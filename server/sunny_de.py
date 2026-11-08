"""Sunshine now and for the next 2 h over Germany ("sunny" in the Germany edition), from DWD satellite data.

Observation: DWD's satellite direct radiation, opendata.dwd.de/weather/satellite/radiation/sid/SIDhr<stamp>EUv4.nc.bz2
(SPECMAGIC_now from Meteosat/MTG, every 10 min, ~17-20 min behind, netCDF3, regular lat/lon grid 0.015 deg = ~1 x 1.7 km,
35.5-59.8 N / 13 W-26 E, int16 W/m2, fill -1, bz2 ~3 MB by day). It holds SID (direct irradiance on the horizontal) and
DNI (direct normal irradiance); the WMO defines sunshine as DNI > 120 W/m2, so DNI is used (SID = DNI * sin(elevation):
the same clear-sky index either way).
Low sun and night: below DAY_ELEV the radiation retrieval says nothing about clouds, so DWD's infrared cloud top height
(opendata.dwd.de/weather/satellite/clouds/CTH/CTHin<stamp>nEU.nc.bz2, MTG FCI IR10.5, every 10 min, 0.03 deg, netCDF3,
metres, works day and night) stands in: low "tops" are the (cold) ground or low fog, high tops are cloud. It keeps the
map's clouds at night and lets the forecast know the clouds that drift into the morning sun. Its "opa" (opacity) field
is 100 % almost everywhere in November and told sunny from cloudy pixels worse (80 % vs 90 % agreement with DNI by day).
Not used: SIS/SISin (global radiation: diffuse light, not sunshine), CAL (cloud albedo: SID already contains it),
SIDin...DEv3 (the same data on a coarse 0.05 deg grid), TS (thunderstorm severity, 800 MB HDF5) and SISfc - an hourly
global radiation forecast for Germany (0.05 deg, +18 h, issued hourly; see server/replay_sun.py for its score): hourly
global radiation cannot tell 10-minute sunshine.

Method (separates the weather from the sun's geometry):
  1. clear-sky index k = DNI / DNI_clear(elevation) (simple clear-sky beam model, air mass after Kasten & Young)
  2. motion: dense optical flow (OpenCV Farneback, as server/clouds.py) between the k fields ~30 min apart, smoothed
  3. extrapolation of k along the motion in 10 min steps to +150 min, blurred with lead time (small clouds are not
     predictable for long)
  4. sunshine at each forecast time from the advected k and that time's sun: DNI = k * DNI_clear(elevation(t)),
     soft threshold around 120 W/m2 -> "sun share" 0..1 (fraction of the 10 min with sunshine); none below SUN_ELEV
Shadows (scored, not served): "pers" (the analysis stays), "kpers" (k stays, only the sun moves), "model" (fading into
the weather model's sunshine in the second hour), "nwp" (weather model only: Open-Meteo minutely_15 sunshine_duration,
self-hosted, best match = ICON-D2 in Germany). The served forecast changes only when a shadow wins (server/score_sun.py).
Replay 07.+08.11. (server/replay_sun.py, 94 runs, daytime): station MAE +60/+120 sat 0.089/0.124, persistence
0.106/0.145, kpers 0.102/0.139, model blend 0.089/0.175, model alone 0.230/0.241 (far too sunny, bias +0.2); grid CSI
+60/+120 sat 0.87/0.77, persistence 0.83/0.71. So the model is not mixed in (to be re-checked with the live logs).

Verification (like server/fcverify.py for rain): every run's leads LEADS are compared with the later SID analysis on
a ~3 km grid inside the shown area (sunny = share >= 0.5, ~9 km tolerance) -> verify_log/sungrid_<YYYYMMDD>.jsonl, and
with DWD's 10-minute station sunshine duration (SD_10, ~70 stations, CDC open data "now" files) ->
verify_log/sunstation_<YYYYMMDD>.jsonl. Daytime only (sun >= SCORE_ELEV at the valid time).

Served by server/rainy_server.py in the Germany edition through the same API as the full edition's clouds.py, so the
app's sunny mode works unchanged:
  GET /api/cloud/frames          -> {"frames": [{"ts", "fc"}], "ref": ms}
  GET /api/cloud/point?lat=&lon= -> {"series": [{"ts", "fc", "cloud": 0..1, "sun": 0..1}], "ref": ms}  ([] outside)
  GET /tiles/cloud/<key>/<ts>/<z>/<x>/<y>.png -> grey = cloud value, alpha = inside the shown area
"cloud" is encoded so that the app's sunShare(cloud) = 1 - (cloud - 0.05) / 0.8 gives the sun share back by day; "sun"
is the sun share itself (0 below SUN_ELEV). /api/health reports "sunny": frames, ref, age, stations.
Env: SUNNY_DE=0 disables it; VERIFY_LOG_DIR as for fcverify.
Memory (measured, 4 h replay): 3 h of observed shares + map values, the k fields of the last 40 min, one forecast
(float16 / uint8 on a ~1.9 km grid, 631 x 593) and the pending verification grids: ~50 MB; ~150 MB transient while a
file is read (bz2 -> 17 MB netCDF). CPU: ~1.3 s per 10-min slot on the x86 dev machine (unpack, regrid, IR, forecast,
shadows); downloads ~3 MB (SID) + ~0.9 MB (CTH, only when the sun is low somewhere) per 10 min, stations ~70 small files
per ~30 min. No lock is shared with the rain processing.
"""
import bz2
import io
import json
import math
import os
import re
import threading
import time
import urllib.parse
import urllib.request
import zipfile
from concurrent.futures import ThreadPoolExecutor, TimeoutError, as_completed
from datetime import datetime, timezone

import cv2
import numpy as np
from PIL import Image
from scipy.io import netcdf_file

import gauges
from clouds import sun_elevation
from mosaic import GermanyDwd, WORLD, merc

SID_URL = "https://opendata.dwd.de/weather/satellite/radiation/sid/"
CTH_URL = "https://opendata.dwd.de/weather/satellite/clouds/CTH/"
SOLAR = "https://opendata.dwd.de/climate_environment/CDC/observations_germany/climate/10_minutes/solar/now/"
LOG_DIR = os.environ.get("VERIFY_LOG_DIR", os.path.join(os.path.dirname(os.path.abspath(__file__)), "verify_log"))

STEP_MS = 600_000
CELL = 2 * GermanyDwd.CELL     # 3000 m Web Mercator = ~1.9 km on the ground (the satellite has ~1 x 1.7 km)
OBS_KEEP_MS = 3 * 3600_000     # observed frames kept (the app shows the last hour; verification needs 2 h)
K_KEEP_MS = 40 * 60_000        # k fields kept for the motion
FC_LEADS = range(10, 151, 10)  # minutes after the newest image (it is ~20 min old): covers "now + 2 h"
POLL_S = 120                   # new files every 10 min
# sun geometry
SUN_ELEV = 3.0                 # below: no sunshine (the beam is too weak and the retrieval unreliable)
DAY_ELEV = 8.0                 # above: k from DNI alone; between SUN_ELEV and DAY_ELEV it fades into the IR cloud top height
DNI_SUN = 120.0                # WMO sunshine threshold, W/m2
CS_A = 0.72                    # clear-sky beam transmittance per air mass (fitted to the clearest DWD pixels, 08.11.)
K_MAX = 1.3
# soft threshold: share = clip((DNI - (DNI_SUN - RAMP)) / (2 RAMP)); against SD_10 (replay_sun.py --calib, 08.11., 2467
# station values): RAMP 40 + 3 x 3 mean MAE 0.072, hit 0.929 (RAMP 20-80 within 0.001, 200: MAE 0.22)
RAMP = 40.0
# IR cloud top height (m) -> k: clear below CTH0, cloud above CTH1. Fitted against daytime DNI on 08.11. (scratch replay,
# 320k pixels, server/replay_sun.py --calib): 90 % agreement on sunny / not sunny; recheck in summer (warm ground, other heights for clear sky)
CTH0, CTH1 = 3000.0, 4000.0
# nowcast
FLOW_DOWN = 3                  # optical flow on a ~6 km grid: large-scale motion only
FLOW_SIGMA = 10.0              # flow smoothing (down pixels, ~55 km)
BLUR = 1.2                     # blur sigma (px) at 10 min lead, grows with sqrt(lead)
# model shadow (Open-Meteo)
MODEL_DEG, MODEL_BATCH, MODEL_REFRESH_S = 0.1, 2000, 30 * 60
MODEL_START, MODEL_END, MODEL_END_W = 60, 120, 0.5   # satellite weight 1 until +60, 0.5 at +120 (shadow "model")
# verification
LEADS = (10, 30, 60, 90, 120)
VERIFY_DOWN = 1.5              # verification grid = sunny grid / 1.5 = ~2.8 km (the rain verification's ~3 km)
TOL_PX = 3                     # ~9 km position tolerance
SCORE_ELEV = 5.0               # daytime only
MIN_PX = 300                   # fewer daytime pixels: no grid line
SUNNY = 0.5                    # "sunny" = share >= 0.5
STATION_WAIT_MS = 4 * 3600_000 # station values are 20-110 min behind; give up on a forecast after this
STATION_LAG_MS = 10 * 60_000   # SD_10 stamped T covers (T-10, T]; the satellite slot S scans Germany ~S+5..S+9
# Bad slots: now and then one SIDhr file shows sun almost everywhere (08.11. 14:50Z: DNI > 120 W/m2 on 91 % of Germany,
# 15 % ten minutes before and 29 % after; the stations saw no such thing). A slot whose daytime sun share differs from
# the previous image by more than JUMP_MAX on average is skipped; after JUMP_ACCEPT skipped slots in a row it counts.
JUMP_MAX = 0.35
JUMP_ACCEPT = 2
TRUTH_KEEP_MS = 8 * 3600_000


def clear_dni(elev):
    """Clear-sky direct normal irradiance (W/m2) for a solar elevation in degrees (array): Meinel-type
    transmittance CS_A ** (AM ** 0.678) with the Kasten & Young air mass. Only its shape over elevation matters for
    the forecast (the index k is divided by it and multiplied back with it at the forecast time)."""
    e = np.maximum(np.asarray(elev, np.float32), 0.5)
    am = 1.0 / (np.sin(np.radians(e)) + 0.50572 * (e + 6.07995) ** -1.6364)
    return (1361.0 * CS_A ** (am ** 0.678)).astype(np.float32)


def share_from_k(k, elev):
    """Sun share 0..1 from the clear-sky index and the solar elevation (NaN where k is unknown)."""
    s = np.clip((k * clear_dni(elev) - (DNI_SUN - RAMP)) / (2 * RAMP), 0.0, 1.0)
    return np.where(elev < SUN_ELEV, np.where(np.isfinite(k), 0.0, np.nan), s).astype(np.float32)


def k_from_cth(cth):
    """Clear-sky index estimated from the IR cloud top height (m)."""
    return np.clip((CTH1 - cth) / (CTH1 - CTH0), 0.0, 1.0).astype(np.float32)


def read_nc(data, names):
    """(lat, lon, {name: float32 grid with NaN for fill values}) of a (bz2) netCDF3 file with a regular lat/lon grid."""
    if data[:3] == b"BZh":
        data = bz2.decompress(data)
    f = netcdf_file(io.BytesIO(data), "r", mmap=False)
    lat, lon = np.array(f.variables["lat"][:], np.float64), np.array(f.variables["lon"][:], np.float64)
    out = {}
    for n in names:
        v = f.variables[n]
        a = np.array(v[0] if len(v.shape) == 3 else v[:], np.float32)
        fill = getattr(v, "_FillValue", None)
        if fill is not None:
            a[a == float(fill)] = np.nan
        out[n] = a
    f.close()
    return lat, lon, out


def stamp_ts(stamp):
    return int(datetime.strptime(stamp, "%Y%m%d%H%M").replace(tzinfo=timezone.utc).timestamp() * 1000)


def ts_stamp(ts):
    return datetime.fromtimestamp(ts / 1000, timezone.utc).strftime("%Y%m%d%H%M")


class SunGrid:
    """The sunny grid: GermanyDwd's Web Mercator box at twice its cell size."""
    BBOX = GermanyDwd.BBOX

    def __init__(self):
        x0, y0, x1, y1 = self.BBOX
        self.nx, self.ny = int(round((x1 - x0) / CELL)), int(round((y1 - y0) / CELL))
        xs = x0 + (np.arange(self.nx) + 0.5) * CELL
        ys = y1 - (np.arange(self.ny) + 0.5) * CELL
        lon = xs / WORLD * 180.0
        lat = np.degrees(2 * np.arctan(np.exp(ys / WORLD * np.pi)) - np.pi / 2)
        self.lat, self.lon = (a.astype(np.float32) for a in np.meshgrid(lat, lon, indexing="ij"))
        self.gx, self.gy = np.meshgrid(np.arange(self.nx, dtype=np.float32), np.arange(self.ny, dtype=np.float32))
        self.maps = {}  # source grid -> (crop rows, crop cols, remap tables)

    def to_pixels(self, x, y):
        x0, _, _, y1 = self.BBOX
        return (np.asarray(x) - x0) / CELL - 0.5, (y1 - np.asarray(y)) / CELL - 0.5

    def elev(self, ts):
        return sun_elevation(ts, self.lat, self.lon).astype(np.float32)

    def regrid(self, lat, lon, a, sigma=(0.0, 0.0)):
        """A field on a regular lat/lon grid onto the sunny grid (bilinear; optional blur in source pixels first, so a
        finer source is averaged rather than sampled)."""
        key = (lat[0], lat[-1], len(lat), lon[0], lon[-1], len(lon))
        if key not in self.maps:
            dlat, dlon = (lat[-1] - lat[0]) / (len(lat) - 1), (lon[-1] - lon[0]) / (len(lon) - 1)
            fy, fx = (self.lat - lat[0]) / dlat, (self.lon - lon[0]) / dlon
            r0, r1 = max(int(np.floor(fy.min())) - 3, 0), min(int(np.ceil(fy.max())) + 4, len(lat))
            c0, c1 = max(int(np.floor(fx.min())) - 3, 0), min(int(np.ceil(fx.max())) + 4, len(lon))
            self.maps[key] = (slice(r0, r1), slice(c0, c1), (fx - c0).astype(np.float32), (fy - r0).astype(np.float32))
        rows, cols, mx, my = self.maps[key]
        crop = np.ascontiguousarray(a[rows, cols], np.float32)
        valid = np.isfinite(crop)
        filled = np.where(valid, crop, 0.0).astype(np.float32)
        w = valid.astype(np.float32)
        if sigma[0] or sigma[1]:
            filled = cv2.GaussianBlur(filled, (0, 0), sigmaX=sigma[0] or 0.01, sigmaY=sigma[1] or 0.01)
            w = cv2.GaussianBlur(w, (0, 0), sigmaX=sigma[0] or 0.01, sigmaY=sigma[1] or 0.01)
        v = cv2.remap(filled, mx, my, cv2.INTER_LINEAR, borderMode=cv2.BORDER_CONSTANT, borderValue=0)
        n = cv2.remap(w, mx, my, cv2.INTER_LINEAR, borderMode=cv2.BORDER_CONSTANT, borderValue=0)
        return np.where(n > 0.5, v / np.maximum(n, 1e-6), np.nan).astype(np.float32)


def fill_nan(a, sigma=15.0):
    """Unknown pixels from their surroundings (normalised blur), what stays unknown -> 0.5."""
    ok = np.isfinite(a)
    if ok.all():
        return a
    w = cv2.GaussianBlur(ok.astype(np.float32), (0, 0), sigma)
    v = cv2.GaussianBlur(np.where(ok, a, 0).astype(np.float32), (0, 0), sigma)
    return np.where(ok, a, np.where(w > 1e-3, v / np.maximum(w, 1e-3), 0.5)).astype(np.float32)


class SunModel:
    """Weather model sunshine share (Open-Meteo minutely_15 sunshine_duration / 900 s) on a 0.1 deg grid over the
    shown area, for the "model" and "nwp" shadows."""

    def __init__(self, grid, url, available, log):
        self.grid, self.url, self.available, self.log = grid, url, available, log
        self.lock = threading.Lock()
        self.times = self.share = None   # (T,) ms at the middle of each 15 min interval; (T, nlat, nlon)
        self.at = 0.0
        self.lats = np.arange(46.0, 56.0 + 1e-6, MODEL_DEG)
        self.lons = np.arange(2.0, 19.0 + 1e-6, MODEL_DEG)
        self.points = None
        self.maps = (((grid.lon - self.lons[0]) / MODEL_DEG).astype(np.float32),
                     ((grid.lat - self.lats[0]) / MODEL_DEG).astype(np.float32))

    def refresh(self, inside, past_steps=4, forecast_steps=14):
        if not self.available() or time.time() - self.at < MODEL_REFRESH_S:
            return 0
        la, lo = (v.ravel() for v in np.meshgrid(self.lats, self.lons, indexing="ij"))
        if self.points is None:  # only where the map shows sunny (plus a margin)
            near = cv2.dilate((inside > 0).astype(np.uint8), np.ones((15, 15), np.uint8))
            col, row = (np.round(v).astype(int) for v in self.grid.to_pixels(*merc(lo, la)))
            ok = (col >= 0) & (col < self.grid.nx) & (row >= 0) & (row < self.grid.ny)
            ok[ok] = near[row[ok], col[ok]] > 0
            self.points = np.flatnonzero(ok)
        la, lo = la[self.points], lo[self.points]
        times, vals = None, None
        for i in range(0, len(la), MODEL_BATCH):
            body = urllib.parse.urlencode({
                "latitude": ",".join(f"{v:.2f}" for v in la[i:i + MODEL_BATCH]),
                "longitude": ",".join(f"{v:.2f}" for v in lo[i:i + MODEL_BATCH]),
                "minutely_15": "sunshine_duration", "past_minutely_15": str(past_steps),
                "forecast_minutely_15": str(forecast_steps), "timeformat": "unixtime"}).encode()
            req = urllib.request.Request(f"{self.url}/v1/forecast", data=body,
                                         headers={"Content-Type": "application/x-www-form-urlencoded"})
            with urllib.request.urlopen(req, timeout=600) as r:
                d = json.loads(r.read())
            d = d if isinstance(d, list) else [d]
            if times is None:
                times = np.array(d[0]["minutely_15"]["time"], np.int64) * 1000 - 450_000  # value = previous 15 min
                vals = np.zeros((len(times), len(la)), np.float32)
            for j, e in enumerate(d):
                vals[:, i + j] = [min(1.0, (v or 0.0) / 900.0) for v in e["minutely_15"]["sunshine_duration"]]
        share = np.full((len(times), len(self.lats) * len(self.lons)), np.nan, np.float32)
        share[:, self.points] = vals
        share = share.reshape(len(times), len(self.lats), len(self.lons))
        with self.lock:
            self.times, self.share = times, share
        self.at = time.time()
        return len(la)

    def field(self, ts):
        with self.lock:
            times, share = self.times, self.share
        if share is None or not times[0] <= ts <= times[-1]:
            return None
        k = min(int(np.searchsorted(times, ts, side="right")) - 1, len(times) - 2)
        f = (ts - times[k]) / (times[k + 1] - times[k])
        m = ((1 - f) * share[k] + f * share[k + 1]).astype(np.float32)
        return cv2.remap(fill_nan(m, 3.0), *self.maps, cv2.INTER_LINEAR, borderMode=cv2.BORDER_REPLICATE)


class SunStations:
    """DWD 10-minute station sunshine duration (SD_10, hours per 10 min -> share = SD_10 * 6), read politely like
    gauges.GaugeDrizzle: one listing request every gauges.LIST_EVERY_S shows which files changed, only those are read,
    gauges.PACE_S apart per worker, stopping after gauges.MAX_ERRORS failures in a row, with an overall deadline."""

    def __init__(self, fetch, log):
        self.fetch, self.log = fetch, log
        self.lock = threading.Lock()
        self.stations = {}  # id -> (lat, lon)
        self.values = {}    # id -> {end time ms: share}
        self.errors = 0
        self.seen, self.listed_day = {}, None

    def _station_list(self):
        text = self.fetch(f"{SOLAR}zehn_now_sd_Beschreibung_Stationen.txt", timeout=60).decode("latin-1")
        recent = time.strftime("%Y%m%d", time.gmtime(time.time() - 3 * 86400))
        out = {}
        for line in text.splitlines()[2:]:
            p = line.split()
            if len(p) >= 6 and p[0].isdigit() and p[2] >= recent:
                out[p[0]] = (float(p[4]), float(p[5]))
        return out

    def _read_one(self, sid):
        if self.errors >= gauges.MAX_ERRORS:
            return None
        time.sleep(gauges.PACE_S)
        try:
            z = zipfile.ZipFile(io.BytesIO(self.fetch(f"{SOLAR}10minutenwerte_SOLAR_{sid}_now.zip", timeout=30, tries=1)))
            rows = z.read(z.namelist()[0]).decode("latin-1").splitlines()
            self.errors = 0
        except Exception:
            self.errors += 1
            return None
        k = [h.strip() for h in rows[0].split(";")].index("SD_10")
        vals = {}
        for r in rows[1:]:
            p = r.split(";")
            try:
                v = float(p[k])
                if v >= 0:
                    vals[stamp_ts(p[1].strip())] = min(1.0, v * 6.0)
            except (ValueError, IndexError):
                continue
        return vals

    def poll(self):
        if time.strftime("%Y%m%d") != self.listed_day:
            self.stations, self.listed_day = self._station_list(), time.strftime("%Y%m%d")
        text = self.fetch(SOLAR, timeout=60).decode("latin-1")
        stamps = {sid: s for sid, s in re.findall(r'SOLAR_(\d{5})_now\.zip</a>\s+(\S+ \S+)', text) if sid in self.stations}
        todo = [sid for sid, s in stamps.items() if self.seen.get(sid) != s]
        if not todo:
            return 0
        self.errors = 0
        pool = ThreadPoolExecutor(gauges.WORKERS)
        futs = {pool.submit(self._read_one, sid): sid for sid in todo}
        got = {}
        try:
            for f in as_completed(futs, timeout=gauges.READ_DEADLINE_S):
                if f.result() is not None:
                    got[futs[f]] = f.result()
        except TimeoutError:
            pass
        pool.shutdown(wait=False, cancel_futures=True)
        cut = time.time() * 1000 - TRUTH_KEEP_MS
        with self.lock:
            for sid, vals in got.items():
                d = self.values.setdefault(sid, {})
                d.update({t: v for t, v in vals.items() if t >= cut})
                for t in [t for t in d if t < cut]:
                    del d[t]
        self.seen.update({sid: stamps[sid] for sid in got})
        return len(got)

    def truth(self, end_ts):
        """{id: share} of the stations that reported the 10 min ending at end_ts."""
        with self.lock:
            return {sid: v[end_ts] for sid, v in self.values.items() if end_ts in v}

    def newest(self):
        with self.lock:
            ends = [max(v) for v in self.values.values() if v]
        return int(np.median(ends)) if ends else 0

    def loop(self):
        while True:
            started = time.time()
            try:
                n = self.poll()
                if n:
                    self.log(f"[{datetime.now():%H:%M:%S}] sunny: {n} station files in {time.time() - started:.0f}s")
            except Exception as e:
                self.log(f"sunny stations: {e}")
            time.sleep(max(30.0, gauges.LIST_EVERY_S - (time.time() - started)))


class SunVerify:
    """Scores every forecast run against the later SID analysis (grid) and the DWD stations (point)."""

    def __init__(self, sun, stations, log, log_dir=None):
        self.sun, self.stations, self.log = sun, stations, log
        self.log_dir = log_dir or LOG_DIR
        g = sun.grid
        self.shape = (int(round(g.nx / VERIFY_DOWN)), int(round(g.ny / VERIFY_DOWN)))  # cv2 (w, h)
        self.pending = {}   # (variant, ref, lead) -> share on the verification grid (uint8, x 250, 255 = unknown)
        self.spending = {}  # (variant, ref, lead) -> {station id: forecast share}
        self.done_ref = None

    def _down(self, a):
        ok = np.isfinite(a)
        v = cv2.resize(np.where(ok, a, 0).astype(np.float32), self.shape, interpolation=cv2.INTER_AREA)
        w = cv2.resize(ok.astype(np.float32), self.shape, interpolation=cv2.INTER_AREA)
        return np.where(w > 0.5, v / np.maximum(w, 1e-6), np.nan).astype(np.float32)

    @staticmethod
    def _counts(fc, obs, mask):
        fc, obs = fc & mask, obs & mask
        k = np.ones((2 * TOL_PX + 1,) * 2, np.uint8)
        near_obs = cv2.dilate(obs.astype(np.uint8), k) > 0
        near_fc = cv2.dilate(fc.astype(np.uint8), k) > 0
        return int((fc & near_obs).sum()), int((obs & ~near_fc).sum()), int((fc & ~near_obs).sum())

    def take(self, ref, variants):
        """Keep the leads of a new run: variants = {name: {valid ts: share grid}}."""
        if ref == self.done_ref:
            return
        sts = self.sun.station_pixels()
        for name, grids in variants.items():
            for lead in LEADS:
                g = grids.get(ref + lead * 60_000)
                if g is None:
                    continue
                d = self._down(g)
                self.pending[(name, ref, lead)] = np.where(np.isfinite(d), np.round(d * 250), 255).astype(np.uint8)
                if sts:
                    self.spending[(name, ref, lead)] = {sid: v for sid, v in self.sun.at_pixels(g, sts).items() if np.isfinite(v)}
        self.done_ref = ref

    def _write(self, kind, lines):
        if not lines:
            return
        os.makedirs(self.log_dir, exist_ok=True)
        by_day = {}
        for row in lines:
            by_day.setdefault(datetime.fromtimestamp(row["valid"] / 1000, timezone.utc).strftime("%Y%m%d"), []).append(row)
        for day, rows in by_day.items():
            with open(os.path.join(self.log_dir, f"{kind}_{day}.jsonl"), "a") as f:
                for row in rows:
                    f.write(json.dumps(row) + "\n")

    def step(self, now_ref):
        """Score what can be scored; now_ref = the newest analysis time (the server's clock for this purpose)."""
        sun, lines, slines = self.sun, [], []
        mask_area = self._down(sun.inside()) > 0.5
        for key, g in list(self.pending.items()):
            variant, r, lead = key
            valid = r + lead * 60_000
            a = sun.obs_share(valid)
            if a is None:
                if now_ref - r > OBS_KEEP_MS:
                    del self.pending[key]
                continue
            del self.pending[key]
            a = self._down(a)
            g = np.where(g == 255, np.nan, g / 250.0).astype(np.float32)
            e = self._down(sun.elev(valid))
            mask = mask_area & (e >= SCORE_ELEV) & np.isfinite(a) & np.isfinite(g)
            n = int(mask.sum())
            if n < MIN_PX:
                continue
            d = g[mask] - a[mask]
            lines.append({"variant": variant, "ref": r, "lead": lead, "valid": valid, "n": n,
                          "obs": int((a[mask] >= SUNNY).sum()), "fcn": int((g[mask] >= SUNNY).sum()),
                          "fc": list(self._counts(g >= SUNNY, a >= SUNNY, mask)),
                          "bias": round(float(d.mean()), 4), "mae": round(float(np.abs(d).mean()), 4)})
        newest = self.stations.newest() if self.stations else 0
        for key, fc in list(self.spending.items()):
            variant, r, lead = key
            valid = r + lead * 60_000
            end = valid + STATION_LAG_MS
            if newest < end:
                if now_ref - r > STATION_WAIT_MS:
                    del self.spending[key]
                continue
            del self.spending[key]
            truth = self.stations.truth(end)
            where = sun.station_coords()
            pairs = [(fc[s], truth[s]) for s in fc if s in truth and s in where
                     and sun_elevation(valid, where[s][0], where[s][1]) >= SCORE_ELEV]
            if len(pairs) < 5:
                continue
            f, o = np.array(pairs, np.float32).T
            slines.append({"variant": variant, "ref": r, "lead": lead, "valid": valid, "n": len(pairs),
                           "mae": round(float(np.abs(f - o).mean()), 4), "bias": round(float((f - o).mean()), 4),
                           "hit": round(float(((f >= SUNNY) == (o >= SUNNY)).mean()), 4),
                           "obs": int((o >= SUNNY).sum()), "fcn": int((f >= SUNNY).sum())})
        self._write("sungrid", lines)
        self._write("sunstation", slines)
        return len(lines), len(slines)


class SunnyDE:
    name = "sunny"

    def __init__(self, fetch, log, de, openmeteo_url=None, openmeteo_available=lambda: False, stations=True, log_dir=None):
        self.fetch, self.log, self.de = fetch, log, de
        self.grid = SunGrid()
        self.lock = threading.Lock()
        self.obs, self.fc, self.ref = {}, {}, None  # ts -> share grid (float16, NaN = unknown); ref = newest image
        self.disp = {}                              # ts -> cloud value for the map (uint8, 255 = unknown)
        self.k = {}                                 # ts -> clear-sky index (float16), last K_KEEP_MS only
        self.shift_ms = 0                           # DEV replay: frames relabelled by this, the sun stays at the real time
        self.frozen = False
        self.stations = SunStations(fetch, log) if stations else None
        self.model = SunModel(self.grid, openmeteo_url, openmeteo_available, log) if openmeteo_url else None
        self.verify = SunVerify(self, self.stations, log, log_dir)
        self._inside = None
        self.cycle_s = None
        self._st_pix = None
        self.skipped, self.skips_in_row = set(), 0  # slots rejected as implausible (JUMP_MAX)

    # ---------- helpers ----------
    def inside(self):
        """Area the map shows (Germany + border zone, from the rain radar's mask) on the sunny grid."""
        if self._inside is None or getattr(self, "_inside_from", None) is not self.de.inside:
            self._inside = cv2.resize(self.de.inside, (self.grid.nx, self.grid.ny), interpolation=cv2.INTER_AREA)
            self._inside_from = self.de.inside
        return self._inside

    def elev(self, ts):
        return self.grid.elev(ts - self.shift_ms)

    def station_coords(self):
        return self.stations.stations if self.stations else {}

    def station_pixels(self):
        where = self.station_coords()
        if not where:
            return {}
        if self._st_pix is None or len(self._st_pix) != len(where):
            pix = {}
            for sid, (lat, lon) in where.items():
                col, row = (int(round(float(v))) for v in self.grid.to_pixels(*merc(lon, lat)))
                if 1 <= col < self.grid.nx - 1 and 1 <= row < self.grid.ny - 1:
                    pix[sid] = (row, col)
            self._st_pix = pix
        return self._st_pix

    @staticmethod
    def at_pixels(g, pix):
        """3 x 3 mean (~6 km, like /api/cloud/point) of a grid at station pixels."""
        out = {}
        for sid, (row, col) in pix.items():
            c = g[row - 1:row + 2, col - 1:col + 2].astype(np.float32)
            out[sid] = float(np.nanmean(c)) if np.isfinite(c).any() else float("nan")
        return out

    def obs_share(self, ts):
        with self.lock:
            g = self.obs.get(ts)
        return None if g is None else g.astype(np.float32)

    # ---------- observation ----------
    def ingest(self, ts, sid_data, cth_data=None):
        """One satellite slot: SIDhr file bytes (+ CTH file bytes) -> k, share and map value at ts."""
        lat, lon, v = read_nc(sid_data, ["DNI"])
        dni = self.grid.regrid(lat, lon, v["DNI"], sigma=(0.8, 0.5))
        del v
        e = self.elev(ts)
        k_dni = np.where(e >= SUN_ELEV, np.clip(dni / clear_dni(e), 0, K_MAX), np.nan).astype(np.float32)
        k_ir = None
        if cth_data is not None:
            lat, lon, c = read_nc(cth_data, ["CTH"])
            k_ir = k_from_cth(self.grid.regrid(lat, lon, c["CTH"]))
        if k_ir is None:
            k = k_dni
        else:  # DNI by day, IR cloud tops at low sun and night, blended in between; either one fills in for the other
            w = np.clip((e - SUN_ELEV) / (DAY_ELEV - SUN_ELEV), 0.0, 1.0)
            kd = np.where(np.isfinite(k_dni), k_dni, k_ir)
            ki = np.where(np.isfinite(k_ir), k_ir, kd)
            k = (w * kd + (1 - w) * ki).astype(np.float32)
        share = share_from_k(k, e)
        return k, share, self._display(k, share, e)

    @staticmethod
    def _display(k, share, e):
        """Cloud value for the map: from the sun share by day, from k at a reference sun near and below the horizon
        (no seam at the terminator); encoded so that the app's sunShare() returns the share by day."""
        pot = share_from_k(k, np.full_like(e, 15.0))
        w = np.clip((e - SUN_ELEV) / (DAY_ELEV - SUN_ELEV), 0.0, 1.0)
        s = w * np.nan_to_num(share) + (1 - w) * np.nan_to_num(pot)
        v = np.clip(0.05 + 0.8 * (1.0 - s), 0, 1) * 254
        return np.where(np.isfinite(k), v, 255).astype(np.uint8)

    # ---------- forecast ----------
    def _motion(self, k0, k1, dt_min):
        img = lambda k: (np.clip(1.0 - fill_nan(k), 0, 1) * 255).astype(np.uint8)
        s = (self.grid.nx // FLOW_DOWN, self.grid.ny // FLOW_DOWN)
        a = cv2.resize(img(k0), s, interpolation=cv2.INTER_AREA)
        b = cv2.resize(img(k1), s, interpolation=cv2.INTER_AREA)
        flow = cv2.calcOpticalFlowFarneback(a, b, None, 0.5, 4, 15, 4, 7, 1.5, 0)
        flow = cv2.GaussianBlur(flow, (0, 0), FLOW_SIGMA)
        return cv2.resize(flow, (self.grid.nx, self.grid.ny), interpolation=cv2.INTER_LINEAR) * (FLOW_DOWN / dt_min)

    def forecast(self, ref):
        """Forecast shares {valid ts: grid} for the served variant and the shadows, from the k fields up to ref."""
        with self.lock:
            k1 = self.k.get(ref)
            prev = next(((self.k[ref - m * 60_000], m) for m in (30, 20, 40, 10) if ref - m * 60_000 in self.k), None)
            obs = self.obs.get(ref)
        if k1 is None:
            return None
        k1 = k1.astype(np.float32)
        flow = self._motion(prev[0].astype(np.float32), k1, prev[1]) if prev else None
        base = fill_nan(k1)
        known = np.isfinite(k1).astype(np.float32)
        out = {"sat": {}, "kpers": {}, "pers": {}, "model": {}, "nwp": {}}
        disp = {}
        for lead in FC_LEADS:
            ts = ref + lead * 60_000
            e = self.elev(ts)
            if flow is not None:
                mx, my = self.grid.gx - flow[..., 0] * lead, self.grid.gy - flow[..., 1] * lead
                kk = cv2.remap(base, mx, my, cv2.INTER_LINEAR, borderMode=cv2.BORDER_REPLICATE)
                kn = cv2.remap(known, mx, my, cv2.INTER_LINEAR, borderMode=cv2.BORDER_REPLICATE)
            else:
                kk, kn = base.copy(), known
            kk = cv2.GaussianBlur(kk, (0, 0), BLUR * math.sqrt(lead / 10))
            kk = np.where(kn > 0.5, kk, np.nan).astype(np.float32)
            s = share_from_k(kk, e)
            out["sat"][ts] = s
            disp[ts] = self._display(kk, s, e)
            if lead in LEADS:  # shadows only where they are scored
                out["kpers"][ts] = share_from_k(k1, e)
                if obs is not None:
                    out["pers"][ts] = obs.astype(np.float32)
                m = self.model.field(ts) if self.model else None
                if m is not None:
                    m = np.where(e >= SUN_ELEV, m, 0.0).astype(np.float32)
                    w = 1.0 if lead <= MODEL_START else 1.0 - min(1.0, (lead - MODEL_START) / (MODEL_END - MODEL_START)) * (1 - MODEL_END_W)
                    out["model"][ts] = np.where(np.isfinite(s), w * s + (1 - w) * m, np.nan).astype(np.float32)
                    out["nwp"][ts] = np.where(np.isfinite(s), m, np.nan).astype(np.float32)
        return out, disp

    # ---------- update ----------
    def _jump(self, ts, share, back=(1, 2)):
        """Mean daytime |share - an earlier image's share| inside the shown area (the first of `back` steps that exists),
        or None without such an image."""
        with self.lock:
            prev = next((self.obs[ts - b * STEP_MS] for b in back if ts - b * STEP_MS in self.obs), None)
        if prev is None:
            return None
        m = (self.inside() > 0.5) & (self.elev(ts) >= SCORE_ELEV) & np.isfinite(share) & np.isfinite(prev.astype(np.float32))
        return float(np.abs(share[m] - prev.astype(np.float32)[m]).mean()) if m.sum() >= 500 else None

    def add_slot(self, ts, sid_data, cth_data=None):
        """Ingest one slot as the newest image, make the forecast, hand it to the verification. False: slot skipped."""
        k, share, disp = self.ingest(ts, sid_data, cth_data)
        jump = self._jump(ts, share)
        if jump is not None and jump > JUMP_MAX and ts - STEP_MS in self.obs:
            before = self._jump(ts, share, back=(2,))
            if before is not None and before <= JUMP_MAX:  # the previous image was the odd one out: drop it
                bad = ts - STEP_MS
                with self.lock:
                    for d in (self.obs, self.k, self.disp):
                        d.pop(bad, None)
                for d in (self.verify.pending, self.verify.spending):
                    for key in [key for key in d if key[1] == bad]:
                        del d[key]
                self.skipped.add(bad)
                self.log(f"sunny: slot {ts_stamp(bad)} dropped (sun share jumped by {jump:.2f} on average and back)")
                jump = None
        if jump is not None and jump > JUMP_MAX and self.skips_in_row < JUMP_ACCEPT:
            self.skipped.add(ts)
            self.skips_in_row += 1
            self.log(f"sunny: slot {ts_stamp(ts)} skipped (sun share jumps by {jump:.2f} on average)")
            return False
        self.skips_in_row = 0
        with self.lock:
            self.k[ts] = k.astype(np.float16)
            self.obs[ts] = share.astype(np.float16)
            self.disp[ts] = disp
            for t in [t for t in self.k if t < ts - K_KEEP_MS]:
                del self.k[t]
            for t in [t for t in self.obs if t < ts - OBS_KEEP_MS]:
                del self.obs[t]
                self.disp.pop(t, None)
            newest = ts >= (self.ref or 0)
        if not newest:
            return True
        res = self.forecast(ts)
        if res is None:
            return True
        variants, disp = res
        with self.lock:
            self.fc = {t: g.astype(np.float16) for t, g in variants["sat"].items()}
            for t, d in disp.items():
                if t not in self.obs:
                    self.disp[t] = d
            for t in [t for t in self.disp if t not in self.obs and t not in self.fc]:
                del self.disp[t]
            self.ref = ts
        self.verify.take(ts, {n: v for n, v in variants.items() if v})
        return True

    def _listing(self, url, pattern):
        text = self.fetch(url, timeout=60).decode("latin-1")
        return {stamp_ts(s): name for name, s in ((m.group(0), m.group(1)) for m in re.finditer(pattern, text))}

    def update(self):
        if self.frozen:
            return 0
        sids = self._listing(SID_URL, r"SIDhr(\d{12})EUv4\.nc\.bz2")
        if not sids:
            return 0
        newest = max(sids)
        self.skipped = {t for t in self.skipped if t > newest - OBS_KEEP_MS}
        # new slots, plus late ones up to K_KEEP_MS before the newest image (not the whole history after a start)
        want = [t for t in sorted(sids) if t > newest - OBS_KEEP_MS and t not in self.obs and t not in self.skipped
                and t > (self.ref or 0) - K_KEEP_MS]
        if not want:
            return 0
        if self.ref is None:  # first start: the last hour (the app's history and the motion), not all 3 h
            want = want[-7:]
        cths = None
        n = 0
        for t in want:
            e = self.elev(t)
            cth = None
            if float(e.min()) < DAY_ELEV:  # IR needed somewhere (low sun, night)
                try:
                    cths = cths if cths is not None else self._listing(CTH_URL, r"CTHin(\d{12})nEU\.nc\.bz2")
                    near = [c for c in cths if abs(c - t) <= 10 * 60_000]
                    if near:
                        c = min(near, key=lambda c: abs(c - t))
                        cth = self.fetch(CTH_URL + cths[c], timeout=120)
                except Exception as ex:
                    self.log(f"sunny: CTH failed: {ex}")
            data = self.fetch(SID_URL + sids[t], timeout=120)
            if t < (self.ref or 0):  # an older slot that arrived late: history only, no forecast from it
                k, share, disp = self.ingest(t, data, cth)
                with self.lock:
                    self.obs[t], self.disp[t] = share.astype(np.float16), disp
                    if t >= self.ref - K_KEEP_MS:
                        self.k[t] = k.astype(np.float16)
            else:
                self.add_slot(t, data, cth)
            n += 1
        return n

    def verify_step(self):
        if self.ref:
            return self.verify.step(self.ref)
        return 0, 0

    def loop(self):
        if self.stations:
            threading.Thread(target=self.stations.loop, daemon=True).start()
        if self.model:
            threading.Thread(target=self.model_loop, daemon=True).start()
        while True:
            started = time.time()
            try:
                n = self.update()
                if n:
                    self.cycle_s = round(time.time() - started, 1)
                    self.log(f"[{datetime.now():%H:%M:%S}] sunny: +{n} images, latest "
                             f"{datetime.fromtimestamp(self.ref / 1000, timezone.utc):%H:%M}Z, {self.cycle_s}s")
            except Exception as e:  # keep serving the last good data
                self.log(f"sunny update failed: {e}")
            try:
                g, s = self.verify_step()
                if g or s:
                    self.log(f"[{datetime.now():%H:%M:%S}] sunny verify: {g} grid, {s} station lines")
            except Exception as e:
                self.log(f"sunny verify failed: {e}")
            time.sleep(POLL_S)

    def model_loop(self):
        while True:
            try:
                n = self.model.refresh(self.inside())
                if n:
                    self.log(f"[{datetime.now():%H:%M:%S}] sunny: model sunshine for {n} points")
            except Exception as e:
                self.log(f"sunny model failed: {e}")
            time.sleep(60)

    # ---------- serving (same API as clouds.CloudNowcast) ----------
    def frames(self):
        with self.lock:
            return ([{"ts": t, "fc": False} for t in sorted(self.obs)] +
                    [{"ts": t, "fc": True} for t in sorted(self.fc) if t not in self.obs]), self.ref

    def health(self):
        frames, ref = self.frames()
        age = round((time.time() * 1000 - ref) / 60_000) if ref else None
        return {"frames": len(frames), "obs": sum(not f["fc"] for f in frames), "ref": ref, "age_min": age,
                "cycle_s": self.cycle_s, "stations": len(self.stations.values) if self.stations else 0,
                "model": bool(self.model and self.model.share is not None), "replay": bool(self.shift_ms),
                "skipped": len(self.skipped)}

    def image(self, ts, bbox, w, h):
        """Cloud value as a grey + alpha PNG (alpha 0 outside the shown area or where unknown); None: unknown frame."""
        with self.lock:
            g = self.disp.get(ts)
        if g is None:
            return None
        minx, miny, maxx, maxy = bbox
        X, Y = np.meshgrid(minx + (np.arange(w) + 0.5) / w * (maxx - minx), maxy - (np.arange(h) + 0.5) / h * (maxy - miny))
        col, row = (v.astype(np.float32) for v in self.grid.to_pixels(X, Y))
        known = (g < 255).astype(np.float32)
        v = cv2.remap(np.where(g < 255, g, 0).astype(np.float32), col, row, cv2.INTER_LINEAR, borderMode=cv2.BORDER_CONSTANT, borderValue=0)
        kn = cv2.remap(known, col, row, cv2.INTER_LINEAR, borderMode=cv2.BORDER_CONSTANT, borderValue=0)
        ins = cv2.remap(self.inside(), col, row, cv2.INTER_LINEAR, borderMode=cv2.BORDER_CONSTANT, borderValue=0)
        grey = np.clip(v / np.maximum(kn, 1e-3), 0, 255).astype(np.uint8)
        alpha = ((ins > 0.5) & (kn > 0.5)).astype(np.uint8) * 255
        buf = io.BytesIO()
        Image.fromarray(np.dstack([grey, alpha]), "LA").save(buf, "PNG", compress_level=3)
        return buf.getvalue()

    def point(self, lat, lon):
        """Series at a place (3 x 3 mean, ~6 km), or [] outside the shown area."""
        col, row = (int(round(float(v))) for v in self.grid.to_pixels(*merc(lon, lat)))
        if not (1 <= col < self.grid.nx - 1 and 1 <= row < self.grid.ny - 1) or self.inside()[row, col] <= 0.5:
            return []
        frames, _ = self.frames()
        out = []
        with self.lock:
            for f in frames:
                g = self.obs.get(f["ts"]) if not f["fc"] else self.fc.get(f["ts"])
                d = self.disp.get(f["ts"])
                if g is None or d is None:
                    continue
                c = d[row - 1:row + 2, col - 1:col + 2]
                c = c[c < 255]
                s = g[row - 1:row + 2, col - 1:col + 2].astype(np.float32)
                if not c.size:
                    continue
                out.append({"ts": f["ts"], "fc": f["fc"], "cloud": round(float(c.mean()) / 254, 3),
                            "sun": round(float(np.nanmean(s)), 3) if np.isfinite(s).any() else 0.0})
        return out

    # ---------- development ----------
    def dev_replay(self, at, folder):
        """DEV only: replay archived files (folder with SIDhr / CTHin files) up to the UTC slot `at` as if it were now:
        frames are relabelled to the current time, the sun geometry stays at the archived time."""
        at_ts = stamp_ts(at)
        now_slot = int(time.time() * 1000) // STEP_MS * STEP_MS - 20 * 60_000
        shift = now_slot - at_ts
        self.frozen = True
        with self.lock:
            self.obs, self.fc, self.disp, self.k, self.ref = {}, {}, {}, {}, None
            self.shift_ms = shift
        n = 0
        for t in range(at_ts - 18 * STEP_MS, at_ts + 1, STEP_MS):
            fp = os.path.join(folder, f"SIDhr{ts_stamp(t)}EUv4.nc.bz2")
            if not os.path.exists(fp):
                continue
            cp = os.path.join(folder, f"CTHin{ts_stamp(t)}nEU.nc.bz2")
            cth = open(cp, "rb").read() if os.path.exists(cp) else None
            self.add_slot(t + shift, open(fp, "rb").read(), cth)
            n += 1
        self.verify.pending.clear()
        self.verify.spending.clear()
        return n

    def dev_reset(self):
        with self.lock:
            self.obs, self.fc, self.disp, self.k, self.ref = {}, {}, {}, {}, None
            self.shift_ms = 0
        self.frozen = False
