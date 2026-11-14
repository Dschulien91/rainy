"""rainy data server.

Builds rain radar with a 2 h nowcast for regions where no official nowcast is freely available, and serves
it to the app. Regions:
  eu - EUMETNET OPERA composite (reflectivity, 1 km, every 5 min, CC BY 4.0), LAEA grid kept at 2 km
  us - NOAA nowCOAST base reflectivity mosaic (public domain, every ~4 min), Web Mercator grid ~3 km
  sat - EUMETSAT H SAF satellite rain estimate (MTG, every 10 min, ~25 min old), Europe/Africa/Atlantic, ~8 km;
        fills the map where no radar exists (shown fainter in the app)
Nowcast: dense optical flow (OpenCV Farneback) between two frames ~30 min apart, then semi-Lagrangian
extrapolation in 5 min steps with smoothing that grows with lead time (small cells are not predictable for
long, as in S-PROG/pySTEPS). Chosen by backtests (server/tune_nowcast.py): +10-25 % CSI at 60-120 min vs. the
first version, and better than persistence over the full 2 h in Europe and the USA.
Where pySTEPS is installed (the Docker image), Europe and the USA use pySTEPS S-PROG instead (server/compare_pysteps.py:
same skill for widespread rain, clearly better for showers in the second hour).
Second hour: the radar nowcast stays as it is, and the precipitation forecast of the best national weather model
for each place (self-hosted Open-Meteo "best match": ICON-D2, AROME, HARMONIE, UKV, HRRR, ...) adds showers only
where the radar nowcast has no rain nearby - the radar cannot see showers that have not formed yet. A plain mix of
radar and model scored worse for heavy rain (server/verify_blend.py). Every model snapshot is logged (model_log/) so that the blend
weights can be verified fairly later (only model runs that were available at the forecast time).

Endpoints (all CORS-enabled, cache headers suitable for a CDN):
  GET /api/health                              (+ "auth": {"google", "email"} = sign-in methods on offer)
  GET /api/radar/frames?region=eu|us          -> {"frames": [{"ts": ms, "fc": bool}], "ref": ms}
  GET /api/radar/image?region=&ts=&bbox=&width=&height=   (bbox in EPSG:3857) -> PNG in the rainy palette
  GET /api/radar/point?lat=&lon=               -> {"region": "eu"|"us"|"sat"|null, "series": [{"ts", "mmh"}], "ref": ms}
  GET /api/radar/mask?bbox=&width=&height=     -> PNG, opaque where radar coverage exists (for the "no data" hatching)
  Seamless map tiles for the app (server/mosaic.py; DWD + OPERA + NOAA + satellite in one, CDN-cacheable):
  GET /api/thunder/areas, /api/thunder/point?lat=&lon=, /api/lightning -> DWD thunderstorm warnings, MTG lightning
  GET /api/rain/frames, /api/rain/point?lat=&lon=[&src=eu] (+ "covered": inside the shown radar area), /tiles/rain/<key>/<ts>/<z>/<x>/<y>.png, /tiles/cloud/...
      (/tiles/raineu/... and src=eu: Europe composite only, without the DWD source in Germany - user-selectable)
  Clouds for the "sunny" map - full edition: server/clouds.py (Europe, EUMETSAT GeoColour + nowcast); Germany edition:
  server/sunny_de.py (Germany + border zone, DWD satellite direct radiation SID/DNI + 2 h nowcast, verified against the
  later analysis and DWD's sunshine stations; outside the area the point series is empty):
  GET /api/cloud/frames                        -> {"frames": [{"ts", "fc"}], "ref": ms}
  GET /api/cloud/image?ts=&bbox=&width=&height= -> grey + alpha PNG, grey = cloudiness 0..255
  GET /api/cloud/point?lat=&lon=               -> {"series": [{"ts", "fc", "cloud": 0..1[, "sun": 0..1]}], "ref": ms}
  (/api/health: "regions"."sunny" = frames, ref, age_min, stations of the Germany sunny source)
  Rain alerts (server/alerts.py, Web Push):
  GET  /api/push/key                           -> {"key": VAPID public key}
  POST /api/push/subscribe   {id?, subscription, settings: {places, own, lead, early, quiet, quietFrom, quietTo, thunder},
                              ownLoc, ownName, tz, api} -> {"id"}   (needs the login session: 401 without)
  POST /api/push/unsubscribe {id?, all?} (all: every device of the logged-in user)
  POST /api/push/test {id};  POST /api/push/remind {id, lat, lon, name, minutes}
  GET  /api/push/preview?lat=&lon=&lead=&tz=   -> what the alert logic sees at a place (sends nothing)
  Optional login to sync favourites and settings (server/auth.py; Cache-Control no-store, credentialed CORS only for
  our own site, writes need a same-site Origin):
  GET  /api/me                                 -> {"email", "providers": {"google", "email"}} (401, email null: logged out)
  GET  /api/auth/google?return=/path           -> 302 to Google (OpenID Connect + PKCE); 503 when not configured
  GET  /api/auth/google/callback               -> session cookie rainy_session, 302 to the return path
  POST /api/auth/email {email, lang, return}   -> {"ok": true} and a sign-in link by mail (valid 15 min, single use)
  GET  /api/auth/email/verify?token=&return=&lang= -> page with a "Sign in" button (mail scanners must not use up links)
  POST /api/auth/email/verify (form token, return) -> session cookie, 302 to the return path (?login=expired if invalid)
  POST /api/auth/logout                        -> {"ok": true}
  GET  /api/user/data                          -> {"data": {...}, "updated": ms};  PUT/POST {data} (object, max 64 KB)
  DELETE /api/user (or POST /api/user/delete)  -> deletes the account with its sessions, data and open login links

Run:  python server/rainy_server.py   (port from env PORT, default 8771; env EDITION=de|full, SUNNY_DE=0 switches the
      Germany sunny source off)
Requests to data sources carry a neutral User-Agent and no identifying information.
"""
from collections import OrderedDict
import contextlib
import io
import json
import math
import os
import re
import sys
import threading
import time
import urllib.parse
import urllib.request
from datetime import datetime, timedelta, timezone
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from urllib.parse import parse_qs, urlparse

import cv2
import h5py
import numpy as np
from PIL import Image
from pyproj import Transformer

import alerts
import auth
import bench_inca
import clouds
import fcverify
import gauges
import monitor
import mosaic
import national
import palette  # stepped rain colours and the rain tile look (same table as the app)
import ptype
import recorder
import sunny_de
import thunder

try:  # pySTEPS builds on Linux (Docker image); without it the simple nowcast is used
    from pysteps import motion as ps_motion, nowcasts as ps_nowcasts
    from pysteps.utils import transformation as ps_transformation
    HAVE_PYSTEPS = True
except ImportError:
    HAVE_PYSTEPS = False

VERSION = "server-v1.24"
# Edition: "de" = first public version (Germany: DWD radar + DWD nowcast, rain alerts, weather model, sunshine from DWD
# satellite data (sunny_de.py); no Europe/USA radar, no EUMETSAT clouds - their loops are not started); "full" = everything;
# "eu" = the Europe test instance next to the Germany one (deploy/docker-compose.yml "api-eu", behind /eu/ on the same
# domain): EUMETNET OPERA radar + our own 2 h nowcast for all of Europe and its verification - no DWD source, no USA,
# no satellite clouds or sunny, no alerts, so that it fits next to the Germany edition on the same 8 GB server.
EDITION = os.environ.get("EDITION", "de")
DEV = os.environ.get("RAINY_DEV") == "1"  # local development only: fake scenarios + dry-run alert outbox (devtools.py)
PORT = int(os.environ.get("PORT", "8771"))
UA = "Mozilla/5.0"          # neutral, carries no identity
HISTORY_MS = 60 * 60_000    # observed frames kept (1 h)
FC_STEPS = 24               # nowcast steps of 5 min (2 h)
STEP_MS = 5 * 60_000
POLL_S = 60
NODATA = np.float32(-1.0)   # outside radar coverage
# Europe grid in km (OPERA comes at 1 km). 2 = a quarter of the memory per frame (8 MB instead of 33 MB, float16) and
# S-PROG at 4 km: the Europe test instance must fit next to the Germany edition (server: 8 GB, no swap at first).
EU_KM = int(os.environ.get("EU_KM", "2" if EDITION == "eu" else "1"))
# Self-hosted Open-Meteo. In production Caddy routes /om to it; locally this server proxies /om when it runs.
OPENMETEO_URL = os.environ.get("OPENMETEO_URL", "http://127.0.0.1:8080")
_om_state = {"at": 0.0, "ok": False}


def openmeteo_available():
    if os.environ.get("OPENMETEO") == "1":
        return True
    if time.time() - _om_state["at"] > 60:
        try:
            urllib.request.urlopen(f"{OPENMETEO_URL}/v1/forecast?latitude=0&longitude=0&hourly=temperature_2m&forecast_hours=1", timeout=3).read()
            _om_state["ok"] = True
        except Exception:
            _om_state["ok"] = False
        _om_state["at"] = time.time()
    return _om_state["ok"]

def rss_mb():
    try:
        with open("/proc/self/status") as f:
            return next(int(l.split()[1]) // 1024 for l in f if l.startswith("VmRSS"))
    except (OSError, StopIteration):
        return None


def memory_loop():
    """Give freed heap back to the OS regularly (glibc keeps it otherwise) and log the memory use."""
    try:
        import ctypes
        trim = ctypes.CDLL("libc.so.6").malloc_trim
    except OSError:
        trim = None
    n = 0
    while True:
        time.sleep(30)
        if trim:
            trim(0)
        n += 1
        if n % 20 == 0:
            log(f"[{datetime.now():%H:%M:%S}] memory {rss_mb()} MB")


def log(msg):
    """Server messages go to stderr: pySTEPS' progress output on stdout is silenced while it runs."""
    sys.stderr.write(msg + "\n")
    sys.stderr.flush()
    monitor.note(msg)  # error messages are counted for the daily report (monitor.py)


def dbz_to_rate(dbz):
    """Marshall-Palmer Z = 200 R^1.6; echoes below 10 dBZ (~0.15 mm/h) count as dry."""
    return np.where(dbz >= 10.0, (10 ** (dbz / 10.0) / 200.0) ** (1 / 1.6), 0.0).astype(np.float32)


def fetch(url, timeout=40, tries=3):
    req = urllib.request.Request(url, headers={"User-Agent": UA})
    for attempt in range(tries):
        try:
            with urllib.request.urlopen(req, timeout=timeout) as r:
                return r.read()
        except OSError:  # short network hiccups: try again after a pause
            if attempt == tries - 1:
                raise
            time.sleep(5 * (attempt + 1))


# ---------- nowcast ----------
def nowcast(prev, last, dt_ms, steps, smooth=1.0, down=2):
    """Extrapolate `last` along the motion observed between `prev` and `last` (dt_ms apart)."""
    ny, nx = last.shape

    def img(r):
        v = np.where(r > 0, np.log1p(np.clip(r, 0, 50)), 0)
        return np.clip(v / math.log1p(50) * 255, 0, 255).astype(np.uint8)
    small = (nx // down, ny // down)  # flow on reduced resolution for speed and stability
    a = cv2.resize(img(prev), small, interpolation=cv2.INTER_AREA)
    b = cv2.resize(img(last), small, interpolation=cv2.INTER_AREA)
    flow = cv2.calcOpticalFlowFarneback(a, b, None, 0.5, 4, 25, 4, 7, 1.5, 0)
    flow = cv2.GaussianBlur(flow, (0, 0), 6)
    flow = cv2.resize(flow, (nx, ny), interpolation=cv2.INTER_LINEAR) * float(down)  # pixels per dt
    flow *= STEP_MS / dt_ms                                                  # pixels per 5 min step
    gx, gy = np.meshgrid(np.arange(nx, dtype=np.float32), np.arange(ny, dtype=np.float32))
    base = np.where(last < 0, 0, last).astype(np.float32)
    out = []
    for k in range(1, steps + 1):
        r = cv2.remap(base, gx - flow[..., 0] * k, gy - flow[..., 1] * k, cv2.INTER_LINEAR,
                      borderMode=cv2.BORDER_CONSTANT, borderValue=0)
        if smooth:
            r = cv2.GaussianBlur(r, (0, 0), smooth * math.sqrt(k))  # grows with lead time
        r[last < 0] = NODATA  # keep the coverage outline stable
        out.append(r)
    return out


# ---------- weather model rain for the second hour ----------
MODEL_REFRESH_S = 15 * 60
MODEL_MAX_AGE_MS = 3 * 3600_000
DATA_DIR = os.environ.get("DATA_DIR", os.path.join(os.path.dirname(os.path.abspath(__file__)), "data"))  # push subscriptions, keys
# contact claim required by Web Push; set it to the app's own domain in production (no personal address)
VAPID_SUBJECT = os.environ.get("VAPID_SUBJECT", "https://rainy.invalid")
MODEL_LOG_DIR = os.environ.get("MODEL_LOG_DIR", os.path.join(os.path.dirname(os.path.abspath(__file__)), "model_log"))
MODEL_LOG_KEEP_S = 48 * 3600
# radar weight: radar only for the first hour, then fading to an even mix at 2 h (to be tuned with the logs)
BLEND_START_MIN, BLEND_END_MIN, BLEND_END_W = 60, 120, 0.5


def radar_weight(lead_min):
    if lead_min <= BLEND_START_MIN:
        return 1.0
    f = min(1.0, (lead_min - BLEND_START_MIN) / (BLEND_END_MIN - BLEND_START_MIN))
    return 1.0 - f * (1.0 - BLEND_END_W)


class ModelRain:
    """Precipitation forecast (15 min steps) of the best national model per place, on a coarse lat/lon grid over a
    radar region, from the self-hosted Open-Meteo. Only grid points inside the radar coverage are requested."""
    BATCH = 2000
    MAP_DOWN = 8  # model -> radar resampling is computed on a coarser grid and scaled up (the model is smooth)

    def __init__(self, region):
        self.region, self.deg = region, region.MODEL_DEG
        self.lock = threading.Lock()
        self.times = self.rate = None   # times: (T,) ms at the middle of each 15 min interval; rate: (T, nlat, nlon) mm/h
        self.issued = 0
        self.points = self.lats = self.lons = self.maps = None

    def _setup(self, g):
        ny, nx = g.shape
        # bounding box of the radar grid in lat/lon, from its edges
        e = np.linspace(0, 1, 200)
        cols = np.concatenate([e * nx, e * nx, np.zeros(200), np.full(200, nx)])
        rows = np.concatenate([np.zeros(200), np.full(200, ny), e * ny, e * ny])
        lon, lat = self.region.pixel_lonlat(cols, rows)
        d = self.deg
        self.lats = np.arange(math.floor(lat.min() / d) * d, lat.max() + d, d)
        self.lons = np.arange(math.floor(lon.min() / d) * d, lon.max() + d, d)
        la, lo = np.meshgrid(self.lats, self.lons, indexing="ij")
        x, y = ll_to_3857.transform(lo.ravel(), np.clip(la.ravel(), -85, 85))
        col, row = (np.round(v).astype(np.int64) for v in self.region.to_pixels(x, y))
        ok = (col >= 0) & (col < nx) & (row >= 0) & (row < ny)
        ok[ok] = g[row[ok], col[ok]] >= 0
        dom = getattr(self.region, "MODEL_DOMAIN", None)
        if dom:  # one point outside the model's area fails the whole request (HTTP 400 "No data is available")
            lat_all, lon_all = la.ravel(), lo.ravel()
            ok &= (lat_all >= dom[0]) & (lat_all <= dom[1]) & (lon_all >= dom[2]) & (lon_all <= dom[3])
        self.points = np.flatnonzero(ok)
        # where each (reduced) radar pixel lies in the model grid
        sy, sx = ny // self.MAP_DOWN, nx // self.MAP_DOWN
        rc, rr = np.meshgrid((np.arange(sx) + 0.5) * nx / sx - 0.5, (np.arange(sy) + 0.5) * ny / sy - 0.5)
        lon, lat = self.region.pixel_lonlat(rc.ravel(), rr.ravel())
        self.maps = (((lon - self.lons[0]) / d).reshape(sy, sx).astype(np.float32),
                     ((lat - self.lats[0]) / d).reshape(sy, sx).astype(np.float32))

    def refresh(self):
        g = self.region.latest()
        if g is None:
            return 0
        if self.points is None:
            self._setup(g)
        la, lo = np.meshgrid(self.lats, self.lons, indexing="ij")
        la, lo = la.ravel()[self.points], lo.ravel()[self.points]
        vals, times = None, None
        for i in range(0, len(self.points), self.BATCH):
            body = urllib.parse.urlencode({
                "latitude": ",".join(f"{v:.3f}" for v in la[i:i + self.BATCH]),
                "longitude": ",".join(f"{v:.3f}" for v in lo[i:i + self.BATCH]),
                "minutely_15": "precipitation", "past_minutely_15": "1", "forecast_minutely_15": "14",
                "timeformat": "unixtime",
                **({"models": self.region.MODELS} if getattr(self.region, "MODELS", None) else {})}).encode()
            req = urllib.request.Request(f"{OPENMETEO_URL}/v1/forecast", data=body,
                                         headers={"Content-Type": "application/x-www-form-urlencoded"})
            with urllib.request.urlopen(req, timeout=600) as r:
                d = json.loads(r.read())
            d = d if isinstance(d, list) else [d]
            if times is None:
                times = np.array(d[0]["minutely_15"]["time"], np.int64) * 1000 - 450_000  # value = sum of the previous 15 min
                vals = np.zeros((len(times), len(self.points)), np.float32)
            for j, e in enumerate(d):
                vals[:, i + j] = [v or 0.0 for v in e["minutely_15"]["precipitation"]]
        rate = np.zeros((len(times), len(self.lats) * len(self.lons)), np.float32)
        rate[:, self.points] = vals * 4.0  # mm per 15 min -> mm/h
        rate = rate.reshape(len(times), len(self.lats), len(self.lons))
        issued = int(time.time() * 1000)
        with self.lock:
            self.times, self.rate, self.issued = times, rate, issued
        self._log(issued, times, rate)
        return len(self.points)

    def _log(self, issued, times, rate):
        try:
            os.makedirs(MODEL_LOG_DIR, exist_ok=True)
            np.savez_compressed(os.path.join(MODEL_LOG_DIR, f"{self.region.name}_{issued}.npz"), times=times,
                                rate=rate.astype(np.float16), lat0=self.lats[0], lon0=self.lons[0], deg=self.deg)
            for f in os.listdir(MODEL_LOG_DIR):
                fp = os.path.join(MODEL_LOG_DIR, f)
                if time.time() - os.path.getmtime(fp) > MODEL_LOG_KEEP_S:
                    os.remove(fp)
        except OSError as e:
            log(f"model log failed: {e}")

    def field(self, ts, shape, times=None, rate=None):
        """Model rain rate at time ts on the radar grid (default: the latest model data), or None."""
        if rate is None:
            with self.lock:
                times, rate = self.times, self.rate
        if rate is None or not times[0] <= ts <= times[-1]:
            return None
        k = min(int(np.searchsorted(times, ts, side="right")) - 1, len(times) - 2)
        f = (ts - times[k]) / (times[k + 1] - times[k])
        m = ((1 - f) * rate[k] + f * rate[k + 1]).astype(np.float32)
        small = cv2.remap(m, self.maps[0], self.maps[1], cv2.INTER_LINEAR, borderMode=cv2.BORDER_CONSTANT, borderValue=0)
        return cv2.resize(small, (shape[1], shape[0]), interpolation=cv2.INTER_LINEAR)

    def blend(self, fc, ref, times=None, rate=None, weight=radar_weight):
        """Mix the radar nowcast with the model rain, more so the longer the lead time."""
        if rate is None:
            with self.lock:
                times, rate, issued = self.times, self.rate, self.issued
            if rate is None or ref - issued > MODEL_MAX_AGE_MS:
                return fc
        for i, g in enumerate(fc):
            w = weight((i + 1) * STEP_MS / 60_000)
            if w >= 1:
                continue
            m = self.field(ref + (i + 1) * STEP_MS, g.shape, times, rate)
            if m is None:
                continue
            cov = g >= 0
            g[cov] = w * g[cov] + (1 - w) * m[cov]
        return fc


# "new cells" blend (candidate): the radar nowcast stays as it is; in the second hour the model only adds showers
# where the radar nowcast has no rain within NEW_RADIUS_KM - i.e. cells that have not formed yet.
NEW_START_MIN, NEW_END_MIN, NEW_END_W, NEW_RADIUS_KM = 60, 120, 0.7, 10.0


def new_cells_weight(lead_min):
    if lead_min <= NEW_START_MIN:
        return 0.0
    return min(1.0, (lead_min - NEW_START_MIN) / (NEW_END_MIN - NEW_START_MIN)) * NEW_END_W


def blend_new_cells(model, fc, ref, times=None, rate=None, km_per_px=None, weight=new_cells_weight):
    if rate is None:
        with model.lock:
            times, rate, issued = model.times, model.rate, model.issued
        if rate is None or ref - issued > MODEL_MAX_AGE_MS:
            return fc
    r = max(1, int(round(NEW_RADIUS_KM / (km_per_px or model.region.KM_PER_PX))))
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (2 * r + 1, 2 * r + 1))
    for i, g in enumerate(fc):
        w = weight((i + 1) * STEP_MS / 60_000)
        if w <= 0:
            continue
        m = model.field(ref + (i + 1) * STEP_MS, g.shape, times, rate)
        if m is None:
            continue
        cov = g >= 0
        near = cv2.dilate((np.where(cov, g, 0) >= 0.1).astype(np.uint8), kernel) > 0
        add = cov & ~near
        g[add] = np.maximum(g[add], w * m[add])
    return fc


def model_loop(model):
    while True:
        started = time.time()
        try:
            if openmeteo_available():
                n = model.refresh()
                if n:
                    log(f"[{datetime.now():%H:%M:%S}] {model.region.name}: model rain for {n} points, {time.time() - started:.0f}s")
        except Exception as e:
            log(f"{model.region.name} model update failed: {e}")
        time.sleep(max(60, MODEL_REFRESH_S - (time.time() - started)) if model.rate is not None else 60)


def _coarse(g, down):
    """Block mean to a coarser grid; blocks without any coverage stay NODATA."""
    g = g.astype(np.float32)
    if down == 1:
        return g
    ny, nx = g.shape
    b = g[: ny // down * down, : nx // down * down].reshape(ny // down, down, nx // down, down)
    return np.where((b >= 0).any(axis=(1, 3)), np.where(b < 0, 0, b).mean(axis=(1, 3)), NODATA).astype(np.float32)


SPROG_LOCK = threading.Lock()  # S-PROG needs a lot of memory: never run two regions at the same time


def sprog_nowcast(seq, mot, steps, down, thr=0.1, probmatch="cdf", levels=6, ar_order=2, motion_out=None):
    """pySTEPS S-PROG: like `nowcast`, but each spatial scale decays at its own observed rate (small showers fade
    fast, large rain areas persist). seq: frames 10/5/0 min old; mot: frames 30/15/0 min old (for the motion).
    Runs on a grid `down` times coarser and is scaled back up. Backtests: equal to `nowcast` for widespread rain,
    clearly better for showers in the second hour (user-view CSI 60-120 min, USA: 0.42 vs 0.35)."""
    last = seq[-1].astype(np.float32)
    ny, nx = last.shape
    zero = 10 * math.log10(thr) - 5.0  # dB value for "dry", below the threshold
    dbz = lambda gs: ps_transformation.dB_transform(np.stack(gs), threshold=thr, zerovalue=zero)[0]
    m = [_coarse(g, down) for g in mot]
    velocity = ps_motion.get_method("LK")(dbz([np.where(g < 0, np.nan, g) for g in m])) / 3.0  # per 15 min -> per 5 min
    if motion_out is not None:  # coarse pixels per 5 min, for advect_blend
        motion_out["velocity"], motion_out["down"] = velocity, down
    s = [_coarse(g, down) for g in seq]
    with SPROG_LOCK, contextlib.redirect_stdout(io.StringIO()):  # one region at a time; pySTEPS prints long tables
        fc = ps_nowcasts.get_method("sprog")(dbz([np.where(g < 0, 0, g) for g in s]), velocity, steps,
                                             precip_thr=10 * math.log10(thr),  # threshold in dB
                                             n_cascade_levels=levels, ar_order=ar_order, num_workers=min(4, os.cpu_count() or 1), probmatching_method=probmatch)
    out = []
    for i in range(len(fc)):  # frame by frame: converting all at once would copy the whole stack several times
        g = np.where(np.isfinite(fc[i]), 10.0 ** (fc[i].astype(np.float32) / 10.0), 0.0).astype(np.float32)  # dB -> mm/h
        g[g < thr] = 0  # below the threshold counts as dry, as in pySTEPS' inverse dB transform
        g = cv2.resize(g, (nx, ny), interpolation=cv2.INTER_LINEAR) if down > 1 else g
        g[g < 0.05] = 0
        g[last < 0] = NODATA
        out.append(g.astype(np.float16))  # compact right away: 24 full-size float32 frames would need ~1.6 GB
    return out


# The step from the last radar image to the first forecast step was a visible break (10.11., live data tiles):
# S-PROG lets small cells decay at once (rainy: small cells 67 -> 40 -> 20 in the first 15 min; Europe on its 4 km
# S-PROG grid: area -14 % and small cells 218 -> 69 -> 18 at +5 min) - while "nothing changes" scored better at +15
# (Europe 0.98 vs 0.96). So in the first hour the last image itself is moved along S-PROG's motion (sharp, every cell
# kept) and handed over to S-PROG gradually: weight 1 - lead / ADV_FADE_MIN. Pure S-PROG stays a scored shadow.
ADV_FADE_MIN = 60


def advect_blend(fc, last, motion, fade_min=ADV_FADE_MIN):
    """fc: S-PROG steps (5 min apart, float16 grids, NODATA outside); last: the newest analysis (same grid); motion:
    sprog_nowcast(motion_out=...). Returns the blended steps (float16)."""
    vel, down = motion.get("velocity"), motion.get("down", 1)
    if vel is None:
        return fc
    last = np.asarray(last, np.float32)
    ny, nx = last.shape
    u = cv2.resize(np.nan_to_num(vel[0]).astype(np.float32), (nx, ny), interpolation=cv2.INTER_LINEAR) * down
    v = cv2.resize(np.nan_to_num(vel[1]).astype(np.float32), (nx, ny), interpolation=cv2.INTER_LINEAR) * down
    gx, gy = np.meshgrid(np.arange(nx, dtype=np.float32), np.arange(ny, dtype=np.float32))
    src = np.where(last < 0, 0, last).astype(np.float32)
    out = []
    for i, g in enumerate(fc):
        lead = (i + 1) * STEP_MS / 60_000
        w = max(0.0, 1.0 - lead / fade_min)
        if w <= 0:
            out.append(g)
            continue
        adv = cv2.remap(src, gx - u * (i + 1), gy - v * (i + 1), cv2.INTER_LINEAR, borderMode=cv2.BORDER_CONSTANT, borderValue=0)
        gf = np.asarray(g, np.float32)
        b = w * adv + (1.0 - w) * np.where(gf < 0, 0, gf)
        b[b < 0.05] = 0
        b[gf < 0] = NODATA
        out.append(b.astype(np.float16))
    return out


# ---------- our own nowcast for Germany ----------
class DeModelArea:
    """The German radar grid as a ModelRain region: model points only where the map shows the radar."""
    name, MODEL_DEG = "de_model", 0.05  # ICON-D2 has ~2.2 km; 0.05 deg (~4-5 km) is plenty for blending
    MODELS = MODEL_DOMAIN = None        # best model per place, no area limit (ModelRain reads both)

    def __init__(self, de):
        self.de, self.KM_PER_PX = de, de.KM_PER_PX
        self.to_ll = Transformer.from_crs("EPSG:3857", "EPSG:4326", always_xy=True)

    def latest(self):
        return np.where(self.de.weight > 0, 0.0, NODATA).astype(np.float32) if self.de.ref else None

    def to_pixels(self, x, y):
        return self.de.to_pixels(x, y)

    def pixel_lonlat(self, col, row):
        x0, _, _, y1 = self.de.BBOX
        return self.to_ll.transform(x0 + (np.asarray(col) + 0.5) * self.de.CELL, y1 - (np.asarray(row) + 0.5) * self.de.CELL)


class OwnNowcast:
    """Our own 2 h forecast on the DWD radar analyses, next to DWD's (RV) - selectable in the app, scored by
    fcverify.py against the later radar image just like DWD's. pySTEPS S-PROG: motion per region (Lucas-Kanade
    optical flow, so showers in different areas can move in different directions) and each spatial scale decays at
    its own observed rate (small showers and drizzle fade, large rain areas persist) - DWD's RV only shifts the
    image. Inputs: analyses 10/5/0 min old (S-PROG) and 30/15/0 min old (motion).
    Pure S-PROG since server-v0.66 (07.11.): mixing in the DWD model (ICON-D2) in the second hour, like DWD's
    SINFONY/INTENSE, made it worse on the same runs (30 h: CSI >= 1 mm/h +120 .31 vs .40; 6 h drizzle: >= 0.1 mm/h
    +120 .27 vs .40, amount .77 vs .98). The model variants stay as shadows "model" (smearing + new showers) and
    "cells" (new showers only) for fcverify, so the model is measured, not assumed."""
    DOWN = 2    # ~2 km (1 km: +0.01..0.03 CSI for twice the CPU - not worth it yet)
    # wet threshold for S-PROG: 0.1 (pySTEPS default) turned light drizzle into "dry" and missed most of it;
    # server/tune_own.py, 8 DWD runs 03./04.11. (drizzle), CSI +15/+30/+60/+90: 0.1 -> 0.38/0.24/0.13/0.01,
    # 0.03 -> 0.72/0.50/0.22/0.04, DWD RV 0.66/0.40/0.23/0.09
    THR = 0.03

    def __init__(self, de):
        self.de, self.done = de, None
        self.model = ModelRain(DeModelArea(de))

    def step(self):
        de = self.de
        if not HAVE_PYSTEPS or getattr(de, "frozen", False):
            return
        with de.lock:
            ref, obs = de.ref, dict(de.obs)
        if ref is None or ref == self.done or ref not in obs:
            return
        near = lambda dt: obs.get(ref - dt * 60_000)
        seq, mot = [near(10), near(5), near(0)], [near(30), near(15), near(0)]
        if any(g is None for g in seq + mot):
            return
        started = time.time()
        motion = {}
        fc0 = sprog_nowcast([g.astype(np.float32) for g in seq], [g.astype(np.float32) for g in mot], FC_STEPS, self.DOWN,
                            thr=self.THR, motion_out=motion)
        fc = advect_blend(fc0, seq[-1], motion)  # no break at "now" (ADV_FADE_MIN)
        pure = {ref + (i + 1) * STEP_MS: g for i, g in enumerate(fc)}
        sprog0 = {ref + (i + 1) * STEP_MS: g.astype(np.float16) for i, g in enumerate(fc0)}  # shadow: before 10.11.
        mixed = [g.astype(np.float32) for g in fc]
        mixed = blend_new_cells(self.model, self.model.blend(mixed, ref), ref, km_per_px=de.KM_PER_PX)
        model = {ref + (i + 1) * STEP_MS: g.astype(np.float16) for i, g in enumerate(mixed)}
        grids = {t: g.astype(np.float16) for t, g in pure.items()}
        # 07.11. (30 h, 361 runs): the model blend lowers CSI >= 1 mm/h at +120 from .40 (pure S-PROG) to .31 but keeps
        # the amount near 1 (pure: 1.26). Which half does it: the smearing towards the model (Region.blend) or the
        # model's new showers (blend_new_cells)? Shadow "cells" = S-PROG + new showers only, no smearing.
        # ("mix" = DWD to +30 fading to own was dropped: by construction it equals dwd early and own late.)
        cells = blend_new_cells(self.model, [g.astype(np.float32) for g in fc], ref, km_per_px=de.KM_PER_PX)
        cells = {ref + (i + 1) * STEP_MS: g.astype(np.float16) for i, g in enumerate(cells)}
        dwd_new = self._dwd_new_cells(ref)  # takes de.lock itself: not inside the block below
        with de.lock:
            if de.ref == ref:
                de.set_own(ref, grids)
                # scored by fcverify only: pure S-PROG; DWD's own forecast plus the model's new showers in the 2nd hour
                de.fc_shadow = {"model": model, "dwdnew": dwd_new, "cells": cells, "sprog0": sprog0}
        self.done = ref
        log(f"[{datetime.now():%H:%M:%S}] own nowcast: run {datetime.fromtimestamp(ref / 1000, timezone.utc):%H:%M}Z in {time.time() - started:.0f}s")

    def _dwd_new_cells(self, ref):
        with self.de.lock:
            fc = dict(self.de.fc)
        times = sorted(fc)
        out = blend_new_cells(self.model, [fc[t].astype(np.float32) for t in times], ref, km_per_px=self.de.KM_PER_PX)
        return {t: g.astype(np.float16) for t, g in zip(times, out)}

    def loop(self):
        while True:
            try:
                self.step()
            except Exception as e:
                log(f"own nowcast failed: {e}")
            time.sleep(10)


# ---------- regions ----------
class Region:
    name = ""
    credit = ""
    SMOOTH = 1.0      # nowcast smoothing in grid pixels (tuned at ~2-3 km)
    FLOW_DOWN = 2     # optical flow is computed at grid resolution / FLOW_DOWN
    MODEL_DEG = None  # lat/lon spacing of the weather model rain used for blending (None: no blending)
    SPROG_DOWN = 0    # S-PROG runs at grid resolution / SPROG_DOWN (0: simple nowcast only)
    RAW_DOWN = 3      # the forecast without the model's new cells (fc_raw) is kept at grid resolution / RAW_DOWN
    MODELS = None     # Open-Meteo "models" for the blend (None: best match per place)
    # A 5-minute step the source skipped is filled with the previous run's forecast for it (regions whose source
    # delivers every 5 minutes). 14.11.2026 02:40 UTC: OPERA skipped one image - for an hour one step of the
    # timeline was empty all over Europe (found by check_tile_gaps: "wet 0 between 620 and 617") and three runs fell
    # back to the simple nowcast (S-PROG wants the images 5, 10, 15 and 30 min back; the simple one showed +40 %
    # rain area, a jump at "now"). The own picture has such holes more often: a step waits up to 20 min for a
    # national layer while the next one is already there. The real image replaces the fill when it arrives.
    FILL_GAPS = False
    SPROG_THR = 0.1   # S-PROG's rain / no rain threshold (mm/h)

    def __init__(self):
        self.lock = threading.Lock()
        self.obs, self.fc, self.ref = {}, {}, None
        self.filled = set()  # steps in obs that are a forecast standing in for a missing image
        self.model = None

    def latest(self):
        with self.lock:
            return self.obs[max(self.obs)] if self.obs else None

    # subclasses: list_times() -> {ts: source}, load(source) -> grid, to_pixels(x3857, y3857) -> (col, row)
    def frames(self):
        with self.lock:
            return ([{"ts": t, "fc": False} for t in sorted(self.obs)] +
                    [{"ts": t, "fc": True} for t in sorted(self.fc)]), self.ref

    def grid(self, ts):
        with self.lock:
            return self.obs.get(ts) if ts in self.obs else self.fc.get(ts)

    def update(self):
        now = int(time.time() * 1000)
        # 15 min more than the hour the timeline shows: its first step (Mosaic.frames, now - 60 min rounded down) must
        # still have its image when it is asked for a few minutes later (10.11.: the first Europe step was empty)
        cutoff = now - HISTORY_MS - 15 * 60_000
        avail = {t: s for t, s in self.list_times().items() if t > cutoff}
        new = [t for t in sorted(avail) if t not in self.obs or t in self.filled]
        for t in new:
            g = self.load(avail[t]).astype(np.float16)  # half the memory; plenty precise for rain rates
            with self.lock:
                self.obs[t] = g
                self.filled.discard(t)
        if not new:
            return 0
        with self.lock:
            for t in list(self.obs):
                if t not in avail and not (t in self.filled and t > cutoff):
                    del self.obs[t]
                    self.filled.discard(t)
            times = sorted(self.obs)
            if self.FILL_GAPS:
                for a, b in zip(times, times[1:]):
                    if STEP_MS < b - a <= 3 * STEP_MS:  # one or two images missing; longer holes stay visible
                        for t in range(a + STEP_MS, b, STEP_MS):
                            if self.fc.get(t) is not None:
                                self.obs[t] = self.fc[t]
                                self.filled.add(t)
                                log(f"{self.name}: no image for {datetime.fromtimestamp(t / 1000, timezone.utc):%H:%MZ} "
                                    f"- filled with the previous run's forecast")
                times = sorted(self.obs)
            last_t = times[-1]
            prev_t = next((t for t in reversed(times) if last_t - t >= 29 * 60_000), times[0])
            prev, last = self.obs[prev_t], self.obs[last_t]
        if prev_t == last_t:
            return len(new)
        fc = None
        motion, pure = {}, None
        if HAVE_PYSTEPS and self.SPROG_DOWN:
            with self.lock:
                near = lambda dt, tol: next((self.obs[t] for t in sorted(self.obs, key=lambda t: abs(last_t - dt - t))
                                             if abs(last_t - dt - t) <= tol), None)
                seq = [near(k * 60_000, 120_000) for k in (10, 5)] + [last]
                mot = [near(k * 60_000, 240_000) for k in (30, 15)] + [last]
            if all(g is not None for g in seq + mot):
                try:
                    pure = sprog_nowcast(seq, mot, FC_STEPS, self.SPROG_DOWN, thr=self.SPROG_THR, motion_out=motion)
                    fc = advect_blend(pure, last, motion)  # no break at "now" (ADV_FADE_MIN)
                except Exception as e:  # never lose the forecast: fall back to the simple method
                    log(f"{self.name} S-PROG failed, using simple nowcast: {e}")
        if fc is None:
            fc = nowcast(prev.astype(np.float32), last.astype(np.float32), last_t - prev_t, FC_STEPS, self.SMOOTH, self.FLOW_DOWN)
        raw = cells = None
        if self.model:
            # The weather model's new showers in the second hour (blend_new_cells) are no longer served, only scored
            # as the shadow "cells" (like the Germany edition since 07.11.). Europe, 10./11.11., 175 runs with the
            # model working: CSI >= 0.1 mm/h +90/+120 0.71/0.66 with them, 0.76/0.71 without (cell by cell 0.39/0.35
            # vs 0.48/0.44), rain amount x1.15/x1.39 vs x0.94/x0.93. And they were visible: the model's wide rain
            # areas were added everywhere except within NEW_RADIUS_KM of the radar rain, which left dry lanes of the
            # same width between every rain band and the model rain around it (the user's "roads", 10.11.).
            d = self.RAW_DOWN
            small = [cv2.resize(np.asarray(g, np.float32), (g.shape[1] // d, g.shape[0] // d), interpolation=cv2.INTER_AREA)
                     for g in fc]
            raw = {last_t + (i + 1) * STEP_MS: g.astype(np.float16) for i, g in enumerate(small)}  # = fc, at 1/d
            cells = blend_new_cells(self.model, small, last_t, km_per_px=self.KM_PER_PX * d)  # changes `small` in place
            cells = {last_t + (i + 1) * STEP_MS: g.astype(np.float16) for i, g in enumerate(cells)}
        d = self.RAW_DOWN
        pure_small = None if pure is None else {
            last_t + (i + 1) * STEP_MS: cv2.resize(np.asarray(g, np.float32), (g.shape[1] // d, g.shape[0] // d),
                                                   interpolation=cv2.INTER_AREA).astype(np.float16) for i, g in enumerate(pure)}
        with self.lock:
            self.fc_raw = raw
            self.fc_cells = cells  # verification shadow "cells": with the model's new showers (served until 11.11.)
            self.fc_pure = pure_small  # S-PROG as before 10.11. (no advection blend): verification shadow "sprog0"
            self.fc = {last_t + (i + 1) * STEP_MS: g.astype(np.float16) for i, g in enumerate(fc)}
            self.ref = last_t
        self.after_forecast()
        return len(new)

    def after_forecast(self):
        """Called after every new forecast run (subclasses: verification notes)."""


class EuropeOpera(Region):
    name, credit = "eu", "EUMETNET OPERA"
    # The provider's two open buckets with the same layout: the regular one, and the backup it moves to when its cloud
    # provider has trouble (11.11.2026 20:10 UTC onwards the regular bucket got no new files for a day and the map was
    # hatched; the backup had them all along). Each image is taken from the first bucket that lists it.
    BUCKETS = ("https://s3.waw3-1.cloudferro.com/openradar-24h",
               "https://object-store.os-api.cci1.ecmwf.int/openradar-24h-backup")
    LAEA = "+proj=laea +lat_0=55.0 +lon_0=10.0 +x_0=1950000.0 +y_0=-2100000.0 +units=m +ellps=WGS84"
    CELL, NY, NX = 1000.0 * EU_KM, 4400 // EU_KM, 3800 // EU_KM   # OPERA's 1 km grid, kept at EU_KM km
    KM_PER_PX = float(EU_KM)
    SMOOTH, FLOW_DOWN = 2.0 / EU_KM, max(1, 4 // EU_KM)  # same physical scales as the backtested 2 km setup
    MODEL_DEG = 0.15
    # S-PROG at 3 km (1 km grid) or 4 km (2 km grid): about half the memory of 2 km (peak ~5.3 GB was too close to the
    # laptop Docker limit)
    SPROG_DOWN = 3 if EU_KM == 1 else 2
    RAW_DOWN = 3 if EU_KM == 1 else 2   # S-PROG alone, at ~3-4 km: the verification's "sprog" variant
    # Europe test instance: ICON-EU only - it is on the server's disk (openmeteo-sync); "best match" read AROME, HARMONIE,
    # MET Nordic ... from the remote mirror for ~60k points per refresh (took > 15 min locally)
    MODELS = os.environ.get("EU_MODELS") or None
    # ICON-EU's area (lat0, lat1, lon0, lon1): until 10.11. the points north of 70.5 N made every refresh fail (400)
    MODEL_DOMAIN = (29.6, 70.4, -23.4, 62.4) if MODELS == "icon_eu" else None
    # OPERA DBZH via Marshall-Palmer shows 1.94x the rain of ~300 German 10-min gauges (22 h, server/check_opera_gauges.py)
    # while DWD shows 0.65x; x0.5 matches the gauges, gives the best CSI and removes the step at the German border.
    GAIN = 0.5

    PX_CACHE = 400  # tile grids whose pixel positions are kept (~0.5 MB each)
    # Radar interference spokes are taken out of every image (spokes.py). In the own picture since v1.15, in the OPERA
    # instance (the app's "European" source) since v1.19 (user, 13.11.2026: yes). check_spokes.py switches it off to
    # see the raw images.
    DESPOKE = True
    FILL_GAPS = True
    # The map draws rain from 0.05 mm/h; with S-PROG's threshold at 0.1 the forecast dropped everything lighter and
    # lost rain area fast (user, 13.11.: "rain only up to now"; tile_gaps.log: -30 % of the wet pixels by +60 min).
    # tune_eu.py, 5 runs of the night 13./14.11.2026, cell by cell on the 2 km grid, +30 / +60 / +120 min:
    # wet area forecast / real 0.92 / 0.82 / 0.82 with 0.1, 1.00 / 1.01 / 1.01 with 0.03; CSI at 0.05 mm/h
    # 0.57 / 0.46 / 0.36 -> 0.60 / 0.50 / 0.39; at 0.3 and 1 mm/h unchanged (0.50 / 0.38 / 0.28, 0.42 / 0.30 /
    # 0.20); amount 0.98 / 0.97 / 1.03. Germany's own forecast has used 0.03 since 04.11.
    SPROG_THR = 0.03
    gauge_px = None  # rain gauges of check_national_gauges.py on this grid: (col, row, on the grid)

    def gauges(self):
        """The gauges' pixels. The OPERA instance takes the list the own-picture instance keeps (same gauges, same
        order: the two instances' forecasts are compared gauge by gauge, check_forecast_gauges.py)."""
        if self.gauge_px is None:
            import check_national_gauges as cng
            shared = os.path.join(cng.LOG_DIR, "eun", "natobs", "stations.json")
            if not EU_NATIONAL and os.path.exists(shared):
                with open(shared) as f:
                    st = [tuple(s) for s in json.load(f)]
            else:
                st = cng.station_list(fetch, log)
            self.gauge_px = cng.pixels(self, st, mosaic.merc) if st else None
        return self.gauge_px

    def after_forecast(self):
        """Every forecast run of a Europe instance is noted at the rain gauges: the picture at the run's time and
        the forecast for +30 / +60 / +90 / +120 min (check_forecast_gauges.py scores both instances against the
        gauges - the only comparison of the two that does not use either one's own radar picture as the truth)."""
        if EDITION != "eu":
            return
        try:
            import check_national_gauges as cng
            px = self.gauges()
            if px is None:
                return
            with self.lock:
                ref = self.ref
                grids = [self.obs.get(ref)] + [self.fc.get(ref + m * 60_000) for m in cng.FC_LEADS]
                # the same run's other forecasts, kept at 1 / RAW_DOWN: S-PROG alone, and with the model's new showers
                more = {name: [d.get(ref + m * 60_000) for m in cng.FC_LEADS]
                        for name, d in (("pure", getattr(self, "fc_pure", None)), ("cells", getattr(self, "fc_cells", None))) if d}
            if ref % 600_000 or any(g is None for g in grids):  # the gauges report every 10 minutes
                return
            d = self.RAW_DOWN
            small = (px[0] // d, px[1] // d, px[2])
            cng.record_fc(ref, np.stack([cng.at_gauges(g, px) for g in grids]),
                          {name: np.stack([cng.at_gauges(g, small, 1) for g in gs])
                           for name, gs in more.items() if all(g is not None for g in gs)})
        except Exception as e:
            log(f"fcgauge failed: {e}")

    def __init__(self):
        super().__init__()
        self.px_cache, self.px_lock = OrderedDict(), threading.Lock()
        self.tr = Transformer.from_crs("EPSG:3857", self.LAEA, always_xy=True)
        self.to_ll = Transformer.from_crs(self.LAEA, "EPSG:4326", always_xy=True)
        import spokes
        self.spokes = spokes.SpokeFilter(self.KM_PER_PX, spokes.sites(self, mosaic.merc))
        self.spoke_log = {}  # radar code -> when its line was last logged
        self.spoke_last = None  # (time step, [(length km, radar code, (col, row))]) of the newest image

    def despoke(self, url, rate):
        if not self.DESPOKE:
            return rate
        try:
            stamp = url.split("OPERA@")[1].split("@")[0]
            ts = int(datetime.strptime(stamp, "%Y%m%dT%H%M").replace(tzinfo=timezone.utc).timestamp() * 1000)
            rate, lines = self.spokes.clean(ts, rate)
            self.spoke_last = (ts, lines)
            for length, code, (c, r) in lines:
                if time.time() - self.spoke_log.get(code, 0) > 3600:
                    self.spoke_log[code] = time.time()
                    lon, lat = self.pixel_lonlat(c, r)
                    log(f"spoke removed: {length} km towards radar {code}, middle {float(lat):.2f} N {float(lon):.2f} E")
        except Exception as e:
            log(f"spoke filter failed: {e}")
        return rate

    def pixel_lonlat(self, col, row):
        return self.to_ll.transform((np.asarray(col) + 0.5) * self.CELL, -(np.asarray(row) + 0.5) * self.CELL)

    def list_times(self):
        now = datetime.now(timezone.utc)
        days = [now] + ([now - timedelta(days=1)] if now.hour == 0 else [])
        out = {}
        failed = []
        for bucket in self.BUCKETS:
            for day in days:
                try:
                    xml = fetch(f"{bucket}/?prefix={day:%Y/%m/%d}/OPERA/COMP/&max-keys=1000").decode()
                except Exception as e:
                    failed.append(e)
                    continue
                for key in re.findall(r"<Key>([^<]+@DBZH\.h5)</Key>", xml):
                    stamp = key.split("OPERA@")[1].split("@")[0]
                    t = int(datetime.strptime(stamp, "%Y%m%dT%H%M").replace(tzinfo=timezone.utc).timestamp() * 1000)
                    out.setdefault(t, f"{bucket}/{key}")
        if not out and failed:
            raise failed[0]
        newest = max(out, default=0)
        if time.time() * 1000 - newest > 25 * 60_000 and time.time() - self.warned > 1800:  # normally 5-10 min old
            self.warned = time.time()
            log(f"eu radar source failed: newest image {datetime.fromtimestamp(newest / 1000, timezone.utc):%d.%m. %H:%MZ}"
                if newest else "eu radar source failed: no images listed in any bucket")
        return out

    warned = 0.0
    BUCKET = BUCKETS[0]  # for check_opera_gauges.py

    def load(self, url):
        with h5py.File(io.BytesIO(fetch(url)), "r") as f:
            d = f["dataset1/data1/data"][:]
            w = f["dataset1/data1/what"].attrs
            gain, offset, nodata, undetect = (float(w[k]) for k in ("gain", "offset", "nodata", "undetect"))
        nod = d == nodata
        dbz = d * gain + offset
        dbz[(d == undetect) | nod] = -32.0
        rate = cv2.medianBlur(dbz_to_rate(dbz), 3) * self.GAIN   # median removes single-pixel clutter
        if EU_KM > 1:  # mean over the covered 1 km pixels; no data where most of the block is outside the radar range
            k = EU_KM
            h, w = rate.shape[0] // k * k, rate.shape[1] // k * k
            cov = (~nod[:h, :w]).reshape(h // k, k, w // k, k).mean(axis=(1, 3))
            tot = np.where(nod[:h, :w], 0, rate[:h, :w]).reshape(h // k, k, w // k, k).mean(axis=(1, 3))
            rate = np.where(cov >= 0.5, tot / np.maximum(cov, 1e-6), NODATA).astype(np.float32)
            return self.despoke(url, rate)
        rate[nod] = NODATA
        return self.despoke(url, rate)

    def to_pixels(self, x, y):
        """Fractional pixel centres. Map tiles: the projection (pyproj, ~2/3 of a Europe tile's render time) is the
        same for every time step of a tile, so the result is kept per tile grid (LRU, float32, ~0.5 MB each): the app
        loads all 36 time steps of a view, the pre-render loop too - 1 transform instead of 36 (Europe tile 13 -> ~5 ms)."""
        big = np.ndim(x) == 2 and np.size(x) >= 4096
        if big:
            key = (float(x[0, 0]), float(y[0, 0]), float(x[-1, -1]), float(y[-1, -1]), np.shape(x))
            with self.px_lock:
                hit = self.px_cache.get(key)
                if hit is not None:
                    self.px_cache.move_to_end(key)
                    return hit
        gx, gy = self.tr.transform(x, y)
        out = np.asarray(gx) / self.CELL - 0.5, -np.asarray(gy) / self.CELL - 0.5
        if big:
            out = tuple(a.astype(np.float32) for a in out)
            with self.px_lock:
                self.px_cache[key] = out
                while len(self.px_cache) > self.PX_CACHE:
                    self.px_cache.popitem(last=False)
        return out


class EuropeNational(EuropeOpera):
    """Our own Europe picture (EU_NATIONAL=1, the third instance behind /eun/): OPERA with national radar products
    laid on top (national.py) - same grid, nowcast, tiles and verification as the OPERA instance."""
    credit = "EUMETNET OPERA + national radar services"

    def __init__(self):
        super().__init__()
        self.started = time.time()
        self.layers = national.start(self, log, UA)
        self.used = {}  # ts -> names of the layers in that image (health)
        self.gauge_px = None  # gauges at which OPERA alone and the composed picture are noted (check_national_gauges.py)

    def list_times(self):
        times = super().list_times()
        now = time.time() * 1000
        # the layers load their last hour first: without them the first images would be composed from OPERA alone
        if time.time() - self.started < 120 and not all(l.first_poll_done for l in self.layers):
            return {}
        alive = [l for l in self.layers if l.alive() and not l.SHADOW]  # a shadow is not waited for
        return {t: (t, url) for t, url in times.items()
                if now - t > national.WAIT_MIN * 60_000 or all(l.has(t) for l in alive)}

    def load(self, src):
        ts, url = src
        rate = super().load(url)  # OPERA as the Europe instance shows it (spokes already out)
        opera = None
        try:
            import check_national_gauges as cng
            if self.gauges() is not None:
                opera = cng.at_gauges(rate, self.gauge_px)
        except Exception as e:
            log(f"natobs failed: {e}")
        self.used[ts] = national.compose(rate, self.layers, ts)
        if opera is not None:
            try:
                col, row, ok = self.gauge_px
                who = national.cover_at(self.layers, ts, np.where(ok, col, -1), np.where(ok, row, -1))
                each = national.values_at(self.layers, ts, np.where(ok, col, -1), np.where(ok, row, -1))
                cng.record(ts, opera, cng.at_gauges(rate, self.gauge_px), who, [l.name for l in self.layers], each,
                           [l.name for l in self.layers if l.SHADOW])
            except Exception as e:
                log(f"natobs failed: {e}")
        for t in [t for t in self.used if t < ts - 3 * HISTORY_MS]:
            del self.used[t]
        return rate


class UsaNowcoast(Region):
    name, credit = "us", "NOAA"
    WMS = "https://nowcoast.noaa.gov/geoserver/observations/weather_radar/ows"
    BBOX = (-14000000.0, 2700000.0, -7400000.0, 6500000.0)  # CONUS in EPSG:3857
    NX, NY = 2200, 1267
    KM_PER_PX = 2.3                     # 3 km Web Mercator pixels at ~40 N
    MODEL_DEG = 0.2
    SPROG_DOWN = 1
    RAMP = [(6, "599494"), (8, "3bb1b1"), (9, "1dcece"), (10, "00ecec"), (11, "00d9ee"), (13, "00c6f1"), (14, "00b3f3"),
            (15, "01a0f6"), (16, "0078f6"), (18, "0050f6"), (19, "0028f6"), (20, "0000f6"), (21, "003fb8"), (23, "007f7b"),
            (24, "00bf3d"), (25, "00ff00"), (26, "00f100"), (28, "00e300"), (29, "00d500"), (30, "00c800"), (31, "00ba00"),
            (33, "00ac00"), (34, "009e00"), (35, "009000"), (36, "3fab00"), (38, "7fc700"), (39, "bfe300"), (40, "ffff00"),
            (41, "f9ef00"), (43, "f3df00"), (44, "edcf00"), (45, "e7c000"), (46, "edb400"), (48, "f3a800"), (49, "f99c00"),
            (50, "ff9000"), (51, "ff6c00"), (53, "ff4800"), (54, "ff2400"), (55, "ff0000"), (56, "f60000"), (58, "ed0000"),
            (59, "e40000"), (60, "dc0000"), (61, "d50000"), (63, "ce0000"), (64, "c70000"), (65, "c00000"), (66, "cf003f"),
            (68, "df007f"), (69, "ef00bf"), (70, "ff00ff"), (71, "e515f1"), (73, "cc2ae4"), (74, "b23fd6"), (75, "9955c9"),
            (76, "747bce"), (78, "4fa1d4"), (79, "2ac7da"), (80, "05ede0")]

    def __init__(self):
        super().__init__()
        self.ramp_rgb = np.array([[int(h[i:i + 2], 16) for i in (0, 2, 4)] for _, h in self.RAMP], np.int32)
        self.ramp_dbz = np.array([d for d, _ in self.RAMP], np.float32)

    def list_times(self):
        xml = fetch(f"{self.WMS}?service=WMS&version=1.3.0&request=GetCapabilities").decode("utf-8", "ignore")
        m = re.search(r"<Name>base_reflectivity_mosaic</Name>.*?<Dimension name=\"time\"[^>]*>([^<]*)", xml, re.S)
        if not m:
            return {}
        return {int(datetime.strptime(t.strip(), "%Y-%m-%dT%H:%M:%S.000Z").replace(tzinfo=timezone.utc).timestamp() * 1000): t.strip()
                for t in m.group(1).split(",") if t.strip()}

    def load(self, t):
        x0, y0, x1, y1 = self.BBOX
        url = (f"{self.WMS}?service=WMS&version=1.3.0&request=GetMap&layers=base_reflectivity_mosaic&styles="
               f"&crs=EPSG:3857&bbox={x0},{y0},{x1},{y1}&width={self.NX}&height={self.NY}&format=image/png&transparent=true&time={t}")
        px = np.asarray(Image.open(io.BytesIO(fetch(url))).convert("RGBA"), np.int32)
        rgb = px[..., :3].reshape(-1, 3)
        packed = (rgb[:, 0] << 16) | (rgb[:, 1] << 8) | rgb[:, 2]
        uniq, inv = np.unique(packed, return_inverse=True)
        ucol = np.stack([(uniq >> 16) & 255, (uniq >> 8) & 255, uniq & 255], 1)
        d2 = ((ucol[:, None, :] - self.ramp_rgb[None, :, :]) ** 2).sum(-1)
        best = d2.argmin(1)
        udbz = np.where(d2[np.arange(len(uniq)), best] <= 3 * 40 * 40, self.ramp_dbz[best], -32.0)  # nearest ramp colour
        dbz = udbz[inv].reshape(self.NY, self.NX).astype(np.float32)
        dbz[px[..., 3] < 100] = -32.0
        return cv2.medianBlur(dbz_to_rate(dbz), 3)

    def to_pixels(self, x, y):
        x0, y0, x1, y1 = self.BBOX
        col = (np.asarray(x) - x0) / (x1 - x0) * self.NX - 0.5
        row = (y1 - np.asarray(y)) / (y1 - y0) * self.NY - 0.5
        return col, row

    def pixel_lonlat(self, col, row):
        x0, y0, x1, y1 = self.BBOX
        x = x0 + (np.asarray(col) + 0.5) / self.NX * (x1 - x0)
        y = y1 - (np.asarray(row) + 0.5) / self.NY * (y1 - y0)
        return ll_to_3857.transform(x, y, direction="INVERSE")


class SatelliteRain(Region):
    """EUMETSAT H SAF blended satellite rain rate (h40b, MTG FCI + microwave). Coarse, but available everywhere
    on the Meteosat disc; used only where no radar exists."""
    name, credit = "sat", "EUMETSAT H SAF"
    WMS = "https://view.eumetsat.int/geoserver/ows"
    BBOX = (-6679169.0, -4865942.0, 6679169.0, 11818999.0)  # 60W-60E, 40S-72N in EPSG:3857
    NX, NY = 1670, 2085                                     # ~8 km
    SMOOTH, FLOW_DOWN = 0.5, 1
    # legend intervals (upper bounds in mm/h) -> representative rate
    CLASSES = [("ccffcc", 0.8), ("99e699", 3), ("66cc66", 5.5), ("33b333", 8.5), ("3399cc", 12.5), ("3366ff", 17.5),
               ("0000ff", 22.5), ("6600cc", 27.5), ("9933cc", 35), ("cc0099", 45), ("800080", 60)]

    def __init__(self):
        super().__init__()
        self.rgb = np.array([[int(h[i:i + 2], 16) for i in (0, 2, 4)] for h, _ in self.CLASSES], np.int32)
        self.val = np.array([v for _, v in self.CLASSES], np.float32)

    def list_times(self):
        xml = fetch(f"{self.WMS[:-4]}/mtg_fd/h40b/ows?service=WMS&version=1.3.0&request=GetCapabilities").decode("utf-8", "ignore")
        m = re.search(r'<Dimension name="time"[^>]*>[^<]*/([0-9T:.\-]+Z)/PT10M', xml)
        if not m:
            return {}
        end = int(datetime.strptime(m.group(1)[:19], "%Y-%m-%dT%H:%M:%S").replace(tzinfo=timezone.utc).timestamp() * 1000)
        return {end - k * 600_000: end - k * 600_000 for k in range(7)}

    def load(self, t):
        x0, y0, x1, y1 = self.BBOX
        stamp = datetime.fromtimestamp(t / 1000, timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")
        url = (f"{self.WMS}?service=WMS&version=1.1.1&request=GetMap&layers=mtg_fd:h40b&styles=&srs=EPSG:3857"
               f"&bbox={x0},{y0},{x1},{y1}&width={self.NX}&height={self.NY}&format=image/png&transparent=true&time={stamp}")
        px = np.asarray(Image.open(io.BytesIO(fetch(url, timeout=90))).convert("RGBA"), np.int32)
        rate = np.zeros((self.NY, self.NX), np.float32)
        wet = px[..., 3] > 100                       # only the (few) rain pixels need colour matching
        rgb = px[wet][:, :3]
        d2 = ((rgb[:, None, :] - self.rgb[None]) ** 2).sum(-1)
        best = d2.argmin(-1)
        rate[wet] = np.where(d2.min(-1) <= 3 * 30 * 30, self.val[best], 0.0)
        return rate

    def to_pixels(self, x, y):
        x0, y0, x1, y1 = self.BBOX
        return (np.asarray(x) - x0) / (x1 - x0) * self.NX - 0.5, (y1 - np.asarray(y)) / (y1 - y0) * self.NY - 0.5


ll_to_3857 = Transformer.from_crs("EPSG:4326", "EPSG:3857", always_xy=True)
EU_NATIONAL = EDITION == "eu" and os.environ.get("EU_NATIONAL") == "1"
REGIONS = {r.name: r for r in (EuropeNational() if EU_NATIONAL else EuropeOpera(), UsaNowcoast(), SatelliteRain())}


def update_loop(region):
    while True:
        started = time.time()
        try:
            n = region.update()
            if n:
                _, ref = region.frames()
                log(f"[{datetime.now():%H:%M:%S}] {region.name}: +{n} frames, latest "
                      f"{datetime.fromtimestamp(ref / 1000, timezone.utc):%H:%M}Z, {time.time() - started:.1f}s")
        except Exception as e:  # keep serving the last good data
            log(f"{region.name} update failed: {e}")
        time.sleep(POLL_S)


# ---------- rendering ----------
def view_coords(bbox, w, h):
    minx, miny, maxx, maxy = bbox
    xs = minx + (np.arange(w) + 0.5) / w * (maxx - minx)
    ys = maxy - (np.arange(h) + 0.5) / h * (maxy - miny)
    return np.meshgrid(xs, ys)


def sample(region, grid, X, Y):
    """Bilinear interpolation between grid pixels (smooth edges instead of blocks); returns rate and coverage.

    Only the window of the grid around the points is converted (the EU grid is 4400 x 3800: converting all of it
    per tile or point cost ~200 MB and seconds each, which made parallel tile requests run out of memory). For a
    dense output grid (map tiles) much coarser than the data, the window is decimated to ~2 data pixels per output
    pixel."""
    col, row = (np.asarray(v, np.float64) for v in region.to_pixels(X, Y))
    h, w = grid.shape
    c0, c1 = max(int(np.floor(col.min())) - 1, 0), min(int(np.ceil(col.max())) + 2, w)
    r0, r1 = max(int(np.floor(row.min())) - 1, 0), min(int(np.ceil(row.max())) + 2, h)
    if c0 >= c1 or r0 >= r1:
        return np.zeros(X.shape, np.float32), np.zeros(X.shape, bool)
    step = 1
    if X.ndim == 2 and min(X.shape) >= 64:
        step = max(1, int(min((c1 - c0) / X.shape[1], (r1 - r0) / X.shape[0]) / 2))
    g = grid[r0:r1:step, c0:c1:step].astype(np.float32)
    col, row = ((col - c0) / step).astype(np.float32), ((row - r0) / step).astype(np.float32)
    rate = cv2.remap(np.where(g < 0, 0, g), col, row, cv2.INTER_LINEAR, borderMode=cv2.BORDER_CONSTANT, borderValue=0)
    cov = cv2.remap((g >= 0).astype(np.float32), col, row, cv2.INTER_LINEAR, borderMode=cv2.BORDER_CONSTANT, borderValue=0)
    return rate, cov > 0.5


def nearest_grid(region, ts, tol_ms=16 * 60_000):
    frames, _ = region.frames()
    if not frames:
        return None
    best = min(frames, key=lambda f: abs(f["ts"] - ts))
    return region.grid(best["ts"]) if abs(best["ts"] - ts) <= tol_ms else None


def render(region, grid, bbox, w, h, ts):
    X, Y = view_coords(bbox, w, h)
    rate, cov = sample(region, grid, X, Y)
    rgba = palette.colorize(rate)
    sat = REGIONS["sat"]
    if region is not sat and not cov.all():
        sgrid = nearest_grid(sat, ts)
        if sgrid is not None:  # satellite estimate where the radar sees nothing, drawn fainter
            srate, _ = sample(sat, sgrid, X, Y)
            srgba = palette.colorize(srate)
            srgba[..., 3] = (srgba[..., 3] * 0.65).astype(np.uint8)
            rgba[~cov] = srgba[~cov]
    elif region is sat:
        rgba[..., 3] = (rgba[..., 3] * 0.65).astype(np.uint8)
    buf = io.BytesIO()
    Image.fromarray(rgba, "RGBA").save(buf, "PNG", compress_level=3)
    return buf.getvalue()


def coverage_mask(bbox, w, h, kind="rain"):
    """Opaque where any radar (Europe from OPERA's real coverage, CONUS box for the USA) exists;
    in the Germany edition exactly Germany. kind "sun" in the Europe instance: the Europe sunny area."""
    X, Y = view_coords(bbox, w, h)
    cov = np.zeros((h, w), bool)
    if kind == "sun" and EDITION == "eu":
        col, row = (v.astype(np.float32) for v in SUNNY.grid.to_pixels(X, Y))
        cov = cv2.remap(SUNNY.inside(), col, row, cv2.INTER_NEAREST, borderMode=cv2.BORDER_CONSTANT, borderValue=0) > 0.5
        img = np.zeros((h, w, 4), np.uint8)
        img[cov] = (0, 0, 0, 255)
        buf = io.BytesIO()
        Image.fromarray(img, "RGBA").save(buf, "PNG", compress_level=3)
        return buf.getvalue()
    if EDITION == "de":
        col, row = (v.astype(np.float32) for v in DE.to_pixels(X, Y))
        cov = cv2.remap(DE.inside, col, row, cv2.INTER_LINEAR, borderMode=cv2.BORDER_CONSTANT, borderValue=0) > 0.5
        img = np.zeros((h, w, 4), np.uint8)
        img[cov] = (0, 0, 0, 255)
        buf = io.BytesIO()
        Image.fromarray(img, "RGBA").save(buf, "PNG", compress_level=3)
        return buf.getvalue()
    eu = REGIONS["eu"]
    frames, _ = eu.frames()
    if frames:
        cov |= sample(eu, eu.grid(frames[0]["ts"]), X, Y)[1]
    if EDITION == "full":
        x0, y0, x1, y1 = UsaNowcoast.BBOX
        cov |= (X >= x0) & (X <= x1) & (Y >= y0) & (Y <= y1)
    img = np.zeros((h, w, 4), np.uint8)
    img[cov] = (0, 0, 0, 255)
    buf = io.BytesIO()
    Image.fromarray(img, "RGBA").save(buf, "PNG", compress_level=3)
    return buf.getvalue()


def point_series(lat, lon):
    x, y = ll_to_3857.transform(lon, lat)
    for region in REGIONS.values():
        col, row = (int(round(float(v))) for v in region.to_pixels(x, y))
        frames, ref = region.frames()
        if not frames:
            continue
        g0 = region.grid(frames[0]["ts"])
        ny, nx = g0.shape
        if not (1 <= col < nx - 1 and 1 <= row < ny - 1):
            continue
        if (g0[row - 1:row + 2, col - 1:col + 2] < 0).all():
            continue  # outside radar coverage
        out = []
        for fr in frames:
            g = region.grid(fr["ts"])
            if g is None:
                continue
            cell = g[row - 1:row + 2, col - 1:col + 2]
            out.append({"ts": fr["ts"], "mmh": round(float(cell[cell >= 0].mean()) if (cell >= 0).any() else 0.0, 3)})
        return region.name, out, ref
    return None, [], None


POINT_CACHE = {}  # (radar key, lat, lon, with DWD) -> rain point answer; the key changes with every new radar run


GEO_CACHE = {}  # proxied geocoding answers (rounded coordinates / query text) -> bytes
GEO_UP = {"/api/geo/reverse": ("https://photon.komoot.io/reverse", ("lat", "lon", "lang")),
          "/api/geo/search": ("https://geocoding-api.open-meteo.com/v1/search", ("name", "count", "language", "format"))}


class Handler(BaseHTTPRequestHandler):
    def geo(self, u):
        """Forward a geocoding request with only the needed parameters, our neutral User-Agent and no client data."""
        up, keys = GEO_UP[u.path]
        q = {k: v[0][:120] for k, v in parse_qs(u.query).items() if k in keys}
        if "lat" in q:
            q["lat"], q["lon"] = f"{float(q['lat']):.3f}", f"{float(q['lon']):.3f}"  # ~100 m
        url = f"{up}?{urllib.parse.urlencode(sorted(q.items()))}"
        body = GEO_CACHE.get(url)
        if body is None:
            body = fetch(url, timeout=20)
            GEO_CACHE[url] = body
            while len(GEO_CACHE) > 50000:
                GEO_CACHE.pop(next(iter(GEO_CACHE)))
        return self.send(200, body, "application/json", 86400)

    server_version = "rainy"
    sys_version = ""

    def log_message(self, *a):
        pass

    def send(self, code, body, ctype, cache=60):
        self.send_response(code)
        self.send_header("Content-Type", ctype)
        self.send_header("Access-Control-Allow-Origin", "*")
        self.send_header("Cache-Control", f"public, max-age={cache}")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def do_GET(self):
        u = urlparse(self.path)
        if auth.is_auth_path(u.path):  # login, session, synced user data (server/auth.py)
            return AUTH.handle(self, "GET", u)
        q = {k: v[0] for k, v in parse_qs(u.query).items()}
        try:
            if u.path.startswith("/om/"):  # local stand-in for the production /om route
                with urllib.request.urlopen(f"{OPENMETEO_URL}{self.path[3:]}", timeout=60) as r:
                    return self.send(r.status, r.read(), r.headers.get("Content-Type", "application/json"), 600)
            if u.path == "/api/ready":  # for uptime monitors: 503 when the German radar data is stale
                src_ref = REGIONS["eu"].ref if EDITION == "eu" else DE.ref
                age = (time.time() * 1000 - src_ref) / 60_000 if src_ref else None
                ok = age is not None and age < 20
                return self.send(200 if ok else 503, json.dumps({"ok": ok, "radar_age_min": age and round(age)}).encode(), "application/json", 0)
            if u.path == "/api/health":
                info = {name: {"frames": len(r.frames()[0]), "ref": r.ref} for name, r in REGIONS.items()}
                info["clouds"] = {"frames": len(CLOUDS.frames()[0]), "ref": CLOUDS.ref}
                info["sunny"] = SUNNY.health()  # Germany edition's sunny source (DWD satellite)
                info["de"] = {"frames": len(DE.frames()[0]), "ref": DE.ref, "vetoed_px": getattr(DE, "vetoed", 0)}
                if EU_NATIONAL:  # per layer: its newest image, and in how many of the map's images it is
                    eu_r = REGIONS["eu"]
                    info["layers"] = {l.name: {"newest": l.newest_ts(), "alive": l.alive(), "shadow": l.SHADOW,
                                               "in_frames": sum(l.name in v for t, v in eu_r.used.items() if t in eu_r.obs)}
                                      for l in eu_r.layers}
                if REGIONS["eu"].spoke_last:  # interference lines removed from the newest image: (km, radar)
                    info["spokes"] = {"ref": REGIONS["eu"].spoke_last[0], "lines": [l[:2] for l in REGIONS["eu"].spoke_last[1]]}
                info["ptype"] = {"frames": len(PTYPE.grids), "ref": max(PTYPE.grids) if PTYPE.grids else None}
                body = {"version": VERSION, "edition": EDITION, "regions": info, "openmeteo": openmeteo_available(), "rss_mb": rss_mb(),
                        "auth": AUTH.providers(),  # sign-in methods the app may offer
                        "tiles": MOSAIC.render_summary()}  # rain tile render time per zoom (server CPU)
                return self.send(200, json.dumps(body).encode(), "application/json", 10)
            if u.path.startswith("/tiles/raindata"):  # data tiles for the app's WebGL map (mosaic.rain_data_tile)
                m = re.fullmatch(r"/tiles/(raindata|raindataown|raindataeu)/([\w-]+)/(\d+)-(\d+)/(\d+)/(\d+)/(\d+)\.png", u.path)
                if not m:
                    return self.send(404, b"not found", "text/plain", 60)
                kind, key = m.group(1), m.group(2)
                t0, n, z, x, y = map(int, m.groups()[2:])
                if not (0 <= x < (1 << z) and 0 <= y < (1 << z)):
                    return self.send(404, b"bad tile", "text/plain", 3600)
                png = MOSAIC.rain_data_tile(key, t0, n, z, x, y, use_de=kind != "raindataeu",
                                            variant="own" if kind == "raindataown" else "dwd")
                if png is None:
                    return self.send(404, b"bad tile", "text/plain", 3600)
                return self.send(200, png, "image/png", 86400 if MOSAIC.key_known(key) else 60)
            if u.path.startswith("/tiles/"):
                m = re.fullmatch(r"/tiles/(rain|rainown|raineu|cloud)/([\w-]+)/(\d+)/(\d+)/(\d+)/(\d+)\.png", u.path)
                if not m:
                    return self.send(404, b"not found", "text/plain", 60)
                kind, key, ts, z, x, y = m.group(1), m.group(2), *map(int, m.groups()[2:])
                variant = "own" if kind == "rainown" else "dwd"
                if not (2 <= z <= 13 and 0 <= x < (1 << z) and 0 <= y < (1 << z)):
                    return self.send(404, b"bad tile", "text/plain", 3600)
                png = (MOSAIC.cloud_tile(key, ts, z, x, y) if kind == "cloud"
                       else MOSAIC.rain_tile(key, ts, z, x, y, use_de=kind != "raineu", variant=variant))
                if png is None:
                    return self.send(404, b"unknown frame", "text/plain", 30)
                # the key names the run: immutable while that run is kept; an older key (a tab asleep for a while)
                # gets the current data, cached only briefly
                return self.send(200, png, "image/png", 86400 if (MOSAIC.cloud_key_known(key) if kind == "cloud" else MOSAIC.key_known(key)) else 60)
            if DEV and u.path.startswith("/api/dev/"):  # never in production (env RAINY_DEV=1 only)
                if u.path == "/api/dev/storm":
                    devtools.storm(DE, PTYPE, THUNDER, float(q["lat"]), float(q["lon"]), int(q.get("arrive", 40)))
                elif u.path == "/api/dev/field":  # widespread rain with all colour classes (look of the rain tiles)
                    devtools.field(DE, PTYPE, int(q.get("seed", 1)))
                elif u.path == "/api/dev/bench":  # render time per rain tile, new look vs. plain (no cache)
                    body = devtools.bench(MOSAIC, int(q.get("z", 8)), int(q.get("n", 36)))
                    return self.send(200, json.dumps(body).encode(), "application/json", 0)
                elif u.path == "/api/dev/reset":
                    devtools.reset(DE, PTYPE, THUNDER)
                    SUNNY.dev_reset()
                elif u.path == "/api/dev/sunreplay":  # archived DWD satellite files (folder in the container) as "now"
                    n = SUNNY.dev_replay(q["at"], q.get("dir", "/tmp/satcache"))
                    MOSAIC.cache.clear()
                    return self.send(200, json.dumps({"slots": n, "ref": SUNNY.ref}).encode(), "application/json", 0)
                elif u.path == "/api/dev/run":  # one alert round now
                    ALERTS.run_once(ll_to_3857)
                elif u.path == "/api/dev/due":  # all reminders due now
                    with ALERTS.lock:
                        for e in ALERTS.subs.values():
                            for r in e.get("reminders", []):
                                r["due"] = 0
                MOSAIC.cache.clear()
                POINT_CACHE.clear()
                return self.send(200, json.dumps({"outbox": devtools.OUTBOX[-10:]}, ensure_ascii=False).encode(), "application/json", 0)
            if u.path in ("/api/geo/reverse", "/api/geo/search"):  # place names / search, anonymised for the user
                return self.geo(u)
            if u.path == "/api/thunder/areas":  # DWD thunderstorm / heavy rain warning areas (GeoJSON)
                with THUNDER.lock:
                    body = THUNDER.geojson
                return self.send(200, body, "application/json", 60)
            if u.path == "/api/thunder/point":
                ws = THUNDER.at(round(float(q["lat"]), 2), round(float(q["lon"]), 2))
                body = [{k: w[k] for k in ("group", "level", "severity", "onset", "expires")} for w in ws]
                return self.send(200, json.dumps(body).encode(), "application/json", 60)
            if u.path == "/api/lightning":  # latest satellite lightning over Germany: {ts, flashes, box}
                with THUNDER.lock:
                    li = THUNDER.lightning
                body = {"ts": li[0], "flashes": li[1]} if li else {}  # flash clusters of the last minutes [[lat, lon]]
                return self.send(200, json.dumps(body).encode(), "application/json", 60)
            if u.path == "/api/rain/frames":
                body = {"key": MOSAIC.key(q.get("fc", "dwd")), "frames": MOSAIC.frames()}
                return self.send(200, json.dumps(body).encode(), "application/json", 30)
            if u.path == "/api/rain/point":
                lat, lon, de = round(float(q["lat"]), 2), round(float(q["lon"]), 2), q.get("src") != "eu"
                variant = "own" if q.get("fc") == "own" else "dwd"  # forecast: DWD nowcast or our own (OwnNowcast)
                ck = (MOSAIC.key(variant), lat, lon, de)  # one answer per ~1 km square and radar run, shared by all users
                body = POINT_CACHE.get(ck)
                if body is None:
                    body = {"key": ck[0], "series": MOSAIC.rain_point(lat, lon, de, variant), "covered": MOSAIC.covered(lat, lon)}
                    POINT_CACHE[ck] = body
                    while len(POINT_CACHE) > 20000:
                        POINT_CACHE.pop(next(iter(POINT_CACHE)))
                return self.send(200, json.dumps(body).encode(), "application/json", 60)
            if u.path == "/api/cloud/frames":
                frames, ref = SKY.frames()
                return self.send(200, json.dumps({"frames": frames, "ref": ref}).encode(), "application/json", 30)
            if u.path == "/api/cloud/image":
                bbox = [float(v) for v in q["bbox"].split(",")]
                w, h = min(int(q["width"]), 2048), min(int(q["height"]), 2048)
                png = SKY.image(int(q["ts"]), bbox, w, h)
                return self.send(200, png, "image/png", 3600) if png else self.send(404, b"unknown frame", "text/plain", 30)
            if u.path == "/api/cloud/point":
                body = {"series": SKY.point(float(q["lat"]), float(q["lon"])), "ref": SKY.ref}
                return self.send(200, json.dumps(body).encode(), "application/json", 60)
            region = REGIONS.get(q.get("region", "eu"))
            if u.path == "/api/radar/frames":
                frames, ref = region.frames()
                return self.send(200, json.dumps({"frames": frames, "ref": ref}).encode(), "application/json", 30)
            if u.path == "/api/radar/image":
                grid = region.grid(int(q["ts"]))
                if grid is None:
                    return self.send(404, b"unknown frame", "text/plain", 30)
                bbox = [float(v) for v in q["bbox"].split(",")]
                w, h = min(int(q["width"]), 1024), min(int(q["height"]), 1024)
                return self.send(200, render(region, grid, bbox, w, h, int(q["ts"])), "image/png", 3600)
            if u.path == "/api/radar/mask":
                bbox = [float(v) for v in q["bbox"].split(",")]
                w, h = min(int(q["width"]), 1024), min(int(q["height"]), 1024)
                return self.send(200, coverage_mask(bbox, w, h, q.get("kind", "rain")), "image/png", 600)
            if u.path == "/api/radar/point":
                name, series, ref = point_series(float(q["lat"]), float(q["lon"]))
                return self.send(200, json.dumps({"region": name, "series": series, "ref": ref}).encode(), "application/json", 60)
            if u.path == "/api/push/key":
                return self.send(200, json.dumps({"key": ALERTS.public_key}).encode(), "application/json", 3600)
            if u.path == "/api/push/preview":  # what the alert logic sees at a place (for testing, sends nothing)
                s = ALERTS.series(float(q["lat"]), float(q["lon"]), ll_to_3857)
                cfg = {"lead": int(q.get("lead", 45)), "early": True, "quiet": False, "tz": q.get("tz", "UTC")}
                kind, info = alerts.AlertEngine.evaluate(s, int(time.time() * 1000), cfg, {}) if s else (None, {})
                text = alerts.AlertEngine.message(kind, info, "Test", cfg["tz"]) if kind else None
                body = {"would_send": kind, "message": text, "info": info,
                        "series": [{"ts": t, "fc": f, "wet_share": round(p, 2), "mmh": round(r, 2)} for t, f, p, r in (s or [])]}
                return self.send(200, json.dumps(body, ensure_ascii=False).encode(), "application/json", 0)
            return self.send(404, b"not found", "text/plain")
        except Exception as e:
            return self.send(400, str(e).encode(), "text/plain", 0)

    def do_OPTIONS(self):  # CORS preflight for the JSON POSTs below
        if auth.is_auth_path(urlparse(self.path).path) or urlparse(self.path).path in PUSH_ACCOUNT:
            return AUTH.preflight(self)
        self.send_response(204)
        self.send_header("Access-Control-Allow-Origin", "*")
        self.send_header("Access-Control-Allow-Methods", "GET, POST, OPTIONS")
        self.send_header("Access-Control-Allow-Headers", "Content-Type")
        self.send_header("Access-Control-Max-Age", "86400")
        self.end_headers()

    def do_PUT(self):
        u = urlparse(self.path)
        if auth.is_auth_path(u.path):
            return AUTH.handle(self, "PUT", u)
        return self.send(404, b"not found", "text/plain", 0)

    def do_DELETE(self):  # DELETE /api/user (account deletion)
        u = urlparse(self.path)
        if auth.is_auth_path(u.path):
            return AUTH.handle(self, "DELETE", u)
        return self.send(404, b"not found", "text/plain", 0)

    def push_account(self, path):
        """Subscribe / unsubscribe rain alerts: tied to the login session (credentialed CORS like the account API).
        Anonymous subscriptions are no longer created (alerts.py); old ones can still be removed by their id."""
        try:
            if not AUTH.same_site(self):
                return AUTH.reply(self, 403, {"error": "origin"})
            body = AUTH.read_json(self, 20_000)
            user = AUTH.session_info(self)
            if path == "/api/push/subscribe":
                if not user:
                    return AUTH.reply(self, 401, {"error": "login"})
                return AUTH.reply(self, 200, {"id": ALERTS.subscribe(body, user[0], user[2])})
            n = ALERTS.remove_user(user[0]) if body.get("all") and user else 0
            if body.get("id"):
                ALERTS.unsubscribe(str(body["id"]))
            return AUTH.reply(self, 200, {"ok": True, "removed": n})
        except Exception as e:
            log(f"push: {path} failed ({type(e).__name__})")
            return AUTH.reply(self, 400, {"error": "bad request"})

    def do_POST(self):
        u = urlparse(self.path)
        if auth.is_auth_path(u.path):  # own size limits (user data up to 64 KB)
            return AUTH.handle(self, "POST", u)
        if u.path in PUSH_ACCOUNT:
            return self.push_account(u.path)
        try:
            n = int(self.headers.get("Content-Length", "0"))
            if n > 20_000:
                return self.send(413, b"too large", "text/plain", 0)
            body = json.loads(self.rfile.read(n) or b"{}")
            if u.path == "/api/push/remind":  # 'remind me again' button of a notification
                ok = ALERTS.remind(str(body.get("id")), body)
                return self.send(200 if ok else 404, json.dumps({"ok": ok}).encode(), "application/json", 0)
            if u.path == "/api/push/test":
                ok = ALERTS.send(str(body.get("id")), {"title": "🌧 Regenwarnung ist aktiv",
                                                       "body": "So sieht eine Warnung aus, wenn Regen auf dich zukommt."})
                return self.send(200 if ok else 404, json.dumps({"ok": ok}).encode(), "application/json", 0)
            return self.send(404, b"not found", "text/plain", 0)
        except Exception as e:
            return self.send(400, str(e).encode(), "text/plain", 0)



CLOUDS = clouds.CloudNowcast(fetch, log, OPENMETEO_URL, openmeteo_available, ll_to_3857)
DE = mosaic.GermanyDwd(fetch, log)
PTYPE = ptype.PrecipType(fetch, log, DE)
DE.drizzle = gauges.GaugeDrizzle(fetch, log, DE, mosaic.merc, state_dir=os.path.join(DATA_DIR, "gauges"))
fcverify.TAG = VERSION
GRID_VERIFY = fcverify.GridVerify(DE, log)
EU_VERIFY = fcverify.RegionVerify(REGIONS["eu"], log) if EDITION == "eu" else None
OWN = OwnNowcast(DE)
# Germany edition: sunshine from DWD satellite data (sunny_de.py) behind the same cloud API and tiles as the full
# edition's EUMETSAT clouds, so the app's sunny mode is the same code for both
# Europe test instance: the same sunny for all of Europe (sunny_de.EUROPE: 4 km grid, ICON-EU, files eusun*)
SUNNY = (sunny_de.SunnyDE(fetch, log, None, OPENMETEO_URL, openmeteo_available, area=sunny_de.EUROPE) if EDITION == "eu"
         else sunny_de.SunnyDE(fetch, log, DE, OPENMETEO_URL, openmeteo_available))
SUNNY_ON = EDITION in ("de", "eu") and os.environ.get("SUNNY_DE", "1") != "0"
SKY = SUNNY if SUNNY_ON else CLOUDS
MOSAIC = mosaic.Mosaic(REGIONS, DE, SKY, sample, palette.paint, log, de_only=EDITION == "de", ptype=PTYPE)
THUNDER = thunder.Thunder(fetch, log)
ALERTS = alerts.AlertEngine(DATA_DIR, {**REGIONS, "de": DE}, log, UA, VAPID_SUBJECT,
                            order=("de",) if EDITION == "de" else ("eu", "us"), thunder=THUNDER,
                            kind_at=lambda lat, lon, ts: ptype.NAMES[int(MOSAIC.kind(ts, *(np.array([[v]]) for v in mosaic.merc(lon, lat)))[0, 0])]
                            if PTYPE.grids else "rain")
AUTH = auth.Auth(DATA_DIR, log)
PUSH_ACCOUNT = ("/api/push/subscribe", "/api/push/unsubscribe")  # need the session cookie (Handler.push_account)
AUTH.on_logout = ALERTS.remove_session
AUTH.on_delete = ALERTS.remove_user
AUTH.on_data = lambda user, data: ALERTS.apply_user(user, data.get("alerts"))
ALERTS.session_alive = AUTH.session_alive
RECORDER = recorder.Recorder(fetch, log, MOSAIC, CLOUDS, mosaic.merc)
if DEV:
    import devtools
    ALERTS.dev, ALERTS.outbox = True, devtools.OUTBOX
CLOUDS.heavy_lock = SPROG_LOCK  # cloud nowcast and S-PROG take turns (Docker VM memory)

def monitor_status():
    """For monitor.py's health rows: how old the newest radar and satellite image are (minutes), and the memory."""
    now = time.time() * 1000
    ref = REGIONS["eu"].ref if EDITION == "eu" else DE.ref
    sun = SUNNY.ref if SUNNY_ON else None
    return {"rss": rss_mb(), "rain_age": round((now - ref) / 60_000, 1) if ref else None,
            "sun_age": round((now - sun) / 60_000, 1) if sun else None}


def gauge_archive_loop():
    """Hourly: the last hours of all rain gauge networks into verify_log/gaugetruth (check_national_gauges.py) - the
    sources keep hours to days, the evaluation needs weeks."""
    import check_national_gauges as cng
    time.sleep(240)
    while True:
        try:
            st = cng.stations()
            if st:
                t0 = time.time()
                n = cng.archive_truth(st)
                log(f"gauge archive: {n} values of {len(st)} gauges, {time.time() - t0:.0f} s")
        except Exception as e:
            log(f"gauge archive failed: {e}")
        time.sleep(3600)


def tile_gap_loop():
    """Every 10 minutes: the rain data tiles of all instances as the app loads them, scanned for time steps that
    drop out (check_tile_gaps.py) -> verify_log/tile_gaps.log."""
    import check_tile_gaps
    time.sleep(420)
    while True:
        try:
            lines = check_tile_gaps.scan(brief=True)
            with open(os.path.join(os.environ.get("VERIFY_LOG_DIR", "."), "tile_gaps.log"), "a") as f:
                f.write("\n".join(lines) + "\n")
        except Exception as e:
            log(f"tile gap scan failed: {e}")
        time.sleep(600)


def main_eu():
    """Europe test instance: OPERA + own nowcast (+ the model's new showers), sunny for Europe, their verification."""
    monitor.start(EDITION, VERSION, monitor_status)
    eu = REGIONS["eu"]
    threading.Thread(target=update_loop, args=(eu,), daemon=True).start()
    if os.environ.get("EU_MODEL_BLEND", "1") != "0":  # only the verification shadow "cells" needs the weather model
        eu.model = ModelRain(eu)
        threading.Thread(target=model_loop, args=(eu.model,), daemon=True).start()
    threading.Thread(target=EU_VERIFY.loop, daemon=True).start()
    if not EU_NATIONAL:  # measurements that are needed once, not per instance: in the OPERA instance
        threading.Thread(target=gauge_archive_loop, daemon=True).start()
        threading.Thread(target=tile_gap_loop, daemon=True).start()
    # Where no radar reaches (Greece, W Turkey, east of Poland ...) the own picture drew the satellite rain estimate
    # (H SAF, ~30 min old, from cloud tops) faintly into the hatched area. User, 13.11.2026: it shows rain inside
    # the "no data" hatching and does not fit the radar picture next to it -> off (SAT_FILL=1 brings it back).
    if EU_NATIONAL and os.environ.get("SAT_FILL") == "1":
        threading.Thread(target=update_loop, args=(REGIONS["sat"],), daemon=True).start()
    # second benchmark: GeoSphere Austria's INCA nowcast against ours at the Austrian rain gauges (bench_inca.py)
    threading.Thread(target=bench_inca.IncaBench(fetch, log, eu, mosaic.merc).loop, daemon=True).start()
    if SUNNY_ON:
        threading.Thread(target=SUNNY.loop, daemon=True).start()
    if os.environ.get("PRERENDER", "1") != "0":  # the third instance draws its tiles on demand (CPU for the other two)
        threading.Thread(target=mosaic.prerender_box_loop, args=(MOSAIC, log, mosaic.EUROPE_BOX, mosaic.EU_PRERENDER_ZOOMS),
                         daemon=True).start()
    threading.Thread(target=memory_loop, daemon=True).start()
    log(f"nowcast: {'pySTEPS S-PROG' if HAVE_PYSTEPS else 'simple (pySTEPS not installed)'}, Europe grid {EU_KM} km"
        f"{', national layers: ' + ', '.join(l.name for l in eu.layers) if EU_NATIONAL else ''}")
    log(f"rainy {VERSION} (eu edition) on http://0.0.0.0:{PORT}")
    ThreadingHTTPServer(("0.0.0.0", PORT), Handler).serve_forever()


if __name__ == "__main__" and EDITION == "eu":
    main_eu()
elif __name__ == "__main__":
    monitor.start(EDITION, VERSION, monitor_status)
    threading.Thread(target=ALERTS.loop, args=(ll_to_3857,), daemon=True).start()
    if EDITION == "de" and not DEV:  # push message to the operator when an instance has no fresh images (opswatch.py)
        import opswatch
        threading.Thread(target=opswatch.OpsWatch(ALERTS, DATA_DIR, log).loop, daemon=True).start()
    threading.Thread(target=mosaic.de_loop, args=(DE, log, PTYPE), daemon=True).start()
    threading.Thread(target=DE.drizzle.loop, daemon=True).start()
    threading.Thread(target=GRID_VERIFY.loop, daemon=True).start()
    if EDITION == "de":
        threading.Thread(target=OWN.loop, daemon=True).start()
        threading.Thread(target=model_loop, args=(OWN.model,), daemon=True).start()
    if SUNNY_ON:
        threading.Thread(target=SUNNY.loop, daemon=True).start()
    if EDITION == "de":  # the daily record of all forecast comparisons (both instances share verify_log)
        import daily_report
        daily_report.start(log)
    threading.Thread(target=mosaic.prerender_loop, args=(MOSAIC, log), daemon=True).start()
    threading.Thread(target=RECORDER.loop, daemon=True).start()
    threading.Thread(target=THUNDER.loop, daemon=True).start()
    threading.Thread(target=memory_loop, daemon=True).start()
    for r in REGIONS.values() if EDITION == "full" else ():
        if r.name == "eu":
            threading.Thread(target=CLOUDS.loop, daemon=True).start()
        threading.Thread(target=update_loop, args=(r,), daemon=True).start()
        if r.MODEL_DEG:
            r.model = ModelRain(r)
            threading.Thread(target=model_loop, args=(r.model,), daemon=True).start()
    log(f"nowcast: {'pySTEPS S-PROG' if HAVE_PYSTEPS else 'simple (pySTEPS not installed)'}")
    log(f"rainy {VERSION} ({EDITION} edition) on http://0.0.0.0:{PORT}")
    ThreadingHTTPServer(("0.0.0.0", PORT), Handler).serve_forever()
