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
  GET /api/health
  GET /api/radar/frames?region=eu|us          -> {"frames": [{"ts": ms, "fc": bool}], "ref": ms}
  GET /api/radar/image?region=&ts=&bbox=&width=&height=   (bbox in EPSG:3857) -> PNG in the rainy palette
  GET /api/radar/point?lat=&lon=               -> {"region": "eu"|"us"|"sat"|null, "series": [{"ts", "mmh"}], "ref": ms}
  GET /api/radar/mask?bbox=&width=&height=     -> PNG, opaque where radar coverage exists (for the "no data" hatching)
  Seamless map tiles for the app (server/mosaic.py; DWD + OPERA + NOAA + satellite in one, CDN-cacheable):
  GET /api/rain/frames, /api/rain/point?lat=&lon=, /tiles/rain/<key>/<ts>/<z>/<x>/<y>.png, /tiles/cloud/...
  Clouds for the "sunny" map (server/clouds.py, Europe, EUMETSAT GeoColour + nowcast):
  GET /api/cloud/frames                        -> {"frames": [{"ts", "fc"}], "ref": ms}
  GET /api/cloud/image?ts=&bbox=&width=&height= -> grey + alpha PNG, grey = cloudiness 0..255
  GET /api/cloud/point?lat=&lon=               -> {"series": [{"ts", "fc", "cloud": 0..1}], "ref": ms}
  Rain alerts (server/alerts.py, Web Push):
  GET  /api/push/key                           -> {"key": VAPID public key}
  POST /api/push/subscribe   {id?, subscription, places, lead, early, quiet, tz} -> {"id"}
  POST /api/push/unsubscribe {id};  POST /api/push/test {id}
  GET  /api/push/preview?lat=&lon=&lead=&tz=   -> what the alert logic sees at a place (sends nothing)

Run:  python server/rainy_server.py   (port from env PORT, default 8771)
Requests to data sources carry a neutral User-Agent and no identifying information.
"""
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
import clouds
import mosaic

try:  # pySTEPS builds on Linux (Docker image); without it the simple nowcast is used
    from pysteps import motion as ps_motion, nowcasts as ps_nowcasts
    from pysteps.utils import transformation as ps_transformation
    HAVE_PYSTEPS = True
except ImportError:
    HAVE_PYSTEPS = False

VERSION = "server-v0.14"
PORT = int(os.environ.get("PORT", "8771"))
UA = "Mozilla/5.0"          # neutral, carries no identity
HISTORY_MS = 60 * 60_000    # observed frames kept (1 h)
FC_STEPS = 24               # nowcast steps of 5 min (2 h)
STEP_MS = 5 * 60_000
POLL_S = 60
NODATA = np.float32(-1.0)   # outside radar coverage
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

def log(msg):
    """Server messages go to stderr: pySTEPS' progress output on stdout is silenced while it runs."""
    sys.stderr.write(msg + "\n")
    sys.stderr.flush()


# ---------- palette (same stops as the app, log interpolation) ----------
STOPS = [(0.1, (154, 208, 255, 0.55)), (0.5, (75, 163, 255, 0.75)), (2, (47, 107, 245, 0.85)),
         (5, (107, 61, 240, 0.9)), (10, (192, 38, 211, 0.95)), (30, (255, 61, 110, 1)), (75, (255, 212, 0, 1))]
LUT_N = 256
LUT_MIN, LUT_MAX = math.log(0.1), math.log(100.0)


def _build_lut():
    lut = np.zeros((LUT_N + 1, 4), np.uint8)  # index 0 = transparent
    for i in range(1, LUT_N + 1):
        mmh = math.exp(LUT_MIN + (LUT_MAX - LUT_MIN) * (i - 1) / (LUT_N - 1))
        c = STOPS[-1][1]
        for (v0, c0), (v1, c1) in zip(STOPS, STOPS[1:]):
            if mmh <= v1:
                t = (math.log(mmh) - math.log(v0)) / (math.log(v1) - math.log(v0))
                c = tuple(a + (b - a) * t for a, b in zip(c0, c1))
                break
        lut[i] = (int(c[0]), int(c[1]), int(c[2]), int(c[3] * 255))
    return lut


LUT = _build_lut()


def colorize(rate):
    idx = np.zeros(rate.shape, np.int32)
    wet = rate >= 0.1
    lr = np.log(np.clip(rate[wet], 0.1, 100.0))
    idx[wet] = 1 + np.round((lr - LUT_MIN) / (LUT_MAX - LUT_MIN) * (LUT_N - 1)).astype(np.int32)
    return LUT[idx]


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
                "timeformat": "unixtime"}).encode()
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


def sprog_nowcast(seq, mot, steps, down):
    """pySTEPS S-PROG: like `nowcast`, but each spatial scale decays at its own observed rate (small showers fade
    fast, large rain areas persist). seq: frames 10/5/0 min old; mot: frames 30/15/0 min old (for the motion).
    Runs on a grid `down` times coarser and is scaled back up. Backtests: equal to `nowcast` for widespread rain,
    clearly better for showers in the second hour (user-view CSI 60-120 min, USA: 0.42 vs 0.35)."""
    last = seq[-1].astype(np.float32)
    ny, nx = last.shape
    dbz = lambda gs: ps_transformation.dB_transform(np.stack(gs), threshold=0.1, zerovalue=-15.0)[0]
    m = [_coarse(g, down) for g in mot]
    velocity = ps_motion.get_method("LK")(dbz([np.where(g < 0, np.nan, g) for g in m])) / 3.0  # per 15 min -> per 5 min
    s = [_coarse(g, down) for g in seq]
    with SPROG_LOCK, contextlib.redirect_stdout(io.StringIO()):  # one region at a time; pySTEPS prints long tables
        fc = ps_nowcasts.get_method("sprog")(dbz([np.where(g < 0, 0, g) for g in s]), velocity, steps,
                                             precip_thr=-10.0,  # 0.1 mm/h in dB
                                             n_cascade_levels=6, num_workers=min(4, os.cpu_count() or 1), probmatching_method="cdf")
    out = []
    for i in range(len(fc)):  # frame by frame: converting all at once would copy the whole stack several times
        g = np.where(np.isfinite(fc[i]), 10.0 ** (fc[i].astype(np.float32) / 10.0), 0.0).astype(np.float32)  # dB -> mm/h
        g[g < 0.1] = 0  # below 0.1 mm/h counts as dry, as in pySTEPS' inverse dB transform
        g = cv2.resize(g, (nx, ny), interpolation=cv2.INTER_LINEAR) if down > 1 else g
        g[g < 0.05] = 0
        g[last < 0] = NODATA
        out.append(g.astype(np.float16))  # compact right away: 24 full-size float32 frames would need ~1.6 GB
    return out


# ---------- regions ----------
class Region:
    name = ""
    credit = ""
    SMOOTH = 1.0      # nowcast smoothing in grid pixels (tuned at ~2-3 km)
    FLOW_DOWN = 2     # optical flow is computed at grid resolution / FLOW_DOWN
    MODEL_DEG = None  # lat/lon spacing of the weather model rain used for blending (None: no blending)
    SPROG_DOWN = 0    # S-PROG runs at grid resolution / SPROG_DOWN (0: simple nowcast only)

    def __init__(self):
        self.lock = threading.Lock()
        self.obs, self.fc, self.ref = {}, {}, None
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
        avail = {t: s for t, s in self.list_times().items() if t > now - HISTORY_MS}
        new = [t for t in sorted(avail) if t not in self.obs]
        for t in new:
            g = self.load(avail[t]).astype(np.float16)  # half the memory; plenty precise for rain rates
            with self.lock:
                self.obs[t] = g
        if not new:
            return 0
        with self.lock:
            for t in list(self.obs):
                if t not in avail:
                    del self.obs[t]
            times = sorted(self.obs)
            last_t = times[-1]
            prev_t = next((t for t in reversed(times) if last_t - t >= 29 * 60_000), times[0])
            prev, last = self.obs[prev_t], self.obs[last_t]
        if prev_t == last_t:
            return len(new)
        fc = None
        if HAVE_PYSTEPS and self.SPROG_DOWN:
            with self.lock:
                near = lambda dt, tol: next((self.obs[t] for t in sorted(self.obs, key=lambda t: abs(last_t - dt - t))
                                             if abs(last_t - dt - t) <= tol), None)
                seq = [near(k * 60_000, 120_000) for k in (10, 5)] + [last]
                mot = [near(k * 60_000, 240_000) for k in (30, 15)] + [last]
            if all(g is not None for g in seq + mot):
                try:
                    fc = sprog_nowcast(seq, mot, FC_STEPS, self.SPROG_DOWN)
                except Exception as e:  # never lose the forecast: fall back to the simple method
                    log(f"{self.name} S-PROG failed, using simple nowcast: {e}")
        if fc is None:
            fc = nowcast(prev.astype(np.float32), last.astype(np.float32), last_t - prev_t, FC_STEPS, self.SMOOTH, self.FLOW_DOWN)
        if self.model:  # backtest (verify_blend.py, variant N): same skill as radar only, plus showers yet to form
            fc = blend_new_cells(self.model, fc, last_t)
        with self.lock:
            self.fc = {last_t + (i + 1) * STEP_MS: g.astype(np.float16) for i, g in enumerate(fc)}
            self.ref = last_t
        return len(new)


class EuropeOpera(Region):
    name, credit = "eu", "EUMETNET OPERA"
    BUCKET = "https://s3.waw3-1.cloudferro.com/openradar-24h"
    LAEA = "+proj=laea +lat_0=55.0 +lon_0=10.0 +x_0=1950000.0 +y_0=-2100000.0 +units=m +ellps=WGS84"
    CELL, NY, NX = 1000.0, 4400, 3800   # full 1 km resolution
    KM_PER_PX = 1.0
    SMOOTH, FLOW_DOWN = 2.0, 4          # same physical scales as the backtested 2 km setup
    MODEL_DEG = 0.15
    SPROG_DOWN = 3                      # S-PROG at 3 km: about half the memory of 2 km (peak ~5.3 GB was too close to the laptop Docker limit)

    def __init__(self):
        super().__init__()
        self.tr = Transformer.from_crs("EPSG:3857", self.LAEA, always_xy=True)
        self.to_ll = Transformer.from_crs(self.LAEA, "EPSG:4326", always_xy=True)

    def pixel_lonlat(self, col, row):
        return self.to_ll.transform((np.asarray(col) + 0.5) * self.CELL, -(np.asarray(row) + 0.5) * self.CELL)

    def list_times(self):
        now = datetime.now(timezone.utc)
        days = [now] + ([now - timedelta(days=1)] if now.hour == 0 else [])
        out = {}
        for day in days:
            xml = fetch(f"{self.BUCKET}/?prefix={day:%Y/%m/%d}/OPERA/COMP/&max-keys=1000").decode()
            for key in re.findall(r"<Key>([^<]+@DBZH\.h5)</Key>", xml):
                stamp = key.split("OPERA@")[1].split("@")[0]
                out[int(datetime.strptime(stamp, "%Y%m%dT%H%M").replace(tzinfo=timezone.utc).timestamp() * 1000)] = key
        return out

    def load(self, key):
        with h5py.File(io.BytesIO(fetch(f"{self.BUCKET}/{key}")), "r") as f:
            d = f["dataset1/data1/data"][:]
            w = f["dataset1/data1/what"].attrs
            gain, offset, nodata, undetect = (float(w[k]) for k in ("gain", "offset", "nodata", "undetect"))
        nod = d == nodata
        dbz = d * gain + offset
        dbz[(d == undetect) | nod] = -32.0
        rate = cv2.medianBlur(dbz_to_rate(dbz), 3)   # removes single-pixel clutter
        rate[nod] = NODATA
        return rate

    def to_pixels(self, x, y):
        gx, gy = self.tr.transform(x, y)
        return np.asarray(gx) / self.CELL - 0.5, -np.asarray(gy) / self.CELL - 0.5  # fractional pixel centres


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
REGIONS = {r.name: r for r in (EuropeOpera(), UsaNowcoast(), SatelliteRain())}


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
    """Bilinear interpolation between grid pixels (smooth edges instead of blocks); returns rate and coverage."""
    col, row = (v.astype(np.float32) for v in region.to_pixels(X, Y))
    g = grid.astype(np.float32)
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
    rgba = colorize(rate)
    sat = REGIONS["sat"]
    if region is not sat and not cov.all():
        sgrid = nearest_grid(sat, ts)
        if sgrid is not None:  # satellite estimate where the radar sees nothing, drawn fainter
            srate, _ = sample(sat, sgrid, X, Y)
            srgba = colorize(srate)
            srgba[..., 3] = (srgba[..., 3] * 0.65).astype(np.uint8)
            rgba[~cov] = srgba[~cov]
    elif region is sat:
        rgba[..., 3] = (rgba[..., 3] * 0.65).astype(np.uint8)
    buf = io.BytesIO()
    Image.fromarray(rgba, "RGBA").save(buf, "PNG", compress_level=3)
    return buf.getvalue()


def coverage_mask(bbox, w, h):
    """Opaque where any radar (Europe from OPERA's real coverage, CONUS box for the USA) exists."""
    X, Y = view_coords(bbox, w, h)
    cov = np.zeros((h, w), bool)
    eu = REGIONS["eu"]
    frames, _ = eu.frames()
    if frames:
        cov |= sample(eu, eu.grid(frames[0]["ts"]), X, Y)[1]
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


class Handler(BaseHTTPRequestHandler):
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
        q = {k: v[0] for k, v in parse_qs(u.query).items()}
        try:
            if u.path.startswith("/om/"):  # local stand-in for the production /om route
                with urllib.request.urlopen(f"{OPENMETEO_URL}{self.path[3:]}", timeout=60) as r:
                    return self.send(r.status, r.read(), r.headers.get("Content-Type", "application/json"), 600)
            if u.path == "/api/health":
                info = {name: {"frames": len(r.frames()[0]), "ref": r.ref} for name, r in REGIONS.items()}
                info["clouds"] = {"frames": len(CLOUDS.frames()[0]), "ref": CLOUDS.ref}
                info["de"] = {"frames": len(DE.frames()[0]), "ref": DE.ref}
                body = {"version": VERSION, "regions": info, "openmeteo": openmeteo_available()}
                return self.send(200, json.dumps(body).encode(), "application/json", 10)
            if u.path.startswith("/tiles/"):
                m = re.fullmatch(r"/tiles/(rain|cloud)/([\w-]+)/(\d+)/(\d+)/(\d+)/(\d+)\.png", u.path)
                if not m:
                    return self.send(404, b"not found", "text/plain", 60)
                kind, key, ts, z, x, y = m.group(1), m.group(2), *map(int, m.groups()[2:])
                if not (2 <= z <= 13 and 0 <= x < (1 << z) and 0 <= y < (1 << z)):
                    return self.send(404, b"bad tile", "text/plain", 3600)
                png = MOSAIC.rain_tile(key, ts, z, x, y) if kind == "rain" else MOSAIC.cloud_tile(key, ts, z, x, y)
                if png is None:
                    return self.send(404, b"unknown frame", "text/plain", 30)
                return self.send(200, png, "image/png", 86400)  # the key in the URL makes the content immutable
            if u.path == "/api/rain/frames":
                body = {"key": MOSAIC.key(), "frames": MOSAIC.frames()}
                return self.send(200, json.dumps(body).encode(), "application/json", 30)
            if u.path == "/api/rain/point":
                body = {"key": MOSAIC.key(), "series": MOSAIC.rain_point(float(q["lat"]), float(q["lon"]))}
                return self.send(200, json.dumps(body).encode(), "application/json", 60)
            if u.path == "/api/cloud/frames":
                frames, ref = CLOUDS.frames()
                return self.send(200, json.dumps({"frames": frames, "ref": ref}).encode(), "application/json", 30)
            if u.path == "/api/cloud/image":
                bbox = [float(v) for v in q["bbox"].split(",")]
                w, h = min(int(q["width"]), 2048), min(int(q["height"]), 2048)
                png = CLOUDS.image(int(q["ts"]), bbox, w, h)
                return self.send(200, png, "image/png", 3600) if png else self.send(404, b"unknown frame", "text/plain", 30)
            if u.path == "/api/cloud/point":
                body = {"series": CLOUDS.point(float(q["lat"]), float(q["lon"])), "ref": CLOUDS.ref}
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
                return self.send(200, coverage_mask(bbox, w, h), "image/png", 600)
            if u.path == "/api/radar/point":
                name, series, ref = point_series(float(q["lat"]), float(q["lon"]))
                return self.send(200, json.dumps({"region": name, "series": series, "ref": ref}).encode(), "application/json", 60)
            if u.path == "/api/push/key":
                return self.send(200, json.dumps({"key": ALERTS.public_key}).encode(), "application/json", 3600)
            if u.path == "/api/push/preview":  # what the alert logic sees at a place (for testing, sends nothing)
                s = ALERTS.series(float(q["lat"]), float(q["lon"]), ll_to_3857)
                cfg = {"lead": int(q.get("lead", 20)), "early": True, "quiet": False, "tz": q.get("tz", "UTC")}
                kind, info = alerts.AlertEngine.evaluate(s, int(time.time() * 1000), cfg, {}) if s else (None, {})
                text = alerts.AlertEngine.message(kind, info, "Test", cfg["tz"]) if kind else None
                body = {"would_send": kind, "message": text, "info": info,
                        "series": [{"ts": t, "fc": f, "wet_share": round(p, 2), "mmh": round(r, 2)} for t, f, p, r in (s or [])]}
                return self.send(200, json.dumps(body, ensure_ascii=False).encode(), "application/json", 0)
            return self.send(404, b"not found", "text/plain")
        except Exception as e:
            return self.send(400, str(e).encode(), "text/plain", 0)

    def do_OPTIONS(self):  # CORS preflight for the JSON POSTs below
        self.send_response(204)
        self.send_header("Access-Control-Allow-Origin", "*")
        self.send_header("Access-Control-Allow-Methods", "GET, POST, OPTIONS")
        self.send_header("Access-Control-Allow-Headers", "Content-Type")
        self.send_header("Access-Control-Max-Age", "86400")
        self.end_headers()

    def do_POST(self):
        u = urlparse(self.path)
        try:
            n = int(self.headers.get("Content-Length", "0"))
            if n > 20_000:
                return self.send(413, b"too large", "text/plain", 0)
            body = json.loads(self.rfile.read(n) or b"{}")
            if u.path == "/api/push/subscribe":
                sid = ALERTS.subscribe(body)
                return self.send(200, json.dumps({"id": sid}).encode(), "application/json", 0)
            if u.path == "/api/push/unsubscribe":
                ALERTS.unsubscribe(str(body.get("id")))
                return self.send(200, b"{}", "application/json", 0)
            if u.path == "/api/push/test":
                ok = ALERTS.send(str(body.get("id")), {"title": "🌧 Regenwarnung ist aktiv",
                                                       "body": "So sieht eine Warnung aus, wenn Regen auf dich zukommt."})
                return self.send(200 if ok else 404, json.dumps({"ok": ok}).encode(), "application/json", 0)
            return self.send(404, b"not found", "text/plain", 0)
        except Exception as e:
            return self.send(400, str(e).encode(), "text/plain", 0)


ALERTS = alerts.AlertEngine(DATA_DIR, REGIONS, log, UA, VAPID_SUBJECT)
CLOUDS = clouds.CloudNowcast(fetch, log, OPENMETEO_URL, openmeteo_available, ll_to_3857)
DE = mosaic.GermanyDwd(fetch, log)
MOSAIC = mosaic.Mosaic(REGIONS, DE, CLOUDS, sample, colorize, log)

if __name__ == "__main__":
    threading.Thread(target=ALERTS.loop, args=(ll_to_3857,), daemon=True).start()
    threading.Thread(target=CLOUDS.loop, daemon=True).start()
    threading.Thread(target=mosaic.de_loop, args=(DE, log), daemon=True).start()
    for r in REGIONS.values():
        threading.Thread(target=update_loop, args=(r,), daemon=True).start()
        if r.MODEL_DEG:
            r.model = ModelRain(r)
            threading.Thread(target=model_loop, args=(r.model,), daemon=True).start()
    log(f"nowcast: {'pySTEPS S-PROG' if HAVE_PYSTEPS else 'simple (pySTEPS not installed)'}")
    log(f"rainy {VERSION} on http://0.0.0.0:{PORT}")
    ThreadingHTTPServer(("0.0.0.0", PORT), Handler).serve_forever()
