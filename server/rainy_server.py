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
  GET  /api/me                                 -> {"email", "name", "picture", "providers": {"google", "email"}} (401, email null: logged out)
  GET  /api/user/picture                       -> the Google profile picture, kept by this server (404 without)
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
import glob
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
import northamerica
import mosaic
import national
import palette  # stepped rain colours and the rain tile look (same table as the app)
import ptype
import recorder
import sunny_de
import sunny_na
import thunder
import tileimg

try:  # pySTEPS builds on Linux (Docker image); without it the simple nowcast is used
    from pysteps import motion as ps_motion, nowcasts as ps_nowcasts
    from pysteps.utils import transformation as ps_transformation
    HAVE_PYSTEPS = True
except ImportError:
    HAVE_PYSTEPS = False

VERSION = "server-v2.04"
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


# WEATHER MODEL PROBE (server-v1.86, 22.11.2026): the user's phone showed "the weather model does not answer" a few
# times in the last days, and nothing on the server could say how often: Open-Meteo logs no failed requests, Caddy no
# access log, and openmeteo_available() is pinned to True in production. Now the Germany instance asks the weather
# model like the app does (a point forecast at a random place, so neither a cache nor a warm file answers it) every
# OM_PROBE_S: inside the network (Open-Meteo itself) and through the public URL (CDN + Caddy + Open-Meteo). The health
# rows carry the count, the failures and the slowest answer of each since the last row (daily_report.py flags them).
OM_PROBE_S = 60
OM_PROBE_Q = ("&minutely_15=temperature_2m,precipitation,cloud_cover,is_day&past_minutely_15=8&forecast_minutely_15=64"
              "&hourly=temperature_2m,precipitation,cloud_cover,is_day,weather_code&forecast_hours=74"
              "&daily=sunrise,sunset&forecast_days=4&timezone=auto&timeformat=unixtime&models=icon_seamless")
_om_probe = {}
_om_probe_lock = threading.Lock()


def om_probe_loop():
    import random
    public = os.environ.get("PUBLIC_URL", "").rstrip("/")
    while True:
        for key, base in (("om", OPENMETEO_URL), ("om_pub", f"{public}/om" if public else None)):
            if not base:
                continue
            # each probe its own random place (Europe, where the app's users are): with one place for both, the
            # public probe always found the files the direct one had just read in the page cache (22.11.: direct up
            # to 13-19 s once an hour, public never above 0.5 s)
            lat, lon = random.uniform(44.0, 55.0), random.uniform(2.0, 16.0)
            t0, err = time.time(), None
            try:
                req = urllib.request.Request(f"{base}/v1/forecast?latitude={lat:.2f}&longitude={lon:.2f}{OM_PROBE_Q}",
                                             headers={"User-Agent": "Mozilla/5.0"})
                with urllib.request.urlopen(req, timeout=20) as r:
                    json.loads(r.read())
            except Exception as e:
                err = str(e)[:80]
            ms = int((time.time() - t0) * 1000)
            with _om_probe_lock:
                s = _om_probe.setdefault(key, {"n": 0, "fail": 0, "max_ms": 0})
                s["n"] += 1
                s["max_ms"] = max(s["max_ms"], ms)
                if err:
                    s["fail"] += 1
                    s["err"] = err
            if err:
                log(f"weather model probe failed ({key}, {ms} ms): {err}")
        time.sleep(OM_PROBE_S)


def om_probe_take():
    """The probe counts since the last call (one health row), then reset."""
    with _om_probe_lock:
        out = {k: dict(v) for k, v in _om_probe.items()}
        _om_probe.clear()
    return out


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


def advect_blend(fc, last, motion, fade_min=ADV_FADE_MIN, speed=1.0, steps=None):
    """fc: S-PROG steps (5 min apart, float16 grids, NODATA outside); last: the newest analysis (same grid); motion:
    sprog_nowcast(motion_out=...). Returns the blended steps (float16). speed scales the motion of the moved last
    image; steps: the step number (1 = +5 min) of each grid in fc when only some steps are passed (shadows)."""
    vel, down = motion.get("velocity"), motion.get("down", 1)
    if vel is None:
        return fc
    last = np.asarray(last, np.float32)
    ny, nx = last.shape
    u = cv2.resize(np.nan_to_num(vel[0]).astype(np.float32), (nx, ny), interpolation=cv2.INTER_LINEAR) * (down * speed)
    v = cv2.resize(np.nan_to_num(vel[1]).astype(np.float32), (nx, ny), interpolation=cv2.INTER_LINEAR) * (down * speed)
    gx, gy = np.meshgrid(np.arange(nx, dtype=np.float32), np.arange(ny, dtype=np.float32))
    src = np.where(last < 0, 0, last).astype(np.float32)
    out = []
    for i, g in enumerate(fc):
        k = steps[i] if steps is not None else i + 1
        lead = k * STEP_MS / 60_000
        w = max(0.0, 1.0 - lead / fade_min)
        if w <= 0:
            out.append(g)
            continue
        adv = cv2.remap(src, gx - u * k, gy - v * k, cv2.INTER_LINEAR, borderMode=cv2.BORDER_CONSTANT, borderValue=0)
        gf = np.asarray(g, np.float32)
        b = w * adv + (1.0 - w) * np.where(gf < 0, 0, gf)
        b[b < 0.05] = 0
        b[gf < 0] = NODATA
        out.append(b.astype(np.float16))
    return out


# The S-PROG forecast moves rain BETWEEN REGIONS (16.11.2026): its probability matching imposes the whole domain's
# intensity distribution of the last image on the forecast field, so large stratiform areas keep high ranks and grow
# (Germany x1.23 / x1.42 of the later image at +60 / +120, Czechia x1.5 / x1.85; at the gauges x1.4-1.5) while decaying
# showers shrink (England x0.35, Alps x0.6) - the whole domain stays at ~0.95. local_amount() puts each region's
# amount back: the reference is the last image moved along the motion (Lagrangian persistence, which keeps every
# region's rain where it travels), compared to the forecast in a wide Gaussian window (sigma LOCAL_PM_KM); the
# forecast is scaled by that ratio (to the power `power`, clipped), the domain total stays S-PROG's.
LOCAL_PM_KM = 150.0
# Served since server-v1.47 (16.11.2026) with keep=False: tune_eu.py, 4 runs of 16.11., amount forecast / later image
# at +60 / +120: Germany 0.74 / 0.67 -> 1.02 / 0.99, Alps 0.67 / 0.73 -> 0.95 / 1.05, England 0.70 / 0.63 -> 0.96 /
# 0.90, Czechia 0.90 / 0.99 -> 1.05 / 1.19; cell-by-cell CSI unchanged (+-0.01). The uncorrected S-PROG stays the
# scored shadow "sprog0" (= the old served forecast from +60 on). LOCAL_AMOUNT=0 switches it off.
LOCAL_AMOUNT = os.environ.get("LOCAL_AMOUNT", "1") == "1"
# Rapid-update weather models (DWD ICON-D2-RUC, Meteo-France AROME-PI; NOAA HRRR for North America) blended into the
# forecast for +45..+120 min - scored shadow "nwp"; served only with NWP_SERVE=1 (below; nwp_blend.py, tune_blend.py).
# NWP_BLEND=0 switches it off; never in the own-picture instance (EU_NATIONAL, CPU for the public instances).
NWP_BLEND = os.environ.get("NWP_BLEND", "1") != "0"
# NWP_SERVE=1 (set for api-eu only, 18.11.2026): the blend is SERVED - live 17./18.11., ~540 Europe runs, CSI >= 1 mm/h
# +90/+120 0.64/0.56 -> 0.66/0.61, amount x0.91/0.89 -> 0.98/1.02, +15..+60 unchanged; North America neutral (shadow
# only). The 4 km shadow is carried to the full grid (nwp_blend.serve_full) before self.fc is set, so tiles, data
# tiles, point series, alerts and gauges all read the blended forecast; the un-blended one is the shadow "own0".
NWP_SERVE = os.environ.get("NWP_SERVE", "0") == "1"


def local_amount(fc, last, motion, km_per_px, sigma_km=LOCAL_PM_KM, power=1.0, clip=3.0, keep=True, steps=None):
    """fc: S-PROG steps (float16, NODATA outside); last: the newest analysis; motion: sprog_nowcast(motion_out=...).
    Returns the steps with every region's rain amount set to the advected last image's (keep: scaled to S-PROG's
    total; else the advected image's own total = Lagrangian persistence). steps: the step number of each grid when
    only some steps are passed (shadows)."""
    vel, down = motion.get("velocity"), motion.get("down", 1)
    if vel is None or not len(fc):
        return fc
    last = np.asarray(last, np.float32)
    ny, nx = last.shape
    c = max(1, int(round(16.0 / km_per_px)))           # factor grid ~16 km
    src = _coarse(np.where(last < 0, 0, last), c)
    src[src < 0] = 0
    cy, cx = src.shape
    # where the moved image comes from outside the radar coverage (rain drifting in from the Atlantic, the domain
    # edge), it knows nothing: there the factor fades to 1 (S-PROG unchanged)
    cov0 = cv2.resize((last[: cy * c, : cx * c] >= 0).astype(np.float32), (cx, cy), interpolation=cv2.INTER_AREA)
    u = cv2.resize(np.nan_to_num(vel[0]).astype(np.float32), (cx, cy), interpolation=cv2.INTER_LINEAR) * down / c
    v = cv2.resize(np.nan_to_num(vel[1]).astype(np.float32), (cx, cy), interpolation=cv2.INTER_LINEAR) * down / c
    gx, gy = np.meshgrid(np.arange(cx, dtype=np.float32), np.arange(cy, dtype=np.float32))
    sig = max(1.0, sigma_km / (c * km_per_px))
    blur = lambda a: cv2.GaussianBlur(a, (0, 0), sig, borderType=cv2.BORDER_CONSTANT)
    norm = np.maximum(blur(np.ones((cy, cx), np.float32)), 1e-6)
    out = []
    for i, g in enumerate(fc):
        gf = np.asarray(g, np.float32)
        f = _coarse(np.where(gf < 0, 0, gf)[: cy * c, : cx * c], c)
        f[f < 0] = 0
        k = steps[i] if steps is not None else i + 1
        mx, my = gx - u * k, gy - v * k
        adv = cv2.remap(src, mx, my, cv2.INTER_LINEAR, borderMode=cv2.BORDER_CONSTANT, borderValue=0)
        cov = cv2.remap(cov0, mx, my, cv2.INTER_LINEAR, borderMode=cv2.BORDER_CONSTANT, borderValue=0)
        tf, ta = float(f.sum()), float(adv.sum())
        if tf <= 0 or ta <= 0:
            out.append(g)
            continue
        if keep:
            adv *= tf / ta                               # keep S-PROG's total: only the distribution moves
        eps = 0.02 * tf / f.size + 1e-6
        r = np.clip((blur(adv) + eps) / (blur(f) + eps), 1.0 / clip, clip) ** power
        wc = np.clip(blur(cov) / norm, 0.0, 1.0) ** 2   # the factor only counts where the source was seen
        r = wc * r + (1.0 - wc)
        r = cv2.resize(r.astype(np.float32), (cx * c, cy * c), interpolation=cv2.INTER_LINEAR)
        r = np.pad(r, ((0, ny - cy * c), (0, nx - cx * c)), mode="edge")
        b = np.where(gf < 0, 0, gf) * r
        b[b < 0.05] = 0
        b[gf < 0] = NODATA
        out.append(b.astype(np.float16))
    return out


# Stationary orographic rain (19.11.2026): rain anchored at the windward slopes (air flows through it, the rain area
# stays) is moved by S-PROG / local_amount like any other rain - across the Alpine ridge into the lee. At the gauges
# 17.-19.11. the Europe forecast / later picture at +120 was x2.0 in Austria south / west (the own picture with the
# Italian layers upstream), x2.6 in Ticino (OPERA), x1.75 in the Italian Alps, and in Switzerland the forecast fell
# below persistence at 1 / 3 mm. oro_forecast() holds the quasi-stationary part in place: S = the minimum of the images
# of the last `w` minutes (lightly smoothed), over mountains only (terrain ORO_H0 -> ORO_H1 m: weight 0 -> 1); the
# forecast loses the share of S the motion carries along (q = moved S / moved last image, ~16 km windows) and gets S
# back where it is, decaying with exp(-lead / tau). lee: a rain-shadow damping exp(min(0, climb) / lee) with climb = the
# metres per 5 min the motion climbs over the smoothed terrain. Scored shadow "oro" (tune_oro.py, dem_eu4km.npz).
# tune_oro.py on the OPERA picture, 10 hourly runs 19.11. 01:50-10:50Z (light rain; amount forecast / later image at
# +120, CSI cell by cell 0.1/1/3 mm): CH south own0 x1.98 -> oro w30 tau240 x1.64 (1 mm +15..+120 +0.02..0.04),
# Italian Alps x1.62 -> x1.51, CH north x1.29 -> x1.11, Germany x1.06 -> x1.02 (CSI equal), Austria west 1 mm +15/+30/
# +60 .38/.26/.16 -> .46/.34/.24 but amount +120 x0.93 -> x0.73, domain CSI equal or +0.01. Weaker variants: tau 120
# (more loss in Austria west), w 60 (smaller S); the rain-shadow damping (lee 150 / 300) cut real rain on the lee
# slopes already at +15 (Austria south 1 mm .20 -> .14) and the unmasked S (everywhere) dried the whole domain
# (x0.87 -> x0.72 at +120) - both rejected. Not yet the target (Austria / CH x1.1): judge the shadow live first.
ORO_H0, ORO_H1 = 400.0, 1200.0
ORO = os.environ.get("ORO", "1") != "0"   # the shadow "oro" in the Europe instances (ORO=0: off)
ORO_W, ORO_TAU, ORO_LEE = 30, 240.0, None   # the variant chosen in tune_oro.py (numbers above)


def oro_part(last, past, terrain, w=60, mask=True):
    """The quasi-stationary rain S (full grid, mm/h): the minimum of the last image and the images up to `w` minutes
    old (past: {minutes ago: image}, every 15 min), each lightly smoothed (6 km box: a few km of jitter is still the
    same place); mask: over mountains only (terrain ORO_H0 -> ORO_H1 m). None when an image is missing."""
    frames = [last] + [past.get(m) for m in sorted(past) if m <= w]
    if not w or any(g is None for g in frames):
        return None
    S = None
    for g in frames:
        g = np.asarray(g, np.float32)
        g = cv2.blur(np.where(g < 0, 0, g), (3, 3))
        S = g if S is None else np.minimum(S, g)
    if mask:
        S *= np.clip((terrain - ORO_H0) / (ORO_H1 - ORO_H0), 0.0, 1.0)
    return S


def oro_forecast(f, lead, last, part, motion, terrain, km_per_px, tau=120.0, lee=None):
    """f: one forecast step (full grid, NODATA outside); lead: minutes; last: the newest image; part: oro_part();
    motion: sprog_nowcast(motion_out=...); terrain: metres on the same grid (smoothed). Returns float32 (NODATA kept)."""
    f = np.asarray(f, np.float32)
    vel, down = motion.get("velocity"), motion.get("down", 1)
    if vel is None:
        return f
    ny, nx = f.shape
    steps = lead / 5.0
    out = np.where(f < 0, 0, f)
    if part is not None and float(part.max()) > 0:
        c = max(1, int(round(16.0 / km_per_px)))
        src_l = _coarse(np.where(np.asarray(last, np.float32) < 0, 0, np.asarray(last, np.float32)), c)
        src_s = _coarse(part, c)
        cy, cx = src_l.shape
        u = cv2.resize(np.nan_to_num(vel[0]).astype(np.float32), (cx, cy), interpolation=cv2.INTER_LINEAR) * down / c
        v = cv2.resize(np.nan_to_num(vel[1]).astype(np.float32), (cx, cy), interpolation=cv2.INTER_LINEAR) * down / c
        gx, gy = np.meshgrid(np.arange(cx, dtype=np.float32), np.arange(cy, dtype=np.float32))
        mx, my = gx - u * steps, gy - v * steps
        al = cv2.remap(np.maximum(src_l, 0), mx, my, cv2.INTER_LINEAR, borderMode=cv2.BORDER_CONSTANT, borderValue=0)
        asl = cv2.remap(np.maximum(src_s, 0), mx, my, cv2.INTER_LINEAR, borderMode=cv2.BORDER_CONSTANT, borderValue=0)
        al, asl = cv2.GaussianBlur(al, (0, 0), 1.0), cv2.GaussianBlur(asl, (0, 0), 1.0)
        q = np.clip(asl / (al + 1e-3), 0.0, 1.0)  # the share of the arriving rain that was anchored where it came from
        q = cv2.resize(q, (cx * c, cy * c), interpolation=cv2.INTER_LINEAR)
        q = np.pad(q, ((0, ny - cy * c), (0, nx - cx * c)), mode="edge")
        out = out * (1.0 - q) + part * math.exp(-lead / tau)
    if lee:
        u = cv2.resize(np.nan_to_num(vel[0]).astype(np.float32), (nx, ny), interpolation=cv2.INTER_LINEAR) * down
        v = cv2.resize(np.nan_to_num(vel[1]).astype(np.float32), (nx, ny), interpolation=cv2.INTER_LINEAR) * down
        dh_row, dh_col = np.gradient(terrain)
        climb = u * dh_col + v * dh_row   # m per 5 min along the motion
        out *= np.exp(np.minimum(0.0, climb) / float(lee))
    out[out < 0.05] = 0
    out[f < 0] = NODATA
    return out


# The radar coverage's INFLOW EDGE (19.11.2026): where a forecast pixel's source - the pixel moved back along the motion
# - lies outside the radar coverage (the Atlantic beyond OPERA's range, the domain edges, gaps between national
# networks), the extrapolation knows nothing and brings in zeros: offline (tune_inflow.py on the 66 archived served
# eu runs 18.11. 20Z-19.11. 12Z) the served forecast kept x0.41 / x0.17 / x0.11 of the rain at +30 / +60 / +120 in
# those cells (CSI 0.1 mm/h .274 / .132 / .074; persistence .222 / .174 / .154). Shadow "inflow": there the forecast
# is replaced, with the weight wout = 1 - (source coverage, smoothed over INFLOW_TAPER_KM), by the larger of
# (a) the last image at the pixel (Eulerian persistence) and (b) the last image extended beyond the coverage edge
# (nearest covered value of a 20 km smoothed copy, fading with the distance outside over INFLOW_DECAY_KM) moved along
# the motion ("the rain keeps coming in as it does at the edge"). Offline in those cells: CSI .308 / .198 / .159,
# amount x0.78 / x0.70 / x0.74; whole domain unchanged (+30 .647 = .647, +60 .541 vs .542, +120 .448 vs .450 at 0.1
# mm/h; 1 mm/h equal; amount x1.00 -> x1.01). NOT the cause of the Galicia / Portugal / Brittany / Ireland drain
# measured at the gauges (their sources lie inside the coverage, wout <= 0.03 at +120 there): there even the last
# image moved along the motion loses the rain (Galicia x0.25 at +120, Brittany x0.60, Portugal x0.49) while the
# image left in place keeps x1.0 - the rain grows at the coast (orography, coastal convergence, or OPERA reading the
# sea far from the radars too weak); the shadow "oro" covers the hilly part. On a rebuilt forecast (S-PROG on 4 km +
# local_amount + advect_blend, 18 runs 18./19.11.) the shadow still helps there: Galicia CSI 0.1 mm/h +120 .128 ->
# .170 (amount x0.12 -> x0.22), Portugal .244 -> .260; whole domain .413 -> .412.
# Computed on the verification grid (1 / RAW_DOWN) at the scored leads only (~5 grids). INFLOW=0: off.
INFLOW = os.environ.get("INFLOW", "1") != "0"
INFLOW_TAPER_KM, INFLOW_DECAY_KM, INFLOW_SMOOTH_KM = 30.0, 150.0, 20.0
INFLOW_LEADS = (15, 30, 60, 90, 120)   # fcverify.LEADS and check_national_gauges.FC_LEADS


def inflow_forecast(fc, last, motion, d, km_per_px, ref, leads=INFLOW_LEADS):
    """fc: the served steps (full grid, < 0 = no radar); last: the newest analysis; motion: sprog_nowcast(motion_out=...)
    (coarse pixels per 5 min); d: RAW_DOWN. Returns {valid ms: float16 grid at 1 / d} for the leads, or None."""
    from scipy import ndimage
    vel, down = motion.get("velocity"), motion.get("down", 1)
    if vel is None or not len(fc):
        return None
    lastf = np.asarray(last, np.float32)
    H, W = lastf.shape[0] // d, lastf.shape[1] // d
    shrink = lambda a: cv2.resize(np.asarray(a, np.float32), (W, H), interpolation=cv2.INTER_AREA)
    cov = shrink(lastf >= 0) > 0.5
    if cov.all() or not cov.any():
        return None
    lz = np.where(cov, shrink(np.where(lastf < 0, 0, lastf)), 0).astype(np.float32)
    km = km_per_px * d
    s = INFLOW_SMOOTH_KM / km / 2
    num = cv2.GaussianBlur(lz, (0, 0), s)
    den = cv2.GaussianBlur(cov.astype(np.float32), (0, 0), s)
    edge = np.where(cov & (den > 0.05), num / np.maximum(den, 1e-6), 0).astype(np.float32)
    dist, (iy, ix) = ndimage.distance_transform_edt(~cov, return_indices=True)
    ext = np.where(cov, lz, edge[iy, ix] * np.exp(-dist * km / INFLOW_DECAY_KM)).astype(np.float32)
    del num, den, edge, dist, iy, ix
    f = down / d
    u = cv2.resize(np.nan_to_num(vel[0]).astype(np.float32), (W, H), interpolation=cv2.INTER_LINEAR) * f
    v = cv2.resize(np.nan_to_num(vel[1]).astype(np.float32), (W, H), interpolation=cv2.INTER_LINEAR) * f
    gx, gy = np.meshgrid(np.arange(W, dtype=np.float32), np.arange(H, dtype=np.float32))
    covf = cov.astype(np.float32)
    taper = INFLOW_TAPER_KM / km / 2
    out = {}
    for i, g in enumerate(fc):
        lead = (i + 1) * STEP_MS // 60_000
        if lead not in leads:
            continue
        mx, my = gx - u * (i + 1), gy - v * (i + 1)
        src = cv2.remap(covf, mx, my, cv2.INTER_LINEAR, borderMode=cv2.BORDER_CONSTANT, borderValue=0)
        wout = 1.0 - np.clip(cv2.GaussianBlur(src, (0, 0), taper), 0.0, 1.0)
        wout[~cov] = 0.0
        fill = np.maximum(lz, cv2.remap(ext, mx, my, cv2.INTER_LINEAR, borderMode=cv2.BORDER_CONSTANT, borderValue=0))
        gf = np.asarray(g, np.float32)
        gs = shrink(np.where(gf < 0, 0, gf))
        o = (1.0 - wout) * gs + wout * fill
        o[o < 0.05] = 0
        o[shrink(gf >= 0) <= 0.5] = NODATA
        out[ref + (i + 1) * STEP_MS] = o.astype(np.float16)
    return out


# The forecast's WET AREA grows (16.11.2026, North America): from +30 min on the served forecast covered x1.14-1.23
# of the later image's wet area while the last image unchanged had x1.0-1.08 - S-PROG runs on a coarse grid (6 km
# there) and the upsampling smears every light-rain edge outwards. A global S-PROG threshold (0.03 -> 0.10) only
# reached x1.11 at a CSI cost. local_area() sets a LOCAL rain / no rain threshold instead: in a Gaussian window
# (sigma LOCAL_AREA_KM) the forecast may be as wet as the last image moved along the motion (Lagrangian persistence)
# times LOCAL_AREA_SLACK; the threshold is the forecast rate that leaves that wet fraction (never below the map's
# 0.05 mm/h: the area is only ever cut, never grown); the removed light rain is given back to the rest of the window
# so every region keeps its amount. Applied to the final field (after advect_blend: the blend of the moved image and
# S-PROG is itself wetter than either at +30). tune_eu.py, 4 runs each, 16.11.2026, on top of the served local150n,
# wet area / CSI at 0.05 0.3 1 mm/h, +30 / +60 / +120:
#   North America  served x1.14 / 1.13 / 1.14  ->  sigma 300, slack 1.05: x1.04 / 1.05 / 1.04, CSI identical
#                  (sigma 300, slack 1: x0.99 / 1.01 / 1.00; sigma 150 slack 1 cuts too much: x0.96 / 0.94 at +60/+120)
#   Europe         served x0.98 / 0.97 / 0.96  ->  x0.98 / 0.97 / 0.95, CSI identical, regional amounts unchanged
# Cost 0.4-0.5 s per run on the server (24 steps, both grids). LOCAL_AREA=1 switches it on (default off).
LOCAL_AREA_KM = 300.0
LOCAL_AREA_SLACK = 1.05
LOCAL_AREA = os.environ.get("LOCAL_AREA", "0") == "1"
# Gauge adjustment of the Europe pictures (gauge_adjust.py, 18.11.2026): a mean-field bias per gauge network from the
# last 12 h at the rain gauges, multiplied into every new image (Austria x1.55 -> x1.03, Wallonia x1.45 -> x1.21,
# Finland x0.53 -> x1.01 at held-out gauges; CSI equal or better overall). GAUGE_ADJUST=1 switches it on (default off;
# not for North America).
GAUGE_ADJUST = os.environ.get("GAUGE_ADJUST", "0") == "1"
# Static clutter mask in the eu / eun / na pictures (clutter.py, 19.11.2026): spots that are wet in most images while
# the country around them is dry, and areas whose mean rate over days no rain reaches (the fixed echo near Dublin
# in OPERA, radar-site echoes in Scandinavia, Poland, Romania, Portugal ...) are replaced by their surroundings before
# the nowcast. Set up in main_eu (the serving process only - offline tools that build a region do not count images).
CLUTTER_MASK = os.environ.get("CLUTTER_MASK", "1") == "1"


def local_area(fc, last, motion, km_per_px, sigma_km=LOCAL_AREA_KM, slack=LOCAL_AREA_SLACK, tmax=1.0, clip=2.0):
    """fc: forecast steps (float16, NODATA outside); last: the newest analysis; motion: sprog_nowcast(motion_out=...).
    Returns the steps with the wet fraction (>= 0.05 mm/h) of every window cut to at most the advected last image's,
    rain amount per window unchanged. `slack`: the window may be that much wetter than the moved image (windows
    where the forecast is drier are never filled up). Local threshold at most `tmax` mm/h, amount factor at most
    `clip`. Rain is only removed or scaled up: dry and NODATA pixels stay exactly as they were."""
    vel, down = motion.get("velocity"), motion.get("down", 1)
    if vel is None or not len(fc):
        return fc
    last = np.asarray(last, np.float32)
    ny, nx = last.shape
    # window statistics on a grid of ~sigma / 5 (at least 16 km); the histograms come from every s-th row / column
    # (~16 pixels per cell), the thresholds are read from a grid of s x s pixel blocks
    c = max(2, 2 * int(round(max(16.0, sigma_km / 5.0) / km_per_px / 2.0)))
    s = max([1] + [d for d in range(1, c // 4 + 1) if c % d == 0])
    cy, cx = ny // c, nx // c
    if cy < 2 or cx < 2:
        return fc
    sy, sx = cy * c // s, cx * c // s
    ncell = cy * cx
    # rate levels 0.05 * 2^(k/4) up to tmax (the last one is tmax itself); lookup table float16 bits -> level
    nlv = max(2, int(math.ceil(4.0 * math.log2(max(tmax, 0.1) / 0.05))) + 1)
    lt = np.log(0.05) + np.arange(nlv) * (0.25 * math.log(2.0))
    lt[-1] = math.log(tmax)
    nb = nlv + 1
    # bin of every sampled pixel = its cell * nb + its level (level from the float16 bits through a lookup table)
    cellnb = ((np.arange(sy) // (c // s))[:, None] * (cx * nb) + (np.arange(sx) // (c // s))[None, :] * nb).ravel()
    with np.errstate(invalid="ignore"):  # the table includes the NaN bit patterns
        f16 = np.nan_to_num(np.arange(65536, dtype=np.uint16).view(np.float16).astype(np.float32), nan=-1.0, posinf=1e4)
    lut = np.clip(np.floor(np.log2(np.maximum(f16, 0.05) * 20.0) * 4.0) + 1.0, 0, nlv - 1)
    lut = np.where(f16 >= tmax, nlv, np.where(f16 >= 0.05, lut, 0)).astype(np.intp)
    g16 = lambda g: np.ascontiguousarray(g, dtype=np.float16)
    samp = lambda a: a[: cy * c : s, : cx * c : s]
    area = lambda a: cv2.resize(a, (cx, cy), interpolation=cv2.INTER_AREA)
    valid = area((samp(g16(fc[0])) >= 0).astype(np.float32))   # NODATA is the same in every step (the last image's)
    lc = last[: cy * c, : cx * c]
    wet0, cov0 = area((lc >= 0.05).astype(np.float32)), area((lc >= 0).astype(np.float32))
    u = cv2.resize(np.nan_to_num(vel[0]).astype(np.float32), (cx, cy), interpolation=cv2.INTER_LINEAR) * down / c
    v = cv2.resize(np.nan_to_num(vel[1]).astype(np.float32), (cx, cy), interpolation=cv2.INTER_LINEAR) * down / c
    gx, gy = np.meshgrid(np.arange(cx, dtype=np.float32), np.arange(cy, dtype=np.float32))
    sig = max(1.0, sigma_km / (c * km_per_px))
    blur = lambda a: cv2.GaussianBlur(a, (0, 0), sig, borderType=cv2.BORDER_CONSTANT)
    norm = np.maximum(blur(np.ones((cy, cx), np.float32)), 1e-6)
    bv = np.maximum(blur(valid), 1e-6)
    # coarse field -> grid of s x s pixel blocks (edge cells continued over the last partial cell), read at the
    # blocks of the pixels `blk`
    ey, ex_ = -(-ny // c) - cy, -(-nx // c) - cx
    fw = (cx + ex_) * (c // s)

    def at(a, blk):
        a = cv2.copyMakeBorder(a, 0, ey, 0, ex_, cv2.BORDER_REPLICATE) if ey or ex_ else a
        return cv2.resize(a, (fw, (cy + ey) * (c // s)), interpolation=cv2.INTER_LINEAR).ravel()[blk]
    out = []
    for i, g in enumerate(fc):
        g = g16(g)
        gs = samp(g)
        h = np.bincount(cellnb + lut[gs.view(np.uint16)].ravel(), minlength=ncell * nb).reshape(cy, cx, nb)
        ex = np.cumsum(h[:, :, :0:-1], axis=2)[:, :, ::-1].astype(np.float32)   # ex[..., k]: sampled pixels >= level k
        mx, my = gx - u * (i + 1), gy - v * (i + 1)
        adv = cv2.remap(wet0, mx, my, cv2.INTER_LINEAR, borderMode=cv2.BORDER_CONSTANT, borderValue=0)
        cov = cv2.remap(cov0, mx, my, cv2.INTER_LINEAR, borderMode=cv2.BORDER_CONSTANT, borderValue=0)
        bc = blur(cov)
        target = blur(adv) / np.maximum(bc, 1e-6) * slack   # wet fraction of the moved last image in the window
        frac = blur(np.ascontiguousarray(ex)) / (bv * (c // s) ** 2)[:, :, None]   # all levels at once (channels)
        # first level whose wet fraction is not above the target; the threshold interpolated in log rate
        k = (frac > target[:, :, None]).sum(axis=2)
        k1 = np.clip(k, 1, nlv - 1)
        fa = np.take_along_axis(frac, (k1 - 1)[:, :, None], 2)[:, :, 0]
        fb = np.take_along_axis(frac, k1[:, :, None], 2)[:, :, 0]
        t = np.clip((fa - target) / np.maximum(fa - fb, 1e-6), 0.0, 1.0)
        logt = np.where(k == 0, lt[0], np.where(k >= nlv, lt[-1], lt[k1 - 1] + t * (lt[k1] - lt[k1 - 1])))
        wc = np.clip(bc / norm, 0.0, 1.0) ** 2           # only where the source was seen (as local_amount)
        logt = (wc * logt + (1.0 - wc) * lt[0]).astype(np.float32)
        if not (logt > lt[0] + 1e-4).any():
            out.append(g)
            continue
        thr = np.exp(logt)
        # give the removed light rain back to the window: amount before / after from the sampled pixels
        g0 = np.maximum(gs.astype(np.float32), 0)
        a0 = area(g0)
        a1 = area(np.where(g0 >= cv2.resize(thr, (sx, sy), interpolation=cv2.INTER_LINEAR), g0, 0).astype(np.float32))
        eps = 0.005 * float(a0.mean()) + 1e-6
        r = np.clip((blur(a0) + eps) / (blur(a1) + eps), 1.0, clip).astype(np.float32)
        # only the wet pixels change (dry and NODATA stay as they are): below the local threshold -> 0, else x r
        b = g.copy().ravel()
        idx = np.flatnonzero((g.view(np.uint16).ravel() - np.uint16(1)) < np.uint16(0x7C00))   # 0 < rate < inf
        rows, cols = np.divmod(idx, nx)
        blk = rows // s * fw + cols // s
        val = b[idx].astype(np.float32)
        b[idx] = np.where(val >= at(thr, blk), val * at(r, blk), 0).astype(np.float16)
        out.append(b.reshape(ny, nx))
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


OWN_MODEL = os.environ.get("OWN_MODEL") == "1"  # Germany: serve the "model" shadow as own (see OwnNowcast.step)


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
        self.de, self.done, self.skipped = de, None, None
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
            # until server-v1.79 this was silent: the run kept DWD's forecast under the "own" choice and nobody knew
            # (audit 20.11.2026); logged once per run - a missing image may still arrive, then the run is made after all
            if self.skipped != ref:
                self.skipped = ref
                gone = [dt for dt in (30, 15, 10, 5) if near(dt) is None]
                log(f"[{datetime.now():%H:%M:%S}] own nowcast: run {datetime.fromtimestamp(ref / 1000, timezone.utc):%H:%M}Z "
                    f"waits - no DWD image {gone} min before it (DWD's forecast is served for 'own' meanwhile)")
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
                # OWN_MODEL=1 (server-v1.90): the "model" shadow is served as our own forecast - it led at the DWD
                # gauges from +60 on 3 rain days (20.-23.11.2026, never worse) and cell by cell from +90 over 7 days;
                # plain S-PROG stays measured as the shadow "pure" (also recorded while OWN_MODEL is off, then = own)
                de.set_own(ref, model if OWN_MODEL else grids)
                # scored by fcverify only; dwdnew = DWD's own forecast plus the model's new showers in the 2nd hour
                de.fc_shadow = {"model": model, "pure": grids, "dwdnew": dwd_new, "cells": cells, "sprog0": sprog0}
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
    ADV_SHADOWS = {}  # name -> (fade_min, speed): scored shadows of the moved last image's hand-over to S-PROG
    LA_SHADOWS = {}   # name -> sigma_km: scored shadows of local_amount's window (without the NWP blend: vs own0)
    RAW_DOWN = 3      # the forecast without the model's new cells (fc_raw) is kept at grid resolution / RAW_DOWN
    ORO_DEM = None    # terrain file for the shadow "oro" (oro_forecast; EuropeOpera: dem_eu4km.npz)
    INFLOW_SHADOW = False   # the shadow "inflow" (inflow_forecast; EuropeOpera: True)
    _terrain = None

    def terrain(self, shape):
        """The terrain height (m) on the region's grid, smoothed ~6 km (from ORO_DEM, made by build_dem.py)."""
        if self._terrain is None or self._terrain.shape != tuple(shape):
            h = np.load(os.path.join(os.path.dirname(os.path.abspath(__file__)), self.ORO_DEM))["h"].astype(np.float32)
            h = cv2.resize(h, (shape[1], shape[0]), interpolation=cv2.INTER_LINEAR)
            self._terrain = cv2.GaussianBlur(h, (0, 0), 3.0)
        return self._terrain
    MODELS = None     # Open-Meteo "models" for the blend (None: best match per place)
    # A 5-minute step the source skipped is filled with the previous run's forecast for it (regions whose source
    # delivers every 5 minutes). 14.11.2026 02:40 UTC: OPERA skipped one image - for an hour one step of the
    # timeline was empty all over Europe (found by check_tile_gaps: "wet 0 between 620 and 617") and three runs fell
    # back to the simple nowcast (S-PROG wants the images 5, 10, 15 and 30 min back; the simple one showed +40 %
    # rain area, a jump at "now"). The own picture has such holes more often: a step waits up to 20 min for a
    # national layer while the next one is already there. The real image replaces the fill when it arrives.
    FILL_GAPS = False
    # BRIDGE (server-v1.92): one radar of a composite drops out for a few images while the others keep coming
    # (23./24.11.2026 radar watch: Iceland, Norway, Ukraine for 5-10 min each, several times a night; Romania for
    # hours). Its area turned "no data" at once and a rain field vanished from the map (user 23.11.: Moldova). Pixels
    # with no data now that were covered one step before get the previous run's forecast for this step, for at most
    # BRIDGE_STEPS steps in a row; a longer outage is hatched as before. Bridged pixels stay out of the verification's
    # truth (fcverify: self.bridged).
    BRIDGE_GAPS = False
    BRIDGE_STEPS = 3
    # OUTAGE MAP (server-v1.92): when every area was last covered (coarse, OUT_DOWN x the grid), for the app's
    # marking of areas whose radar is out right now (/api/radar/mask?kind=outage) - unlike areas no radar ever reaches
    OUT_DOWN = 4
    OUT_KEEP_MIN = 6 * 60
    SPROG_THR = 0.1   # S-PROG's rain / no rain threshold (mm/h)

    def __init__(self):
        self.lock = threading.Lock()
        self.obs, self.fc, self.ref = {}, {}, None
        self.filled = set()  # steps in obs that are a forecast standing in for a missing image
        self.bridged = {}    # step -> uint8 grid: >0 where a dropped-out radar's area was bridged (BRIDGE_GAPS)
        self.cov_seen = None  # int32 minutes, coarse: when each area was last covered (outage map)
        self.model = None

    def latest(self):
        with self.lock:
            return self.obs[max(self.obs)] if self.obs else None

    def bridge(self, t):
        """BRIDGE_GAPS (under the lock): fill the area of a radar that just dropped out from the previous run's
        forecast for step t, for at most BRIDGE_STEPS steps in a row."""
        g, prev, f = self.obs[t], self.obs.get(t - STEP_MS), self.fc.get(t)
        if prev is None or f is None or prev.shape != g.shape or f.shape != g.shape:
            return
        lost = (g < 0) & (prev >= 0)
        if not lost.any():
            return
        before = self.bridged.get(t - STEP_MS)
        count = np.where(lost, (before.astype(np.int32) if before is not None else 0) + 1, 0)
        sel = lost & (count <= self.BRIDGE_STEPS) & (f >= 0)
        n = int(sel.sum())
        if not n:
            return
        g = g.copy()
        g[sel] = f[sel]
        self.obs[t] = g
        self.bridged[t] = np.where(sel, count, 0).astype(np.uint8)
        if n > 2000 and int(count[sel].max()) == 1:  # a radar's area, not a few edge pixels: once per outage
            log(f"{self.name}: {n} pixels without data at {datetime.fromtimestamp(t / 1000, timezone.utc):%H:%MZ} "
                f"bridged with the forecast (a radar or the whole source dropped out)")

    def note_coverage(self, t):
        """Outage map (under the lock): the coarse cells covered in step t get its time."""
        g = self.obs[t]
        d = self.OUT_DOWN
        h, w = g.shape[0] // d, g.shape[1] // d
        cov = np.asarray(g[:h * d, :w * d] >= 0).reshape(h, d, w, d).mean(axis=(1, 3)) >= 0.5
        if self.cov_seen is None or self.cov_seen.shape != cov.shape:
            self.cov_seen = np.full(cov.shape, -10 ** 9, np.int32)
        self.cov_seen[cov] = np.maximum(self.cov_seen[cov], t // 60_000)

    def outage(self, X, Y):
        """True where the newest image has no data but the area was covered within OUT_KEEP_MIN: a radar is out."""
        with self.lock:
            seen = self.cov_seen
            newest = max((t for t in self.obs if t not in self.filled), default=None)
        if seen is None or newest is None:
            return np.zeros(np.shape(X), bool)
        recent = seen >= newest // 60_000 - self.OUT_KEEP_MIN
        col, row = (np.asarray(v, np.float64) for v in self.to_pixels(X, Y))
        d, (h, w) = self.OUT_DOWN, seen.shape
        r, c = np.floor(row / d).astype(np.int64), np.floor(col / d).astype(np.int64)
        inside = (r >= 0) & (r < h) & (c >= 0) & (c < w)
        out = np.zeros(np.shape(X), bool)
        out[inside] = recent[r[inside], c[inside]]
        return out & ~sample(self, self.grid(newest), X, Y)[1]

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
        listed = self.list_times()
        self.src_newest = max(listed, default=None)  # the newest step the source offers (source watch)
        avail = {t: s for t, s in listed.items() if t > cutoff}
        new = [t for t in sorted(avail) if t not in self.obs or t in self.filled or self.replace_obs(t, avail[t])]
        for t in new:
            g = self.load(avail[t]).astype(np.float16)  # half the memory; plenty precise for rain rates
            with self.lock:
                self.obs[t] = g
                self.filled.discard(t)
                self.bridged.pop(t, None)
                if self.BRIDGE_GAPS:
                    self.bridge(t)
                self.note_coverage(t)
        if not new:
            return 0
        with self.lock:
            for t in list(self.obs):
                if t not in avail and not (t in self.filled and t > cutoff):
                    del self.obs[t]
                    self.filled.discard(t)
            for t in [t for t in self.bridged if t not in self.obs]:
                del self.bridged[t]
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
        fc = adv = None
        motion, pure, pure_small = {}, None, None
        if HAVE_PYSTEPS and self.SPROG_DOWN:
            with self.lock:
                near = lambda dt, tol: next((self.obs[t] for t in sorted(self.obs, key=lambda t: abs(last_t - dt - t))
                                             if abs(last_t - dt - t) <= tol), None)
                seq = [near(k * 60_000, 120_000) for k in (10, 5)] + [last]
                mot = [near(k * 60_000, 240_000) for k in (30, 15)] + [last]
            if all(g is not None for g in seq + mot):
                try:
                    pure = sprog_nowcast(seq, mot, FC_STEPS, self.SPROG_DOWN, thr=self.SPROG_THR, motion_out=motion)
                    d = self.RAW_DOWN  # the uncorrected S-PROG, kept small for the shadow "sprog0"
                    pure_small = {last_t + (i + 1) * STEP_MS: cv2.resize(np.asarray(g, np.float32), (g.shape[1] // d, g.shape[0] // d),
                                  interpolation=cv2.INTER_AREA).astype(np.float16) for i, g in enumerate(pure)}
                    ks = [lead // 5 for lead in INFLOW_LEADS]
                    raw_k = [pure[k - 1] for k in ks] if self.LA_SHADOWS and LOCAL_AMOUNT else None
                    if LOCAL_AMOUNT:  # every region keeps its rain amount (local_amount, 16.11.2026)
                        pure = local_amount(pure, last, motion, self.KM_PER_PX, keep=False)
                    fc = advect_blend(pure, last, motion)  # no break at "now" (ADV_FADE_MIN)
                    if self.ADV_SHADOWS or raw_k is not None:  # other variants, at the scored leads only, at 1 / RAW_DOWN
                        try:
                            d = self.RAW_DOWN
                            small = lambda gs: {last_t + k * STEP_MS: cv2.resize(np.asarray(g, np.float32), (g.shape[1] // d, g.shape[0] // d),
                                                                                 interpolation=cv2.INTER_AREA).astype(np.float16)
                                                for k, g in zip(ks, gs)}
                            adv = {n: small(advect_blend([pure[k - 1] for k in ks], last, motion, f, sp, steps=ks))
                                   for n, (f, sp) in self.ADV_SHADOWS.items()}
                            for n, sig in self.LA_SHADOWS.items() if raw_k is not None else ():
                                la = local_amount(raw_k, last, motion, self.KM_PER_PX, sigma_km=sig, keep=False, steps=ks)
                                adv[n] = small(advect_blend(la, last, motion, steps=ks))
                        except Exception as e:
                            log(f"{self.name} extra shadows failed: {e}")
                            adv = None
                    raw_k = None
                    if LOCAL_AREA:  # off by default: wet area per window <= the moved last image's (local_area)
                        fc = local_area(fc, last, motion, self.KM_PER_PX)
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
        oro_small = oro = None
        if self.ORO_DEM and ORO and motion.get("velocity") is not None:
            # shadow "oro" (19.11.2026): the stationary orographic part held in place (oro_forecast), at 1 / RAW_DOWN
            try:
                t0 = time.time()
                with self.lock:
                    past = {m: near(m * 60_000, 120_000) for m in (15, 30, 45, 60)}
                lastf = np.asarray(last, np.float32)
                terr = self.terrain(lastf.shape)
                part = oro_part(lastf, past, terr, w=ORO_W)
                d = self.RAW_DOWN
                oro_small = []
                for i, g in enumerate(fc):
                    o = oro_forecast(g, (i + 1) * STEP_MS / 60_000, lastf, part, motion, terr, self.KM_PER_PX,
                                     tau=ORO_TAU, lee=ORO_LEE)
                    oro_small.append(cv2.resize(o, (o.shape[1] // d, o.shape[0] // d),
                                                interpolation=cv2.INTER_AREA).astype(np.float16))
                del part, past, lastf
                self.oro_s = round(time.time() - t0, 2)
            except Exception as e:
                log(f"{self.name} oro shadow failed: {e}")
                oro_small = None
        nwp = own0 = None
        pure = None  # S-PROG's own steps: not needed any more (frees the steps serve() replaces)
        if getattr(self, "nwp", None) is not None:  # shadow "nwp": the served forecast with the rapid-update models
            try:
                d = self.RAW_DOWN
                shrink = lambda g: cv2.resize(np.asarray(g, np.float32), (g.shape[1] // d, g.shape[0] // d),
                                              interpolation=cv2.INTER_AREA)
                small = [shrink(g) for g in fc]
                nwp = self.nwp.shadow(small, shrink(last), last_t)
                if NWP_SERVE:  # the blend served: fc becomes the shadow carried to the full grid (nwp_blend.serve_full)
                    own0 = {last_t + (i + 1) * STEP_MS: g.astype(np.float16) for i, g in enumerate(small)}
                    if nwp:
                        fc = self.nwp.serve(fc, small, nwp, last_t)
                del small
                if oro_small is not None and nwp and NWP_SERVE:  # "oro" with the same blend, comparable to "own"
                    oro = self.nwp.blend_again(oro_small, last_t)
            except Exception as e:  # never lose the forecast: fc is only replaced once serve() has finished
                log(f"{self.name} nwp shadow{' / serve' if NWP_SERVE else ''} failed: {e}")
            self.nwp.prep = None
        inflow = None
        if self.INFLOW_SHADOW and INFLOW and motion.get("velocity") is not None:
            # shadow "inflow" (19.11.2026): the served forecast where its source lies outside the coverage, at 1 / RAW_DOWN
            try:
                inflow = inflow_forecast(fc, last, motion, self.RAW_DOWN, self.KM_PER_PX, last_t)
            except Exception as e:
                log(f"{self.name} inflow shadow failed: {e}")
        if oro is None and oro_small is not None and not (NWP_SERVE and nwp):  # no served blend: compare with own as is
            oro = {last_t + (i + 1) * STEP_MS: g.astype(np.float16) for i, g in enumerate(oro_small)}
        oro_small = None
        with self.lock:
            self.fc_own0 = own0  # NWP_SERVE: verification shadow "own0" = the forecast without the blend, at 1 / RAW_DOWN
            self.fc_oro = oro  # verification shadow "oro": the stationary orographic part held in place, at 1 / RAW_DOWN
            self.fc_inflow = inflow  # verification shadow "inflow": the coverage's inflow edge filled, at 1 / RAW_DOWN
            self.fc_adv = adv  # verification shadows ADV_SHADOWS / LA_SHADOWS, at 1 / RAW_DOWN
            self.fc_nwp = nwp  # verification shadow "nwp" (fcverify.RegionVerify), at 1 / RAW_DOWN
            self.fc_raw = raw
            self.fc_cells = cells  # verification shadow "cells": with the model's new showers (served until 11.11.)
            self.fc_pure = pure_small  # S-PROG without advection blend and amount control: verification shadow "sprog0"
            self.fc = {last_t + (i + 1) * STEP_MS: g.astype(np.float16) for i, g in enumerate(fc)}
            self.ref = last_t
        self.after_forecast()
        return len(new)

    def after_forecast(self):
        """Called after every new forecast run (subclasses: verification notes)."""

    def replace_obs(self, t, src):
        """True when the step t, already in obs, is to be loaded again from src (EuropeNational: OPERA came late)."""
        return False


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
    ORO_DEM = "dem_eu4km.npz"   # terrain for the shadow "oro" (build_dem.py; the OPERA grid at 4 km)
    INFLOW_SHADOW = True        # the shadow "inflow": the coverage's inflow edge (inflow_forecast)
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
    BRIDGE_GAPS = os.environ.get("BRIDGE_GAPS", "1") != "0"
    # WARM START: the images and the forecast of the last run are kept on disk, and a restarted instance continues
    # with them at once instead of serving nothing, then measured images without a forecast, for 2-4 minutes
    # (user, 13.11.2026: "mal war was da und manchmal nicht ... nur bis 'jetzt' was da" - three restarts while he
    # was looking). Same run -> same tile key, so the app does not notice. Older than 20 min: a cold start.
    STATE_DIR = os.path.join(os.environ.get("VERIFY_LOG_DIR", "."), "state") if EDITION == "eu" else None

    def save_state(self):
        d = self.STATE_DIR
        if not d or not self.state_lock.acquire(blocking=False):  # the previous save is still writing
            return
        try:
            os.makedirs(d, exist_ok=True)
            with self.lock:
                obs, fc, ref, filled = dict(self.obs), dict(self.fc), self.ref, set(self.filled)
            have = {int(f[4:-4]) for f in os.listdir(d) if f.startswith("obs_") and f.endswith(".npz")}
            for t in (obs.keys() - have) | (self.saved_filled - filled) & obs.keys():  # new, or real image for a fill
                tmp = os.path.join(d, f"tmp_{t}.npz")
                np.savez_compressed(tmp, g=obs[t])
                os.replace(tmp, os.path.join(d, f"obs_{t}.npz"))
            for t in have - obs.keys():
                os.remove(os.path.join(d, f"obs_{t}.npz"))
            tmp = os.path.join(d, "tmp_fc.npz")
            np.savez_compressed(tmp, **{str(t): g for t, g in fc.items()})
            os.replace(tmp, os.path.join(d, "fc.npz"))
            with open(os.path.join(d, "tmp_state.json"), "w") as f:
                json.dump({"ref": ref, "fc": sorted(fc), "filled": sorted(filled), "version": VERSION}, f)
            os.replace(os.path.join(d, "tmp_state.json"), os.path.join(d, "state.json"))
            self.saved_filled = filled
        except Exception as e:
            log(f"{self.name}: state not saved: {e}")
        finally:
            self.state_lock.release()

    def load_state(self):
        """True when the last run before the restart was taken over."""
        d = self.STATE_DIR
        try:
            with open(os.path.join(d, "state.json")) as f:
                meta = json.load(f)
            ref, now = meta["ref"], time.time() * 1000
            if now - ref > 20 * 60_000:
                return False
            z = np.load(os.path.join(d, "fc.npz"))
            fc = {int(k): z[k] for k in z.files}
            if sorted(fc) != meta["fc"]:  # stopped between the two files
                return False
            obs = {}
            for name in os.listdir(d):
                if name.startswith("obs_") and name.endswith(".npz"):
                    t = int(name[4:-4])
                    if now - HISTORY_MS - 15 * 60_000 < t <= ref:
                        obs[t] = np.load(os.path.join(d, name))["g"]
            if ref not in obs or len(obs) < 7:
                return False
            with self.lock:
                self.obs, self.fc, self.ref = obs, fc, ref
                self.filled = set(meta["filled"]) & set(obs)
                self.fc_raw = self.fc_cells = self.fc_pure = self.fc_nwp = self.fc_oro = self.fc_inflow = self.fc_adv = None
            self.saved_filled = set(self.filled)
            log(f"{self.name}: continuing with the run of {datetime.fromtimestamp(ref / 1000, timezone.utc):%H:%MZ} "
                f"from before the restart ({len(obs)} images)")
            return True
        except FileNotFoundError:
            return False
        except Exception as e:
            log(f"{self.name}: state not read: {e}")
            return False
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
        threading.Thread(target=self.save_state, daemon=True).start()
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
                        for name, d in (("pure", getattr(self, "fc_pure", None)), ("cells", getattr(self, "fc_cells", None)),
                                        ("own0", getattr(self, "fc_own0", None)),  # own0: without the blend (NWP_SERVE)
                                        ("oro", getattr(self, "fc_oro", None)),  # oro: stationary orographic part
                                        ("inflow", getattr(self, "fc_inflow", None)),  # inflow: coverage edge filled
                                        *(getattr(self, "fc_adv", None) or {}).items()) if d}  # hand-over shadows
            if ref % 600_000 or any(g is None for g in grids):  # the gauges report every 10 minutes
                return
            d = self.RAW_DOWN
            small = (px[0] // d, px[1] // d, px[2])
            cng.record_fc(ref, np.stack([cng.at_gauges(g, px) for g in grids]),
                          {name: np.stack([cng.at_gauges(g, small, 1) for g in gs])
                           for name, gs in more.items() if all(g is not None for g in gs)})
        except Exception as e:
            log(f"fcgauge failed: {e}")
        try:
            self.record_alert_inputs()
        except Exception as e:
            log(f"fcalert failed: {e}")

    def record_alert_inputs(self):
        """The rain alerts' input at every gauge (16.11.2026: alerts verified outside Germany - verify_alerts_eu.py):
        per run, for the picture and every 5-minute forecast step, the wet share of the RADIUS_KM circle (>= alerts.WET,
        the real alert rule's quantity) and the rate alerts.series reports (max of the 3x3 mean at the place and the
        circle's 75th percentile) -> verify_log/<instance>/fcalert/<ref>.npz. Same circle as alerts.series."""
        import alerts
        import check_national_gauges as cng
        px = self.gauges()
        with self.lock:
            ref = self.ref
            grids = [self.obs.get(ref)] + [self.fc.get(ref + (i + 1) * STEP_MS) for i in range(FC_STEPS)]
        if px is None or ref is None or ref % 600_000 or any(g is None for g in grids):
            return
        col, row, ok = px
        r = max(1, int(round(alerts.RADIUS_KM / self.KM_PER_PX)))
        o = np.arange(-r, r + 1)
        dx, dy = np.meshgrid(o, o)
        disc = dx ** 2 + dy ** 2 <= r * r
        dx, dy = dx[disc], dy[disc]
        o3 = np.array([-1, 0, 1])
        ex, ey = (a.ravel() for a in np.meshgrid(o3, o3))
        ny, nx = grids[0].shape
        col, row = np.asarray(col), np.asarray(row)
        cc = np.clip(col[:, None] + dx[None], 0, nx - 1)
        rr = np.clip(row[:, None] + dy[None], 0, ny - 1)
        c3 = np.clip(col[:, None] + ex[None], 0, nx - 1)
        r3 = np.clip(row[:, None] + ey[None], 0, ny - 1)
        share = np.full((len(grids), len(col)), 255, np.uint8)
        peak = np.full((len(grids), len(col)), np.nan, np.float16)
        use = np.asarray(ok, bool)
        for i, g in enumerate(grids):
            g = np.asarray(g, np.float32)
            v = g[rr, cc]
            good = v >= 0
            nval = good.sum(axis=1)
            valid = use & (nval >= 0.5 * v.shape[1])   # alerts.series: skipped when mostly outside the radar
            sh = ((v >= alerts.WET) & good).sum(axis=1) / np.maximum(1, nval)
            vm = np.where(good, v, np.nan)
            p75 = np.nanpercentile(np.where(valid[:, None], vm, 0), 75, axis=1)
            e3 = g[r3, c3]
            m3 = np.where(e3 >= 0, e3, 0).sum(axis=1) / np.maximum(1, (e3 >= 0).sum(axis=1))
            share[i, valid] = np.round(sh[valid] * 250).astype(np.uint8)
            peak[i, valid] = np.maximum(m3, p75)[valid]
        cng.record_alert(ref, share, peak)

    def __init__(self):
        super().__init__()
        self.px_cache, self.px_lock = OrderedDict(), threading.Lock()
        self.tr = Transformer.from_crs("EPSG:3857", self.LAEA, always_xy=True)
        self.to_ll = Transformer.from_crs(self.LAEA, "EPSG:4326", always_xy=True)
        import spokes
        self.spokes = spokes.SpokeFilter(self.KM_PER_PX, spokes.sites(self, mosaic.merc))
        self.spoke_log = {}  # radar code -> when its line was last logged
        self.spoke_last = None  # (time step, [(length km, radar code, (col, row))]) of the newest image
        self.state_lock, self.saved_filled = threading.Lock(), set()
        self.gadj = None
        if GAUGE_ADJUST and not NA:
            import gauge_adjust
            self.gadj = gauge_adjust.GaugeAdjust(self, "nat" if EU_NATIONAL else "opera", log)

    def gauge_adjusted(self, ts, rate):
        """The image with the gauge adjustment (GAUGE_ADJUST); ts: ms or the OPERA file's url."""
        if self.gadj is None:
            return rate
        try:
            if not isinstance(ts, (int, np.integer)):
                stamp = ts.split("OPERA@")[1].split("@")[0]
                ts = int(datetime.strptime(stamp, "%Y%m%dT%H%M").replace(tzinfo=timezone.utc).timestamp() * 1000)
            return self.gadj.apply(int(ts), rate)
        except Exception as e:
            log(f"gauge adjust failed: {e}")
            return rate

    clutter = None  # clutter.ClutterMask (main_eu, CLUTTER_MASK)

    def declutter(self, ts, rate):
        """The image with the static clutter spots replaced by their surroundings (clutter.py); ts: ms or the OPERA
        file's url. The mask counts the image before it is applied."""
        if self.clutter is None:
            return rate
        if not isinstance(ts, (int, np.integer)):
            stamp = ts.split("OPERA@")[1].split("@")[0]
            ts = int(datetime.strptime(stamp, "%Y%m%dT%H%M").replace(tzinfo=timezone.utc).timestamp() * 1000)
        return self.clutter.apply(int(ts), rate)

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
        self.opera_newest = newest or None
        if time.time() * 1000 - newest > 25 * 60_000 and time.time() - self.warned > 1800:  # normally 5-10 min old
            self.warned = time.time()
            log(f"eu radar source failed: newest image {datetime.fromtimestamp(newest / 1000, timezone.utc):%d.%m. %H:%MZ}"
                if newest else "eu radar source failed: no images listed in any bucket")
        return out

    warned = 0.0
    opera_newest = None  # the newest OPERA image listed (ms), for the source watch (monitor_status)
    BUCKET = BUCKETS[0]  # for check_opera_gauges.py

    # Radar watch (server-v1.91): which national radars made each image (the file's how/nodes). User 23.11.2026: "over
    # Moldova a huge cloud suddenly vanishes at about 12:20" - the Romanian radar robar (Barnova near Iasi, it covers
    # NE Romania and Moldova) left the composite at 11:20Z; OPERA marks its area "no data", the map hatches it and the
    # place forecast switches to the weather model, but nobody knew. Now a radar that leaves (or comes back) is logged
    # and the health rows carry the missing ones for the daily report ("Radare ... ausgefallen").
    NODE_KEEP_MS = 6 * 3600_000   # a radar unseen for longer is no longer expected (decommissioned, seasonal)
    node_seen = None              # {radar code: newest image time (ms) it was in}
    node_ts = 0                   # the newest image the watch has seen (images can load out of order)
    nodes_missing = ()            # radars of the last 6 h that are not in the newest image

    def watch_nodes(self, url, nodes):
        try:
            stamp = url.split("OPERA@")[1].split("@")[0]
            ts = int(datetime.strptime(stamp, "%Y%m%dT%H%M").replace(tzinfo=timezone.utc).timestamp() * 1000)
            codes = set(re.findall(r"[a-z]{5}", nodes.decode() if isinstance(nodes, bytes) else str(nodes)))
            if not codes or ts <= self.node_ts:
                return
            seen = self.node_seen if self.node_seen is not None else {}
            for c in codes:
                seen[c] = ts
            missing = sorted(c for c, t in seen.items() if c not in codes and ts - t <= self.NODE_KEEP_MS)
            if self.node_seen is not None:  # not on the first image after a start (everything would be "new")
                hhmm = f"{datetime.fromtimestamp(ts / 1000, timezone.utc):%H:%MZ}"
                for c in sorted(set(missing) - set(self.nodes_missing)):
                    log(f"radar {c} left the OPERA composite at {hhmm} (its area shows no data)")
                for c in sorted(set(self.nodes_missing) - set(missing)):
                    log(f"radar {c} is back in the OPERA composite at {hhmm}")
            self.node_seen, self.node_ts, self.nodes_missing = seen, ts, tuple(missing)
            for c in [c for c, t in seen.items() if ts - t > self.NODE_KEEP_MS]:
                del seen[c]
        except Exception as e:  # never lose an image over the watch
            log(f"radar watch failed: {e}")

    def load(self, url):
        with h5py.File(io.BytesIO(fetch(url)), "r") as f:
            d = f["dataset1/data1/data"][:]
            w = f["dataset1/data1/what"].attrs
            gain, offset, nodata, undetect = (float(w[k]) for k in ("gain", "offset", "nodata", "undetect"))
            if "how" in f and "nodes" in f["how"].attrs:
                self.watch_nodes(url, f["how"].attrs["nodes"])
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
        else:
            rate[nod] = NODATA
        rate = self.despoke(url, rate)
        # the own picture (EuropeNational.load) removes clutter and adjusts after its layers are laid on
        return rate if EU_NATIONAL else self.gauge_adjusted(url, self.declutter(url, rate))

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
    # OPERA LATE (19.11.2026, server-v1.70): OPERA's composite stopped at 21:10 UTC while the national products (DWD,
    # MeteoSwiss, Radar-DPC) kept coming - and the own picture stopped with it, the map showed the 21:10 run for over
    # an hour (also over Germany). A step OPERA has not delivered OPERA_LATE_MIN after the measurement is now composed
    # from the national layers alone; everywhere else it has no data (hatched in the app, never old rain as "now").
    # OPERA's image replaces it when it arrives. Normally OPERA is 5-10 min late, so this does not happen.
    OPERA_LATE_MIN = 15
    # LOCAL WINDOW (20.11.2026): Wallonia +120 x2.6 at the gauges (19./20.11., later picture x0.95) while NL x0.43 and
    # Flanders x0.59. A stationary heavy-rain maximum over northern France was carried east along a W-E band by the
    # moved last image; local_amount's 150 km window then filled the decay S-PROG let happen over NL / Flanders back into
    # the band's core. Offline (12 event runs + 12 runs of 16./17.11., +120 vs the later picture): sigma 50 km -> Wallonia
    # x1.79 -> 1.61, NL x0.53 -> 0.76 in the event, Wallonia 1.29 -> 1.14 on the quiet days, domain CSI unchanged.
    # Scored shadow "la50" (without the NWP blend: compare with own0).
    LA_SHADOWS = {"la50": 50.0}

    def __init__(self):
        super().__init__()
        self.started = time.time()
        self.layers = national.start(self, log, UA)
        self.used = {}  # ts -> names of the layers in that image (health)
        self.gauge_px = None  # gauges at which OPERA alone and the composed picture are noted (check_national_gauges.py)
        self.partial = set()  # steps composed without OPERA (replaced when OPERA's image comes)
        self.partial_warned = 0.0
        self.backfilled = set()  # (shadow layer, step) whose late image was added to the natobs file

    def list_times(self):
        try:
            times = super().list_times()
        except Exception as e:  # both OPERA buckets unreachable: the national layers still count
            if time.time() - self.partial_warned > 1800:
                log(f"eu radar: OPERA not listed ({e})")
            times = {}
        now = time.time() * 1000
        # the layers load their last hour first: without them the first images would be composed from OPERA alone
        if time.time() - self.started < 120 and not all(l.first_poll_done for l in self.layers):
            return {}
        alive = [l for l in self.layers if l.alive() and not l.SHADOW]  # a shadow is not waited for
        out = {t: (t, url) for t, url in times.items()
               if now - t > national.WAIT_MIN * 60_000 or all(l.has(t) for l in alive)}
        if alive:
            # only after OPERA's newest image: a single image OPERA skipped is filled from the forecast (FILL_GAPS)
            # instead, so the rest of Europe does not blink out for one step
            last = int(now - self.OPERA_LATE_MIN * 60_000) // STEP_MS * STEP_MS
            t, n, opera_last = last, 0, max(times, default=0)
            while t > max(opera_last, now - HISTORY_MS - 15 * 60_000):
                if any(l.has(t) for l in alive):
                    out[t] = (t, None)
                    n += 1
                t -= STEP_MS
            # steps composed without OPERA stay while OPERA does not deliver them later (OPERA back without its backlog)
            for t in self.partial:
                if t not in times and t not in out and t > now - HISTORY_MS - 15 * 60_000 and any(l.has(t) for l in alive):
                    out[t] = (t, None)
            if n and time.time() - self.partial_warned > 1800:
                self.partial_warned = time.time()
                log(f"eu radar without OPERA: {n} steps from the national layers alone ({', '.join(l.name for l in alive)})")
        return out

    def replace_obs(self, t, src):
        return t in self.partial and src[1] is not None

    def backfill_shadows(self, ts, cng):
        """A shadow layer whose source publishes later than the step is noted at the gauges (DMI: 8-13 min after its
        time, OPERA 5-10) had no image then: its values are added to the steps of the last 30 min once the image is
        there, so it is scored like the others (server-v1.88). The picture never waits for a shadow."""
        col, row, ok = self.gauge_px
        for k, l in enumerate(self.layers):
            if not l.SHADOW:
                continue
            for t in range(ts - 30 * 60_000, ts, STEP_MS):
                if (l.name, t) in self.backfilled or l.get(t) is None:
                    continue
                self.backfilled.add((l.name, t))
                v = national.values_at([l], t, np.where(ok, col, -1), np.where(ok, row, -1))[0]
                cng.backfill(t, k, v)
        self.backfilled = {x for x in self.backfilled if x[1] > ts - 2 * 3600_000}

    def load(self, src):
        ts, url = src
        if url is None:  # OPERA late: the national layers alone, no data elsewhere
            rate = np.full((self.NY, self.NX), NODATA, np.float32)
            self.used[ts] = national.compose(rate, self.layers, ts)
            self.partial.add(ts)
            for t in [t for t in self.partial if t < ts - 3 * HISTORY_MS]:
                self.partial.discard(t)
            return self.gauge_adjusted(ts, self.declutter(ts, rate))
        self.partial.discard(ts)
        rate = super().load(url)  # OPERA as the Europe instance shows it (spokes already out)
        opera = None
        try:
            import check_national_gauges as cng
            if self.gauges() is not None:
                opera = cng.at_gauges(rate, self.gauge_px)
        except Exception as e:
            log(f"natobs failed: {e}")
        self.used[ts] = national.compose(rate, self.layers, ts)
        rate = self.declutter(ts, rate)  # before natobs: the gauges see the picture as shown
        if opera is not None:
            try:
                col, row, ok = self.gauge_px
                who = national.cover_at(self.layers, ts, np.where(ok, col, -1), np.where(ok, row, -1))
                each = national.values_at(self.layers, ts, np.where(ok, col, -1), np.where(ok, row, -1))
                cng.record(ts, opera, cng.at_gauges(rate, self.gauge_px), who, [l.name for l in self.layers], each,
                           [l.name for l in self.layers if l.SHADOW])
                self.backfill_shadows(ts, cng)
            except Exception as e:
                log(f"natobs failed: {e}")
        for t in [t for t in self.used if t < ts - 3 * HISTORY_MS]:
            del self.used[t]
        return self.gauge_adjusted(ts, rate)  # natobs above keeps the unadjusted pictures


class NorthAmericaMrms(EuropeOpera):
    """North America (NA=1, the instance behind /na/): NOAA MRMS in place of OPERA (northamerica.py) - the same
    nowcast, tiles, timeline, verification and warm start as the Europe instances, registered as region "eu" so all
    of that runs unchanged. Grid: MRMS's 0.01 deg cells in 2 x 2 blocks (0.02 deg, ~2.2 km north-south, 1.4-2 km
    east-west), 3500 x 1750."""
    ORO_DEM = None  # no terrain file for North America: no shadow "oro"
    credit = "NOAA MRMS"
    # HAND-OVER (20.11.2026): the served forecast hands the moved last image over to S-PROG by +60 (ADV_FADE_MIN); on
    # North America's ~6 km S-PROG grid that loses the scattered fine structure, and the motion of the cells is faster
    # than that of the rain areas (78 boxes, 72 h: areas 2.4 px/h, motion field 5.6 px/h; the West's mountain rain and a
    # stationary front in the Southeast). Offline, 47 + 35 runs, tolerant CSI >= 1 mm/h +60/+90/+120: persistence
    # .766/.678/.616, own .748/.657/.585, fade 240 .800/.702/.623, fade 240 + motion x0.7 .803/.712/.639 (cell by cell
    # -0.01..0.02 vs own, still +0.04..0.05 over persistence; amount .98/.95 vs .93/.91). Scored shadows first.
    ADV_SHADOWS = {"adv240": (240, 1.0), "adv240s": (240, 0.7)}
    K = 2
    DEG = northamerica.DEG0 * K
    LAT_TOP = northamerica.LAT1 + northamerica.DEG0 / 2
    LON_LEFT = northamerica.LON0 - northamerica.DEG0 / 2
    NX, NY = northamerica.NX0 // K, northamerica.NY0 // K
    KM_PER_PX = 2.0
    GAIN = 1.0          # MRMS's rate is already in mm/h (radar with its own gauge / model corrections)
    DESPOKE = False     # no OPERA radar list for it; MRMS has its own quality control
    BOX = (-130.0, 20.0, -60.0, 55.0)
    # S-PROG on grid / 3 (~6 x 5 km) instead of / 2: at / 2 pySTEPS needed +1.4 GB per run on top of ~0.9 GB held, over
    # the container's 2.5 GB - the instance was OOM-killed on every second run from 14.11. 23:13 UTC on (135 starts),
    # and its restarts pushed the whole host into OOM kills of the Germany and Europe instances. Measured locally
    # (15.11.): / 3 needs +0.45 GB and half the time (17 s instead of 34 s); 5 issue times, cell by cell on 4x4 blocks,
    # CSI +15/+30 at 0.1 mm/h 0.759/0.666 vs 0.767/0.673 (the first hour is the advected full-resolution image).
    SPROG_DOWN = 3

    def __init__(self):
        super().__init__()
        self.listing = northamerica.MrmsListing(fetch)
        self.metar = northamerica.MetarGauges(fetch, os.environ.get("VERIFY_LOG_DIR", "."), log)
        self.gauge_at = 0.0

    def pixel_lonlat(self, col, row):
        return self.LON_LEFT + (np.asarray(col) + 0.5) * self.DEG, self.LAT_TOP - (np.asarray(row) + 0.5) * self.DEG

    def to_pixels(self, x, y):
        lon, lat = northamerica.merc_to_lonlat(x, y)
        return ((lon - self.LON_LEFT) / self.DEG - 0.5).astype(np.float32), ((self.LAT_TOP - lat) / self.DEG - 0.5).astype(np.float32)

    def list_times(self):
        out = self.listing.times()
        newest = max(out, default=0)
        if time.time() * 1000 - newest > 25 * 60_000 and time.time() - self.warned > 1800:
            self.warned = time.time()
            log("na radar source failed: no MRMS file in the last 25 min")
        return {t: (t, url) for t, url in out.items()}

    def load(self, src):
        ts, url = src
        rate = northamerica.block_mean(northamerica.decode(fetch(url, timeout=60)), self.K, NODATA)
        rate = self.declutter(ts, rate)
        try:  # the picture at the US rain gauges (METAR), every image: check_na_gauges.py
            import check_national_gauges as cng
            px = self.gauges()
            if px is not None:
                v = cng.at_gauges(rate, px)
                cng.record(ts, v, v)
        except Exception as e:
            log(f"natobs failed: {e}")
        return rate

    def gauges(self):
        """The US METAR stations with a precipitation sensor (northamerica.MetarGauges; the list grows while new
        stations report, so it is read again every hour)."""
        if self.gauge_px is None or time.time() - self.gauge_at > 3600:
            import check_national_gauges as cng
            st = self.metar.stations()
            self.gauge_px = cng.pixels(self, st, mosaic.merc) if st else None
            self.gauge_at = time.time()
        return self.gauge_px


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
NA = EDITION == "eu" and os.environ.get("NA") == "1"   # the North America instance (NOAA MRMS), behind /na/
# which of the four instances this is, for logs and /api/health (all three non-German ones are EDITION "eu" - the
# North America one logged "eu edition" and "eu: +1 frames", audit 20.11.2026; region names stay "eu": files use them)
INSTANCE = "na" if NA else "eun" if EU_NATIONAL else EDITION
REGIONS = {r.name: r for r in (EuropeNational() if EU_NATIONAL else NorthAmericaMrms() if NA else EuropeOpera(),
                               UsaNowcoast(), SatelliteRain())}


def update_loop(region):
    while True:
        started = time.time()
        try:
            n = region.update()
            if n:
                _, ref = region.frames()
                log(f"[{datetime.now():%H:%M:%S}] {INSTANCE if EDITION == 'eu' else region.name}: +{n} frames, latest "
                      f"{datetime.fromtimestamp(ref / 1000, timezone.utc):%H:%M}Z, {time.time() - started:.1f}s")
        except Exception as e:  # keep serving the last good data
            import traceback  # where it failed (a bare message like "NoneType / int" cannot be traced afterwards)
            where = " <- ".join(f"{os.path.basename(f.filename)}:{f.lineno} {f.name}" for f in traceback.extract_tb(e.__traceback__)[-3:][::-1])
            log(f"{region.name} update failed: {e} [{where}]")
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
    if kind == "outage" and EDITION != "eu":  # only the radar composites know outages (server-v1.92): none here
        kind, X = "none", X[:0]
    if kind == "none":
        buf = io.BytesIO()
        Image.fromarray(np.zeros((h, w, 4), np.uint8), "RGBA").save(buf, "PNG", compress_level=3)
        return buf.getvalue()
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
    if kind == "outage":  # areas whose radar is out right now (server-v1.92): opaque there
        cov = eu.outage(X, Y)
        img = np.zeros((h, w, 4), np.uint8)
        img[cov] = (0, 0, 0, 255)
        buf = io.BytesIO()
        Image.fromarray(img, "RGBA").save(buf, "PNG", compress_level=3)
        return buf.getvalue()
    frames, _ = eu.frames()
    obs = [f["ts"] for f in frames if not f["fc"]]
    if obs:  # the newest image's coverage (v1.70: before the first one's - an OPERA outage looked dry, not hatched)
        cov |= sample(eu, eu.grid(max(obs)), X, Y)[1]
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
                info["de"] = {"frames": len(DE.frames()[0]), "ref": DE.ref, "vetoed_px": getattr(DE, "vetoed", 0),
                              # our own forecast's run (server-v1.79): own_ref != ref = "own" serves DWD's forecast now
                              "own_ref": DE.own_ref, "own_current": DE.ref is not None and DE.own_ref == DE.ref}
                info["sources"] = source_ages(time.time() * 1000)  # minutes since each source's newest data (v1.70)
                if EU_NATIONAL:  # per layer: its newest image, and in how many of the map's images it is
                    eu_r = REGIONS["eu"]
                    info["layers"] = {l.name: {"newest": l.newest_ts(), "alive": l.alive(), "shadow": l.SHADOW,
                                               "in_frames": sum(l.name in v for t, v in eu_r.used.items() if t in eu_r.obs)}
                                      for l in eu_r.layers}
                if REGIONS["eu"].spoke_last:  # interference lines removed from the newest image: (km, radar)
                    info["spokes"] = {"ref": REGIONS["eu"].spoke_last[0], "lines": [l[:2] for l in REGIONS["eu"].spoke_last[1]]}
                info["ptype"] = {"frames": len(PTYPE.grids), "ref": max(PTYPE.grids) if PTYPE.grids else None}
                body = {"version": VERSION, "edition": EDITION, "instance": INSTANCE, "regions": info, "openmeteo": openmeteo_available(), "rss_mb": rss_mb(),
                        "auth": AUTH.providers(),  # sign-in methods the app may offer
                        "tiles": MOSAIC.render_summary()}  # rain tile render time per zoom (server CPU)
                return self.send(200, json.dumps(body).encode(), "application/json", 10)
            if u.path.startswith("/tiles/raindata"):  # data tiles for the app's WebGL map (mosaic.rain_data_tile)
                m = re.fullmatch(r"/tiles/(raindata|raindataown|raindataeu)/([\w-]+)/(\d+)-(\d+)/(\d+)/(\d+)/(\d+)\.(png|webp)", u.path)
                if not m:
                    return self.send(404, b"not found", "text/plain", 60)
                kind, key = m.group(1), m.group(2)
                t0, n, z, x, y = map(int, m.groups()[2:7])
                fmt = m.group(8)  # lossless WebP (server-v1.69, tileimg.py) or PNG: the same values
                if not (0 <= x < (1 << z) and 0 <= y < (1 << z)):
                    return self.send(404, b"bad tile", "text/plain", 3600)
                png = MOSAIC.rain_data_tile(key, t0, n, z, x, y, use_de=kind != "raindataeu",
                                            variant="own" if kind == "raindataown" else "dwd", fmt=fmt)
                if png is None:
                    return self.send(404, b"bad tile", "text/plain", 3600)
                return self.send(200, png, tileimg.TYPES[fmt], 86400 if MOSAIC.key_known(key) else 60)
            if u.path.startswith("/tiles/clouddata/"):  # all sunny frames of one tile in one atlas (app v1.13)
                # "<t0>-<n>m8": cells with 8 px of the neighbour tiles around them (server-v2.02, app v1.59)
                m = re.fullmatch(r"/tiles/clouddata/(\d+)/(\d+)-(\d+)(m8)?/(\d+)/(\d+)/(\d+)\.(png|webp)", u.path)
                if not m:
                    return self.send(404, b"not found", "text/plain", 60)
                key = m.group(1)
                t0, n = int(m.group(2)), int(m.group(3))
                z, x, y = map(int, m.groups()[4:7])
                fmt, margin = m.group(8), 8 if m.group(4) else 0
                if not (0 <= x < (1 << z) and 0 <= y < (1 << z)):
                    return self.send(404, b"bad tile", "text/plain", 3600)
                png = MOSAIC.cloud_data_tile(key, t0, n, z, x, y, fmt=fmt, margin=margin)
                if png is None:
                    return self.send(404, b"no atlas", "text/plain", 60)
                return self.send(200, png, tileimg.TYPES[fmt], 86400 if MOSAIC.cloud_key_known(key) else 60)
            if u.path.startswith("/tiles/"):
                m = re.fullmatch(r"/tiles/(rain|rainown|raineu|cloud)/([\w-]+)/(\d+)/(\d+)/(\d+)/(\d+)\.(png|webp)", u.path)
                if not m:
                    return self.send(404, b"not found", "text/plain", 60)
                kind, key, ts, z, x, y = m.group(1), m.group(2), *map(int, m.groups()[2:6])
                fmt = m.group(7)
                variant = "own" if kind == "rainown" else "dwd"
                if fmt == "webp" and kind != "cloud":  # rain pictures stay PNG
                    return self.send(404, b"not found", "text/plain", 3600)
                if not (2 <= z <= 13 and 0 <= x < (1 << z) and 0 <= y < (1 << z)):
                    return self.send(404, b"bad tile", "text/plain", 3600)
                png = (MOSAIC.cloud_tile(key, ts, z, x, y, fmt) if kind == "cloud"
                       else MOSAIC.rain_tile(key, ts, z, x, y, use_de=kind != "raineu", variant=variant))
                if png is None:
                    return self.send(404, b"unknown frame", "text/plain", 30)
                # the key names the run: immutable while that run is kept; an older key (a tab asleep for a while)
                # gets the current data, cached only briefly
                return self.send(200, png, tileimg.TYPES[fmt], 86400 if (MOSAIC.cloud_key_known(key) if kind == "cloud" else MOSAIC.key_known(key)) else 60)
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
            if u.path == "/api/warnings/point" and WARNINGS is not None:  # official warnings at a place
                import official_warnings
                ws = WARNINGS.at(round(float(q["lat"]), 2), round(float(q["lon"]), 2), lang=q.get("lang", "de")[:5])
                attr = {src: official_warnings.ATTRIBUTION.get(src) for src in {w["source"] for w in ws}}
                return self.send(200, json.dumps({"warnings": ws, "attribution": attr}, ensure_ascii=False).encode(),
                                 "application/json", 60)
            if u.path == "/api/warnings/areas" and WARNINGS is not None:  # ?bbox=west,south,east,north (optional)
                bbox = tuple(map(float, q["bbox"].split(","))) if q.get("bbox") else None
                return self.send(200, json.dumps(WARNINGS.areas_geojson(bbox)).encode(), "application/json", 60)
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
                    cov = MOSAIC.covered(lat, lon)
                    # no radar here (Europe / North America instances, server-v1.79): no series, so the app shows the
                    # weather model instead of a run of zeros ("dry for 2 hours" under the hatching); the Germany
                    # edition keeps its series (the app uses the model there itself when covered is false)
                    series = MOSAIC.rain_point(lat, lon, de, variant) if cov or EDITION == "de" else []
                    body = {"key": ck[0], "series": series, "covered": cov}
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
                return self.send(200, coverage_mask(bbox, w, h, q.get("kind", "rain")), "image/png", 120)
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
# sunny: Germany (DWD satellite) / Europe (DWD satellite, EU area) / the Americas (GOES-East, sunny_na.py; North America
# instance, 15.11.2026)
SUNNY = (sunny_na.SunnyGoes(fetch, log, sky=REGIONS["eu"].metar.sky) if NA
         else sunny_de.SunnyDE(fetch, log, None, OPENMETEO_URL, openmeteo_available, area=sunny_de.EUROPE) if EDITION == "eu"
         else sunny_de.SunnyDE(fetch, log, DE, OPENMETEO_URL, openmeteo_available))
SUNNY_ON = EDITION in ("de", "eu") and os.environ.get("SUNNY_DE", "1") != "0"
SKY = SUNNY if SUNNY_ON else CLOUDS
MOSAIC = mosaic.Mosaic(REGIONS, DE, SKY, sample, palette.paint, log, de_only=EDITION == "de", ptype=PTYPE)
THUNDER = thunder.Thunder(fetch, log)
WARNINGS = None  # official warnings (official_warnings.py): DWD in the de instance, MeteoAlarm in api-eu, NWS/ECCC in api-na
# Last successful refresh (ms) of the thunder and warning feeds, for source_ages (server-v1.79): a failing feed kept
# its old areas on the map with nothing watching it (audit 20.11.2026: 3 failed lightning fetches on 19.11. unnoticed).
FEED_OK = {}
STARTED_MS = time.time() * 1000


def watch_feed(obj, method, name):
    """Wrap obj.method so each run that does not raise stamps FEED_OK[name] (or FEED_OK[name + ":" + arg])."""
    fn = getattr(obj, method, None)
    if fn is None:
        return

    def run(*a, **k):
        out = fn(*a, **k)
        FEED_OK[f"{name}:{a[0]}" if a else name] = time.time() * 1000
        return out
    setattr(obj, method, run)


watch_feed(THUNDER, "refresh_warnings", "thunder")


def warnings_start(regions):
    """Official severe-weather warnings for the app's place card / map (16.11.2026; competitors show them): only served
    by /api/warnings/point and /api/warnings/areas - the app does not use them yet (design to be chosen by the user)."""
    global WARNINGS
    import official_warnings
    WARNINGS = official_warnings.Warnings(fetch, log, os.path.join(DATA_DIR, "warnings"), regions=regions)
    watch_feed(WARNINGS, "refresh", "warn")  # one age per source: warn:dwd, warn:meteoalarm, warn:nws, ...
    for s in WARNINGS.sources:
        FEED_OK.setdefault(f"warn:{s}", None)
    threading.Thread(target=WARNINGS.loop, daemon=True).start()
    # 18.11.2026: the warnings check our own data (warn_check.py) - archive them and our forecasts for the daily check
    import warn_check
    name = "de" if EDITION == "de" else "na" if NA else "eu"
    threading.Thread(target=warn_check.Recorder(
        log, WARNINGS, THUNDER if EDITION == "de" else None, warn_check.RegionFc(DE if EDITION == "de" else REGIONS["eu"], name),
        name, skip=() if EDITION == "de" else ("dwd",)).loop, daemon=True).start()  # api-eu: DWD is archived by api
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
    out = {"rss": rss_mb(), "rain_age": round((now - ref) / 60_000, 1) if ref else None,
           "sun_age": round((now - sun) / 60_000, 1) if sun else None, "src": source_ages(now)}
    if sun and sun in getattr(SUNNY, "ir_only", ()):  # newest sunny slot from the IR cloud tops alone (server-v1.94)
        out["sun_ir_only"] = True
    if EDITION == "de":
        out.update(om_probe_take())  # "om" / "om_pub": weather model probes since the last row (server-v1.86)
    elif not NA and REGIONS["eu"].node_seen is not None:  # radars missing from the newest OPERA image (server-v1.91)
        out["radars_out"] = list(REGIONS["eu"].nodes_missing)
        dead = {l.name: len(l.dead) for l in getattr(REGIONS["eu"], "layers", []) if getattr(l, "dead", None)}
        if dead:
            out["layer_radars_out"] = dead  # national layers: boxes of a dropped-out radar (national.COLLAPSE_MIN)
    return out


def source_ages(now):
    """Minutes since the newest data of every source this instance reads (server-v1.70, user 19.11.2026: "we cannot
    trust our sources" - the daily report counts how long each one was missing)."""
    out = {}
    try:  # never lose the health row over this
        if EDITION == "de":
            out["dwd"] = DE.ref
        else:
            eu = REGIONS["eu"]
            out["mrms" if NA else "opera"] = getattr(eu, "src_newest", None) if NA else eu.opera_newest
            for l in getattr(eu, "layers", ()):
                out[l.name] = l.newest_ts()
        if SUNNY_ON or NA:
            out["sunny"] = SUNNY.ref
        feeds = {k: v for k, v in FEED_OK.items() if k.startswith("warn:")}  # official warnings per source
        if EDITION == "de":  # the thunder feeds run in the Germany instance only (server-v1.79)
            feeds["thunder"] = FEED_OK.get("thunder")  # DWD NowCastMIX warning areas
            feeds["lightning"] = THUNDER.lightning[0] if THUNDER.lightning else None  # EUMETSAT MTG LI
        # a feed that has not answered yet counts as missing only after the first 15 min (no alarm at every start)
        out.update({k: v for k, v in feeds.items() if v or now - STARTED_MS > 15 * 60_000})
    except Exception as e:
        log(f"source ages failed: {e}")
    return {k: round((now - v) / 60_000, 1) if v else None for k, v in out.items()}


def model_at_gauges_loop(stations=None, out_dir=None, every_h=3):
    """Every 3 hours (North America: every hour, its METAR / SWOB stations - the model baseline next to the
    nowcast's +60 / +120 in check_na_gauges.py): the weather model's forecast (Open-Meteo best match, as the app's 24 h / 3-day views show it) for the next
    24 hours at every rain gauge of check_national_gauges -> verify_log/modelgauge/<issue time>.npz (t, precip mm per
    hour, temperature 2 m; 14 days). The truth is the gauge archive: check_model_gauges.py scores the 24 h forecast
    by lead and region - the base for correcting it per place (user, 14./15.11.2026; meteoblue / WetterOnline do this)."""
    import check_national_gauges as cng
    out_dir = out_dir or os.path.join(cng.ROOT, "modelgauge")
    stations = stations or cng.stations
    time.sleep(300)
    while True:
        try:
            st = stations()
            if st and openmeteo_available():
                issue = int(time.time() // 3600 * 3600 * 1000)
                t, pr, tt = None, [], []
                for i in range(0, len(st), 150):
                    part = st[i:i + 150]
                    q = (f"{OPENMETEO_URL}/v1/forecast?latitude={','.join(f'{x[2]:.3f}' for x in part)}"
                         f"&longitude={','.join(f'{x[3]:.3f}' for x in part)}&hourly=precipitation,temperature_2m"
                         f"&forecast_hours=25&timeformat=unixtime")
                    d = json.loads(fetch(q, timeout=120))
                    d = d if isinstance(d, list) else [d]
                    for x in d:
                        h = x["hourly"]
                        t = np.array(h["time"], np.int64) * 1000
                        pr.append([np.nan if v is None else v for v in h["precipitation"]][:len(t)])
                        tt.append([np.nan if v is None else v for v in h["temperature_2m"]][:len(t)])
                os.makedirs(out_dir, exist_ok=True)
                tmp = os.path.join(out_dir, f"{issue}.tmp.npz")
                np.savez_compressed(tmp, t=t, precip=np.array(pr, np.float16), temp=np.array(tt, np.float16),
                                    ids=np.array([f"{x[0]}:{x[1]}" for x in st]))
                os.replace(tmp, os.path.join(out_dir, f"{issue}.npz"))
                old = time.time() * 1000 - 14 * 86400_000
                for fp in glob.glob(os.path.join(out_dir, "*.npz")):
                    if os.path.basename(fp)[:-4].isdigit() and int(os.path.basename(fp)[:-4]) < old:
                        os.remove(fp)
        except Exception as e:
            log(f"model at gauges failed: {e}")
        # every 3 hours at :10 (one pass reads ~3600 places and took 6 min the first time: the Nordic / UK best-match
        # models are read from the remote mirror) - 8 issue times a day are plenty for scores by lead
        time.sleep(every_h * 3600 - time.time() % (every_h * 3600) + 600)


def synop_start():
    """All-Europe SYNOP reports (synop.py): hourly rain at stations in every country (gauge network "sy") and the cloud
    cover as a second station truth for sunny Europe (files eusysunstation_*: score_sun.py --prefix eusy)."""
    import synop
    feed = synop.SynopFeed(fetch, log, os.environ.get("VERIFY_LOG_DIR", "."))
    threading.Thread(target=feed.loop, daemon=True).start()
    if SUNNY_ON:
        def _cc(sid):
            try:
                return synop.country(int(sid))
            except ValueError:
                return None
        SUNNY.extra_verify.append(sunny_de.SunVerify(SUNNY, feed.sky, log, prefix="eusy", stations_only=True, group=_cc))
        # measured sunshine (19.11.2026): SYNOP hourly sunshine duration at ~340 stations in 16 countries, scored with
        # the forecast for the middle of the hour (synop.HourlySun) - the fair Europe-wide sunny truth; the cloud cover
        # rows (eusy) stay as a cloud truth (score_sun.py ... --prefix eusd --broken --by)
        SUNNY.extra_verify.append(sunny_de.SunVerify(SUNNY, feed.sun, log, prefix="eusd", stations_only=True, group=_cc,
                                                     lag_ms=30 * 60_000))


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
    if CLUTTER_MASK:
        try:
            import clutter
            vdir = os.environ.get("VERIFY_LOG_DIR", ".")
            eu.clutter = clutter.ClutterMask("na" if NA else "eun" if EU_NATIONAL else "eu", (eu.NY, eu.NX), eu.KM_PER_PX,
                                             vdir, log, archive=os.path.join(vdir, "obs_eu"), lonlat=eu.pixel_lonlat)
        except Exception as e:  # never keep the instance from starting
            log(f"clutter mask not started: {e}")
    eu.load_state()
    threading.Thread(target=update_loop, args=(eu,), daemon=True).start()
    if os.environ.get("EU_MODEL_BLEND", "1") != "0":  # only the verification shadow "cells" needs the weather model
        eu.model = ModelRain(eu)
        threading.Thread(target=model_loop, args=(eu.model,), daemon=True).start()
    # own-picture instance (EU_NATIONAL, the app's default Europe picture) only when NWP_BLEND=1 is set explicitly
    if NWP_BLEND and (not EU_NATIONAL or os.environ.get("NWP_BLEND") == "1"):  # the newest rapid-update model runs for the shadow "nwp" (nwp_blend.py)
        import nwp_blend
        eu.nwp = nwp_blend.NwpLive(eu, "na" if NA else "eu", eu.RAW_DOWN, log, lock=SPROG_LOCK)
        threading.Thread(target=eu.nwp.loop, daemon=True).start()
    threading.Thread(target=EU_VERIFY.loop, daemon=True).start()
    if not EU_NATIONAL and not NA:  # measurements that are needed once, not per instance: in the OPERA instance
        threading.Thread(target=gauge_archive_loop, daemon=True).start()
        threading.Thread(target=model_at_gauges_loop, daemon=True).start()
        threading.Thread(target=tile_gap_loop, daemon=True).start()
        synop_start()
        # competitors' open nowcasts at our gauges (bench_competitors.py, 16.11.2026: MET Norway / Yr, RMI INCA-BE,
        # CHMI COTREC, HungaroMet MEANDER, KNMI; 17.11.2026: DWD RV, MeteoSwiss, GeoSphere INCA)
        # - score_competitors.py
        import bench_competitors
        import check_national_gauges as cng
        threading.Thread(target=bench_competitors.CompetitorBench(
            fetch, log, os.environ.get("VERIFY_LOG_DIR", "."), cng.stations).loop, daemon=True).start()
    # Where no radar reaches (Greece, W Turkey, east of Poland ...) the own picture drew the satellite rain estimate
    # (H SAF, ~30 min old, from cloud tops) faintly into the hatched area. User, 13.11.2026: it shows rain inside
    # the "no data" hatching and does not fit the radar picture next to it -> off (SAT_FILL=1 brings it back).
    if EU_NATIONAL and os.environ.get("SAT_FILL") == "1":
        threading.Thread(target=update_loop, args=(REGIONS["sat"],), daemon=True).start()
    # second benchmark: GeoSphere Austria's INCA nowcast against ours at the Austrian rain gauges (bench_inca.py)
    if not NA:
        threading.Thread(target=bench_inca.IncaBench(fetch, log, eu, mosaic.merc).loop, daemon=True).start()
    else:  # the US / Canadian rain gauges (METAR, SWOB) and the model at them for the North America instance
        threading.Thread(target=eu.metar.loop, daemon=True).start()
        if os.environ.get("NA_MINUTE_GAUGES", "1") != "0":
            # 10-minute sums at ~1100 of those gauges (ASOS 5-minute reports via the IEM, ECCC SWOB minute records)
            # -> verify_log/na/gaugetruth: the truth for alerts and forecast timing (minute_gauges.py, 17.11.2026)
            import minute_gauges
            threading.Thread(target=minute_gauges.MinuteGauges(
                fetch, os.environ.get("VERIFY_LOG_DIR", "."), log, eu.metar.stations, eu.metar.precip_ok).loop,
                daemon=True).start()
        import bench_competitors  # ECCC's radar extrapolation (Canada + USA, to +72 min) at a sample of our gauges
        threading.Thread(target=bench_competitors.CompetitorBench(
            fetch, log, os.environ.get("VERIFY_LOG_DIR", "."), eu.metar.stations, na=True).loop, daemon=True).start()
        threading.Thread(target=model_at_gauges_loop, daemon=True,
                         kwargs={"stations": eu.metar.stations, "every_h": 1,
                                 "out_dir": os.path.join(os.environ.get("VERIFY_LOG_DIR", "."), "modelgauge")}).start()
    if SUNNY_ON:
        threading.Thread(target=SUNNY.loop, daemon=True).start()
    if not EU_NATIONAL:  # the own-picture instance shares api-eu's warnings (the app asks /eu or /na)
        warnings_start(("na",) if NA else ("eu",))
    if os.environ.get("PRERENDER", "1") != "0":  # the third instance draws its tiles on demand (CPU for the other two)
        threading.Thread(target=mosaic.prerender_box_loop, args=(MOSAIC, log, eu.BOX if NA else mosaic.EUROPE_BOX, mosaic.EU_PRERENDER_ZOOMS),
                         daemon=True).start()
    if eu.gadj is not None:  # GAUGE_ADJUST=1: the network factors, every 10 minutes (gauge_adjust.py)
        threading.Thread(target=eu.gadj.loop, daemon=True).start()
    threading.Thread(target=memory_loop, daemon=True).start()
    log(f"nowcast: {'pySTEPS S-PROG' if HAVE_PYSTEPS else 'simple (pySTEPS not installed)'}, Europe grid {EU_KM} km"
        f"{', national layers: ' + ', '.join(l.name for l in eu.layers) if EU_NATIONAL else ''}")
    log(f"rainy {VERSION} ({INSTANCE} instance, eu edition) on http://0.0.0.0:{PORT}")
    ThreadingHTTPServer(("0.0.0.0", PORT), Handler).serve_forever()


if __name__ == "__main__" and EDITION == "eu":
    main_eu()
elif __name__ == "__main__":
    monitor.start(EDITION, VERSION, monitor_status)
    threading.Thread(target=ALERTS.loop, args=(ll_to_3857,), daemon=True).start()
    if EDITION == "de" and not DEV:  # push message to the operator when an instance has no fresh images (opswatch.py)
        import opswatch
        threading.Thread(target=opswatch.OpsWatch(ALERTS, DATA_DIR, log).loop, daemon=True).start()
        threading.Thread(target=om_probe_loop, daemon=True).start()  # is the weather model answering? (server-v1.86)
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
    if EDITION == "de":
        warnings_start(("de",))
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
