"""rainy data server (local prototype).

Pulls the pan-European EUMETNET OPERA radar composite (reflectivity, 1 km, every 5 min, CC BY 4.0),
converts it to rain rate, computes a 2 h nowcast (optical flow + semi-Lagrangian extrapolation),
and serves it to the app:

  GET /api/radar/frames                  -> {"frames": [{"ts": ms, "fc": bool}], "ref": ms}
  GET /api/radar/image?ts=&bbox=&width=&height=   (bbox in EPSG:3857) -> PNG in the rainy palette
  GET /api/radar/point?lat=&lon=         -> {"series": [{"ts": ms, "mmh": float}], "ref": ms} or {"series": []}

Run:  python server/rainy_server.py   (port 8771)
No identifying headers are sent to data sources.
"""
import io
import json
import math
import threading
import time
import urllib.request
from datetime import datetime, timedelta, timezone
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from urllib.parse import parse_qs, urlparse

import cv2
import h5py
import numpy as np
from PIL import Image
from pyproj import Transformer

VERSION = "server-v0.1"
PORT = 8771
BUCKET = "https://s3.waw3-1.cloudferro.com/openradar-24h"
HISTORY = 12            # observed frames kept (1 h)
FC_STEPS = 24           # nowcast steps of 5 min (2 h)
STEP_MS = 5 * 60_000
POLL_S = 60
UA = "Mozilla/5.0"      # neutral, carries no identity

# OPERA grid: LAEA, upper-left corner at (0, 0), 1 km source pixels; we keep 2 km
LAEA = "+proj=laea +lat_0=55.0 +lon_0=10.0 +x_0=1950000.0 +y_0=-2100000.0 +units=m +ellps=WGS84"
CELL = 2000.0
NY, NX = 2200, 1900
to_grid = Transformer.from_crs("EPSG:3857", LAEA, always_xy=True)
ll_to_grid = Transformer.from_crs("EPSG:4326", LAEA, always_xy=True)

NODATA = np.float32(-1.0)  # outside radar coverage

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


# ---------- data ----------
def fetch(url, timeout=30):
    req = urllib.request.Request(url, headers={"User-Agent": UA})
    with urllib.request.urlopen(req, timeout=timeout) as r:
        return r.read()


def list_keys(day):
    xml = fetch(f"{BUCKET}/?prefix={day:%Y/%m/%d}/OPERA/COMP/&max-keys=1000").decode()
    keys = []
    for part in xml.split("<Key>")[1:]:
        k = part.split("</Key>")[0]
        if k.endswith("@DBZH.h5"):
            keys.append(k)
    return keys


def key_time(key):
    stamp = key.split("OPERA@")[1].split("@")[0]  # 20261027T1355
    return int(datetime.strptime(stamp, "%Y%m%dT%H%M").replace(tzinfo=timezone.utc).timestamp() * 1000)


def load_rate(raw):
    """OPERA DBZH (1 km) -> rain rate mm/h on the 2 km grid; NODATA outside coverage."""
    with h5py.File(io.BytesIO(raw), "r") as f:
        d = f["dataset1/data1/data"][:]
        w = f["dataset1/data1/what"].attrs
        gain, offset, nodata, undetect = (float(w[k]) for k in ("gain", "offset", "nodata", "undetect"))
    nod = d == nodata
    und = d == undetect
    dbz = d * gain + offset
    dbz[und | nod] = -32.0
    rate = np.where(dbz >= 10.0, (10 ** (dbz / 10.0) / 200.0) ** (1 / 1.6), 0.0).astype(np.float32)  # Marshall-Palmer
    rate = rate.reshape(NY, 2, NX, 2).mean(axis=(1, 3))        # 1 km -> 2 km
    covered = (~nod).reshape(NY, 2, NX, 2).any(axis=(1, 3))
    rate = cv2.medianBlur(rate, 3)                               # remove single-pixel clutter
    rate[~covered] = NODATA
    return rate


def nowcast(prev, last, steps):
    """Extrapolate `last` along the motion between `prev` and `last` (15 min apart)."""
    def img(r):
        v = np.where(r > 0, np.log1p(np.clip(r, 0, 50)), 0)
        return np.clip(v / math.log1p(50) * 255, 0, 255).astype(np.uint8)
    small = (NX // 2, NY // 2)  # optical flow on 4 km for speed and stability
    a = cv2.resize(img(prev), small, interpolation=cv2.INTER_AREA)
    b = cv2.resize(img(last), small, interpolation=cv2.INTER_AREA)
    flow = cv2.calcOpticalFlowFarneback(a, b, None, 0.5, 4, 25, 4, 7, 1.5, 0)
    flow = cv2.GaussianBlur(flow, (0, 0), 6)
    flow = cv2.resize(flow, (NX, NY), interpolation=cv2.INTER_LINEAR) * 2.0  # back to 2 km pixels, per 15 min
    flow /= 3.0                                                              # per 5 min step
    gx, gy = np.meshgrid(np.arange(NX, dtype=np.float32), np.arange(NY, dtype=np.float32))
    base = np.where(last < 0, 0, last).astype(np.float32)
    out = []
    for k in range(1, steps + 1):
        mx = gx - flow[..., 0] * k
        my = gy - flow[..., 1] * k
        r = cv2.remap(base, mx, my, cv2.INTER_LINEAR, borderMode=cv2.BORDER_CONSTANT, borderValue=0)
        r[last < 0] = NODATA  # keep coverage outline stable
        out.append(r)
    return out


class Store:
    def __init__(self):
        self.lock = threading.Lock()
        self.obs = {}          # ts -> rate grid
        self.fc = {}           # ts -> rate grid
        self.ref = None

    def frames(self):
        with self.lock:
            return ([{"ts": t, "fc": False} for t in sorted(self.obs)] +
                    [{"ts": t, "fc": True} for t in sorted(self.fc)]), self.ref

    def grid(self, ts):
        with self.lock:
            return self.obs.get(ts) if ts in self.obs else self.fc.get(ts)


STORE = Store()


def update_loop():
    while True:
        try:
            now = datetime.now(timezone.utc)
            keys = list_keys(now) + (list_keys(now - timedelta(days=1)) if now.hour == 0 else [])
            keys = sorted(keys, key=key_time)[-HISTORY:]
            new = [k for k in keys if key_time(k) not in STORE.obs]
            for k in new:
                rate = load_rate(fetch(f"{BUCKET}/{k}"))
                with STORE.lock:
                    STORE.obs[key_time(k)] = rate
            if new:
                with STORE.lock:
                    keep = {key_time(k) for k in keys}
                    for t in list(STORE.obs):
                        if t not in keep:
                            del STORE.obs[t]
                    times = sorted(STORE.obs)
                    last_t = times[-1]
                    prev_t = next((t for t in reversed(times) if last_t - t >= 15 * 60_000), times[0])
                    prev, last = STORE.obs[prev_t], STORE.obs[last_t]
                started = time.time()
                fc = nowcast(prev, last, FC_STEPS)
                with STORE.lock:
                    STORE.fc = {last_t + (i + 1) * STEP_MS: g for i, g in enumerate(fc)}
                    STORE.ref = last_t
                print(f"[{datetime.now():%H:%M:%S}] +{len(new)} frames, latest {datetime.fromtimestamp(last_t / 1000, timezone.utc):%H:%M}Z, "
                      f"nowcast {time.time() - started:.1f}s", flush=True)
        except Exception as e:  # keep serving the last good data
            print("update failed:", e, flush=True)
        time.sleep(POLL_S)


# ---------- rendering ----------
def render(grid, bbox, w, h):
    minx, miny, maxx, maxy = bbox
    xs = minx + (np.arange(w) + 0.5) / w * (maxx - minx)
    ys = maxy - (np.arange(h) + 0.5) / h * (maxy - miny)
    X, Y = np.meshgrid(xs, ys)
    gx, gy = to_grid.transform(X, Y)
    col = (gx / CELL).astype(np.int32)
    row = (-gy / CELL).astype(np.int32)
    inside = (col >= 0) & (col < NX) & (row >= 0) & (row < NY)
    rate = np.zeros((h, w), np.float32)
    rate[inside] = grid[row[inside], col[inside]]
    rgba = colorize(np.where(rate < 0, 0, rate))
    buf = io.BytesIO()
    Image.fromarray(rgba, "RGBA").save(buf, "PNG", optimize=False, compress_level=3)
    return buf.getvalue()


def point_series(lat, lon):
    gx, gy = ll_to_grid.transform(lon, lat)
    col, row = int(gx / CELL), int(-gy / CELL)
    if not (1 <= col < NX - 1 and 1 <= row < NY - 1):
        return []
    frames, _ = STORE.frames()
    out = []
    for fr in frames:
        g = STORE.grid(fr["ts"])
        if g is None:
            continue
        cell = g[row - 1:row + 2, col - 1:col + 2]
        if (cell < 0).all():
            return []  # no radar coverage here
        out.append({"ts": fr["ts"], "mmh": round(float(cell[cell >= 0].mean()), 3)})
    return out


class Handler(BaseHTTPRequestHandler):
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
            if u.path == "/api/radar/frames":
                frames, ref = STORE.frames()
                return self.send(200, json.dumps({"frames": frames, "ref": ref, "version": VERSION}).encode(), "application/json", 30)
            if u.path == "/api/radar/image":
                grid = STORE.grid(int(q["ts"]))
                if grid is None:
                    return self.send(404, b"unknown frame", "text/plain")
                bbox = [float(v) for v in q["bbox"].split(",")]
                w, h = min(int(q["width"]), 1024), min(int(q["height"]), 1024)
                return self.send(200, render(grid, bbox, w, h), "image/png", 600)
            if u.path == "/api/radar/point":
                series = point_series(float(q["lat"]), float(q["lon"]))
                _, ref = STORE.frames()
                return self.send(200, json.dumps({"series": series, "ref": ref}).encode(), "application/json", 60)
            return self.send(404, b"not found", "text/plain")
        except Exception as e:
            return self.send(400, str(e).encode(), "text/plain", 0)


if __name__ == "__main__":
    threading.Thread(target=update_loop, daemon=True).start()
    print(f"rainy {VERSION} on http://0.0.0.0:{PORT}", flush=True)
    ThreadingHTTPServer(("0.0.0.0", PORT), Handler).serve_forever()
