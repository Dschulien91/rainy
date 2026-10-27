"""rainy data server.

Builds rain radar with a 2 h nowcast for regions where no official nowcast is freely available, and serves
it to the app. Regions:
  eu - EUMETNET OPERA composite (reflectivity, 1 km, every 5 min, CC BY 4.0), LAEA grid kept at 2 km
  us - NOAA nowCOAST base reflectivity mosaic (public domain, every ~4 min), Web Mercator grid ~3 km
Nowcast: dense optical flow (OpenCV Farneback) between two frames ~15 min apart, then semi-Lagrangian
extrapolation in 5 min steps (same principle as pySTEPS' extrapolation nowcast).

Endpoints (all CORS-enabled, cache headers suitable for a CDN):
  GET /api/health
  GET /api/radar/frames?region=eu|us          -> {"frames": [{"ts": ms, "fc": bool}], "ref": ms}
  GET /api/radar/image?region=&ts=&bbox=&width=&height=   (bbox in EPSG:3857) -> PNG in the rainy palette
  GET /api/radar/point?lat=&lon=               -> {"region": "eu"|"us"|null, "series": [{"ts", "mmh"}], "ref": ms}

Run:  python server/rainy_server.py   (port from env PORT, default 8771)
Requests to data sources carry a neutral User-Agent and no identifying information.
"""
import io
import json
import math
import os
import re
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

VERSION = "server-v0.2"
PORT = int(os.environ.get("PORT", "8771"))
UA = "Mozilla/5.0"          # neutral, carries no identity
HISTORY_MS = 60 * 60_000    # observed frames kept (1 h)
FC_STEPS = 24               # nowcast steps of 5 min (2 h)
STEP_MS = 5 * 60_000
POLL_S = 60
NODATA = np.float32(-1.0)   # outside radar coverage

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


def fetch(url, timeout=40):
    req = urllib.request.Request(url, headers={"User-Agent": UA})
    with urllib.request.urlopen(req, timeout=timeout) as r:
        return r.read()


# ---------- nowcast ----------
def nowcast(prev, last, dt_ms, steps):
    """Extrapolate `last` along the motion observed between `prev` and `last` (dt_ms apart)."""
    ny, nx = last.shape

    def img(r):
        v = np.where(r > 0, np.log1p(np.clip(r, 0, 50)), 0)
        return np.clip(v / math.log1p(50) * 255, 0, 255).astype(np.uint8)
    small = (nx // 2, ny // 2)  # flow on half resolution for speed and stability
    a = cv2.resize(img(prev), small, interpolation=cv2.INTER_AREA)
    b = cv2.resize(img(last), small, interpolation=cv2.INTER_AREA)
    flow = cv2.calcOpticalFlowFarneback(a, b, None, 0.5, 4, 25, 4, 7, 1.5, 0)
    flow = cv2.GaussianBlur(flow, (0, 0), 6)
    flow = cv2.resize(flow, (nx, ny), interpolation=cv2.INTER_LINEAR) * 2.0  # pixels per dt
    flow *= STEP_MS / dt_ms                                                  # pixels per 5 min step
    gx, gy = np.meshgrid(np.arange(nx, dtype=np.float32), np.arange(ny, dtype=np.float32))
    base = np.where(last < 0, 0, last).astype(np.float32)
    out = []
    for k in range(1, steps + 1):
        r = cv2.remap(base, gx - flow[..., 0] * k, gy - flow[..., 1] * k, cv2.INTER_LINEAR,
                      borderMode=cv2.BORDER_CONSTANT, borderValue=0)
        r[last < 0] = NODATA  # keep the coverage outline stable
        out.append(r)
    return out


# ---------- regions ----------
class Region:
    name = ""
    credit = ""

    def __init__(self):
        self.lock = threading.Lock()
        self.obs, self.fc, self.ref = {}, {}, None

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
            g = self.load(avail[t])
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
            prev_t = next((t for t in reversed(times) if last_t - t >= 14 * 60_000), times[0])
            prev, last = self.obs[prev_t], self.obs[last_t]
        if prev_t == last_t:
            return len(new)
        fc = nowcast(prev, last, last_t - prev_t, FC_STEPS)
        with self.lock:
            self.fc = {last_t + (i + 1) * STEP_MS: g for i, g in enumerate(fc)}
            self.ref = last_t
        return len(new)


class EuropeOpera(Region):
    name, credit = "eu", "EUMETNET OPERA"
    BUCKET = "https://s3.waw3-1.cloudferro.com/openradar-24h"
    LAEA = "+proj=laea +lat_0=55.0 +lon_0=10.0 +x_0=1950000.0 +y_0=-2100000.0 +units=m +ellps=WGS84"
    CELL, NY, NX = 2000.0, 2200, 1900

    def __init__(self):
        super().__init__()
        self.tr = Transformer.from_crs("EPSG:3857", self.LAEA, always_xy=True)

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
        rate = dbz_to_rate(dbz).reshape(self.NY, 2, self.NX, 2).mean(axis=(1, 3))   # 1 km -> 2 km
        covered = (~nod).reshape(self.NY, 2, self.NX, 2).any(axis=(1, 3))
        rate = cv2.medianBlur(rate, 3)                                               # single-pixel clutter
        rate[~covered] = NODATA
        return rate

    def to_pixels(self, x, y):
        gx, gy = self.tr.transform(x, y)
        return (np.asarray(gx) / self.CELL).astype(np.int32), (-np.asarray(gy) / self.CELL).astype(np.int32)


class UsaNowcoast(Region):
    name, credit = "us", "NOAA"
    WMS = "https://nowcoast.noaa.gov/geoserver/observations/weather_radar/ows"
    BBOX = (-14000000.0, 2700000.0, -7400000.0, 6500000.0)  # CONUS in EPSG:3857
    NX, NY = 2200, 1267
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
        col = ((np.asarray(x) - x0) / (x1 - x0) * self.NX).astype(np.int32)
        row = ((y1 - np.asarray(y)) / (y1 - y0) * self.NY).astype(np.int32)
        return col, row


REGIONS = {r.name: r for r in (EuropeOpera(), UsaNowcoast())}
ll_to_3857 = Transformer.from_crs("EPSG:4326", "EPSG:3857", always_xy=True)


def update_loop(region):
    while True:
        started = time.time()
        try:
            n = region.update()
            if n:
                _, ref = region.frames()
                print(f"[{datetime.now():%H:%M:%S}] {region.name}: +{n} frames, latest "
                      f"{datetime.fromtimestamp(ref / 1000, timezone.utc):%H:%M}Z, {time.time() - started:.1f}s", flush=True)
        except Exception as e:  # keep serving the last good data
            print(f"{region.name} update failed: {e}", flush=True)
        time.sleep(POLL_S)


# ---------- rendering ----------
def render(region, grid, bbox, w, h):
    minx, miny, maxx, maxy = bbox
    xs = minx + (np.arange(w) + 0.5) / w * (maxx - minx)
    ys = maxy - (np.arange(h) + 0.5) / h * (maxy - miny)
    X, Y = np.meshgrid(xs, ys)
    col, row = region.to_pixels(X, Y)
    ny, nx = grid.shape
    inside = (col >= 0) & (col < nx) & (row >= 0) & (row < ny)
    rate = np.zeros((h, w), np.float32)
    rate[inside] = grid[row[inside], col[inside]]
    buf = io.BytesIO()
    Image.fromarray(colorize(np.where(rate < 0, 0, rate)), "RGBA").save(buf, "PNG", compress_level=3)
    return buf.getvalue()


def point_series(lat, lon):
    x, y = ll_to_3857.transform(lon, lat)
    for region in REGIONS.values():
        col, row = (int(v) for v in region.to_pixels(x, y))
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
            if u.path == "/api/health":
                info = {name: {"frames": len(r.frames()[0]), "ref": r.ref} for name, r in REGIONS.items()}
                body = {"version": VERSION, "regions": info, "openmeteo": os.environ.get("OPENMETEO") == "1"}
                return self.send(200, json.dumps(body).encode(), "application/json", 10)
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
                return self.send(200, render(region, grid, bbox, w, h), "image/png", 3600)
            if u.path == "/api/radar/point":
                name, series, ref = point_series(float(q["lat"]), float(q["lon"]))
                return self.send(200, json.dumps({"region": name, "series": series, "ref": ref}).encode(), "application/json", 60)
            return self.send(404, b"not found", "text/plain")
        except Exception as e:
            return self.send(400, str(e).encode(), "text/plain", 0)


if __name__ == "__main__":
    for r in REGIONS.values():
        threading.Thread(target=update_loop, args=(r,), daemon=True).start()
    print(f"rainy {VERSION} on http://0.0.0.0:{PORT}", flush=True)
    ThreadingHTTPServer(("0.0.0.0", PORT), Handler).serve_forever()
