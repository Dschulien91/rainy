"""Our own Europe radar picture: the European composite (EUMETNET OPERA) with national radar products laid on top.

Why (user, 13.11.2026): OPERA has holes - measured on 13.11.: southern Italy 12 % covered, central Italy 79 %,
Sardinia 62 %, Greece 0 % - and it is one uniform 2 km product, while several weather services publish their own
national composite: finer, gauge-adjusted, with their own radars. The third instance of the server (EU_NATIONAL=1,
behind /eun/) serves this mosaic with the same nowcast, tiles and verification as the OPERA instance, so the two can
be compared number by number and switched in the app.

A layer = one national product: it polls its source, keeps the last ~80 min resampled to the host grid (OPERA's
grid at EU_KM) inside its own window, and says how much it is trusted per pixel:
  weight 1 inside its coverage, falling to 0 over FEATHER_KM towards the edge of its coverage - no hard seams where
  it meets OPERA; where OPERA has nothing, the layer counts alone.
The host region (rainy_server.EuropeNational) waits with a new time step until every layer that is alive has its
image for that step (the sources publish 5-12 min after the measurement), at most WAIT_MIN; a layer whose source is
down is skipped (OPERA alone there) and logged.

Layers:
  it  Italy - Dipartimento della Protezione Civile, Radar-DPC: SRI (surface rain intensity, mm/h, gauge-adjusted),
      1 km, every 5 min, GeoTIFF, transverse Mercator (lat0 42, lon0 12.5, WGS84), no account.
      Licence CC BY-SA ("citare la fonte Radar-DPC", derived works under the same licence).
"""
import io
import json
import threading
import time
import urllib.request
from datetime import datetime, timezone

import cv2
import numpy as np
from PIL import Image
from pyproj import Transformer

STEP_MS = 5 * 60_000
KEEP_MS = 80 * 60_000
FEATHER_KM = 30.0
WAIT_MIN = 20          # a time step is composed without a layer's image after this long
ALIVE_MIN = 30         # a layer that delivered within this time is waited for
NEAR_MS = 10 * 60_000  # without the exact image: the nearest one within this distance


class Layer:
    name, credit = "", ""
    PROJ = ""           # proj string of the product's grid
    X0, Y1, CELL, NX, NY = 0.0, 0.0, 1000.0, 0, 0   # top-left corner, cell size (m), size

    def __init__(self, host, log, ua):
        self.host, self.log, self.ua = host, log, ua
        self.lock = threading.Lock()
        self.frames = {}        # ts -> (rate on the host window, float16, < 0 = no data; weight float16)
        self.first_poll_done = False
        self.failed = 0.0
        self._window()

    # ---------- geometry ----------
    def _window(self):
        """The part of the host grid this product covers, and where each of its pixels lies in the product."""
        to_ll = Transformer.from_crs(self.PROJ, "EPSG:4326", always_xy=True)
        xs = self.X0 + np.array([0, self.NX, self.NX, 0, self.NX / 2, self.NX / 2, 0, self.NX]) * self.CELL
        ys = self.Y1 - np.array([0, 0, self.NY, self.NY, 0, self.NY, self.NY / 2, self.NY / 2]) * self.CELL
        lon, lat = to_ll.transform(xs, ys)
        m = Transformer.from_crs("EPSG:4326", "EPSG:3857", always_xy=True)
        col, row = self.host.to_pixels(*m.transform(lon, lat))
        h = self.host
        c0, c1 = max(int(np.floor(col.min())), 0), min(int(np.ceil(col.max())) + 1, h.NX)
        r0, r1 = max(int(np.floor(row.min())), 0), min(int(np.ceil(row.max())) + 1, h.NY)
        self.win = (slice(r0, r1), slice(c0, c1))
        cc, rr = np.meshgrid(np.arange(c0, c1), np.arange(r0, r1))
        lon, lat = h.pixel_lonlat(cc, rr)
        x, y = Transformer.from_crs("EPSG:4326", self.PROJ, always_xy=True).transform(lon, lat)
        self.map_col = ((np.asarray(x) - self.X0) / self.CELL - 0.5).astype(np.float32)
        self.map_row = ((self.Y1 - np.asarray(y)) / self.CELL - 0.5).astype(np.float32)
        self.k = max(1, int(round(h.KM_PER_PX * 1000 / self.CELL)))  # product pixels per host pixel

    def to_host(self, rate):
        """Product grid (mm/h, < 0 = no data) -> (rate, weight) on the host window. Mean over the product pixels of a
        host cell (box filter before the bilinear lookup), no data where less than half of them have data."""
        ok = (rate >= 0).astype(np.float32)
        val = np.where(rate >= 0, rate, 0).astype(np.float32)
        if self.k > 1:
            ok, val = cv2.blur(ok, (self.k, self.k)), cv2.blur(val, (self.k, self.k))
        kw = dict(interpolation=cv2.INTER_LINEAR, borderMode=cv2.BORDER_CONSTANT, borderValue=0)
        cov = cv2.remap(ok, self.map_col, self.map_row, **kw)
        v = cv2.remap(val, self.map_col, self.map_row, **kw)
        inside = cov >= 0.5
        out = np.where(inside, v / np.maximum(cov, 1e-6), -1.0).astype(np.float32)
        dist = cv2.distanceTransform(inside.astype(np.uint8), cv2.DIST_L2, 3) * self.host.KM_PER_PX
        return out.astype(np.float16), np.clip(dist / FEATHER_KM, 0, 1).astype(np.float16)

    # ---------- source (subclasses) ----------
    def newest(self):
        """Time (ms) of the newest image the source has."""
        raise NotImplementedError

    def load(self, ts):
        """The product grid (float32 mm/h, < 0 = no data) of the image at ts; raises when it does not exist."""
        raise NotImplementedError

    def _get(self, url, data=None, timeout=40):
        req = urllib.request.Request(url, data=data, headers={"User-Agent": self.ua, **(
            {"Content-Type": "application/json"} if data else {})})
        with urllib.request.urlopen(req, timeout=timeout) as r:
            return r.read()

    # ---------- store ----------
    def poll(self):
        newest = self.newest()
        now = time.time() * 1000
        want = [t for t in range(newest - KEEP_MS + STEP_MS, newest + 1, STEP_MS) if t not in self.frames and t > now - KEEP_MS]
        got = 0
        for t in sorted(want, reverse=True):  # the newest first: the map waits for it
            try:
                frame = self.to_host(self.load(t))
            except Exception:
                if t == newest:
                    raise
                frame = None  # an older step the source does not have (any more): not asked again
            with self.lock:
                if frame is not None:
                    self.frames[t] = frame
                    got += 1
                else:
                    self.frames.setdefault(t, None)  # not asked again
        with self.lock:
            for t in [t for t in self.frames if t < now - KEEP_MS]:
                del self.frames[t]
        return got

    def loop(self):
        while True:
            try:
                n = self.poll()
                if n:
                    with self.lock:
                        last = max(t for t, f in self.frames.items() if f is not None)
                    self.log(f"[{datetime.now():%H:%M:%S}] layer {self.name}: +{n} images, latest "
                             f"{datetime.fromtimestamp(last / 1000, timezone.utc):%H:%MZ}")
            except Exception as e:
                if time.time() - self.failed > 600:
                    self.failed = time.time()
                    self.log(f"layer {self.name} failed: {e}")
            self.first_poll_done = True
            time.sleep(30)

    def newest_ts(self):
        with self.lock:
            return max((t for t, f in self.frames.items() if f is not None), default=None)

    def alive(self):
        t = self.newest_ts()
        return t is not None and time.time() * 1000 - t < ALIVE_MIN * 60_000

    def has(self, ts):
        with self.lock:
            return self.frames.get(ts) is not None

    def get(self, ts):
        """(rate, weight) of the image at ts, else of the nearest one within NEAR_MS, else None."""
        with self.lock:
            ok = [t for t, f in self.frames.items() if f is not None and abs(t - ts) <= NEAR_MS]
            return self.frames[min(ok, key=lambda t: abs(t - ts))] if ok else None

    def blend(self, grid, ts):
        """Lay this layer's image over `grid` (host grid, float32, < 0 = no data), in place. True when it was there."""
        f = self.get(ts)
        if f is None:
            return False
        rate, w = (a.astype(np.float32) for a in f)
        g = grid[self.win]
        have, base = w > 0, g >= 0
        g[...] = np.where(have & base, w * np.maximum(rate, 0) + (1 - w) * g, np.where(have, np.maximum(rate, 0), g))
        return True


class ItalyDpc(Layer):
    name, credit = "it", "Radar-DPC (Dipartimento della Protezione Civile, CC BY-SA)"
    API = "https://radar-api.protezionecivile.it"
    PROJ = "+proj=tmerc +lat_0=42 +lon_0=12.5 +k=1 +x_0=0 +y_0=0 +ellps=WGS84 +units=m"
    X0, Y1, CELL, NX, NY = -600000.0, 650000.0, 1000.0, 1200, 1400
    PRODUCT = "SRI"

    def newest(self):
        d = json.loads(self._get(f"{self.API}/findLastProductByType?type={self.PRODUCT}", timeout=30))
        return int(d["lastProducts"][0]["time"])

    def load(self, ts):
        d = json.loads(self._get(f"{self.API}/downloadProduct", timeout=30,
                                 data=json.dumps({"productType": self.PRODUCT, "productDate": int(ts)}).encode()))
        a = np.asarray(Image.open(io.BytesIO(self._get(d["url"], timeout=60))), np.float32)
        if a.shape != (self.NY, self.NX):
            raise ValueError(f"unexpected grid {a.shape}")
        return np.where(np.isfinite(a) & (a >= 0), np.minimum(a, 300.0), -1.0).astype(np.float32)


LAYERS = (ItalyDpc,)


def start(host, log, ua):
    """Create the layers for the host region and start their polling threads."""
    layers = []
    for cls in LAYERS:
        try:
            layer = cls(host, log, ua)
            threading.Thread(target=layer.loop, daemon=True).start()
            layers.append(layer)
        except Exception as e:
            log(f"layer {cls.name} failed to start: {e}")
    return layers
