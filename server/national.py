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
      Used only over Italy and the seas around it (ItalyDpc.AREA): far from the Italian radars the product is much too
      wet - 13.11.2026, 33 Austrian gauges in Carinthia over 2.5 h: gauges 0.07 mm/h, OPERA 0.32, SRI 2.71 (wet at
      460 of 475 gauge readings, the gauges at 35); Klagenfurt 3.9 mm/h at 20 dBZ maximum reflectivity.
  de  Germany - Deutscher Wetterdienst, open data RV composite: the analysis of each 5-min run (mm per 5 min, 1 km,
      polar stereographic, ODIM HDF5 in a tar with the 24 forecast steps), x GAIN 1.5 as in the Germany edition
      (measured against ~300 DWD gauges, mosaic.GermanyDwd.GAIN_RV). Published ~5 min after the measurement.
  ch  Switzerland - MeteoSwiss open data (CC BY), product RZC: rain rate (mm/h) from the five Swiss radars with
      MeteoSwiss' own corrections for the Alps, 1 km, every 5 min, ODIM HDF5, Swiss oblique Mercator; one STAC item
      per day (data.geo.admin.ch, collection ch.meteoschweiz.ogd-radar-precip), files rzcYYDDDHHMM??.001.h5 stamped
      with the END of the 5 minutes, published ~4 min later.
"""
import io
import json
import re
import tarfile
import threading
import time
import urllib.request
from datetime import datetime, timedelta, timezone

import cv2
import h5py
import numpy as np
from PIL import Image, ImageDraw
from pyproj import Transformer

import mosaic

STEP_MS = 5 * 60_000
KEEP_MS = 80 * 60_000
FEATHER_KM = 30.0
WAIT_MIN = 20          # a time step is composed without a layer's image after this long
ALIVE_MIN = 30         # a layer that delivered within this time is waited for
NEAR_MS = 10 * 60_000  # without the exact image: the nearest one within this distance


class Layer:
    name, credit = "", ""
    SHIFT_MS = 0        # an image stamped T belongs to the host's time step T + SHIFT_MS (see timing())
    PROJ = ""           # proj string of the product's grid
    X0, Y1, CELL, NX, NY = 0.0, 0.0, 1000.0, 0, 0   # top-left corner, cell size (m), size
    AREA = None         # [(lon, lat), ...]: the product is used only inside this polygon (None: everywhere it has data)

    def __init__(self, host, log, ua):
        self.host, self.log, self.ua = host, log, ua
        self.lock = threading.Lock()
        self.frames = {}        # ts -> (rate on the host window, float16, < 0 = no data; weight float16)
        self.first_poll_done = False
        self.failed = 0.0
        self.geometry()
        self._window()

    def geometry(self):
        """Products that describe their own grid set PROJ, X0, Y1, CELL, NX, NY here (from a first file)."""

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
        self.area = None
        if self.AREA:
            x, y = Transformer.from_crs("EPSG:4326", self.PROJ, always_xy=True).transform(*zip(*self.AREA))
            im = Image.new("L", (self.NX, self.NY), 0)
            ImageDraw.Draw(im).polygon([((a - self.X0) / self.CELL, (self.Y1 - b) / self.CELL) for a, b in zip(x, y)], fill=1)
            self.area = np.asarray(im).astype(bool)

    def to_host(self, rate):
        """Product grid (mm/h, < 0 = no data) -> (rate, weight) on the host window. Mean over the product pixels of a
        host cell (box filter before the bilinear lookup), no data where less than half of them have data."""
        if self.area is not None:
            rate = np.where(self.area, rate, -1.0)
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
            return self.frames.get(ts - self.SHIFT_MS) is not None

    def get(self, ts):
        """(rate, weight) of the image at ts, else of the nearest one within NEAR_MS, else None."""
        ts -= self.SHIFT_MS
        with self.lock:
            ok = [t for t, f in self.frames.items() if f is not None and abs(t - ts) <= NEAR_MS]
            return self.frames[min(ok, key=lambda t: abs(t - ts))] if ok else None



def compose(grid, layers, ts):
    """Lay the layers' images for the step ts over `grid` (host grid, float32, < 0 = no data), in place; returns the
    names of the layers that had an image. Where layers overlap each counts with its weight (the one whose coverage
    edge is farther away counts more); what is left of the weight (1 - sum, near the edges) goes to OPERA, and where
    OPERA has nothing the layers count alone."""
    num = np.zeros(grid.shape, np.float32)
    den = np.zeros(grid.shape, np.float32)
    used = []
    for l in layers:
        f = l.get(ts)
        if f is None:
            continue
        rate, w = (a.astype(np.float32) for a in f)
        num[l.win] += w * np.maximum(rate, 0)
        den[l.win] += w
        used.append(l.name)
    if used:
        have, base = den > 0, grid >= 0
        mean = num / np.maximum(den, 1e-6)
        rest = np.clip(1 - den, 0, 1)
        grid[...] = np.where(have & base, (1 - rest) * mean + rest * grid, np.where(have, mean, grid))
    return used


class ItalyDpc(Layer):
    name, credit = "it", "Radar-DPC (Dipartimento della Protezione Civile, CC BY-SA)"
    API = "https://radar-api.protezionecivile.it"
    PROJ = "+proj=tmerc +lat_0=42 +lon_0=12.5 +k=1 +x_0=0 +y_0=0 +ellps=WGS84 +units=m"
    X0, Y1, CELL, NX, NY = -600000.0, 650000.0, 1000.0, 1200, 1400
    PRODUCT = "SRI"
    # Italy's borders (to ~10 km, the layer fades out over FEATHER_KM inside them anyway) and the seas around it up to
    # about the middle towards the neighbours; Corsica is left to OPERA (French radar).
    AREA = [(7.53, 43.78), (7.70, 44.10), (7.20, 44.20), (6.90, 44.40), (6.98, 44.68), (6.72, 44.93), (6.63, 45.11),
            (6.90, 45.22), (7.10, 45.45), (6.85, 45.68), (6.86, 45.83), (7.04, 45.92), (7.17, 45.87), (7.66, 45.98),
            (7.87, 45.93), (8.12, 46.22), (8.38, 46.45), (8.45, 46.25), (8.60, 46.12), (8.85, 46.00), (8.95, 45.83),
            (9.08, 45.90), (9.02, 46.05), (9.17, 46.17), (9.28, 46.50), (9.45, 46.50), (9.58, 46.29), (9.95, 46.38),
            (10.10, 46.23), (10.17, 46.40), (10.08, 46.60), (10.45, 46.55), (10.47, 46.86), (10.75, 46.80),
            (11.05, 46.80), (11.15, 46.95), (11.50, 47.00), (12.13, 47.09), (12.22, 46.88), (12.40, 46.70),
            (12.75, 46.64), (13.20, 46.59), (13.72, 46.52), (13.66, 46.44), (13.40, 46.32), (13.48, 46.20),
            (13.52, 46.05), (13.64, 45.95), (13.60, 45.80), (13.80, 45.73), (13.92, 45.63), (13.72, 45.60),
            (13.40, 45.50), (13.05, 45.10), (13.30, 44.50), (14.30, 43.60), (15.40, 43.00), (17.00, 42.10),
            (18.30, 41.20), (18.95, 40.30), (19.20, 39.40), (19.60, 34.50), (11.50, 34.50), (11.50, 37.30),
            (10.20, 37.90), (8.00, 38.20), (7.30, 38.40), (7.30, 41.20), (8.60, 41.33), (9.26, 41.31), (9.60, 41.36),
            (9.75, 41.60), (9.75, 42.90), (9.30, 43.35), (8.50, 43.50), (7.60, 43.60)]
    # timing() on 13.11. (twice, slow rain, so a flat curve): CSI against OPERA 0.43/0.44 unshifted, 0.44/0.45 with the
    # image stamped 5 min earlier, 0.44/0.45 with 10 - and the file is written ~7 min after its stamp, which fits a
    # stamp at the START of the 5-min scan (DWD, MeteoSwiss and OPERA stamp the end). To be re-checked on a day with
    # fast-moving rain over Italy.
    SHIFT_MS = STEP_MS

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


def odim_grid(f):
    """(proj string, x0, y1, cell, nx, ny) of an ODIM HDF5 composite from its /where group."""
    w = f["where"].attrs
    proj = w["projdef"].decode() if isinstance(w["projdef"], bytes) else str(w["projdef"])
    x, y = Transformer.from_crs("EPSG:4326", proj, always_xy=True).transform(float(w["UL_lon"]), float(w["UL_lat"]))
    return proj, float(x), float(y), float(w["xscale"]), int(w["xsize"]), int(w["ysize"])


class GermanyRv(Layer):
    name, credit = "de", "Deutscher Wetterdienst (open data)"
    RV = "https://opendata.dwd.de/weather/radar/composite/rv/"
    GAIN = 1.5

    def _runs(self):
        names = re.findall(r"composite_rv_(\d{8}_\d{4})\.tar", self._get(self.RV, timeout=60).decode("latin-1"))
        return sorted({int(datetime.strptime(n, "%Y%m%d_%H%M").replace(tzinfo=timezone.utc).timestamp() * 1000) for n in names})

    def _analysis(self, ts):
        name = datetime.fromtimestamp(ts / 1000, timezone.utc).strftime("composite_rv_%Y%m%d_%H%M.tar")
        with tarfile.open(fileobj=io.BytesIO(self._get(self.RV + name, timeout=120))) as tar:
            m = next(m for m in tar.getmembers() if m.name.rsplit("_", 1)[1].startswith("000"))
            return h5py.File(io.BytesIO(tar.extractfile(m).read()), "r")

    def geometry(self):
        with self._analysis(self._runs()[-1]) as f:
            self.PROJ, self.X0, self.Y1, self.CELL, self.NX, self.NY = odim_grid(f)

    def newest(self):
        return self._runs()[-1]

    def load(self, ts):
        with self._analysis(ts) as f:
            w = {k: float(f["dataset1/data1/what"].attrs[k]) for k in ("gain", "offset", "nodata")}
            raw = f["dataset1/data1/data"][:]
        nod = raw == w["nodata"]
        mm5 = mosaic.despeckle(np.where(nod, 0.0, np.maximum(raw * w["gain"] + w["offset"], 0.0)).astype(np.float32))
        return np.where(nod, -1.0, mm5 * 12.0 * self.GAIN).astype(np.float32)


class SwissRzc(Layer):
    name, credit = "ch", "MeteoSwiss (open data, CC BY)"
    STAC = "https://data.geo.admin.ch/api/stac/v1/collections/ch.meteoschweiz.ogd-radar-precip/items"

    def __init__(self, *a):
        self.files, self.listed = {}, 0.0   # ts -> url
        super().__init__(*a)

    def _list(self):
        if time.time() - self.listed < 45:
            return
        now = datetime.now(timezone.utc)
        days = [now] + ([now - timedelta(days=1)] if now.hour < 2 else [])
        for day in days:
            d = json.loads(self._get(f"{self.STAC}/{day:%Y%m%d}-ch", timeout=60))
            for k, a in d.get("assets", {}).items():
                m = re.match(r"rzc(\d{2})(\d{3})(\d{4})..\.001\.h5$", k)
                if m:
                    t = datetime.strptime(f"20{m.group(1)}{m.group(2)}{m.group(3)}", "%Y%j%H%M").replace(tzinfo=timezone.utc)
                    self.files[int(t.timestamp() * 1000)] = a["href"]
        old = time.time() * 1000 - 3 * KEEP_MS
        self.files = {t: u for t, u in self.files.items() if t > old}
        self.listed = time.time()

    def geometry(self):
        self._list()
        with h5py.File(io.BytesIO(self._get(self.files[max(self.files)], timeout=60)), "r") as f:
            self.PROJ, self.X0, self.Y1, self.CELL, self.NX, self.NY = odim_grid(f)

    def newest(self):
        self._list()
        return max(self.files)

    def load(self, ts):
        with h5py.File(io.BytesIO(self._get(self.files[ts], timeout=60)), "r") as f:
            a = np.asarray(f["dataset1/data1/data"][:], np.float32)
        return np.where(np.isfinite(a) & (a >= 0), np.minimum(a, 300.0), -1.0).astype(np.float32)


LAYERS = (ItalyDpc, GermanyRv, SwissRzc)


def timing(host, layer, shifts=(-10, -5, 0, 5, 10)):
    """How well the layer's wet area matches the host's own source (OPERA alone) for each time shift in minutes:
    {shift: mean CSI of "wet >= 0.3 mm/h" over the host's images}. The best shift says which of the layer's time
    stamps belongs to which host step (the services stamp the start or the end of the measurement)."""
    base = super(type(host), host)
    times = base.list_times()
    out = {}
    grids = {t: base.load(times[t])[layer.win] for t in sorted(times)[-8:]}
    for sh in shifts:
        scores = []
        for t, g in grids.items():
            with layer.lock:
                f = layer.frames.get(t - sh * 60_000)
            if f is None:
                continue
            rate, w = (a.astype(np.float32) for a in f)
            both = (w >= 1) & (g >= 0)
            a, b = (rate >= 0.3) & both, (g >= 0.3) & both
            if (a | b).sum() > 200:
                scores.append((a & b).sum() / (a | b).sum())
        out[sh] = round(float(np.mean(scores)), 3) if scores else None
    return out


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
