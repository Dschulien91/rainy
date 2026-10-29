"""One seamless, cacheable map for the whole app: map tiles (z/x/y, 256 px, Web Mercator) per time step.

Why tiles: every user gets the very same files for a time step, so a CDN (Cloudflare) serves almost all
requests and the server renders each tile only once per run - instead of one image per user and map view.
Every tile URL carries a key of the data runs it was made from, so a URL never changes its content
("immutable"): new data -> new key -> new URLs.

Rain mosaic (5 min steps, last hour + 2 h):
  Germany  - DWD radar composite with DWD's official 2 h nowcast (RV), fetched here once for all users;
             its weight fades from 1 inside Germany to 0 at FADE_KM beyond the border, so there is no seam
  Europe   - EUMETNET OPERA + our nowcast (rainy_server Region "eu")
  USA      - NOAA + our nowcast (Region "us")
  elsewhere on the Meteosat disc - EUMETSAT H SAF satellite estimate (Region "sat"), drawn fainter
Clouds: the cloudiness field of clouds.py, coloured in the app (sharp edges at every zoom level).

Endpoints (added to rainy_server.Handler):
  GET /api/rain/frames               -> {"key", "frames": [{"ts", "fc"}]}
  GET /api/rain/point?lat=&lon=      -> {"key", "series": [{"ts", "fc", "mmh"}]}   (same data as the map)
  GET /tiles/rain/<key>/<ts>/<z>/<x>/<y>.png
  GET /tiles/cloud/<key>/<ts>/<z>/<x>/<y>.png   (grey = cloudiness, alpha = covered)
"""
import io
import json
import os
import re
import threading
import tarfile
import time
from collections import OrderedDict
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime, timezone

import cv2
import h5py
import numpy as np
from PIL import Image

from ptype import NAMES as ptype_names, dwd_grid_maps

WORLD = 20037508.342789244
STEP_MS = 5 * 60_000
PRERENDER_ZOOMS = range(4, 9)  # Germany at zoom 4-8: ~120 tiles per frame
CACHE_TILES = 12000            # in-process tile cache (~1-3 KB each); holds all pre-rendered tiles of a run
TILE = 256
FADE_KM = 30.0


SPECKLE_PX = 10        # isolated echoes below ~10 km2 on the 1 km DWD grid ...
SPECKLE_MMH = 3.0      # ... that are also strong: clutter spikes are 20-60 mm/h, small real drizzle spots are weak


def despeckle(mm5):
    """Remove isolated tiny echoes: wind farms, ground clutter and interference show up as a few 1 km pixels with
    20-60 mm/h in dry surroundings (29.10.: six such spots, no lightning, nothing at WetterOnline). The DWD nowcast
    extrapolates them as standing 'cells', so they are removed in every frame. Small weak spots (drizzle) stay:
    WetterOnline shows them too."""
    wet = (mm5 > 0.05 / 12 / GermanyDwd.GAIN_RV).astype(np.uint8)  # the lowest drawn rate: drizzle fields stay whole
    n, lab, stats, _ = cv2.connectedComponentsWithStats(wet, connectivity=8)
    peak = np.zeros(n, np.float32)
    np.maximum.at(peak, lab.ravel(), mm5.ravel())
    small = np.zeros(n, bool)
    small[1:] = (stats[1:, cv2.CC_STAT_AREA] < SPECKLE_PX) & (peak[1:] * 12 * GermanyDwd.GAIN_RV > SPECKLE_MMH)
    return np.where(small[lab], 0.0, mm5).astype(np.float32)


def merc(lon, lat):
    x = np.asarray(lon, np.float64) * WORLD / 180.0
    y = np.log(np.tan((90.0 + np.asarray(lat, np.float64)) * np.pi / 360.0)) * WORLD / np.pi
    return x, y


def tile_xy(z, x, y, size=TILE):
    """Web Mercator coordinates of the pixel centres of tile z/x/y."""
    span = 2 * WORLD / (1 << z)
    xs = -WORLD + x * span + (np.arange(size) + 0.5) / size * span
    ys = WORLD - y * span - (np.arange(size) + 0.5) / size * span
    return np.meshgrid(xs, ys)


class GermanyDwd:
    """DWD radar composite plus DWD's own nowcast (RV) on a fixed Web Mercator grid.

    Source: the DWD open data RV composite (opendata.dwd.de/weather/radar/composite/rv): one file per 5-min run with
    the analysis and 24 forecast steps to +120 min as real values (mm per 5 min, 1 km). The DWD WMS (colour images
    in 15 classes) is only the fallback: it stalled for hours on 29.10. while open data kept updating."""
    RV = "https://opendata.dwd.de/weather/radar/composite/rv/"
    # Open data RV analysis against ~300 DWD 10-min gauges (1.9 days, server/check_rain_gauges.py --source rv): 0.67 of
    # the measured rain like the WMS; x1.5 matches the gauge totals (1.01) with the best CSI (>= 1 mm/h: 0.42)
    GAIN_RV = 1.5
    name = "de"
    WMS = "https://maps.dwd.de/geoserver/dwd/wms"
    LAYER = "dwd:Niederschlagsradar"
    BBOX = (*merc(2.0, 46.0), *merc(19.0, 56.0))   # x0, y0, x1, y1
    CELL = 1500.0                                   # ~0.95 km on the ground at 51 N
    KM_PER_PX = 0.95
    # The WMS radar is not gauge-adjusted: against ~300 DWD 10-min gauges it shows 0.65-0.67 of the measured rain
    # (2.5 days and 22 h, server/check_rain_gauges.py); x1.5 matches the gauge totals and gives the best CSI.
    GAIN = 1.5
    # legend colour -> representative mm/h (same table as the app's DWD fallback)
    CLASSES = {"33ffff": 0.14, "1acc9a": 0.28, "019934": 0.63, "4db31b": 1.4, "99cc01": 2.4, "cce601": 3.9,
               "ffff01": 6.1, "ffc401": 8.7, "ff8901": 12, "ff4501": 21, "fe0000": 37, "e5004c": 58,
               "cc0098": 87, "6600cb": 120, "0000fe": 160}

    def __init__(self, fetch, log):
        self.fetch, self.log = fetch, log
        x0, y0, x1, y1 = self.BBOX
        self.nx, self.ny = int(round((x1 - x0) / self.CELL)), int(round((y1 - y0) / self.CELL))
        self.lock = threading.Lock()
        self.obs, self.fc, self.ref = {}, {}, None
        self.rgb = np.array([[int(h[i:i + 2], 16) for i in (0, 2, 4)] for h in self.CLASSES], np.int32)
        self.val = np.array(list(self.CLASSES.values()), np.float32)
        self.rv_maps = None
        self.weight = self._germany_weight()
        # Germany for the hatching: the outline leaves estuaries (Elbe, Weser) and lagoons out as sea, so close
        # gaps up to ~10 km - inland waters must not look like "no data"
        inside = (self.weight > 0.99).astype(np.uint8)
        self.inside = cv2.morphologyEx(inside, cv2.MORPH_CLOSE, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (11, 11))).astype(np.float32)
        self.pool = ThreadPoolExecutor(4)

    def _germany_weight(self):
        """1 inside Germany, fading to 0 at FADE_KM outside (Natural Earth outline, public domain)."""
        with open(os.path.join(os.path.dirname(os.path.abspath(__file__)), "germany.json"), encoding="utf-8") as f:
            polys = json.load(f)["coordinates"]
        mask = np.zeros((self.ny, self.nx), np.uint8)
        for poly in polys:
            ring = np.array(poly[0], np.float64)
            col, row = self.to_pixels(*merc(ring[:, 0], ring[:, 1]))
            cv2.fillPoly(mask, [np.stack([col, row], 1).round().astype(np.int32)], 1)
        dist_px = cv2.distanceTransform((1 - mask).astype(np.uint8), cv2.DIST_L2, 5)
        km = dist_px * self.CELL * np.cos(np.radians(51.0)) / 1000.0
        return np.clip(1.0 - km / FADE_KM, 0.0, 1.0).astype(np.float32)

    def to_pixels(self, x, y):
        x0, y0, x1, y1 = self.BBOX
        return (np.asarray(x) - x0) / self.CELL - 0.5, (y1 - np.asarray(y)) / self.CELL - 0.5

    def frames(self):
        with self.lock:
            return ([{"ts": t, "fc": False} for t in sorted(self.obs)] +
                    [{"ts": t, "fc": True} for t in sorted(self.fc)]), self.ref

    def grid(self, ts):
        with self.lock:
            return self.obs.get(ts) if ts in self.obs else self.fc.get(ts)

    def _latest_run(self):
        xml = self.fetch(f"{self.WMS.rsplit('/', 1)[0]}/Niederschlagsradar/ows?service=WMS&version=1.3.0&request=GetCapabilities").decode("utf-8", "ignore")
        # newest run in the list of available runs: the "default" attribute now and then names a run hours old
        m = re.search(r'<Dimension name="REFERENCE_TIME"[^>]*>([^<]+)<', xml)
        runs = [int(datetime.strptime(v.strip()[:19], "%Y-%m-%dT%H:%M:%S").replace(tzinfo=timezone.utc).timestamp() * 1000)
                for v in m.group(1).split(",") if v.strip()[:2] == "20"]
        now = time.time() * 1000
        return max(t for t in runs if t <= now + 300_000)

    def _load(self, ts, ref=None):
        x0, y0, x1, y1 = self.BBOX
        iso = lambda t: datetime.fromtimestamp(t / 1000, timezone.utc).strftime("%Y-%m-%dT%H:%M:%S.000Z")
        url = (f"{self.WMS}?service=WMS&version=1.1.1&request=GetMap&layers={self.LAYER}&styles=&srs=EPSG:3857"
               f"&bbox={x0},{y0},{x1},{y1}&width={self.nx}&height={self.ny}&format=image/png&transparent=true&time={iso(ts)}"
               + (f"&DIM_REFERENCE_TIME={iso(ref)}" if ref else ""))
        for attempt in range(4):  # the WMS now and then answers with a random service exception: ask again
            try:
                px = np.asarray(Image.open(io.BytesIO(self.fetch(url, timeout=120))).convert("RGBA"), np.int32)
                break
            except OSError:
                if attempt == 3:
                    raise
                time.sleep(2 + 3 * attempt)
        rate = np.zeros((self.ny, self.nx), np.float32)
        wet = px[..., 3] > 200
        d2 = ((px[wet][:, :3][:, None, :] - self.rgb[None]) ** 2).sum(-1)
        rate[wet] = np.where(d2.min(-1) <= 3 * 30 * 30, self.val[d2.argmin(-1)], 0.0)  # outline etc. -> 0
        return (rate * self.GAIN).astype(np.float16)

    def _rv_runs(self):
        names = re.findall(r"composite_rv_(\d{8}_\d{4})\.tar", self.fetch(self.RV, timeout=60).decode("latin-1"))
        return sorted({int(datetime.strptime(n, "%Y%m%d_%H%M").replace(tzinfo=timezone.utc).timestamp() * 1000) for n in names})

    def _load_rv(self, run):
        """{ts: rate grid (mm/h, float16)} for the analysis and the 24 forecast steps of one RV run."""
        name = datetime.fromtimestamp(run / 1000, timezone.utc).strftime("composite_rv_%Y%m%d_%H%M.tar")
        out = {}
        with tarfile.open(fileobj=io.BytesIO(self.fetch(self.RV + name, timeout=120))) as tar:
            for m in tar.getmembers():
                lead = int(m.name.rsplit("_", 1)[1].split("-")[0])
                with h5py.File(io.BytesIO(tar.extractfile(m).read()), "r") as f:
                    if self.rv_maps is None:
                        self.rv_maps = dwd_grid_maps(f, self)
                    w = f["dataset1/data1/what"].attrs
                    raw = f["dataset1/data1/data"][:]
                    mm5 = despeckle(np.where(raw == w["nodata"], 0.0, np.maximum(raw * float(w["gain"]) + float(w["offset"]), 0.0)).astype(np.float32))
                g = cv2.remap(mm5 * 12.0 * self.GAIN_RV, *self.rv_maps, cv2.INTER_LINEAR, borderMode=cv2.BORDER_CONSTANT, borderValue=0)
                out[run + lead * 60_000] = g.astype(np.float16)
        return out

    def update(self):
        if getattr(self, "frozen", False):  # devtools.py test scenario in place
            return 0
        try:
            return self._update_rv()
        except Exception as e:
            self.log(f"de: open data RV failed ({e}), using the WMS")
            return self._update_wms()

    def _update_rv(self):
        runs = self._rv_runs()
        ref = runs[-1]
        if ref == self.ref:
            return 0
        obs_t = [ref - k * STEP_MS for k in range(12, -1, -1)]
        got = self._load_rv(ref)
        obs = {t: g for t, g in self.obs.items() if t in obs_t}
        obs[ref] = got[ref]
        n = 25
        for t in obs_t:  # history (first start): the analysis of each earlier run
            if t not in obs and t in runs:
                obs[t] = self._load_rv(t)[t]
                n += 1
        with self.lock:
            self.obs = obs
            self.fc, self.ref = {t: g for t, g in got.items() if t > ref}, ref
        return n

    def _update_wms(self):
        ref = self._latest_run()
        if self.ref and ref < self.ref:  # the capabilities now and then announce an old run as the default
            return 0
        obs_t = [ref - k * STEP_MS for k in range(12, -1, -1)]
        fc_t = [ref + k * STEP_MS for k in range(1, 25)]
        new_obs = [t for t in obs_t if t not in self.obs]
        new_fc = fc_t if ref != self.ref else []
        if not new_obs and not new_fc:
            return 0
        got_obs = dict(zip(new_obs, self.pool.map(self._load, new_obs)))
        got_fc = dict(zip(new_fc, self.pool.map(lambda t: self._load(t, ref), new_fc)))
        with self.lock:
            self.obs = {t: g for t, g in {**self.obs, **got_obs}.items() if t in obs_t}
            if got_fc:
                self.fc, self.ref = got_fc, ref
        return len(new_obs) + len(new_fc)


class Mosaic:
    def __init__(self, regions, de, clouds, sample, colorize, log, de_only=False, ptype=None):
        self.regions, self.de, self.clouds, self.sample, self.colorize, self.log = regions, de, clouds, sample, colorize, log
        self.ptype = ptype  # precipitation type from the DWD radar (Germany), or None
        self.de_only = de_only  # Germany edition: DWD only, fading out 30 km beyond the border, no satellite estimate
        self.cache, self.cache_lock = OrderedDict(), threading.Lock()  # small in-process cache; the CDN does the rest

    # ---------- timeline ----------
    def key(self):
        refs = [self.de.ref] + [self.regions[n].ref for n in ("eu", "us", "sat")]
        return "-".join(str((r or 0) // 60_000) for r in refs)

    def frames(self):
        now = int(time.time() * 1000)
        eu_ref = self.regions["eu"].ref or self.de.ref or now
        start = (now - 60 * 60_000) // STEP_MS * STEP_MS
        end = max(self.de.ref or 0, eu_ref) + 120 * 60_000
        last_obs = min(t for t in (self.de.ref, self.regions["eu"].ref) if t) if (self.de.ref or self.regions["eu"].ref) else now
        return [{"ts": t, "fc": t > last_obs} for t in range(start, end + 1, STEP_MS)]

    @staticmethod
    def _nearest(src, ts, tol_ms):
        frames, _ = src.frames()
        if not frames:
            return None
        best = min(frames, key=lambda f: abs(f["ts"] - ts))
        return src.grid(best["ts"]) if abs(best["ts"] - ts) <= tol_ms else None

    # ---------- composite ----------
    def rain(self, ts, X, Y, use_de=True):
        """Rain rate and a 'faint' flag (satellite estimate) at Web Mercator points X, Y.
        use_de=False: OPERA + our nowcast only (recorded in parallel to decide whether DWD is worth keeping)."""
        rate = np.zeros(X.shape, np.float32)
        covered = np.zeros(X.shape, bool)
        for name, tol in (("eu", 150_000), ("us", 180_000)):
            g = self._nearest(self.regions[name], ts, tol)
            if g is not None:
                r, cov = self.sample(self.regions[name], g, X, Y)
                rate[cov] = r[cov]
                covered |= cov
        g = self._nearest(self.de, ts, 150_000) if use_de else None
        if g is not None:
            col, row = (v.astype(np.float32) for v in self.de.to_pixels(X, Y))
            d = cv2.remap(g.astype(np.float32), col, row, cv2.INTER_LINEAR, borderMode=cv2.BORDER_CONSTANT, borderValue=0)
            w = cv2.remap(self.de.weight, col, row, cv2.INTER_LINEAR, borderMode=cv2.BORDER_CONSTANT, borderValue=0)
            rate = np.where(covered, w * d + (1 - w) * rate, w * d if self.de_only else np.where(w > 0, d, rate))
            covered |= w > 0
        faint = np.zeros(X.shape, bool) if self.de_only else ~covered
        if faint.any():
            g = self._nearest(self.regions["sat"], ts, 360_000)
            if g is not None:
                r, _ = self.sample(self.regions["sat"], g, X, Y)
                rate[faint] = r[faint]
        return rate, faint

    def _cached(self, key, make):
        with self.cache_lock:
            if key in self.cache:
                self.cache.move_to_end(key)
                return self.cache[key]
        png = make()
        with self.cache_lock:
            self.cache[key] = png
            while len(self.cache) > CACHE_TILES:
                self.cache.popitem(last=False)
        return png

    def rain_tile(self, key, ts, z, x, y, use_de=True):
        def make():
            X, Y = tile_xy(z, x, y)
            rate, faint = self.rain(ts, X, Y, use_de)
            rgba = self.colorize(rate, self.kind(ts, X, Y))
            rgba[..., 3] = np.where(faint, (rgba[..., 3] * 0.65).astype(np.uint8), rgba[..., 3])
            buf = io.BytesIO()
            Image.fromarray(rgba, "RGBA").save(buf, "PNG", compress_level=6)
            return buf.getvalue()
        return self._cached(f"{'r' if use_de else 're'}/{key}/{ts}/{z}/{x}/{y}", make)

    def cloud_tile(self, key, ts, z, x, y):
        span = 2 * WORLD / (1 << z)
        bbox = (-WORLD + x * span, WORLD - (y + 1) * span, -WORLD + (x + 1) * span, WORLD - y * span)
        return self._cached(f"c/{key}/{ts}/{z}/{x}/{y}", lambda: self.clouds.image(ts, bbox, TILE, TILE))

    def kind(self, ts, X, Y):
        """Precipitation type per point (ptype kinds) or None outside the data / without the classification."""
        if self.ptype is None:
            return None
        obs = [t for t, _ in self.de.obs.items()] if self.de.obs else []
        return self.ptype.kind(ts, X, Y, forecast=not obs or ts > max(obs))

    def rain_point(self, lat, lon, use_de=True):
        """Same composite as the map, mean over ~3 km around the place."""
        cx, cy = merc(lon, lat)
        d = np.array([-1500.0, 0.0, 1500.0]) / np.cos(np.radians(lat))
        X, Y = np.meshgrid(cx + d, cy + d)
        out = []
        for f in self.frames():
            rate, faint = self.rain(f["ts"], X, Y, use_de)
            e = {"ts": f["ts"], "fc": f["fc"], "mmh": round(float(rate.mean()), 3), "est": bool(faint.all())}
            k = self.kind(f["ts"], X, Y)
            if k is not None and (rate >= 0.1).any():  # most frequent type where it is wet
                e["kind"] = ptype_names[int(np.bincount(k[rate >= 0.1].ravel(), minlength=6).argmax())]
            out.append(e)
        return out


def prerender_loop(mosaic, log):
    """Render the German rain tiles of every frame (zoom PRERENDER_ZOOMS) as soon as a new radar run is in, so no
    user waits for them: the app asks for tiles up to zoom 8 only (enlarges them above), i.e. ~120 tiles per frame."""
    done = None
    while True:
        try:
            key = mosaic.key()
            if key != done and mosaic.de.ref:
                started, n = time.time(), 0
                (x0, x1), (y0, y1) = merc(np.array([5.5, 15.4]), np.array([47.0, 55.3]))  # Germany + ~30 km
                now = time.time() * 1000
                for f in sorted(mosaic.frames(), key=lambda f: abs(f["ts"] - now)):  # frames near "now" first
                    for z in PRERENDER_ZOOMS:
                        k = (1 << z) / (2 * WORLD)
                        for x in range(int((x0 + WORLD) * k), int((x1 + WORLD) * k) + 1):
                            for y in range(int((WORLD - y1) * k), int((WORLD - y0) * k) + 1):
                                mosaic.rain_tile(key, f["ts"], z, x, y)
                                n += 1
                    if mosaic.key() != key:  # a newer run arrived meanwhile: start over with it
                        break
                done = key
                log(f"[{datetime.now():%H:%M:%S}] prerender: {n} tiles in {time.time() - started:.0f}s")
        except Exception as e:
            log(f"prerender failed: {e}")
        time.sleep(15)


def de_loop(de, log, ptype=None):
    while True:
        started = time.time()
        try:
            n = de.update()
            if n:
                log(f"[{datetime.now():%H:%M:%S}] de: +{n} DWD frames, run "
                    f"{datetime.fromtimestamp(de.ref / 1000, timezone.utc):%H:%M}Z, {time.time() - started:.0f}s")
        except Exception as e:  # keep serving the last good data
            log(f"de update failed: {e}")
        if ptype is not None and not getattr(de, "frozen", False):
            try:
                ptype.update(list(de.obs))
            except Exception as e:
                log(f"ptype update failed: {e}")
        time.sleep(60)
