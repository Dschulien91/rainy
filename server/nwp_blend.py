"""Rapid-update weather models blended into the radar nowcast for +60..+120 min (scored shadow "nwp").

Why: the radar extrapolation (S-PROG + local_amount + advect_blend) cannot make new rain or kill old rain; a weather
model that assimilates radar every hour knows about growth, decay and new cells, but places rain worse in the first
hour. Literature: NWP alone crosses over extrapolation at 2-3 h, so it can only help with a weight that grows with the
lead time, measured from skill, after the model is corrected - the naive "add the model's rain" (blend_new_cells,
rejected 11.11.2026) painted dry lanes and x1.39 amount.

Models (Open-Meteo's open-data copies on AWS, read directly as .om files with the `omfiles` package; no API calls):
  ruc    DWD ICON-D2-RUC (2.1 km, hourly runs with radar latent-heat nudging, 15-min precipitation), Germany + all
         neighbours (43-58 N, 4 W-20 E). Licence: DWD open data, CC BY 4.0. Open-Meteo latency ~45 min.
  arome  Meteo-France AROME-PI (the 1.3 km nowcasting AROME, hourly runs to +6 h, 15-min precipitation), France +
         neighbours (37.5-55.4 N, 12 W-16 E). Licence: Etalab Licence Ouverte 2.0 (commercial use with attribution).
         Latency ~35 min.
  hrrr   NOAA HRRR CONUS (3 km, hourly runs with radar reflectivity assimilation, 15-min precipitation). US public
         domain. Latency ~75-90 min.
Not used: UKMO UK 2 km (CC BY-SA, ~4 h latency), KNMI / DMI HARMONIE (2.5-3 h latency, hourly steps only), MET Nordic
(hourly steps only), ICON-D2 (3-hourly runs, 85 min latency: RUC is the same model, newer), RRFS (2.6 h latency).

Method (per forecast run at analysis time ref, on the verification grid ~4 km):
 1. newest run per model that was available at ref (Open-Meteo's availability time), not older than MAX_AGE;
    its 15-min accumulations -> mm/h at the interval centres, linear in time to every 5-min step.
 2. correction against the radar analysis at ref, over the model's area: (a) quantile mapping of the model's field at
    ref onto the radar's distribution (amount and wet area); the same mapping for every lead; (b) optional: position
    correction - optical flow from the model's field at ref to the radar analysis, applied to every lead.
 3. models combined by their edge weights (0 at the model-area edge, 1 after EDGE_KM inside: no seams).
 4. blend: out = (1 - w) * extrapolation + w * model, w = weight(lead) * edge weight [* skill factor]; weight 0 in the
    first 30-45 min; optionally probability-matched to the w-mix of both fields' distributions (keeps the amount and
    the wet area between the two instead of the smeared union).
All of it is measured offline first (tune_blend.py) and live as the shadow "nwp" (fcverify).
"""
import bz2
import json
import math
import os
import threading
import time
import urllib.request
from datetime import datetime, timezone

import cv2
import numpy as np

S3 = "https://openmeteo.s3.amazonaws.com"
UA = {"User-Agent": "Mozilla/5.0"}  # anonymous, neutral: never anything that identifies us
DWD_GRID = "https://opendata.dwd.de/weather/nwp/icon-d2/grib/{h:02d}/{v}/icon-d2_germany_icosahedral_time-invariant_{d}{h:02d}_000_0_{v}.grib2.bz2"
HRRR_WKT = ('PROJCRS["LCC",BASEGEOGCRS["GCS_Sphere",DATUM["D_Sphere",ELLIPSOID["Sphere",6371229.0,0.0]]],'
            'CONVERSION["LCC",METHOD["Lambert Conic Conformal (2SP)"],PARAMETER["Latitude of 1st standard parallel",38.5],'
            'PARAMETER["Latitude of 2nd standard parallel",38.5],PARAMETER["Latitude of false origin",0.0],'
            'PARAMETER["Longitude of false origin",-97.5]],CS[Cartesian,2],AXIS["easting",east],AXIS["northing",north],'
            'LENGTHUNIT["metre",1]]')
MODELS = {
    "ruc": {"domain": "dwd_icon_d2_ruc_native_15min", "grid": ("icon_d2",), "region": "eu"},
    "arome": {"domain": "meteofrance_arome_france_hd_15min", "grid": ("latlon", 37.5, -12.0, 0.01, 1791, 2801), "region": "eu"},
    "hrrr": {"domain": "ncep_hrrr_conus_15min",
             "grid": ("lcc", HRRR_WKT, 21.137995, -122.72, 47.842403, -60.918, 1059, 1799), "region": "na"},
}
REGION_MODELS = {"eu": ("ruc", "arome"), "na": ("hrrr",)}
EDGE_KM = 50.0          # the weight fades to 0 over this distance at a model area's edge
MAX_AGE_MS = 4 * 3600_000  # a run older than this (init -> ref) is not used
CACHE_DIR = os.environ.get("NWP_CACHE_DIR", os.path.join(os.environ.get("DATA_DIR", "/tmp"), "nwp_cache"))
LOCAL_DIR = os.environ.get("NWP_DATA_DIR", "")  # Open-Meteo's data volume (read-only mount), if the models are synced


def _get(url, timeout=120):
    with urllib.request.urlopen(urllib.request.Request(url, headers=UA), timeout=timeout) as r:
        return r.read()


def _ms(s):
    """'2026-11-16T18:00Z' or '...:00:00Z' -> ms."""
    s = s.rstrip("Z")
    fmt = "%Y-%m-%dT%H:%M:%S" if s.count(":") == 2 else "%Y-%m-%dT%H:%M"
    return int(datetime.strptime(s, fmt).replace(tzinfo=timezone.utc).timestamp() * 1000)


# ---------- model grids -> the region's verification grid ----------
def model_points(grid):
    """lat, lon (float64, flat, in the model's storage order) of every model grid point."""
    kind = grid[0]
    if kind == "latlon":
        _, lat0, lon0, d, ny, nx = grid
        la, lo = np.meshgrid(lat0 + d * np.arange(ny), lon0 + d * np.arange(nx), indexing="ij")
        return la.ravel(), lo.ravel()
    if kind == "lcc":
        from pyproj import CRS, Transformer
        _, wkt, lat0, lon0, lat1, lon1, ny, nx = grid
        fwd = Transformer.from_crs("EPSG:4326", CRS.from_wkt(wkt), always_xy=True)
        x0, y0 = fwd.transform(lon0, lat0)
        x1, y1 = fwd.transform(lon1, lat1)
        xs, ys = np.meshgrid(np.linspace(x0, x1, nx), np.linspace(y0, y1, ny))  # row 0 = south (Open-Meteo order)
        lo, la = Transformer.from_crs(CRS.from_wkt(wkt), "EPSG:4326", always_xy=True).transform(xs.ravel(), ys.ravel())
        return np.asarray(la), np.asarray(lo)
    if kind == "icon_d2":  # ICON-D2's triangular grid: cell centres from DWD's time-invariant fields (0.8 MB, cached)
        path = os.path.join(CACHE_DIR, "icon_d2_grid.npz")
        if not os.path.exists(path):
            import eccodes
            out = {}
            day = datetime.now(timezone.utc)
            for v in ("clat", "clon"):
                for back in range(8):  # the newest 3-hourly run directory that has the file
                    t = datetime.fromtimestamp(day.timestamp() - back * 3 * 3600, timezone.utc)
                    h = t.hour // 3 * 3
                    try:
                        data = bz2.decompress(_get(DWD_GRID.format(h=h, v=v, d=t.strftime("%Y%m%d"))))
                        break
                    except Exception:
                        continue
                hd = eccodes.codes_new_from_message(data)
                out[v] = eccodes.codes_get_values(hd).astype(np.float32)
                eccodes.codes_release(hd)
            os.makedirs(CACHE_DIR, exist_ok=True)
            np.savez_compressed(path, lat=out["clat"], lon=out["clon"])
        z = np.load(path)
        return z["lat"].astype(np.float64), z["lon"].astype(np.float64)
    raise ValueError(kind)


class Mapper:
    """Model grid -> verification grid (shape h x w = the region grid / down): every model point falls into one
    pixel, a pixel's value is the mean of its points (the models are 1.3-3 km, the grid ~4 km); pixels without a point
    inside the model's area get their nearest neighbour's value. edge: 0..1, 0 outside the model's area, ramping up
    over EDGE_KM inside it."""

    def __init__(self, grid, region, down, shape, km_per_px, valid=None):
        from pyproj import Transformer
        from scipy import ndimage
        la, lo = model_points(grid)
        x, y = Transformer.from_crs("EPSG:4326", "EPSG:3857", always_xy=True).transform(lo, np.clip(la, -85, 85))
        col, row = region.to_pixels(np.asarray(x), np.asarray(y))
        c = np.floor((np.asarray(col, np.float64) + 0.5) / down).astype(np.int64)
        r = np.floor((np.asarray(row, np.float64) + 0.5) / down).astype(np.int64)
        h, w = shape
        ok = (c >= 0) & (c < w) & (r >= 0) & (r < h)
        if valid is not None:
            ok &= valid
        self.sel = np.flatnonzero(ok).astype(np.int32)   # model points on the grid
        self.pix = (r[ok] * w + c[ok]).astype(np.int32)  # their pixel
        del x, y, col, row, c, r, ok, la, lo
        self.shape, self.n, self.km = shape, h * w, km_per_px
        self.geo = self._area(np.bincount(self.pix, minlength=self.n))  # (fill, area, edge) of all points

    def _area(self, cnt):
        """Per-pixel point counts -> (nearest pixel with points, the model's area, edge weight 0..1)."""
        from scipy import ndimage
        h, w = self.shape
        has = cnt.reshape(self.shape) > 0
        # the model's area: pixels with points, small gaps closed (coarse grid spacing vs. the grid)
        area = cv2.morphologyEx(has.astype(np.uint8), cv2.MORPH_CLOSE, np.ones((5, 5), np.uint8)) > 0
        area = ndimage.binary_fill_holes(area)
        _, idx = ndimage.distance_transform_edt(~has, return_indices=True)
        fill = (idx[0] * w + idx[1]).ravel().astype(np.int32)
        edge = np.clip(ndimage.distance_transform_edt(area) * self.km / EDGE_KM, 0, 1).astype(np.float32)
        return fill, area, edge

    @property
    def edge(self):
        return self.geo[2]

    def finish(self, s, cnt, geo):
        """Per-pixel sums and counts of the model points (flat) -> grid (float32), 0 outside the model's area."""
        fill, area, _ = geo
        g = (np.asarray(s, np.float32) / np.maximum(cnt, 1))[fill].reshape(self.shape)
        g[~area] = 0
        return g

    def read(self, reader, idx, scale=1.0, band_values=4_000_000):
        """Time indices idx of an .om reader (spatial dims..., time) -> ((len(idx), h, w) float16 grids, edge weight).
        Missing values (NaN: AROME's grid box is larger than its area) count as outside the model's area; the area and
        its edge weight come from the first index. Read in bands of the first spatial dimension that has more than one
        row, all needed times at once (the files are chunked along time: one read per step would decode every chunk
        again), so memory stays ~band_values floats."""
        shp = tuple(reader.shape)
        sp, k0, k1 = shp[:-1], min(idx), max(idx) + 1
        dim = 0 if sp[0] > 1 else 1
        inner = int(np.prod(sp[dim + 1:])) if dim + 1 < len(sp) else 1
        rows = max(1, band_values // max(1, inner * (k1 - k0)))
        sums = np.zeros((len(idx), self.n), np.float32)
        cnt = np.zeros(self.n, np.float32)
        cols = [k - k0 for k in idx]
        for r0 in range(0, sp[dim], rows):
            r1 = min(sp[dim], r0 + rows)
            sl = [slice(None)] * len(sp) + [slice(k0, k1)]
            sl[dim] = slice(r0, r1)
            a = np.asarray(reader[tuple(sl)], np.float32).reshape(-1, k1 - k0)
            f0, f1 = r0 * inner, r1 * inner          # flat range of this band (dim 0 of size 1 when dim == 1)
            lo, hi = np.searchsorted(self.sel, f0), np.searchsorted(self.sel, f1)
            if hi <= lo:
                continue
            loc, pix = self.sel[lo:hi] - f0, self.pix[lo:hi]
            vals = a[loc][:, cols]
            ok = np.isfinite(vals[:, 0])
            pix, vals = pix[ok], np.nan_to_num(vals[ok], nan=0.0) * scale
            cnt += np.bincount(pix, minlength=self.n).astype(np.float32)
            for j in range(len(idx)):
                sums[j] += np.bincount(pix, weights=vals[:, j], minlength=self.n).astype(np.float32)
        geo = self._area(cnt)
        return np.stack([self.finish(s, cnt, geo) for s in sums]).astype(np.float16), geo[2]


# ---------- model runs ----------
class RunField:
    """One model run on the verification grid: rate(t) in mm/h for any t inside the run."""

    def __init__(self, model, init, avail, times, fields, edge):
        self.model, self.init, self.avail, self.edge = model, init, avail, edge  # edge: 0..1, this run's area
        self.times = np.asarray(times, np.int64)  # ms, centres of the 15-min intervals
        self.fields = fields                      # (T, h, w) float16 mm/h

    def at(self, t):
        ts = self.times
        if not ts[0] <= t <= ts[-1]:
            return None
        k = min(int(np.searchsorted(ts, t, side="right")) - 1, len(ts) - 2)
        f = (t - ts[k]) / (ts[k + 1] - ts[k])
        return ((1 - f) * self.fields[k].astype(np.float32) + f * self.fields[k + 1].astype(np.float32))


def _om_reader(path):
    from omfiles import OmFileReader
    return OmFileReader(path)


def archived_runs(model, day):
    """Open-Meteo's archive of past runs (data_run/, every 3rd run): [(init ms, run path)] of one UTC day."""
    import re
    dom = MODELS[model]["domain"]
    xml = _get(f"{S3}/?list-type=2&prefix=data_run/{dom}/{day:%Y/%m/%d}/&delimiter=/").decode()
    out = []
    for p in re.findall(r"<Prefix>(data_run/[^<]+/(\d{4})Z/)</Prefix>", xml):
        hhmm = p[1]
        t = day.replace(hour=int(hhmm[:2]), minute=int(hhmm[2:]), second=0, microsecond=0)
        out.append((int(t.timestamp() * 1000), p[0]))
    return out


def load_archived(model, run_path, mapper, t0, t1):
    """One archived run's fields for t0..t1 (ms, interval centres) -> RunField (None if missing)."""
    os.makedirs(CACHE_DIR, exist_ok=True)
    key = run_path.strip("/").replace("/", "_")
    om = os.path.join(CACHE_DIR, key + "_precipitation.om")
    meta_p = os.path.join(CACHE_DIR, key + "_meta.json")
    if not os.path.exists(meta_p):
        meta = json.loads(_get(f"{S3}/{run_path}meta.json"))
        with open(meta_p, "w") as f:
            json.dump(meta, f)
    meta = json.load(open(meta_p))
    if "precipitation" not in meta.get("variables", ["precipitation"]):
        return None
    if not os.path.exists(om):
        data = _get(f"{S3}/{run_path}precipitation.om", timeout=600)
        with open(om + ".tmp", "wb") as f:
            f.write(data)
        os.replace(om + ".tmp", om)
    ends = np.array([_ms(v) for v in meta["valid_times"]], np.int64)
    dt = int(meta.get("temporal_resolution_seconds", 900)) * 1000
    centres = ends - dt // 2
    rd = _om_reader(om)
    # HRRR's file has the run's start too (sum 0), ICON-D2-RUC's and AROME's begin with the first 15 minutes
    off = len(ends) - rd.shape[-1]
    idx = [k for k in range(max(1, off), len(ends)) if t0 - dt <= centres[k] <= t1 + dt]
    if len(idx) < 2:
        rd.close()
        return None
    fields, edge = mapper.read(rd, [k - off for k in idx], 3600_000 / dt)  # mm per interval -> mm/h
    rd.close()
    return RunField(model, _ms(meta["reference_time"]), _ms(meta["created_at"]), centres[idx], fields, edge)


def load_latest(model, mapper, t0, t1, known=None):
    """The newest run (Open-Meteo's data/: each new run overwrites the times it covers) for t0..t1 (ms) -> RunField.
    From the local Open-Meteo volume when the model is synced (NWP_DATA_DIR), else from Open-Meteo's open-data copy.
    known: the modification time of the run held already - returns None when nothing has changed since."""
    dom = MODELS[model]["domain"]
    local = os.path.join(LOCAL_DIR, dom) if LOCAL_DIR else None
    if local and os.path.exists(os.path.join(local, "static", "meta.json")):
        meta = json.load(open(os.path.join(local, "static", "meta.json")))
    else:
        local = None
        meta = json.loads(_get(f"{S3}/data/{dom}/static/meta.json"))
    if known is not None and meta.get("last_run_modification_time") == known:
        return None
    dt = int(meta["temporal_resolution_seconds"]) * 1000
    span = int(meta["chunk_time_length"]) * dt
    init, avail = meta["last_run_initialisation_time"] * 1000, meta["last_run_availability_time"] * 1000
    t0 = max(t0, init + dt // 2)
    # interval ends from the first needed one to the last
    t1 = min(t1, meta.get("data_end_time", 1 << 40) * 1000 - dt)
    ends = np.arange((t0 + dt // 2) // dt * dt, t1 + dt // 2 + dt, dt, dtype=np.int64)
    fields, centres, edge = [], [], None
    for chunk in sorted({int(e // span) for e in ends}):
        path = os.path.join(local, "precipitation", f"chunk_{chunk}.om") if local else None
        if not path or not os.path.exists(path):
            os.makedirs(CACHE_DIR, exist_ok=True)
            path = os.path.join(CACHE_DIR, f"latest_{dom}_{chunk}.om")
            url = f"{S3}/data/{dom}/precipitation/chunk_{chunk}.om"
            with urllib.request.urlopen(urllib.request.Request(url, headers=UA, method="HEAD"), timeout=60) as r:
                etag = r.headers.get("ETag", "")
            stamp = path + ".etag"  # the chunk is rewritten by every run: download only when it changed
            if not (os.path.exists(path) and os.path.exists(stamp) and open(stamp).read() == etag):
                data = _get(url, timeout=180)
                with open(path + ".tmp", "wb") as f:
                    f.write(data)
                os.replace(path + ".tmp", path)
                with open(stamp, "w") as f:
                    f.write(etag)
        rd = _om_reader(path)
        sel = [e for e in ends if e // span == chunk]
        idx = [int((e - chunk * span) // dt) for e in sel]
        f, ed = mapper.read(rd, idx, 3600_000 / dt)
        fields.append(f)
        edge = ed if edge is None else np.minimum(edge, ed)
        centres += [e - dt // 2 for e in sel]
        rd.close()
    try:  # downloaded chunks of past days (the newest run has moved on to the next chunk): delete
        for f in os.listdir(CACHE_DIR):
            p = os.path.join(CACHE_DIR, f)
            if f.startswith(f"latest_{dom}_") and time.time() - os.path.getmtime(p) > 86400:
                os.remove(p)
    except OSError:
        pass
    if len(centres) < 2:
        return None
    rf = RunField(model, init, avail, centres, np.concatenate(fields), edge)
    rf.mod = meta.get("last_run_modification_time")
    return rf


# ---------- correction and blending ----------
def quantile_map(src, ref, mask):
    """A function mapping src's values onto ref's distribution (both sampled on mask: same pixels)."""
    s = np.sort(src[mask].astype(np.float32))
    r = np.sort(ref[mask].astype(np.float32))
    n = len(s)
    if n < 100:
        return None

    def f(x):
        x = np.asarray(x, np.float32)
        p = np.searchsorted(s, x, side="left") + np.searchsorted(s, x, side="right")  # mid-rank of ties
        y = r[np.clip(p // 2, 0, n - 1)]
        return np.where(x <= 0, 0, y).astype(np.float32)
    return f


def _db(g):
    return np.where(g >= 0.1, 10 * np.log10(np.maximum(g, 0.1)), -10).astype(np.float32)


def position_flow(model0, radar0, mask, km_per_px, max_km=100.0, coarse_km=8.0):
    """Displacement (u, v in pixels of the grid) that moves the model's rain at ref onto the radar's, or None.
    Farneback optical flow on dB fields at ~8 km, smoothed, clipped to max_km."""
    h, w = model0.shape
    k = max(1, int(round(coarse_km / km_per_px)))
    small = lambda g: cv2.resize(np.where(mask, g, 0).astype(np.float32), (w // k, h // k), interpolation=cv2.INTER_AREA)
    a, b = _db(small(model0)), _db(small(radar0))
    to8 = lambda g: np.clip((g + 10) * 4, 0, 255).astype(np.uint8)
    flow = cv2.calcOpticalFlowFarneback(to8(a), to8(b), None, 0.5, 4, 9, 5, 7, 1.5, 0)
    flow = cv2.GaussianBlur(flow, (0, 0), 3)
    mag = np.hypot(flow[..., 0], flow[..., 1]) * k * km_per_px
    scale = np.minimum(1.0, max_km / np.maximum(mag, 1e-6))
    u = cv2.resize(flow[..., 0] * scale, (w, h), interpolation=cv2.INTER_LINEAR) * k
    v = cv2.resize(flow[..., 1] * scale, (w, h), interpolation=cv2.INTER_LINEAR) * k
    return u.astype(np.float32), v.astype(np.float32)


def warp(g, flow):
    """g moved along flow (backward sampling: out(x) = g(x - d))."""
    u, v = flow
    h, w = g.shape
    gx, gy = np.meshgrid(np.arange(w, dtype=np.float32), np.arange(h, dtype=np.float32))
    return cv2.remap(g.astype(np.float32), gx - u, gy - v, cv2.INTER_LINEAR, borderMode=cv2.BORDER_CONSTANT, borderValue=0)


def fss(f, o, mask, th, n):
    """Fractions skill score parts (numerator, denominator) at threshold th, window n x n pixels."""
    m = mask.astype(np.float32)
    pf = cv2.blur(((f >= th) & mask).astype(np.float32), (n, n))
    po = cv2.blur(((o >= th) & mask).astype(np.float32), (n, n))
    mm = cv2.blur(m, (n, n)) > 0.5
    return float(((pf - po) ** 2)[mm].sum()), float((pf ** 2 + po ** 2)[mm].sum())


def ramp(lead, start, end, wmax):
    """0 until `start` min, linear to wmax at `end` min, flat after."""
    if lead <= start:
        return 0.0
    return wmax * min(1.0, (lead - start) / max(1e-6, end - start))


# the shadow's settings (tune_blend.py, 16.11.2026, same past runs for every variant, models as old as live (Europe
# ~65 min, HRRR ~105 min); CSI cell by cell >= 0.1 / >= 1 mm/h, +90 / +120 min, served -> variant):
#   Europe, 40 runs: linear blend from +60 to w 0.3 at +120 with the per-model skill factor plus the model's growth /
#     decay ("trend30+s"): 0.502/0.457 -> 0.504/0.464, 0.303/0.252 -> 0.315/0.269, FSS 20 km >= 1 at +120 0.543 ->
#     0.574, amount x0.94/0.93 -> 1.01/1.06; +30/+60 unchanged.
#   North America, 30 runs: any linear blend lowers CSI >= 0.1 (best 0.482/0.432 -> 0.466/0.415) while it raises
#     >= 1 (0.354/0.307 -> 0.360/0.318): HRRR's drizzle is placed worse. Growth / decay alone ("trend30h") is neutral
#     to slightly better: 0.483/0.432, 0.355/0.311, amount 0.96/1.02 (served 0.90/0.90).
DEFAULT = {"start": 60, "end": 120, "wmax": 0.3, "cdf": True, "shift": False, "pm": True, "skill": True,
           "trend": 30, "tstart": 30, "tend": 120, "twmax": 0.5}
REGION_DEFAULT = {"eu": DEFAULT, "na": dict(DEFAULT, wmax=0.0)}


def prepare(runs, ref, last, opts=DEFAULT, km_per_px=4.0):
    """The corrected, combined model for this run: (model(t) -> grid or None, edge weight E, diagnostics).
    runs: [(RunField, Mapper)] - the newest usable run per model; last: the radar analysis at ref (grid, < 0 = none)."""
    cov = last >= 0
    a0 = np.where(cov, last, 0).astype(np.float32)
    parts, diag = [], {}
    for rf, mp in runs:
        if rf is None or ref - rf.init > MAX_AGE_MS or rf.avail > ref:
            continue
        m0 = rf.at(ref)
        if m0 is None:
            continue
        mask = cov & (rf.edge > 0.5)
        if mask.sum() < 1000:
            continue
        qm = quantile_map(m0, a0, mask) if opts.get("cdf", True) else None
        corr = qm if qm is not None else (lambda x: np.asarray(x, np.float32))
        m0c = corr(m0)
        flow = position_flow(m0c, a0, mask, km_per_px) if opts.get("shift") else None
        n = max(1, int(round(20 / km_per_px)))
        num, den = fss(m0c if flow is None else warp(m0c, flow), a0, mask, 1.0, n)
        f0 = 1 - num / den if den > 0 else 0.0
        diag[rf.model] = {"age_min": round((ref - rf.init) / 60_000), "fss0": round(f0, 3),
                          "amount0": round(float(m0[mask].sum() / max(1e-6, a0[mask].sum())), 2)}
        parts.append((rf, mp, corr, flow, f0))
    if not parts:
        return None, None, diag
    E = np.max([rf.edge for rf, *_ in parts], axis=0)

    memo = {}

    def model_at(t):
        if t in memo:  # the field at ref is asked for at every step (trend)
            return memo[t]
        num = np.zeros_like(E)
        den = np.zeros_like(E)
        for rf, mp, corr, flow, f0 in parts:
            m = rf.at(t)
            if m is None:
                continue
            m = corr(m)
            if flow is not None:
                m = warp(m, flow)
            wgt = rf.edge * (max(0.0, min(1.0, (f0 - 0.2) / 0.4)) if opts.get("skill") else 1.0)
            num += wgt * m
            den += wgt
        out = np.where(den > 0, num / np.maximum(den, 1e-6), 0).astype(np.float32)
        if t == ref:
            memo[t] = out
        return out
    if opts.get("skill"):  # pixels where every model scored badly at ref get no model weight
        E = np.max([rf.edge * max(0.0, min(1.0, (f0 - 0.2) / 0.4)) for rf, _, _, _, f0 in parts], axis=0)
    return model_at, E, diag


def pm_mix(b, f, m, w, mask):
    """b's values replaced, rank by rank on mask, by the w-mix of f's and m's distributions (STEPS-style probability
    matching of a linear blend: the blend keeps where the rain is, the amount and the wet area are the mix)."""
    if mask.sum() < 100:
        return b
    sf, sm = np.sort(f[mask]), np.sort(m[mask])
    target = (1 - w) * sf + w * sm
    order = np.argsort(b[mask], kind="stable")
    out = b.copy()
    vals = np.empty_like(target)
    vals[order] = target
    out[mask] = vals
    return out


def blend(fc, ref, last, model_at, E, opts=DEFAULT, step_ms=300_000):
    """fc: the served steps on the verification grid (5 min apart, < 0 = no radar). Returns the blended steps
    (float32 grids, < 0 kept)."""
    return [blend_step(g, (i + 1) * step_ms / 60_000, ref + (i + 1) * step_ms, model_at, E, opts)
            for i, g in enumerate(fc)]


def blend_step(g, lead, t, model_at, E, opts=DEFAULT):
    """One step: g = the served forecast for t (lead minutes after ref). Two optional parts:
    trend (opts "trend" = window sigma in km): the model's growth / decay - the served forecast times the model's
      smoothed rain at t / at ref (clipped 1/3..3), to the power ramp(tstart, tend, twmax) * edge weight; the model's
      rain does not have to be in the right place for this, only in the right region;
    linear blend (wmax > 0): (1 - w) * forecast + w * model, then probability-matched to the w-mix (pm)."""
    g = np.asarray(g, np.float32)
    if model_at is None:
        return g
    w = ramp(lead, opts["start"], opts["end"], opts["wmax"])
    a = ramp(lead, opts.get("tstart", 30), opts.get("tend", 120), opts.get("twmax", 1.0)) if opts.get("trend") else 0.0
    if w <= 0 and a <= 0:
        return g
    cov = g >= 0
    f = np.where(cov, g, 0)
    m = model_at(t)
    if a > 0:
        k = opts["trend"] / opts.get("km", 4.0)
        blur = lambda x: cv2.GaussianBlur(x, (0, 0), k)
        ratio = np.clip((blur(m) + 0.2) / (blur(model_at(t - int(round(lead * 60_000)))) + 0.2), 1 / 3, 3)
        f = f * ratio ** (a * E)
    b = f
    if w > 0:
        W = w * E
        b = (1 - W) * f + W * m
        if opts.get("pm", True):
            mask = cov & (E > 0)
            pmd = pm_mix(b, f, m, w, mask)
            b = np.where(mask, E * pmd + (1 - E) * f, b)  # full matching inside, fading to the plain forecast at the edge
    b = np.array(b, np.float32)
    b[b < 0.05] = 0
    b[~cov] = -1
    return b


# ---------- live: the newest runs for a region (background refresh) ----------
class NwpLive:
    """Keeps the newest run of every model of a region on the verification grid (region grid / down), refreshed every
    5 min (new runs come about hourly). shadow(fc_small, last_small, ref) -> {valid ms: float16 grid}: the scored
    shadow "nwp". lock: held while a model grid's mapping is built (the instance's S-PROG lock: never both at once).
    Memory: ~0.15 GB held (mappings + two runs of 20 steps at ~4 km), ~0.3 GB once at the start."""

    def __init__(self, region, region_key, down, log=print, opts=None, lock=None):
        self.region, self.key, self.down, self.log = region, region_key, down, log
        self.km = region.KM_PER_PX * down
        self.opts = dict(REGION_DEFAULT[region_key], km=self.km, **(opts or {}))
        self.lock = lock or threading.Lock()
        self.mappers, self.runs, self.shape, self.diag = {}, {}, None, {}

    def refresh(self):
        if self.shape is None:
            g = self.region.latest()
            if g is None:
                return
            self.shape = (g.shape[0] // self.down, g.shape[1] // self.down)
        now = int(time.time() * 1000)
        for m in REGION_MODELS[self.key]:
            try:
                if m not in self.mappers:
                    with self.lock:  # once: ~0.3 GB for a moment (AROME has 5 million points)
                        self.mappers[m] = Mapper(MODELS[m]["grid"], self.region, self.down, self.shape, self.km)
                old = self.runs.get(m)
                # not under the lock: a slow download must never hold up the forecast (~0.1 GB while mapping)
                rf = load_latest(m, self.mappers[m], now - 30 * 60_000, now + 4 * 3600_000, getattr(old, "mod", None))
                if rf is not None:
                    self.runs[m] = rf
                    if old is None or rf.init != old.init:
                        self.log(f"nwp {m}: run {datetime.fromtimestamp(rf.init / 1000, timezone.utc):%d.%m. %H:%MZ}, "
                                 f"{len(rf.times)} steps")
            except Exception as e:
                self.log(f"nwp {m} refresh failed: {e}")

    def loop(self):
        while True:
            self.refresh()
            time.sleep(300)

    def shadow(self, fc_small, last_small, ref):
        runs = [(self.runs.get(m), self.mappers.get(m)) for m in REGION_MODELS[self.key]]
        runs = [(rf, mp) for rf, mp in runs if rf is not None and mp is not None]
        if not runs or np.shape(last_small) != self.shape:
            return None
        last = np.asarray(last_small, np.float32)
        model_at, E, diag = prepare(runs, ref, last, self.opts, self.km)
        self.diag = diag
        if model_at is None:
            return None
        out = blend(fc_small, ref, last, model_at, E, self.opts)
        return {ref + (i + 1) * 300_000: g.astype(np.float16) for i, g in enumerate(out)}
