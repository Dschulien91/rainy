"""Offline test of the sunny nowcast beyond +60 min (sunny_de.py): weather-model blend, motion field, model trend.

Input: archived DWD SIDhr files (as server/replay_sun.py: --cache), the DWD 10-minute station sunshine
(verify_log/truth_sun_<day>.json: --truth) and Open-Meteo's archive of past weather-model runs (data_run/, read as .om
files with `omfiles`, anonymous; --nwp is the local copy, --fetch downloads what is missing). Every model run is used
only from the time it became available (meta.json created_at), so the model is exactly as old as it would have been.

Per run (every satellite slot with an image 10-40 min before it) the served forecast "sat" is rebuilt exactly like
SunnyDE.forecast (Farneback flow, blur ramp, k -> share), plus variants:
  pers                 the analysis stays
  nwp / nwpqm          the model's sun share alone: raw / quantile-mapped (k space) onto the satellite at ref
  b<w> / bk<w> / braw<w>  constant weight w at every lead: (1 - w) * sat + w * nwpqm in share space / in k space /
                       with the raw model (group "blend": choose the weight per lead)
  s_ / sb_ / sk_<start>_<w120>  lead schedules (0 until start, linear to w120 at +120) over sat / over shblur / in k
  s55sk<n>             schedule 30/0.55 scaled down when the corrected model's share MAE at ref was high (group "sched")
  flow variants        f_s5 / f_s20 (flow smoothing), f_d2 (finer flow grid), f_dis (DIS flow), f_avg (mean of the
                       20/30/40 min pairs), f_sl (semi-Lagrangian backward trajectories in 10 min steps), f_60 (60 min pair)
  tr<a>                Lagrangian model trend: k + a * (model k at t - model k at the origin at ref), qm'd model
Scores (daytime, sun >= 5 deg at the valid time), like SunVerify: grid against the later analysis (CSI with ~9 km
tolerance at share >= 0.5, MAE, bias, sunny area fc/obs) and the DWD stations (MAE, hit rate of "sunny or not", bias).

  python -I tune_sun.py --cache DIR --truth DIR --nwp DIR [--fetch] [--days 20261115,20261116] [--model dwd_icon_d2_15min]
         [--groups base,blend,flow,trend,sched] [--coarse 5] [--region na]
  (models: dwd_icon_d2_15min, dwd_icon_d2_ruc_native_15min, dwd_icon_eu; --region na: GOES-East ACMF files in --cache,
   CONUS grid, model ncep_hrrr_conus_15min with shortwave/diffuse_radiation.om, grid scores only; needs omfiles and
   eccodes (ICON-D2-RUC grid); NWP_CACHE_DIR for nwp_blend's grid cache; ~1 GB RAM for 3 days in Germany)
"""
import argparse
import glob
import json
import math
import os
import re
import sys
import time
import urllib.request
from collections import defaultdict
from datetime import datetime, timezone

import cv2
import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import sunny_de as sd  # noqa: E402
from clouds import sun_elevation  # noqa: E402
from mosaic import merc  # noqa: E402

S3 = "https://openmeteo.s3.amazonaws.com"
UA = {"User-Agent": "Mozilla/5.0"}
GRIDS = {"dwd_icon_d2_15min": ("latlon", 43.18, -3.94, 0.02, 746, 1215),
         "dwd_icon_d2_ruc_native_15min": ("icon_d2",),
         "dwd_icon_eu": ("latlon", 29.5, -23.5, 0.0625, 657, 1377),  # hourly (the Europe instance's model)
         "ncep_hrrr_conus_15min": "hrrr"}  # direct = shortwave - diffuse (no direct_radiation file); grid from nwp_blend
LEADS = (10, 30, 60, 90, 120)
WEIGHTS = (0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.8)
SAT_LAG_MS = 7 * 60_000  # the SID slot S scans Germany ~S+5..S+9
# lead-dependent model weights: 0 until `start` min, linear to wmax at +120 (name: (start, wmax))
SCHEDULES = {"30_45": (30, 0.45), "30_55": (30, 0.55), "45_45": (45, 0.45), "60_45": (60, 0.45), "30_35": (30, 0.35)}


SKILL = {"sk1": (0.10, 0.10), "sk2": (0.15, 0.10), "sk3": (0.08, 0.06)}  # weight x clip(1 - (mae0 - a) / b)
MAE0 = []


def ramp(lead, start, end, wmax):
    return 0.0 if lead <= start else wmax * min(1.0, (lead - start) / (end - start))


def _ms(s):
    s = s.rstrip("Z")
    fmt = "%Y-%m-%dT%H:%M:%S" if s.count(":") == 2 else "%Y-%m-%dT%H:%M"
    return int(datetime.strptime(s, fmt).replace(tzinfo=timezone.utc).timestamp() * 1000)


def _get(url, timeout=600):
    with urllib.request.urlopen(urllib.request.Request(url, headers=UA), timeout=timeout) as r:
        return r.read()


def fetch_runs(root, model, days):
    """Download meta.json + direct_radiation.om of the archived runs 00-12Z of the given days (missing ones only)."""
    for day in days:
        d = datetime.strptime(day, "%Y%m%d")
        xml = _get(f"{S3}/?list-type=2&prefix=data_run/{model}/{d:%Y/%m/%d}/&delimiter=/").decode()
        for p, hhmm in re.findall(r"<Prefix>(data_run/[^<]+/(\d{4})Z/)</Prefix>", xml):
            if int(hhmm[:2]) > 12:
                continue
            out = os.path.join(root, model, day[6:], hhmm)
            os.makedirs(out, exist_ok=True)
            for f in ("meta.json", "direct_radiation.om"):
                fp = os.path.join(out, f)
                if not os.path.exists(fp):
                    data = _get(f"{S3}/{p}{f}")
                    with open(fp + ".tmp", "wb") as fh:
                        fh.write(data)
                    os.replace(fp + ".tmp", fp)


quantile_map = sd.quantile_map  # the live shadow's mapping (sunny_de.SunnyDE._nwp_qm)


class ModelRuns:
    """Archived runs of one model as clear-sky index fields on the sunny grid (only the steps a test needs)."""

    def __init__(self, root, model, grid, t0, t1, coarse=1):
        """coarse > 1: the model sampled every `coarse` pixels and interpolated back (as the live SunModel's point
        grid: 0.1 deg in Germany = ~4-6 pixels, 0.25 deg in Europe)."""
        import nwp_blend as nb
        from omfiles import OmFileReader
        self.runs = []  # (avail ms, init ms, centres ms (T,), k fields (T, ny, nx) float16 NaN = none)
        g = GRIDS[model]
        mapper = nb.Mapper(nb.MODELS[g]["grid"] if isinstance(g, str) else g, grid, 1, (grid.ny, grid.nx), grid.cell / 1000.0)
        for mp in sorted(glob.glob(os.path.join(root, model, "*", "*", "meta.json"))):
            meta = json.load(open(mp))
            om = os.path.join(os.path.dirname(mp), "direct_radiation.om")
            sw, df = (os.path.join(os.path.dirname(mp), f"{v}_radiation.om") for v in ("shortwave", "diffuse"))
            if not os.path.exists(om) and not (os.path.exists(sw) and os.path.exists(df)):
                continue
            init, avail = _ms(meta["reference_time"]), _ms(meta["created_at"])
            if avail > t1 or init + 8 * 3600_000 < t0:
                continue
            ends = np.array([_ms(v) for v in meta["valid_times"]], np.int64)
            centres = ends - int(meta.get("temporal_resolution_seconds", 900)) * 500  # backward means
            rd = OmFileReader(om if os.path.exists(om) else sw)
            off = len(ends) - rd.shape[-1]
            # the steps a forecast from this run can need: from its availability to 3.5 h + 2 h later
            idx = [k for k in range(max(1, off), len(ends))
                   if avail - 30 * 60_000 <= centres[k] <= avail + 6 * 3600_000 and t0 - 3600_000 <= centres[k] <= t1 + 3 * 3600_000]
            if len(idx) < 2:
                rd.close()
                continue
            fields, edge = mapper.read(rd, [k - off for k in idx])
            rd.close()
            if not os.path.exists(om):  # direct = global - diffuse
                rd = OmFileReader(df)
                dif, _ = mapper.read(rd, [k - off for k in idx])
                rd.close()
                fields = np.maximum(fields.astype(np.float32) - dif.astype(np.float32), 0).astype(np.float16)
                del dif
            ks = np.empty(fields.shape, np.float16)
            for j, k in enumerate(idx):
                e = sun_elevation(int(centres[k]), grid.lat, grid.lon).astype(np.float32)
                dni = fields[j].astype(np.float32) / np.sin(np.radians(np.maximum(e, 1.0)))
                kk = np.clip(dni / sd.clear_dni(e), 0, sd.K_MAX)
                if coarse > 1:
                    small = sd.fill_nan(np.where((e >= sd.SUN_ELEV) & (edge > 0.5), kk, np.nan).astype(np.float32))[::coarse, ::coarse]
                    kk = cv2.resize(small, (grid.nx, grid.ny), interpolation=cv2.INTER_LINEAR)
                ks[j] = np.where((e >= sd.SUN_ELEV) & (edge > 0.5), kk, np.nan)
            self.runs.append((avail, init, centres[idx], ks))
            print(f"model {model} run {datetime.fromtimestamp(init / 1000, timezone.utc):%d.%m. %H:%MZ} "
                  f"(available {datetime.fromtimestamp(avail / 1000, timezone.utc):%H:%MZ}): {len(idx)} steps", flush=True)
        self.runs.sort(key=lambda r: r[0])

    def run_for(self, ref):
        ok = [r for r in self.runs if r[0] <= ref]
        return ok[-1] if ok else None

    @staticmethod
    def k_at(run, t):
        _, _, c, ks = run
        if not c[0] <= t <= c[-1]:
            return None
        j = min(int(np.searchsorted(c, t, side="right")) - 1, len(c) - 2)
        f = (t - c[j]) / (c[j + 1] - c[j])
        return ((1 - f) * ks[j].astype(np.float32) + f * ks[j + 1].astype(np.float32))


# ---------- motion ----------
def to_img(k):
    return (np.clip(1.0 - sd.fill_nan(k), 0, 1) * 255).astype(np.uint8)


def flow_field(grid, k0, k1, dt_min, down=sd.FLOW_DOWN, sigma=sd.FLOW_SIGMA, method="farneback"):
    s = (grid.nx // down, grid.ny // down)
    a = cv2.resize(to_img(k0), s, interpolation=cv2.INTER_AREA)
    b = cv2.resize(to_img(k1), s, interpolation=cv2.INTER_AREA)
    if method == "dis":
        dis = cv2.DISOpticalFlow_create(cv2.DISOPTICAL_FLOW_PRESET_MEDIUM)
        flow = dis.calc(a, b, None)
    else:
        flow = cv2.calcOpticalFlowFarneback(a, b, None, 0.5, 4, 15, 4, 7, 1.5, 0)
    flow = cv2.GaussianBlur(flow, (0, 0), sigma)
    return cv2.resize(flow, (grid.nx, grid.ny), interpolation=cv2.INTER_LINEAR) * (down / dt_min)


def origins(grid, flow, lead, semi_lagrangian=False):
    """Where the air at each pixel was `lead` minutes earlier (backward trajectories)."""
    if not semi_lagrangian:
        return grid.gx - flow[..., 0] * lead, grid.gy - flow[..., 1] * lead
    x, y = grid.gx.copy(), grid.gy.copy()
    n = max(1, int(round(lead / 10)))
    dt = lead / n
    fu, fv = flow[..., 0], flow[..., 1]
    for _ in range(n):
        u = cv2.remap(fu, x, y, cv2.INTER_LINEAR, borderMode=cv2.BORDER_REPLICATE)
        v = cv2.remap(fv, x, y, cv2.INTER_LINEAR, borderMode=cv2.BORDER_REPLICATE)
        x, y = x - u * dt, y - v * dt
    return x, y


def advect(grid, base, known, flow, lead, e, semi_lagrangian=False, blur="k"):
    """The served forecast's steps (SunnyDE.forecast): k moved, blurred with the lead (ramp to SHARP_MIN), -> share."""
    mx, my = origins(grid, flow, lead, semi_lagrangian)
    kk = cv2.remap(base, mx, my, cv2.INTER_LINEAR, borderMode=cv2.BORDER_REPLICATE)
    kn = cv2.remap(known, mx, my, cv2.INTER_LINEAR, borderMode=cv2.BORDER_REPLICATE)
    sig = max(0.3, sd.BLUR * math.sqrt(lead / 10) * min(1.0, lead / sd.SHARP_MIN))
    if blur == "k":
        kk = cv2.GaussianBlur(kk, (0, 0), sig)
        kk = np.where(kn > 0.5, kk, np.nan).astype(np.float32)
        return kk, sd.share_from_k(kk, e), (mx, my)
    sh = sd.share_from_k(np.where(kn > 0.5, kk, np.nan).astype(np.float32), e)
    return kk, np.where(np.isfinite(sh), cv2.GaussianBlur(sd.fill_nan(sh), (0, 0), sig), np.nan).astype(np.float32), (mx, my)


# ---------- scoring ----------
class Scores:
    def __init__(self, grid, stations, pix):
        self.grid, self.st, self.pix = grid, stations, pix
        self.shape = (int(round(grid.nx / sd.VERIFY_DOWN)), int(round(grid.ny / sd.VERIFY_DOWN)))
        self.g = defaultdict(lambda: np.zeros(7))   # (variant, lead) -> h, m, f, sum|d|, sum d, n, fcn/obs ratio parts
        self.go = defaultdict(float)                 # (variant, lead) -> observed sunny pixels
        self.s = defaultdict(lambda: np.zeros(4))   # (variant, lead) -> sum|d|, hits, sum d, n
        self.day = defaultdict(lambda: np.zeros(4))  # (variant, lead, day) -> station sum|d|, hits, n, grid sum|d| / n
        self.truth_cache = {}

    def _down(self, a):
        ok = np.isfinite(a)
        v = cv2.resize(np.where(ok, a, 0).astype(np.float32), self.shape, interpolation=cv2.INTER_AREA)
        w = cv2.resize(ok.astype(np.float32), self.shape, interpolation=cv2.INTER_AREA)
        return np.where(w > 0.5, v / np.maximum(w, 1e-6), np.nan).astype(np.float32)

    def prepare(self, valid, ana, area):
        """Everything about the truth at one valid time (shared by all variants)."""
        a = self._down(ana)
        e = self._down(self.grid.elev(valid))
        mask = (e >= sd.SCORE_ELEV) & np.isfinite(a) & (self._down(area) > 0.5)
        truth = self.st.truth(valid + sd.STATION_LAG_MS)
        keys = [s for s in truth if s in self.pix and sun_elevation(valid, *self.st.stations[s]) >= sd.SCORE_ELEV]
        return a, mask, truth, keys

    def add(self, name, lead, fc, prep, day):
        a, mask, truth, keys = prep
        g = self._down(fc)
        m = mask & np.isfinite(g)
        n = int(m.sum())
        if n >= sd.MIN_PX:
            fcb, obb = (g >= sd.SUNNY), (a >= sd.SUNNY)
            k = np.ones((2 * sd.TOL_PX + 1,) * 2, np.uint8)
            f_, o_ = fcb & m, obb & m
            near_o = cv2.dilate(o_.astype(np.uint8), k) > 0
            near_f = cv2.dilate(f_.astype(np.uint8), k) > 0
            d = g[m] - a[m]
            self.g[(name, lead)] += [int((f_ & near_o).sum()), int((o_ & ~near_f).sum()), int((f_ & ~near_o).sum()),
                                     float(np.abs(d).sum()), float(d.sum()), n, int(f_.sum())]
            self.go[(name, lead)] += int(o_.sum())
            self.day[(name, lead, day)][3] += float(np.abs(d).mean())
        if len(keys) >= 5:
            fv = np.array([self._at(fc, self.pix[s]) for s in keys], np.float32)
            ov = np.array([truth[s] for s in keys], np.float32)
            ok = np.isfinite(fv)
            if ok.sum() >= 5:
                fv, ov = fv[ok], ov[ok]
                hits = float(((fv >= sd.SUNNY) == (ov >= sd.SUNNY)).sum())
                self.s[(name, lead)] += [float(np.abs(fv - ov).sum()), hits, float((fv - ov).sum()), len(fv)]
                self.day[(name, lead, day)][:3] += [float(np.abs(fv - ov).sum()), hits, len(fv)]

    @staticmethod
    def _at(g, rc):
        r, c = rc
        v = g[r - 1:r + 2, c - 1:c + 2]
        return float(np.nanmean(v)) if np.isfinite(v).any() else float("nan")

    def report(self, names, leads=LEADS):
        print("variant          lead   CSI   POD   FAR  bias   MAE  fc/obs |  st MAE  hit   bias      n")
        for name in names:
            for lead in leads:
                g, s = self.g.get((name, lead)), self.s.get((name, lead))
                if g is None and s is None:
                    continue
                line = f"{name:15s} +{lead:3d}"
                if g is not None:
                    h, m, f, ad, sdv, n, fcn = g
                    line += (f"  {h / max(h + m + f, 1):.2f}  {h / max(h + m, 1):.2f}  {f / max(h + f, 1):.2f}"
                             f" {sdv / max(n, 1):+.2f}  {ad / max(n, 1):.3f} {fcn / max(self.go[(name, lead)], 1):.2f}")
                else:
                    line += " " * 40
                if s is not None:
                    ad, hits, sdv, n = s
                    line += f" | {ad / n:.3f} {hits / n:.3f} {sdv / n:+.3f} {int(n):6d}"
                print(line, flush=True)


# ---------- main ----------
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--cache", required=True)
    ap.add_argument("--truth", required=True)
    ap.add_argument("--nwp", required=True)
    ap.add_argument("--model", default="dwd_icon_d2_15min")
    ap.add_argument("--fetch", action="store_true")
    ap.add_argument("--days", default="")
    ap.add_argument("--groups", default="base,blend,flow,trend")
    ap.add_argument("--every", type=int, default=1, help="use every n-th run")
    ap.add_argument("--coarse", type=int, default=1, help="model sampled every n pixels (live point grid)")
    ap.add_argument("--region", default="de", help="de (DWD SID, stations) or na (GOES-East clear sky mask in --cache, "
                                                    "CONUS grid, no station truth)")
    a = ap.parse_args()
    groups = set(a.groups.split(","))
    na = a.region == "na"
    if na:
        goes = {}
        for f in glob.glob(os.path.join(a.cache, "OR_ABI-L2-ACMF*.nc")):
            m = re.search(r"_s(\d{4})(\d{3})(\d{2})(\d{2})", os.path.basename(f))
            t = datetime.strptime("".join(m.groups()), "%Y%j%H%M").replace(tzinfo=timezone.utc)
            goes[int(t.timestamp() * 1000) // sd.STEP_MS * sd.STEP_MS] = f
        slots = sorted(goes)
    else:
        files = sorted(glob.glob(os.path.join(a.cache, "SIDhr*EUv4.nc.bz2")))
        slots = sorted(sd.stamp_ts(os.path.basename(f)[5:17]) for f in files)
    if a.days:
        slots = [t for t in slots if sd.ts_stamp(t)[:8] in a.days.split(",")]
    days = sorted({sd.ts_stamp(t)[:8] for t in slots})
    print(f"{len(slots)} slots on {', '.join(days)}", flush=True)
    if a.fetch:
        fetch_runs(a.nwp, a.model, days)
    if na:
        import sunny_na

        class NoStations:
            stations = {}

            def truth(self, end):
                return {}
        st = NoStations()
        x, y = merc(np.array([-125.0, -67.0]), np.array([24.0, 50.0]))
        grid = sd.SunGrid((float(x[0]), float(y[0]), float(x[1]), float(y[1])), sunny_na.AMERICA["cell"])
        goes_reader = sunny_na.SunnyGoes.__new__(sunny_na.SunnyGoes)  # only its cloud_probability()
        goes_reader.grid, goes_reader._idx = grid, None
    else:
        from replay_sun import ReplayStations
        st = ReplayStations(a.truth, days)
        grid = sd.SunGrid()
    pix = {}
    for sid, (lat, lon) in st.stations.items():
        col, row = (int(round(float(v))) for v in grid.to_pixels(*merc(lon, lat)))
        if 1 <= col < grid.nx - 1 and 1 <= row < grid.ny - 1:
            pix[sid] = (row, col)
    models = ModelRuns(a.nwp, a.model, grid, slots[0], slots[-1], a.coarse) if groups & {"blend", "trend", "base", "sched"} else None
    # all analyses: k and share per slot (implausible jumps dropped as in SunnyDE.add_slot)
    K, SH = {}, {}
    prev = None
    for t in slots:
        e = grid.elev(t)
        if na:  # as sunny_na.SunnyGoes.ingest
            p = goes_reader.cloud_probability(open(goes[t], "rb").read())
            k = np.where(np.isfinite(p), 1.0 - 0.95 * np.clip(p, 0, 1), np.nan).astype(np.float32)
        else:
            lat, lon, v = sd.read_nc(open(os.path.join(a.cache, f"SIDhr{sd.ts_stamp(t)}EUv4.nc.bz2"), "rb").read(), ["DNI"])
            dni = grid.regrid(lat, lon, v["DNI"], sigma=(0.8, 0.5))
            k = np.where(e >= sd.SUN_ELEV, np.clip(dni / sd.clear_dni(e), 0, sd.K_MAX), np.nan).astype(np.float32)
        s = sd.share_from_k(k, e)
        if prev is not None and t - prev <= 2 * sd.STEP_MS:
            m = (e >= sd.SCORE_ELEV) & np.isfinite(s) & np.isfinite(SH[prev].astype(np.float32))
            if m.sum() >= 500 and np.abs(s[m] - SH[prev].astype(np.float32)[m]).mean() > sd.JUMP_MAX:
                print(f"slot {sd.ts_stamp(t)} skipped (jump)")
                continue
        K[t], SH[t] = k.astype(np.float16), s.astype(np.float16)
        prev = t
    print(f"{len(K)} analyses read", flush=True)
    area = np.ones((grid.ny, grid.nx), np.float32)
    if models is not None:  # score only where the model exists (all variants on the same pixels)
        cover = np.zeros_like(area)
        for r in models.runs:
            cover = np.maximum(cover, np.isfinite(r[3]).any(axis=0).astype(np.float32))
        area = cover
    sc = Scores(grid, st, {s: rc for s, rc in pix.items() if area[rc] > 0.5})
    refs = [t for t in sorted(K) if any(t - m * 60_000 in K for m in (30, 20, 40, 10))][::a.every]
    started = time.time()
    for i, ref in enumerate(refs):
        day = sd.ts_stamp(ref)[:8]
        k1 = K[ref].astype(np.float32)
        pm = next(m for m in (30, 20, 40, 10) if ref - m * 60_000 in K)
        k0 = K[ref - pm * 60_000].astype(np.float32)
        valid_leads = [L for L in LEADS if ref + L * 60_000 in K]
        if not valid_leads:
            continue
        base, known = sd.fill_nan(k1), np.isfinite(k1).astype(np.float32)
        flows = {"sat": (flow_field(grid, k0, k1, pm), False)}
        if "flow" in groups:
            flows["f_s5"] = (flow_field(grid, k0, k1, pm, sigma=5.0), False)
            flows["f_s20"] = (flow_field(grid, k0, k1, pm, sigma=20.0), False)
            flows["f_d2"] = (flow_field(grid, k0, k1, pm, down=2, sigma=15.0), False)
            flows["f_dis"] = (flow_field(grid, k0, k1, pm, method="dis"), False)
            flows["f_sl"] = (flows["sat"][0], True)
            pairs = [(m, K[ref - m * 60_000].astype(np.float32)) for m in (20, 30, 40) if ref - m * 60_000 in K]
            if len(pairs) >= 2:
                flows["f_avg"] = (np.mean([flow_field(grid, kp, k1, m) for m, kp in pairs], axis=0), False)
            if ref - 3600_000 in K:
                flows["f_60"] = (flow_field(grid, K[ref - 3600_000].astype(np.float32), k1, 60), False)
        run = models.run_for(ref) if models is not None else None
        m0 = models.k_at(run, ref + SAT_LAG_MS) if run is not None else None
        qm = None
        if m0 is not None:
            e0 = grid.elev(ref)
            msk = np.isfinite(m0) & np.isfinite(k1) & (e0 >= sd.DAY_ELEV)
            qm = quantile_map(m0[msk], k1[msk])
        mae0 = None
        if qm is not None:  # the corrected model's error against the analysis at ref (its skill today)
            s0 = sd.share_from_k(qm(m0), grid.elev(ref))
            sel = np.isfinite(s0) & np.isfinite(SH[ref].astype(np.float32)) & (grid.elev(ref) >= sd.SCORE_ELEV)
            mae0 = float(np.abs(s0[sel] - SH[ref].astype(np.float32)[sel]).mean()) if sel.sum() > 500 else None
            MAE0.append(mae0 if mae0 is not None else np.nan)
        if models is not None and qm is None:
            continue  # every variant on the same cases: runs without a usable model are left out for all
        for L in valid_leads:
            ts = ref + L * 60_000
            e = grid.elev(ts)
            m1 = models.k_at(run, ts + SAT_LAG_MS) if run is not None else None
            if models is not None and m1 is None:
                continue
            prep = sc.prepare(ts, SH[ts].astype(np.float32), area)
            sc.add("pers", L, SH[ref].astype(np.float32), prep, day)
            kk_sat, s_sat, (mx, my) = advect(grid, base, known, flows["sat"][0], L, e)
            sc.add("sat", L, s_sat, prep, day)
            if "base" in groups:
                sc.add("shblur", L, advect(grid, base, known, flows["sat"][0], L, e, blur="share")[1], prep, day)
                sc.add("kpers", L, sd.share_from_k(k1, e), prep, day)
            for name, (fl, slg) in flows.items():
                if name != "sat":
                    sc.add(name, L, advect(grid, base, known, fl, L, e, semi_lagrangian=slg)[1], prep, day)
            if m1 is None or qm is None:
                continue
            age = (ref - run[1]) / 3600_000
            m1q = qm(m1)
            s_nwp = sd.share_from_k(m1, e)
            s_q = sd.share_from_k(m1q, e)
            fill = lambda x, y: np.where(np.isfinite(x), x, y)
            sc.add("nwp", L, fill(s_nwp, s_sat), prep, day)
            sc.add("nwpqm", L, fill(s_q, s_sat), prep, day)
            sc.add(f"nwpqm_age{int(age)}", L, fill(s_q, s_sat), prep, day)
            if "blend" in groups or "sched" in groups:
                ok = np.isfinite(s_q) & np.isfinite(s_sat)
                s_sb = advect(grid, base, known, flows["sat"][0], L, e, blur="share")[1]
                for sname, (a0, b0) in SKILL.items():  # 30/0.55 scaled down when the model was bad at ref
                    f = 0.0 if mae0 is None else float(np.clip(1 - (mae0 - a0) / b0, 0, 1))
                    w = ramp(L, 30, 120, 0.55) * f
                    sc.add(f"s55{sname}", L, np.where(ok, (1 - w) * s_sat + w * s_q, s_sat), prep, day)
                for sname, (start, wmax) in SCHEDULES.items():
                    w = ramp(L, start, 120, wmax)
                    sc.add(f"s_{sname}", L, np.where(ok, (1 - w) * s_sat + w * s_q, s_sat), prep, day)
                    sc.add(f"sb_{sname}", L, np.where(ok, (1 - w) * s_sb + w * s_q, s_sb), prep, day)
                    kb = np.where(ok, (1 - w) * kk_sat + w * m1q, kk_sat).astype(np.float32)
                    sc.add(f"sk_{sname}", L, sd.share_from_k(kb, e), prep, day)
            if "blend" in groups:
                for w in WEIGHTS:
                    sc.add(f"b{w:.1f}", L, np.where(ok, (1 - w) * s_sat + w * s_q, s_sat), prep, day)
                    kb = np.where(ok, (1 - w) * kk_sat + w * m1q, kk_sat)
                    sc.add(f"bk{w:.1f}", L, sd.share_from_k(kb.astype(np.float32), e), prep, day)
                    sc.add(f"braw{w:.1f}", L, np.where(ok, (1 - w) * s_sat + w * fill(s_nwp, s_sat), s_sat), prep, day)
            if "trend" in groups:
                # the model's own change along the motion: model k at t minus the model k at ref where the air came from
                sm = lambda x: cv2.GaussianBlur(sd.fill_nan(x), (0, 0), 3.0)
                d = sm(m1q) - cv2.remap(sm(qm(m0)), mx, my, cv2.INTER_LINEAR, borderMode=cv2.BORDER_REPLICATE)
                for amt in (0.5, 1.0):
                    kt = np.clip(kk_sat + amt * d, 0, sd.K_MAX).astype(np.float32)
                    sc.add(f"tr{amt:.1f}", L, sd.share_from_k(np.where(np.isfinite(m1), kt, kk_sat), e), prep, day)
        if i % 10 == 0:
            print(f"run {i + 1}/{len(refs)} {sd.ts_stamp(ref)} ({time.time() - started:.0f}s)"
                  + (f" model age {(ref - run[1]) / 60_000:.0f} min" if run is not None else ""), flush=True)
    have = set(sc.s) | set(sc.g)
    names = ["pers", "sat", "shblur", "kpers", "nwp", "nwpqm"] + sorted({n for n, _ in have if n.startswith("nwpqm_age")})
    names += [n for n in ("f_s5", "f_s20", "f_d2", "f_dis", "f_sl", "f_avg", "f_60") if (n, 60) in have]
    names += [n for n in ("tr0.5", "tr1.0") if (n, 60) in have]
    names += [f"{p}_{s}" for p in ("s", "sb", "sk") for s in SCHEDULES if (f"{p}_{s}", 60) in have]
    names += [f"s55{s}" for s in SKILL if (f"s55{s}", 60) in have]
    if MAE0:
        print("model share MAE at ref (corrected) percentiles 10/25/50/75/90:",
              np.round(np.nanpercentile(MAE0, [10, 25, 50, 75, 90]), 3))
    names += [f"{p}{w:.1f}" for p in ("b", "bk", "braw") for w in WEIGHTS if (f"{p}{w:.1f}", 60) in have]
    sc.report(names)
    print("per day (station MAE / hit, grid MAE per run summed):")
    for name in [n for n in ("pers", "sat", "nwpqm", "b0.3", "b0.5", "bk0.3", "f_sl", "f_avg", "s_30_45", "s_30_55", "sb_30_45", "s55sk1", "s55sk2", "s55sk3") if (n, 120) in sc.s]:
        for L in (60, 120):
            print(f"  {name:8s} +{L:3d} " + "  ".join(
                f"{d}: {v[0] / max(v[2], 1):.3f}/{v[1] / max(v[2], 1):.3f}" for d in days
                for v in [sc.day.get((name, L, d), np.zeros(4))]))


if __name__ == "__main__":
    main()
