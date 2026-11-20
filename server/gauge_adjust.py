"""Gauge adjustment of a Europe radar picture: a mean-field bias per gauge network, learned at the rain gauges on a
rolling window and applied to every new image (switch: GAUGE_ADJUST=1 in rainy_server.py, default off).

Why (18.11.2026, competitor bench and check at the gauges, 16.-18.11.): the pictures are too wet in Austria (TAWES
gauges, analysis x1.7 / x1.9 in "European" / own picture), Wallonia (SPW x1.4 / x1.3), Hungary (x1.7), France (SYNOP
x1.4) and too dry in Sweden / Finland (x0.35 / x0.5) and in Slovenia (x0.35) - while Germany and Czechia are ~x1.0.
These are radar calibration / range effects of the composite (OPERA has no Austrian, Italian or Pasja ravan radar:
Austria is seen from far beyond its borders, at the height of the melting layer and above), steady over many hours,
so the gauges of the last hours tell how far off a network's area is now.

How:
  * the picture BEFORE the adjustment is noted at every gauge (5 x 5 cells, as check_national_gauges.at_gauges) for
    every 10-minute image; at start the last hours are taken from verify_log/eun/natobs (key "opera" = the OPERA
    instance's picture, "nat" = the own picture, both unadjusted);
  * every 10 minutes, per network (SYNOP: per country) over the last WIN_H hours that end LAG_H before now, only the
    10-minute steps where the gauge or the picture is at least WET mm/h:
        factor = (sum of gauge mm + PRIOR) / (sum of picture mm + PRIOR), clipped to [MIN_F, MAX_F]
    (a ratio of sums - wet hours dominate; a dry network stays at 1); truth = the gauge archive (gaugetruth, written
    hourly by the OPERA instance) - only values the gauges really reported count, without implausible ones
    (check_national_gauges.implausible) and with no gauge above SHARE_MAX of its network's sum (server-v1.79);
  * the factor field: every gauge carries its network's factor (log), spread with a Gaussian of FIELD_KM on a
    ~16 km grid, fading to 1 where gauges are few or none (FIELD_D0); multiplied into every rain pixel of the image
    (no data and dry pixels stay as they are).

Offline test (check_gauge_adjust.py; 2-fold spatial split: factors learned on half of every network's gauges, scored
on the other half; tuned on 15./16.11., held out 17./18.11.2026, hourly sums CSI at 0.2 / 1 / 3 mm, amount):
  "European" all gauges x0.91 0.61/0.47/0.28 -> x0.98 0.63/0.48/0.29; own picture x0.94 0.61/0.49/0.29 ->
  x1.01 0.63/0.49/0.29. Austria x1.55 -> x1.03 (CSI 0.49/0.40 -> 0.55/0.33: its excess is light echo over dry
  gauges, which a factor can only scale), Wallonia x1.45 -> x1.21 (0.66/0.53 -> 0.68/0.59 at 1 / 3 mm), Germany
  x0.90 -> x1.00, Hungary x1.71 -> x1.08, Finland x0.53 -> x1.01 (0.51/0.17 -> 0.69/0.28), Slovenia x0.39 -> x0.81,
  Portugal x0.34 -> x0.81 (0.46/0.21/0.04 -> 0.52/0.35/0.26), Spain SYNOP x0.31 -> x0.93, France SYNOP x1.39 -> x1.15.
Intensity: split by the PICTURE's hourly class (what a correction can use), Germany / Czechia - x1.0 overall - read
x0.84-0.88 / x0.91-0.94 / x1.5-1.7 in hours of 0.2-1 / 1-5 / > 5 mm (split by the GAUGE's class: x1.1 / x0.8 /
x0.65 - the usual scatter of a 10 km mean against a point, both ways). Austria (x1.71 / x1.27), Spain / Portugal
(x0.4-0.6) and Finland are off by a level in every class, not by intensity; a factor per intensity class (picture
< 1 / >= 1 mm/h) made the picture too dry (x0.85) with lower CSI at 1 mm on the tuning days -> one factor.
"""
import glob
import os
import threading
import time

import cv2
import numpy as np

import check_national_gauges as cng

WIN_H = float(os.environ.get("GAUGE_ADJUST_WIN_H", 12))
LAG_H = 1.5            # the newest gauge values are ~1 h late (archive hourly, networks 10-90 min behind)
PRIOR = 3.0            # mm per network: shrinks the factor to 1 when little rain was seen
WET = 0.3              # mm/h: steps where neither the gauge nor the picture reach this do not count
MIN_F, MAX_F = 0.8, 3.0   # 19.11.2026: min 0.5 -> 0.8 (held-out 17.-19.11.: ALL x0.99 -> x1.02, CSI 1/3 mm .49/.29 ->
                          # .50/.30, Austria 1 mm .28 -> .37 - a deep cut also removes real heavy rain)
FIELD_KM = 40.0        # spread of the gauges' factors on the map
FIELD_D0 = 0.02        # gauges per 16 km cell (after the spread) at which a factor counts half
CELL_KM = 16.0
EVERY_S = 600
# networks that never take part (comma-separated codes, SYNOP countries as "sy-xx"), e.g. GAUGE_ADJUST_SKIP=de,bf
# Default "ch,iv" (19.11.2026, held-out 58 h): Switzerland raw x1.04 CSI 1/3 mm .63/.28 vs adjusted x0.88 .58/.16,
# Veneto 3 mm .27 -> .08 with no amount gain - the factor cuts the heavy cores there. GAUGE_ADJUST_SKIP= (empty) = none.
SKIP = {s.strip() for s in os.environ.get("GAUGE_ADJUST_SKIP", "ch,iv").split(",") if s.strip()}
MS10 = 600_000
# no single gauge may carry more than SHARE_MAX of its network's gauge sum - a safety net behind the plausibility
# check (server-v1.79, 20.11.2026: en:1552 with a faulty 1742 mm was 97 % of England's sum -> x3.0 for 12 h; with
# this cap alone x1.4-1.5). Its gauge and picture sums are scaled down together, so a real heavy shower keeps its
# ratio, only its weight. Networks with fewer than SHARE_MIN_N wet gauges: no cap. Replay 14.-20.11. (54 runs):
# 1/3 + 20 changed one legit factor (se 2.83 -> 2.70); 0.2 + 10 changed 63, mostly small SYNOP countries (up to -0.7)
SHARE_MAX, SHARE_MIN_N = 1 / 3, 20


def group(s):
    return f"sy-{s[1].split('-', 1)[0]}" if s[0] == "sy" else s[0]


def network_factors(gs, rs, groups, seen):
    """Per gauge sums of the gauge (gs) and picture (rs, mm) over the window -> (factor per gauge, {network: (factor,
    gauges, gauge mm)}). seen: the gauges that take part."""
    fac = np.ones(len(gs), np.float32)
    factors = {}
    for g in np.unique(groups[seen]):
        m = seen & (groups == g)
        a, b = gs[m].astype(np.float64), rs[m].astype(np.float64)
        if (a > 0).sum() >= SHARE_MIN_N:
            cap = SHARE_MAX / (1 - SHARE_MAX) * (a.sum() - a)   # SHARE_MAX of the sum with the others as they are
            w = np.where(a > cap, cap / np.maximum(a, 1e-9), 1.0)
            a, b = a * w, b * w
        f = float(np.clip((a.sum() + PRIOR) / (b.sum() + PRIOR), MIN_F, MAX_F))
        fac[m] = f
        factors[g] = (round(f, 2), int(m.sum()), round(float(a.sum()), 1))
    return fac, factors


class GaugeAdjust:
    def __init__(self, host, key, log):
        """host: the Europe region (grid, to_pixels, gauges()); key: the natobs field of this picture ("opera" for
        the OPERA instance, "nat" for the own picture) - for the hours before the start."""
        self.host, self.key, self.log = host, key, log
        self.lock = threading.Lock()
        self.pic = {}           # 10-min time -> picture at the gauges before the adjustment (float16)
        self.field = None       # factor on the coarse grid (float32), None = not yet / all 1
        self.factors = {}       # network -> factor (for the log / health)
        self.st = None
        self.logged = 0.0
        self.backfilled = False

    # ---------- the pictures at the gauges ----------
    def _gauges(self):
        if self.st is None:
            st = cng.stations()
            px = self.host.gauges()
            if not st or px is None or len(px[0]) != len(st):
                return None
            self.st = st
            self.groups = np.array([group(s) for s in st])
            self.use = np.array([s[0] not in SKIP and group(s) not in SKIP for s in st])
        return self.host.gauge_px

    def _backfill(self):
        """The last hours from natobs (written by the own-picture instance for both pictures, unadjusted)."""
        self.backfilled = True
        now = time.time() * 1000
        lo = now - (WIN_H + LAG_H + 1) * 3600_000
        for fp in glob.glob(os.path.join(cng.ROOT, "eun", "natobs", "*.npz")):
            b = os.path.basename(fp)[:-4]
            if not b.isdigit() or int(b) % MS10 or int(b) < lo:
                continue
            try:
                v = np.load(fp)[self.key].astype(np.float16)
            except Exception:
                continue
            full = np.full(len(self.st), np.nan, np.float16)
            full[:min(len(v), len(full))] = v[:len(full)]
            with self.lock:
                self.pic.setdefault(int(b), full)

    def note(self, ts, rate):
        if ts % MS10:
            return
        px = self._gauges()
        if px is None:
            return
        v = cng.at_gauges(rate, px).astype(np.float16)
        with self.lock:
            self.pic[int(ts)] = v
            old = time.time() * 1000 - (WIN_H + LAG_H + 2) * 3600_000
            for t in [t for t in self.pic if t < old]:
                del self.pic[t]

    # ---------- the factors ----------
    def compute(self):
        if self._gauges() is None:
            return
        if not self.backfilled:
            self._backfill()
        now = time.time() * 1000
        hi = (now - LAG_H * 3600_000) // MS10 * MS10
        lo = hi - WIN_H * 3600_000
        with self.lock:
            ts = sorted(t for t in self.pic if lo < t <= hi)
            R = np.stack([self.pic[t] for t in ts]).astype(np.float32) if ts else None
        if R is None:
            return
        G = np.full(R.shape, np.nan, np.float32)
        idx = {f"{s[0]}:{s[1]}": k for k, s in enumerate(self.st)}
        ts_a = np.array(ts, np.int64)
        rejected = {}  # implausible gauge values (check_national_gauges.implausible), dropped by _day_load
        for day in sorted({t // 86400_000 for t in ts}):
            rows = cng._day_load(day, rejected)
            sel = np.flatnonzero(ts_a // 86400_000 == day)
            col = (ts_a[sel] % 86400_000) // MS10
            for key, row in rows.items():
                k = idx.get(key)
                if k is not None:
                    G[sel, k] = row[col]
        ok = np.isfinite(G) & np.isfinite(R) & (R >= 0)
        wet = ok & ((G * 6 >= WET) | (R >= WET))
        gs = np.where(wet, G, 0).sum(0)
        rs = np.where(wet, R / 6, 0).sum(0)
        seen = ok.any(0) & self.use
        fac, factors = network_factors(gs, rs, self.groups, seen)
        # the field: log factors of the gauges, spread and faded to 1 where there are few
        col, row, inside = self.host.gauge_px
        c = max(1, int(round(CELL_KM / self.host.KM_PER_PX)))
        cy, cx = -(-self.host.NY // c), -(-self.host.NX // c)
        num = np.zeros((cy, cx), np.float32)
        den = np.zeros((cy, cx), np.float32)
        k = np.flatnonzero(seen & inside)
        np.add.at(num, (row[k] // c, col[k] // c), np.log(fac[k]))
        np.add.at(den, (row[k] // c, col[k] // c), 1.0)
        sig = FIELD_KM / (c * self.host.KM_PER_PX)
        num = cv2.GaussianBlur(num, (0, 0), sig, borderType=cv2.BORDER_CONSTANT)
        den = cv2.GaussianBlur(den, (0, 0), sig, borderType=cv2.BORDER_CONSTANT)
        field = np.exp(num / (den + FIELD_D0)).astype(np.float32)
        with self.lock:
            self.field, self.factors, self.cell = field, factors, c
        if time.time() - self.logged > 3600:
            self.logged = time.time()
            top = sorted(factors.items(), key=lambda x: -x[1][2])[:16]
            self.log("gauge adjust (" + self.key + "): " + ", ".join(f"{g} x{f[0]}" for g, f in top))
            if rejected:
                bad = sorted(rejected.items(), key=lambda x: -x[1])
                self.log(f"gauge adjust ({self.key}): implausible values dropped at {len(bad)} gauges: "
                         + ", ".join(f"{k} {v:.1f}" for k, v in bad[:8]) + (" ..." if len(bad) > 8 else ""))

    def loop(self):
        time.sleep(60)
        while True:
            try:
                self.compute()
            except Exception as e:
                self.log(f"gauge adjust failed: {e}")
            time.sleep(EVERY_S)

    # ---------- the image ----------
    def apply(self, ts, rate):
        """rate (float32, host grid, < 0 = no data) -> adjusted copy; notes the unadjusted picture at the gauges."""
        try:
            self.note(ts, rate)
        except Exception as e:
            self.log(f"gauge adjust (note) failed: {e}")
        with self.lock:
            field, c = self.field, getattr(self, "cell", 1)
        if field is None:
            return rate
        ny, nx = rate.shape
        f = cv2.resize(field, (field.shape[1] * c, field.shape[0] * c), interpolation=cv2.INTER_LINEAR)[:ny, :nx]
        return np.where(rate > 0, rate * f, rate).astype(np.float32)
