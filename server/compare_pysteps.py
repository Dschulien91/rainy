"""Compare the server's own nowcast with pySTEPS methods on the same recent radar cases (runs in the Docker image).

Variants (all on the 2 km grid, verified against the later observed radar):
  C  current server method (Farneback flow from a 30 min pair, lead-dependent smoothing)
  E  pySTEPS: Lucas-Kanade motion + plain semi-Lagrangian extrapolation
  S  pySTEPS S-PROG: E + scale-dependent decay (small cells fade, large fronts stay)
  N  pySTEPS ANVIL: E + local growth/decay of rain intensity from the recent trend
  P  persistence (baseline)
Two kinds of scores:
  pixel CSI  - strict: rain at exactly this 2 km pixel at exactly this lead time
  user view  - what an app user experiences: for places on a 10 km raster, "will it rain here (>= 0.2 mm/h) within
               the next 0-60 / 60-120 min?" as yes/no (hit rate, false alarms, CSI), plus the median error of the
               predicted rain start in minutes
Usage:  python server/compare_pysteps.py eu|us [hours=8] [step_min=40] [variants=CESNP]
"""
import os
import sys
import time

import numpy as np
from pysteps import motion, nowcasts
from pysteps.utils import transformation

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import rainy_server as rs  # noqa: E402
from verify_nowcast import Cache, counts  # noqa: E402

LEADS = (15, 30, 60, 90, 120)
THR = (0.5, 2.0)
STEPS = 24  # 5 min steps
WORKERS = os.cpu_count() or 1


class Cache2(Cache):
    """Keeps frames on the 2 km grid only (full 1 km frames would need gigabytes)."""
    def get(self, t):
        if t not in self.data:
            self.data[t] = to_2km(self.region.load(self.times[t]), self.region)
            if len(self.data) > self.size:
                self.data.popitem(last=False)
        self.data.move_to_end(t)
        return self.data[t]


PLACE_STEP = 5      # places every 5 grid pixels (~10 km)
WET = 0.2           # mm/h counted as "rain at my place"
WINDOWS = ((0, 60), (60, 120))


def place_series(frames):
    """Rain rate at each place (3x3 mean around it) for a list of frames -> (T, places)."""
    out = []
    for g in frames:
        g = np.where(g < 0, np.nan, g)
        ny, nx = g.shape
        ys, xs = np.mgrid[2:ny - 2:PLACE_STEP, 2:nx - 2:PLACE_STEP]
        acc = sum(g[ys + dy, xs + dx] for dy in (-1, 0, 1) for dx in (-1, 0, 1)) / 9.0
        out.append(acc.ravel())
    return np.array(out)


def user_scores(fc_series, ob_series, acc, key):
    """fc_series/ob_series: (24, places) for leads 5..120 min."""
    valid = ~np.isnan(ob_series).any(0) & ~np.isnan(fc_series).any(0)
    for a, b in WINDOWS:
        sl = slice(a // 5, b // 5)
        f, o = (fc_series[sl, valid] >= WET), (ob_series[sl, valid] >= WET)
        fy, oy = f.any(0), o.any(0)
        st = acc.setdefault((key, a), {"c": np.zeros(3, np.int64), "dt": []})
        st["c"] += [(fy & oy).sum(), (~fy & oy).sum(), (fy & ~oy).sum()]
        both = fy & oy & ~f[0] & ~o[0]  # start error only where it is dry at the beginning of the window
        if both.any():
            st["dt"].extend((np.abs(f[:, both].argmax(0) - o[:, both].argmax(0)) * 5).tolist())


def to_2km(g, region):
    g = g.astype(np.float32)
    if region.name != "eu":  # US grid is ~3 km already
        return g
    ny, nx = g.shape
    g = g[: ny // 2 * 2, : nx // 2 * 2].reshape(ny // 2, 2, nx // 2, 2)
    cov = (g >= 0).any(axis=(1, 3))
    rate = np.where(g < 0, 0, g).mean(axis=(1, 3))
    return np.where(cov, rate, rs.NODATA).astype(np.float32)


def lk_motion(frames):
    """Lucas-Kanade motion from frames 15 min apart, returned in pixels per 5 min step."""
    db = np.stack([transformation.dB_transform(np.where(f < 0, np.nan, f), threshold=0.1, zerovalue=-15.0)[0]
                   for f in frames])
    return motion.get_method("LK")(db) / 3.0


def finish(fc, last):
    fc = np.nan_to_num(np.asarray(fc, np.float32), nan=0.0)
    fc[fc < 0.05] = 0
    fc[:, last < 0] = rs.NODATA
    return list(fc)


def variant_E(seq, v):
    return finish(nowcasts.get_method("extrapolation")(np.where(seq[-1] < 0, 0, seq[-1]), v, STEPS), seq[-1])


def variant_S(seq, v):
    db, meta = transformation.dB_transform(np.stack([np.where(f < 0, 0, f) for f in seq[-3:]]), threshold=0.1, zerovalue=-15.0)
    fc = nowcasts.get_method("sprog")(db, v, STEPS, precip_thr=meta["threshold"], n_cascade_levels=6,
                                      num_workers=WORKERS, probmatching_method="cdf")
    return finish(transformation.dB_transform(fc, threshold=-10.0, inverse=True)[0], seq[-1])


def variant_N(seq, v):
    rr = np.stack([np.where(f < 0, 0, f) for f in seq[-4:]])
    fc = nowcasts.get_method("anvil")(rr, v, STEPS, n_cascade_levels=6, ar_window_radius=25, num_workers=WORKERS)
    return finish(fc, seq[-1])


def main(name, hours, step, variants):
    region = rs.REGIONS[name]
    times = region.list_times()
    cache = Cache2(region, times, size=40)
    ts = sorted(times)
    end, start = ts[-1] - 120 * 60_000, max(ts[0] + 30 * 60_000, ts[-1] - hours * 3600_000)
    acc, uacc, secs = {}, {}, {v: 0.0 for v in variants}
    t, n = start, 0
    while t <= end:
        base = cache.nearest(t)
        prev5 = [cache.nearest(t - k * 60_000, 2) for k in (15, 10, 5)]
        p30, p15 = cache.nearest(t - 30 * 60_000, 4), cache.nearest(t - 15 * 60_000, 4)
        if base and all(prev5) and p30 and p15 and len({base, *prev5, p30}) == 5:
            seq = [cache.get(x) for x in (*prev5, base)]  # t-15, t-10, t-5, t
            last, prev30 = seq[-1], cache.get(p30)
            fcs = {}
            v = None
            for var in variants:
                t0 = time.time()
                if var in "ESN" and v is None:
                    v = lk_motion([prev30, cache.get(p15), last])
                if var == "C":
                    fcs[var] = rs.nowcast(prev30, last, base - p30, STEPS, 1.0, 2)
                elif var == "E":
                    fcs[var] = variant_E(seq, v)
                elif var == "S":
                    fcs[var] = variant_S(seq, v)
                elif var == "N":
                    fcs[var] = variant_N(seq, v)
                elif var == "P":
                    fcs[var] = [last] * STEPS
                secs[var] += time.time() - t0
            obs_t = [cache.nearest(base + k * 5 * 60_000) for k in range(1, STEPS + 1)]
            if all(obs_t):
                ob_s = place_series([cache.get(x) for x in obs_t])
                for var, fc in fcs.items():
                    user_scores(place_series(fc), ob_s, uacc, var)
            for lead in LEADS:
                valid = cache.nearest(base + lead * 60_000)
                if not valid:
                    continue
                ob = cache.get(valid)
                for var, fc in fcs.items():
                    for thr in THR:
                        acc.setdefault((var, lead, thr), np.zeros(3, np.int64))
                        acc[(var, lead, thr)] += counts(fc[lead // 5 - 1], ob, thr)
            n += 1
            print(f"  case {n} ({time.strftime('%H:%MZ', time.gmtime(base / 1000))}) done", flush=True)
        t += step * 60_000
    print(f"\n=== {name}: {n} cases; CSI by variant (P = persistence) ===")
    for thr in THR:
        print(f"rain >= {thr} mm/h      " + "  ".join(f"{v:>5}" for v in variants))
        for lead in LEADS:
            row = []
            for var in variants:
                h, m, f = acc.get((var, lead, thr), np.zeros(3))
                row.append(f"{h / max(1, h + m + f):5.2f}")
            print(f"  {lead:3d}m             " + "  ".join(row))
    print("\nuser view: will it rain at my place (10 km raster) in the window?")
    for a, b in WINDOWS:
        print(f"  {a:3d}-{b:3d} min   " + "  ".join(f"{v:>22}" for v in variants))
        for name_, fn in (("hit rate", lambda h, m, f: h / max(1, h + m)), ("false alarms", lambda h, m, f: f / max(1, h + f)),
                          ("CSI", lambda h, m, f: h / max(1, h + m + f))):
            print(f"  {name_:13s}" + "  ".join(f"{fn(*uacc[(v, a)]['c']) if (v, a) in uacc else 0:22.2f}" for v in variants))
        print(f"  {'start error':13s}" + "  ".join(
            f"{np.median(uacc[(v, a)]['dt']) if (v, a) in uacc and uacc[(v, a)]['dt'] else 0:19.0f} min" for v in variants))
    print("seconds per case: " + ", ".join(f"{v} {secs[v] / max(1, n):.0f}" for v in variants), flush=True)


if __name__ == "__main__":
    a = sys.argv
    main(a[1], float(a[2]) if len(a) > 2 else 8, int(a[3]) if len(a) > 3 else 40, a[4] if len(a) > 4 else "CESNP")
