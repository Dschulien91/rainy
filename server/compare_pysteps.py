"""Compare the server's own nowcast with pySTEPS methods on the same recent radar cases (runs in the Docker image).

Variants (all on the 2 km grid, verified against the later observed radar):
  C  current server method (Farneback flow from a 30 min pair, lead-dependent smoothing)
  E  pySTEPS: Lucas-Kanade motion + plain semi-Lagrangian extrapolation
  S  pySTEPS S-PROG: E + scale-dependent decay (small cells fade, large fronts stay)
  N  pySTEPS ANVIL: E + local growth/decay of rain intensity from the recent trend
  P  persistence (baseline)
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
    cache = Cache(region, times, size=80)
    ts = sorted(times)
    end, start = ts[-1] - 120 * 60_000, max(ts[0] + 30 * 60_000, ts[-1] - hours * 3600_000)
    acc, secs = {}, {v: 0.0 for v in variants}
    t, n = start, 0
    while t <= end:
        base = cache.nearest(t)
        prev5 = [cache.nearest(t - k * 60_000, 2) for k in (15, 10, 5)]
        p30, p15 = cache.nearest(t - 30 * 60_000, 4), cache.nearest(t - 15 * 60_000, 4)
        if base and all(prev5) and p30 and p15 and len({base, *prev5, p30}) == 5:
            seq = [to_2km(cache.get(x), region) for x in (*prev5, base)]  # t-15, t-10, t-5, t
            last, prev30 = seq[-1], to_2km(cache.get(p30), region)
            fcs = {}
            v = None
            for var in variants:
                t0 = time.time()
                if var in "ESN" and v is None:
                    v = lk_motion([prev30, to_2km(cache.get(p15), region), last])
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
            for lead in LEADS:
                valid = cache.nearest(base + lead * 60_000)
                if not valid:
                    continue
                ob = to_2km(cache.get(valid), region)
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
    print("seconds per case: " + ", ".join(f"{v} {secs[v] / max(1, n):.0f}" for v in variants), flush=True)


if __name__ == "__main__":
    a = sys.argv
    main(a[1], float(a[2]) if len(a) > 2 else 8, int(a[3]) if len(a) > 3 else 40, a[4] if len(a) > 4 else "CESNP")
