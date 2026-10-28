"""Backtest of blending the radar nowcast with the weather model rain (second hour).

Uses the model snapshots the server logs (model_log/<region>_<issued ms>.npz). For each base time only the newest
snapshot fetched *before* that time is used, so the test sees exactly what the live server would have had.
Blend variants (radar weight by lead time):
  R   radar nowcast only
  B   current server setting (radar only until 60 min, 50/50 at 120 min)
  N   "new cells": radar unchanged, model adds showers only where the radar has no rain within 10 km
  M   model only (reference)
Usage:  python server/verify_blend.py eu|us [hours=10] [step_min=30]
"""
import glob
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import rainy_server as rs  # noqa: E402
from verify_nowcast import Cache, counts  # noqa: E402

LEADS = (15, 30, 60, 90, 120)
DOWN = 3  # evaluation grid: 3 km
THR = (0.5, 2.0)


VARIANTS = {  # (model, fc, base, times, rate) -> blended fc
    "R": lambda md, fc, b, t, r: fc,
    "B": lambda md, fc, b, t, r: md.blend(fc, b, t, r, rs.radar_weight),
    "N": lambda md, fc, b, t, r: rs.blend_new_cells(md, fc, b, t, r, km_per_px=md.region.KM_PER_PX * DOWN),
    "M": lambda md, fc, b, t, r: md.blend(fc, b, t, r, lambda m: 0.0),
}


def snapshots(name):
    out = []
    for f in glob.glob(os.path.join(rs.MODEL_LOG_DIR, f"{name}_*.npz")):
        out.append((int(os.path.basename(f).split("_")[1].split(".")[0]), f))
    return sorted(out)


def main(name, hours, step):
    region = rs.REGIONS[name]
    snaps = snapshots(name)
    if not snaps:
        print("no model snapshots yet - let the server run for a few hours first")
        return
    times = region.list_times()
    cache = Cache(region, times, size=12)  # 1 km frames are big: keep only what one case needs
    ts = sorted(times)
    end = ts[-1] - 120 * 60_000
    start = max(ts[0] + 30 * 60_000, ts[-1] - hours * 3600_000, snaps[0][0] + 60_000)
    model = rs.ModelRain(region)
    acc, n, t = {}, 0, start
    while t <= end:
        base, prev = cache.nearest(t), cache.nearest(t - 30 * 60_000, 4)
        snap = [f for issued, f in snaps if issued <= (base or 0) and base - issued <= rs.MODEL_MAX_AGE_MS]
        if base and prev and prev != base and snap:
            full = cache.get(base)
            if model.maps is None:
                model._setup(full)
            # evaluated on a 3 km grid: all blend variants of a 24-step 1 km nowcast at once would need ~8 GB
            last, prev_g = rs._coarse(full, DOWN), rs._coarse(cache.get(prev), DOWN)
            z = np.load(snap[-1])
            mt, mr = z["times"], z["rate"].astype(np.float32)
            fc0 = rs.nowcast(prev_g, last, base - prev, rs.FC_STEPS, region.SMOOTH / DOWN, max(1, region.FLOW_DOWN // DOWN))
            fcs = {v: f(model, [g.copy() for g in fc0], base, mt, mr) for v, f in VARIANTS.items()}
            for lead in LEADS:
                valid = cache.nearest(base + lead * 60_000)
                if not valid:
                    continue
                ob = rs._coarse(cache.get(valid), DOWN)
                for v, fc in fcs.items():
                    for thr in THR:
                        acc.setdefault((v, lead, thr), np.zeros(3, np.int64))
                        acc[(v, lead, thr)] += counts(fc[lead // 5 - 1], ob, thr)
            n += 1
            print(f"  case {n} done", flush=True)
        t += step * 60_000
    print(f"\n=== {name}: {n} cases; CSI by blend variant ===")
    for thr in THR:
        print(f"rain >= {thr} mm/h      " + "  ".join(f"{v:>5}" for v in VARIANTS))
        for lead in LEADS:
            row = []
            for v in VARIANTS:
                h, m, f = acc.get((v, lead, thr), np.zeros(3))
                row.append(f"{h / max(1, h + m + f):5.2f}")
            print(f"  {lead:3d}m             " + "  ".join(row))


if __name__ == "__main__":
    a = sys.argv
    main(a[1], float(a[2]) if len(a) > 2 else 10, int(a[3]) if len(a) > 3 else 30)
