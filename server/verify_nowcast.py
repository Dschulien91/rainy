"""Backtest of the rainy nowcast against what the radar observed later.

For base times over the last hours: forecast +15..+120 min from the radar at the base time (same code as the
server), then compare with the observed radar at the valid time. Baseline: persistence ("everything stays where
it is"), i.e. the base-time radar used as forecast for every lead time.

Scores per threshold (rain rate in mm/h) on the 2-3 km grid, pixels with radar coverage only:
  POD = hits / (hits + misses)                  share of observed rain that was forecast
  FAR = false alarms / (hits + false alarms)    share of forecast rain that did not happen
  CSI = hits / (hits + misses + false alarms)   overall hit score, 1 = perfect

Usage:  python server/verify_nowcast.py [hours=10] [step_min=30]
"""
import os
import sys
import time
from collections import OrderedDict

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import rainy_server as rs  # noqa: E402  (no threads start on import)

LEADS = (15, 30, 60, 90, 120)
THRESHOLDS = (0.5, 2.0)


class Cache:
    def __init__(self, region, times, size=40):
        self.region, self.times, self.size, self.data = region, times, size, OrderedDict()

    def nearest(self, t, tol_min=2.6):
        best = min(self.times, key=lambda x: abs(x - t))
        return best if abs(best - t) <= tol_min * 60_000 else None

    def get(self, t):
        if t not in self.data:
            self.data[t] = self.region.load(self.times[t])
            if len(self.data) > self.size:
                self.data.popitem(last=False)
        self.data.move_to_end(t)
        return self.data[t]


def counts(fc, ob, thr):
    m = (fc >= 0) & (ob >= 0)
    f, o = fc[m] >= thr, ob[m] >= thr
    return np.array([(f & o).sum(), (~f & o).sum(), (f & ~o).sum()], np.int64)


def run(region, hours, step_min):
    times = region.list_times()
    if not times:
        print(f"{region.name}: no data")
        return
    cache = Cache(region, times)
    ts = sorted(times)
    end = ts[-1] - max(LEADS) * 60_000
    start = max(ts[0] + 30 * 60_000, ts[-1] - hours * 3600_000)
    acc = {(k, lead, thr): np.zeros(3, np.int64) for k in ("nowcast", "persistence") for lead in LEADS for thr in THRESHOLDS}
    bases = 0
    t = start
    while t <= end:
        base = cache.nearest(t)
        prev = cache.nearest(t - 30 * 60_000, tol_min=4)  # same pairing as the server
        if base and prev and prev != base:
            started = time.time()
            last = cache.get(base)
            fc = rs.nowcast(cache.get(prev), last, base - prev, max(LEADS) // 5)
            for lead in LEADS:
                valid = cache.nearest(base + lead * 60_000)
                if not valid:
                    continue
                ob = cache.get(valid)
                for thr in THRESHOLDS:
                    acc[("nowcast", lead, thr)] += counts(fc[lead // 5 - 1], ob, thr)
                    acc[("persistence", lead, thr)] += counts(last, ob, thr)
            bases += 1
            print(f"  {region.name} base {time.strftime('%H:%M', time.gmtime(base / 1000))}Z done ({time.time() - started:.0f}s)", flush=True)
        t += step_min * 60_000

    print(f"\n=== {region.name} ({region.credit}): {bases} forecasts, last {hours} h ===")
    for thr in THRESHOLDS:
        print(f"rain >= {thr} mm/h")
        print("  lead   CSI nowcast  CSI persist.  POD    FAR")
        for lead in LEADS:
            h, m, f = acc[("nowcast", lead, thr)]
            ph, pm, pf = acc[("persistence", lead, thr)]
            csi = h / max(1, h + m + f)
            pcsi = ph / max(1, ph + pm + pf)
            pod, far = h / max(1, h + m), f / max(1, h + f)
            print(f"  {lead:3d}m   {csi:6.2f}       {pcsi:6.2f}      {pod:5.2f}  {far:5.2f}")
    print(flush=True)


if __name__ == "__main__":
    hours = float(sys.argv[1]) if len(sys.argv) > 1 else 10
    step = int(sys.argv[2]) if len(sys.argv) > 2 else 30
    for r in (rs.EuropeOpera(), rs.UsaNowcoast()):
        run(r, hours, step)
