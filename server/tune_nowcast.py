"""Compare nowcast variants on the same recent cases (Europe or USA) to tune the server's method.

Variants:
  A  current: flow from frames 15 min apart, no smoothing
  B  flow from frames 30 min apart
  C  B + scale-dependent smoothing that grows with lead time (small cells fade out, as in S-PROG/pySTEPS)
  D  C + flow averaged from the 15 min and 30 min pairs
Usage:  python server/tune_nowcast.py eu|us [hours=8] [step_min=40]
"""
import math
import os
import sys

import cv2
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import rainy_server as rs  # noqa: E402
from verify_nowcast import Cache, counts  # noqa: E402

LEADS = (15, 30, 60, 90, 120)
THR = (0.5, 2.0)


def flow_between(prev, last, dt_ms):
    ny, nx = last.shape

    def img(r):
        v = np.where(r > 0, np.log1p(np.clip(r, 0, 50)), 0)
        return np.clip(v / math.log1p(50) * 255, 0, 255).astype(np.uint8)
    small = (nx // 2, ny // 2)
    a = cv2.resize(img(prev), small, interpolation=cv2.INTER_AREA)
    b = cv2.resize(img(last), small, interpolation=cv2.INTER_AREA)
    f = cv2.calcOpticalFlowFarneback(a, b, None, 0.5, 4, 25, 4, 7, 1.5, 0)
    f = cv2.GaussianBlur(f, (0, 0), 6)
    return cv2.resize(f, (nx, ny), interpolation=cv2.INTER_LINEAR) * 2.0 * (rs.STEP_MS / dt_ms)


def extrapolate(last, flow, steps, smooth):
    ny, nx = last.shape
    gx, gy = np.meshgrid(np.arange(nx, dtype=np.float32), np.arange(ny, dtype=np.float32))
    base = np.where(last < 0, 0, last).astype(np.float32)
    out = []
    for k in range(1, steps + 1):
        r = cv2.remap(base, gx - flow[..., 0] * k, gy - flow[..., 1] * k, cv2.INTER_LINEAR,
                      borderMode=cv2.BORDER_CONSTANT, borderValue=0)
        if smooth:
            sigma = smooth * math.sqrt(k)  # grows with lead time
            r = cv2.GaussianBlur(r, (0, 0), sigma)
        r[last < 0] = rs.NODATA
        out.append(r)
    return out


def main(name, hours, step):
    region = rs.REGIONS[name]
    times = region.list_times()
    cache = Cache(region, times, size=60)
    ts = sorted(times)
    end, start = ts[-1] - 120 * 60_000, max(ts[0] + 30 * 60_000, ts[-1] - hours * 3600_000)
    acc = {}
    t, n = start, 0
    while t <= end:
        base, p15, p30 = cache.nearest(t), cache.nearest(t - 15 * 60_000, 4), cache.nearest(t - 30 * 60_000, 4)
        if base and p15 and p30 and len({base, p15, p30}) == 3:
            last = cache.get(base)
            f15 = flow_between(cache.get(p15), last, base - p15)
            f30 = flow_between(cache.get(p30), last, base - p30)
            fcs = {"A": extrapolate(last, f15, 24, 0), "B": extrapolate(last, f30, 24, 0),
                   "C": extrapolate(last, f30, 24, 1.0), "D": extrapolate(last, (f15 + f30) / 2, 24, 1.0),
                   "P": [last] * 24}
            for lead in LEADS:
                valid = cache.nearest(base + lead * 60_000)
                if not valid:
                    continue
                ob = cache.get(valid)
                for v, fc in fcs.items():
                    for thr in THR:
                        acc.setdefault((v, lead, thr), np.zeros(3, np.int64))
                        acc[(v, lead, thr)] += counts(fc[lead // 5 - 1], ob, thr)
            n += 1
            print(f"  case {n} done", flush=True)
        t += step * 60_000
    print(f"\n=== {name}: {n} cases; CSI by variant (P = persistence) ===")
    for thr in THR:
        print(f"rain >= {thr} mm/h      " + "  ".join(f"{v:>5}" for v in "ABCDP"))
        for lead in LEADS:
            row = []
            for v in "ABCDP":
                h, m, f = acc.get((v, lead, thr), np.zeros(3))
                row.append(f"{h / max(1, h + m + f):5.2f}")
            print(f"  {lead:3d}m             " + "  ".join(row))


if __name__ == "__main__":
    main(sys.argv[1], float(sys.argv[2]) if len(sys.argv) > 2 else 8, int(sys.argv[3]) if len(sys.argv) > 3 else 40)
