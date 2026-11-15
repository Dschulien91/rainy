"""Would a running gauge-based bias correction make the radar picture better? An offline test on the recorded data.

The radar's rain amount is off by region (15.11.2026, hourly sums at the gauges: England x1.17, Finland x0.45, Sweden
x0.32, the Italian layer in drizzle x7): the classic remedy in hydrology is a bias field from the gauges (mean-field
bias, then local bias spread with a kernel, smoothed in time - the Kalman-filter literature). Before anything goes
into the picture this scores it honestly:
  * pairs: per gauge and full hour, the gauge's mm and the picture's mean mm/h (verify_log/eun/natobs: "opera" = the
    European picture, "nat" = the own picture; truth from the gauge archive, check_national_gauges.truth);
  * for the hour h the factor at a gauge uses only hours <= h - LAG (gauges report late) of OTHER gauges (leave one
    out), weighted by a Gaussian kernel in distance (KERNEL_KM) and an exponential decay in time (TAU_H):
        factor = (sum w G + PRIOR) / (sum w R + PRIOR), clipped to [MIN_F, MAX_F]
    - a ratio of sums (wet hours dominate), shrunk to 1 where little rain was seen (PRIOR mm);
  * scores raw and corrected per country: amount ratio, CSI / POD / FAR at 0.2 / 1 / 3 mm per hour.

  docker exec -w /app deploy-api-eun-1 python check_bias.py [--hours 40] [--kernel 60] [--tau 6] [--lag 2] [--prior 2]
"""
import glob
import os
import sys
import time

import numpy as np

import check_national_gauges as cng


def arg(name, default):
    return float(sys.argv[sys.argv.index(name) + 1]) if name in sys.argv else default


HOURS, KERNEL_KM, TAU_H, LAG_H, PRIOR = arg("--hours", 40), arg("--kernel", 60), arg("--tau", 6), arg("--lag", 2), arg("--prior", 2)
MIN_F, MAX_F = 0.25, 4.0
THS = (0.2, 1.0, 3.0)

st = cng.stations()
n = len(st)
lat = np.radians([s[2] for s in st]); lon = np.radians([s[3] for s in st])
country = np.array([s[0] for s in st])
now = time.time() * 1000
files = sorted(f for f in glob.glob(os.path.join(cng.OBS_DIR, "*.npz"))
               if int(os.path.basename(f)[:-4]) % 600_000 == 0 and int(os.path.basename(f)[:-4]) > now - HOURS * 3600_000)
ts = [int(os.path.basename(f)[:-4]) for f in files]
truth = cng.truth(st, ts[0] - 1200_000, ts[-1] + 1200_000, live=False)
hours = sorted({(t - 1) // 3600_000 for t in ts})
H = len(hours)
G = np.full((H, n), np.nan, np.float32)
R = {k: np.full((H, n), np.nan, np.float32) for k in ("opera", "nat")}
acc = {k: np.zeros((H, n), np.float32) for k in ("G", "opera", "nat")}
cnt = np.zeros((H, n), np.int16)
hidx = {h: i for i, h in enumerate(hours)}
for f, t in zip(files, ts):
    z = np.load(f)
    m = min(n, len(z["opera"]))
    i = hidx[(t - 1) // 3600_000]
    o, a = z["opera"][:m].astype(np.float32), z["nat"][:m].astype(np.float32)
    for k in range(m):
        q = truth.get(st[k][0], {}).get(st[k][1], {})
        v = q.get(t)
        if v is None or not np.isfinite(o[k]) or not np.isfinite(a[k]):
            continue
        acc["G"][i, k] += v; acc["opera"][i, k] += o[k]; acc["nat"][i, k] += a[k]; cnt[i, k] += 1
full = cnt == 6
G[full] = acc["G"][full]
for k in ("opera", "nat"):
    R[k][full] = acc[k][full] / 6.0
print(f"{len(files)} images, {H} hours, {int(full.sum())} gauge-hours with all six 10-min values, {n} gauges")

# kernel between gauges (leave one out: zero on the diagonal)
d = np.arccos(np.clip(np.sin(lat)[:, None] * np.sin(lat)[None] + np.cos(lat)[:, None] * np.cos(lat)[None] * np.cos(lon[:, None] - lon[None]), -1, 1)) * 6371
K = np.exp(-0.5 * (d / KERNEL_KM) ** 2).astype(np.float32)
np.fill_diagonal(K, 0)
K[d > 3 * KERNEL_KM] = 0
del d


def corrected(Rk):
    out = np.full_like(Rk, np.nan)
    lag = int(LAG_H)
    for i in range(H):
        past = [j for j in range(i - lag + 1) if j >= 0]
        if not past:
            continue
        w = np.exp(-(i - np.array(past)) / TAU_H)[:, None].astype(np.float32)
        ok = np.isfinite(G[past]) & np.isfinite(Rk[past])
        g = np.where(ok, G[past], 0) * w
        r = np.where(ok, Rk[past], 0) * w
        sg, sr = g.sum(0), r.sum(0)            # per gauge, weighted in time
        fg, fr = K @ sg, K @ sr                # spread to every gauge from the others
        f = np.clip((fg + PRIOR) / (fr + PRIOR), MIN_F, MAX_F)
        out[i] = Rk[i] * f
    return out


def scores(g, r):
    ok = np.isfinite(g) & np.isfinite(r)
    g, r = g[ok], r[ok]
    if not len(g) or g.sum() < 1:
        return None
    cells = []
    for th in THS:
        h, m, f = ((g >= th) & (r >= th)).sum(), ((g >= th) & (r < th)).sum(), ((g < th) & (r >= th)).sum()
        cells.append(f"{th}: CSI {h / max(1, h + m + f):.2f} POD {h / max(1, h + m):.2f} FAR {f / max(1, h + f):.2f}")
    return f"x{r.sum() / g.sum():5.2f}  mae {np.abs(r - g).mean():.3f}   " + "   ".join(cells), len(g), g.mean()


print(f"kernel {KERNEL_KM:.0f} km, tau {TAU_H:.0f} h, lag {LAG_H:.0f} h, prior {PRIOR} mm, factor [{MIN_F}, {MAX_F}]")
cor = {k: corrected(R[k]) for k in R}
for name in sorted(set(country)) + ["all"]:
    sel = np.ones(n, bool) if name == "all" else country == name
    lines = []
    for k, label in (("opera", "European"), ("nat", "own")):
        a, b = scores(G[:, sel], R[k][:, sel]), scores(G[:, sel], cor[k][:, sel])
        if a:
            lines.append(f"  {label:9s} raw        {a[0]}")
            lines.append(f"  {label:9s} corrected  {b[0]}")
    if lines:
        s = scores(G[:, sel], R["nat"][:, sel])
        print(f"\n{cng.NAMES.get(name, name)}: {s[1]} gauge-hours, gauge mean {s[2]:.3f} mm")
        print("\n".join(lines))
