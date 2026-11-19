"""Offline test: the served Europe forecast where its source lies OUTSIDE the radar coverage (Atlantic inflow edge).

Input: the served forecasts archived by warn_check's recorder (warnfc_eu/<ref>.npz: r30/r60/r120 on the obs_eu grid)
and the 4 km analyses (obs_eu). The motion is re-estimated like sprog_nowcast (pySTEPS LK on dB of the frames
30/15/0 min old, per 5 min). For each lead the pixel's source = the pixel moved back along the motion (as
advect_blend / local_amount); cov_src = the coverage there, blurred (taper). Variants replace the forecast by a fill
with weight wout = 1 - cov_src:
  euler   the last image at the pixel (Eulerian persistence)
  edge    the last image extended beyond the coverage edge (nearest covered value, smoothed along the edge, fading
          with the distance outside), moved along the motion = "the rain keeps coming in as it does at the edge"
  edgemax max(forecast, edge fill) in the outside part
Scored per coastal region and the whole domain against the later analysis (cell by cell, 0.1 / 1 mm/h, amount).

  lagr    the last image moved along the motion (Lagrangian persistence), reference
Served shadow since 19.11.2026: rainy_server.inflow_forecast = "edgeeul" with taper 30 km, decay 150 km.

  docker run --rm -m 1g --cpus 1 -v deploy_verify-log:/vl:ro -v <dir with this file>:/w:ro -w /app deploy-api-eu      python /w/tune_inflow.py [--dir /vl] [--fc warnfc_eu] [--obs obs_eu] [--every 2] [--taper 30,60] [--decay 150]
(warnfc_eu: the served forecast's archive kept by warn_check.py's recorder since server-v1.58; ~0.3 MB per run)
"""
import os
import sys
import json
import time

import cv2
import numpy as np
from scipy import ndimage

args = sys.argv[1:]
opt = lambda n, d: args[args.index(n) + 1] if n in args else d
ROOT = opt("--dir", "/vl")
FC_DIR = os.path.join(ROOT, opt("--fc", "warnfc_eu"))
OBS_DIR = os.path.join(ROOT, opt("--obs", "obs_eu"))
EVERY = int(opt("--every", 1))
TAPERS = [float(v) for v in opt("--taper", "30").split(",")]
DECAY = float(opt("--decay", 150))
KM = 4.0
LEADS = (30, 60, 120)
REGIONS = {  # lat0, lat1, lon0, lon1
    "Galicia": (41.8, 43.9, -9.6, -6.7), "Portugal": (36.9, 41.8, -9.8, -6.2), "Ireland": (51.3, 55.5, -10.8, -5.8),
    "Scotland": (54.6, 59.5, -8.5, -0.5), "W Norway": (58.0, 63.5, 4.5, 8.0), "Brittany": (47.0, 49.0, -5.3, -1.5),
    "Iberia N coast": (42.8, 44.0, -6.7, -1.5), "Wales/SW Eng": (50.0, 53.5, -5.8, -2.5)}


def unq(q):
    q = np.asarray(q)
    return (np.exp2(q.astype(np.float32) / 32.0) - 1.0) if q.dtype == np.uint8 else q.astype(np.float32)


def tname(t):
    return time.strftime("%Y%m%d%H%M", time.gmtime(t / 1000))


obs_cache = {}


def obs(t):
    if t in obs_cache:
        return obs_cache[t]
    fp = os.path.join(OBS_DIR, tname(t) + ".npz")
    g = np.load(fp)["rate"].astype(np.float32) if os.path.exists(fp) else None
    if len(obs_cache) > 60:
        obs_cache.pop(next(iter(obs_cache)))
    obs_cache[t] = g
    return g


from pysteps import motion as ps_motion
from pysteps.utils import transformation as ps_tr
from pyproj import Transformer

meta = json.load(open(os.path.join(OBS_DIR, "grid.json")))
refs = sorted(int(f[:-4]) for f in os.listdir(FC_DIR) if f.endswith(".npz"))[::EVERY]
shape = None
for r in refs:
    g = obs(r)
    if g is not None:
        shape = g.shape
        break
ny, nx = shape
inv = Transformer.from_crs(meta["proj"], "EPSG:4326", always_xy=True)
gx, gy = np.meshgrid(np.arange(nx, dtype=np.float32), np.arange(ny, dtype=np.float32))
lon, lat = inv.transform((gx + 0.5) * meta["cell"], -(gy + 0.5) * meta["cell"])
reg = {k: (lat >= a) & (lat <= b) & (lon >= c) & (lon <= d) for k, (a, b, c, d) in REGIONS.items()}
print(f"grid {shape}, {len(refs)} runs {tname(refs[0])}..{tname(refs[-1])}, tapers {TAPERS}, decay {DECAY} km", flush=True)

thr = 0.1
zero = 10 * np.log10(thr) - 5.0
dbz = lambda gs: ps_tr.dB_transform(np.stack(gs), threshold=thr, zerovalue=zero)[0]

acc = {}


def add(name, L, region, f, o, m):
    a = acc.setdefault((name, L, region), np.zeros(9))
    for j, th in enumerate((0.1, 1.0)):
        wf, wo = (f >= th) & m, (o >= th) & m
        a[3 * j] += (wf & wo).sum(); a[3 * j + 1] += (~wf & wo).sum(); a[3 * j + 2] += (wf & ~wo).sum()
    a[6] += f[m].sum(); a[7] += o[m].sum(); a[8] += m.sum()


def extend(last, cov):
    """The last image beyond the coverage: nearest covered value of a 20 km smoothed copy, fading with the distance."""
    src = np.where(cov, last, 0).astype(np.float32)
    k = (0, 0)
    s = 20.0 / KM / 2
    num = cv2.GaussianBlur(src, k, s)
    den = cv2.GaussianBlur(cov.astype(np.float32), k, s)
    sm = np.where(den > 0.05, num / np.maximum(den, 1e-6), 0)
    edge = np.where(cov, sm, 0)
    d, (iy, ix) = ndimage.distance_transform_edt(~cov, return_indices=True)
    ext = np.where(cov, last, edge[iy, ix] * np.exp(-d * KM / DECAY)).astype(np.float32)
    return ext


n_done = 0
t0 = time.time()
wout_stats = {}
for ref in refs:
    z = np.load(os.path.join(FC_DIR, f"{ref}.npz"))
    frames = [obs(ref - 1800_000), obs(ref - 900_000), obs(ref)]
    if any(f is None for f in frames):
        continue
    last = frames[-1]
    cov = last >= 0
    if cov.sum() == 0:
        continue
    vel = ps_motion.get_method("LK")(dbz([np.where(g < 0, np.nan, g) for g in frames])) / 3.0
    u, v = np.nan_to_num(vel[0]).astype(np.float32), np.nan_to_num(vel[1]).astype(np.float32)
    ext = extend(last, cov)
    lastz = np.where(cov, last, 0).astype(np.float32)
    for L in LEADS:
        o = obs(ref + L * 60_000)
        if o is None or f"r{L}" not in z.files:
            continue
        n = L // 5
        mx, my = gx - u * n, gy - v * n
        covsrc = cv2.remap(cov.astype(np.float32), mx, my, cv2.INTER_LINEAR, borderMode=cv2.BORDER_CONSTANT, borderValue=0)
        edgef = cv2.remap(ext, mx, my, cv2.INTER_LINEAR, borderMode=cv2.BORDER_CONSTANT, borderValue=0)
        fc = unq(z[f"r{L}"])
        m = cov & (o >= 0)
        oo = np.where(o >= 0, o, 0)
        lagr = cv2.remap(lastz, mx, my, cv2.INTER_LINEAR, borderMode=cv2.BORDER_CONSTANT, borderValue=0)
        variants = {"served": fc, "pers": lastz, "lagr": lagr}
        for tp in TAPERS:
            s = tp / KM / 2
            wout = 1.0 - np.clip(cv2.GaussianBlur(covsrc, (0, 0), s), 0, 1)
            wout[~cov] = 0
            sfx = "" if len(TAPERS) == 1 else f"@{tp:g}"
            variants["euler" + sfx] = (1 - wout) * fc + wout * lastz
            variants["edge" + sfx] = (1 - wout) * fc + wout * edgef
            variants["edgemax" + sfx] = (1 - wout) * fc + wout * np.maximum(fc, edgef)
            variants["edgeeul" + sfx] = (1 - wout) * fc + wout * np.maximum(lastz, edgef)
            if tp == TAPERS[0]:
                for rk, rm in reg.items():
                    ws = wout_stats.setdefault((rk, L), [0.0, 0])
                    ws[0] += float(wout[rm & cov].sum()); ws[1] += int((rm & cov).sum())
                outside = wout > 0.05
        for name, f in variants.items():
            f = np.where(f >= 0.05, f, 0)
            add(name, L, "domain", f, oo, m)
            add(name, L, "inflow zone", f, oo, m & outside)
            for rk, rm in reg.items():
                add(name, L, rk, f, oo, m & rm)
    n_done += 1
    if n_done % 10 == 0:
        print(f"{n_done} runs, {time.time() - t0:.0f} s", flush=True)

print(f"\n{n_done} runs. per region and lead: CSI 0.1 / 1.0 mm/h cell by cell, amount fc / later analysis, "
      f"(mean outside weight of the region's covered cells, wet cells of the truth)")
names = list(dict.fromkeys(k[0] for k in acc))
for region in ["domain", "inflow zone"] + list(REGIONS):
    print(f"\n{region}")
    for L in LEADS:
        ws = wout_stats.get((region, L))
        a0 = acc.get(("served", L, region))
        if a0 is None:
            continue
        extra = f" wout {ws[0] / max(1, ws[1]):.2f}" if ws else ""
        print(f"  +{L}{extra}  obs mean {a0[7] / max(1, a0[8]):.3f} mm/h")
        for name in names:
            a = acc.get((name, L, region))
            if a is None:
                continue
            c1 = a[0] / max(1, a[0] + a[1] + a[2]); c2 = a[3] / max(1, a[3] + a[4] + a[5])
            print(f"    {name:12s} CSI {c1:.3f} {c2:.3f}  amount x{a[6] / max(1e-9, a[7]):.2f}")
