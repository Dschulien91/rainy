"""Where does the radar show rain at gauges that stay dry? Per gauge of one country: the share of the recorded
images in which OPERA alone is wet there, OPERA's and the own picture's mean rate, and what the gauge measured.

  docker exec -w /app deploy-api-eun-1 python check_dry_echoes.py [at|de|ch|it|iv|cz]

Reads the files check_national_gauges.record() wrote (verify_log/eun/natobs, last 4 days) - light. Single gauges that
are wet in every image while their neighbours are not = clutter; one connected area of light echo over dry gauges =
rain aloft that does not reach the ground (13.11.2026: all of Carinthia and Styria next to rain over Italy)."""
import glob
import json
import os
import sys

import numpy as np

import check_national_gauges as cng

country = sys.argv[1] if len(sys.argv) > 1 else "at"
st = [tuple(s) for s in json.load(open(os.path.join(cng.OBS_DIR, "stations.json")))]
files = sorted(glob.glob(os.path.join(cng.OBS_DIR, "*.npz")))
ts = [int(os.path.basename(f)[:-4]) for f in files]
idx = [k for k, s in enumerate(st) if s[0] == country]
op = np.full((len(files), len(st)), np.nan, np.float32)
na = np.full((len(files), len(st)), np.nan, np.float32)
for i, f in enumerate(files):
    z = np.load(f)
    n = len(z["opera"])
    op[i, :n], na[i, :n] = z["opera"], z["nat"]
truth = cng.GAUGES[country]([st[k][1] for k in idx], ts[0] - 1200_000, ts[-1] + 1200_000)
print(len(files), "images", (ts[-1] - ts[0]) / 3600_000, "h;", len(idx), "gauges")
rows = []
for k in idx:
    g = truth.get(st[k][1], {})
    gs = sum(v for t, v in g.items() if ts[0] <= t <= ts[-1] + 600_000)
    o = op[:, k]
    ok = np.isfinite(o)
    if ok.sum() < 5 or not g:
        continue
    rows.append((float((o[ok] >= 0.1).mean()), float(o[ok].mean()), float(np.nanmean(na[:, k])), gs, st[k][2], st[k][3], st[k][1]))
rows.sort(reverse=True)
wet = [r for r in rows if r[0] > 0]
print("gauges with any OPERA echo:", len(wet), "of", len(rows), "; gauges with rain measured:", sum(r[3] > 0 for r in rows))
print("share of images wet | OPERA mean mm/h | own mean | gauge sum mm | lat lon id")
for r in rows[:40]:
    print(f"{r[0]:5.2f}  {r[1]:6.3f}  {r[2]:6.3f}  {r[3]:5.1f}   {r[4]:.2f} {r[5]:.2f} {r[6]}")
tot = sum(r[1] for r in rows)
print("share of OPERA's amount at the 10 wettest-in-OPERA gauges:", sum(sorted((r[1] for r in rows), reverse=True)[:10]) / max(tot, 1e-9))
