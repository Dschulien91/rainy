"""How far from the Swiss radars is the Swiss layer (ch) still right? Gauges of all countries where ch has data,
by distance to the nearest Swiss radar: gauge mean against ch alone, OPERA, the own picture and Italy's layer, and
how often ch is wet where the gauge is (POD) / dry where ch is wet (FAR). Basis of national.SwissRzc.RANGE_KM.

  docker exec -w /app deploy-api-eun-1 python check_swiss_range.py [hours=12]

Since server-v1.18 ch has no data beyond RANGE_KM, so the outer classes stay empty: to measure farther out again,
widen RANGE_KM for a day or note ch as a second, unlimited shadow layer."""
import glob
import json
import os
import sys
import time

import numpy as np

import check_national_gauges as cng
import national

HOURS = int(sys.argv[1]) if len(sys.argv) > 1 else 12
RADARS = national.SwissRzc.RADARS
st = [tuple(s) for s in json.load(open(os.path.join(cng.OBS_DIR, "stations.json")))]
since = time.time() * 1000 - HOURS * 3600_000
files = sorted(f for f in glob.glob(os.path.join(cng.OBS_DIR, "*.npz"))
               if int(os.path.basename(f)[:-4]) % 600_000 == 0 and int(os.path.basename(f)[:-4]) >= since)
ts = [int(os.path.basename(f)[:-4]) for f in files]
lat, lon = np.array([s[2] for s in st]), np.array([s[3] for s in st])
dist = np.min([np.hypot((lat - a) * 111.2, (lon - b) * 111.2 * np.cos(np.radians(a))) for a, b in RADARS], 0)
truth = {}
for c in sorted({s[0] for s in st}):
    truth[c] = cng.GAUGES[c]([s[1] for s in st if s[0] == c], ts[0] - 1200_000, ts[-1] + 1200_000)
rows = []  # country, distance, gauge, opera, own, ch, it, de
for f, t in zip(files, ts):
    z = np.load(f)
    if "each" not in z.files:
        continue
    layers = [str(x) for x in z["layers"]]
    each = z["each"].astype(np.float32)
    if "ch" not in layers:
        continue
    ch = each[layers.index("ch")]
    other = {n: each[layers.index(n)] for n in ("it", "de") if n in layers}
    for k in np.flatnonzero(np.isfinite(ch)):
        c, sid = st[k][0], st[k][1]
        q = truth[c].get(sid, {})
        v = [q.get(t), q.get(t + 600_000)]
        if None in v or not np.isfinite(z["opera"][k]):
            continue
        rows.append((c, dist[k], np.mean(v) * 6, float(z["opera"][k]), float(z["nat"][k]), float(ch[k]),
                     float(other.get("it", ch * np.nan)[k]), float(other.get("de", ch * np.nan)[k])))
C = np.array([r[0] for r in rows])
D, G, O, N, S, I, E = (np.array([r[i] for r in rows], np.float32) for i in range(1, 8))
print(len(rows), "pairs with the Swiss layer,", len(files), "images")
for name, sel in (("all countries", np.ones(len(C), bool)), ("Italy (it, iv, ie)", np.isin(C, ("it", "iv", "ie"))),
                  ("Switzerland", C == "ch"), ("Germany", C == "de"), ("Austria", C == "at")):
    print(f"\n{name}:  km from the nearest Swiss radar | pairs | gauge | ch | OPERA | own | it (where it has data: pairs, gauge, it)"
          " | ch: POD at gauge >= 0.3, FAR at ch >= 0.3")
    for lo, hi in ((0, 60), (60, 100), (100, 140), (140, 180), (180, 220), (220, 270), (270, 999)):
        m = sel & (D >= lo) & (D < hi)
        if m.sum() < 30:
            continue
        mi = m & np.isfinite(I)
        wet, sw = G[m] >= 0.3, S[m] >= 0.3
        pod = (wet & sw).sum() / max(1, wet.sum())
        far = (~wet & sw).sum() / max(1, sw.sum())
        print(f"  {lo:3d}-{hi:3d}  {m.sum():6d}  {G[m].mean():6.3f}  {S[m].mean():6.3f}  {O[m].mean():6.3f}  {N[m].mean():6.3f}"
              f"   ({mi.sum():5d}, {G[mi].mean() if mi.any() else 0:6.3f}, {I[mi].mean() if mi.any() else 0:6.3f})"
              f"   POD {pod:.2f} ({wet.sum()})  FAR {far:.2f} ({sw.sum()})")
