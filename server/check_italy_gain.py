"""Refit the damping of the Italian radar layer (national.ItalyDpc.damp) against rain gauges.

Hourly sums of the ARPA Veneto (iv) and ARPAE Emilia-Romagna (ie) gauges (readers of check_national_gauges.py)
against: OPERA as the Europe instance shows it, Radar-DPC SRI raw, SRI with the damping in use, flat gains, and a
grid of r * min(c, a r^b) applied to every 5-min image before summing. Needs rain there in the last hours. Run in
the OPERA instance (it loads OPERA images itself and so does not disturb the throttled third instance), with the
current files next to it:

  docker exec deploy-api-eu-1 mkdir -p /tmp/t
  for f in check_italy_gain.py check_national_gauges.py; do docker cp /opt/rainy/server/$f deploy-api-eu-1:/tmp/t/; done
  docker exec -e PYTHONPATH=/tmp/t:/app -w /app deploy-api-eu-1 python /tmp/t/check_italy_gain.py [hours=4] [every n-th gauge=2]
  (... check_italy_gain.py --load  repeats the tables on the pairs of the last run, kept in /tmp/italy_pairs.npz:
  G = gauge mm per hour, O = OPERA mean, S = the twelve raw SRI values of the hour, R = region)

Read it as: the damping should keep the amount near x1.0-1.2 and the CSI at 0.2 / 1 / 3 / 8 mm per hour as high as
possible in BOTH regions; the last table (raw SRI class -> what the gauges measured) shows where the product
overstates. Change DAMP_A / DAMP_B / DAMP_C only on that evidence (history: national.ItalyDpc).
"""
import io
import json
import sys
import time
import urllib.request

import numpy as np
from PIL import Image
from pyproj import Transformer

import rainy_server as rs
import check_national_gauges as cng

UA = {"User-Agent": rs.UA, "Content-Type": "application/json"}
API = "https://radar-api.protezionecivile.it"
PROJ = "+proj=tmerc +lat_0=42 +lon_0=12.5 +k=1 +x_0=0 +y_0=0 +ellps=WGS84 +units=m"
X0, Y1 = -600000.0, 650000.0
HOURS = int(sys.argv[1]) if len(sys.argv) > 1 and sys.argv[1].isdigit() else 4
EVERY = int(sys.argv[2]) if len(sys.argv) > 2 else 2
A, B, C = 0.2, 0.8, 0.7   # national.ItalyDpc.DAMP_A, DAMP_B, DAMP_C
THS = (0.2, 1.0, 3.0, 8.0)


def get(url, data=None):
    with urllib.request.urlopen(urllib.request.Request(url, data=data, headers=UA), timeout=60) as r:
        return r.read()


def sri(ts):
    u = json.loads(get(f"{API}/downloadProduct", json.dumps({"productType": "SRI", "productDate": ts}).encode()))
    return np.asarray(Image.open(io.BytesIO(get(u["url"]))), np.float32)


PAIRS = "/tmp/italy_pairs.npz"   # the collected pairs: "--load" as first argument refits on them without downloading
LOAD = len(sys.argv) > 1 and sys.argv[1] == "--load"
if LOAD:
    HOURS = 0
eu = rs.REGIONS["eu"]
st = [s for c in ("iv", "ie") for s in cng.LISTS[c](rs.fetch)[::EVERY]] if not LOAD else []
lon, lat = np.array([s[3] for s in st]), np.array([s[2] for s in st])
col, row = eu.to_pixels(*rs.mosaic.merc(lon, lat))
col, row = np.round(col).astype(int), np.round(row).astype(int)
x, y = Transformer.from_crs("EPSG:4326", PROJ, always_xy=True).transform(lon, lat)
ic, ir = ((x - X0) / 1000).astype(int), ((Y1 - y) / 1000).astype(int)

times = eu.list_times() if not LOAD else {}
T1 = int(time.time() * 1000 - 25 * 60_000) // 3600_000 * 3600_000
truth = {}
for c in ("iv", "ie"):  # {gauge id: {end of 10 min (ms): mm}}
    truth.update({(c, k): v for k, v in cng.GAUGES[c]([s[1] for s in st if s[0] == c], T1 - (HOURS + 1) * 3600_000, T1).items()})
gauge = [truth.get((s[0], s[1]), {}) for s in st]

G, O, S, R = [], [], [], []
for T in [T1 - k * 3600_000 for k in range(HOURS)]:
    # gauges and OPERA stamp the END of 5 minutes: T-55 .. T; SRI stamps the start: T-60 .. T-5
    steps = [T - k * 300_000 for k in range(12)]
    if any(t not in times for t in steps):
        print("OPERA incomplete for", time.strftime("%H:%M", time.gmtime(T / 1000)))
        continue
    op = np.zeros((12, len(st)), np.float32)
    sr = np.zeros((12, len(st)), np.float32)
    try:
        for k, t in enumerate(steps):
            g = eu.load(times[t])
            for j in range(len(st)):
                w = g[row[j] - 1:row[j] + 2, col[j] - 1:col[j] + 2]
                op[k, j] = w.mean() if (w >= 0).all() else np.nan
            a = sri(t - 300_000)
            for j in range(len(st)):
                w = a[ir[j] - 1:ir[j] + 2, ic[j] - 1:ic[j] + 2]
                sr[k, j] = w.mean() if (w >= 0).all() else np.nan
    except Exception as e:
        print("hour failed", e)
        continue
    for j in range(len(st)):
        v = [gauge[j].get(T - k * 600_000) for k in range(6)]
        if None in v or not np.isfinite(op[:, j]).all() or not np.isfinite(sr[:, j]).all():
            continue
        G.append(sum(v)); O.append(op[:, j].mean()); S.append(sr[:, j]); R.append(st[j][0])
    print("hour", time.strftime("%H:%M", time.gmtime(T / 1000)), "pairs so far", len(G), flush=True)
if LOAD:
    z = np.load(PAIRS)
    G, O, S, R = z["G"], z["O"], z["S"], z["R"]
else:
    G, O, S, R = np.array(G, np.float32), np.array(O, np.float32), np.array(S, np.float32), np.array(R)
    if not len(G):
        sys.exit("no pairs")
    np.savez_compressed(PAIRS, G=G, O=O, S=S, R=R)


def csis(r, m):
    out = []
    for th in THS:
        h, mi, f = ((G >= th) & (r >= th) & m).sum(), ((G >= th) & (r < th) & m).sum(), ((G < th) & (r >= th) & m).sum()
        out.append((h / max(1, h + mi + f), h / max(1, h + mi), f / max(1, h + f)))
    return out


def show(label, r, m):
    cells = "   ".join("%.1f: CSI %.2f POD %.2f FAR %.2f" % (th, *c) for th, c in zip(THS, csis(r, m)))
    print("  %-26s amount x%.2f mae %.2f   %s" % (label, r[m].sum() / G[m].sum(), np.abs(r - G)[m].mean(), cells))


damp = lambda a, b, c: (S * np.minimum(c, a * np.maximum(S, 1e-6) ** b)).mean(1)
everything = np.ones(len(G), bool)
res = []
for a in (0.15, 0.2, 0.25, 0.3, 0.4):
    for b in (0.3, 0.4, 0.5, 0.6, 0.8):
        for c in (0.5, 0.6, 0.7, 0.75, 0.85, 1.0):
            r = damp(a, b, c)
            # the amount right over all pairs and in neither region far off (13.11.: Veneto raw 2.8x, Emilia 1.44x)
            if 0.9 <= r.sum() / G.sum() <= 1.2 and all(not m.any() or 0.85 <= r[m].sum() / G[m].sum() <= 1.4
                                                         for m in (R == "iv", R == "ie")):
                res.append((sum(x[0] for m in (R == "iv", R == "ie") if m.any() for x in csis(r, m)), a, b, c))
best = sorted(res, reverse=True)[:6]
for name, m in (("Veneto + Emilia-Romagna", everything), ("Veneto", R == "iv"), ("Emilia-Romagna", R == "ie")):
    if not m.any():
        continue
    print("\n%s: %d pairs, gauge mean %.2f mm per hour" % (name, m.sum(), G[m].mean()))
    show("OPERA (as shown)", O, m)
    show("SRI raw", S.mean(1), m)
    show("SRI damped (in use)", damp(A, B, C), m)
    show("first fit: 0.25, 0.4, 1", damp(0.25, 0.4, 1.0), m)
    for g in (0.4, 0.5):
        show("SRI x%.1f" % g, g * S.mean(1), m)
    print("  grid r * min(c, a r^b): best by the sum of the CSI of both regions, amount x0.9-1.2 (per region x0.85-1.4):")
    for s, a, b, c in best:
        show("a %.2f b %.2f c %.2f" % (a, b, c), damp(a, b, c), m)
print("\nraw SRI class (hourly mean, mm/h) -> pairs | raw mean | gauge mean | gauge / raw | damped (in use)")
raw, use = S.mean(1), damp(A, B, C)
for lo, hi in ((0.2, 0.5), (0.5, 1), (1, 2), (2, 4), (4, 8), (8, 16), (16, 32), (32, 999)):
    m = (raw >= lo) & (raw < hi)
    if m.sum():
        print("  %5.1f-%5.1f  %5d   %6.2f   %6.2f   %5.2f   %6.2f" % (lo, hi, m.sum(), raw[m].mean(), G[m].mean(), G[m].mean() / raw[m].mean(), use[m].mean()))
