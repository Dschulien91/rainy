"""Refit the damping of the Italian radar layer (national.ItalyDpc.damp) against rain gauges.

ARPA Veneto gauges (hourly sums of their 5-min values) against: OPERA as the Europe instance shows it, Radar-DPC SRI
raw, SRI with the damping in use, flat gains, and a grid of r * min(1, a r^b) applied to every 5-min image before
summing. Needs rain over Veneto in the last hours. Run in the OPERA instance (it loads OPERA images itself and so
does not disturb the throttled third instance):

  docker cp /opt/rainy/server/check_italy_gain.py deploy-api-eu-1:/app/   (only if the image is older than the file)
  docker exec -e PYTHONPATH=/app -w /app deploy-api-eu-1 python check_italy_gain.py [hours=4] [every n-th gauge=2]

Read it as: the damping should keep the amount near x1.0-1.2 and the CSI at 0.2 / 1 / 3 mm/h as high as possible;
change DAMP_A / DAMP_B only when several days agree (13.11.2026: a 0.25, b 0.4 - see national.ItalyDpc).
"""
import io
import json
import sys
import time
import urllib.request
from datetime import datetime, timezone

import numpy as np
from PIL import Image
from pyproj import Transformer

import rainy_server as rs

UA = {"User-Agent": rs.UA, "Content-Type": "application/json"}
API = "https://radar-api.protezionecivile.it"
VEN = "https://api.arpa.veneto.it/REST/v1"
PROJ = "+proj=tmerc +lat_0=42 +lon_0=12.5 +k=1 +x_0=0 +y_0=0 +ellps=WGS84 +units=m"
X0, Y1 = -600000.0, 650000.0
HOURS = int(sys.argv[1]) if len(sys.argv) > 1 else 4
EVERY = int(sys.argv[2]) if len(sys.argv) > 2 else 2


def get(url, data=None):
    with urllib.request.urlopen(urllib.request.Request(url, data=data, headers=UA), timeout=60) as r:
        return r.read()


def sri(ts):
    u = json.loads(get(f"{API}/downloadProduct", json.dumps({"productType": "SRI", "productDate": ts}).encode()))
    return np.asarray(Image.open(io.BytesIO(get(u["url"]))), np.float32)


eu = rs.REGIONS["eu"]
st = json.loads(get(f"{VEN}/meteo_meteogrammi?rete=MGRAMMI&coordcd=23&orario=0"))["data"][::EVERY]
lon = np.array([float(s["longitudine"]) for s in st]); lat = np.array([float(s["latitudine"]) for s in st])
col, row = eu.to_pixels(*rs.mosaic.merc(lon, lat))
col, row = np.round(col).astype(int), np.round(row).astype(int)
x, y = Transformer.from_crs("EPSG:4326", PROJ, always_xy=True).transform(lon, lat)
ic, ir = ((x - X0) / 1000).astype(int), ((Y1 - y) / 1000).astype(int)

gauge = []
for s in st:
    mm = {}
    try:
        for r in json.loads(get(f"{VEN}/meteo_meteogrammi_tabella?codseqst={s['codseqst']}"))["data"]:
            if r["tipo"] == "PREC" and r["valore"] not in (None, ""):
                mm[int(datetime.fromisoformat(r["dataora"]).replace(tzinfo=timezone.utc).timestamp() * 1000) - 3600_000] = float(r["valore"])
    except Exception as e:
        print("station failed", s["codice_stazione"], e)
    gauge.append(mm)
    time.sleep(0.2)

times = eu.list_times()
T1 = int(time.time() * 1000 - 25 * 60_000) // 3600_000 * 3600_000
G, O, S = [], [], []
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
        v = [gauge[j].get(t) for t in steps]
        if None in v or not np.isfinite(op[:, j]).all() or not np.isfinite(sr[:, j]).all():
            continue
        G.append(sum(v)); O.append(op[:, j].mean()); S.append(sr[:, j])
    print("hour", time.strftime("%H:%M", time.gmtime(T / 1000)), "pairs so far", len(G), flush=True)
G, O, S = np.array(G, np.float32), np.array(O, np.float32), np.array(S, np.float32)


def show(label, r):
    cells = []
    for th in (0.2, 1.0, 3.0):
        h, m, f = ((G >= th) & (r >= th)).sum(), ((G >= th) & (r < th)).sum(), ((G < th) & (r >= th)).sum()
        cells.append("%.1f: CSI %.2f POD %.2f FAR %.2f" % (th, h / max(1, h + m + f), h / max(1, h + m), f / max(1, h + f)))
    print("  %-26s amount x%.2f corr %.2f mae %.2f   %s" % (label, r.sum() / G.sum(), np.corrcoef(G, r)[0, 1], np.abs(r - G).mean(), "   ".join(cells)))


print(len(G), "pairs, gauge mean %.2f mm/h" % G.mean())
if not len(G):
    sys.exit()
A, B = 0.25, 0.4   # national.ItalyDpc.DAMP_A, DAMP_B
damp = lambda a, b: (S * np.minimum(1, a * np.maximum(S, 1e-6) ** b)).mean(1)
show("OPERA (as shown)", O)
show("SRI raw", S.mean(1))
show("SRI damped (in use)", damp(A, B))
for g in (0.3, 0.4, 0.5):
    show("SRI x%.1f" % g, g * S.mean(1))
print("grid r * min(1, a r^b), best by the sum of the three CSI with the amount within x0.9-1.25:")
res = []
for a in (0.1, 0.15, 0.2, 0.25, 0.3, 0.4, 0.5):
    for b in (0.2, 0.3, 0.4, 0.5, 0.6, 0.75):
        r = damp(a, b)
        if 0.9 <= r.sum() / G.sum() <= 1.25:
            csi = []
            for th in (0.2, 1.0, 3.0):
                h, m, f = ((G >= th) & (r >= th)).sum(), ((G >= th) & (r < th)).sum(), ((G < th) & (r >= th)).sum()
                csi.append(h / max(1, h + m + f))
            res.append((sum(csi), a, b))
for s, a, b in sorted(res, reverse=True)[:6]:
    show("a %.2f b %.2f" % (a, b), damp(a, b))
