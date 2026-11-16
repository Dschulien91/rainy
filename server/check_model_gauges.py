"""The weather model's 24 h forecast (the app's 24 h / 3-day views) against the rain gauges, and a simple correction.

Since server v1.30 the OPERA instance notes the forecast (Open-Meteo best match, hourly precipitation and
temperature, next 24 h) at every rain gauge of check_national_gauges every 3 hours: verify_log/modelgauge/<issue>.npz.
Open-Meteo's precipitation at hour t is the sum of (t - 1 h, t]; the truth is the gauges' 10-minute sums of that hour
(gauge archive). Per country and lead bucket: amount ratio, MAE, CSI at 0.2 / 1 mm per hour - raw, and corrected
with a per-country factor learned only from issues at least 24 h older (sum of gauges / sum of model, so a
correction is judged on hours it has not seen; shrunk to 1 with PRIOR mm).

  docker exec -w /app deploy-api-eu-1 python check_model_gauges.py [--days 7]

Read it as: where "corrected" beats "raw" at the same leads over several rainy days, a per-country (later per
station / per weather type) correction of the app's 24 h view is worth building (meteoblue / WetterOnline do this).
"""
import glob
import os
import sys
import time

import numpy as np

import check_national_gauges as cng
import synop

DAYS = float(sys.argv[sys.argv.index("--days") + 1]) if "--days" in sys.argv else 7.0
PRIOR = 5.0
THS = (0.2, 1.0)
BUCKETS = ((1, 1), (2, 2), (3, 3), (4, 6), (7, 12), (13, 24))  # +1 h / +2 h: next to the nowcast's +60 / +120 hours
now = time.time() * 1000
files = sorted(f for f in glob.glob(os.path.join(cng.ROOT, "modelgauge", "*.npz")) if os.path.basename(f)[:-4].isdigit()
               and int(os.path.basename(f)[:-4]) > now - DAYS * 86400_000)
if not files:
    sys.exit("no model forecasts noted yet")
issues = [int(os.path.basename(f)[:-4]) for f in files]
st = cng.stations()
truth = cng.truth(st, issues[0], now, live=False)
rows = []   # (issue, lead h, country, gauge mm, model mm)
for f, issue in zip(files, issues):
    z = np.load(f)
    t, pr = z["t"], z["precip"].astype(np.float32)
    ids = [str(x) for x in z["ids"]]
    for k, key in enumerate(ids[:pr.shape[0]]):
        c, sid = key.split(":", 1)
        g = truth.get(c, {}).get(sid)
        if not g:
            continue
        if c == "sy":  # SYNOP stations: per country (as in check_forecast_gauges.py)
            c = f"sy-{sid.split('-', 1)[0]}"
        for i, end in enumerate(t):
            lead = int(round((end - issue) / 3600_000))
            if not 1 <= lead <= 24 or not np.isfinite(pr[k, i]):
                continue
            v = [g.get(int(end) - m * 600_000) for m in range(6)]
            if None in v:
                continue
            rows.append((issue, lead, c, float(sum(v)), float(pr[k, i])))
if not rows:
    sys.exit("no pairs yet (the gauge archive must cover the forecast hours)")
R = np.array([(r[0], r[1], r[3], r[4]) for r in rows], np.float64)
C = np.array([r[2] for r in rows])
print(f"{len(files)} issues, {len(rows)} gauge-hours, {len(set(C))} networks, last {DAYS:.0f} days")
# the correction: per country, from issues at least 24 h before this one
corr = np.ones(len(R))
for c in set(C):
    m = C == c
    for issue in sorted(set(R[m, 0])):
        past = m & (R[:, 0] <= issue - 86400_000)
        f = (R[past, 2].sum() + PRIOR) / (R[past, 3].sum() + PRIOR) if past.any() else 1.0
        corr[m & (R[:, 0] == issue)] = np.clip(f, 0.33, 3.0)


def line(g, f):
    cells = []
    for th in THS:
        h, mi, fa = ((g >= th) & (f >= th)).sum(), ((g >= th) & (f < th)).sum(), ((g < th) & (f >= th)).sum()
        cells.append(f"{th}: CSI {h / max(1, h + mi + fa):.2f} POD {h / max(1, h + mi):.2f} FAR {fa / max(1, h + fa):.2f}")
    ratio = f"x{f.sum() / g.sum():5.2f}" if g.sum() > 1 else "x   - "
    return f"{ratio}  mae {np.abs(f - g).mean():.3f}   " + "   ".join(cells)


for name in sorted(set(C)) + ["all"]:
    sel = np.ones(len(R), bool) if name == "all" else C == name
    if R[sel, 2].sum() < 1:
        continue
    label = f"{synop.NAMES.get(name[3:], name[3:])} (SYNOP)" if name.startswith("sy-") else cng.NAMES.get(name, name)
    print(f"\n{label}: {int(sel.sum())} gauge-hours, gauge mean {R[sel, 2].mean():.3f} mm")
    for lo, hi in BUCKETS:
        m = sel & (R[:, 1] >= lo) & (R[:, 1] <= hi)
        if m.sum() < 50:
            continue
        g, f = R[m, 2], R[m, 3]
        print(f"  +{lo:2d}-{hi:2d} h raw        {line(g, f)}")
        print(f"  +{lo:2d}-{hi:2d} h corrected  {line(g, f * corr[m])}")
