"""The two Europe instances' FORECASTS against rain gauges - "European" (OPERA) and the own picture, same runs, same
gauges, same truth. The grid scores (eugrid_*) measure each instance against its own later radar picture and cannot
compare them where the pictures differ (Italy: OPERA shows a tenth of the rain); this can.

Both instances note every forecast run on a 10-minute step at the gauges of check_national_gauges.py (since
server-v1.20): the picture at the run's time and the forecast for +30 / +60 / +90 / +120 min, in
verify_log/fcgauge/ (OPERA instance) and verify_log/eun/fcgauge/ (own picture), kept 8 days.

  docker exec -w /app deploy-api-eun-1 python check_forecast_gauges.py [--hours 24]

Per country (and Italy together, and all) and lead, on the runs both instances have:
  European     the OPERA instance's forecast
  own          the own picture's forecast
  Eur. persist / own persist: the picture at the run's time, unchanged (what a forecast must beat)
"x" = rain amount against the gauges, CSI / POD / FAR at 0.1 / 0.3 / 1 mm/h on 10-minute pairs (the forecast for the
time T against the gauge sums ending at T and T + 10 min, as in check_national_gauges.py) and on sums over full
hours ("hours": six runs' forecasts with the same lead, 0.2 / 1 / 3 mm - judge light rain there).
Read it as: own must beat European where a national layer counts (Italy, Germany, Switzerland) and equal it where
none does (Austria, Czechia); a forecast below its own persistence at +30 means the motion is wrong there.
"""
import glob
import json
import os
import sys
import time

import numpy as np

import check_national_gauges as cng

HOURS = float(sys.argv[sys.argv.index("--hours") + 1]) if "--hours" in sys.argv else 24.0
ROOT = os.path.dirname(cng.LOG_DIR) if os.path.basename(cng.LOG_DIR.rstrip("/")) == "eun" else cng.LOG_DIR
DIRS = {"European": os.path.join(ROOT, "fcgauge"), "own": os.path.join(ROOT, "eun", "fcgauge")}
with open(os.path.join(ROOT, "eun", "natobs", "stations.json")) as f:
    st = [tuple(s) for s in json.load(f)]
since = time.time() * 1000 - HOURS * 3600_000
refs = None
for d in DIRS.values():
    have = {int(os.path.basename(p)[:-4]) for p in glob.glob(os.path.join(d, "*.npz"))}
    refs = have if refs is None else refs & have
refs = sorted(t for t in refs if t >= since)
if not refs:
    sys.exit("no forecast run that both instances noted in this time")
print(f"{len(refs)} forecast runs of both instances, {time.strftime('%d.%m. %H:%M', time.gmtime(refs[0] / 1000))} - "
      f"{time.strftime('%d.%m. %H:%M', time.gmtime(refs[-1] / 1000))} UTC, {len(st)} gauges")
truth = {c: cng.GAUGES[c]([s[1] for s in st if s[0] == c], refs[0] - 1200_000, refs[-1] + 150 * 60_000)
         for c in sorted({s[0] for s in st})}
data = {name: {t: np.load(os.path.join(d, f"{t}.npz"))["fc"].astype(np.float32) for t in refs} for name, d in DIRS.items()}
rows = {}   # (country, lead) -> lists: gauge, European, own, Eur. persist, own persist, gauge index, valid time
for t in refs:
    a, b = data["European"][t], data["own"][t]
    n = min(a.shape[1], b.shape[1], len(st))
    for li, lead in enumerate(cng.FC_LEADS):
        T = t + lead * 60_000
        for k in range(n):
            q = truth[st[k][0]].get(st[k][1], {})
            v = [q.get(T), q.get(T + 600_000)]
            x = (a[li + 1, k], b[li + 1, k], a[0, k], b[0, k])
            if None in v or not np.isfinite(x).all():
                continue
            rows.setdefault((st[k][0], lead), []).append((np.mean(v) * 6, *x, k, T))


def hourly(r):
    """Six consecutive 10-minute rows of a gauge within a full hour -> their mean (= mm in that hour)."""
    groups = {}
    for x in r:
        groups.setdefault((x[5], x[6] // 3600_000), []).append(x[:5])
    return np.array([np.mean(g, 0) for g in groups.values() if len(g) == 6], np.float32).reshape(-1, 5)


LABELS = ("European", "own", "Eur. persist", "own persist")
groups = [(cng.NAMES[c], (c,)) for c in sorted({s[0] for s in st})]
groups += [("Italy (all regions)", ("it", "iv", "ie")), ("All gauges", tuple(sorted({s[0] for s in st})))]
for name, cs in groups:
    for lead in cng.FC_LEADS:
        r = [x for c in cs for x in rows.get((c, lead), [])]
        if not r:
            continue
        a = np.array([x[:5] for x in r], np.float32)
        cng.table(f"{name}, +{lead} min", a[:, 0], {l: a[:, i + 1] for i, l in enumerate(LABELS)})
        h = hourly(r)
        if len(h):
            cng.table(f"{name}, +{lead} min, hours", h[:, 0], {l: h[:, i + 1] for i, l in enumerate(LABELS)},
                      ths=(0.2, 1.0, 3.0), unit="mm")
