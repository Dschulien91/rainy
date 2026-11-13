"""The two Europe instances' FORECASTS against rain gauges - "European" (OPERA) and the own picture, same runs, same
gauges, same truth. The grid scores (eugrid_*) measure each instance against its own later radar picture and cannot
compare them where the pictures differ (Italy: OPERA shows a tenth of the rain); this can.

Both instances note every forecast run on a 10-minute step at the gauges of check_national_gauges.py (since
server-v1.20): the picture at the run's time and the forecast for +30 / +60 / +90 / +120 min, in
verify_log/fcgauge/ (OPERA instance) and verify_log/eun/fcgauge/ (own picture), kept 14 days; since v1.22 also the
run's other forecasts where the instance has them. The gauges' values come from the server's archive
(verify_log/gaugetruth, hourly) plus what the sources have right now ("--archive": the archive alone, fast).

  docker exec -w /app deploy-api-eun-1 python check_forecast_gauges.py [--hours 24] [--archive]

Per country (and Italy together, and all) and lead, on the runs both instances have:
  European / own           the forecast the instance serves
  ... S-PROG               S-PROG alone (without the advected last image in the first hour)
  ... +model               S-PROG with the weather model's new showers (OPERA instance only)
  ... persist              the picture at the run's time, unchanged (what a forecast must beat)
"x" = rain amount against the gauges, CSI / POD / FAR at 0.1 / 0.3 / 1 mm/h on 10-minute pairs (the forecast for the
time T against the gauge sums ending at T and T + 10 min, as in check_national_gauges.py) and on sums over full
hours ("hours": six runs' forecasts with the same lead, 0.2 / 1 / 3 mm - judge light rain there).
Read it as: own must beat European where a national layer counts (Italy, Germany, Switzerland) and equal it where
none does (Austria, Czechia); a forecast below its own persistence at +30 means the motion is wrong there; a
forecast whose amount ("x") falls with the lead while persistence keeps it loses rain area (13.11.: -35 % in 1 h).
"""
import glob
import os
import sys
import time

import numpy as np

import check_national_gauges as cng

HOURS = float(sys.argv[sys.argv.index("--hours") + 1]) if "--hours" in sys.argv else 24.0
DIRS = {"European": os.path.join(cng.ROOT, "fcgauge"), "own": os.path.join(cng.ROOT, "eun", "fcgauge")}
EXTRA = {"pure": "S-PROG", "cells": "+model"}
st = cng.stations()
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
truth = cng.truth(st, refs[0] - 1200_000, refs[-1] + 150 * 60_000, live="--archive" not in sys.argv)
data = {name: {t: {k: v.astype(np.float32) for k, v in np.load(os.path.join(d, f"{t}.npz")).items()} for t in refs}
        for name, d in DIRS.items()}
# columns: per instance the served forecast, the variants every run of it has, its persistence
cols = []   # (label, instance, key or None for persistence)
for name in DIRS:
    cols.append((name, name, "fc"))
    cols += [(f"{name[:3]}. {EXTRA[k]}", name, k) for k in EXTRA if all(k in data[name][t] for t in refs)]
    cols.append((f"{name[:3]}. persist", name, None))
rows = {}   # (country, lead) -> [(gauge mm/h, one value per column ..., gauge index, valid time)]
n = min(len(st), *(z["fc"].shape[1] for inst in data.values() for z in inst.values()))
for t in refs:
    for li, lead in enumerate(cng.FC_LEADS):
        T = t + lead * 60_000
        vals = np.stack([data[inst][t]["fc"][0] if key is None else data[inst][t][key][li + (key == "fc")]
                         for _, inst, key in cols])[:, :n]
        ok = np.isfinite(vals).all(0)
        for k in np.flatnonzero(ok):
            q = truth.get(st[k][0], {}).get(st[k][1], {})
            v = [q.get(T), q.get(T + 600_000)]
            if None not in v:
                rows.setdefault((st[k][0], lead), []).append((np.mean(v) * 6, *vals[:, k], k, T))


def hourly(r):
    """Six consecutive 10-minute rows of a gauge within a full hour -> their mean (= mm in that hour)."""
    groups = {}
    for x in r:
        groups.setdefault((x[-2], x[-1] // 3600_000), []).append(x[:-2])
    return np.array([np.mean(g, 0) for g in groups.values() if len(g) == 6], np.float32).reshape(-1, len(cols) + 1)


countries = sorted({s[0] for s in st})
groups = [(cng.NAMES[c], (c,)) for c in countries]
groups += [("Italy (all regions)", ("it", "iv", "ie")), ("All gauges", tuple(countries))]
for name, cs in groups:
    for lead in cng.FC_LEADS:
        r = [x for c in cs for x in rows.get((c, lead), [])]
        if not r:
            continue
        a = np.array([x[:-2] for x in r], np.float32)
        cng.table(f"{name}, +{lead} min", a[:, 0], {c[0]: a[:, i + 1] for i, c in enumerate(cols)})
        h = hourly(r)
        if len(h):
            cng.table(f"{name}, +{lead} min, hours", h[:, 0], {c[0]: h[:, i + 1] for i, c in enumerate(cols)},
                      ths=(0.2, 1.0, 3.0), unit="mm")
