"""North America: the MRMS picture and our forecast against the US rain gauges (METAR, hourly sums).

The North America instance (api-na) keeps (server v1.31):
  verify_log/na/metar/<day>.jsonl   each automated US station's hourly precipitation (northamerica.MetarGauges)
  verify_log/na/natobs/<ts>.npz     the picture at those stations, every 5-min image
  verify_log/na/fcgauge/<ref>.npz   every 10-min run's forecast at them: picture at ref, +30 / +60 / +90 / +120
A METAR hour ends at the report time (~:53) and starts at the previous routine report (taken as 60 min before).
  picture   mean of the images inside the hour (mm/h = mm in the hour)
  forecast  per lead: mean of the forecasts of the runs whose valid time falls inside the hour
  persist   the same runs' picture at their start (what the forecast must beat)
Since 16.11.2026: Canada's federal stations (ECCC SWOB, hourly, ids CA-...) and the weather model at all of them
(verify_log/na/modelgauge, every hour: "model +60 / +120" = its hour issued 1 / 2 h before the hour's end - the
baseline a forecast app without its own nowcast shows). Tables per country (USA, Canada) and all.
Scores per lead: amount ratio, CSI / POD / FAR at 0.25 / 1 / 3 mm per hour (0.25 mm = the gauges' 0.01 inch step).

  docker exec -w /app deploy-api-na-1 python check_na_gauges.py [--hours 24]
"""
import glob
import json
import os
import sys
import time

import numpy as np

import check_national_gauges as cng

HOURS = float(sys.argv[sys.argv.index("--hours") + 1]) if "--hours" in sys.argv else 24.0
LOG = cng.LOG_DIR
THS = (0.25, 1.0, 3.0)
with open(os.path.join(LOG, "metar", "stations.json")) as f:
    st = [tuple(s) for s in json.load(f)]
idx = {s[1]: k for k, s in enumerate(st)}
now = time.time() * 1000
since = now - HOURS * 3600_000
truth = []
for fp in glob.glob(os.path.join(LOG, "metar", "*.jsonl")):
    for line in open(fp):
        r = json.loads(line)
        if since <= r["t"] <= now - 30 * 60_000 and r["id"] in idx:
            truth.append((idx[r["id"]], r["t"], r["mm"]))
obs_files = {int(os.path.basename(f)[:-4]): f for f in glob.glob(os.path.join(cng.OBS_DIR, "*.npz"))}
fc_files = {int(os.path.basename(f)[:-4]): f for f in glob.glob(os.path.join(cng.FC_DIR, "*.npz"))}
obs_ts, fc_ts = np.array(sorted(obs_files)), np.array(sorted(fc_files))
cache = {}
model = {}   # issue ms -> (t, precip, {id: row})
for fp in glob.glob(os.path.join(LOG, "modelgauge", "*.npz")):
    name = os.path.basename(fp)[:-4]
    if name.isdigit() and int(name) >= since - 3 * 3600_000:
        z = np.load(fp)
        model[int(name)] = (z["t"], z["precip"].astype(np.float32), {str(x): i for i, x in enumerate(z["ids"])})


def model_at(k, t1, lead_h):
    """The model's precipitation in the hour ending t1, issued lead_h hours before t1 (to the full hour)."""
    end = (t1 + 1800_000) // 3600_000 * 3600_000  # METAR hours end ~:53, SWOB at :00
    m = model.get(end - lead_h * 3600_000)
    if m is None:
        return None
    t, pr, ids = m
    i = ids.get(f"{st[k][0]}:{st[k][1]}")
    j = np.flatnonzero(t == end)
    if i is None or not len(j) or not np.isfinite(pr[i, j[0]]):
        return None
    return float(pr[i, j[0]])


def load(path, key):
    if path not in cache:
        z = np.load(path)
        cache[path] = {k: z[k].astype(np.float32) for k in z.files}
    return cache[path][key]


print(f"{len(truth)} station hours from {len(st)} stations, {len(obs_ts)} images, {len(fc_ts)} forecast runs, last {HOURS:.0f} h")
rows = {}
for k, t1, mm in truth:
    c = st[k][0]
    t0 = t1 - 3600_000
    ins = obs_ts[(obs_ts > t0) & (obs_ts <= t1)]
    vals = [load(obs_files[t], "nat")[k] for t in ins if k < len(load(obs_files[t], "nat"))]
    if len(vals) >= 9 and np.isfinite(vals).all():
        rows.setdefault((c, "picture"), []).append((mm, float(np.mean(vals))))
    for li, lead in enumerate(cng.FC_LEADS):
        refs = fc_ts[(fc_ts + lead * 60_000 > t0) & (fc_ts + lead * 60_000 <= t1)]
        f, p = [], []
        for r in refs:
            a = load(fc_files[r], "fc")
            if k < a.shape[1] and np.isfinite(a[li + 1, k]) and np.isfinite(a[0, k]):
                f.append(a[li + 1, k]); p.append(a[0, k])
        if len(f) >= 4:
            rows.setdefault((c, f"forecast +{lead}"), []).append((mm, float(np.mean(f))))
            rows.setdefault((c, f"persist  +{lead}"), []).append((mm, float(np.mean(p))))
    # the weather model (HRRR & co. as Open-Meteo's best match; the baseline many apps show for the next hours):
    # its hour ending t1 from the issue 1 h / 2 h before (modelgauge, hourly since 16.11.2026)
    for lead_h in (1, 2):
        v = model_at(k, t1, lead_h)
        if v is not None:
            rows.setdefault((c, f"model    +{lead_h * 60}"), []).append((mm, v))
NAMES = {"us": "USA (METAR gauges)", "ca": "Canada (ECCC SWOB gauges)", "all": "All"}
LINES = ["picture"] + [n for lead in cng.FC_LEADS for n in (f"forecast +{lead}", f"persist  +{lead}", f"model    +{lead}")]
for c, name in ((c, name) for c in ("us", "ca", "all") for name in [None] + LINES):
    if name is None:
        print(f"\n{NAMES[c]}:")
        continue
    r = [x for (cc, nn), rr in rows.items() if nn == name and c in (cc, "all") for x in rr]
    if not r:
        continue
    g, q = np.array(r, np.float32).T
    cells = []
    for th in THS:
        h, m, fa = ((g >= th) & (q >= th)).sum(), ((g >= th) & (q < th)).sum(), ((g < th) & (q >= th)).sum()
        cells.append(f"{th}: CSI {h / max(1, h + m + fa):.2f} POD {h / max(1, h + m):.2f} FAR {fa / max(1, h + fa):.2f}"
                     if h + m + fa else f"{th}: nothing wet")
    ratio = f"x{q.sum() / g.sum():5.2f}" if g.sum() > 1 else "x   - "
    print(f"    {name:14s} {len(g):6d} hours  gauge mean {g.mean():.3f}  {ratio}   " + "   ".join(cells))
