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

Since 17.11.2026 also 10-MINUTE tables at the gauges of the 10-minute archive (verify_log/na/gaugetruth,
minute_gauges.py: US ASOS 5-minute reports, Canadian SWOB minute records), as check_forecast_gauges.py in Europe:
the forecast for the time T (run + lead) against the gauge's mean rate over T-10 .. T+10 (two 10-minute sums),
CSI / POD / FAR at 0.1 / 0.3 / 1 mm/h, and on sums over full hours (six runs' forecasts with the same lead; 0.2 / 1 /
3 mm). Columns: forecast, persist (the run's picture at its time), later pic (the picture at T: the radar's own error
at the gauges - the forecast cannot beat it). US gauges count in 0.254 mm steps: one tip in the 20 minutes is
0.76 mm/h, so 0.1 and 0.3 mm/h both mean "it rained at all" there (Canada: 0.1 mm steps).

  docker exec -w /app deploy-api-na-1 python check_na_gauges.py [--hours 24] [--ten-only]
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
for k, t1, mm in (truth if "--ten-only" not in sys.argv else []):
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


# ---------- 10-minute truth (minute_gauges.py, since 17.11.2026) ----------
TEN = 600_000
t_lo = int(since) // 3600_000 * 3600_000 - 3600_000
t_hi = int(now) // TEN * TEN
nb = (t_hi - t_lo) // TEN + 1
tm = np.full((len(st), nb), np.nan, np.float32)    # mm in the 10 minutes ending t_lo + b * TEN
twx = np.full((len(st), nb), 255, np.uint8)        # present weather with precipitation (US; 255 unknown)
key_idx = {f"{s[0]}:{s[1]}": k for k, s in enumerate(st)}
for day in range(t_lo // 86400_000, t_hi // 86400_000 + 1):
    try:
        z = np.load(cng._day_file(day))
        ids, mm = [str(x) for x in z["ids"]], z["mm"].astype(np.float32)
        wx = z["wx"] if "wx" in z.files else np.full(mm.shape, 255, np.uint8)
    except (OSError, ValueError, KeyError):
        continue
    tt = day * 86400_000 + np.arange(144, dtype=np.int64) * TEN
    inside = (tt >= t_lo) & (tt <= t_hi)
    for i, key in enumerate(ids):
        k = key_idx.get(key)
        if k is None:
            continue
        ok = np.isfinite(mm[i]) & inside
        tm[k, (tt[ok] - t_lo) // TEN] = mm[i, ok]
        ok = (wx[i] != 255) & inside
        twx[k, (tt[ok] - t_lo) // TEN] = wx[i, ok]
ten = np.isfinite(tm).any(axis=1)
if not ten.any():
    print("\n(no 10-minute gauge truth in this time: verify_log/na/gaugetruth, minute_gauges.py)")
    sys.exit()
country = np.array([s[0] for s in st])
print(f"\n===== 10-minute gauges: {int(ten.sum())} gauges with 10-minute truth "
      f"({int((ten & (country == 'us')).sum())} USA ASOS, {int((ten & (country == 'ca')).sum())} Canada SWOB minute) =====")


def truth_at(T):
    """(mean rate over T-10 .. T+10 in mm/h, rain yes / no there: 1 / 0 / NaN) for all gauges. Rain = the gauge
    measured anything or (US) a report had precipitation as present weather; no rain = both sums 0 and no such
    report. The rate is NaN unless both 10-minute sums are known."""
    b = (T - t_lo) // TEN
    if b < 0 or b + 1 >= nb:
        nan = np.full(len(st), np.nan, np.float32)
        return nan, nan
    m, w = tm[:, b:b + 2], twx[:, b:b + 2]
    wet = (m > 0).any(axis=1) | (w == 1).any(axis=1)
    dry = (m == 0).all(axis=1) & ~(w == 1).any(axis=1)
    return m.sum(axis=1) * 3, np.where(wet, 1.0, np.where(dry, 0.0, np.nan)).astype(np.float32)


LABELS = ("forecast", "persist", "later pic")
ten_rows = {}   # lead -> [rows: gauge rate, rain yes/no, forecast, persist, later pic, gauge, hour of T]
for r in fc_ts:
    if r < since:
        continue
    a = load(fc_files[r], "fc")
    n = min(a.shape[1], len(st))
    sel = np.flatnonzero(ten[:n])
    for li, lead in enumerate(cng.FC_LEADS):
        T = int(r + lead * 60_000)
        g, yn = truth_at(T)
        later = load(obs_files[T], "nat")[:n][sel] if T in obs_files else np.full(len(sel), np.nan, np.float32)
        vals = np.stack([g[sel], yn[sel], a[li + 1, sel], a[0, sel], later]).astype(np.float64)
        ok = np.isfinite(vals[2:]).all(axis=0) & (np.isfinite(vals[0]) | np.isfinite(vals[1]))
        if ok.any():
            ten_rows.setdefault(lead, []).append(np.concatenate(
                [vals[:, ok], sel[ok][None].astype(np.float64), np.full((1, int(ok.sum())), T // 3600_000, np.float64)]))
pic = []        # the picture at every image time T against the gauges at T
for T in obs_ts:
    if T < since:
        continue
    v = load(obs_files[T], "nat")
    n = min(len(v), len(st))
    g, yn = truth_at(int(T))
    ok = (np.isfinite(g[:n]) | np.isfinite(yn[:n])) & np.isfinite(v[:n]) & ten[:n]
    if ok.any():
        pic.append(np.stack([g[:n][ok], yn[:n][ok], v[:n][ok], np.flatnonzero(ok)]).astype(np.float64))


def hours_of(a):
    """Rows (gauge, hour) with six runs, every rate known -> the mean of the six (= mm in that hour, same lead)."""
    a = a[:, np.isfinite(a[0])]
    key = a[-2].astype(np.int64) * 10_000_000 + a[-1].astype(np.int64)
    order = np.argsort(key, kind="stable")
    a, key = a[:, order], key[order]
    _, start, count = np.unique(key, return_index=True, return_counts=True)
    s = start[count == 6]
    return a[:-2][:, s[:, None] + np.arange(6)].mean(axis=2)


def yesno(title, wet, cols, th=0.1):
    """Rain yes / no (gauge or present weather) against "forecast >= th mm/h"."""
    ok = np.isfinite(wet)
    if not ok.any():
        return
    w = wet[ok] > 0.5
    print(f"\n{title}: {int(ok.sum())} pairs, rain at the gauge in {w.mean():.1%}")
    for label, q in cols.items():
        f = q[ok] >= th
        h, m, fa = (w & f).sum(), (w & ~f).sum(), (~w & f).sum()
        print(f"  {label:12s} rain >= {th} mm/h: CSI {h / max(1, h + m + fa):.2f} POD {h / max(1, h + m):.2f} "
              f"FAR {fa / max(1, h + fa):.2f}")


NAMES10 = {"us": "USA (ASOS 5-minute reports)", "ca": "Canada (SWOB minute records)", "all": "All North America"}
for c in ("us", "ca", "all"):
    def want(k, c=c):
        return np.ones(len(k), bool) if c == "all" else country[k.astype(int)] == c
    if pic:
        p = np.concatenate(pic, axis=1)
        p = p[:, want(p[3])]
        if p.shape[1]:
            r = np.isfinite(p[0])
            cng.table(f"{NAMES10[c]}, picture at the image time", p[0, r], {"picture": p[2, r]})
            yesno(f"{NAMES10[c]}, picture at the image time, rain yes / no", p[1], {"picture": p[2]})
    for lead in cng.FC_LEADS:
        if lead not in ten_rows:
            continue
        a = np.concatenate(ten_rows[lead], axis=1)
        a = a[:, want(a[5])]
        if not a.shape[1]:
            continue
        r = np.isfinite(a[0])
        cng.table(f"{NAMES10[c]}, +{lead} min", a[0, r], {lb: a[i + 2, r] for i, lb in enumerate(LABELS)})
        yesno(f"{NAMES10[c]}, +{lead} min, rain yes / no", a[1], {lb: a[i + 2] for i, lb in enumerate(LABELS)})
        h = hours_of(np.delete(a, 1, axis=0))
        if h.shape[1]:
            cng.table(f"{NAMES10[c]}, +{lead} min, hours", h[0], {lb: h[i + 1] for i, lb in enumerate(LABELS)},
                      ths=(0.2, 1.0, 3.0), unit="mm")
