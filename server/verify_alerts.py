"""How often would a rain alert have been right? Hit rates by lead time and intensity, from recorded forecasts.

For every recorded rain forecast (server/recorder.py) and DWD rain gauge that was dry when the forecast was issued,
an alert rule is applied to the gauge's forecast series, and the alert counts as right when the gauge measures rain
(>= 0.05 mm per 10 min) within +-15 min of the predicted start. Misses (rain that came without any predicted start)
are counted too. Rules:
  point       rain starts when the forecast at the gauge is >= 0.3 mm/h for two 5-minute steps in a row (the only
              rule until 12.11.; NOT what the alerts do)
  circle      the real alert rule (alerts.evaluate): >= P_START of the 5 km circle wet for MIN_STEPS steps, DWD's
              forecast - from the recorder's "share" rows (since server-v0.98)
  circle-own  the same with our own forecast ("share_own")
Reliability (queue: alerts from a probability): for every lead, how often it really rained at the gauge depending on
the forecast wet share of the circle - the measured probability a trigger and the percentage in the push text can be
taken from (alerts.HIT_RATE).

  python verify_alerts.py [--since 2026-10-28T20:35 | --day 20261110] [--rate 0.3] [--truth radar]

10.11.: over 08.-10.11. the hit rates were 12-20 % against the gauges and 26-37 % against the radar - well below
alerts.HIT_RATE. 12.11.: those numbers were about half of the truth. The gauges report on full 10 minutes, the hit
window was built from the predicted start in 10-minute steps (start - 15 min, ...): for a start on a full 10 minutes
it never met a gauge time, so every such alert (half of all) counted as wrong (0 % of 4151 vs 47 % of 3842 for the
others, 08.-12.11.). The window is now the gauge intervals that overlap start +-15 min, for every start.
"""
import argparse
import os
import sys
from datetime import datetime, timezone

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from alerts import MIN_STEPS, P_START  # noqa: E402
from score import load_truth, records  # noqa: E402

LEADS = [(5, 15), (15, 30), (30, 45), (45, 60), (60, 90), (90, 120)]
PEAKS = [(0.3, 1.0, "leicht"), (1.0, 4.0, "mäßig"), (4.0, 999, "stark")]
TEN = 600_000
REL_LEADS = (15, 30, 45, 60, 90, 120)                     # reliability: forecast step closest to issue + lead
REL_BINS = (0.0, 0.1, 0.3, 0.5, 0.7, 0.9, 1.0001)         # forecast wet share of the circle


def window(t0):
    """End times of the gauge's 10-minute intervals (t - 10 min, t] that overlap t0 +- 15 min."""
    t0 = int(t0)
    return range((t0 - 900_000 + TEN - 1) // TEN * TEN, t0 + 1500_000, TEN)


def first_start(ok, fc, ts, issued, steps):
    """First forecast step after the issue time from which the rule holds for `steps` steps in a row."""
    for i in range(len(ts) - steps + 1):
        if fc[i] and ts[i] > issued and all(ok[i + k] for k in range(steps)):
            return i
    return None


def table(title, acc, miss):
    print(f"{title}; rain onsets without alert: {miss[0]}/{miss[1]}")
    print("lead      " + "  ".join(f"{name:>12s}" for _, _, name in PEAKS) + "      all")
    for b, (l0, l1) in enumerate(LEADS):
        cells, h, n = [], 0, 0
        for p in range(len(PEAKS)):
            e = acc.get((b, p), [0, 0])
            h, n = h + e[0], n + e[1]
            cells.append(f"{e[0] / e[1]:6.0%} ({e[1]:4d})" if e[1] else f"{'-':>12s}")
        print(f"{l0:3d}-{l1:3d}  " + "  ".join(cells) + (f"   {h / n:5.0%} ({n})" if n else ""))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--since", default="2026-10-28T20:35")
    ap.add_argument("--rate", type=float, default=0.3)
    ap.add_argument("--truth", choices=("gauge", "radar"), default="gauge")
    ap.add_argument("--day", help="only forecasts issued on this UTC day, e.g. 20261110 (daily_report.py)")
    a = ap.parse_args()
    since = datetime.fromisoformat(a.since).replace(tzinfo=timezone.utc).timestamp() * 1000
    if a.day:
        import score
        since = datetime.strptime(a.day, "%Y%m%d").replace(tzinfo=timezone.utc).timestamp() * 1000
        score.UNTIL = since + 86400_000
    truth = load_truth("rain", since - 3600_000)
    if a.truth == "radar":  # later radar observation at the station (what the map showed), 10-min bins like the gauges
        radar = {}
        for r in records("rain", since):
            for i, t in enumerate(r["ts"]):
                if not r["fc"][i]:
                    for j, sid in enumerate(r["ids"]):
                        k = (int(t) + TEN - 1) // TEN * TEN
                        d = radar.setdefault(str(sid), {})
                        d[k] = max(d.get(k, 0.0), float(r["rate"][i, j]) / 6)  # mm/h -> mm per 10 min
        truth = radar
    rules = ("point", "circle", "circle-own")
    acc = {k: {} for k in rules}        # rule -> (lead bucket, peak class) -> [hits, total]
    miss = {k: [0, 0] for k in rules}   # rain onsets at gauges without a predicted start in the next 2 h / all onsets
    rel = {}                            # (source, lead, share bin) -> [rain cases, all cases]
    for r in records("rain", since):
        ts, issued = r["ts"].astype(np.int64), int(r["issued"])
        fc = r["fc"].astype(bool)
        if len(ts) < 2:  # recorded while no radar data was loaded
            continue
        series = {"point": (r["rate"] >= a.rate, r["rate"], 2)}
        if "share" in r:
            series["circle"] = (r["share"] >= P_START, r["rate"], MIN_STEPS)
            if not np.isnan(r["share_own"]).all():
                series["circle-own"] = (r["share_own"] >= P_START, r["own"], MIN_STEPS)
        rel_at = [(lead, int(np.abs(ts - (issued + lead * 60_000)).argmin())) for lead in REL_LEADS]
        rel_at = [(lead, i) for lead, i in rel_at if fc[i] and i + 1 < len(ts) and abs(ts[i] - issued - lead * 60_000) <= 150_000]
        for j, sid in enumerate(r["ids"]):
            obs = truth.get(str(sid))
            if not obs:
                continue
            # dry at issue time (last 20 min at the gauge)
            if any(obs.get(t, 0) >= 0.05 for t in range(issued // TEN * TEN - 2 * TEN, issued + 1, TEN)):
                continue
            later = [t for t in sorted(obs) if issued < t <= issued + 120 * 60_000]
            if not later or later[-1] < issued + 120 * 60_000 - TEN:
                continue  # truth not complete yet
            onset = next((t for t in later if obs[t] >= 0.05), None)
            for rule, (ok, rate, steps) in series.items():
                start = first_start(ok[:, j], fc, ts, issued, steps)
                if onset is not None:
                    miss[rule][1] += 1
                    miss[rule][0] += start is None
                if start is None:
                    continue
                lead = (ts[start] - issued) / 60_000
                b = next((k for k, (l0, l1) in enumerate(LEADS) if l0 <= lead < l1), None)
                if b is None:
                    continue
                peak = max(float(np.nanmax(rate[start:start + 6, j])), PEAKS[0][0])
                p = next(k for k, (v0, v1, _) in enumerate(PEAKS) if v0 <= peak < v1)
                e = acc[rule].setdefault((b, p), [0, 0])
                e[0] += any(obs.get(t, 0) >= 0.05 for t in window(ts[start]))
                e[1] += 1
            for src in ("share", "share_own"):
                if src not in r:
                    continue
                for lead, i in rel_at:
                    v = min(float(r[src][i, j]), float(r[src][i + 1, j]))  # both steps, as the rule needs
                    if v != v:
                        continue
                    k = next(n for n in range(len(REL_BINS) - 1) if REL_BINS[n] <= v < REL_BINS[n + 1])
                    e = rel.setdefault((src, lead, k), [0, 0])
                    e[0] += any(obs.get(t, 0) >= 0.05 for t in window(ts[i]))
                    e[1] += 1
    table("point rule (forecast at the gauge), hit rate (n) by lead time", acc["point"], miss["point"])
    for rule, name in (("circle", "the real alert rule (5 km circle, DWD's forecast)"),
                       ("circle-own", "the real alert rule with our own forecast")):
        if acc[rule]:
            print()
            table(f"{name}, hit rate (n) by lead time", acc[rule], miss[rule])
    for src, name in (("share", "DWD's forecast"), ("share_own", "our own forecast")):
        if not any(k[0] == src for k in rel):
            continue
        print(f"\nreliability, {name}: how often it rained at the gauge (+-15 min) by forecast wet share of the circle  % (n)")
        print("share     " + "  ".join(f"{f'+{lead}':>12s}" for lead in REL_LEADS))
        for k in range(len(REL_BINS) - 1):
            cells = []
            for lead in REL_LEADS:
                e = rel.get((src, lead, k), [0, 0])
                cells.append(f"{e[0] / e[1]:5.0%} ({e[1]:5d})" if e[1] else f"{'-':>12s}")
            print(f"{REL_BINS[k]:.1f}-{min(REL_BINS[k + 1], 1.0):.1f}   " + "  ".join(cells))


if __name__ == "__main__":
    main()
