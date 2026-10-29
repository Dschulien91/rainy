"""How often would a rain alert have been right? Hit rates by lead time and intensity, from recorded forecasts.

For every recorded rain forecast (server/recorder.py) and DWD rain gauge that was dry when the forecast was issued,
the alert rule is applied to the gauge's forecast series: rain starts when the forecast is >= 0.3 mm/h for two
5-minute steps in a row. The alert counts as right when the gauge measures rain (>= 0.05 mm per 10 min) within
+-15 min of the predicted start. The hit rates per lead time are the alert probabilities shown in the push
messages (alerts.HIT_RATE); misses (rain that came without any predicted start) are counted too.

  python verify_alerts.py [--since 2026-10-28T20:35] [--rate 0.3]
"""
import argparse
import os
import sys
from datetime import datetime, timezone

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from score import load_truth, records  # noqa: E402

LEADS = [(5, 15), (15, 30), (30, 45), (45, 60), (60, 90), (90, 120)]
PEAKS = [(0.3, 1.0, "leicht"), (1.0, 4.0, "mäßig"), (4.0, 999, "stark")]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--since", default="2026-10-28T20:35")
    ap.add_argument("--rate", type=float, default=0.3)
    ap.add_argument("--truth", choices=("gauge", "radar"), default="gauge")
    a = ap.parse_args()
    since = datetime.fromisoformat(a.since).replace(tzinfo=timezone.utc).timestamp() * 1000
    truth = load_truth("rain", since - 3600_000)
    if a.truth == "radar":  # later radar observation at the station (what the map showed), 10-min bins like the gauges
        radar = {}
        for r in records("rain", since):
            for i, t in enumerate(r["ts"]):
                if not r["fc"][i]:
                    for j, sid in enumerate(r["ids"]):
                        k = (int(t) + 599_999) // 600_000 * 600_000
                        d = radar.setdefault(str(sid), {})
                        d[k] = max(d.get(k, 0.0), float(r["rate"][i, j]) / 6)  # mm/h -> mm per 10 min
        truth = radar
    acc = {}  # (lead bucket, peak class) -> [hits, total]
    miss = [0, 0]  # rain onsets at gauges that had no predicted start in the next 2 h / all onsets
    for r in records("rain", since):
        ts, issued = r["ts"].astype(np.int64), int(r["issued"])
        fc = r["fc"].astype(bool)
        for j, sid in enumerate(r["ids"]):
            obs = truth.get(str(sid))
            if not obs:
                continue
            # dry at issue time (last 20 min at the gauge)
            if any(obs.get(t, 0) >= 0.05 for t in range(issued // 600_000 * 600_000 - 1200_000, issued + 1, 600_000)):
                continue
            series = r["rate"][:, j]
            start = None
            for i in range(len(ts) - 1):
                if fc[i] and ts[i] > issued and series[i] >= a.rate and series[i + 1] >= a.rate:
                    start = i
                    break
            later = [t for t in sorted(obs) if issued < t <= issued + 120 * 60_000]
            if not later or later[-1] < issued + 120 * 60_000 - 600_000:
                continue  # truth not complete yet
            onset = next((t for t in later if obs[t] >= 0.05), None)
            if onset is not None:
                miss[1] += 1
                if start is None:
                    miss[0] += 1
            if start is None:
                continue
            lead = (ts[start] - issued) / 60_000
            b = next((k for k, (l0, l1) in enumerate(LEADS) if l0 <= lead < l1), None)
            if b is None:
                continue
            peak = float(series[start:start + 6].max())
            p = next(k for k, (v0, v1, _) in enumerate(PEAKS) if v0 <= peak < v1)
            hit = any(obs.get(t, 0) >= 0.05 for t in range(int(ts[start]) - 900_000, int(ts[start]) + 1500_000, 600_000))
            e = acc.setdefault((b, p), [0, 0])
            e[0] += hit
            e[1] += 1
    print(f"alerts that would have been sent, hit rate (n) by lead time; rain onsets without alert: {miss[0]}/{miss[1]}")
    print("lead      " + "  ".join(f"{name:>12s}" for _, _, name in PEAKS) + "      all")
    for b, (l0, l1) in enumerate(LEADS):
        cells, h, n = [], 0, 0
        for p in range(len(PEAKS)):
            e = acc.get((b, p), [0, 0])
            h, n = h + e[0], n + e[1]
            cells.append(f"{e[0] / e[1]:6.0%} ({e[1]:4d})" if e[1] else f"{'-':>12s}")
        print(f"{l0:3d}-{l1:3d}  " + "  ".join(cells) + (f"   {h / n:5.0%} ({n})" if n else ""))


if __name__ == "__main__":
    main()
