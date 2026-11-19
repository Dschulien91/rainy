"""Sunny forecast variants (server/sunny_de.py) compared on exactly the same runs, per lead, daytime only.

Grid (verify_log/sungrid_*.jsonl, against the later DWD satellite analysis, ~3 km, ~9 km tolerance, sunny = share >= 0.5):
CSI / POD / FAR, mean share bias and MAE. Stations (verify_log/sunstation_*.jsonl, against DWD's 10-minute sunshine
duration SD_10): MAE of the sun share and the hit rate of "sunny or not".

  python score_sun.py sat pers kpers model nwp [--hours 24 | --all | --day 20261110] [--dir verify_log] [--prefix eu] [--broken] [--by]
  (--by: MAE per station group, the country for --prefix eusy, rows from server-v1.60 on)
  (--prefix eusd: SYNOP hourly sunshine duration, ~340 stations in 16 countries, server-v1.62 on - the fair
   Europe-wide sunny truth; --prefix eusy = total cloud cover, a cloud truth: unfair to the sun share, see synop.HourlySun)
  (--broken: also the stations that were in broken cloud when the run started - compare "sat" with "sharp")
  (--prefix eu: the Europe instance's files eusungrid_* / eusunstation_*)
  (weather-model blend beyond +30 min, 17.11.2026: python score_sun.py sat satnwp nwp pers - "satnwp" is the shadow;
   with SUNNY_NWP=1 it is served as "sat" and the plain advection is "satadv"; offline test: tune_sun.py)
The last line is the analysis itself at the stations (variant "ana", lead 0): how well the newest satellite image
says "sun or not" where a gauge measures it - no forecast can be better than that.
"""
import glob
import json
import os
import sys
import time
from collections import defaultdict
from datetime import datetime, timezone

args = sys.argv[1:]
opt = lambda name, default: args[args.index(name) + 1] if name in args else default
hours = float(opt("--hours", 24))
log_dir = opt("--dir", os.environ.get("VERIFY_LOG_DIR", os.path.join(os.path.dirname(os.path.abspath(__file__)), "verify_log")))
prefix = opt("--prefix", "")
names = [a for i, a in enumerate(args) if not a.startswith("--") and (i == 0 or args[i - 1] not in ("--hours", "--dir", "--prefix", "--day"))] or ["sat", "pers"]
since, until, span = (time.time() - hours * 3600) * 1000 if "--all" not in args else 0, float("inf"), f"last {hours:.0f} h"
if "--day" in args:  # one UTC day, by the time the forecast is for
    since = datetime.strptime(opt("--day", ""), "%Y%m%d").replace(tzinfo=timezone.utc).timestamp() * 1000
    until, span = since + 86400_000, opt("--day", "") + " UTC"
ANA = []


def load(kind):
    rows = defaultdict(dict)  # (ref, lead) -> variant -> row
    for fp in glob.glob(os.path.join(log_dir, f"{prefix}{kind}_*.jsonl")):
        for line in open(fp):
            r = json.loads(line)
            if since <= r["valid"] < until:
                rows[(r["ref"], r["lead"])][r["variant"]] = r
                if r["variant"] == "ana":
                    ANA.append(r)
    have = [n for n in names if any(n in v for v in rows.values())]  # a new shadow without rows yet is left out
    if len(have) < len(names):
        print(f"({kind}: no data yet: {', '.join(n for n in names if n not in have)})")
    load.names = have
    return {k: v for k, v in rows.items() if all(n in v for n in have)}


grid = load("sungrid")
leads = sorted({lead for _, lead in grid})
print(f"grid: {len(grid)} checks, {len({r for r, _ in grid})} runs, {span}   (sunny = share >= 0.5)")
print("  variant   lead   CSI   POD   FAR  bias   MAE   sunny fc/obs")
for n in load.names:
    for lead in leads:
        h = m = f = b = e = w = fcn = obs = 0
        for (_, l), v in grid.items():
            if l == lead:
                r = v[n]
                h += r["fc"][0]; m += r["fc"][1]; f += r["fc"][2]
                b += r["bias"] * r["n"]; e += r["mae"] * r["n"]; w += r["n"]
                fcn += r["fcn"]; obs += r["obs"]
        print(f"  {n:8s} +{lead:3d}  {h / max(h + m + f, 1):.2f}  {h / max(h + m, 1):.2f}  {f / max(h + f, 1):.2f}"
              f" {b / max(w, 1):+.2f}  {e / max(w, 1):.2f}   {fcn / max(obs, 1):.2f}")
st = load("sunstation")
leads = sorted({lead for _, lead in st})
print(f"stations: {len(st)} checks, {len({r for r, _ in st})} runs   (share vs DWD SD_10)")
print("  variant   lead   MAE   hit  bias      n")
for n in load.names:
    for lead in leads:
        e = b = h = w = 0
        for (_, l), v in st.items():
            if l == lead:
                r = v[n]
                e += r["mae"] * r["n"]; b += r["bias"] * r["n"]; h += r["hit"] * r["n"]; w += r["n"]
        print(f"  {n:8s} +{lead:3d}  {e / max(w, 1):.3f} {h / max(w, 1):.3f} {b / max(w, 1):+.3f} {w:6d}")
if "--broken" in args:  # stations in broken cloud at the run's start (sunny_de.SunVerify.step: nb, mae_b, ...)
    print("stations in broken cloud at the start (analysis share 0.15-0.85):")
    print("  variant   lead   MAE   hit  bias      n")
    for n in load.names:
        for lead in leads:
            e = b = h = w = 0
            for (_, l), v in st.items():
                r = v[n]
                if l == lead and r.get("nb"):
                    e += r["mae_b"] * r["nb"]; b += r["bias_b"] * r["nb"]; h += r["hit_b"] * r["nb"]; w += r["nb"]
            if w:
                print(f"  {n:8s} +{lead:3d}  {e / w:.3f} {h / w:.3f} {b / w:+.3f} {w:6d}")
if "--by" in args:  # per group (the country for --prefix eusy; rows since server-v1.60 carry "by")
    groups = {}
    for (_, l), v in st.items():
        for n in load.names:
            for g, (k, ae, se, hi) in (v[n].get("by") or {}).items():
                a = groups.setdefault((g, n, l), [0, 0.0, 0.0, 0])
                a[0] += k; a[1] += ae; a[2] += se; a[3] += hi
    print("stations by group: MAE at +60 / +120 per variant (n = values at +120)")
    print("  group   " + "".join(f" {n:>13s}" for n in load.names) + "       n")
    for g in sorted({g for g, _, _ in groups}):
        cells = []
        for n in load.names:
            a, b = groups.get((g, n, 60)), groups.get((g, n, 120))
            cells.append(f"{a[1] / a[0]:.3f}/{b[1] / b[0]:.3f}" if a and b and a[0] and b[0] else "-")
        nn = max((groups.get((g, n, 120)) or [0])[0] for n in load.names)
        print(f"  {g:8s}" + "".join(f" {c:>13s}" for c in cells) + f"  {nn:6d}")
if ANA:
    w = sum(r["n"] for r in ANA)
    print(f"analysis at the stations (newest image vs gauge, {len(ANA)} images, {w} values): "
          f"MAE {sum(r['mae'] * r['n'] for r in ANA) / w:.3f}  hit {sum(r['hit'] * r['n'] for r in ANA) / w:.3f}  "
          f"bias {sum(r['bias'] * r['n'] for r in ANA) / w:+.3f}")
