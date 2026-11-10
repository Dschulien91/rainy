"""Forecast variants compared on exactly the same runs (score_grid.py can mix periods after a restart).

  python score_same.py dwd own pers [--hours 12 | --day 20261110] [--strict]
  (--strict: cell by cell on the ~3 km grid instead of the ~9 km tolerance - rows since server-v0.93)
  python score_same.py own sprog pers --log eugrid [--area wo]   (the Europe test instance, fcverify.RegionVerify;
                                                                    area wo = WetterOnline's Europe view only)
"pers" = nothing moves (the image at issue time): the baseline any forecast must beat. Only runs and leads that every
named variant has are counted - so name a new shadow in a call of its own (daily_report.py does), or the comparison
shrinks to the hours since it exists.
"""
import glob
import json
import os
import sys
import time
from collections import defaultdict
from datetime import datetime, timezone

LOG_DIR = os.environ.get("VERIFY_LOG_DIR", os.path.join(os.path.dirname(os.path.abspath(__file__)), "verify_log"))
args = sys.argv[1:]
OPTS = ("--hours", "--log", "--area", "--day")
hours = float(args[args.index("--hours") + 1]) if "--hours" in args else 12
prefix = args[args.index("--log") + 1] if "--log" in args else "fcgrid"
area = args[args.index("--area") + 1] if "--area" in args else "all"
names = [a for i, a in enumerate(args) if not a.startswith("--") and (i == 0 or args[i - 1] not in OPTS)]
KEY = "st" if "--strict" in args else "fc"
if "--day" in args:  # one UTC day, by the time the forecast is for
    since = datetime.strptime(args[args.index("--day") + 1], "%Y%m%d").replace(tzinfo=timezone.utc).timestamp() * 1000
    until, span = since + 86400_000, args[args.index("--day") + 1] + " UTC"
else:
    since, until, span = (time.time() - hours * 3600) * 1000, float("inf"), f"last {hours:.0f} h"
rows = defaultdict(dict)  # (ref, lead) -> variant -> row
for fp in glob.glob(os.path.join(LOG_DIR, f"{prefix}_*.jsonl")):
    for line in open(fp):
        r = json.loads(line)
        if since <= r["valid"] < until and r.get("area", "all") == area and f"{KEY}0.1" in r:
            rows[(r["ref"], r["lead"])][r.get("variant", "dwd")] = r
            pk = "pst" if KEY == "st" else "pers"
            if f"{pk}0.1" in r:  # Germany: persistence is written into DWD's row
                rows[(r["ref"], r["lead"])]["pers"] = {f"{KEY}0.1": r[f"{pk}0.1"], f"{KEY}1.0": r[f"{pk}1.0"]}
present = {n for v in rows.values() for n in v}
missing = [n for n in names if n not in present]
names = [n for n in names if n in present]  # a variant without rows yet (new shadow) must not empty the comparison
if missing:
    print(f"(no data yet: {', '.join(missing)})")
common = {k: v for k, v in rows.items() if all(n in v for n in names)}
leads = sorted({lead for _, lead in common})
versions = sorted({r["v"] for v in common.values() for r in v.values() if r.get("v")})
print(f"{len(common)} checks, {len({r for r, _ in common})} runs, {span}   "
      f"CSI {'cell by cell' if KEY == 'st' else '(amount forecast/real)'}"
      + (f"   versions {', '.join(versions)}" if versions else ""))
if names and leads:  # how much rain there was at all: a dry day says nothing
    wet = [v[n].get("obs0.1") for (_, lead), v in common.items() if lead == leads[0] for n in names[:1]]
    wet = [w for w in wet if w is not None]
    if wet:
        print(f"wet cells in the truth per run (>= 0.1 mm/h): mean {sum(wet) / len(wet):.0f}")
for th in ("0.1", "1.0"):
    print(f">= {th} mm/h")
    for n in names:
        cells = []
        for lead in leads:
            h = m = f = fc = re = 0
            for (_, l), v in common.items():
                if l == lead:
                    a, b, c = v[n][f"{KEY}{th}"]
                    h += a; m += b; f += c
                    if "sum" in v[n] and KEY == "fc":
                        fc += v[n]["sum"][0]; re += v[n]["sum"][1]
            cells.append(f"+{lead}: {h / max(h + m + f, 1):.2f}" + (f" ({fc / max(re, 1):.2f})" if re else ""))
        print(f"  {n:7s} " + " | ".join(cells))
