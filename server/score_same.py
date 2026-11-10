"""Forecast variants compared on exactly the same runs (score_grid.py can mix periods after a restart).

  python score_same.py dwd dwdnew own [--hours 12]
  python score_same.py own sprog pers --log eugrid [--area wo]   (the Europe test instance, fcverify.RegionVerify;
                                                                    area wo = WetterOnline's Europe view only)
"""
import glob
import json
import os
import sys
import time
from collections import defaultdict

LOG_DIR = os.environ.get("VERIFY_LOG_DIR", os.path.join(os.path.dirname(os.path.abspath(__file__)), "verify_log"))
args = sys.argv[1:]
hours = float(args[args.index("--hours") + 1]) if "--hours" in args else 12
prefix = args[args.index("--log") + 1] if "--log" in args else "fcgrid"
area = args[args.index("--area") + 1] if "--area" in args else "all"
names = [a for i, a in enumerate(args) if not a.startswith("--") and (i == 0 or args[i - 1] not in ("--hours", "--log", "--area"))]
since = (time.time() - hours * 3600) * 1000
rows = defaultdict(dict)  # (ref, lead) -> variant -> row
for fp in glob.glob(os.path.join(LOG_DIR, f"{prefix}_*.jsonl")):
    for line in open(fp):
        r = json.loads(line)
        if r["valid"] >= since and r.get("area", "all") == area:
            rows[(r["ref"], r["lead"])][r.get("variant", "dwd")] = r
present = {n for v in rows.values() for n in v}
missing = [n for n in names if n not in present]
names = [n for n in names if n in present]  # a variant without rows yet (new shadow) must not empty the comparison
if missing:
    print(f"(no data yet: {', '.join(missing)})")
common = {k: v for k, v in rows.items() if all(n in v for n in names)}
leads = sorted({lead for _, lead in common})
print(f"{len(common)} checks, {len({r for r, _ in common})} runs, last {hours:.0f} h   CSI (amount forecast/real)")
for th in ("0.1", "1.0"):
    print(f">= {th} mm/h")
    for n in names:
        cells = []
        for lead in leads:
            h = m = f = fc = re = 0
            for (_, l), v in common.items():
                if l == lead:
                    a, b, c = v[n][f"fc{th}"]
                    h += a; m += b; f += c
                    if "sum" in v[n]:
                        fc += v[n]["sum"][0]; re += v[n]["sum"][1]
            cells.append(f"+{lead}: {h / max(h + m + f, 1):.2f} ({fc / max(re, 1):.2f})")
        print(f"  {n:7s} " + " | ".join(cells))
