"""Forecast variants compared on exactly the same runs (score_grid.py can mix periods after a restart).

  python score_same.py dwd dwdnew own [--hours 12]
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
names = [a for i, a in enumerate(args) if not a.startswith("--") and (i == 0 or args[i - 1] != "--hours")]
since = (time.time() - hours * 3600) * 1000
rows = defaultdict(dict)  # (ref, lead) -> variant -> row
for fp in glob.glob(os.path.join(LOG_DIR, "fcgrid_*.jsonl")):
    for line in open(fp):
        r = json.loads(line)
        if r["valid"] >= since:
            rows[(r["ref"], r["lead"])][r.get("variant", "dwd")] = r
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
