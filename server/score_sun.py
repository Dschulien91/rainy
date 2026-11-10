"""Sunny forecast variants (server/sunny_de.py) compared on exactly the same runs, per lead, daytime only.

Grid (verify_log/sungrid_*.jsonl, against the later DWD satellite analysis, ~3 km, ~9 km tolerance, sunny = share >= 0.5):
CSI / POD / FAR, mean share bias and MAE. Stations (verify_log/sunstation_*.jsonl, against DWD's 10-minute sunshine
duration SD_10): MAE of the sun share and the hit rate of "sunny or not".

  python score_sun.py sat pers kpers model nwp [--hours 24 | --all] [--dir verify_log] [--prefix eu]
  (--prefix eu: the Europe instance's files eusungrid_* / eusunstation_*)
"""
import glob
import json
import os
import sys
import time
from collections import defaultdict

args = sys.argv[1:]
opt = lambda name, default: args[args.index(name) + 1] if name in args else default
hours = float(opt("--hours", 24))
log_dir = opt("--dir", os.environ.get("VERIFY_LOG_DIR", os.path.join(os.path.dirname(os.path.abspath(__file__)), "verify_log")))
prefix = opt("--prefix", "")
names = [a for i, a in enumerate(args) if not a.startswith("--") and (i == 0 or args[i - 1] not in ("--hours", "--dir", "--prefix"))] or ["sat", "pers"]
since = (time.time() - hours * 3600) * 1000 if "--all" not in args else 0


def load(kind):
    rows = defaultdict(dict)  # (ref, lead) -> variant -> row
    for fp in glob.glob(os.path.join(log_dir, f"{prefix}{kind}_*.jsonl")):
        for line in open(fp):
            r = json.loads(line)
            if r["valid"] >= since:
                rows[(r["ref"], r["lead"])][r["variant"]] = r
    have = [n for n in names if any(n in v for v in rows.values())]  # a new shadow without rows yet is left out
    if len(have) < len(names):
        print(f"({kind}: no data yet: {', '.join(n for n in names if n not in have)})")
    load.names = have
    return {k: v for k, v in rows.items() if all(n in v for n in have)}


grid = load("sungrid")
leads = sorted({lead for _, lead in grid})
print(f"grid: {len(grid)} checks, {len({r for r, _ in grid})} runs, last {hours:.0f} h   (sunny = share >= 0.5)")
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
