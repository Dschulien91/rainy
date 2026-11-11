"""Summary of bench_inca.py: our Europe rain forecast, GeoSphere Austria's INCA nowcast and "nothing moves", scored
at the Austrian rain gauges on exactly the same gauges and half hours.

Per half hour after the moment the forecasts were taken (+30 = the first half hour ... +120 = the fourth):
CSI (POD / FAR) for "wet" (>= 0.1 mm per 30 min) and "rain" (>= 0.5 mm per 30 min = 1 mm/h), the amount forecast/real,
and how old each forecast's data was when it was taken.

  python score_inca.py [--hours 24 | --day 20261111 | --all] [--dir verify_log]
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
log_dir = opt("--dir", os.environ.get("VERIFY_LOG_DIR", os.path.join(os.path.dirname(os.path.abspath(__file__)), "verify_log")))
hours = float(opt("--hours", 24))
since, until, span = (time.time() - hours * 3600) * 1000 if "--all" not in args else 0, float("inf"), f"last {hours:.0f} h"
if "--day" in args:
    since = datetime.strptime(opt("--day", ""), "%Y%m%d").replace(tzinfo=timezone.utc).timestamp() * 1000
    until, span = since + 86400_000, opt("--day", "") + " UTC"
rows = defaultdict(dict)  # (t0, lead) -> variant -> row
for fp in glob.glob(os.path.join(log_dir, "inca_*.jsonl")):
    for line in open(fp):
        r = json.loads(line)
        if since <= r["valid"] < until:
            rows[(r["t0"], r["lead"])][r["variant"]] = r
names = ("inca", "own", "pers")
common = {k: v for k, v in rows.items() if all(n in v for n in names)}
leads = sorted({lead for _, lead in common})
if not common:
    print("(noch keine Zeilen: der Vergleich braucht ~2,5 h vom Abruf bis zur Auswertung)")
    sys.exit()
first = [v for (_, lead), v in common.items() if lead == leads[0]]
print(f"{len(first)} Abrufe, {span}; im Mittel {sum(v['own']['n'] for v in first) / len(first):.0f} Regenmesser je Abruf; "
      f"Alter der Daten beim Abruf: eigene {sum(v['own']['age'] for v in first) / len(first):.0f} min, "
      f"INCA {sum(v['inca']['age'] for v in first) / len(first):.0f} min")
for key, label in (("wet", "nass >= 0.1 mm/30 min"), ("rain", "Regen >= 0.5 mm/30 min")):
    n_obs = sum(v["own"][key][0] + v["own"][key][1] for v in common.values())
    print(f"{label}   CSI (POD/FAR), Menge Vorhersage/real   [{n_obs} nasse Fälle]")
    for n in names:
        cells = []
        for lead in leads:
            h = m = f = fc = ob = 0
            for (_, l), v in common.items():
                if l == lead:
                    a, b, c = v[n][key]
                    h += a; m += b; f += c
                    fc += v[n]["sum"][0]; ob += v[n]["sum"][1]
            cells.append(f"+{lead}: {h / max(h + m + f, 1):.2f} ({h / max(h + m, 1):.2f}/{f / max(h + f, 1):.2f}) x{fc / max(ob, 0.01):.2f}")
        print(f"  {n:5s} " + " | ".join(cells))
