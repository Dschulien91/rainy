"""Summary of server/fcverify.py: forecast vs later DWD radar image on the full grid, per lead, against persistence.

  python score_grid.py [--hours 48]
"""
import argparse
import glob
import json
import os
import time
from collections import defaultdict

LOG_DIR = os.environ.get("VERIFY_LOG_DIR", os.path.join(os.path.dirname(os.path.abspath(__file__)), "verify_log"))


def csi(h, m, f):
    return h / max(h + m + f, 1), h / max(h + m, 1), f / max(h + f, 1)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--hours", type=float, default=48)
    a = ap.parse_args()
    since = (time.time() - a.hours * 3600) * 1000
    tot = defaultdict(lambda: [0, 0, 0])
    runs = defaultdict(set)
    for fp in sorted(glob.glob(os.path.join(LOG_DIR, "fcgrid_*.jsonl"))):
        for line in open(fp):
            r = json.loads(line)
            if r["valid"] < since:
                continue
            for th in ("0.1", "1.0"):
                for kind in ("fc", "pers"):
                    v = r.get(f"{kind}{th}")
                    if v:
                        name = "pers" if kind == "pers" else r.get("variant", "dwd")
                        t = tot[(th, name, r["lead"])]
                        t[0] += v[0]; t[1] += v[1]; t[2] += v[2]
                runs[(r.get("variant", "dwd"), r["lead"])].add(r["ref"])
    leads = sorted({lead for _, lead in runs})
    if not leads:
        print("no verified forecasts yet")
        return
    print(f"Vorhersage vs. spaeteres DWD-Radarbild (~9 km Toleranz), letzte {a.hours:.0f} h   CSI (POD/FAR)")
    for th, name in (("0.1", "nass >= 0.1 mm/h"), ("1.0", "Regen >= 1 mm/h")):
        print(f"  {name}")
        for kind, label in (("dwd", "DWD    "), ("own", "eigene "), ("pers", "bleibt ")):
            if not any(tot[(th, kind, lead)] != [0, 0, 0] for lead in leads):
                continue
            cells = []
            for lead in leads:
                c, p, f = csi(*tot[(th, kind, lead)])
                cells.append(f"+{lead}: {c:.2f} ({p:.2f}/{f:.2f})")
            print(f"    {label} " + " | ".join(cells))
    for v in ("dwd", "own"):
        if any(runs[(v, l)] for l in leads):
            print(f"  Laeufe {v}: " + ", ".join(f"+{l}: {len(runs[(v, l)])}" for l in leads))


if __name__ == "__main__":
    main()
