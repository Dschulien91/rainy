"""Jumps a user sees when moving the timeline, summed up from monitor.py's timeline rows (verify_log/timeline_*.jsonl).

Per product (rain: dwd = DWD's forecast, own = ours, eu = Europe; sun: de, eu) how much the map changes from one step
to the next - between measured images, at "now" (last measured image -> first forecast step) and inside the forecast -
and the single steps that stand out. Rain: share of the wet area that differs (only with enough rain, MIN_AREA cells);
sun: mean change of the displayed value (0 = overcast .. 1 = clear).
  break at "now":   the step at "now" changes clearly more than the steps between measured images
  jump:             a measured step far above the usual one (sunset method switch, a bad satellite or radar image)
  forecast step:    one lead whose step is always larger than the others (two methods handing over)

  python score_jumps.py [--hours 24 | --day 20261110] [--dir verify_log]
"""
import glob
import json
import os
import sys
import time
from collections import defaultdict
from datetime import datetime, timezone
from statistics import median

args = sys.argv[1:]
opt = lambda name, default: args[args.index(name) + 1] if name in args else default
log_dir = opt("--dir", os.environ.get("VERIFY_LOG_DIR", os.path.join(os.path.dirname(os.path.abspath(__file__)), "verify_log")))
hours = float(opt("--hours", 24))
since, until = (time.time() - hours * 3600) * 1000, float("inf")
if "--day" in args:
    since = datetime.strptime(opt("--day", ""), "%Y%m%d").replace(tzinfo=timezone.utc).timestamp() * 1000
    until = since + 86400_000
MIN_AREA = 500                           # rain: wet cells needed for a step to count
FLOOR = {"rain": 0.35, "sun": 0.08}      # a step below this is never called a jump
NAMES = {("rain", "dwd"): "Regen DWD-Vorhersage", ("rain", "own"): "Regen eigene Vorhersage", ("rain", "eu"): "Regen Europa",
         ("sun", "de"): "sunny Deutschland", ("sun", "eu"): "sunny Europa"}

runs = defaultdict(list)
for fp in glob.glob(os.path.join(log_dir, "timeline_*.jsonl")):
    for line in open(fp):
        r = json.loads(line)
        if since <= r["ref"] < until:
            runs[(r["kind"], r["product"])].append(r)
if not runs:
    print("(noch keine Zeitachsen-Daten)")
hm = lambda ms: datetime.fromtimestamp(ms / 1000, timezone.utc).strftime("%d.%m. %H:%MZ")
q = lambda v, p: sorted(v)[min(len(v) - 1, int(p * len(v)))]
for (kind, product), rs in sorted(runs.items()):
    ok = lambda r, i: (r["chg"][i] is not None and
                       (kind == "sun" or max(r["area"][i], r["area"][i - 1]) >= MIN_AREA))
    measured = {}                      # valid time of the later image -> (change, ref): the newest run counts
    now, ratio, fc = [], [], defaultdict(list)
    for r in sorted(rs, key=lambda r: r["ref"]):
        n = r["n_obs"]
        for i in range(1, len(r["chg"])):
            if not ok(r, i):
                continue
            if i < n:
                measured[r["ref"] + r["t"][i] * 60_000] = r["chg"][i]
            elif i == n:
                now.append((r["chg"][i], r["ref"]))
                a0, a1 = r["area"][i - 1], r["area"][i]
                if a0 is not None and a1 is not None:
                    ratio.append(a1 / max(a0, 1) if kind == "rain" else a1 - a0)
            else:
                fc[r["t"][i]].append(r["chg"][i])
    print(f"{NAMES.get((kind, product), kind + ' ' + product)}: {len(rs)} Läufe")
    if not measured or not now:
        print("  (zu wenig Regen bzw. Daten für eine Aussage)")
        continue
    m = median(measured.values())
    mn = median(c for c, _ in now)
    what = "Regenfläche x" if kind == "rain" else "sonniger Anteil "
    print(f"  gemessen -> gemessen:        median {m:.3f}, 90 % unter {q(list(measured.values()), 0.9):.3f}")
    print(f"  jetzt -> erster Vorhersageschritt: median {mn:.3f}, 90 % unter {q([c for c, _ in now], 0.9):.3f}"
          f"   ({what}{median(ratio):{'.2f' if kind == 'rain' else '+.3f'}})"
          + ("   <-- BRUCH bei 'jetzt'" if mn > 1.5 * m and mn > 0.5 * FLOOR[kind] else ""))
    med = {lead: median(v) for lead, v in fc.items() if len(v) >= 5}
    if med:
        allm = median(med.values())
        top = max(med, key=med.get)
        print(f"  Vorhersage Schritt zu Schritt: median {allm:.3f}, größter bei +{top} min ({med[top]:.3f})"
              + ("   <-- STUFE in der Vorhersage" if med[top] > 1.6 * allm and med[top] > 0.5 * FLOOR[kind] else ""))
    limit = max(3 * m, FLOOR[kind])
    odd = sorted(((c, t) for t, c in measured.items() if c > limit), reverse=True)
    if odd:
        print(f"  SPRÜNGE zwischen gemessenen Bildern (> {limit:.2f}): {len(odd)}, die größten: "
              + ", ".join(f"{hm(t)} {c:.2f}" for c, t in odd[:6]))
