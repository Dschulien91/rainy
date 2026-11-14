"""Offline variants of the Europe nowcast on past runs: does the forecast keep the rain area?

For runs of the last hours (OPERA keeps 24 h): the S-PROG forecast as served (advection blend in the first hour)
with different S-PROG rain thresholds, against the real later images. Per lead: wet area (>= 0.05 mm/h, the map's
lowest class) forecast / real, rain amount forecast / real, CSI cell by cell at 0.05 / 0.3 / 1 mm/h; "pers" = the
last image unchanged. Run in the OPERA instance (two S-PROG runs per case and variant - not in the throttled one):

  docker exec -w /app deploy-api-eu-1 python tune_eu.py [runs=6] [hours between runs=1]

History: 14.11.2026 - the served forecast lost 30-40 % of the wet area in the first hour (tile_gaps.log) while the
measured trend was -9 %: S-PROG's threshold was 0.1 mm/h, the map draws from 0.05 (Germany's own forecast has used
0.03 since 04.11.).
"""
import sys
import time

import numpy as np

import rainy_server as rs

RUNS = int(sys.argv[1]) if len(sys.argv) > 1 else 6
EVERY = float(sys.argv[2]) if len(sys.argv) > 2 else 1.0
VARIANTS = {"thr 0.10 (old)": dict(thr=0.1), "thr 0.05": dict(thr=0.05), "thr 0.03": dict(thr=0.03)}
LEADS = (30, 60, 120)
THS = (0.05, 0.3, 1.0)
rs.EuropeOpera.DESPOKE = False  # raw images in any order: the filter needs them in sequence
eu = rs.REGIONS["eu"]
times = eu.list_times()
newest = max(times)
cache = {}


def img(t):
    if t not in cache:
        cache[t] = eu.load(times[t]).astype(np.float32) if t in times else None
    return cache[t]


acc = {}   # (variant, lead) -> [wet fc, wet real, sum fc, sum real, {th: [h, m, f]}]
done = 0
for k in range(RUNS):
    R = (newest - 125 * 60_000 - int(k * EVERY * 3600_000)) // rs.STEP_MS * rs.STEP_MS
    need = [R - m * 60_000 for m in (30, 15, 10, 5, 0)] + [R + m * 60_000 for m in LEADS]
    if any(img(t) is None for t in need):
        print("run", time.strftime("%H:%M", time.gmtime(R / 1000)), "skipped (images missing)")
        continue
    last = img(R)
    seq, mot = [img(R - 600_000), img(R - 300_000), last], [img(R - 1800_000), img(R - 900_000), last]
    for name, kw in [("pers", None)] + list(VARIANTS.items()):
        if kw is None:
            fc = {lead: last for lead in LEADS}
        else:  # one variant at a time, only the scored steps kept (three full forecasts at once ran out of memory)
            motion = {}
            pure = rs.sprog_nowcast(seq, mot, rs.FC_STEPS, eu.SPROG_DOWN, motion_out=motion, **kw)
            full = rs.advect_blend(pure, last, motion)
            fc = {lead: full[lead // 5 - 1] for lead in LEADS}
            del pure, full
        for lead in LEADS:
            f, o = np.asarray(fc[lead], np.float32), img(R + lead * 60_000)
            ok = (f >= 0) & (o >= 0)
            a = acc.setdefault((name, lead), [0, 0, 0.0, 0.0, {th: [0, 0, 0] for th in THS}])
            a[0] += int((f[ok] >= 0.05).sum()); a[1] += int((o[ok] >= 0.05).sum())
            a[2] += float(f[ok].sum()); a[3] += float(o[ok].sum())
            for th in THS:
                wf, wo = f[ok] >= th, o[ok] >= th
                a[4][th][0] += int((wf & wo).sum()); a[4][th][1] += int((~wf & wo).sum()); a[4][th][2] += int((wf & ~wo).sum())
    done += 1
    print("run", time.strftime("%H:%M", time.gmtime(R / 1000)), "done", flush=True)
    cache.clear()
print(f"\n{done} runs; wet area and amount = forecast / real later image; CSI cell by cell at {THS} mm/h")
for name in ["pers"] + list(VARIANTS):
    for lead in LEADS:
        a = acc.get((name, lead))
        if a:
            csi = " ".join("%.2f" % (h / max(1, h + m + f)) for h, m, f in (a[4][th] for th in THS))
            print(f"  {name:16s} +{lead:3d}  wet area x{a[0] / max(1, a[1]):.2f}  amount x{a[2] / max(1e-9, a[3]):.2f}  CSI {csi}")
