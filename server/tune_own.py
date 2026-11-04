"""Offline tuning of the own nowcast (rainy_server.OwnNowcast) against DWD's RV forecast on the same cases: past DWD
runs (open data keeps ~2 days) are issued every STEP_ISSUE minutes over the last HOURS hours and scored at +15..+90
against the real analyses with the rules of fcverify.py (3 km, ~9 km tolerance, >= 0.1 mm/h).
  docker exec -w /app rainy-api python tune_own.py [HOURS]
"""
import sys
import numpy as np
import cv2
import rainy_server as rs
import mosaic
import fcverify

HOURS = float(sys.argv[1]) if len(sys.argv) > 1 else 3
STEP_ISSUE, LEADS = 30, (15, 30, 60, 90)
de = mosaic.GermanyDwd(rs.fetch, print)
runs = de._rv_runs()
ref = runs[-1]
need = [t for t in runs if t >= ref - (HOURS + 1.5) * 3600_000]
obs, dwd_fc = {}, {}
for t in need:  # analyses of every run, and the full DWD forecast of the issue runs
    got = de._load_rv(t)
    obs[t] = got[t]
    if (ref - t) % (STEP_ISSUE * 60_000) == 0:
        dwd_fc[t] = got
shape = (de.nx // fcverify.DOWN, de.ny // fcverify.DOWN)
mask = fcverify._down(de.inside, shape) > 0.5
VARIANTS = {  # current: thr 0.03, down 2, cdf
    "cdf down2": dict(down=2, thr=0.03),
    "mean down2": dict(down=2, thr=0.03, probmatch="mean"),
    "none down2": dict(down=2, thr=0.03, probmatch=None),
    "cdf down1": dict(down=1, thr=0.03),
}
THS = (0.1, 1.0)  # scoring thresholds: wet, real rain (mm/h)
issues = [t for t in dwd_fc if t - 30 * 60_000 in obs and t + 90 * 60_000 in obs]
tot = {}
for issue in issues:
    near = lambda dt: obs[issue - dt * 60_000].astype(np.float32)
    seq, mot = [near(10), near(5), near(0)], [near(30), near(15), near(0)]
    for name, v in VARIANTS.items():
        fc = rs.sprog_nowcast(seq, mot, 18, v["down"], thr=v["thr"], probmatch=v.get("probmatch", "cdf"))
        for lead in LEADS:
            a = fcverify._down(obs[issue + lead * 60_000], shape)
            g = fcverify._down(fc[lead // 5 - 1], shape)
            for th in THS:
                h, m, f = fcverify._counts(g >= th, a >= th, mask)
                t = tot.setdefault((th, name, lead), [0, 0, 0])
                t[0] += h; t[1] += m; t[2] += f
    for lead in LEADS:  # DWD's forecast and persistence on the same cases
        a = fcverify._down(obs[issue + lead * 60_000], shape)
        for name, g in (("DWD", dwd_fc[issue][issue + lead * 60_000]), ("persistence", obs[issue])):
            for th in THS:
                h, m, f = fcverify._counts(fcverify._down(g, shape) >= th, a >= th, mask)
                t = tot.setdefault((th, name, lead), [0, 0, 0]); t[0] += h; t[1] += m; t[2] += f
print(f"{len(issues)} cases   CSI (POD/FAR)")
for th in THS:
    print(f">= {th} mm/h")
    for name in ["DWD", "persistence"] + list(VARIANTS):
        print(f"  {name:13s} " + " | ".join(
            f"+{lead}: {h / max(h + m + f, 1):.2f} ({h / max(h + m, 1):.2f}/{f / max(h + f, 1):.2f})"
            for lead in LEADS for h, m, f in [tot.get((th, name, lead), [0, 0, 0])]))
