"""Offline test of the rapid-update model blend (nwp_blend.py) on past runs, scored like the shadows.

Truth and input: the radar analyses every instance keeps at ~4 km (verify_log/obs_eu/, verify_log/na/obs_eu/, 14 days;
fcverify.RegionVerify). Models "as they were at run time": Open-Meteo's archive of past runs (data_run/, every 3rd
hourly run, with the time it became available). A case is a forecast run at ref = a run's availability time + 0, 20
or 40 min - the model is then exactly as old as the newest hourly run is in live use. The served forecast is rebuilt
on the 4 km grid (S-PROG on 4 km, local_amount, advect_blend: as in Region.update), then every variant is scored at
+30/+60/+90/+120 min against the later analysis, inside the radar coverage and the models' area (edge weight > 0.5):
CSI cell by cell at 0.1 / 1.0 mm/h, FSS at 20 km, wet area (>= 0.1) and amount forecast / real.

  docker run --rm -m 2000m --cpus 1.5 -e EDITION=eu [-e NA=1] -e VERIFY_LOG_DIR=/tmp/vl -e NWP_CACHE_DIR=/w/cache \
     -v /root/nwptools:/w -v deploy_verify-log:/vl:ro -w /app -e PYTHONPATH=/w:/app nwp-tools \
     python /w/tune_blend.py [max cases=40] [variant,variant,...] [--offsets 0,20,40] [--days 7]
(the nwp-tools image = deploy-api-eu + `pip install omfiles`; /root/nwptools holds nwp_blend.py, tune_blend.py)
"""
import os
import sys
import time
from collections import OrderedDict
from datetime import datetime, timedelta, timezone

import cv2
import numpy as np

import nwp_blend as nb
import rainy_server as rs

KEY = "na" if rs.NA else "eu"
REG = rs.REGIONS["eu"]
DOWN = 2                                    # the obs_eu grid = region grid / 2 (~4 km)
KM = REG.KM_PER_PX * DOWN
OBS_DIR = os.environ.get("OBS_DIR", "/vl/na/obs_eu" if KEY == "na" else "/vl/obs_eu")
args = sys.argv[1:]
opt = lambda name, d: args[args.index(name) + 1] if name in args else d
pos = [a for i, a in enumerate(args) if not a.startswith("--") and (i == 0 or not args[i - 1].startswith("--"))]
MAX_CASES = int(pos[0]) if pos else 40
OFFSETS = [int(v) for v in opt("--offsets", "0,20,40").split(",")]
DAYS = float(opt("--days", "7"))
LEADS = (30, 60, 90, 120)
THS = (0.1, 1.0)
FSS_N = max(1, int(round(20 / KM)))
D = dict(nb.DEFAULT, skill=False, start=45, end=120, wmax=0.4, trend=0, km=KM)  # the first test's base
ALL = OrderedDict([
    ("pers", None), ("served", None),
    # the corrected model alone (weight 1 inside its area): its own skill per lead
    ("model_raw", dict(D, start=0, end=0.01, wmax=1.0, cdf=False, pm=False)),
    ("model", dict(D, start=0, end=0.01, wmax=1.0, pm=False)),
    ("model_shift", dict(D, start=0, end=0.01, wmax=1.0, pm=False, shift=True)),
    # constant weights at every lead: which weight is best at which lead
    ("w0.2", dict(D, start=0, end=0.01, wmax=0.2)), ("w0.35", dict(D, start=0, end=0.01, wmax=0.35)),
    ("w0.5", dict(D, start=0, end=0.01, wmax=0.5)), ("w0.7", dict(D, start=0, end=0.01, wmax=0.7)),
    # schedules (weight 0 until start, linear to wmax at end)
    ("nwp", dict(D)),
    ("nwp_nopm", dict(D, pm=False)),
    ("nwp_raw", dict(D, cdf=False)),
    ("nwp_shift", dict(D, shift=True)),
    ("nwp_skill", dict(D, skill=True)),
    ("nwp_30_50", dict(D, start=30, end=120, wmax=0.5)),
    ("nwp_60_30", dict(D, start=60, end=120, wmax=0.3)),
    ("nwp_30_30", dict(D, start=30, end=120, wmax=0.3)),
    ("nwp_45_25", dict(D, start=45, end=120, wmax=0.25)),
    ("skill_60_30", dict(D, start=60, end=120, wmax=0.3, skill=True)),
    ("skill_30_40", dict(D, start=30, end=120, wmax=0.4, skill=True)),
    ("shskill_60_30", dict(D, start=60, end=120, wmax=0.3, skill=True, shift=True)),
    # growth / decay from the model (blend_step "trend"), alone and with the linear blend
    ("trend30", dict(D, wmax=0, skill=True, trend=30, tstart=30, tend=120, twmax=1.0)),
    ("trend30h", dict(D, wmax=0, skill=True, trend=30, tstart=30, tend=120, twmax=0.5)),
    ("trend60", dict(D, wmax=0, skill=True, trend=60, tstart=30, tend=120, twmax=1.0)),
    ("trend30+s", dict(D, start=60, end=120, wmax=0.3, skill=True, trend=30, tstart=30, tend=120, twmax=0.5)),
    ("live", dict(nb.REGION_DEFAULT[KEY], km=KM)),   # what the shadow "nwp" runs (nwp_blend.REGION_DEFAULT)
])
VARIANTS = OrderedDict((k, ALL[k]) for k in opt("--variants", ",".join(ALL)).split(",")) if "--variants" in args else ALL
if len(pos) > 1:
    VARIANTS = OrderedDict((k, ALL[k]) for k in ["pers", "served"] + [v for v in pos[1].split(",") if v not in ("pers", "served")])


def tname(t):
    return datetime.fromtimestamp(t / 1000, timezone.utc).strftime("%Y%m%d%H%M")


obs_times = sorted(int(datetime.strptime(f[:12], "%Y%m%d%H%M").replace(tzinfo=timezone.utc).timestamp() * 1000)
                   for f in os.listdir(OBS_DIR) if f.endswith(".npz"))
obs_set = set(obs_times)
cutoff = obs_times[-1] - DAYS * 86400_000
cache = OrderedDict()


def obs(t):
    if t not in obs_set:
        return None
    if t not in cache:
        cache[t] = np.load(os.path.join(OBS_DIR, tname(t) + ".npz"))["rate"].astype(np.float32)
        while len(cache) > 40:
            cache.popitem(last=False)
    return cache[t]


shape = obs(obs_times[-1]).shape
models = nb.REGION_MODELS[KEY]
t0 = time.time()
mappers = {m: nb.Mapper(nb.MODELS[m]["grid"], REG, DOWN, shape, KM) for m in models}
print(f"{KEY}: grid {shape}, {KM:.1f} km, mappers {time.time() - t0:.0f} s, obs {tname(obs_times[0])}..{tname(obs_times[-1])}",
      flush=True)
E_all = np.max([mp.edge for mp in mappers.values()], axis=0)

# the archived runs of every model, with the time they became available
runs = {}
day = datetime.fromtimestamp(max(obs_times[0], cutoff) / 1000, timezone.utc).replace(hour=0, minute=0, second=0, microsecond=0)
while day.timestamp() * 1000 <= obs_times[-1]:
    for m in models:
        for init, path in nb.archived_runs(m, day):
            runs.setdefault(m, []).append([init, path, None])
    day += timedelta(days=1)


def avail(m, r):
    if r[2] is None:
        import json
        p = os.path.join(nb.CACHE_DIR, r[1].strip("/").replace("/", "_") + "_meta.json")
        if not os.path.exists(p):
            os.makedirs(nb.CACHE_DIR, exist_ok=True)
            with open(p, "w") as f:
                f.write(nb._get(f"{nb.S3}/{r[1]}meta.json").decode())
        r[2] = nb._ms(json.load(open(p))["created_at"])
    return r[2]


# cases: ref = availability of the primary model's run + offset; each model's newest run must be "hourly fresh"
primary = models[0]
cases = []
for r in runs.get(primary, []):
    if r[0] < cutoff - 3 * 3600_000 or r[0] > obs_times[-1]:
        continue
    try:
        av = avail(primary, r)
    except Exception:
        continue
    for off in OFFSETS:
        ref = (av + off * 60_000 + 299_999) // 300_000 * 300_000
        need = [ref - k * 60_000 for k in (30, 15, 10, 5, 0)] + [ref + L * 60_000 for L in LEADS]
        if any(t not in obs_set for t in need):
            continue
        use = {}
        for m in models:
            best = None
            for rr in runs.get(m, []):
                if rr[0] <= ref and ref - rr[0] <= 4 * 3600_000:
                    try:
                        a = avail(m, rr)
                    except Exception:
                        continue
                    if a <= ref and ref - a <= 60 * 60_000 and (best is None or rr[0] > best[0]):
                        best = rr
            if best:
                use[m] = best
        if use:
            cases.append((ref, use))
# rain inside the models' area at ref or at +120
wet = []
for ref, use in cases:
    a, b = obs(ref), obs(ref + 120 * 60_000)
    m = (a >= 0) & (E_all > 0.5)
    wet.append(max(((a >= 0.1) & m).sum(), ((b >= 0.1) & m).sum()) / max(1, m.sum()))
cases = [c for c, w in zip(cases, wet) if w >= 0.01]
if len(cases) > MAX_CASES:
    cases = [cases[int(i)] for i in np.linspace(0, len(cases) - 1, MAX_CASES)]
print(f"{len(cases)} cases with rain in the models' area (of {len(wet)}); variants: {', '.join(VARIANTS)}", flush=True)

runcache = OrderedDict()


def runfield(m, r, ref):
    key = r[1]
    if key not in runcache:
        # the cases using this run have ref in [available, available + 60 min] and need ref .. ref + 120 min
        runcache[key] = nb.load_archived(m, r[1], mappers[m], r[2] - 15 * 60_000, r[2] + 200 * 60_000)
        while len(runcache) > 4:
            runcache.popitem(last=False)
    return runcache[key]


acc = {}
diags = {}


def score(name, lead, f, o, mask):
    a = acc.setdefault((name, lead), {"wet": [0, 0], "sum": [0.0, 0.0], **{f"csi{th}": [0, 0, 0] for th in THS},
                                        **{f"fss{th}": [0.0, 0.0] for th in THS}})
    a["wet"][0] += int(((f >= 0.1) & mask).sum()); a["wet"][1] += int(((o >= 0.1) & mask).sum())
    a["sum"][0] += float(f[mask].sum()); a["sum"][1] += float(o[mask].sum())
    for th in THS:
        wf, wo = (f >= th) & mask, (o >= th) & mask
        c = a[f"csi{th}"]
        c[0] += int((wf & wo).sum()); c[1] += int((~wf & wo).sum()); c[2] += int((wf & ~wo).sum())
        num, den = nb.fss(f, o, mask, th, FSS_N)
        a[f"fss{th}"][0] += num; a[f"fss{th}"][1] += den


done = 0
for ref, use in cases:
    t_case = time.time()
    last = obs(ref)
    seq = [obs(ref - 600_000), obs(ref - 300_000), last]
    mot = [obs(ref - 1800_000), obs(ref - 900_000), last]
    motion = {}
    try:
        pure = rs.sprog_nowcast(seq, mot, rs.FC_STEPS, 1, thr=REG.SPROG_THR, motion_out=motion)
        pure = rs.local_amount(pure, last, motion, KM, keep=False)
        fc = rs.advect_blend(pure, last, motion)
    except Exception as e:
        print("case", tname(ref), "S-PROG failed:", e)
        continue
    del pure
    rfs = [(runfield(m, r, ref), mappers[m]) for m, r in use.items()]
    prepared = {}
    for name, o in VARIANTS.items():
        if o is None:
            continue
        k = (o.get("cdf", True), o.get("shift", False), o.get("skill", False))
        if k not in prepared:
            prepared[k] = nb.prepare(rfs, ref, last, o, KM)
            for mm, d in prepared[k][2].items():
                diags.setdefault((mm, k), []).append(d)
    base = prepared.get((True, False, False)) or next(iter(prepared.values()))
    if base[0] is None:
        print("case", tname(ref), "no usable model run")
        continue
    Ecase = base[1]
    for L in LEADS:
        o = obs(ref + L * 60_000)
        g = np.asarray(fc[L // 5 - 1], np.float32)
        mask = (last >= 0) & (o >= 0) & (Ecase > 0.5)
        oo = np.where(o >= 0, o, 0)
        for name, op in VARIANTS.items():
            if name == "pers":
                f = np.where(last >= 0, last, 0)
            elif name == "served":
                f = np.where(g >= 0, g, 0)
            else:
                k = (op.get("cdf", True), op.get("shift", False), op.get("skill", False))
                model_at, E, _ = prepared[k]
                f = nb.blend_step(g, L, ref + L * 60_000, model_at, E, op)
                f = np.where(f >= 0, f, 0)
            score(name, L, f, oo, mask)
    done += 1
    ages = {m: round((ref - r[0]) / 60_000) for m, r in use.items()}
    print(f"case {tname(ref)} done ({time.time() - t_case:.0f} s), model age min {ages}, "
          f"fss0 {', '.join(f'{m} {d[m]["fss0"]}' for d in [base[2]] for m in d)}", flush=True)

print(f"\n{KEY}: {done} runs, inside radar coverage and the models' area; CSI cell by cell at 0.1 / 1.0 mm/h, "
      f"FSS {FSS_N * KM:.0f} km at 0.1 / 1.0, wet area (>= 0.1) and amount forecast / later radar")
for name in VARIANTS:
    for L in LEADS:
        a = acc.get((name, L))
        if not a:
            continue
        csi = " ".join("%.3f" % (c[0] / max(1, sum(c))) for c in (a[f"csi{th}"] for th in THS))
        fs = " ".join("%.3f" % (1 - n / d if d else 0) for n, d in (a[f"fss{th}"] for th in THS))
        print(f"  {name:12s} +{L:3d}  CSI {csi}  FSS {fs}  wet x{a['wet'][0] / max(1, a['wet'][1]):.2f}  "
              f"amount x{a['sum'][0] / max(1e-9, a['sum'][1]):.2f}")
for (m, k), ds in diags.items():
    if k == (True, False, False):
        print(f"  {m}: {len(ds)} runs, age {np.mean([d['age_min'] for d in ds]):.0f} min, FSS at ref (1 mm/h, 20 km, "
              f"corrected) {np.mean([d['fss0'] for d in ds]):.3f}, raw amount model/radar at ref x{np.mean([d['amount0'] for d in ds]):.2f}")
