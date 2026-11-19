"""Offline variants for stationary orographic rain in the Europe nowcast (19.11.2026).

Problem (gauges, 17.-19.11.): S-PROG and local_amount move the rain that is anchored at the windward slopes across the
Alpine ridge - forecast / later picture at +120 in Austria south / west x2.0, Ticino x2.6, Italian Alps x1.75, and in
Switzerland the forecast falls below persistence at 1 / 3 mm. Variants on top of the served forecast before the model
blend ("own0" = S-PROG + local_amount + advect_blend):
  oro<W>_<tau>[_all]  the quasi-stationary part S = minimum of the images of the last W minutes (every 15 min, lightly
                      smoothed), over mountains only (terrain mask; "_all": everywhere), is held in place and decays with
                      exp(-lead / tau); the forecast loses the share of S that the motion carries along (q = moved S /
                      moved last image, 16 km windows)
  lee<W0>             rain shadow: the forecast is damped where the motion goes down a slope, exp(min(0, climb) / W0),
                      climb = metres per 5 min along the motion over the smoothed terrain
Truth: the later OPERA image cell by cell (2 km); per region the amount forecast / later image and CSI at 0.1 / 1 / 3
mm/h; "pers" = the last image unchanged. One S-PROG run per case, all variants from it.

  docker run --rm -m 1800m --cpus 1.5 -e EDITION=eu deploy-api-eu python tune_oro.py [runs=10] [hours between=2] [dem]
"""
import math
import sys
import time

import cv2
import numpy as np

import rainy_server as rs

RUNS = int(sys.argv[1]) if len(sys.argv) > 1 else 10
EVERY = float(sys.argv[2]) if len(sys.argv) > 2 else 2.0
DEM = sys.argv[3] if len(sys.argv) > 3 else "dem_eu4km.npz"
LEADS = (15, 30, 60, 90, 120)
THS = (0.1, 1.0, 3.0)
REGIONS = {"AT south": (12.3, 15.0, 46.4, 47.1), "AT west": (9.5, 12.5, 46.8, 47.6), "AT north": (13.0, 16.5, 47.6, 48.8),
           "CH south": (7.0, 9.3, 45.8, 46.5), "CH north": (7.0, 9.6, 46.8, 47.8), "Ital Alps": (9.3, 12.5, 45.8, 46.8),
           "Slovenia": (13.4, 16.6, 45.4, 46.9), "NL coast": (3.3, 5.0, 51.3, 53.5), "Germany": (5.9, 15.0, 47.3, 55.0),
           "Norway W": (4.5, 8.0, 58.0, 63.0), "domain": None}
VARIANTS = {"oro30_120": dict(w=30, tau=120), "oro60_120": dict(w=60, tau=120), "oro60_240": dict(w=60, tau=240),
            "oro60_inf": dict(w=60, tau=1e9), "oro30_240": dict(w=30, tau=240), "oro60_120_all": dict(w=60, tau=120, mask=False),
            "lee150": dict(lee=150), "lee300": dict(lee=300), "oro60_120+lee300": dict(w=60, tau=120, lee=300)}
if len(sys.argv) > 4:
    VARIANTS = {k: VARIANTS[k] for k in sys.argv[4].split(",")}

rs.EuropeOpera.DESPOKE = False
eu = rs.REGIONS["eu"]
times = eu.list_times()
newest = max(times)
cache = {}


def img(t):
    if t not in cache:
        cache[t] = eu.load(times[t]).astype(np.float32) if t in times else None
    return cache[t]


def region_masks(shape):
    rows, cols = np.mgrid[0:shape[0], 0:shape[1]]
    lon, lat = eu.pixel_lonlat(cols.ravel(), rows.ravel())
    lon, lat = np.asarray(lon).reshape(shape), np.asarray(lat).reshape(shape)
    return {n: (np.ones(shape, bool) if b is None else (lon >= b[0]) & (lon <= b[1]) & (lat >= b[2]) & (lat <= b[3]))
            for n, b in REGIONS.items()}


REG = None
H = None   # terrain on the fine grid (m), smoothed
acc = {}   # (variant, lead, region) -> [sum fc, sum obs, {th: [h, m, f]}]


def note(name, lead, f, o):
    ok = (f >= 0) & (o >= 0)
    for rn, m in REG.items():
        k = ok & m
        a = acc.setdefault((name, lead, rn), [0.0, 0.0, {th: [0, 0, 0] for th in THS}])
        ff, oo = f[k], o[k]
        a[0] += float(ff.sum()); a[1] += float(oo.sum())
        for th in THS:
            wf, wo = ff >= th, oo >= th
            a[2][th][0] += int((wf & wo).sum()); a[2][th][1] += int((~wf & wo).sum()); a[2][th][2] += int((wf & ~wo).sum())


done = 0
t_start = time.time()
for k in range(RUNS):
    R = (newest - 125 * 60_000 - int(k * EVERY * 3600_000)) // rs.STEP_MS * rs.STEP_MS
    need = [R - m * 60_000 for m in (60, 45, 30, 15, 10, 5, 0)] + [R + m * 60_000 for m in LEADS]
    if any(img(t) is None for t in need):
        print("run", time.strftime("%d.%m %H:%M", time.gmtime(R / 1000)), "skipped (images missing)", flush=True)
        cache.clear()
        continue
    last = img(R)
    if REG is None:
        REG = region_masks(last.shape)
        h4 = np.load(DEM)["h"].astype(np.float32)
        H = cv2.resize(h4, (last.shape[1], last.shape[0]), interpolation=cv2.INTER_LINEAR)
        H = cv2.GaussianBlur(H, (0, 0), 3.0)  # ~6 km
    seq, mot = [img(R - 600_000), img(R - 300_000), last], [img(R - 1800_000), img(R - 900_000), last]
    motion = {}
    t0 = time.time()
    pure = rs.sprog_nowcast(seq, mot, rs.FC_STEPS, eu.SPROG_DOWN, thr=eu.SPROG_THR, motion_out=motion)
    pure = rs.local_amount(pure, last, motion, eu.KM_PER_PX, keep=False)
    fc = rs.advect_blend(pure, last, motion)
    del pure
    fc = {lead: np.asarray(fc[lead // 5 - 1], np.float32) for lead in LEADS}
    t_sprog = time.time() - t0
    t0 = time.time()
    for lead in LEADS:
        note("pers", lead, last, img(R + lead * 60_000))
        note("own0", lead, fc[lead], img(R + lead * 60_000))
    past = {m: img(R - m * 60_000) for m in (15, 30, 45, 60)}
    parts = {}
    for name, kw in VARIANTS.items():
        pk = (kw.get("w", 0), kw.get("mask", True))
        if pk not in parts:
            parts[pk] = rs.oro_part(last, past, H, w=pk[0], mask=pk[1])
        for lead in LEADS:
            f = rs.oro_forecast(fc[lead], lead, last, parts[pk], motion, H, eu.KM_PER_PX, tau=kw.get("tau", 120.0),
                                lee=kw.get("lee"))
            note(name, lead, f, img(R + lead * 60_000))
    del parts
    done += 1
    print("run", time.strftime("%d.%m %H:%M", time.gmtime(R / 1000)), f"done (S-PROG {t_sprog:.0f} s, variants "
          f"{time.time() - t0:.0f} s)", flush=True)
    del fc
    cache.clear()

print(f"\n{done} runs, {time.time() - t_start:.0f} s. amount = forecast / later image; CSI cell by cell at {THS} mm/h")
names = ["pers", "own0"] + list(VARIANTS)
for rn in REGIONS:
    print(f"\n== {rn}")
    for name in names:
        cells = []
        for lead in LEADS:
            a = acc.get((name, lead, rn))
            if not a or a[1] <= 0:
                cells.append(f"+{lead} -")
                continue
            csi = "/".join("%.2f" % (h / max(1, h + m + f)) for h, m, f in (a[2][th] for th in THS))
            cells.append(f"+{lead} x{a[0] / a[1]:.2f} {csi}")
        print(f"  {name:18s} " + "  ".join(cells))
