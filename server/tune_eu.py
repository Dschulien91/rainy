"""Offline variants of the Europe nowcast on past runs: does the forecast keep the rain area?

For runs of the last hours (OPERA keeps 24 h): the S-PROG forecast as served (advection blend in the first hour)
with different S-PROG rain thresholds, against the real later images. Per lead: wet area (>= 0.05 mm/h, the map's
lowest class) forecast / real, rain amount forecast / real, CSI cell by cell at 0.05 / 0.3 / 1 mm/h; "pers" = the
last image unchanged. Run in the OPERA instance (two S-PROG runs per case and variant - not in the throttled one):

  docker exec -w /app deploy-api-eu-1 python tune_eu.py [runs=6] [hours between runs=1] [variant names, comma-separated]
  (16.11.: better in a throwaway container: docker run --rm -m 2000m --cpus 1.5 deploy-api-eu python tune_eu.py ...)

Since 16.11. also per REGION (boxes below): rain amount forecast / real later image. At the gauges the forecast had
x1.4 the rain of the later picture in Germany (x1.5 Czechia, x1.25 Switzerland) while the whole-domain amount was
~0.97 - the forecast moves rain between regions. "probmatch mean" keeps S-PROG's own field and only scales its mean;
"cdf" (served) imposes the whole domain's distribution of the last image.

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
ALL = {"thr 0.10 (old)": dict(thr=0.1), "thr 0.05": dict(thr=0.05), "thr 0.03": dict(thr=0.03),
       "thr 0.03 pm-mean": dict(thr=0.03, probmatch="mean"), "thr 0.10 pm-mean": dict(thr=0.1, probmatch="mean"),
       # 16.11.: local amount control after S-PROG (rainy_server.local_amount), window sigma in km / power
       "served": dict(thr=0.03), "local150": dict(thr=0.03, local=(150, 1.0)), "local80": dict(thr=0.03, local=(80, 1.0)),
       "local300": dict(thr=0.03, local=(300, 1.0)), "local150h": dict(thr=0.03, local=(150, 0.5)),
       "local150n": dict(thr=0.03, local=(150, 1.0, False)), "local80n": dict(thr=0.03, local=(80, 1.0, False)),
       # 16.11.: local wet-area control (rainy_server.local_area) on top of the served local150n: window sigma in km,
       # where: "mid" = after local_amount (before the advection blend), "pre" = before local_amount, "post" = after
       # the advection blend (the final field); optional: slack (window may be that much wetter than the moved image)
       "area150": dict(thr=0.03, local=(150, 1.0, False), area=(150, "mid")),
       "area80": dict(thr=0.03, local=(150, 1.0, False), area=(80, "mid")),
       "area300": dict(thr=0.03, local=(150, 1.0, False), area=(300, "mid")),
       "area150pre": dict(thr=0.03, local=(150, 1.0, False), area=(150, "pre")),
       "area150post": dict(thr=0.03, local=(150, 1.0, False), area=(150, "post")),
       "area80post": dict(thr=0.03, local=(150, 1.0, False), area=(80, "post")),
       "area300post": dict(thr=0.03, local=(150, 1.0, False), area=(300, "post")),
       "area150postS": dict(thr=0.03, local=(150, 1.0, False), area=(150, "post", 1.05)),
       "area300postS": dict(thr=0.03, local=(150, 1.0, False), area=(300, "post", 1.05))}
VARIANTS = ({k: ALL[k] for k in sys.argv[3].split(",")} if len(sys.argv) > 3 else
            {k: ALL[k] for k in ("thr 0.10 (old)", "thr 0.05", "thr 0.03")})
REGIONS = {"Germany": (5.9, 15.0, 47.3, 55.0), "Alps CH/AT": (6.0, 16.5, 45.8, 47.8), "Czechia": (12.1, 18.9, 48.6, 51.0),
           "England": (-5.7, 1.8, 50.0, 55.5), "Italy N": (7.0, 13.5, 43.5, 45.8)}
if rs.NA:  # the North America instance (same script: docker run ... deploy-api-na python tune_eu.py ...)
    REGIONS = {"Northeast": (-80.0, -67.0, 38.0, 47.5), "Southeast": (-91.0, -75.0, 25.0, 36.5),
               "Midwest": (-97.0, -80.0, 37.0, 49.0), "Plains/Texas": (-106.0, -94.0, 26.0, 40.0),
               "West": (-125.0, -110.0, 32.0, 49.0), "S Canada": (-100.0, -60.0, 49.0, 55.0)}
LEADS = (30, 60, 120)
THS = (0.05, 0.3, 1.0)
rs.EuropeOpera.DESPOKE = False  # raw images in any order: the filter needs them in sequence
eu = rs.REGIONS["eu"]
times = eu.list_times()
REG = None  # region -> boolean mask on the Europe grid (built from the first image)


def region_masks(shape):
    rows, cols = np.mgrid[0:shape[0], 0:shape[1]]
    lon, lat = eu.pixel_lonlat(cols.ravel(), rows.ravel())
    lon, lat = np.asarray(lon).reshape(shape), np.asarray(lat).reshape(shape)
    return {n: (lon >= a) & (lon <= b) & (lat >= c) & (lat <= d) for n, (a, b, c, d) in REGIONS.items()}
newest = max(times)
cache = {}


def img(t):
    if t not in cache:
        cache[t] = eu.load(times[t]).astype(np.float32) if t in times else None
    return cache[t]


AREA_S = {}  # variant -> seconds per local_area call (24 steps)
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
            kw = dict(kw)
            loc = kw.pop("local", None)
            area = kw.pop("area", None)
            pure = rs.sprog_nowcast(seq, mot, rs.FC_STEPS, eu.SPROG_DOWN, motion_out=motion, **kw)

            def area_ctl(f, where):
                if not area or area[1] != where:
                    return f
                t0 = time.time()
                f = rs.local_area(f, last, motion, eu.KM_PER_PX, sigma_km=area[0], slack=area[2] if len(area) > 2 else 1.0)
                AREA_S.setdefault(name, []).append(time.time() - t0)
                return f
            pure = area_ctl(pure, "pre")
            if loc:
                pure = rs.local_amount(pure, last, motion, eu.KM_PER_PX, sigma_km=loc[0], power=loc[1],
                                       keep=loc[2] if len(loc) > 2 else True)
            pure = area_ctl(pure, "mid")
            full = area_ctl(rs.advect_blend(pure, last, motion), "post")
            fc = {lead: full[lead // 5 - 1] for lead in LEADS}
            del pure, full
        for lead in LEADS:
            f, o = np.asarray(fc[lead], np.float32), img(R + lead * 60_000)
            ok = (f >= 0) & (o >= 0)
            if REG is None:
                REG = region_masks(f.shape)
            for rn, m in REG.items():
                b = acc.setdefault((name, lead, rn), [0.0, 0.0])
                b[0] += float(f[ok & m].sum()); b[1] += float(o[ok & m].sum())
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
for name, ts in AREA_S.items():
    print(f"  {name:16s} local_area {np.mean(ts):.2f} s per run (max {max(ts):.2f})")
print()
print("rain amount forecast / real later image per region (the gauges said: forecast x1.2-1.5 of the later picture)")
for name in ["pers"] + list(VARIANTS):
    for lead in LEADS:
        cells = []
        for rn in REGIONS:
            b = acc.get((name, lead, rn))
            cells.append(f"{rn} x{b[0] / b[1]:.2f}" if b and b[1] > 0 else f"{rn} -")
        print(f"  {name:16s} +{lead:3d}  " + "  ".join(cells))
