"""Does adding the infrared cloud test by day (calibrate_clouds_allday.py) hurt the sun share?

Cold thin cirrus counts as cloud cover but lets the sun through. Same samples as calibrate_sun.py (GeoColour by day,
DWD 10-minute sunshine), sun share = 1 - clamp((c - a) / b) searched for both c = day photo only and
c = max(day photo, infrared); fitted on even days, scored on odd days.

  python check_sun_ir.py [--days 14]
"""
import argparse
import os
import sys
import time
from concurrent.futures import ThreadPoolExecutor

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import calibrate_clouds_allday as ca  # noqa: E402
import calibrate_sun as cs  # noqa: E402
from clouds import sun_elevation  # noqa: E402

IR0, IR_K, IR_S = 80.0, 2.5, 10.0  # from calibrate_clouds_allday.py


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--days", type=int, default=14)
    a = ap.parse_args()
    cache = os.path.join(os.path.dirname(os.path.abspath(__file__)), "cloud_log", "calib")
    now = int(time.time() * 1000)
    since = (now - a.days * 86400_000) // 86400_000 * 86400_000
    st = cs.stations()
    with ThreadPoolExecutor(8) as ex:
        sun = dict(zip([s[0] for s in st], ex.map(lambda s: cs.sunshine(s[0], since), st)))
        temp = {}
        for s, t in zip(st, ex.map(lambda s: {**ca.table(f"{ca.CDC}/10_minutes/air_temperature/recent/10minutenwerte_TU_{s[0]}_akt.zip", "TT_10"),
                                               **ca.table(f"{ca.CDC}/10_minutes/air_temperature/now/10minutenwerte_TU_{s[0]}_now.zip", "TT_10")}, st)):
            temp[s[0]] = t
    times = [t for t in range(since, now - 3600_000, 1800_000) if sun_elevation(t, 51.0, 10.0) > 12]
    # IR over the same box and size as the GeoColour cache of calibrate_sun.py
    ca.LON0, ca.LAT0, ca.LON1, ca.LAT1, ca.W, ca.H = cs.LON0, cs.LAT0, cs.LON1, cs.LAT1, cs.W, cs.H
    with ThreadPoolExecutor(4) as ex:
        imgs = list(ex.map(lambda t: (t, cs.image(t, cache), ca.image("mtg_fd:ir105_hrfi", "mtg_fd:mtg_fd_ir105_hrfi_grayscale", t, cache)), times))
    cv, ci, obs, day = [], [], [], []
    for t, m, ir in imgs:
        if ir is None or (m < 6).mean() > 0.5:
            continue
        for sid, lat, lon in st:
            v, tt = [sun[sid].get(t), sun[sid].get(t + 600_000)], temp[sid].get(t)
            e = float(sun_elevation(t, lat, lon))
            if None in v or tt is None or e < 10:
                continue
            r0, r1 = int((cs.LAT1 - lat - cs.WIN_LAT) / (cs.LAT1 - cs.LAT0) * cs.H), int((cs.LAT1 - lat + cs.WIN_LAT) / (cs.LAT1 - cs.LAT0) * cs.H) + 1
            c0, c1 = int((lon - cs.WIN_LON - cs.LON0) / (cs.LON1 - cs.LON0) * cs.W), int((lon + cs.WIN_LON - cs.LON0) / (cs.LON1 - cs.LON0) * cs.W) + 1
            d = cs.cloudiness(m[r0:r1, c0:c1], e, 45, 200, 110)
            i = np.clip((ir[r0:r1, c0:c1, 0] - (IR0 - IR_K * tt)) / IR_S, 0, 1)
            cv.append(d.mean()); ci.append(np.maximum(d, i).mean()); obs.append(np.mean(v)); day.append(t // 86400_000)
    cv, ci, obs, day = map(np.array, (cv, ci, obs, day))
    even, odd = day % 2 == 0, day % 2 == 1
    print(f"{len(obs)} samples")
    for name, c in (("photo only", cv), ("photo + infrared", ci)):
        best = min(((np.abs(1 - np.clip((c[even] - sa) / sb, 0, 1) - obs[even]).mean(), sa, sb)
                    for sa in np.arange(0, 0.61, 0.05) for sb in np.arange(0.2, 1.21, 0.05)))
        _, sa, sb = best
        p = 1 - np.clip((c[odd] - sa) / sb, 0, 1)
        print(f"{name:17s} sun map ({sa:.2f}, {sb:.2f}): odd days MAE {np.abs(p - obs[odd]).mean():.3f} "
              f"hit {((p >= 0.5) == (obs[odd] >= 0.5)).mean():.1%}")


if __name__ == "__main__":
    main()
