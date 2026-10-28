"""Same check as check_rain_gauges.py for the EUMETNET OPERA composite (our rain outside Germany).

OPERA's open archive keeps 24 h, so this compares the last ~22 h every 30 min at the German gauges, with the rain
rate exactly as the server derives it (rainy_server.EuropeOpera.load). If OPERA and DWD are off by a similar factor,
one correction can be applied to both without a step at the German border.

  python check_opera_gauges.py
"""
import os
import re
import sys
import time
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime, timedelta, timezone

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import check_rain_gauges as crg  # noqa: E402
import mosaic  # noqa: E402
import rainy_server as rs  # noqa: E402


def main():
    eu = rs.EuropeOpera()
    now = datetime.now(timezone.utc)
    keys = {}
    for day in (now - timedelta(days=1), now):
        xml = rs.fetch(f"{eu.BUCKET}/?prefix={day:%Y/%m/%d}/OPERA/COMP/&max-keys=1000").decode()
        for key in re.findall(r"<Key>([^<]+@DBZH\.h5)</Key>", xml):
            t = datetime.strptime(key.split("OPERA@")[1].split("@")[0], "%Y%m%dT%H%M").replace(tzinfo=timezone.utc)
            if t.minute in (0, 30) and t > now - timedelta(hours=23):
                keys[int(t.timestamp() * 1000)] = key
    st = crg.stations()
    since = min(keys) - 3600_000
    with ThreadPoolExecutor(8) as ex:
        g = dict(zip([s[0] for s in st], ex.map(lambda s: crg.gauge(s[0], since), st)))
    col, row = eu.to_pixels(*mosaic.merc(np.array([s[2] for s in st]), np.array([s[1] for s in st])))
    col, row = np.round(col).astype(int), np.round(row).astype(int)
    print(f"{len(st)} gauges, {len(keys)} OPERA images", flush=True)
    pr, pg = [], []
    for t in sorted(keys):
        if t + 600_000 > max(max(v, default=0) for v in g.values()):
            continue
        rate = eu.load(keys[t])
        rate = np.where(rate < 0, 0, rate)
        for k, (sid, lat, lon) in enumerate(st):
            v = [g[sid].get(t), g[sid].get(t + 600_000)]
            if None in v:
                continue
            pr.append(rate[row[k] - 2:row[k] + 3, col[k] - 2:col[k] + 3].mean())
            pg.append(np.mean(v) * 6)
    pr, pg = np.array(pr), np.array(pg)
    print(f"{len(pr)} pairs; mean rain gauge {pg.mean():.3f} mm/h, OPERA {pr.mean():.3f} mm/h "
          f"(OPERA/gauge {pr.sum() / max(pg.sum(), 1e-9):.2f})")
    for fac in (0.4, 0.5, 0.6, 0.8, 1.0, 1.25, 1.5, 1.75):
        q = pr * fac
        cs = []
        for th in (0.1, 0.3, 1.0):
            wo, wf = pg >= th, q >= th
            h, m, f = (wo & wf).sum(), (wo & ~wf).sum(), (~wo & wf).sum()
            cs.append(f"{h / max(1, h + m + f):.2f}")
        print(f"factor {fac}: ratio {q.sum() / max(pg.sum(), 1e-9):.2f}  CSI 0.1/0.3/1.0: {' '.join(cs)}")
    for th in (0.1, 0.3, 1.0):
        wo, wf = pg >= th, pr >= th
        h, m, f = (wo & wf).sum(), (wo & ~wf).sum(), (~wo & wf).sum()
        print(f">= {th} mm/h: gauge wet {wo.sum()}, OPERA wet {wf.sum()}, CSI {h / max(1, h + m + f):.2f} "
              f"POD {h / max(1, h + m):.2f} FAR {f / max(1, h + f):.2f}")


if __name__ == "__main__":
    main()
