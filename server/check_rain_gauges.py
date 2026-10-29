"""Do our rain rates match the rain gauges? Past days, DWD radar as the app shows it vs DWD 10-minute gauges.

Fetches past DWD radar images (the WMS keeps ~3 days) every 30 min over Germany, decodes them with the same
colour table as server/mosaic.py (GermanyDwd), and compares the ~5 km mean at ~300 gauges with the gauge rain
of the 20 minutes around the image: overall bias, hit rates, and the radar value by gauge intensity class.

  python check_rain_gauges.py [--days 2.5] [--cache DIR]
"""
import argparse
import io
import os
import sys
import time
import urllib.request
import zipfile
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime, timezone

import numpy as np
from PIL import Image

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from mosaic import GermanyDwd  # noqa: E402
from recorder import RAIN_CELL  # noqa: E402

UA = {"User-Agent": "Mozilla/5.0"}
CDC = "https://opendata.dwd.de/climate_environment/CDC/observations_germany/climate/10_minutes/precipitation"
WMS = "https://maps.dwd.de/geoserver/dwd/wms"
LON0, LAT0, LON1, LAT1, W, H = 5.5, 47.0, 15.5, 55.2, 700, 900  # EPSG:4326, ~1 km pixels
R = 2  # +-2 px: ~5 km window like /api/rain/point


def get(url, timeout=120):
    for k in range(4):
        try:
            with urllib.request.urlopen(urllib.request.Request(url, headers=UA), timeout=timeout) as r:
                return r.read()
        except Exception:
            if k == 3:
                raise
            time.sleep(3)


def stations():
    text = get(f"{CDC}/now/zehn_now_rr_Beschreibung_Stationen.txt").decode("latin-1")
    recent = time.strftime("%Y%m%d", time.gmtime(time.time() - 3 * 86400))
    cells = {}
    for line in text.splitlines()[2:]:
        p = line.split()
        if len(p) >= 6 and p[0].isdigit() and p[2] >= recent:
            s = (p[0], float(p[4]), float(p[5]))
            if LAT0 < s[1] < LAT1 and LON0 < s[2] < LON1:
                cells.setdefault((int(s[1] // RAIN_CELL[0]), int(s[2] // RAIN_CELL[1])), s)
    return list(cells.values())


def gauge(sid, since):
    vals = {}
    for url in (f"{CDC}/recent/10minutenwerte_nieder_{sid}_akt.zip", f"{CDC}/now/10minutenwerte_nieder_{sid}_now.zip"):
        try:
            z = zipfile.ZipFile(io.BytesIO(get(url)))
        except Exception:
            continue
        rows = z.read(z.namelist()[0]).decode("latin-1").splitlines()
        k = [h.strip() for h in rows[0].split(";")].index("RWS_10")
        for r in rows[1:]:
            p = r.split(";")
            ts = int(datetime.strptime(p[1].strip(), "%Y%m%d%H%M").replace(tzinfo=timezone.utc).timestamp() * 1000)
            if ts >= since and float(p[k]) > -999:
                vals[ts] = float(p[k])
    return vals


def radar(ts, cache, dec):
    fp = os.path.join(cache, f"dwd_{ts}.png")
    if not os.path.exists(fp):
        stamp = datetime.fromtimestamp(ts / 1000, timezone.utc).strftime("%Y-%m-%dT%H:%M:%S.000Z")
        for k in range(4):  # the WMS answers now and then with a random service exception
            data = get(f"{WMS}?service=WMS&version=1.3.0&request=GetMap&layers=dwd:Niederschlagsradar&styles=&crs=EPSG:4326"
                       f"&bbox={LAT0},{LON0},{LAT1},{LON1}&width={W}&height={H}&format=image/png&transparent=true&time={stamp}")
            try:
                Image.open(io.BytesIO(data))
                break
            except OSError:
                if k == 3:
                    return None
                time.sleep(3)
        with open(fp, "wb") as f:
            f.write(data)
    px = np.asarray(Image.open(fp).convert("RGBA"), np.int32)
    rate = np.zeros((H, W), np.float32)
    wet = px[..., 3] > 200
    d2 = ((px[wet][:, :3][:, None, :] - dec.rgb[None]) ** 2).sum(-1)
    rate[wet] = np.where(d2.min(-1) <= 3 * 30 * 30, dec.val[d2.argmin(-1)], 0.0)
    return rate


def rv_analysis(ts, cache):
    """DWD open data RV run at ts: analysis (lead 0) in mm/h on the 1 km DWD grid, plus its 'where' attributes."""
    import tarfile
    import h5py
    fp = os.path.join(cache, f"rv_{ts}.npz")
    if not os.path.exists(fp):
        name = datetime.fromtimestamp(ts / 1000, timezone.utc).strftime("composite_rv_%Y%m%d_%H%M.tar")
        try:
            data = get(f"https://opendata.dwd.de/weather/radar/composite/rv/{name}")
        except Exception:
            return None
        with tarfile.open(fileobj=io.BytesIO(data)) as tar:
            m = next(m for m in tar.getmembers() if m.name.endswith("_000-hd5"))
            with h5py.File(io.BytesIO(tar.extractfile(m).read()), "r") as f:
                w = f["dataset1/data1/what"].attrs
                raw = f["dataset1/data1/data"][:]
                rate = np.where(raw == w["nodata"], 0.0, np.maximum(raw * float(w["gain"]) + float(w["offset"]), 0.0)) * 12
                where = {k: (v.decode() if isinstance(v, bytes) else v) for k, v in f["where"].attrs.items()}
        np.savez_compressed(fp, rate=rate.astype(np.float32), projdef=where["projdef"], ul=[where["UL_lon"], where["UL_lat"]])
    z = np.load(fp)
    return z["rate"], str(z["projdef"]), z["ul"]


def report(pr, pg, what):
    print(what)
    print(f"{len(pr)} pairs; mean rain gauge {pg.mean():.3f} mm/h, radar {pr.mean():.3f} mm/h "
          f"(radar/gauge {pr.sum() / max(pg.sum(), 1e-9):.2f})")
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
        print(f">= {th} mm/h: gauge wet {wo.sum()}, radar wet {wf.sum()}, CSI {h / max(1, h + m + f):.2f} "
              f"POD {h / max(1, h + m):.2f} FAR {f / max(1, h + f):.2f}")
    for lo, hi in ((0.1, 0.5), (0.5, 1), (1, 3), (3, 100)):
        s = (pg >= lo) & (pg < hi)
        if s.any():
            print(f"gauge {lo}-{hi} mm/h: n {s.sum()}, radar mean {pr[s].mean():.2f}, median {np.median(pr[s]):.2f}, "
                  f"radar dry {np.mean(pr[s] < 0.1):.0%}")



def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--days", type=float, default=2.5)
    ap.add_argument("--source", choices=("wms", "rv"), default="wms", help="DWD WMS images or open data RV analysis")
    ap.add_argument("--cache", default=os.path.join(os.path.dirname(os.path.abspath(__file__)), "cloud_log", "dwdradar"))
    a = ap.parse_args()
    os.makedirs(a.cache, exist_ok=True)
    dec = GermanyDwd.__new__(GermanyDwd)
    dec.rgb = np.array([[int(h[i:i + 2], 16) for i in (0, 2, 4)] for h in GermanyDwd.CLASSES], np.int32)
    dec.val = np.array(list(GermanyDwd.CLASSES.values()), np.float32)
    now = int(time.time() * 1000)
    since = (now - int(a.days * 86400_000)) // 1800_000 * 1800_000
    st = stations()
    with ThreadPoolExecutor(8) as ex:
        g = dict(zip([s[0] for s in st], ex.map(lambda s: gauge(s[0], since - 3600_000), st)))
    last = max((t for v in g.values() for t in v), default=since)
    times = list(range(since, last - 600_000, 1800_000))
    print(f"{len(st)} gauges, {len(times)} radar images", flush=True)
    if a.source == "rv":
        from pyproj import Transformer
        with ThreadPoolExecutor(4) as ex:
            got = list(ex.map(lambda t: (t, rv_analysis(t, a.cache)), times))
        got = [(t, g) for t, g in got if g is not None]
        rate0, projdef, ul = got[0][1]
        to = Transformer.from_crs("EPSG:4326", projdef, always_xy=True)
        ux, uy = to.transform(*ul)
        sx, sy = to.transform(np.array([s[2] for s in st]), np.array([s[1] for s in st]))
        rc = [(int(round((uy - y) / 1000 - 0.5)), int(round((x - ux) / 1000 - 0.5))) for x, y in zip(sx, sy)]
        pr, pg = [], []
        for t, (rate, _, _) in got:
            for (sid, _, _), (r, c) in zip(st, rc):
                v = [g[sid].get(t), g[sid].get(t + 600_000)]
                if None in v:
                    continue
                pr.append(rate[r - R:r + R + 1, c - R:c + R + 1].mean())
                pg.append(np.mean(v) * 6)
        return report(np.array(pr), np.array(pg), f"{len(got)} RV runs")
    with ThreadPoolExecutor(4) as ex:
        imgs = list(ex.map(lambda t: (t, radar(t, a.cache, dec)), times))
    pr, pg = [], []
    imgs = [(t, rate) for t, rate in imgs if rate is not None]
    print(f"{len(imgs)} images usable", flush=True)
    for t, rate in imgs:
        for sid, lat, lon in st:
            v = [g[sid].get(t), g[sid].get(t + 600_000)]  # (t-10, t] and (t, t+10]
            if None in v:
                continue
            r = int((LAT1 - lat) / (LAT1 - LAT0) * H)
            c = int((lon - LON0) / (LON1 - LON0) * W)
            pr.append(rate[max(r - R, 0):r + R + 1, max(c - R, 0):c + R + 1].mean())
            pg.append(np.mean(v) * 6)  # mm per 10 min -> mm/h
    return report(np.array(pr), np.array(pg), f"{len(imgs)} WMS images")

if __name__ == "__main__":
    main()
