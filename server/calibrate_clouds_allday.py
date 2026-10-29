"""Cloud detection for every sun elevation (day, twilight, night), fitted against DWD hourly cloud cover.

The GeoColour photo only works by day: at dawn/dusk it is dim and in the night part it shows infrared clouds and
city lights, so the day formula saw clear sky under thick cloud (29.10. morning: western Europe clear in the app,
overcast at WetterOnline). Here the truth is the hourly total cloud cover (octas, ceilometers) of ~200 DWD stations,
which is independent of the sun, and the candidate formula combines
  - day: the darkest GeoColour channel on the sun-elevation scale (as calibrated by calibrate_sun.py)
  - infrared: MTG IR10.5 grey value colder than the clear-sky value for the station's air temperature
  - fog: the light-blue GeoColour night channel (low cloud the infrared barely sees)
with a weight from the sun elevation. Constants are searched on even days and checked on odd days.

  python calibrate_clouds_allday.py [--days 12]
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
from clouds import sun_elevation  # noqa: E402

UA = {"User-Agent": "Mozilla/5.0"}
CDC = "https://opendata.dwd.de/climate_environment/CDC/observations_germany/climate"
WMS = "https://view.eumetsat.int/geoserver/ows"
LON0, LAT0, LON1, LAT1, W, H = 5.5, 47.0, 15.5, 55.2, 500, 600  # Germany, ~1.4 km pixels
WIN = 4  # +-4 px ~ 12 km


def get(url, timeout=120):
    for k in range(4):
        try:
            with urllib.request.urlopen(urllib.request.Request(url, headers=UA), timeout=timeout) as r:
                return r.read()
        except Exception:
            if k == 3:
                raise
            time.sleep(3)


def ts_of(s):
    fmt = "%Y%m%d%H%M" if len(s) == 12 else "%Y%m%d%H"
    return int(datetime.strptime(s, fmt).replace(tzinfo=timezone.utc).timestamp() * 1000)


def table(url, col):
    try:
        z = zipfile.ZipFile(io.BytesIO(get(url)))
    except Exception:
        return {}
    name = next(n for n in z.namelist() if n.startswith("produkt"))
    rows = z.read(name).decode("latin-1").splitlines()
    k = [h.strip() for h in rows[0].split(";")].index(col)
    out = {}
    for r in rows[1:]:
        p = r.split(";")
        v = float(p[k])
        if v > -999:
            out[ts_of(p[1].strip())] = v
    return out


def stations():
    text = get(f"{CDC}/hourly/cloudiness/recent/N_Stundenwerte_Beschreibung_Stationen.txt").decode("latin-1")
    recent = time.strftime("%Y%m%d", time.gmtime(time.time() - 10 * 86400))
    out = []
    for line in text.splitlines()[2:]:
        p = line.split()
        if len(p) >= 6 and p[0].isdigit() and p[2] >= recent:
            lat, lon = float(p[4]), float(p[5])
            if LAT0 + 0.15 < lat < LAT1 - 0.15 and LON0 + 0.2 < lon < LON1 - 0.2:
                out.append((p[0], lat, lon))
    return out


def image(layer, style, ts, cache):
    fp = os.path.join(cache, f"{layer.split(':')[1]}_{ts}.png")
    if not os.path.exists(fp):
        stamp = datetime.fromtimestamp(ts / 1000, timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")
        fmt = "image/jpeg" if "geocolour" in layer else "image/png"
        for k in range(4):
            data = get(f"{WMS}?service=WMS&version=1.3.0&request=GetMap&layers={layer}&styles={style}&crs=EPSG:4326"
                       f"&bbox={LAT0},{LON0},{LAT1},{LON1}&width={W}&height={H}&format={fmt}&time={stamp}")
            try:
                Image.open(io.BytesIO(data))
                break
            except OSError:
                if k == 3:
                    return None
                time.sleep(3)
        with open(fp, "wb") as f:
            f.write(data)
    return np.asarray(Image.open(fp).convert("RGB")).astype(np.float32)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--days", type=int, default=12)
    ap.add_argument("--fixed", help="score only these constants: ir0,ir_k,ir_s,fog0,fog_s,e0,e1,ir_day")
    ap.add_argument("--cache", default=os.path.join(os.path.dirname(os.path.abspath(__file__)), "cloud_log", "allday"))
    a = ap.parse_args()
    os.makedirs(a.cache, exist_ok=True)
    st = stations()
    with ThreadPoolExecutor(8) as ex:
        cover = dict(zip([s[0] for s in st], ex.map(
            lambda s: table(f"{CDC}/hourly/cloudiness/recent/stundenwerte_N_{s[0]}_akt.zip", "V_N"), st)))
        temp = dict(zip([s[0] for s in st], ex.map(
            lambda s: table(f"{CDC}/10_minutes/air_temperature/recent/10minutenwerte_TU_{s[0]}_akt.zip", "TT_10"), st)))
    end = max(max(v, default=0) for v in cover.values())
    times = list(range(end - a.days * 86400_000, end + 1, 3600_000))
    print(f"{len(st)} stations ({sum(1 for s in st if temp[s[0]])} with temperature), {len(times)} hours", flush=True)

    def both(t):
        return t, image("mtg_fd:rgb_geocolour", "", t, a.cache), image("mtg_fd:ir105_hrfi", "mtg_fd:mtg_fd_ir105_hrfi_grayscale", t, a.cache)
    with ThreadPoolExecutor(4) as ex:
        imgs = list(ex.map(both, times))

    rows = []  # m (darkest channel), blue excess, ir grey (white = cold), elev, T2m, observed cover, day index
    for t, geo, ir in imgs:
        if geo is None or ir is None or (geo.max(axis=2) < 6).mean() > 0.3:
            continue
        for sid, lat, lon in st:
            o, tt = cover[sid].get(t), temp[sid].get(t)
            if o is None or tt is None or o > 8:  # 9 = sky not visible (fog) -> skip
                continue
            r = int((LAT1 - lat) / (LAT1 - LAT0) * H)
            c = int((lon - LON0) / (LON1 - LON0) * W)
            g = geo[r - WIN:r + WIN + 1, c - WIN:c + WIN + 1]
            rows.append((g.min(axis=2).mean(), (g[..., 2] - g[..., 0]).mean(), ir[r - WIN:r + WIN + 1, c - WIN:c + WIN + 1, 0].mean(),
                         float(sun_elevation(t, lat, lon)), tt, o / 8, t // 86400_000))
    d = np.array(rows)
    m, blue, irg, elev, t2m, obs, day = d.T
    print(f"{len(d)} samples; mean observed cover {obs.mean():.2f}", flush=True)

    def current(sel):  # live formula (server-v0.23)
        top = 45 + 200 * np.sqrt(np.clip(np.sin(np.radians(elev[sel])), 0, 1))
        cd = np.clip((m[sel] - 110) / np.maximum(top - 110, 30), 0, 1)
        cn = np.clip((m[sel] - 80) / 90, 0, 1)
        w = np.clip((elev[sel] + 3) / 9, 0, 1)
        return w * cd + (1 - w) * cn

    def candidate(p, sel):
        ir0, ir_k, ir_s, fog0, fog_s, e0, e1, ir_day = p
        top = 45 + 200 * np.sqrt(np.clip(np.sin(np.radians(elev[sel])), 0, 1))
        cd = np.clip((m[sel] - 110) / np.maximum(top - 110, 30), 0, 1)
        # infrared: colder (higher inverted grey) than expected for the air temperature -> cloud
        ci = np.clip((irg[sel] - (ir0 + ir_k * -t2m[sel])) / ir_s, 0, 1)
        cf = np.clip((blue[sel] - fog0) / fog_s, 0, 1)
        cn = np.maximum(ci, cf)
        cd = np.maximum(cd, ir_day * ci)  # by day the infrared can add clouds the photo shows too dark
        w = np.clip((elev[sel] - e0) / max(e1 - e0, 1), 0, 1)
        return w * cd + (1 - w) * cn

    bins = [("Nacht", -90, -6), ("Dämmerung", -6, 12), ("Tag", 12, 90)]

    def report(name, pred_fn, sel):
        cells = []
        for bn, lo, hi in bins:
            s = sel & (elev >= lo) & (elev < hi)
            if s.any():
                p = pred_fn(s)
                cells.append(f"{bn}: MAE {np.abs(p - obs[s]).mean():.3f} hit {((p >= 0.5) == (obs[s] >= 0.5)).mean():.0%}")
        print(f"{name:22s} " + " | ".join(cells))

    even, odd = day % 2 == 0, day % 2 == 1
    report("live (odd days)", lambda s: current(s), odd)
    if a.fixed:  # score one parameter set only: ir0, ir_k, ir_s, fog0, fog_s, e0, e1, ir_day
        report("fixed (odd days)", lambda s: candidate([float(v) for v in a.fixed.split(",")], s), odd)
        return
    best = None
    rng = np.random.default_rng(1)
    grid = {"ir0": np.arange(40, 201, 5), "ir_k": np.arange(0, 6.1, 0.5), "ir_s": [2, 4, 6, 8, 10, 15, 20],
            "fog0": [20, 30, 40, 50, 60, 80, 999], "fog_s": [2, 5, 10], "e0": [0, 3, 6, 10, 15],
            "e1": [10, 15, 20, 25, 30, 40], "ir_day": [0, 0.5, 1]}
    keys = list(grid)
    for _ in range(40000):  # random search over the grid
        p = [rng.choice(grid[k]) for k in keys]
        if p[6] <= p[5]:
            continue
        mae = np.abs(candidate(p, even) - obs[even]).mean()
        if best is None or mae < best[0]:
            best = (mae, p)
    print("best on even days:", dict(zip(keys, [float(v) for v in best[1]])))
    report("candidate (odd days)", lambda s: candidate(best[1], s), odd)


if __name__ == "__main__":
    main()
