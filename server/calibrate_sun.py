"""Calibrate "satellite image -> sun share" against DWD sunshine duration (past days, no waiting for logs).

Fetches past MTG GeoColour images over Germany (EUMETSAT View WMS keeps the archive) every 30 min by day and the
DWD 10-minute sunshine duration of ~70 stations (recent archive + today), then scores the formula the server and
app use (cloudiness per pixel from the darkest channel with a sun-elevation scale, ~12 km mean, sun share =
1 - clamp((c - a) / b)) and searches for better constants.

  python calibrate_sun.py [--days 14] [--cache DIR]
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
CDC = "https://opendata.dwd.de/climate_environment/CDC/observations_germany/climate/10_minutes/solar"
WMS = "https://view.eumetsat.int/geoserver/ows"
LON0, LAT0, LON1, LAT1, W, H = 2.0, 46.0, 19.0, 56.0, 850, 500  # ~1.4 x 2.2 km pixels
WIN_LAT, WIN_LON = 0.055, 0.085  # ~12 km window, like the app's point value


def get(url, timeout=120):
    for k in range(3):
        try:
            with urllib.request.urlopen(urllib.request.Request(url, headers=UA), timeout=timeout) as r:
                return r.read()
        except Exception:
            if k == 2:
                raise
            time.sleep(3)


def stations():
    text = get(f"{CDC}/now/zehn_now_sd_Beschreibung_Stationen.txt").decode("latin-1")
    recent = time.strftime("%Y%m%d", time.gmtime(time.time() - 3 * 86400))
    out = []
    for line in text.splitlines()[2:]:
        p = line.split()
        if len(p) >= 6 and p[0].isdigit() and p[2] >= recent and LAT0 + 0.2 < float(p[4]) < LAT1 - 0.2:
            out.append((p[0], float(p[4]), float(p[5])))
    return out


def sunshine(sid, since):
    vals = {}
    for url in (f"{CDC}/recent/10minutenwerte_SOLAR_{sid}_akt.zip", f"{CDC}/now/10minutenwerte_SOLAR_{sid}_now.zip"):
        try:
            z = zipfile.ZipFile(io.BytesIO(get(url)))
        except Exception:
            continue
        rows = z.read(z.namelist()[0]).decode("latin-1").splitlines()
        k = [h.strip() for h in rows[0].split(";")].index("SD_10")
        for r in rows[1:]:
            p = r.split(";")
            ts = int(datetime.strptime(p[1].strip(), "%Y%m%d%H%M").replace(tzinfo=timezone.utc).timestamp() * 1000)
            if ts >= since and float(p[k]) > -999:
                vals[ts] = min(1.0, float(p[k]) * 6)
    return vals


def image(ts, cache):
    fp = os.path.join(cache, f"geo_{ts}.jpg")
    if not os.path.exists(fp):
        stamp = datetime.fromtimestamp(ts / 1000, timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")
        data = get(f"{WMS}?service=WMS&version=1.3.0&request=GetMap&layers=mtg_fd:rgb_geocolour&styles=&crs=EPSG:4326"
                   f"&bbox={LAT0},{LON0},{LAT1},{LON1}&width={W}&height={H}&format=image/jpeg&time={stamp}")
        with open(fp, "wb") as f:
            f.write(data)
    return np.asarray(Image.open(fp).convert("RGB")).min(axis=2).astype(np.float32)


def cloudiness(m, elev, base, amp, floor):
    top = base + amp * np.sqrt(np.clip(np.sin(np.radians(elev)), 0, 1))
    return np.clip((m - floor) / np.maximum(top - floor, 30), 0, 1)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--days", type=int, default=14)
    ap.add_argument("--per-pixel", action="store_true", help="sun share per pixel, then the window mean")
    ap.add_argument("--cache", default=os.path.join(os.path.dirname(os.path.abspath(__file__)), "cloud_log", "calib"))
    a = ap.parse_args()
    os.makedirs(a.cache, exist_ok=True)
    now = int(time.time() * 1000)
    since = (now - a.days * 86400_000) // 86400_000 * 86400_000
    st = stations()
    with ThreadPoolExecutor(8) as ex:
        truth = dict(zip([s[0] for s in st], ex.map(lambda s: sunshine(s[0], since), st)))
    times = [t for t in range(since, now - 3600_000, 1800_000) if sun_elevation(t, 51.0, 10.0) > 12]
    print(f"{len(st)} stations, {len(times)} images", flush=True)
    with ThreadPoolExecutor(4) as ex:
        imgs = list(ex.map(lambda t: (t, image(t, a.cache)), times))

    # samples: the pixels of each station's window, sun elevation, observed sun share (20 min around the image)
    win, elev, obs, when = [], [], [], []
    for t, m in imgs:
        if (m < 6).mean() > 0.5:  # missing image
            continue
        for sid, lat, lon in st:
            v = [truth[sid].get(t), truth[sid].get(t + 600_000)]
            if None in v:
                continue
            e = float(sun_elevation(t, lat, lon))
            if e < 10:
                continue
            r0, r1 = int((LAT1 - lat - WIN_LAT) / (LAT1 - LAT0) * H), int((LAT1 - lat + WIN_LAT) / (LAT1 - LAT0) * H) + 1
            c0, c1 = int((lon - WIN_LON - LON0) / (LON1 - LON0) * W), int((lon + WIN_LON - LON0) / (LON1 - LON0) * W) + 1
            w = m[r0:r1, c0:c1].ravel()
            win.append(w[:36] if len(w) >= 36 else np.pad(w, (0, 36 - len(w)), mode="edge"))
            elev.append(e); obs.append(np.mean(v)); when.append(t)
    win, elev, obs = np.array(win), np.array(elev)[:, None], np.array(obs)
    print(f"{len(obs)} samples, observed mean sun share {obs.mean():.2f}", flush=True)

    def evaluate(prm, sel=slice(None)):
        base, amp, floor, sa, sb = prm
        c = cloudiness(win[sel], elev[sel], base, amp, floor).mean(axis=1)
        p = 1 - np.clip((c - sa) / sb, 0, 1)
        if a.per_pixel:  # sun share per pixel, then the mean (clear gaps count fully)
            p = (1 - np.clip((cloudiness(win[sel], elev[sel], base, amp, floor) - sa) / sb, 0, 1)).mean(axis=1)
        o = obs[sel]
        return np.abs(p - o).mean(), ((p >= 0.5) == (o >= 0.5)).mean(), (p - o).mean(), c

    def search(sel):
        best = None
        for base in range(30, 201, 15):
            for amp in range(0, 241, 20):
                for floor in (80, 90, 100, 110, 120, 130):
                    cp = cloudiness(win[sel], elev[sel], base, amp, floor)
                    c = cp.mean(axis=1)
                    for sa in np.arange(0.0, 0.51, 0.05):
                        for sb in np.arange(0.2, 1.01, 0.05):
                            if a.per_pixel:
                                mae = np.abs((1 - np.clip((cp - sa) / sb, 0, 1)).mean(axis=1) - obs[sel]).mean()
                            else:
                                mae = np.abs(1 - np.clip((c - sa) / sb, 0, 1) - obs[sel]).mean()
                            if best is None or mae < best[0]:
                                best = (mae, (base, amp, floor, round(float(sa), 2), round(float(sb), 2)))
        return best[1]

    for name, prm in (("v0.44 (fest)", (170, 0, 80, 0.15, 0.35)), ("v0.45 (jetzt)", (120, 120, 85, 0.35, 0.45))):
        mae, hit, bias, _ = evaluate(prm)
        mo, ho, _, _ = evaluate(prm, (np.array(when) // 86400_000) % 2 == 1)
        print(f"{name:14s} odd days: MAE {mo:.3f}  Treffer {100 * ho:.1f}%")
        print(f"{name:14s} MAE {mae:.3f}  Treffer {100 * hit:.1f}%  Bias {bias:+.3f}")
    # fit on even days, score on odd days (honest check against overfitting), then fit on all
    even = (np.array(when) // 86400_000) % 2 == 0
    prm = search(even)
    mae, hit, bias, _ = evaluate(prm, ~even)
    print(f"fit even -> odd days {prm}  MAE {mae:.3f}  Treffer {100 * hit:.1f}%  Bias {bias:+.3f}")
    prm = search(slice(None))
    mae, hit, bias, c = evaluate(prm)
    print(f"best (all) {prm}  MAE {mae:.3f}  Treffer {100 * hit:.1f}%  Bias {bias:+.3f}")
    bins = np.digitize(c, np.arange(0.1, 1.0, 0.1))
    print("c-bin: n / observed sun share  " + "  ".join(
        f"{k / 10:.1f}:{(bins == k).sum()}/{obs[bins == k].mean():.2f}" for k in range(10) if (bins == k).any()))

if __name__ == "__main__":
    main()
