"""Backtest of the 2 h sun forecast on past days, scored against DWD sunshine duration.

Replays the server's cloud nowcast (server/clouds.py: dense optical flow between images 30 min apart, advection,
Lagrangian growth/decay from the model, blur with lead time, blend into the model) on archived GeoColour images
over Germany (same cache as calibrate_sun.py) and the weather model's cloud cover of those days (self-hosted
Open-Meteo, past_days), then scores the sun share at ~65 stations by lead time against what really happened.
Variants are compared side by side, so a change only goes live when it wins here.

  python backtest_clouds.py [--days 14] [--variants persist,server,...]
"""
import argparse
import json
import math
import os
import sys
import time
import urllib.parse
import urllib.request
from concurrent.futures import ThreadPoolExecutor

import cv2
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import calibrate_sun as cs  # noqa: E402
from clouds import sun_elevation  # noqa: E402

OM = os.environ.get("OPENMETEO_URL", "http://127.0.0.1:8080")
LEADS = list(range(0, 121, 10))
BUCKETS = [("jetzt", 0, 5), ("+30", 25, 35), ("+60", 55, 65), ("+90", 85, 95), ("+120", 115, 125)]
# cloudiness + sun share constants (base, amp, floor, sa, sb) and whether the sun share is taken per pixel
CAL = {"base": 45, "amp": 200, "floor": 110, "sa": 0.05, "sb": 0.8, "per_pixel": False}  # live values


def model_grid(days):
    """Hourly model cloud cover 0..1 on a 0.25 deg grid over the image box: times (ms), lats (desc), lons, cover."""
    lats = np.arange(cs.LAT1, cs.LAT0 - 1e-6, -0.25)
    lons = np.arange(cs.LON0, cs.LON1 + 1e-6, 0.25)
    la, lo = (v.ravel() for v in np.meshgrid(lats, lons, indexing="ij"))
    cache = os.path.join(os.path.dirname(os.path.abspath(__file__)), "cloud_log", "calib", f"model_{days}_{time.strftime('%Y%m%d%H')}.npz")
    if os.path.exists(cache):
        z = np.load(cache)
        return z["times"], z["cover"]
    cover, times = [], None
    for i in range(0, len(la), 200):  # small batches: big ones overload the self-hosted Open-Meteo (S3 timeouts)
        body = urllib.parse.urlencode({"latitude": ",".join(f"{v:.2f}" for v in la[i:i + 200]),
                                       "longitude": ",".join(f"{v:.2f}" for v in lo[i:i + 200]),
                                       "hourly": "cloud_cover", "past_days": str(days + 1), "forecast_days": "1",
                                       "timeformat": "unixtime"}).encode()
        for k in range(5):
            try:
                with urllib.request.urlopen(urllib.request.Request(f"{OM}/v1/forecast", data=body), timeout=300) as r:
                    d = json.loads(r.read())
                break
            except Exception:
                if k == 4:
                    raise
                time.sleep(30)
        d = d if isinstance(d, list) else [d]
        times = np.array(d[0]["hourly"]["time"], np.int64) * 1000
        cover += [[(v or 0) / 100 for v in e["hourly"]["cloud_cover"]] for e in d]
    cover = np.array(cover, np.float32).T.reshape(len(times), len(lats), len(lons))
    np.savez_compressed(cache, times=times, cover=cover)
    return times, cover


class Model:
    def __init__(self, days):
        self.times, self.cover = model_grid(days)

    def field(self, ts):
        k = min(max(int(np.searchsorted(self.times, ts, side="right")) - 1, 0), len(self.times) - 2)
        f = min(max((ts - self.times[k]) / (self.times[k + 1] - self.times[k]), 0), 1)
        small = (1 - f) * self.cover[k] + f * self.cover[k + 1]
        return cv2.resize(small, (cs.W, cs.H), interpolation=cv2.INTER_LINEAR)


def cloud_field(m, ts):
    """Cloudiness from the darkest channel, sun elevation per pixel (coarse)."""
    lat = np.linspace(cs.LAT1, cs.LAT0, 50)[:, None] * np.ones((1, 85))
    lon = np.ones((50, 1)) * np.linspace(cs.LON0, cs.LON1, 85)[None, :]
    elev = cv2.resize(sun_elevation(ts, lat, lon).astype(np.float32), (cs.W, cs.H))
    return cs.cloudiness(m, elev, CAL["base"], CAL["amp"], CAL["floor"]).astype(np.float32)


GX, GY = np.meshgrid(np.arange(cs.W, dtype=np.float32), np.arange(cs.H, dtype=np.float32))


def flow_field(prev, last, down=4, sigma=10):
    s = (cs.W // down, cs.H // down)
    a = (cv2.resize(prev, s, interpolation=cv2.INTER_AREA) * 255).astype(np.uint8)
    b = (cv2.resize(last, s, interpolation=cv2.INTER_AREA) * 255).astype(np.uint8)
    f = cv2.calcOpticalFlowFarneback(a, b, None, 0.5, 4, 21, 4, 7, 1.5, 0)
    f = cv2.GaussianBlur(f, (0, 0), sigma)
    return cv2.resize(f, (cs.W, cs.H), interpolation=cv2.INTER_LINEAR) * (down / 30.0)  # px per minute


def sat_weight(lead, full, zero):
    return 1.0 if lead <= full else max(0.0, 1 - (lead - full) / (zero - full))


def forecast(variant, field, ref, model):
    """{lead: cloudiness field} for one issue time; field(ts) gives the observed cloudiness (or None)."""
    last = field(ref)
    out = {0: last}
    if variant == "persist":
        return {L: last for L in LEADS}
    if variant == "model":
        return {L: model.field(ref + L * 60_000) for L in LEADS}
    opt = VARIANTS[variant]
    prev = field(ref - opt.get("dt", 30) * 60_000)
    if prev is None:
        prev = field(ref - 30 * 60_000)
    flow = flow_field(prev, last, sigma=opt.get("sigma", 10)) * (30.0 / opt.get("dt", 30) if prev is not None else 1)
    m0 = model.field(ref)
    for L in LEADS[1:]:
        mx, my = GX - flow[..., 0] * L, GY - flow[..., 1] * L
        c = cv2.remap(last, mx, my, cv2.INTER_LINEAR, borderMode=cv2.BORDER_REPLICATE)
        m1 = model.field(ref + L * 60_000)
        if opt.get("trend", 0):
            a0 = cv2.remap(m0, mx, my, cv2.INTER_LINEAR, borderMode=cv2.BORDER_REPLICATE)
            a1 = a0 + opt["trend"] * (m1 - a0)
            thin = np.where(a0 > 0.05, a1 / np.maximum(a0, 1e-3), 1.0)
            form = np.where(a0 < 0.95, (a1 - a0) / np.maximum(1 - a0, 1e-3), 0.0)
            c = np.where(a1 < a0, c * thin, c + (1 - c) * form * 0.5)
        if opt.get("blur", 0):
            c = cv2.GaussianBlur(c, (0, 0), opt["blur"] * math.sqrt(L / 10))
        w = sat_weight(L, *opt.get("blend", (60, 240)))
        if w < 1:
            c = w * c + (1 - w) * m1
        out[L] = np.clip(c, 0, 1)
    return out


VARIANTS = {
    "persist": {}, "model": {},
    "server": {"trend": 0.5, "blur": 0.6, "blend": (60, 240)},
    "advect": {"blur": 0.6, "blend": (9999, 10000)},
    "notrend": {"blur": 0.6, "blend": (60, 240)},
    "blend30": {"trend": 0.5, "blur": 0.6, "blend": (30, 180)},
    "blend0": {"trend": 0.5, "blur": 0.6, "blend": (0, 240)},
    "trend1": {"trend": 1.0, "blur": 0.6, "blend": (60, 240)},
    "blur2": {"trend": 0.5, "blur": 1.5, "blend": (60, 240)},
    "adv_b1": {"blur": 1.0, "blend": (9999, 10000)},
    "adv_b2": {"blur": 2.0, "blend": (9999, 10000)},
    "adv_b3": {"blur": 3.0, "blend": (9999, 10000)},
    "adv_b4": {"blur": 4.0, "blend": (9999, 10000)},
    "adv_b6": {"blur": 6.0, "blend": (9999, 10000)},
    "adv_s5": {"blur": 0.6, "sigma": 5, "blend": (9999, 10000)},
    "adv_s20": {"blur": 0.6, "sigma": 20, "blend": (9999, 10000)},
    "adv_dt60": {"blur": 0.6, "dt": 60, "blend": (9999, 10000)},
    "adv_b1_dt60": {"blur": 1.0, "dt": 60, "blend": (9999, 10000)},
}


def sun_share(field, r0, r1, c0, c1):
    w = field[r0:r1, c0:c1]
    if CAL["per_pixel"]:
        return float((1 - np.clip((w - CAL["sa"]) / CAL["sb"], 0, 1)).mean())
    return float(1 - np.clip((w.mean() - CAL["sa"]) / CAL["sb"], 0, 1))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--days", type=int, default=14)
    ap.add_argument("--variants", default="persist,model,server,advect,notrend,blend30,blend0,trend1,blur2")
    ap.add_argument("--cal", help="json overriding the cloudiness/sun constants, e.g. '{\"base\": 90}'")
    ap.add_argument("--cache", default=os.path.join(os.path.dirname(os.path.abspath(__file__)), "cloud_log", "calib"))
    a = ap.parse_args()
    if a.cal:
        CAL.update(json.loads(a.cal))
    now = int(time.time() * 1000)
    since = (now - a.days * 86400_000) // 86400_000 * 86400_000
    st = cs.stations()
    with ThreadPoolExecutor(8) as ex:
        truth = dict(zip([s[0] for s in st], ex.map(lambda s: cs.sunshine(s[0], since), st)))
    model = Model(a.days)
    wins = []
    for sid, lat, lon in st:
        r0 = int((cs.LAT1 - lat - cs.WIN_LAT) / (cs.LAT1 - cs.LAT0) * cs.H)
        c0 = int((lon - cs.WIN_LON - cs.LON0) / (cs.LON1 - cs.LON0) * cs.W)
        wins.append((sid, lat, lon, r0, r0 + 6, c0, c0 + 6))
    files = sorted(f for f in os.listdir(a.cache) if f.startswith("geo_"))
    times = [int(f[4:-4]) for f in files]
    have = set(times)
    issues = [t for t in times if t - 1800_000 in have and t >= since]
    variants = a.variants.split(",")
    acc = {}
    for t in issues:
        fields = {}

        def field(x):
            if x not in fields:
                if x not in have:
                    return None
                m = cs.image(x, a.cache)
                fields[x] = None if (m < 6).mean() > 0.3 else cloud_field(m, x)
            return fields[x]
        if field(t) is None or field(t - 1800_000) is None:
            continue
        for v in variants:
            fc = forecast(v, field, t, model)
            for name, l0, l1 in BUCKETS:
                for L in LEADS:
                    if not l0 <= L <= l1:
                        continue
                    vt = t + L * 60_000
                    for sid, lat, lon, r0, r1, c0, c1 in wins:
                        o = [truth[sid].get(vt), truth[sid].get(vt + 600_000)]
                        if None in o or sun_elevation(vt, lat, lon) < 10:
                            continue
                        p, ob = sun_share(fc[L], r0, r1, c0, c1), float(np.mean(o))
                        e = acc.setdefault((v, name), [[], []])
                        e[0].append(abs(p - ob)); e[1].append((p >= 0.5) == (ob >= 0.5))
    print(f"{len(issues)} issue times, CAL {CAL}")
    print("variant   " + " | ".join(f"{n:>12s}" for n, _, _ in BUCKETS) + "   (MAE sun share / hit rate)")
    for v in variants:
        cells = [f"{np.mean(acc[(v, n)][0]):.3f} {100 * np.mean(acc[(v, n)][1]):3.0f}%" if (v, n) in acc else "-" for n, _, _ in BUCKETS]
        print(f"{v:9s} " + " | ".join(f"{c:>12s}" for c in cells))


if __name__ == "__main__":
    main()
