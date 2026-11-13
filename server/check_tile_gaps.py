"""Do the WebGL rain data tiles have time steps that drop out? (user, 13.11.2026: "data flickers in some countries,
sometimes the forecast is cut off and there is rain only up to now")

For each instance (European, own picture, Germany): the app's timeline, every zoom-5 tile over Europe, wet pixels per
time step, loaded through the public address exactly as the app does. Reports: a step whose wet area collapses
against both neighbours, tiles with rain in the measured steps but almost none in the forecast, a jump at "now",
tiles that fail. Every scan also notes the wet area at the first step / now / +30 / +60 / +120 min - how fast the
forecast loses rain area (13.11.: -35 % in the first hour, the measured trend was -9 %).

The OPERA instance runs the scan every 10 minutes (rainy_server.tile_gap_loop) into verify_log/tile_gaps.log, with
each instance's radar age: findings while an instance was starting are expected, findings in normal operation are
bugs. By hand, with the per-step numbers:

  docker exec -w /app deploy-api-eu-1 python check_tile_gaps.py
"""
import io
import json
import time
import urllib.request

import numpy as np
from PIL import Image

BASE = "https://hellorainy.com"
S, COLS = 132, 7
INSTANCES = (("European", "/eu", "raindataeu", "?src=eu", range(14, 20), range(8, 13)),
             ("own picture", "/eun", "raindataeu", "?src=eu", range(14, 20), range(8, 13)),
             ("Germany", "", "raindata", "", range(16, 18), range(10, 12)))


def get(url):
    with urllib.request.urlopen(urllib.request.Request(url, headers={"User-Agent": "Mozilla/5.0"}), timeout=60) as r:
        return r.read()


def scan(brief=False):
    out = []
    stamp = time.strftime("%d.%m. %H:%M", time.gmtime())
    for name, prefix, kind, q, xs, ys in INSTANCES:
        try:
            age = json.loads(get(f"{BASE}{prefix}/api/ready")).get("radar_age_min")
        except Exception as e:
            age = f"not ready ({e})"
        try:
            d = json.loads(get(f"{BASE}{prefix}/api/rain/frames{q}"))
            fr = d["frames"]
            n, t0 = len(fr), fr[0]["ts"]
        except Exception as e:
            out.append(f"{stamp} {name}: no timeline ({e}), radar age {age}")
            continue
        nobs = sum(not f["fc"] for f in fr)
        complete = all(b["ts"] - a["ts"] == 300_000 for a, b in zip(fr, fr[1:]))
        tot, found, z = np.zeros(n), [], 5
        for x in xs:
            for y in ys:
                try:
                    img = np.asarray(Image.open(io.BytesIO(get(f"{BASE}{prefix}/tiles/{kind}/{d['key']}/{t0}-{n}/{z}/{x}/{y}.png"))))
                except Exception as e:
                    found.append(f"   tile {z}/{x}/{y}: {e}")
                    continue
                wet = np.array([(img[(i // COLS) * S:(i // COLS + 1) * S, (i % COLS) * S:(i % COLS + 1) * S, 0] > 0).sum()
                                for i in range(n)])
                tot += wet
                for i in range(1, n - 1):
                    lo = min(wet[i - 1], wet[i + 1])
                    if lo >= 150 and wet[i] < 0.5 * lo:
                        found.append(f"   tile {z}/{x}/{y} step {i} ({'forecast' if fr[i]['fc'] else 'measured'}): "
                                     f"wet {wet[i]} between {wet[i - 1]} and {wet[i + 1]}")
                if 0 < nobs < n:
                    if wet[:nobs].max() >= 150 and wet[nobs:].max() < 0.2 * wet[:nobs].max():
                        found.append(f"   tile {z}/{x}/{y}: rain in the measured steps (up to {wet[:nobs].max()} px), almost none in the forecast")
                    if wet[nobs - 1] >= 300 and abs(int(wet[nobs]) - int(wet[nobs - 1])) > 0.35 * wet[nobs - 1]:
                        found.append(f"   tile {z}/{x}/{y}: jump at 'now': {wet[nobs - 1]} -> {wet[nobs]} wet px")
        at = lambda i: int(tot[min(max(i, 0), n - 1)])
        out.append(f"{stamp} {name}: key {d['key']}, {n} steps ({nobs} measured), sequence complete {complete}, radar age {age}, "
                   f"findings {len(found)}, wet px first/now/+30/+60/+120: {at(0)}/{at(nobs - 1)}/{at(nobs + 5)}/{at(nobs + 11)}/{at(n - 1)}")
        out += found
        if not brief:
            out.append("   wet pixels per step, all tiles: " + " ".join(str(int(v)) for v in tot))
    return out


if __name__ == "__main__":
    print("\n".join(scan()))
