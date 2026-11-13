"""Do the WebGL rain data tiles have time steps that drop out? (user, 13.11.2026: "data flickers in some countries,
sometimes the forecast is cut off and there is rain only up to now")

For each instance (European, own picture, Germany): the app's timeline, every zoom-5 tile over Europe, wet pixels per
time step. Reports: a step whose wet area collapses against both neighbours, tiles with rain in the measured steps
but almost none in the forecast, a jump at "now", tiles that fail. With --log only one line per instance and the
findings (for the server's cron: every 10 minutes into verify_log/tile_gaps.log, with each instance's /api/ready -
findings while an instance was restarting are expected, findings in normal operation are bugs).

  docker exec deploy-api-1 python /app/verify_log/check_tile_gaps.py [--log]
"""
import io
import json
import sys
import urllib.request

import numpy as np
from PIL import Image

BASE = "https://hellorainy.com"
LOG = "--log" in sys.argv
import time
STAMP = time.strftime("%d.%m. %H:%M", time.gmtime())
S, COLS = 132, 7


def get(url):
    with urllib.request.urlopen(urllib.request.Request(url, headers={"User-Agent": "Mozilla/5.0"}), timeout=60) as r:
        return r.read()


for name, prefix, kind, q in (("European", "/eu", "raindataeu", "?src=eu"), ("own picture", "/eun", "raindataeu", "?src=eu"),
                              ("Germany", "", "raindata", "")):
    try:
        ready = json.loads(get(f"{BASE}{prefix}/api/ready")).get("radar_age_min")
    except Exception as e:
        ready = f"not ready ({e})"
    try:
        d = json.loads(get(f"{BASE}{prefix}/api/rain/frames{q}"))
    except Exception as e:
        print(f"{STAMP} {name}: no timeline ({e}), radar age {ready}")
        continue
    fr = d["frames"]
    n, t0 = len(fr), fr[0]["ts"]
    nobs = sum(not f["fc"] for f in fr)
    steps_ok = all(b["ts"] - a["ts"] == 300_000 for a, b in zip(fr, fr[1:]))
    print(f"\n{name}: key {d['key']}, {n} steps ({nobs} measured, {n - nobs} forecast), 5-minute sequence complete: {steps_ok}")
    z = 5
    xs, ys = (range(14, 20), range(8, 13)) if name != "Germany" else (range(16, 18), range(10, 12))
    tot = np.zeros(n)
    bad = 0
    for x in xs:
        for y in ys:
            try:
                img = np.asarray(Image.open(io.BytesIO(get(f"{BASE}{prefix}/tiles/{kind}/{d['key']}/{t0}-{n}/{z}/{x}/{y}.png"))))
            except Exception as e:
                print(f"   tile {z}/{x}/{y}: {e}")
                bad += 1
                continue
            wet = np.array([(img[(i // COLS) * S:(i // COLS + 1) * S, (i % COLS) * S:(i % COLS + 1) * S, 0] > 0).sum() for i in range(n)])
            tot += wet
            for i in range(1, n - 1):
                lo = min(wet[i - 1], wet[i + 1])
                if lo >= 150 and wet[i] < 0.5 * lo:
                    print(f"   tile {z}/{x}/{y} step {i} ({'forecast' if fr[i]['fc'] else 'measured'}): wet {wet[i]} between {wet[i - 1]} and {wet[i + 1]}")
                    bad += 1
            if wet[:nobs].max() >= 150 and wet[nobs:].max() < 0.2 * wet[:nobs].max():
                print(f"   tile {z}/{x}/{y}: rain in the measured steps (up to {wet[:nobs].max()} px) but almost none in the forecast")
                bad += 1
            if wet[nobs - 1] >= 300 and abs(int(wet[nobs]) - int(wet[nobs - 1])) > 0.35 * wet[nobs - 1]:
                print(f"   tile {z}/{x}/{y}: jump at 'now': {wet[nobs - 1]} -> {wet[nobs]} wet px")
                bad += 1
    if not LOG:
        print("   wet pixels per step, all tiles:", " ".join(str(int(v)) for v in tot))
    print(f"   findings: {bad}; wet px first / now / last step: {int(tot[0])} / {int(tot[nobs - 1])} / {int(tot[-1])}")
