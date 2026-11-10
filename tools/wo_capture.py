"""Screenshots of WetterOnline's RegenRadar map (the map frame of www.wetteronline.de/regenradar) for tools/wo_verify.py:
their latest radar image (L0) and their forecast at +15/+30/+45/+60/+75/+90 min, plus their last radar images (WO's
own truth for earlier forecasts). A headless Edge with a fresh profile and its default user agent shows the map
exactly as a visitor sees it (anonymous: nothing identifies us; no data files are fetched directly, only the page).

Files: tools/fc_shots/wom_<valid UTC YYYYmmddHHMM>_L<lead>.png, 640x650 (map frame without its bottom bar).
Usage: python tools/wo_capture.py        (one run takes ~3 min; run every 15-30 min)
       python tools/wo_capture.py --eu   (their Europe view for the Europe beta: centre 48.5 N 11.0 E, zoom 5 = Paris -
                                          Warsaw - Rome, 15 min steps; files woe_<...>.png, scored by tools/wo_verify_eu.py)
"""
import base64
import json
import os
import shutil
import socket
import subprocess
import sys
import tempfile
import time
import urllib.request
from datetime import datetime, timedelta, timezone

import cv2
import numpy as np
import websocket

EDGE = r"C:\Program Files (x86)\Microsoft\Edge\Application\msedge.exe"
MAP = "https://radar.wo-cloud.com/desktop/rr/compact?wrx=51.014,10.503&wrm=6&wrp=periodCurrentHighRes"
MAP_EU = "https://radar.wo-cloud.com/desktop/rr/compact?wrx=48.5,11.0&wrm=5&wrp=periodCurrentHighRes"
EU_CENTER, EU_ZOOM = (11.0, 48.5), 5   # lon, lat of MAP_EU (its centre pixel is at (320, 345) like the Germany view)
W, H, CROP_H = 640, 690, 650
SHOTS = os.path.join(os.path.dirname(os.path.abspath(__file__)), "fc_shots")
LEADS = (15, 30, 45, 60, 75, 90)
PAST = (15, 30, 45, 60)       # minutes before L0: their radar images, truth for earlier forecasts
LOAD_TIMEOUT_S = 60


class Cdp:
    def __init__(self, port):
        for _ in range(50):
            try:
                tabs = json.load(urllib.request.urlopen(f"http://127.0.0.1:{port}/json"))
                page = next(t for t in tabs if t["type"] == "page")
                break
            except Exception:
                time.sleep(0.3)
        self.ws = websocket.create_connection(page["webSocketDebuggerUrl"], timeout=60)
        self.n = 0

    def call(self, method, **params):
        self.n += 1
        self.ws.send(json.dumps({"id": self.n, "method": method, "params": params}))
        while True:
            m = json.loads(self.ws.recv())
            if m.get("id") == self.n:
                if "error" in m:
                    raise RuntimeError(m["error"])
                return m.get("result", {})

    def eval(self, js):
        return self.call("Runtime.evaluate", expression=js, returnByValue=True)["result"].get("value")

    def shot(self):
        png = base64.b64decode(self.call("Page.captureScreenshot", format="png")["data"])
        return cv2.imdecode(np.frombuffer(png, np.uint8), cv2.IMREAD_COLOR)


def loaded(img):
    """The map fades in from white: done when the land is grey, not white."""
    land = img[150:550, 150:500]
    return float(land.mean()) < 225


def clock(c):
    """The frame's clock label (local time HH:MM) and whether it is a forecast (orange)."""
    return c.eval("(() => { const e=[...document.querySelectorAll('*')].find(e => /^\\d\\d:\\d\\d$/.test("
                  "(e.textContent||'').trim()) && e.children.length===0); return e ? e.textContent.trim() : null })()")


def capture(c, url):
    c.call("Page.navigate", url=url)
    t0 = time.time()
    time.sleep(8)
    while time.time() - t0 < LOAD_TIMEOUT_S:
        img = c.shot()
        if loaded(img):
            time.sleep(4)  # let the last tiles arrive
            return c.shot(), clock(c)
        time.sleep(3)
    return None, clock(c)


def utc_of(label, near):
    """Local HH:MM label -> UTC datetime closest to `near`."""
    hh, mm = map(int, label.split(":"))
    local_now = near.astimezone()
    cand = local_now.replace(hour=hh, minute=mm, second=0, microsecond=0)
    best = min((cand + timedelta(days=d) for d in (-1, 0, 1)), key=lambda t: abs((t - local_now).total_seconds()))
    return best.astimezone(timezone.utc)


def wid(t):
    return t.strftime("%Y%m%d-%H%M") + "-1"


def main(url=MAP, prefix="wom"):
    os.makedirs(SHOTS, exist_ok=True)
    prof = tempfile.mkdtemp(prefix="edge-wo-")
    with socket.socket() as s:
        s.bind(("127.0.0.1", 0))
        port = s.getsockname()[1]
    proc = subprocess.Popen([EDGE, "--headless=new", "--hide-scrollbars", f"--user-data-dir={prof}",
                             f"--remote-debugging-port={port}", f"--window-size={W},{H}", "--no-first-run",
                             "--mute-audio", "about:blank"], stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
    saved = []
    try:
        c = Cdp(port)
        c.call("Page.enable")
        c.call("Emulation.setDeviceMetricsOverride", width=W, height=H, deviceScaleFactor=1, mobile=False)
        img, label = capture(c, url)
        if img is None or not label:
            print("live map did not load"); return 1
        l0 = utc_of(label, datetime.now(timezone.utc))
        frames = [(l0, 0, img)]
        for lead in LEADS:
            t = l0 + timedelta(minutes=lead)
            im, lab = capture(c, url + "&wrt=" + wid(t))
            if im is not None and lab and utc_of(lab, t) == t:
                frames.append((t, lead, im))
            else:
                print(f"+{lead}: not shown (label {lab})")
        for back in PAST:
            t = l0 - timedelta(minutes=back)
            if os.path.exists(os.path.join(SHOTS, f"{prefix}_{t:%Y%m%d%H%M}_L0.png")):
                continue
            im, lab = capture(c, url + "&wrt=" + wid(t))
            if im is not None and lab and utc_of(lab, t) == t:
                frames.append((t, 0, im))
        for t, lead, im in frames:
            p = os.path.join(SHOTS, f"{prefix}_{t:%Y%m%d%H%M}_L{lead}.png")
            cv2.imwrite(p, im[:CROP_H, :W])
            saved.append(os.path.basename(p))
    finally:
        proc.terminate()
        try:
            proc.wait(10)
        except Exception:
            proc.kill()
        shutil.rmtree(prof, ignore_errors=True)
    print(" ".join(saved))
    return 0


if __name__ == "__main__":
    sys.exit(main(MAP_EU, "woe") if "--eu" in sys.argv else main())
