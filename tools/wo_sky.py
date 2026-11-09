"""Rough visual comparison of sunny with WetterOnline's cloud map (WetterRadar, cloud layer), for the user's eyes.

Takes screenshots of WetterOnline's WetterRadar map frame (radar.wo-cloud.com/desktop/wr/compact, the frame of
www.wetteronline.de/wetterradar) at the same view as tools/wo_capture.py (centre 51.014 N 10.503 E, zoom 6, Web Mercator
with 256 px tiles) for now and later steps, and draws our sunny map for the same times and the same extent from our
own tiles (hellorainy.com/tiles/cloud, zoom 6 = the same pixel grid). Writes side by side images:
  tools/fc_shots/sky_<valid UTC YYYYmmddHHMM>_L<lead>.png   (left WetterOnline, right sunny, Germany outlined in both)
  tools/fc_shots/wos_<valid UTC YYYYmmddHHMM>_L<lead>.png   (their raw frame, for a score against the DWD stations)
A headless Edge with a fresh profile and its default user agent, like wo_capture.py (anonymous; only the page is
loaded, no data files). Usage: python tools/wo_sky.py   (~1 min; tools/wo_loop.py runs it once an hour)
"""
import io
import json
import math
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

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from wo_capture import EDGE, W, H, CROP_H, SHOTS, Cdp, clock, utc_of, wid  # noqa: E402

MAP = "https://radar.wo-cloud.com/desktop/wr/compact?wrx=51.014,10.503&wrm=6"
SITE = "https://hellorainy.com"
LEADS = (0, 60, 120)
CENTER = (10.503, 51.014)
CENTER_PX = (320.0, 345.0)
OFFSET = (1.0, 0.0)   # as tools/wo_verify.py (--fit against the DWD radar)
R = 6378137.0
WORLD = math.pi * R
HERE = os.path.dirname(os.path.abspath(__file__))
UA = {"User-Agent": "Mozilla/5.0"}


def get(url):
    return urllib.request.urlopen(urllib.request.Request(url, headers=UA), timeout=60).read()


def merc(lon, lat):
    return R * math.radians(lon), R * math.log(math.tan(math.pi / 4 + math.radians(lat) / 2))


def frame_origin():
    """World pixel (zoom 6, 256 px tiles) of the frame's top left pixel."""
    cx, cy = merc(*CENTER)
    n = 256 * 2 ** 6
    return (cx + WORLD) / (2 * WORLD) * n - CENTER_PX[0] - OFFSET[0], (WORLD - cy) / (2 * WORLD) * n - CENTER_PX[1] - OFFSET[1]


def outline(img, color):
    gj = json.load(open(os.path.join(HERE, "..", "server", "germany.json")))
    ox, oy = frame_origin()
    n = 256 * 2 ** 6
    for poly in gj["coordinates"]:
        for ring in poly:
            pts = []
            for lon, lat in ring:
                mx, my = merc(lon, lat)
                pts.append(((mx + WORLD) / (2 * WORLD) * n - ox, (WORLD - my) / (2 * WORLD) * n - oy))
            cv2.polylines(img, [np.round(np.array(pts)).astype(np.int32)], True, color, 1, cv2.LINE_AA)
    return img


def ours(ts_ms):
    """Our sunny map for valid time ts_ms (nearest frame within 6 min) on the frame's pixel grid, or None."""
    d = json.loads(get(f"{SITE}/api/cloud/frames"))
    best = min(d["frames"], key=lambda f: abs(f["ts"] - ts_ms), default=None)
    if not best or abs(best["ts"] - ts_ms) > 6 * 60_000:
        return None, None
    ox, oy = frame_origin()
    x0, y0 = int(ox // 256), int(oy // 256)
    x1, y1 = int((ox + W) // 256), int((oy + CROP_H) // 256)
    big = np.zeros(((y1 - y0 + 1) * 256, (x1 - x0 + 1) * 256, 2), np.uint8)
    for ty in range(y0, y1 + 1):
        for tx in range(x0, x1 + 1):
            try:
                png = get(f"{SITE}/tiles/cloud/{d['ref']}/{best['ts']}/6/{tx}/{ty}.png")
            except Exception:
                continue
            t = cv2.imdecode(np.frombuffer(png, np.uint8), cv2.IMREAD_UNCHANGED)
            if t is not None and t.ndim == 3:  # grey + alpha decodes as BGRA
                big[(ty - y0) * 256:(ty - y0 + 1) * 256, (tx - x0) * 256:(tx - x0 + 1) * 256] = t[..., [0, 3]]
    sx, sy = int(round(ox - x0 * 256)), int(round(oy - y0 * 256))
    g = big[sy:sy + CROP_H, sx:sx + W]
    v, a = g[..., 0].astype(np.float32) / 254, g[..., 1] > 127
    sun = np.clip(1 - (v - 0.05) / 0.8, 0, 1)  # the app's sunShare(): grey = 0.05 + 0.8 * (1 - share)
    # sun: warm yellow, cloud: light grey, outside the shown area: dark
    col = np.zeros((CROP_H, W, 3), np.float32)
    for k, (s, c) in enumerate(zip((90, 200, 245), (225, 225, 225))):  # BGR
        col[..., k] = sun * s + (1 - sun) * c
    col[~a] = (60, 60, 60)
    return col.astype(np.uint8), best


def main():
    os.makedirs(SHOTS, exist_ok=True)
    prof = tempfile.mkdtemp(prefix="edge-wos-")
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
        l0 = None
        for lead in LEADS:
            url = MAP if lead == 0 else MAP + "&wrt=" + wid(l0 + timedelta(minutes=lead))
            c.call("Page.navigate", url=url)
            time.sleep(14)  # the map fades in; the cloud layer is the default
            img, label = c.shot(), clock(c)
            if not label:
                print(f"+{lead}: no clock label"); continue
            t = utc_of(label, datetime.now(timezone.utc) if l0 is None else l0 + timedelta(minutes=lead))
            if l0 is None:
                l0 = t
            elif t != l0 + timedelta(minutes=lead):
                print(f"+{lead}: not shown (label {label})"); continue
            # raw frame for a later score against the DWD sunshine stations (green = cloud-free on their map)
            cv2.imwrite(os.path.join(SHOTS, f"wos_{t:%Y%m%d%H%M}_L{lead}.png"), img[:CROP_H, :W])
            mine, frame = ours(int(t.timestamp() * 1000))
            wo = outline(img[:CROP_H, :W].copy(), (0, 0, 200))
            if mine is None:
                print(f"+{lead}: no sunny frame for {t:%H:%M}Z"); continue
            right = outline(mine, (0, 0, 200))
            pad = np.full((CROP_H, 8, 3), 255, np.uint8)
            both = np.hstack([wo, pad, right])
            kind = "Messung" if not frame["fc"] else "Vorhersage"
            cv2.putText(both, f"WetterOnline {label}", (10, CROP_H - 12), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0, 0, 0), 2)
            local = datetime.fromtimestamp(frame["ts"] / 1000).strftime("%H:%M")
            cv2.putText(both, f"sunny {local} ({kind})", (W + 18, CROP_H - 12), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0, 0, 0), 2)
            p = os.path.join(SHOTS, f"sky_{t:%Y%m%d%H%M}_L{lead}.png")
            cv2.imwrite(p, both)
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
    sys.exit(main())
