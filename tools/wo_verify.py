"""WetterOnline's rain forecast scored against the DWD radar, by the same rules as DWD's forecast and ours.

Input: screenshots of WetterOnline's RegenRadar map frame (www.wetteronline.de/regenradar, the embedded map at its
default view: centre 51.014 N 10.503 E, zoom 6, 640x690 px, Web Mercator with 256 px tiles), cropped to the frame's
top 640x650 px, saved as tools/fc_shots/wom_<valid UTC YYYYmmddHHMM>_L<lead min>.png (L0 = their latest radar image).

Truth: the DWD radar analyses the server saves at ~3 km (server/verify_log/obs3km, fcverify._save_obs).
Each WetterOnline frame is mapped onto that grid; a cell is wet when rain covers >= WET_FRAC of it. Scoring is
fcverify's: inside the area our map shows, ~9 km tolerance (TOL_PX 3 cells). L0 rows say how well their *displayed*
radar matches the DWD radar at all (the ceiling for their forecast under these rules).

For the same issue times and leads, DWD's and our forecast are read from verify_log/fcgrid_*.jsonl (fcverify), so
all three are compared on the same cases:

  python tools/wo_verify.py          -> CSI (POD/FAR) per lead for wo, dwd, own on the same cases
  python tools/wo_verify.py -v       -> one line per frame
  python tools/wo_verify.py --fit    -> find the pixel offset of the map against the DWD radar (georeference check)
"""
import glob
import json
import math
import os
import re
import sys
from collections import defaultdict
from datetime import datetime, timezone

import cv2
import numpy as np
from PIL import Image

HERE = os.path.dirname(os.path.abspath(__file__))
SHOTS = os.path.join(HERE, "fc_shots")
LOG = os.path.join(HERE, "..", "server", "verify_log")
OBS = os.path.join(LOG, "obs3km")

R = 6378137.0
CENTER = (10.503, 51.014)          # lon, lat of the frame centre (iframe URL wrx=51.014,10.503)
CENTER_PX = (320.0, 345.0)         # frame 640x690
M_PER_PX = 2 * math.pi * R / (256 * 2 ** 6)
OFFSET = (1.0, 0.0)                # px correction from --fit (07.11. 10:20Z: best IoU at dx 0..+2, dy 0)
WET_FRAC = 0.3
TOL_PX = 3
LEADS = (15, 30, 60, 90, 120)
LEAD_SLACK = 7                     # their steps after +35 min are on :00/:15/:30/:45, so leads like +50/+65/+95 occur
# UI on the frame (x0, y0, x1, y1): clock, search, layer buttons, zoom buttons, 15/30 min, info, menu arrow
UI = [(0, 0, 115, 55), (420, 0, 640, 50), (585, 50, 640, 150), (70, 595, 195, 650), (15, 625, 65, 650),
      (585, 595, 640, 650)]


def merc(lon, lat):
    return R * np.radians(lon), R * np.log(np.tan(np.pi / 4 + np.radians(lat) / 2))


def rain_mask(path):
    a = np.asarray(Image.open(path).convert("RGB")).astype(np.int16)
    sat = a.max(2) - a.min(2)
    # rain is coloured (the faintest class light cyan); basemap, borders, labels, water are grey/white/black
    faint = cv2.morphologyEx(((a[..., 2] - a[..., 0] > 25) & (a.max(2) > 150)).astype(np.uint8), cv2.MORPH_OPEN,
                             np.ones((2, 2), np.uint8)) > 0
    m = ((sat > 45) & (a.max(2) > 120)) | faint
    valid = np.ones(m.shape, bool)
    for x0, y0, x1, y1 in UI:
        valid[y0:y1, x0:x1] = False
    return m & valid, valid


def grid():
    with open(os.path.join(OBS, "grid.json")) as f:
        g = json.load(f)
    return g, np.load(os.path.join(OBS, "mask.npy"))


def maps(g, offset=OFFSET):
    """For every 3 km cell centre: the frame pixel (float) it falls on."""
    x0, y0, x1, y1 = g["bbox"]
    h, w = g["shape"]
    cx, cy = merc(*CENTER)
    mx = x0 + (np.arange(w) + 0.5) * g["cell"]
    my = y1 - (np.arange(h) + 0.5) * g["cell"]
    px = CENTER_PX[0] + offset[0] + (mx - cx) / M_PER_PX
    py = CENTER_PX[1] + offset[1] - (my - cy) / M_PER_PX
    return np.meshgrid(px.astype(np.float32), py.astype(np.float32))


def to_grid(path, g, mp):
    m, valid = rain_mask(path)
    cell_px = g["cell"] / M_PER_PX  # ~1.8 frame px per 3 km cell
    k = max(1, int(round(cell_px)))
    frac = cv2.blur(m.astype(np.float32), (k, k))
    ok = cv2.erode(valid.astype(np.uint8), np.ones((k, k), np.uint8)).astype(np.float32)
    wet = cv2.remap(frac, *mp, cv2.INTER_LINEAR, borderMode=cv2.BORDER_CONSTANT, borderValue=0) >= WET_FRAC
    inside = cv2.remap(ok, *mp, cv2.INTER_NEAREST, borderMode=cv2.BORDER_CONSTANT, borderValue=0) > 0.5
    return wet, inside


def counts(fc, obs, mask):
    fc, obs = fc & mask, obs & mask
    k = np.ones((2 * TOL_PX + 1,) * 2, np.uint8)
    near_obs = cv2.dilate(obs.astype(np.uint8), k) > 0
    near_fc = cv2.dilate(fc.astype(np.uint8), k) > 0
    return int((fc & near_obs).sum()), int((obs & ~near_fc).sum()), int((fc & ~near_obs).sum())


def truth(valid):
    p = os.path.join(OBS, valid.strftime("%Y%m%d%H%M") + ".npz")
    return np.load(p)["rate"].astype(np.float32) if os.path.exists(p) else None


def shots():
    out = []
    for p in sorted(glob.glob(os.path.join(SHOTS, "wom_*_L*.png"))):
        m = re.match(r"wom_(\d{12})_L(\d+)\.png", os.path.basename(p))
        if m:
            out.append((datetime.strptime(m.group(1), "%Y%m%d%H%M").replace(tzinfo=timezone.utc), int(m.group(2)), p))
    return out


def fcgrid_rows():
    rows = {}
    for p in glob.glob(os.path.join(LOG, "fcgrid_*.jsonl")):
        with open(p) as f:
            for line in f:
                r = json.loads(line)
                rows[(r.get("variant", "dwd"), r["ref"], r["lead"])] = r  # early rows: DWD only
    return rows


def fmt(t):
    h, m, f = t[:3]
    return f"{h / max(h + m + f, 1):.2f} ({h / max(h + m, 1):.2f}/{f / max(h + f, 1):.2f})"


def fit():
    g, mask = grid()
    best = []
    for valid, lead, p in shots():
        a = truth(valid)
        if lead or a is None:
            continue
        for dx in range(-8, 9, 2):
            for dy in range(-8, 9, 2):
                wet, inside = to_grid(p, g, maps(g, (dx, dy)))
                o = a >= 0.1
                mk = mask & inside
                inter = (wet & o & mk).sum()
                union = ((wet | o) & mk).sum()
                best.append((inter / max(union, 1), dx, dy, valid))
    for row in sorted(best, reverse=True)[:8]:
        print(f"IoU {row[0]:.3f}  dx {row[1]:+d} dy {row[2]:+d}  {row[3]:%d.%m. %H:%M}Z")


def main():
    if "--fit" in sys.argv:
        return fit()
    g, mask = grid()
    mp = maps(g)
    rows = fcgrid_rows()
    tot = {k: defaultdict(lambda: [0, 0, 0, 0]) for k in ("wo", "dwd", "own")}
    l0 = [0, 0, 0, 0, 0, 0]  # counts vs >= 0.1, wo wet cells, dwd wet cells
    for valid, lead, p in shots():
        a = truth(valid)
        if a is None:
            continue
        wet, inside = to_grid(p, g, mp)
        mk = mask & inside
        o = a >= 0.1
        c = counts(wet, o, mk)
        if lead == 0:
            for i in range(3):
                l0[i] += c[i]
            l0[3] += 1; l0[4] += int((wet & mk).sum()); l0[5] += int((o & mk).sum())
            if "-v" in sys.argv:
                print(f"  L0   {valid:%d.%m. %H:%M}Z  wo radar vs DWD radar CSI {fmt(c)}  wet cells wo {int((wet & mk).sum())} dwd {int((o & mk).sum())}")
            continue
        std = min(LEADS, key=lambda L: abs(L - lead))
        if abs(std - lead) > LEAD_SLACK:
            continue
        issue = valid.timestamp() * 1000 - lead * 60_000
        ref = int(round(issue / 300_000) * 300_000)  # their issue time on the 5-min run grid
        d, o_ = rows.get(("dwd", ref, std)), rows.get(("own", ref, std))
        if d is None or o_ is None:  # DWD and our forecast for the same run and lead must exist: same cases only
            continue
        for name, t in (("wo", c), ("dwd", d["fc0.1"]), ("own", o_["fc0.1"])):
            s = tot[name][std]
            s[0] += t[0]; s[1] += t[1]; s[2] += t[2]; s[3] += 1
        if "-v" in sys.argv:
            print(f"  +{lead:3d} {valid:%d.%m. %H:%M}Z  wo {fmt(c)}  dwd {fmt(d['fc0.1'])}  own {fmt(o_['fc0.1'])}")
    if l0[3]:
        print(f"WetterOnline's radar image vs DWD radar ({l0[3]} images): CSI {fmt(l0)}, wet area wo/dwd "
              f"{l0[4] / max(l0[5], 1):.2f}")
    print(">= 0.1 mm/h (DWD radar), CSI (POD/FAR), same cases")
    for name in ("wo", "dwd", "own"):
        print(f"  {name:4s}" + " | ".join(f"+{L}: {fmt(tot[name][L])} [{tot[name][L][3]}]" for L in LEADS if tot[name][L][3]))


if __name__ == "__main__":
    main()
