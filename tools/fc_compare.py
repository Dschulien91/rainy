"""Forecast check for the WetterOnline comparison: how well did each app's rain forecast match that app's own
later radar image?

Screenshots live in tools/fc_shots/ named <src>_<valid UTC YYYYmmddHHMM>_L<lead min>.png (L0 = the actual image at
that time). src "ours" = tools/compare_shot.ps1 (900x820, light theme, Germany zoom 6); src "wo" = the WetterOnline
RegenRadar Deutschland map cropped to (0,30)-(520,590) at scale 0.6. Rain = saturated, bright pixels inside the
map area (UI elements masked). A forecast pixel counts as a hit when actual rain lies within TOL_PX (~10 km), so
small position errors are forgiven like in a gauge check with a window.

  python tools/fc_compare.py            -> CSI / POD / FAR per source and lead, and the pairs used
"""
import glob
import os
import re
import sys
from collections import defaultdict
from datetime import datetime

import cv2
import numpy as np
from PIL import Image

DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "fc_shots")
MATCH_MIN = 10  # an actual image within this many minutes of the forecast's valid time
AREA = {        # (x0, y0, x1, y1) of the map without UI, as a fraction of the image size
    "ours": (0.0, 0.0, 0.92, 0.66),
    "wo": (0.02, 0.10, 0.89, 0.86),
}
TOL_PX = {"ours": 4, "wo": 3}


def rain_mask(path, src):
    a = np.asarray(Image.open(path).convert("RGB")).astype(np.int16)
    h, w = a.shape[:2]
    sat = a.max(2) - a.min(2)
    # coloured and bright (UI blue is dark, basemap and hatching grey), or bluish: the faintest drizzle class;
    # thin bluish lines (borders) are opened away
    faint = cv2.morphologyEx((a[..., 2] - a[..., 0] > 20).astype(np.uint8), cv2.MORPH_OPEN, np.ones((3, 3), np.uint8)) > 0
    m = ((sat > 45) & (a.max(2) > 140)) | faint
    x0, y0, x1, y1 = AREA[src]
    keep = np.zeros_like(m)
    keep[int(y0 * h):int(y1 * h), int(x0 * w):int(x1 * w)] = True
    if src == "ours":  # location dot in the middle of the map, logo top left
        cv2.circle(keep.view(np.uint8), (int(0.501 * w), int(0.337 * h)), int(0.02 * w), 0, -1)
        keep[:int(0.07 * h), :int(0.17 * w)] = False
    return m & keep


def score(fc, act, tol):
    if not fc.any() and not act.any():
        return None
    d_act = cv2.distanceTransform((~act).astype(np.uint8), cv2.DIST_L2, 3)
    d_fc = cv2.distanceTransform((~fc).astype(np.uint8), cv2.DIST_L2, 3)
    hits = int((fc & (d_act <= tol)).sum())
    false = int((fc & (d_act > tol)).sum())
    miss = int((act & (d_fc > tol)).sum())
    return hits, miss, false


def main():
    shots = defaultdict(dict)  # src -> {(valid, lead): path}
    for p in glob.glob(os.path.join(DIR, "*_L*.*")):
        m = re.match(r"(\w+?)_(\d{12})_L(\d+)\.", os.path.basename(p))
        if m:
            shots[m.group(1)][(datetime.strptime(m.group(2), "%Y%m%d%H%M"), int(m.group(3)))] = p
    for src, d in sorted(shots.items()):
        actual = {v: p for (v, lead), p in d.items() if lead == 0}
        tot = defaultdict(lambda: [0, 0, 0, 0])
        for (valid, lead), p in sorted(d.items()):
            if lead == 0 or not actual:
                continue
            best = min(actual, key=lambda v: abs((v - valid).total_seconds()))
            if abs((best - valid).total_seconds()) > MATCH_MIN * 60:
                continue
            s = score(rain_mask(p, src), rain_mask(actual[best], src), TOL_PX[src])
            if s is None:  # both dry: nothing to score
                continue
            t = tot[lead]
            t[0] += s[0]; t[1] += s[1]; t[2] += s[2]; t[3] += 1
            if "-v" in sys.argv:
                print(f"  {src} +{lead:3d} valid {valid:%d.%m. %H:%M}Z  hits {s[0]} miss {s[1]} false {s[2]}")
        for lead in sorted(tot):
            h, m, f, n = tot[lead]
            csi, pod, far = h / max(h + m + f, 1), h / max(h + m, 1), f / max(h + f, 1)
            print(f"{src:4s} +{lead:3d} min: CSI {csi:.2f}  POD {pod:.2f}  FAR {far:.2f}   ({n} pairs)")


if __name__ == "__main__":
    main()
