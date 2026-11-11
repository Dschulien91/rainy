"""WetterOnline's Europe rain forecast scored against the OPERA radar, by the same rules as our Europe forecast
(Europe beta, server EDITION=eu, fcverify.RegionVerify).

Input: screenshots of WetterOnline's map frame at its Europe view (tools/wo_capture.py --eu: centre 48.5 N 11.0 E,
zoom 5, 640x650 px, Web Mercator with 256 px tiles), tools/fc_shots/woe_<valid UTC YYYYmmddHHMM>_L<lead min>.png.
Truth: the OPERA analyses the Europe server saves on its ~4 km verification grid (verify_log/obs_eu, LAEA, -1 = outside
the radar range), copied from the server:
  ssh ... "cd /var/lib/docker/volumes/deploy_verify-log/_data && tar cf - obs_eu eugrid_*.jsonl" | tar xf - -C server/verify_log
Our forecasts for the same issue times and leads come from verify_log/eugrid_*.jsonl, area "wo" (exactly the ground of
their view): "own" (served), "sprog" (S-PROG alone), "pers" (nothing moves).

  python tools/wo_verify_eu.py        -> CSI (POD/FAR) per lead: wo, wo-self (vs their own later image), own, sprog, pers
  python tools/wo_verify_eu.py -v     -> one line per frame
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
from pyproj import Transformer

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from wo_verify import rain_mask, WET_FRAC, LEAD_SLACK, LEADS, fmt  # noqa: E402  the same colour reading

HERE = os.path.dirname(os.path.abspath(__file__))
SHOTS = os.path.join(HERE, "fc_shots")
LOG = os.path.join(HERE, "..", "server", "verify_log")
OBS = os.path.join(LOG, "obs_eu")
R = 6378137.0
CENTER = (11.0, 48.5)              # lon, lat of the frame centre (wo_capture.MAP_EU)
CENTER_PX = (320.0, 345.0)         # as the Germany view
ZOOM = 5
M_PER_PX = 2 * math.pi * R / (256 * 2 ** ZOOM)
OFFSET = (1.0, 0.0)                # as the Germany view (same map frame); to be checked with a fit once data exists
TOL_PX = 3                         # fcverify.TOL_PX on the ~4 km grid: the same rule as our rows


def merc(lon, lat):
    return R * np.radians(lon), R * np.log(np.tan(np.pi / 4 + np.radians(lat) / 2))


def grid_maps(shape):
    """For every verification cell centre: the frame pixel (float) it falls on, and the cell's lon/lat."""
    with open(os.path.join(OBS, "grid.json")) as f:
        g = json.load(f)
    h, w = shape
    to_ll = Transformer.from_crs(g["proj"], "EPSG:4326", always_xy=True)
    cols, rows = np.meshgrid(np.arange(w) + 0.5, np.arange(h) + 0.5)
    lon, lat = to_ll.transform(cols * g["cell"], -rows * g["cell"])
    mx, my = merc(lon, np.clip(lat, -85, 85))
    cx, cy = merc(*CENTER)
    px = CENTER_PX[0] + OFFSET[0] + (mx - cx) / M_PER_PX
    py = CENTER_PX[1] + OFFSET[1] - (my - cy) / M_PER_PX
    return (px.astype(np.float32), py.astype(np.float32)), g["cell"]


def to_grid(path, mp, cell):
    m, valid = rain_mask(path)
    k = max(1, int(round(cell / M_PER_PX)))  # ~1 frame px per 4 km cell at zoom 5
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
    for p in sorted(glob.glob(os.path.join(SHOTS, "woe_*_L*.png"))):
        m = re.match(r"woe_(\d{12})_L(\d+)\.png", os.path.basename(p))
        if m:
            out.append((datetime.strptime(m.group(1), "%Y%m%d%H%M").replace(tzinfo=timezone.utc), int(m.group(2)), p))
    return out


def rows():
    out = {}
    for p in glob.glob(os.path.join(LOG, "eugrid_*.jsonl")):
        with open(p) as f:
            for line in f:
                r = json.loads(line)
                if r.get("area") == "wo":
                    out[(r["variant"], r["ref"], r["lead"])] = r
    return out


def main():
    rr = rows()
    names = ("wo", "wo-self", "own", "pers", "sprog", "cells")  # sprog: rows until server-v0.93, cells: since
    tot = {k: defaultdict(lambda: [0, 0, 0, 0]) for k in names}
    l0_imgs = {v: p for v, lead, p in shots() if lead == 0}
    l0 = [0, 0, 0, 0, 0, 0]
    mp = cell = None
    for valid, lead, p in shots():
        a = truth(valid)
        if a is None:
            continue
        if mp is None:
            mp, cell = grid_maps(a.shape)
        wet, inside = to_grid(p, mp, cell)
        mk = (a >= 0) & inside
        o = a >= 0.1
        c = counts(wet, o, mk)
        if lead == 0:
            for i in range(3):
                l0[i] += c[i]
            l0[3] += 1; l0[4] += int((wet & mk).sum()); l0[5] += int((o & mk).sum())
            if "-v" in sys.argv:
                print(f"  L0   {valid:%d.%m. %H:%M}Z  wo radar vs OPERA CSI {fmt(c)}  wet cells wo {int((wet & mk).sum())} opera {int((o & mk).sum())}")
            continue
        std = min(LEADS, key=lambda L: abs(L - lead))
        if abs(std - lead) > LEAD_SLACK:
            continue
        ref = int(round((valid.timestamp() * 1000 - lead * 60_000) / 300_000) * 300_000)
        ours = {n: rr.get((n, ref, std)) for n in ("own", "pers", "sprog", "cells")}
        if ours["own"] is None or ours["pers"] is None:  # same cases only
            continue
        ours = {n: v for n, v in ours.items() if v is not None}
        for name, t in [("wo", c)] + [(n, v["fc0.1"]) for n, v in ours.items()]:
            s = tot[name][std]
            s[0] += t[0]; s[1] += t[1]; s[2] += t[2]; s[3] += 1
        cs = None
        if valid in l0_imgs:
            own_truth, inside2 = to_grid(l0_imgs[valid], mp, cell)
            cs = counts(wet, own_truth, mk & inside2)
            s = tot["wo-self"][std]
            s[0] += cs[0]; s[1] += cs[1]; s[2] += cs[2]; s[3] += 1
        if "-v" in sys.argv:
            print(f"  +{lead:3d} {valid:%d.%m. %H:%M}Z  wo {fmt(c)}  wo-self {fmt(cs) if cs else '-':16s}  "
                  + "  ".join(f"{n} {fmt(v['fc0.1'])}" for n, v in ours.items()))
    if l0[3]:
        print(f"WetterOnline's radar image vs OPERA ({l0[3]} images): CSI {fmt(l0)}, wet area wo/opera "
              f"{l0[4] / max(l0[5], 1):.2f}")
    print(">= 0.1 mm/h, CSI (POD/FAR) [cases]; truth = OPERA radar, for wo-self = WetterOnline's own later radar image")
    for name in names:
        cells = [f"+{L}: {fmt(tot[name][L])} [{tot[name][L][3]}]" for L in LEADS if tot[name][L][3]]
        print(f"  {name:7s} " + " | ".join(cells) if cells else f"  {name:7s} (no cases yet)")


if __name__ == "__main__":
    main()
