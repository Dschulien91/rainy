"""Find straight rain edges on the live map that are geometry, not weather (daily evaluation, STEP 2).

Why (22.11.2026): the user saw a dead-straight east-west rain edge near Aarau (zoom 12, Klassisch design). Rain moves;
an edge made by the processing (a layer window, a weight seam, a coarse grid, a source's internal seam) stays where it
is. So this tool loads the WebGL data tiles the app shows (zoom 8, all time steps of the current run, one class per
1/8 step as the app draws it) for an area and reports class edges that run exactly along one tile row or column for
at least --min-px texels AND stay at that row/column in at least --frames measured steps. It saves the class grids
(npz) for every run so a hit can be traced afterwards with tools/recompose_national.py.

Usage (from the repo root, anonymous requests only):
  python -I tools/check_map_edges.py                        # eun (the app's default), northern Switzerland + SW Germany
  python -I tools/check_map_edges.py --base https://hellorainy.com/eu --tiles 128-139,84-92 --out C:/tmp/edges
  python -I tools/check_map_edges.py --fc                   # also forecast steps (edges that stay over the lead times)

Reading the result: a hit in eun but not in eu (same area, same time) comes from the national layers (national.py);
in both: OPERA, the nowcast or the tiles. Rain bands that lie still (a front along the Alps, a valley) also give
fixed edges, mostly east-west: compare with the moving rain before calling a hit a bug. First case, 22.11.2026
09:45-10:20Z: 47.93 N 8.17-8.44 E (south-east Black Forest, ~20 km ENE of the DWD Feldberg radar), eun only;
tools/recompose_national.py reproduced it in the mix of the DWD and Swiss layers (there the DWD layer read 2-5 mm/h,
the Swiss one 0.5-1.3, at the same places), while neither layer alone has it that straight - cause not settled.
Last line (eu / eun, since 26.11.2026): "grid lock" - the share of straight edges on the 4 km S-PROG grid's lines,
measured steps vs forecast steps (see grid_lock()). A forecast share well above the measured one says the forecast's
straight edges come from upsampling the coarse grid (26.11.: Alps measured 0.25, forecast 0.35 eun / 0.41 eu).
"""
import argparse
import io
import json
import math
import os
import time
import urllib.request
from collections import defaultdict

import numpy as np
from PIL import Image

UA = {"User-Agent": "Mozilla/5.0"}   # anonymous, neutral
S, M, COLS = 132, 2, 7               # data tile cell with its margin (index.html GL_DATA), steps per atlas row


def get(url, tries=3):
    for k in range(tries):
        try:
            return urllib.request.urlopen(urllib.request.Request(url, headers=UA), timeout=90).read()
        except Exception:
            if k == tries - 1:
                raise
            time.sleep(5)


def load(base, kind, tx, ty, z=8):
    fr = json.loads(get(base + "/api/rain/frames"))
    key, frames = fr["key"], fr["frames"]
    t0, n = frames[0]["ts"], len(frames)
    h, w = (ty[1] - ty[0] + 1) * 128, (tx[1] - tx[0] + 1) * 128
    q = np.zeros((n, h, w), np.uint8)
    for y in range(ty[0], ty[1] + 1):
        for x in range(tx[0], tx[1] + 1):
            im = np.asarray(Image.open(io.BytesIO(get(f"{base}/tiles/{kind}/{key}/{t0}-{n}/{z}/{x}/{y}.webp"))).convert("RGB"))
            for i in range(n):
                r, c = divmod(i, COLS)
                q[i, (y - ty[0]) * 128:(y - ty[0] + 1) * 128, (x - tx[0]) * 128:(x - tx[0] + 1) * 128] = \
                    im[r * S + M:(r + 1) * S - M, c * S + M:(c + 1) * S - M, 0]
            time.sleep(0.2)   # gentle on the server
    return key, frames, q


def straight_runs(cls, min_px):
    """(axis, k, start, length) of class edges exactly between line k and k+1 (axis 0: rows, 1: columns), with
    the neighbouring lines unchanged on that stretch (a step, not a gradient)."""
    out = []
    for axis in (0, 1):
        g = cls if axis == 0 else cls.T
        d = g[1:] != g[:-1]
        for k in range(1, g.shape[0] - 2):
            run = d[k] & ~d[k - 1] & ~d[k + 1]
            x = np.concatenate([[0], run.astype(np.int8), [0]])
            st, en = np.flatnonzero(np.diff(x) == 1), np.flatnonzero(np.diff(x) == -1)
            out += [(axis, k, int(s), int(e - s)) for s, e in zip(st, en) if e - s >= min_px]
    return out


def lat_of(ytile):
    return math.degrees(math.atan(math.sinh(math.pi - 2 * math.pi * ytile / 256)))


def lon_of(xtile):
    return xtile / 256 * 360 - 180


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--base", default="https://hellorainy.com/eun")
    ap.add_argument("--kind", default="raindataown")
    ap.add_argument("--tiles", default="131-136,86-90", help="x0-x1,y0-y1 at zoom 8")
    ap.add_argument("--min-px", type=int, default=10, help="straight stretch in zoom-8 texels (~0.8 km each)")
    ap.add_argument("--frames", type=int, default=4, help="measured steps the edge must stay in")
    ap.add_argument("--fc", action="store_true", help="also forecast steps")
    ap.add_argument("--out", default="")
    a = ap.parse_args()
    xs, ys = ([int(v) for v in p.split("-")] for p in a.tiles.split(","))
    key, frames, q = load(a.base, a.kind, xs, ys)
    if a.out:
        os.makedirs(a.out, exist_ok=True)
        np.savez_compressed(os.path.join(a.out, f"edges_{a.base.rstrip('/').rsplit('/', 1)[-1]}_{key}.npz"), q=q,
                            ts=np.array([f["ts"] for f in frames]), fc=np.array([f["fc"] for f in frames]), x0=xs[0], y0=ys[0])
    seen = defaultdict(set)
    for i, f in enumerate(frames):
        if f["fc"] and not a.fc:
            continue
        for axis, k, s, n in straight_runs((q[i] // 8).astype(np.int16), a.min_px):
            for b in range(s // 8, (s + n - 1) // 8 + 1):   # 8-texel bins along the edge
                seen[(axis, k, b)].add(i)
    hits = sorted(((len(v), key_, sorted(v)) for key_, v in seen.items() if len(v) >= a.frames), reverse=True)
    stamp = lambda i: time.strftime("%H:%MZ", time.gmtime(frames[i]["ts"] / 1000)) + ("fc" if frames[i]["fc"] else "")
    print(f"{a.base} {a.kind} key {key}: {len(frames)} steps, tiles {a.tiles}, {len(hits)} fixed edge bins")
    for n, (axis, k, b), fr in hits[:30]:
        if axis == 0:
            where = f"E-W edge at {lat_of(ys[0] + (k + 1) / 128):.3f} N, {lon_of(xs[0] + b / 16):.2f}-{lon_of(xs[0] + (b + 1) / 16):.2f} E"
        else:
            where = f"N-S edge at {lon_of(xs[0] + (k + 1) / 128):.3f} E, {lat_of(ys[0] + (b + 1) / 16):.2f}-{lat_of(ys[0] + b / 16):.2f} N"
        print(f"  {where}: {n} steps ({stamp(fr[0])} .. {stamp(fr[-1])})")
    if "/na" not in a.base:
        grid_lock(q, frames, xs, ys, a.min_px)


# Europe's forecast runs S-PROG on a 4 km grid (2 km picture grid / SPROG_DOWN 2, LAEA below; pixel 0 at x = 0) and
# scales it up. 26.11.2026: the straight forecast edges lay ON that grid's lines - light rain zeroed below the threshold
# on the 4 km grid, upsampled bilinearly and cut at 0.05 mm/h has its outline exactly between two coarse pixels.
# grid_lock() measures that: the LAEA position of every straight class edge (all steps, not only the fixed ones) modulo
# 4 km; "locked" = within 1/8 of a coarse pixel boundary (1/4 of the edges by chance). Measured steps are the control
# (their 2 km grid has no 4 km structure); a forecast share well above the measured one = edges made by the coarse
# grid, not by the weather. Near 10 E the grid runs along the map's rows, so the test is sharpest there (Alps,
# Germany); far east or west the grid is rotated and few straight runs are grid lines in the first place.
LAEA = "+proj=laea +lat_0=55.0 +lon_0=10.0 +x_0=1950000.0 +y_0=-2100000.0 +units=m +ellps=WGS84"
COARSE_M = 4000.0


def grid_lock(q, frames, xs, ys, min_px):
    try:
        from pyproj import Transformer
    except ImportError:
        print("grid lock: pyproj missing, skipped")
        return
    tr = Transformer.from_crs("EPSG:4326", LAEA, always_xy=True)
    share = {}
    for kind in (False, True):
        ph = []
        for i, f in enumerate(frames):
            if bool(f["fc"]) != kind:
                continue
            for axis, k, s, n in straight_runs((q[i] // 8).astype(np.int16), min_px):
                t = np.arange(s, s + n, 3) + 0.5
                if axis == 0:   # E-W edge: its northing
                    lat = np.full(len(t), lat_of(ys[0] + (k + 1) / 128))
                    lon = np.array([lon_of(xs[0] + v / 128) for v in t])
                    v = np.asarray(tr.transform(lon, lat)[1])
                else:           # N-S edge: its easting
                    lon = np.full(len(t), lon_of(xs[0] + (k + 1) / 128))
                    lat = np.array([lat_of(ys[0] + v / 128) for v in t])
                    v = np.asarray(tr.transform(lon, lat)[0])
                ph.append((v % COARSE_M) / COARSE_M)
        ph = np.concatenate(ph) if ph else np.zeros(0)
        share["fc" if kind else "obs"] = (float(((ph < 0.125) | (ph >= 0.875)).mean()) if len(ph) else float("nan"), len(ph))
    (so, no), (sf, nf) = share["obs"], share["fc"]
    verdict = "FORECAST EDGES ON THE 4 KM GRID" if nf >= 200 and sf >= max(0.32, so + 0.06) else "no grid lock"
    print(f"grid lock (edges within 1/8 of a 4 km line, 0.25 by chance): measured {so:.2f} ({no} points), forecast "
          f"{sf:.2f} ({nf} points) -> {verdict}")


if __name__ == "__main__":
    main()
