"""Removes radar interference spokes from a composite: thin straight lines that do not move and point at a radar.

Why (13.11.2026): the European composite showed a 90 km long, 2-4 km wide line of 15-200 mm/h across western Slovenia
(45.6 N 14.28 E -> 46.3 N 14.46 E) inside light rain of ~1 mm/h, at the same cells for more than an hour: the beam of
the Croatian radar Goli (45.02 N 14.12 E) towards a transmitter in its band (wireless link), which the composite's
quality control let through. On the map it is a "storm" that is not there, and the nowcast carries it forward.

What is removed - all of it must hold, so that no real rain goes:
  thin      the cell stands out from the opening of the field with a ~14 km disc by more than MIN_RES mm/h and by more
            than the surroundings themselves (at least twice their level), and it is not part of a patch 6 km wide;
  unmoved   it does so in at least SHARE of the images within +-WINDOW (at least MIN_IMAGES of them) - rain moves, a
            narrow cold-front line crosses its own width in one or two images;
  a line    such cells form a straight segment of at least MIN_LEN_KM (gaps up to MAX_GAP_KM);
  radial    and the segment's line passes a radar site (opera_radars.json = EUMETNET's public list, plus EXTRA_SITES)
            that lies within MAX_RANGE_KM - a band of sea-effect showers is thin, long, straight and stays for hours
            too, but it does not point at a radar.
Small stationary cores (a thunderstorm anchored at a mountain, also ground clutter) are NOT touched. The removed cells
get the level of their surroundings (the opening), not zero.

Test on the live images: spoke_test.py in EVALUATION.md 8b (lists every removed line with its radar).
"""
import json
import os

import cv2
import numpy as np

DISC_KM = 14.0
MIN_RES = 1.0          # mm/h above the surroundings
WINDOW_MS = 30 * 60_000
MIN_IMAGES = 5
SHARE = 0.7
MIN_LEN_KM = 30.0
MAX_GAP_KM = 8.0
MAX_RANGE_KM = 250.0   # a radar's range: the site may be this far from the segment
NEAR_KM = 3.0          # how near the line must pass the site ... plus SLOPE km per km between the segment and the site
SLOPE = 0.02           # (the direction of a 30 km segment of 2 km cells is known to ~1 degree)
KEEP_MS = 3 * 3600_000
# Radars of the composite that the public list does not hold (positions to ~1 km): Austria
EXTRA_SITES = [("atrau", 48.074, 16.536), ("atfel", 48.065, 13.062), ("atzir", 47.072, 14.560),
               ("atpat", 47.209, 11.461), ("atval", 47.158, 10.213)]


def sites(host, merc):
    """[(code, col, row)] of the radar sites on the host's grid."""
    with open(os.path.join(os.path.dirname(os.path.abspath(__file__)), "opera_radars.json")) as f:
        rows = [tuple(r) for r in json.load(f)] + EXTRA_SITES
    col, row = host.to_pixels(*merc(np.array([r[2] for r in rows]), np.array([r[1] for r in rows])))
    return [(r[0], float(c), float(q)) for r, c, q in zip(rows, col, row) if 0 <= c < host.NX and 0 <= q < host.NY]


def thin(rate, km):
    """(mask of the cells that stand out as a thin structure, the field's level without them)."""
    r = np.maximum(rate, 0).astype(np.float32)
    k = max(3, int(round(DISC_KM / km)) | 1)
    bg = cv2.morphologyEx(r, cv2.MORPH_OPEN, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (k, k)))
    return r - bg > np.maximum(MIN_RES, bg), bg


class SpokeFilter:
    def __init__(self, km, radar_sites):
        self.km = km
        self.codes = [s[0] for s in radar_sites]
        self.xy = np.array([s[1:] for s in radar_sites], np.float64).reshape(-1, 2)
        self.seen = {}   # time -> flat indices of the thin cells (widened by one cell: a spoke may wobble by one)

    def clean(self, ts, rate):
        """rate (float32, < 0 = no data) without the spokes; also returns the removed lines as
        [(length km, radar code, (col, row) of its middle)]."""
        km = self.km
        cand, bg = thin(rate, km)
        self.seen[ts] = np.flatnonzero(cv2.dilate(cand.astype(np.uint8), np.ones((3, 3), np.uint8))).astype(np.int32)
        for t in [t for t in self.seen if t < max(self.seen) - KEEP_MS]:
            del self.seen[t]
        near = [m for t, m in self.seen.items() if abs(t - ts) <= WINDOW_MS]
        if len(near) < MIN_IMAGES:
            return rate, []
        count = np.zeros(rate.size, np.uint8)
        for m in near:
            count[m] += 1
        stay = (cand & (count.reshape(rate.shape) >= SHARE * len(near))).astype(np.uint8)
        three = np.ones((3, 3), np.uint8)
        line = stay & ~cv2.dilate(cv2.erode(stay, three), three)   # without the patches of 3 x 3 cells and more
        segs = cv2.HoughLinesP(line * 255, 1, np.pi / 360, threshold=int(MIN_LEN_KM / km * 0.5),
                               minLineLength=MIN_LEN_KM / km, maxLineGap=MAX_GAP_KM / km)
        if segs is None:
            return rate, []
        ys, xs = np.nonzero(line)
        pts = np.stack([xs, ys], 1).astype(np.float64)
        ays, axs = np.nonzero(stay)   # removed along a found line: every thin unmoved cell, patches too
        apts = np.stack([axs, ays], 1).astype(np.float64)
        remove = np.zeros(rate.shape, np.uint8)
        found = []
        for x1, y1, x2, y2 in segs.reshape(-1, 4).astype(np.float64):
            u = np.array([x2 - x1, y2 - y1])
            length = float(np.hypot(*u))
            u /= length
            rel = pts - (x1, y1)
            on = (np.abs(rel @ (-u[1], u[0])) <= 2.0) & (rel @ u >= -3) & (rel @ u <= length + 3)
            if on.sum() < length * 0.5:
                continue
            # the line through those cells (least squares), then: does it pass a radar?
            vx, vy, x0, y0 = cv2.fitLine(pts[on].astype(np.float32), cv2.DIST_L2, 0, 0.01, 0.01).ravel()
            d = self.xy - (x0, y0)
            along, across = d @ (vx, vy), np.abs(d @ (-vy, vx))
            a = (pts[on] - (x0, y0)) @ (vx, vy)
            gap = np.maximum(0, np.maximum(a.min() - along, along - a.max()))  # site to the segment, along the line
            hit = (gap * km <= MAX_RANGE_KM) & (across * km <= NEAR_KM + SLOPE * gap * km)
            if not hit.any():
                continue
            k = int(np.argmin(np.where(hit, across, np.inf)))
            if not remove[ys[on], xs[on]].all():
                found.append((round((a.max() - a.min()) * km), self.codes[k], (int(xs[on].mean()), int(ys[on].mean()))))
            d = apts - (x0, y0)
            sel = (np.abs(d @ (-vy, vx)) <= 2.0) & (d @ (vx, vy) >= a.min() - 3) & (d @ (vx, vy) <= a.max() + 3)
            remove[ays[sel], axs[sel]] = 1
        if not found:
            return rate, []
        remove = cv2.dilate(remove, three).astype(bool) & (rate >= 0)   # with the line's weaker flanks
        return np.where(remove, np.minimum(bg, rate), rate).astype(np.float32), found
