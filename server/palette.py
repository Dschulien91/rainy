"""Rain colours and the rain tile look ("variant D", v0.76).

Discrete colour classes per rain band with hard steps between them, rounded outlines and a thin darker contour
along every class boundary. The app (index.html, RAIN_STEPS / KIND_STEPS) uses the same table for the timeline bars
and the legend, so map and timeline match.

Tile pipeline (paint):
  rate on a coarse grid (~2 samples per 1 km radar cell, never finer than the tile pixels, with a margin)
  -> log rate -> slight Gaussian blur -> bicubic upsampling to the 2x supersampled tile -> classify into bands
  -> darker contour on the inner side of each class boundary -> RGBA -> area downsampling (premultiplied alpha)
  = anti-aliased, rounded edges instead of 1 km pixel staircases at every zoom level.
"""
import cv2
import numpy as np

BANDS = (0.05, 0.1, 0.5, 2.0, 5.0, 10.0, 30.0, 75.0)  # mm/h: lower bound of classes 1..8 (0 = dry)
DRAW_MIN = BANDS[0]
LOG_BANDS = np.log(np.array(BANDS, np.float32))

# RGBA (alpha 0-255) per precipitation kind (ptype.py: rain, sleet, snow, freezing rain, graupel, hail) and class 1..8
STEPS = [
    [(150, 200, 255, 150), (110, 175, 255, 190), (40, 130, 255, 215), (20, 70, 235, 235),
     (120, 50, 235, 245), (205, 30, 205, 250), (255, 50, 90, 255), (255, 215, 0, 255)],       # rain
    [(190, 232, 240, 130), (170, 225, 236, 185), (115, 200, 226, 210), (60, 160, 212, 228),
     (60, 115, 220, 240), (70, 90, 215, 246), (60, 70, 200, 255), (40, 50, 160, 255)],          # sleet
    [(236, 240, 255, 140), (225, 232, 255, 195), (195, 205, 255, 215), (155, 160, 255, 232),
     (125, 110, 240, 242), (110, 85, 230, 248), (95, 60, 215, 255), (70, 40, 175, 255)],        # snow
    [(255, 190, 220, 140), (255, 170, 212, 200), (250, 120, 190, 222), (235, 70, 160, 238),
     (205, 25, 130, 248), (180, 0, 115, 252), (150, 0, 95, 255), (115, 0, 70, 255)],            # freezing rain
    [(200, 242, 204, 140), (185, 238, 190, 205), (150, 225, 165, 220), (105, 200, 130, 232),
     (75, 180, 105, 240), (50, 155, 85, 246), (25, 120, 60, 255), (15, 90, 45, 255)],           # graupel
    [(255, 130, 85, 160), (255, 105, 55, 238), (250, 75, 40, 244), (235, 35, 25, 248),
     (220, 15, 10, 252), (195, 0, 0, 255), (150, 0, 0, 255), (110, 0, 0, 255)],                # hail: red, unlike heavy rain
]
COLORS = np.zeros((len(STEPS), len(BANDS) + 1, 4), np.uint8)  # [kind, class] -> RGBA, class 0 transparent
for _k, _row in enumerate(STEPS):
    COLORS[_k, 1:] = _row

CONTOUR = 0.7      # contour colour: this share of the inner class colour's brightness
SS = 2             # supersampling factor (anti-aliased edges)
FLOOR = 0.02       # mm/h standing for "dry" in log space (the 0.05 edge then sits between dry and light rain)
BLUR = 0.6         # Gaussian sigma in coarse cells (~0.3 radar cells at city zoom) before upsampling
FAINT_ALPHA = 0.65  # satellite estimate (outside the radars): drawn fainter
_KERNEL = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (2 * SS + 1, 2 * SS + 1))  # ~1 display px contour
_maps = {}

# Fast classification: log rate -> uint8 (Q_LO..Q_HI in 255 steps of ~4 %) -> class via a 256-entry table
Q_LO, Q_HI = np.log(0.005), np.log(250.0)
Q_K = 255.0 / (Q_HI - Q_LO)
_CLASS_LUT = np.searchsorted(LOG_BANDS, Q_LO + np.arange(256) / Q_K, side="right").astype(np.uint8)
# Paint table, index = kind * 18 + contour * 9 + class (+ 108 when faint): premultiplied RGBA (uint8 downsampling)
_NK, _NC = len(STEPS), len(BANDS) + 1
_PAINT = np.zeros((256, 1, 4), np.uint8)
for _k in range(_NK):
    for _e in (0, 1):
        for _f in (0, 1):
            _c = COLORS[_k].astype(np.float32)
            _c[:, :3] *= CONTOUR if _e else 1.0
            _c[:, 3] *= FAINT_ALPHA if _f else 1.0
            _c[:, :3] *= _c[:, 3:4] / 255.0
            _PAINT[_f * _NK * 2 * _NC + _k * 2 * _NC + _e * _NC + np.arange(_NC), 0] = np.round(_c)


def classes(rate):
    """Band class 0..8 per rate (0 = below DRAW_MIN)."""
    r = np.asarray(rate, np.float32)
    return np.searchsorted(LOG_BANDS, np.log(np.maximum(r, 1e-6)), side="right").astype(np.uint8)


def colorize(rate, kind=None):
    """Stepped colours without smoothing (whole-view radar images of the full edition)."""
    c = classes(rate)
    return COLORS[0][c] if kind is None else COLORS[kind, c]


def _upsample_maps(n, margin, size):
    """remap() coordinates of the supersampled tile pixel centres in the coarse grid (coarse sample i sits at
    (i - margin + 0.5) coarse cells from the tile's corner)."""
    key = (n, margin, size)
    if key not in _maps:
        u = ((np.arange(size, dtype=np.float32) + 0.5) * n / size + margin - 0.5).astype(np.float32)
        _maps[key] = cv2.convertMaps(*np.meshgrid(u, u), cv2.CV_16SC2)  # fixed-point maps: faster remap
        if len(_maps) > 64:
            _maps.pop(next(iter(_maps)))
    return _maps[key]


def paint(rate, kind, faint, n, margin, tile=256):
    """RGBA tile (tile x tile) from rate / kind / faint on the coarse grid ((n + 2 margin)^2, see _upsample_maps).
    Returns None when the tile is dry."""
    wet = rate >= DRAW_MIN * 0.5
    if not wet.any():
        return None
    m1, m2 = _upsample_maps(n, margin, tile * SS)
    lg = np.log(np.maximum(rate, FLOOR)).astype(np.float32)
    lg = cv2.GaussianBlur(lg, (0, 0), BLUR, borderType=cv2.BORDER_REPLICATE)
    v = cv2.remap(lg, m1, m2, cv2.INTER_CUBIC, borderMode=cv2.BORDER_REPLICATE)
    # |.|: cubic undershoot below Q_LO folds back to small values, which are dry as well
    cls = cv2.LUT(cv2.convertScaleAbs(v, alpha=Q_K, beta=-Q_LO * Q_K), _CLASS_LUT)
    if not cls.any():
        return None
    idx = cls.copy()
    # contour: inner side of every boundary to a lower class (incl. the outer edge to dry)
    idx[cv2.erode(cls, _KERNEL, borderType=cv2.BORDER_REPLICATE) < cls] += _NC
    # precipitation kind: smooth boundaries too (indicator per kind present, upsampled, the largest wins)
    if kind is not None:
        present = np.unique(kind[wet])
        if len(present) == 1:
            idx += np.uint8(present[0] * 2 * _NC)
        elif len(present) > 1:
            best, ks = None, np.zeros(cls.shape, np.uint8)
            for k in present:
                ind = cv2.GaussianBlur((kind == k).astype(np.float32), (0, 0), BLUR, borderType=cv2.BORDER_REPLICATE)
                w = cv2.remap(ind, m1, m2, cv2.INTER_LINEAR, borderMode=cv2.BORDER_REPLICATE)
                if best is None:
                    best, ks[:] = w, k
                else:
                    more = w > best
                    ks[more], best = k, np.maximum(best, w)
            idx += ks * np.uint8(2 * _NC)
    if faint is not None and faint.any():
        f = cv2.remap(faint.astype(np.float32), m1, m2, cv2.INTER_LINEAR, borderMode=cv2.BORDER_REPLICATE) > 0.5
        idx[f] += np.uint8(_NK * 2 * _NC)
    rgba = cv2.LUT(cv2.merge([idx] * 4), _PAINT)  # premultiplied
    small = cv2.resize(rgba, (tile, tile), interpolation=cv2.INTER_AREA).astype(np.float32)
    a = small[..., 3:4]
    small[..., :3] *= 255.0 / np.maximum(a, 1.0)  # back to straight alpha
    return np.clip(small + 0.5, 0, 255).astype(np.uint8)
