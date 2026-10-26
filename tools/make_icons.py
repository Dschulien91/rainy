# Renders the rainy app icons (PNG) from the same geometry as logo.svg:
# a raindrop outline tilted by 18 degrees with a smaller filled drop inside.
# Run: python tools/make_icons.py   (needs Pillow)
import math
from pathlib import Path
from PIL import Image, ImageDraw

NAVY = (11, 42, 91, 255)
OUT = Path(__file__).resolve().parent.parent
TILT = math.radians(18)
PIVOT = (32, 36)


def bezier(p0, p1, p2, p3, n=400):
    for i in range(n + 1):
        t = i / n
        u = 1 - t
        yield (u**3 * p0[0] + 3 * u * u * t * p1[0] + 3 * u * t * t * p2[0] + t**3 * p3[0],
               u**3 * p0[1] + 3 * u * u * t * p1[1] + 3 * u * t * t * p2[1] + t**3 * p3[1])


def drop():
    """Points along the drop in viewBox units (0..64), starting at the tip, clockwise."""
    pts = list(bezier((32, 5), (32, 5), (50, 27), (50, 40)))
    pts += [(32 + 18 * math.cos(a), 40 + 18 * math.sin(a)) for a in (math.pi * i / 400 for i in range(401))]
    pts += list(bezier((14, 40), (14, 27), (32, 5), (32, 5)))
    return pts


def rotate(p):
    x, y = p[0] - PIVOT[0], p[1] - PIVOT[1]
    return (PIVOT[0] + x * math.cos(TILT) - y * math.sin(TILT), PIVOT[1] + x * math.sin(TILT) + y * math.cos(TILT))


def render(size):
    scale = 8  # supersampling
    s = size * scale
    img = Image.new("RGBA", (s, s), (255, 255, 255, 255))
    d = ImageDraw.Draw(img)
    pad = 0.14 * s
    k = (s - 2 * pad) / 68  # viewBox -2..66
    to_px = lambda p: (pad + (p[0] + 2) * k, pad + (p[1] + 2) * k)
    r = 3 * k  # stroke width 6
    for p in drop():
        cx, cy = to_px(rotate(p))
        d.ellipse([cx - r, cy - r, cx + r, cy + r], fill=NAVY)
    inner = [to_px(rotate((32 + (x - 32) * 0.42, 42 + (y - 40) * 0.42))) for x, y in drop()]
    d.polygon(inner, fill=NAVY)
    return img.resize((size, size), Image.LANCZOS).convert("RGB")


if __name__ == "__main__":
    for name, size in (("apple-touch-icon.png", 180), ("icon-192.png", 192), ("icon-512.png", 512)):
        render(size).save(OUT / name)
    print("icons written to", OUT)
