# Renders the rainy app icons (PNG) from the same geometry as logo.svg.
# Run: python tools/make_icons.py   (needs Pillow)
import math
from pathlib import Path
from PIL import Image, ImageDraw

NAVY = (11, 42, 91, 255)
SUN = (245, 167, 0, 255)
OUT = Path(__file__).resolve().parent.parent


def bezier(p0, p1, p2, p3, n=400):
    for i in range(n + 1):
        t = i / n
        u = 1 - t
        yield (u**3 * p0[0] + 3 * u * u * t * p1[0] + 3 * u * t * t * p2[0] + t**3 * p3[0],
               u**3 * p0[1] + 3 * u * u * t * p1[1] + 3 * u * t * t * p2[1] + t**3 * p3[1])


def drop_outline():
    """Points along the drop (viewBox 0..64), starting at the tip, clockwise."""
    pts = list(bezier((32, 5), (32, 5), (50, 27), (50, 40)))
    pts += [(32 + 18 * math.cos(a), 40 + 18 * math.sin(a))
            for a in (math.pi * i / 400 for i in range(401))]  # bottom half circle, right to left
    pts += list(bezier((14, 40), (14, 27), (32, 5), (32, 5)))
    return pts


def render(size, background):
    scale = 8  # supersampling
    s = size * scale
    img = Image.new("RGBA", (s, s), background)
    d = ImageDraw.Draw(img)
    pad = 0.16 * s
    k = (s - 2 * pad) / 64
    pts = drop_outline()
    lengths = [0.0]
    for a, b in zip(pts, pts[1:]):
        lengths.append(lengths[-1] + math.dist(a, b))
    total = lengths[-1]
    r = 2 * k  # stroke width 4 in viewBox units
    for (x, y), l in zip(pts, lengths):
        if 0.09 * total < l < 0.185 * total:  # the gap for the sun dot
            continue
        cx, cy = pad + x * k, pad + y * k
        d.ellipse([cx - r, cy - r, cx + r, cy + r], fill=NAVY)
    cx, cy, rr = pad + 42.7 * k, pad + 20.4 * k, 2.8 * k
    d.ellipse([cx - rr, cy - rr, cx + rr, cy + rr], fill=SUN)
    return img.resize((size, size), Image.LANCZOS)


if __name__ == "__main__":
    render(180, (255, 255, 255, 255)).convert("RGB").save(OUT / "apple-touch-icon.png")
    render(192, (255, 255, 255, 255)).convert("RGB").save(OUT / "icon-192.png")
    render(512, (255, 255, 255, 255)).convert("RGB").save(OUT / "icon-512.png")
    print("icons written to", OUT)
