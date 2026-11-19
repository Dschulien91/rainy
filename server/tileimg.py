"""Encoding of the app's data tiles: PNG or lossless WebP (server-v1.69).

The app asks for .webp when the browser decodes it (all current ones) and .png otherwise; both hold the same values
bit for bit. Measured on the server (4 cores, 19.11.2026) with live tiles:
  rain data tiles (RGB 924 x 660): WebP method 0 is 10-30 % smaller than PNG level 6 and ~3x faster to encode
    (20-28 ms vs 68-96 ms)
  sunny atlases (grey + alpha 2048 x 768): WebP method 1 is 35-75 % smaller than PNG level 3 (e.g. 268 -> 166 KB,
    98 -> 49 KB, 28 -> 7 KB) at ~2-5x the encode time (28-290 ms vs 17-63 ms), once per tile and run, then cached
"""
import io

from PIL import Image

WEBP = {"rain": {"method": 0, "quality": 0}, "cloud": {"method": 1, "quality": 25}}
TYPES = {"png": "image/png", "webp": "image/webp"}


def encode(img, fmt="png", kind="rain", png_level=6):
    """PIL image -> bytes. The colour under fully transparent pixels may change in WebP (exact=False): the app
    never reads it."""
    buf = io.BytesIO()
    if fmt == "webp":
        if img.mode == "LA":  # WebP has no grey + alpha mode; the grey is stored once (subtract-green transform)
            img = img.convert("RGBA")
        img.save(buf, "WEBP", lossless=True, exact=False, **WEBP[kind])
    else:
        img.save(buf, "PNG", compress_level=png_level)
    return buf.getvalue()


def transcode(png, fmt, kind="cloud"):
    """A PNG from a source that only makes PNGs, in the asked format."""
    return png if fmt == "png" or png is None else encode(Image.open(io.BytesIO(png)), fmt, kind)
