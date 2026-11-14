"""North America for the rain radar: NOAA MRMS (Multi-Radar Multi-Sensor), the US national radar mosaic, which also
takes in the Canadian radars near the border (14.11.2026: 84 % of 49-52 N covered, 42 % of 52-55 N).

Source: s3://noaa-mrms-pds (NOAA open data, no account), product CONUS/PrecipRate_00.00 - surface precipitation rate
(mm/h), 0.01 deg (~1 km), a new file every 2 minutes, ~2-5 min after the measurement. GRIB2, data packed as a 16-bit
PNG (data representation template 5.41): value = (reference + raw * 2^E) / 10^D; -3 = outside the radars' reach.
Read here without a GRIB library (the image has none): only the sections this one product needs.

The instance (compose service api-na, behind /na/, env NA=1) runs the Europe edition's whole pipeline - nowcast,
tiles, timeline, verification against its own later images, warm start - with this region in place of OPERA. MRMS
writes even minutes: the 5-minute step T takes the file of T, else T+1 min, else T-1 min (a one-minute shift does not
matter for 5-minute rain motion).
"""
import gzip
import io
import re
import struct
import time
import urllib.request
from datetime import datetime, timedelta, timezone

import numpy as np
from PIL import Image

BUCKET = "https://noaa-mrms-pds.s3.amazonaws.com"
PRODUCT = "CONUS/PrecipRate_00.00"
LAT1, LON0, DEG0 = 54.995, -129.995, 0.01   # centre of MRMS's first cell (north-west), cell size
NX0, NY0 = 7000, 3500


def decode(raw):
    """MRMS GRIB2 (PNG packed) -> float32 mm/h on the 0.01 deg grid, < 0 = no data."""
    if raw[:2] == b"\x1f\x8b":
        raw = gzip.decompress(raw)
    if raw[:4] != b"GRIB" or raw[7] != 2:
        raise ValueError("not GRIB2")
    i, ref, E, D, png, ni, nj = 16, None, 0, 0, None, None, None
    while i < len(raw) - 4 and raw[i:i + 4] != b"7777":
        ln, num = struct.unpack(">IB", raw[i:i + 5])
        sec = raw[i:i + ln]
        if num == 3:
            ni, nj = struct.unpack(">II", sec[30:38])
        elif num == 5:
            if struct.unpack(">H", sec[9:11])[0] != 41:
                raise ValueError("not PNG packed")
            ref = struct.unpack(">f", sec[11:15])[0]
            E, D = struct.unpack(">hh", sec[15:19])
        elif num == 7:
            png = sec[5:]
        i += ln
    if png is None or ref is None or (ni, nj) != (NX0, NY0):
        raise ValueError(f"unexpected MRMS file (grid {ni} x {nj})")
    a = np.asarray(Image.open(io.BytesIO(png))).astype(np.float32)
    return (ref + a * 2.0 ** E) / 10.0 ** D


class MrmsListing:
    """5-minute steps -> the URL of the MRMS file used for it."""

    def __init__(self, fetch):
        self.fetch = fetch

    def times(self):
        now = datetime.now(timezone.utc)
        days = [now] + ([now - timedelta(days=1)] if now.hour < 3 else [])
        files = {}
        for day in days:
            token = None
            while True:
                url = f"{BUCKET}/?list-type=2&prefix={PRODUCT}/{day:%Y%m%d}/&max-keys=1000" + (
                    f"&continuation-token={urllib.request.quote(token)}" if token else "")
                xml = self.fetch(url).decode()
                for key in re.findall(r"<Key>([^<]+\.grib2\.gz)</Key>", xml):
                    m = re.search(r"_(\d{8})-(\d{6})\.grib2\.gz$", key)
                    if m:
                        t = int(datetime.strptime(m.group(1) + m.group(2), "%Y%m%d%H%M%S").replace(tzinfo=timezone.utc).timestamp() * 1000)
                        files[t // 60_000 * 60_000] = f"{BUCKET}/{key}"
                token = re.search(r"<NextContinuationToken>([^<]+)</NextContinuationToken>", xml)
                if not token:
                    break
                token = token.group(1)
        out = {}
        step = 5 * 60_000
        for t in range(min(files, default=0) // step * step, max(files, default=0) + 1, step):
            for dt in (0, 60_000, -60_000):
                if t + dt in files:
                    out[t] = files[t + dt]
                    break
        return out


def block_mean(a, k, nodata):
    """k x k block mean of a (< 0 = no data), no data where most of the block has none."""
    h, w = a.shape[0] // k * k, a.shape[1] // k * k
    ok = a[:h, :w] >= 0
    cov = ok.reshape(h // k, k, w // k, k).mean(axis=(1, 3))
    tot = np.where(ok, a[:h, :w], 0).reshape(h // k, k, w // k, k).mean(axis=(1, 3))
    return np.where(cov >= 0.5, tot / np.maximum(cov, 1e-6), nodata).astype(np.float32)


def merc_to_lonlat(x, y):
    r = 6378137.0
    return np.degrees(np.asarray(x) / r), np.degrees(np.arctan(np.sinh(np.asarray(y) / r)))


if __name__ == "__main__":  # quick check: newest file, decode time, coverage
    get = lambda u: urllib.request.urlopen(urllib.request.Request(u, headers={"User-Agent": "Mozilla/5.0"}), timeout=60).read()
    t = MrmsListing(get).times()
    ts = max(t)
    t0 = time.time()
    v = decode(get(t[ts]))
    print(len(t), "steps, newest", datetime.fromtimestamp(ts / 1000, timezone.utc), f"decoded in {time.time() - t0:.1f}s,",
          f"covered {np.mean(v >= 0):.2f}, wet {np.mean(v >= 0.1):.3f}, max {v.max():.0f} mm/h")
