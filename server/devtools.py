"""Development only (env RAINY_DEV=1): fake weather scenarios and a dry-run alert outbox for end-to-end tests.

Never active in production. A scenario paints a storm into the live German grids that moves towards a place:
rain with a snow part and a hail core, a DWD-style thunderstorm warning over the place and lightning. Push
subscriptions whose endpoint starts with "dev:" are not sent anywhere; their messages go to an outbox instead,
so the whole alert flow (alert, probability, 'remind me' button, reminder / all-clear) can be checked.
"""
import time

import numpy as np

import ptype
from mosaic import merc

OUTBOX = []


def storm(de, pt, th, lat, lon, arrive_min=40, speed_kmh=30):
    """Storm blob moving east, reaching (lat, lon) after arrive_min; freezes the DWD grids until reset."""
    de.frozen = True
    cx, cy = merc(lon, lat)
    cols, rows = np.meshgrid(np.arange(de.nx), np.arange(de.ny))
    x0, _, _, y1 = de.BBOX
    X, Y = x0 + (cols + 0.5) * de.CELL, y1 - (rows + 0.5) * de.CELL
    k = 1 / np.cos(np.radians(lat))  # ground metres -> Web Mercator metres
    now = max(de.obs) if de.obs else int(time.time() * 1000)

    def blob(ts):
        shift = (arrive_min - (ts - now) / 60_000) / 60 * speed_kmh * 1000 * k  # blob centre west of the place
        d2 = ((X - (cx - shift)) / k) ** 2 + ((Y - cy) / k) ** 2
        return (12.0 * np.exp(-d2 / (2 * 5000.0 ** 2))).astype(np.float32), Y > cy - 3000 * k, d2 < 2500.0 ** 2

    for grids in (de.obs, de.fc):
        for ts in list(grids):
            rate, _, _ = blob(ts)
            grids[ts] = np.maximum(grids[ts].astype(np.float32), np.where(rate >= 0.1, rate, 0)).astype(np.float16)
    with pt.lock:
        for ts in list(de.obs):
            rate, north, core = blob(ts)
            g = np.full((de.ny, de.nx), ptype.RAIN, np.uint8)
            g[north] = ptype.SNOW
            g[core] = ptype.HAIL
            pt.grids[ts] = g
    ring = [[lon - 0.3, lat - 0.2], [lon + 0.3, lat - 0.2], [lon + 0.3, lat + 0.2], [lon - 0.3, lat + 0.2], [lon - 0.3, lat - 0.2]]
    onset = int(time.time() * 1000) + (arrive_min - 10) * 60_000
    w = {"id": f"dev-{onset}", "group": "Gewitter", "severity": "Moderate", "level": "Starkes Gewitter",
         "onset": onset, "expires": onset + 60 * 60_000, "rings": [ring]}
    th.frozen = True  # the 2-min DWD refresh must not replace the scenario
    with th.lock:
        th.warnings = [w]
        th.geojson = (b'{"type":"FeatureCollection","features":[{"type":"Feature","geometry":{"type":"Polygon","coordinates":['
                      + str(ring).encode() + b']},"properties":{"group":"Gewitter","level":"Starkes Gewitter","severity":"Moderate",'
                      + f'"onset":{w["onset"]},"expires":{w["expires"]}'.encode() + b'}}]}')
    # lightning: a few flash clusters in the storm's current position
    sx = lon - arrive_min / 60 * speed_kmh / (111.0 * np.cos(np.radians(lat)))
    rng = np.random.default_rng(1)
    with th.lock:
        th.lightning = (int(time.time() * 1000), [[round(lat + rng.normal(0, 0.04), 3), round(sx + rng.normal(0, 0.06), 3)] for _ in range(12)])
    de.ref = (de.ref or now) + 60_000  # new tile key: browsers must not reuse the tiles from before the scenario


def field(de, pt, seed=1, speed_kmh=40):
    """Widespread rain with showers and a few heavy cells over all of Germany, moving east (look of the rain tiles:
    every colour class, ragged outlines, snow in the north-east). Freezes the DWD grids until reset."""
    import cv2
    de.frozen = True
    rng = np.random.default_rng(seed)
    ny, nx = de.ny, de.nx
    pad = int(2 * 60 * speed_kmh / 60 / de.KM_PER_PX) + 8  # room for 2 h of motion
    f = np.zeros((ny, nx + pad), np.float32)
    for cell_km, amp in ((180, 1.0), (60, 0.7), (20, 0.45), (6, 0.25)):
        n = max(2, int((nx + pad) * de.KM_PER_PX / cell_km))
        g = rng.normal(size=(max(2, int(ny * de.KM_PER_PX / cell_km)) + 1, n + 1)).astype(np.float32)
        f += amp * cv2.resize(g, (nx + pad, ny), interpolation=cv2.INTER_CUBIC)
    base = np.where(f > 0.3, np.minimum(0.05 * np.exp(2.6 * (f - 0.3)), 120.0), 0.0).astype(np.float32)
    now = max(de.obs) if de.obs else int(time.time() * 1000)
    for grids in (de.obs, de.fc):
        for ts in list(grids):
            shift = int(round((ts - now) / 3_600_000 * speed_kmh / de.KM_PER_PX)) + pad // 2
            grids[ts] = base[:, pad - shift:pad - shift + nx].astype(np.float16) if 0 <= shift <= pad else grids[ts]
    with pt.lock:
        for ts in list(pt.grids):
            g = np.full((ny, nx), ptype.RAIN, np.uint8)
            g[: ny // 4, nx // 2:] = ptype.SNOW
            pt.grids[ts] = g
    de.ref = (de.ref or now) + 60_000  # new tile key


def bench(mosaic, z, n=40):
    """Render time per rain tile (ms, this thread only, no cache): the v0.76 look against the plain look before
    (rate per tile pixel + colour table + PNG). Tiles of the latest frame around Germany's centre."""
    import io
    from PIL import Image
    import mosaic as mm
    import palette
    ts = max(f["ts"] for f in mosaic.frames() if not f["fc"])
    key = mosaic.key()
    k = (1 << z) / (2 * mm.WORLD)
    cx, cy = mm.merc(10.4, 51.0)
    x0, y0 = int((cx + mm.WORLD) * k), int((mm.WORLD - cy) * k)
    side = max(1, int(n ** 0.5))
    tiles = [(x0 - side // 2 + i, y0 - side // 2 + j) for i in range(side) for j in range(side)]
    out = {}
    t = time.perf_counter()
    for x, y in tiles:
        mosaic.cache.pop(f"rdwd/{key}/{ts}/{z}/{x}/{y}", None)
        mosaic.rain_tile(key + "-bench", ts, z, x, y)
    out["new_ms"] = round((time.perf_counter() - t) / len(tiles) * 1000, 1)
    t = time.perf_counter()
    for x, y in tiles:
        X, Y = mm.tile_xy(z, x, y)
        rate, faint = mosaic.rain(ts, X, Y)
        rgba = palette.colorize(rate, mosaic.kind(ts, X, Y))
        buf = io.BytesIO()
        Image.fromarray(rgba, "RGBA").save(buf, "PNG", compress_level=6)
    out["old_ms"] = round((time.perf_counter() - t) / len(tiles) * 1000, 1)
    with mosaic.cache_lock:
        for kk in [kk for kk in mosaic.cache if "-bench/" in kk]:
            del mosaic.cache[kk]
    return {"z": z, "tiles": len(tiles), **out}


def reset(de, pt, th):
    de.frozen = False
    de.obs.clear(); de.fc.clear(); de.ref = None  # reloaded from the DWD within a minute
    with pt.lock:
        pt.grids.clear()
    th.frozen = False
    with th.lock:
        th.warnings = []
        th.geojson = b'{"type":"FeatureCollection","features":[]}'
        th.lightning = None
