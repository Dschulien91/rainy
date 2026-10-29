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
