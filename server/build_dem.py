"""Terrain height on the Europe radar grid, for the orographic shadow "oro" (rainy_server.oro_forecast).

Asks the local Open-Meteo instance (the deploy's own container, http://openmeteo:8080/v1/elevation; nothing leaves
the server under our name) for the height of every 4 km cell centre of the OPERA grid (EuropeOpera at EU_KM=2, cell
pairs averaged = SPROG_DOWN 2) and stores it as int16 metres (sea and no data = 0):

  docker exec -w /app deploy-api-eu-1 python build_dem.py /tmp/dem_eu4km.npz   (~10k requests, ~6 min)

The file is committed as server/dem_eu4km.npz (copied into the image by the Dockerfile).
"""
import json
import sys
import time
import urllib.request
from concurrent.futures import ThreadPoolExecutor

import numpy as np

import rainy_server as rs

OUT = sys.argv[1] if len(sys.argv) > 1 else "dem_eu4km.npz"
URL = "http://openmeteo:8080/v1/elevation?latitude={}&longitude={}"
D = 2  # 4 km cells on the 2 km grid
eu = rs.REGIONS["eu"]
ny, nx = eu.NY // D, eu.NX // D
rows, cols = np.mgrid[0:ny, 0:nx]
# the 4 km cell centre = the corner between the 2x2 fine pixels: fine (col, row) = D * c + (D - 1) / 2
lon, lat = eu.pixel_lonlat(cols.ravel() * D + (D - 1) / 2.0, rows.ravel() * D + (D - 1) / 2.0)
lon, lat = np.asarray(lon, np.float64), np.asarray(lat, np.float64)
h = np.zeros(lon.size, np.float32)
t0 = time.time()


def fetch(i):
    la = ",".join(f"{v:.4f}" for v in lat[i:i + 100])
    lo = ",".join(f"{v:.4f}" for v in lon[i:i + 100])
    for attempt in range(3):
        try:
            with urllib.request.urlopen(URL.format(la, lo), timeout=30) as r:
                e = json.loads(r.read())["elevation"]
            h[i:i + len(e)] = [0.0 if v is None or v != v else v for v in e]
            return
        except Exception as ex:  # transient: retry, then leave 0
            if attempt == 2:
                print("failed", i, ex, flush=True)
            time.sleep(1)


with ThreadPoolExecutor(4) as ex:  # a light job: 4 requests at a time to the local instance
    for k, _ in enumerate(ex.map(fetch, range(0, lon.size, 100))):
        if k % 1000 == 0:
            print(f"{k * 100}/{lon.size} {time.time() - t0:.0f} s", flush=True)
h = np.clip(h, 0, 5000).reshape(ny, nx).astype(np.int16)
np.savez_compressed(OUT, h=h, down=D, km=eu.KM_PER_PX * D)
print("done", OUT, h.shape, "max", int(h.max()), f"{time.time() - t0:.0f} s")
