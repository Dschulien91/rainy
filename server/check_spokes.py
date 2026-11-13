"""What does the spoke filter (spokes.py) remove from the European composite's last images?

Lists every removed line with the radar it points at, per image, and the Slovenian line of 13.11.2026 row by row.
Run it in the OPERA instance (it loads 13 composites - too heavy for the throttled third instance), with the current
spokes.py next to it:

  docker exec deploy-api-eu-1 mkdir -p /tmp/t
  for f in spokes.py opera_radars.json check_spokes.py; do docker cp /opt/rainy/server/$f deploy-api-eu-1:/tmp/t/; done
  docker exec -e PYTHONPATH=/tmp/t:/app -w /app deploy-api-eu-1 python /tmp/t/check_spokes.py [images=13]

The running Europe instances (OPERA and own picture, both filter since server-v1.19) report the lines of their newest
image in /eu/api/health and /eun/api/health -> regions.spokes and log "spoke removed: ..." (once an hour per radar)."""
import sys
import time

import numpy as np

import rainy_server as rs
import spokes

rs.EuropeOpera.DESPOKE = False  # the raw images: the filter runs here by hand
eu = rs.REGIONS["eu"]
times = eu.list_times()
n = int(sys.argv[1]) if len(sys.argv) > 1 else 13
ts = sorted(times)[-n:]
st = spokes.sites(eu, rs.mosaic.merc)
print(len(st), "radar sites on the grid")
f = spokes.SpokeFilter(eu.KM_PER_PX, st)
grids = {}
for t in ts:
    grids[t] = eu.load(times[t])
    f.clean(t, grids[t])
for t in ts[::3] + [ts[-1]]:
    t0 = time.time()
    out, lines = f.clean(t, grids[t])
    dt = time.time() - t0
    stamp = rs.datetime.fromtimestamp(t / 1000, rs.timezone.utc).strftime("%H:%M")
    changed = out != grids[t]
    print(stamp, "lines", len(lines), "cells changed", int(changed.sum()),
          "rain removed (mm/h x cells)", round(float((grids[t] - out)[changed].sum()), 1) if changed.any() else 0, f"{dt:.2f} s")
    for length, code, (c, r) in lines:
        lon, lat = eu.pixel_lonlat(c, r)
        a, b = grids[t][max(r - 30, 0):r + 30, max(c - 30, 0):c + 30], out[max(r - 30, 0):r + 30, max(c - 30, 0):c + 30]
        print(f"   {length} km towards {code}, middle {lat:.2f} N {lon:.2f} E; max around it before {float(a.max()):.1f}, after {float(b.max()):.1f} mm/h")
# the Slovenian line of 13.11., row by row, before and after (last image)
t = ts[-1]
out, _ = f.clean(t, grids[t])
lats = np.arange(45.5, 46.41, 0.1)
lons = np.arange(13.9, 14.81, 0.02)
LO, LA = np.meshgrid(lons, lats)
col, row = eu.to_pixels(*rs.mosaic.merc(LO.ravel(), LA.ravel()))
col, row = np.round(col).astype(int).reshape(LO.shape), np.round(row).astype(int).reshape(LO.shape)
for i, la in enumerate(lats):
    print(f"   {la:.1f}N  max before {grids[t][row[i], col[i]].max():6.1f}  after {out[row[i], col[i]].max():6.1f}")
