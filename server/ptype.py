"""Precipitation type (rain, sleet, snow, freezing rain, graupel, hail) from the DWD radar.

Source: DWD HymecNG composite (opendata.dwd.de/weather/radar/composite/hymecng, every 5 min, 1 km polar
stereographic, open data): a hydrometeor classification from the dual-polarisation radars - ready-made, nothing is
estimated here. Its classes are mapped onto our German radar grid. Precipitating pixels without a class get the
class of the nearest classified pixel within ~20 km (the classification has gaps at the edge of rain areas).
Forecast frames keep the latest type for snow, sleet and freezing rain (they change slowly with the air mass);
graupel and hail belong to moving cells, so they are only shown where they were measured.
"""
import io
import threading
from datetime import datetime, timezone

import cv2
import h5py
import numpy as np
from pyproj import Transformer

URL = "https://opendata.dwd.de/weather/radar/composite/hymecng/composite_HymecNG_{:%Y%m%d_%H%M}_000-hd5"
# our kinds (index into the colour tables): 0 rain, 1 sleet, 2 snow, 3 freezing rain, 4 graupel, 5 hail
RAIN, SLEET, SNOW, FREEZING, GRAUPEL, HAIL = range(6)
NAMES = ("rain", "sleet", "snow", "freezing", "graupel", "hail")
CLASS_TO_KIND = {2: RAIN, 3: RAIN, 4: FREEZING, 5: FREEZING, 6: SLEET, 7: SNOW, 8: GRAUPEL, 9: HAIL, 10: HAIL}
FILL_KM = 20
KEEP = 14


def dwd_grid_maps(f, de):
    """cv2.remap tables from our German Web Mercator grid (de) to a DWD composite grid (HDF5 'where' attributes)."""
    w = f["where"].attrs
    proj = w["projdef"].decode() if isinstance(w["projdef"], bytes) else w["projdef"]
    to = Transformer.from_crs("EPSG:4326", proj, always_xy=True)
    ux, uy = to.transform(float(w["UL_lon"]), float(w["UL_lat"]))
    cols, rows = np.meshgrid(np.arange(de.nx), np.arange(de.ny))
    x0, _, _, y1 = de.BBOX
    mx, my = x0 + (cols + 0.5) * de.CELL, y1 - (rows + 0.5) * de.CELL
    lon, lat = Transformer.from_crs("EPSG:3857", "EPSG:4326", always_xy=True).transform(mx, my)
    sx, sy = to.transform(lon, lat)
    return (((sx - ux) / float(w["xscale"]) - 0.5).astype(np.float32),
            ((uy - sy) / float(w["yscale"]) - 0.5).astype(np.float32))


class PrecipType:
    def __init__(self, fetch, log, de):
        self.fetch, self.log, self.de = fetch, log, de
        self.lock = threading.Lock()
        self.grids = {}       # ts -> uint8 kind grid on the DE grid (255 = unknown)
        self.maps = None      # remap tables DE grid -> HymecNG pixels

    def _load(self, ts):
        data = self.fetch(URL.format(datetime.fromtimestamp(ts / 1000, timezone.utc)), timeout=60)
        with h5py.File(io.BytesIO(data), "r") as f:
            cls = f["dataset1/data1/data"][:]
            if self.maps is None:
                self.maps = dwd_grid_maps(f, self.de)
        lut = np.full(256, 255, np.uint8)
        for c, k in CLASS_TO_KIND.items():
            lut[c] = k
        kind = cv2.remap(lut[cls], *self.maps, cv2.INTER_NEAREST, borderMode=cv2.BORDER_CONSTANT, borderValue=255)
        # gaps: take the nearest classified pixel within FILL_KM
        unknown = (kind == 255).astype(np.uint8)
        if unknown.all():
            return kind
        dist, labels = cv2.distanceTransformWithLabels(unknown, cv2.DIST_L2, 5, labelType=cv2.DIST_LABEL_PIXEL)
        known = np.argwhere(unknown.ravel() == 0).ravel()
        lab_to_idx = np.zeros(labels.max() + 1, np.int64)
        lab_to_idx[labels.ravel()[known]] = known
        filled = kind.ravel()[lab_to_idx[labels.ravel()]].reshape(kind.shape)
        km = dist * self.de.KM_PER_PX
        return np.where((kind == 255) & (km <= FILL_KM), filled, kind).astype(np.uint8)

    def update(self, obs_times):
        """Load the classification for the observed radar frames (newest first); returns the number loaded."""
        n = 0
        for ts in sorted(obs_times, reverse=True)[:KEEP]:
            if ts in self.grids:
                continue
            try:
                g = self._load(ts)
            except Exception as e:  # not published yet / gap: the frame stays "rain"
                self.log(f"ptype {datetime.fromtimestamp(ts / 1000, timezone.utc):%H:%M}: {e}")
                continue
            with self.lock:
                self.grids[ts] = g
            n += 1
        with self.lock:
            for ts in sorted(self.grids)[:-KEEP]:
                del self.grids[ts]
        return n

    def kind(self, ts, X, Y, forecast):
        """Kind per Web Mercator point (0 rain .. 5 hail) for a frame; forecast frames use the latest field
        without graupel and hail."""
        with self.lock:
            if not self.grids:
                return None
            if forecast or ts not in self.grids:
                g = self.grids[max(t for t in self.grids if t <= ts)] if any(t <= ts for t in self.grids) else self.grids[max(self.grids)]
            else:
                g = self.grids[ts]
        col, row = (np.round(v).astype(np.int32) for v in self.de.to_pixels(X, Y))
        ok = (col >= 0) & (col < g.shape[1]) & (row >= 0) & (row < g.shape[0])
        k = np.zeros(X.shape, np.uint8)
        k[ok] = g[row[ok], col[ok]]
        k[k == 255] = RAIN
        if forecast:
            k[(k == GRAUPEL) | (k == HAIL)] = RAIN
        return k
