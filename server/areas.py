"""The town of the user's blue dot as an area (app v1.63: a light tint and its boundary line on the map).

User, 26.11.2026: "die Stadt, in der ich mich befinde, sollte besser dargestellt werden, mit recht früh sichtbaren
Außengrenzen ... das ganze Stadtgebiet leicht hervorheben farblich" (picked variant B of four).

Data: Overture Maps "divisions / division_area" (OpenStreetMap boundaries, ODbL): localities, localadmins and
town-sized counties and regions, cut to the app's areas and simplified by tools/city_areas.py into one SQLite file
with an R*Tree of the bounding boxes (DATA_DIR/areas.sqlite). The lookup runs here, so no third party learns the
user's position (Overpass and Nominatim want an identifying User-Agent, Photon's public instance has no polygons).
Without the file the endpoint answers {} (the app then shows no area).

  GET /api/geo/area?lat=&lon=&name=  -> {"name", "geometry"} (GeoJSON) or {} when no area holds the point

Pick: among the areas that hold the point, the smallest whose names include `name` (the place name the app shows),
else the smallest whose name starts with it or it with that name ("Frankfurt" / "Frankfurt am Main"); else none - the
smallest locality would often be a borough (Munich's localities are its districts, the town itself is a county).
"""
import json
import os
import sqlite3
import threading

import numpy as np

DB_PATH = os.path.join(os.environ.get("DATA_DIR", os.path.dirname(os.path.abspath(__file__))), "areas.sqlite")
_local = threading.local()
_cache, _cache_lock = {}, threading.Lock()


def _db():
    con = getattr(_local, "con", None)
    if con is None:
        if not os.path.exists(DB_PATH):
            return None
        con = _local.con = sqlite3.connect(f"file:{DB_PATH}?mode=ro", uri=True, check_same_thread=False)
    return con


def _rings(geom):
    polys = geom["coordinates"] if geom["type"] == "MultiPolygon" else [geom["coordinates"]]
    for poly in polys:
        for ring in poly:
            yield np.asarray(ring, dtype=np.float64)


def contains(geom, lon, lat):
    """Even-odd rule over every ring (holes included)."""
    inside = False
    for r in _rings(geom):
        x0, y0, x1, y1 = r[:-1, 0], r[:-1, 1], r[1:, 0], r[1:, 1]
        cross = (y0 > lat) != (y1 > lat)
        with np.errstate(divide="ignore", invalid="ignore"):
            xs = x0 + (lat - y0) * (x1 - x0) / (y1 - y0)
        inside ^= bool(np.count_nonzero(cross & (lon < xs)) % 2)
    return inside


def lookup(lat, lon, name=""):
    """The area dict for the app ({} when none)."""
    key = (round(lat, 3), round(lon, 3), name)
    with _cache_lock:
        if key in _cache:
            return _cache[key]
    con = _db()
    if con is None:
        return {}
    rows = con.execute("SELECT a.name, a.names, a.subtype, a.km2, a.geom FROM area_idx i JOIN area a ON a.id = i.id "
                       "WHERE i.xmin <= ? AND i.xmax >= ? AND i.ymin <= ? AND i.ymax >= ?", (lon, lon, lat, lat)).fetchall()
    hits = []
    for nm, names, subtype, km2, g in rows:
        geom = json.loads(g)
        if contains(geom, lon, lat):
            hits.append((km2, nm, set(json.loads(names)), subtype, geom))
    want = name.strip().casefold()
    names = [(h, {n.casefold() for n in h[2] | {h[1]}}) for h in hits]
    pool = [h for h, ns in names if want in ns] if want else []
    if want and not pool:
        pool = [h for h, ns in names if any(n.startswith(want) or want.startswith(n) for n in ns)]
    out = {}
    if pool:
        km2, nm, _, _, geom = min(pool, key=lambda h: h[0])
        out = {"name": nm, "geometry": geom}
    with _cache_lock:
        _cache[key] = out
        while len(_cache) > 2000:
            _cache.pop(next(iter(_cache)))
    return out
