"""Thunderstorms: DWD automatic short-range warnings (NowCastMIX) and satellite lightning, for the map and alerts.

Warnings: DWD "Autowarn_Vorhersage" (WFS, open data) - NowCastMIX areas where a thunderstorm (or heavy rain) is
expected within about the next hour, updated every 5 min, with onset/expiry and severity (Minor = Gewitter,
Moderate = starkes Gewitter, Severe = schweres Gewitter, Extreme = extremes Gewitter). Ready-made by the DWD from
radar cell tracking, lightning and the model; nothing is computed here.
Lightning: MTG Lightning Imager "accumulated flash area" (EUMETSAT View WMS, every 5 min, CC BY), fetched once for
Germany and served to all users as one image.
"""
import io
import json
import re
import threading
import time
import urllib.parse
from datetime import datetime, timezone

import cv2
import numpy as np
from PIL import Image

from mosaic import merc

WFS = "https://maps.dwd.de/geoserver/dwd/ows"
EUM = "https://view.eumetsat.int/geoserver/ows"
GROUPS = ("Gewitter", "Starkregen")       # warning groups kept (map + alerts use thunderstorms; heavy rain is shown)
LEVEL = {"Minor": "Gewitter", "Moderate": "Starkes Gewitter", "Severe": "Schweres Gewitter", "Extreme": "Extremes Gewitter"}
LIGHTNING_BOX = (4.5, 46.5, 16.5, 55.8)  # lon/lat box around Germany
LIGHTNING_SIZE = (720, 900)
REFRESH_S = 120
# DWD area (lon, lat ranges incl. the neighbours NowCastMIX covers): they do not overlap, so they tell which
# coordinate of a feature is the longitude (axis order check, server-v1.79)
LON_RANGE, LAT_RANGE = (-10.0, 30.0), (40.0, 62.0)
# lightning: the WMS layer's data time; the image arrives ~15 min after it (20.11.2026), so flashes are dropped only
# when the data is clearly stale (feed stalled or our fetch failing), not at the normal delivery delay
LIGHTNING_TIME = "https://view.eumetsat.int/geoserver/mtg_fd/li_afa/ows?service=WMS&version=1.3.0&request=GetCapabilities"
LIGHTNING_MAX_AGE_MS = 35 * 60_000


def iso_ms(s):
    return int(datetime.strptime(s[:19], "%Y-%m-%dT%H:%M:%S").replace(tzinfo=timezone.utc).timestamp() * 1000)


def lon_lat(coords):
    """Polygon / MultiPolygon coordinates in lon/lat order, or None when the axis order cannot be told.
    The DWD WFS stores the axis order per feature, not per request: in one GetFeature answer (20.11.2026, same
    srsName EPSG:4326) most features came lon/lat and some lat/lon, and srsName CRS84, the EPSG URI, WFS 1.0.0 or
    EPSG:3857 did not change that (3857 just projects the swapped pair). So each feature is checked against the DWD
    area instead of trusting the request (server-v1.79; server-v1.58 swapped all of them, which put every
    lon/lat-stored storm near the Horn of Africa)."""
    pts = []
    stack = [coords]
    while stack:
        c = stack.pop()
        if c and isinstance(c[0], (int, float)):
            pts.append(c[:2])
        elif c:
            stack.extend(c)
    if not pts:
        return None
    a = np.asarray(pts, np.float64)
    x, y = a[:, 0], a[:, 1]
    ok = lambda v, r: bool(((v >= r[0]) & (v <= r[1])).all())
    if ok(x, LON_RANGE) and ok(y, LAT_RANGE):
        return coords
    if ok(x, LAT_RANGE) and ok(y, LON_RANGE):
        flip = lambda c: [c[1], c[0]] if c and not isinstance(c[0], list) else [flip(v) for v in c]
        return flip(coords)
    return None


def inside(lon, lat, ring):
    """Ray casting point-in-polygon for one ring [[lon, lat], ...]."""
    xs, ys = np.asarray([p[0] for p in ring]), np.asarray([p[1] for p in ring])
    x2, y2 = np.roll(xs, -1), np.roll(ys, -1)
    cross = (ys > lat) != (y2 > lat)
    xint = xs + (lat - ys) * (x2 - xs) / np.where(y2 == ys, 1e-12, y2 - ys)
    return bool(np.count_nonzero(cross & (lon < xint)) % 2)


class Thunder:
    def __init__(self, fetch, log):
        self.fetch, self.log = fetch, log
        self.lock = threading.Lock()
        self.warnings = []       # [{id, group, level, severity, onset, expires, rings}]
        self._geojson, self._feats = b'{"type":"FeatureCollection","features":[]}', []
        self._lightning = None   # (data time ms, [[lat, lon], ...])
        self.lightning_time = None  # WMS time of the last image fetched

    # Served as properties so that the endpoints (rainy_server.py, read under self.lock) never show warnings that
    # expired or lightning that went stale since the last successful refresh, e.g. while the DWD / EUMETSAT fetch
    # fails (server-v1.79). devtools scenarios set both directly.
    @property
    def geojson(self):
        """Map areas (GeoJSON bytes) without the warnings that have expired by now."""
        raw, feats = self._geojson, self._feats
        if feats is None:
            return raw
        now = int(time.time() * 1000)
        live = [f for f in feats if f["properties"]["expires"] > now]
        return raw if len(live) == len(feats) else json.dumps({"type": "FeatureCollection", "features": live}).encode()

    @geojson.setter
    def geojson(self, body):
        self._geojson, self._feats = body, None

    @property
    def lightning(self):
        """(data time ms, flash clusters) or None when the data is older than LIGHTNING_MAX_AGE_MS."""
        li = self._lightning
        return li if li and time.time() * 1000 - li[0] <= LIGHTNING_MAX_AGE_MS else None

    @lightning.setter
    def lightning(self, li):
        self._lightning = li

    # ---------- warnings ----------
    def refresh_warnings(self):
        now = datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")
        groups = ",".join(f"'{g}'" for g in GROUPS)
        q = urllib.parse.urlencode({
            "service": "WFS", "version": "2.0.0", "request": "GetFeature", "typeName": "dwd:Autowarn_Vorhersage",
            "outputFormat": "application/json", "srsName": "EPSG:4326",
            "CQL_FILTER": f"EXPIRES > '{now}' AND EC_GROUP IN ({groups})",
            "propertyName": "ID_ALERT,EC_GROUP,EC_II,SEVERITY,ONSET,EXPIRES,THE_GEOM"})
        d = json.loads(self.fetch(f"{WFS}?{q}", timeout=60))
        out, feats = [], []
        for f in d.get("features", []):
            p, g = f["properties"], f.get("geometry") or {}
            # GeoJSON, Leaflet and inside() want lon/lat; the WFS mixes both orders (see lon_lat)
            coords = lon_lat(g.get("coordinates") or [])
            if coords is None:
                self.log(f"thunder: skipped warning {p.get('ID_ALERT')}: coordinates outside the DWD area")
                continue
            g = {**g, "coordinates": coords}
            polys = g.get("coordinates", []) if g.get("type") == "MultiPolygon" else [g.get("coordinates", [])] if g.get("type") == "Polygon" else []
            rings = [poly[0] for poly in polys if poly]
            if not rings:
                continue
            w = {"id": p.get("ID_ALERT") or f.get("id"), "group": p["EC_GROUP"], "severity": p.get("SEVERITY", "Minor"),
                 "level": LEVEL.get(p.get("SEVERITY"), "Gewitter") if p["EC_GROUP"] == "Gewitter" else "Starkregen",
                 "onset": iso_ms(p["ONSET"]), "expires": iso_ms(p["EXPIRES"]), "rings": rings}
            out.append(w)
            feats.append({"type": "Feature", "geometry": g, "properties": {k: w[k] for k in ("group", "level", "severity", "onset", "expires")}})
        with self.lock:
            self.warnings = out
            self._geojson, self._feats = json.dumps({"type": "FeatureCollection", "features": feats}).encode(), feats

    def at(self, lat, lon, now_ms=None, within_ms=3600_000):
        """Warnings covering a place that are active now or start within the next hour, earliest first."""
        now_ms = now_ms or int(time.time() * 1000)
        with self.lock:
            ws = list(self.warnings)
        hit = [w for w in ws if w["onset"] <= now_ms + within_ms and w["expires"] > now_ms and any(inside(lon, lat, r) for r in w["rings"])]
        return sorted(hit, key=lambda w: w["onset"])

    # ---------- lightning ----------
    def refresh_lightning(self):
        # the data time from the layer's own capabilities (~7 KB): the map gets the time of the flashes, not of our
        # fetch, and an unchanged time step is not fetched again (server-v1.79)
        t = None
        try:
            m = re.search(rb'<Dimension name="time" default="([0-9T:.Z-]+)"', self.fetch(LIGHTNING_TIME, timeout=30))
            t = m.group(1).decode() if m else None
        except Exception as e:
            self.log(f"thunder: lightning time failed: {e}")
        if t and t == self.lightning_time:
            return
        (x0, x1), (y0, y1) = merc(np.array(LIGHTNING_BOX[0::2]), np.array(LIGHTNING_BOX[1::2]))
        url = (f"{EUM}?service=WMS&version=1.3.0&request=GetMap&layers=mtg_fd:li_afa&styles=&crs=EPSG:3857"
               f"&bbox={x0},{y0},{x1},{y1}&width={LIGHTNING_SIZE[0]}&height={LIGHTNING_SIZE[1]}&format=image/png&transparent=true")
        img = Image.open(io.BytesIO(self.fetch(url + (f"&time={t}" if t else ""), timeout=60))).convert("RGBA")
        self.set_lightning(np.asarray(img)[..., 3] > 0, iso_ms(t) if t else None)
        self.lightning_time = t

    def set_lightning(self, flash, ts=None):
        """Flash areas (image over LIGHTNING_BOX in Web Mercator) -> one point per flash cluster for the map."""
        n, _, _, cent = cv2.connectedComponentsWithStats(flash.astype(np.uint8), connectivity=8)
        h, w = flash.shape
        (x0, x1), (y0, y1) = merc(np.array(LIGHTNING_BOX[0::2]), np.array(LIGHTNING_BOX[1::2]))
        pts = []
        for cx, cy in cent[1:301]:
            mx, my = x0 + (cx + 0.5) / w * (x1 - x0), y1 - (cy + 0.5) / h * (y1 - y0)
            lon = mx / 20037508.342789244 * 180.0
            lat = np.degrees(2 * np.arctan(np.exp(my / 20037508.342789244 * np.pi)) - np.pi / 2)
            pts.append([round(float(lat), 3), round(float(lon), 3)])
        with self.lock:
            self.lightning = (ts or int(time.time() * 1000), pts)

    def loop(self):
        while True:
            for fn in () if getattr(self, "frozen", False) else (self.refresh_warnings, self.refresh_lightning):  # devtools scenario
                try:
                    fn()
                except Exception as e:
                    self.log(f"thunder: {fn.__name__} failed: {e}")
            time.sleep(REFRESH_S)
