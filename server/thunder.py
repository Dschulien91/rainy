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


def iso_ms(s):
    return int(datetime.strptime(s[:19], "%Y-%m-%dT%H:%M:%S").replace(tzinfo=timezone.utc).timestamp() * 1000)


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
        self.geojson = b'{"type":"FeatureCollection","features":[]}'
        self.lightning = None    # (ts ms, png bytes)

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
            self.geojson = json.dumps({"type": "FeatureCollection", "features": feats}).encode()

    def at(self, lat, lon, now_ms=None, within_ms=3600_000):
        """Warnings covering a place that are active now or start within the next hour, earliest first."""
        now_ms = now_ms or int(time.time() * 1000)
        with self.lock:
            ws = list(self.warnings)
        hit = [w for w in ws if w["onset"] <= now_ms + within_ms and w["expires"] > now_ms and any(inside(lon, lat, r) for r in w["rings"])]
        return sorted(hit, key=lambda w: w["onset"])

    # ---------- lightning ----------
    def refresh_lightning(self):
        (x0, x1), (y0, y1) = merc(np.array(LIGHTNING_BOX[0::2]), np.array(LIGHTNING_BOX[1::2]))
        url = (f"{EUM}?service=WMS&version=1.3.0&request=GetMap&layers=mtg_fd:li_afa&styles=&crs=EPSG:3857"
               f"&bbox={x0},{y0},{x1},{y1}&width={LIGHTNING_SIZE[0]}&height={LIGHTNING_SIZE[1]}&format=image/png&transparent=true")
        img = Image.open(io.BytesIO(self.fetch(url, timeout=60))).convert("RGBA")
        self.set_lightning(np.asarray(img)[..., 3] > 0)

    def set_lightning(self, flash):
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
            self.lightning = (int(time.time() * 1000), pts)

    def loop(self):
        while True:
            for fn in (self.refresh_warnings, self.refresh_lightning):
                try:
                    fn()
                except Exception as e:
                    self.log(f"thunder: {fn.__name__} failed: {e}")
            time.sleep(REFRESH_S)
