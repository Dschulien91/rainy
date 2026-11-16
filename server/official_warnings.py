"""Official severe-weather warnings (thunderstorm, rain, snow/ice, wind, ...) of the national weather services for a place.

Named official_warnings.py, not warnings.py: a warnings.py next to rainy_server.py would shadow Python's own `warnings`
module (numpy and the standard library import it) and break the whole server.

Sources (all keyless, open, fetched with the server's anonymous fetch(); nothing identifying is ever sent):
  dwd         Germany: DWD CAP "COMMUNEUNION_DWD_STAT" (opendata.dwd.de/weather/alerts/cap/), the latest snapshot
              of all warnings as zipped CAP XML in 8 languages, with the warned area as polygons (union of the warned
              municipalities, EXCLUDE_POLYGON = holes). Coast and lake warnings carry only WARNCELLIDs: their areas come
              from the DWD GeoServer WFS (dwd:Warngebiete_Kueste, dwd:Warngebiete_Binnenseen), cached on disk.
  meteoalarm  Europe (EUMETNET MeteoAlarm): JSON feed per country (feeds.meteoalarm.org/api/v1/warnings/feeds-<country>),
              full CAP alerts. Areas are EMMA_ID / NUTS / FIPS / WARNCELLID geocodes or CAP polygons. Geocode
              geometry: MeteoAlarm's own public geocode GeoJSON + alias CSV for redistributors (GitLab
              meteoalarm-pm-group/documents, linked from meteoalarm.org "Redistribution Hub"), downloaded at run time,
              simplified and cached on disk (not kept in the repository: the T&C do not clearly license the geometry).
              Germany is skipped there when the dwd source runs (DWD's own data is more precise).
  nws         USA: api.weather.gov/alerts/active (GeoJSON). Few alerts carry a polygon; the rest name forecast / county /
              marine zones whose geometry is fetched once from api.weather.gov/zones/... and cached on disk.
  eccc        Canada: api.weather.gc.ca/collections/weather-alerts (OGC API, one polygon per alert region, en + fr).

Normalised level (1-4), so all sources look alike in the app:
  MeteoAlarm  awareness_level parameter: 1 green = no warning (dropped), 2 yellow, 3 orange, 4 red.
              Fallback without it: CAP severity Minor 1 (dropped) / Moderate 2 / Severe 3 / Extreme 4.
  DWD         its own CAP severities are shifted against MeteoAlarm (DWD feeds MeteoAlarm Minor -> yellow):
              Minor 2 (Wetterwarnung, yellow), Moderate 3 (markantes Wetter, orange), Severe 4 (Unwetter, red),
              Extreme 4 (extremes Unwetter); "VORABINFORMATION ..." (pre-information) 1.
  NWS         by product kind: Statement / Outlook / Message / Special Weather Statement 1, Advisory 2, Watch 2,
              Warning / Emergency 3, any product with severity Extreme 4 (Tornado / Hurricane / Extreme Wind Warning).
  ECCC        colour when given (yellow 2, orange 3, red 4), else by type: statement 1, advisory 2, watch 2, warning 3.
Normalised event: thunderstorm, rain, snow-ice, wind, fog, heat, cold, flood, coastal, avalanche, fire, other.

Licences / attribution (see ATTRIBUTION; show it in the app next to the warnings):
  MeteoAlarm: terms equivalent to CC BY 4.0 plus: show the national service's name (or "EUMETNET - MeteoAlarm" for
              more than one country), the time of issue ("sent"), a link to meteoalarm.org, the delay disclaimer, keep the
              original text next to any modified one ("text" next to the shortened "description"), real time
              (<5 min average, never >10 min delay).
  DWD:        DWD open data, CC BY 4.0; "Deutscher Wetterdienst" and the geobasis note "(c) GeoBasis-DE / BKG".
  NWS:        US government work, public domain.
  ECCC:       ECCC Data Server End-use Licence (commercial use allowed): "Data Source: Environment and Climate Change
              Canada", no implied endorsement, watches and warnings reproduced without altering content or intent.
Every warning carries "text" (the original description, unchanged) next to the shortened "description".

Use:  w = Warnings(fetch, log, cache_dir, regions=("de",));  threading.Thread(target=w.loop, daemon=True).start()
      w.at(lat, lon, lang="de") -> [warning]      w.areas_geojson((west, south, east, north)) -> FeatureCollection
Test: python official_warnings.py --once eu|na|de [--at LAT,LON ...] [--lang de] [--samples]
"""
import csv
import io
import json
import math
import os
import re
import threading
import time
import xml.etree.ElementTree as ET
import zipfile
from datetime import datetime, timezone

import numpy as np

REFRESH_S = 300                 # each source every 5 min (MeteoAlarm T&C: < 5 min average delay)
LOOKAHEAD_MS = 24 * 3600_000    # warnings starting later than this are not shown yet
SIMPLIFY_DEG = 0.005            # Douglas-Peucker tolerance (~0.5 km): borders of neighbouring areas stay close
CELL_DEG = 1.0                  # grid cell of the point lookup index
GEOMETRY_MAX_AGE_S = 7 * 86400  # static area geometry (MeteoAlarm geocodes, DWD coast / lakes) is checked weekly
ZONE_MAX_AGE_S = 180 * 86400    # NWS zone geometry is re-fetched after half a year
ZONE_FETCH_MAX = 400            # NWS zones fetched per cycle at most (politeness); the rest follow next cycle
ZONE_PAUSE_S = 0.25
DESC_CHARS = 300

DWD_CAP = ("https://opendata.dwd.de/weather/alerts/cap/COMMUNEUNION_DWD_STAT/"
           "Z_CAP_C_EDZW_LATEST_PVW_STATUS_PREMIUMDWD_COMMUNEUNION_MUL.zip")
DWD_WFS = "https://maps.dwd.de/geoserver/dwd/ows"
DWD_CELL_LAYERS = ("dwd:Warngebiete_Kueste", "dwd:Warngebiete_Binnenseen")
MA_FEED = "https://feeds.meteoalarm.org/api/v1/warnings/feeds-{}"
MA_INDEX = "https://feeds.meteoalarm.org/"
MA_DOCS_TREE = "https://gitlab.com/api/v4/projects/meteoalarm-pm-group%2Fdocuments/repository/tree?per_page=100"
MA_DOCS_RAW = "https://gitlab.com/meteoalarm-pm-group/documents/-/raw/master/{}?inline=false"
MA_COUNTRIES = (  # feed slugs as listed on feeds.meteoalarm.org (re-read from there daily)
    "andorra", "austria", "belgium", "bosnia-herzegovina", "bulgaria", "croatia", "cyprus", "czechia", "denmark",
    "estonia", "finland", "france", "germany", "greece", "hungary", "iceland", "ireland", "israel", "italy", "latvia",
    "lithuania", "luxembourg", "malta", "moldova", "montenegro", "netherlands", "norway", "poland", "portugal",
    "republic-of-north-macedonia", "romania", "serbia", "slovakia", "slovenia", "spain", "sweden", "switzerland",
    "ukraine", "united-kingdom")
NWS_ALERTS = "https://api.weather.gov/alerts/active?status=actual"
NWS_ZONE_PREFIX = "https://api.weather.gov/zones/"
ECCC_ALERTS = "https://api.weather.gc.ca/collections/weather-alerts/items?f=json&limit={}&offset={}"
ECCC_PAGE = 500

REGIONS = {"de": ("dwd",), "eu": ("dwd", "meteoalarm"), "na": ("nws", "eccc")}

ATTRIBUTION = {
    "dwd": {"name": "Deutscher Wetterdienst", "url": "https://www.dwd.de/warnungen", "licence": "CC BY 4.0",
            "note": "Geobasisdaten: © GeoBasis-DE / BKG (Daten modifiziert)"},
    "meteoalarm": {"name": "EUMETNET – MeteoAlarm", "url": "https://meteoalarm.org", "licence": "CC BY 4.0 (equivalent) + MeteoAlarm T&C",
                   "note": "Time delays between this website and the www.meteoalarm.org website are possible. For the most "
                           "up-to-date awareness information as published by the participating National Meteorological and "
                           "Hydrological Services, please refer to www.meteoalarm.org."},
    "nws": {"name": "NOAA / National Weather Service", "url": "https://www.weather.gov", "licence": "public domain"},
    "eccc": {"name": "Environment and Climate Change Canada", "url": "https://weather.gc.ca",
             "licence": "ECCC Data Server End-use Licence (https://eccc-msc.github.io/open-data/licence/readme_en/)",
             "note": "Data Source: Environment and Climate Change Canada"},
}

# MeteoAlarm awareness_type number -> normalised event
MA_TYPES = {1: "wind", 2: "snow-ice", 3: "thunderstorm", 4: "fog", 5: "heat", 6: "cold", 7: "coastal", 8: "fire",
            9: "avalanche", 10: "rain", 11: "other", 12: "flood", 13: "flood"}
# DWD eventCode GROUP -> normalised event
DWD_GROUPS = {"THUNDERSTORM": "thunderstorm", "HAIL": "thunderstorm", "TORNADO": "thunderstorm", "WIND": "wind",
              "RAIN": "rain", "SNOWFALL": "snow-ice", "SNOWDRIFT": "snow-ice", "GLAZE": "snow-ice",
              "SLIPPERINESS": "snow-ice", "FROST": "cold", "FOG": "fog", "THAW": "flood", "HEAT": "heat", "UV": "heat"}
SEVERITY_MA = {"Minor": 1, "Moderate": 2, "Severe": 3, "Extreme": 4}
SEVERITY_DWD = {"Minor": 2, "Moderate": 3, "Severe": 4, "Extreme": 4}
COLOUR = {"yellow": 2, "orange": 3, "red": 4}
# free-text event names (NWS, ECCC) -> normalised event; first match wins; words must start at a word boundary
EVENT_WORDS = tuple((event, re.compile(r"\b(?:" + words + ")")) for event, words in (
    ("coastal", r"coastal|lakeshore|surf\b|rip currents?\b|beach|seas\b|tsunami|small craft|freezing spray|marine|storm surge|"
                r"high water|low water|côti|vagues|onde de tempête"),
    ("thunderstorm", r"tornad|thunder|orage|waterspout|trombe"),
    ("flood", r"flood|inond|crue"),
    ("snow-ice", r"winter|snow|blizzard|ice storm|freezing rain|freezing drizzle|sleet|lake effect|ice\b|icy\b|neige|verglas|"
                 r"pluie verglaçante|bruine verglaçante|poudrerie|hiver"),
    ("cold", r"freeze\b|frost|cold\b|extreme cold|wind chill|arctic outflow|gel\b|gelée|froid|refroidissement"),
    ("heat", r"heat\b|chaleur"),
    ("fog", r"fog\b|smoke|brouillard|fumée|visibility|visibilité"),
    ("fire", r"fire\b|red flag|feux?\b"),
    ("avalanche", r"avalanche"),
    ("wind", r"hurricane|tropical|cyclone|typhoon|winds?\b|gale|storm\b|squall|vents?\b|ouragan|tempête|bourrasque"),
    ("rain", r"rain|pluie|precipitation"),
))


def now_ms():
    return int(time.time() * 1000)


def iso_ms(s):
    """ISO time ('2026-11-16T03:06:00+01:00', '...Z', '...986Z') -> ms since epoch, None if missing."""
    if not s or s == "None":
        return None
    try:
        d = datetime.fromisoformat(str(s).strip().replace("Z", "+00:00"))
    except ValueError:
        return None
    if d.tzinfo is None:
        d = d.replace(tzinfo=timezone.utc)
    return int(d.timestamp() * 1000)


def short(s, n=DESC_CHARS):
    s = " ".join(str(s or "").split())
    if len(s) <= n:
        return s
    cut = s[:n]
    i = cut.rfind(". ")
    return cut[:i + 1] if i > n // 2 else cut.rstrip() + "…"


def classify(name):
    t = (name or "").lower()
    for event, pattern in EVENT_WORDS:
        if pattern.search(t):
            return event
    return "other"


def refs_of(s):
    """CAP <references> 'sender,identifier,sent sender,identifier,sent' -> identifiers."""
    return {p.split(",")[1] for p in str(s or "").split() if p.count(",") >= 2}


# ---------- geometry ----------
def simplify(ring, tol=SIMPLIFY_DEG):
    """Douglas-Peucker for one ring (Nx2 lon/lat, first == last); returns float32 Nx2 or None when it collapses."""
    p = np.asarray(ring, np.float64)
    if p.ndim != 2 or len(p) < 4:
        return None
    if not np.array_equal(p[0], p[-1]):
        p = np.vstack([p, p[:1]])
    n = len(p)
    keep = np.zeros(n, bool)
    keep[0] = keep[-1] = True
    # a closed ring has start == end: split it at the point farthest from the start first
    far = int(np.argmax(np.hypot(*(p - p[0]).T)))
    keep[far] = True
    stack = [(0, far), (far, n - 1)]
    while stack:
        i, j = stack.pop()
        if j <= i + 1:
            continue
        a, b, seg = p[i], p[j], p[i + 1:j]
        d = b - a
        ln = math.hypot(d[0], d[1])
        dist = np.hypot(*(seg - a).T) if ln == 0 else np.abs(d[0] * (seg[:, 1] - a[1]) - d[1] * (seg[:, 0] - a[0])) / ln
        k = int(np.argmax(dist))
        if dist[k] > tol:
            m = i + 1 + k
            keep[m] = True
            stack += [(i, m), (m, j)]
    q = p[keep]
    if len(q) < 4:  # tiny ring (islet, single municipality): keep its bounding box shape rather than nothing
        x0, y0 = p.min(0)
        x1, y1 = p.max(0)
        if x1 - x0 <= 0 or y1 - y0 <= 0:
            return None
        q = np.array([[x0, y0], [x1, y0], [x1, y1], [x0, y1], [x0, y0]])
    return q.astype(np.float32)


def in_ring(lon, lat, r):
    """Ray casting point-in-polygon for one ring (float32 Nx2 lon/lat)."""
    xs, ys = r[:, 0].astype(np.float64), r[:, 1].astype(np.float64)
    x2, y2 = np.roll(xs, -1), np.roll(ys, -1)
    cross = (ys > lat) != (y2 > lat)
    with np.errstate(divide="ignore", invalid="ignore"):
        xint = xs + (lat - ys) * (x2 - xs) / np.where(y2 == ys, 1e-12, y2 - ys)
    return bool(np.count_nonzero(cross & (lon < xint)) % 2)


class Shape:
    """Polygons [(bbox, outer ring, [hole rings])] with their overall bbox (west, south, east, north)."""
    __slots__ = ("polys", "bbox")

    def __init__(self, polys):
        self.polys = polys
        self.bbox = (min(p[0][0] for p in polys), min(p[0][1] for p in polys),
                     max(p[0][2] for p in polys), max(p[0][3] for p in polys))

    def contains(self, lon, lat):
        x0, y0, x1, y1 = self.bbox
        if not (x0 <= lon <= x1 and y0 <= lat <= y1):
            return False
        for (a, b, c, d), outer, holes in self.polys:
            if a <= lon <= c and b <= lat <= d and in_ring(lon, lat, outer) and not any(in_ring(lon, lat, h) for h in holes):
                return True
        return False

    def geojson(self):
        def ring(r):
            return np.round(r.astype(np.float64), 4).tolist()
        return {"type": "MultiPolygon", "coordinates": [[ring(o)] + [ring(h) for h in hs] for _, o, hs in self.polys]}

    def compact(self):
        """For the disk cache: [[outer, hole, ...], ...] as rounded lists."""
        r = lambda a: np.round(a.astype(np.float64), 4).tolist()  # float64: float32 values would print 16 digits
        return [[r(o)] + [r(h) for h in hs] for _, o, hs in self.polys]


def make_shape(polys, presimplified=False):
    """polys: [[outer ring, hole ring, ...], ...] as [[lon, lat], ...] -> Shape (None if nothing is left)."""
    out = []
    for rings in polys:
        if not rings:
            continue
        rs = [np.asarray(r, np.float32) if presimplified else simplify(r) for r in rings]
        if rs[0] is None or len(rs[0]) < 4:
            continue
        o = rs[0]
        bb = (float(o[:, 0].min()), float(o[:, 1].min()), float(o[:, 0].max()), float(o[:, 1].max()))
        out.append((bb, o, [h for h in rs[1:] if h is not None and len(h) >= 4]))
    return Shape(out) if out else None


def geojson_polys(g):
    """GeoJSON geometry (Polygon, MultiPolygon, GeometryCollection) -> [[outer, holes...], ...]."""
    if not g:
        return []
    t = g.get("type")
    if t == "Polygon":
        return [g.get("coordinates") or []]
    if t == "MultiPolygon":
        return list(g.get("coordinates") or [])
    if t == "GeometryCollection":
        return [p for gg in g.get("geometries") or [] for p in geojson_polys(gg)]
    return []


def cap_ring(s):
    """CAP polygon 'lat,lon lat,lon ...' -> [[lon, lat], ...]."""
    pts = []
    for pair in str(s).split():
        a, _, b = pair.partition(",")
        try:
            pts.append([float(b), float(a)])
        except ValueError:
            pass
    return pts


def cap_shape(polygons, excludes=()):
    """CAP polygons (+ DWD EXCLUDE_POLYGONs as holes of the polygon containing them) -> Shape."""
    outers = [r for r in (simplify(cap_ring(p)) for p in polygons) if r is not None]
    holes = [[] for _ in outers]
    for e in excludes:
        h = simplify(cap_ring(e))
        if h is None:
            continue
        for i, o in enumerate(outers):
            if in_ring(float(h[0, 0]), float(h[0, 1]), o):
                holes[i].append(h)
                break
    return make_shape([[o] + hs for o, hs in zip(outers, holes)], presimplified=True)


class State:
    """Result of one source refresh (immutable once built): alerts, area shapes, and a grid index over them."""

    def __init__(self, source, alerts, links, shapes, stats=None):
        self.source = source
        self.alerts = alerts          # id -> alert
        self.links = links            # shape key -> [(alert id, area name)]
        self.shapes = {k: shapes[k] for k in links if k in shapes}
        self.stats = stats or {}
        self.grid = {}
        for k, s in self.shapes.items():
            x0, y0, x1, y1 = s.bbox
            for ix in range(math.floor(x0 / CELL_DEG), math.floor(x1 / CELL_DEG) + 1):
                for iy in range(math.floor(y0 / CELL_DEG), math.floor(y1 / CELL_DEG) + 1):
                    self.grid.setdefault((ix, iy), []).append(k)

    def hits(self, lat, lon):
        for k in self.grid.get((math.floor(lon / CELL_DEG), math.floor(lat / CELL_DEG)), ()):
            if self.shapes[k].contains(lon, lat):
                yield k


def build_state(source, alerts, shape_of, stats_extra=None):
    """alerts: parsed alerts with 'areas' [(shape key or None, area name, geocode text)] -> State + coverage stats."""
    links, shapes, unresolved = {}, {}, {}
    n_areas = n_res = 0
    for a in alerts.values():
        for key, name, code in a["areas"]:
            n_areas += 1
            s = shape_of(a, key) if key else None
            if s is None:
                unresolved[code or name] = unresolved.get(code or name, 0) + 1
                continue
            n_res += 1
            shapes[key] = s
            links.setdefault(key, []).append((a["id"], name))
    with_geo = sum(1 for a in alerts.values() if any(k in shapes for k, _, _ in a["areas"]))
    stats = {"alerts": len(alerts), "with_geometry": with_geo, "areas": n_areas, "resolved": n_res,
             "unresolved": n_areas - n_res, "unresolved_examples": sorted(unresolved, key=unresolved.get, reverse=True)[:5]}
    stats.update(stats_extra or {})
    return State(source, alerts, links, shapes, stats)


def pick_info(infos, lang):
    """infos: [(language, event name, headline, full text)] -> the requested language, else English, else first."""
    if not infos:
        return ("", "", "", "")
    lang = (lang or "").lower().split("-")[0]
    for want in (lang, "en"):
        for i in infos:
            if i[0].lower().split("-")[0] == want:
                return i
    return infos[0]


class Warnings:
    def __init__(self, fetch, log, cache_dir, regions=("eu",), sources=None, meteoalarm_skip=None):
        self.fetch, self.log, self.cache_dir = fetch, log, cache_dir
        os.makedirs(cache_dir, exist_ok=True)
        self.sources = list(sources or dict.fromkeys(s for r in regions for s in REGIONS[r]))
        # Germany comes from DWD directly when that source runs
        self.ma_skip = set(meteoalarm_skip if meteoalarm_skip is not None else (("germany",) if "dwd" in self.sources else ()))
        self.states = {}               # name ("dwd", "nws", "meteoalarm:spain", ...) -> State; swapped atomically
        self.lock = threading.Lock()   # guards the geometry stores below while they are (re)loaded
        self.ma_geo = {}               # EMMA_ID -> Shape
        self.ma_alias = {}             # "TYPE:VALUE" and "VALUE" -> EMMA_ID
        self.ma_geo_checked = 0.0
        self.ma_countries = list(MA_COUNTRIES)
        self.ma_countries_checked = 0.0
        self.ma_parsed = {}            # country -> {alert identifier: parsed alert} (parse only new ones)
        self.dwd_cells = {}            # WARNCELLID (coast, lakes) -> (Shape, name)
        self.dwd_cells_checked = 0.0
        self.nws_zones = None          # zone id -> {"n": name, "p": compact polys, "t": fetched}
        self.nws_shapes = {}           # zone id -> Shape
        self.nws_failed = {}           # zone id -> time of the last failed fetch

    # ---------- public ----------
    def loop(self):
        """One thread per source, each refreshing every REFRESH_S; a slow or failing source never blocks the others."""
        threads = [threading.Thread(target=self._source_loop, args=(s,), daemon=True, name=f"warnings-{s}") for s in self.sources]
        for t in threads:
            t.start()
        for t in threads:
            t.join()

    def refresh(self, source):
        getattr(self, f"refresh_{source}")()

    def at(self, lat, lon, lang="de", now=None):
        """Active warnings for a place (expired ones and those starting more than 24 h ahead left out), worst first."""
        now = now or now_ms()
        out, seen = [], set()
        for st in list(self.states.values()):
            for key in st.hits(lat, lon):
                for aid, area in st.links[key]:
                    a = st.alerts.get(aid)
                    if a is None or aid in seen or not self._active(a, now):
                        continue
                    seen.add(aid)
                    out.append(self._render(a, lang, area))
        return sorted(out, key=lambda w: (-w["level"], w["onset"] or 0))

    def areas_geojson(self, bbox=None, now=None):
        """Warned areas for the map: one feature per area shape with the highest level of its active warnings."""
        now = now or now_ms()
        feats = []
        for st in list(self.states.values()):
            for key, links in st.links.items():
                s = st.shapes.get(key)
                if s is None:
                    continue
                if bbox and (s.bbox[0] > bbox[2] or s.bbox[2] < bbox[0] or s.bbox[1] > bbox[3] or s.bbox[3] < bbox[1]):
                    continue
                act = [st.alerts[aid] for aid, _ in links if aid in st.alerts and self._active(st.alerts[aid], now)]
                if not act:
                    continue
                top = max(act, key=lambda a: (a["level"], -(a["onset"] or 0)))
                exp = [a["expires"] for a in act]
                feats.append({"type": "Feature", "geometry": s.geojson(), "properties": {
                    "source": st.source, "level": top["level"], "event": top["event"],
                    "events": sorted({a["event"] for a in act}), "count": len(act),
                    "onset": min(a["onset"] or 0 for a in act), "expires": None if None in exp else max(exp),
                    "now": any((a["onset"] or 0) <= now for a in act)}})
        return {"type": "FeatureCollection", "features": feats}

    def stats(self):
        return {name: st.stats for name, st in sorted(self.states.items())}

    # ---------- internals ----------
    def _source_loop(self, source):
        while True:
            t0 = time.time()
            try:
                self.refresh(source)
            except Exception as e:
                self.log(f"warnings: {source} failed: {e}")
            time.sleep(max(30.0, REFRESH_S - (time.time() - t0)))

    @staticmethod
    def _active(a, now):
        return (a["expires"] is None or a["expires"] > now) and (a["onset"] or 0) <= now + LOOKAHEAD_MS

    @staticmethod
    def _render(a, lang, area):
        language, name, headline, desc = pick_info(a["infos"], lang)
        if a.get("fr_area") and language.startswith("fr"):  # ECCC names its regions in both languages
            area = a["fr_area"]
        return {"source": a["source"], "id": a["id"], "country": a.get("country"), "event": a["event"],
                "level": a["level"], "title": name, "headline": headline or name, "description": short(desc), "text": desc,
                "onset": a["onset"], "expires": a["expires"], "sent": a.get("sent"), "sender": a["sender"],
                "language": language, "area": area, "web": a.get("web")}

    def _json(self, url, timeout=60):
        return json.loads(self.fetch(url, timeout=timeout))

    def _cache_path(self, name):
        return os.path.join(self.cache_dir, name)

    def _write_cache(self, name, obj):
        tmp = self._cache_path(name + ".tmp")
        with open(tmp, "w", encoding="utf-8") as f:
            json.dump(obj, f, separators=(",", ":"))
        os.replace(tmp, self._cache_path(name))

    def _read_cache(self, name):
        try:
            with open(self._cache_path(name), encoding="utf-8") as f:
                return json.load(f)
        except (OSError, ValueError):
            return None

    @staticmethod
    def _alert(source, aid, **kw):
        a = {"source": source, "id": aid, "country": None, "sent": None, "onset": None, "expires": None, "level": 2,
             "event": "other", "sender": "", "web": None, "infos": [], "areas": [], "own": {}}
        a.update(kw)
        return a

    # ---------- DWD (Germany) ----------
    def _dwd_cells(self):
        """Coast and lake warning areas (WARNCELLID -> shape) from the DWD WFS, cached on disk, checked weekly."""
        if self.dwd_cells and time.time() - self.dwd_cells_checked < GEOMETRY_MAX_AGE_S:
            return
        cached = self._read_cache("dwd_cells.json")
        if cached and time.time() - cached.get("t", 0) < GEOMETRY_MAX_AGE_S:
            cells = cached["cells"]
        else:
            cells = {}
            try:
                for layer in DWD_CELL_LAYERS:
                    q = (f"{DWD_WFS}?service=WFS&version=2.0.0&request=GetFeature&typeName={layer}"
                         f"&outputFormat=application/json&srsName=EPSG:4326")
                    for f in self._json(q).get("features", []):
                        p = f.get("properties") or {}
                        s = make_shape(geojson_polys(f.get("geometry")))
                        if s and p.get("WARNCELLID"):
                            cells[str(p["WARNCELLID"])] = {"n": p.get("NAME"), "p": s.compact()}
                self._write_cache("dwd_cells.json", {"t": time.time(), "cells": cells})
            except Exception as e:
                self.log(f"warnings: DWD coast/lake areas failed: {e}")
                cells = (cached or {}).get("cells", {})
        self.dwd_cells = {k: (make_shape(v["p"], presimplified=True), v["n"]) for k, v in cells.items()}
        self.dwd_cells_checked = time.time()

    def refresh_dwd(self):
        self._dwd_cells()
        z = zipfile.ZipFile(io.BytesIO(self.fetch(DWD_CAP, timeout=60)))
        now = now_ms()
        alerts, refs = {}, set()
        for name in z.namelist():
            try:
                a = self._parse_dwd(z.read(name), now)
            except ET.ParseError as e:
                self.log(f"warnings: DWD {name}: {e}")
                continue
            if a:
                refs |= a.pop("refs")
                alerts[a["id"]] = a
        alerts = {k: a for k, a in alerts.items() if k not in refs}

        def shape_of(a, key):
            if key in a["own"]:
                return a["own"][key]
            c = self.dwd_cells.get(key.split(":", 1)[1]) if key.startswith("cell:") else None
            return c[0] if c else None
        self.states["dwd"] = build_state("dwd", alerts, shape_of)

    def _parse_dwd(self, xml, now):
        ns = {"c": "urn:oasis:names:tc:emergency:cap:1.2"}
        root = ET.fromstring(xml)
        txt = lambda el, tag: (el.findtext(f"c:{tag}", "", ns) or "").strip()
        if txt(root, "status") != "Actual" or txt(root, "msgType") == "Cancel":
            return None
        infos = root.findall("c:info", ns)
        if not infos:
            return None
        i0 = infos[0]
        codes = {e.findtext("c:valueName", "", ns): e.findtext("c:value", "", ns) for e in i0.findall("c:eventCode", ns)}
        event = txt(i0, "event")
        expires = iso_ms(txt(i0, "expires"))
        if expires is not None and expires <= now:
            return None
        level = 1 if event.upper().startswith("VORABINFORMATION") else SEVERITY_DWD.get(txt(i0, "severity"), 2)
        aid = txt(root, "identifier")
        a = self._alert("dwd", aid, country="DE", sent=iso_ms(txt(root, "sent")),
                        onset=iso_ms(txt(i0, "onset")) or iso_ms(txt(i0, "effective")) or iso_ms(txt(root, "sent")),
                        expires=expires, level=level, event=DWD_GROUPS.get(codes.get("GROUP", "").upper(), "other"),
                        sender=txt(i0, "senderName") or "Deutscher Wetterdienst", web=txt(i0, "web") or None,
                        infos=[(txt(i, "language"), txt(i, "event"), txt(i, "headline"), txt(i, "description")) for i in infos])
        a["refs"] = refs_of(txt(root, "references"))
        cells = []
        for n, ar in enumerate(i0.findall("c:area", ns)):
            polys = [p.text for p in ar.findall("c:polygon", ns) if p.text]
            gcs = [(g.findtext("c:valueName", "", ns), g.findtext("c:value", "", ns)) for g in ar.findall("c:geocode", ns)]
            desc = txt(ar, "areaDesc")
            if polys:
                s = cap_shape(polys, [v for k, v in gcs if k == "EXCLUDE_POLYGON"])
                key = f"{aid}#{n}"
                if s:
                    a["own"][key] = s
                a["areas"].append((key, None, f"polygon {aid[-12:]}"))
            else:
                cells += [(v, desc) for k, v in gcs if k == "WARNCELLID"]
        # The polygonal event area is the union of the warned municipalities (WARNCELLID 8..., city parts 7...), which
        # are listed as areas without geometry. Coast and lake areas (5..., 2...) are not in the polygon.
        has_poly = bool(a["areas"])
        municipalities = []
        for c, desc in cells:
            if has_poly and c not in self.dwd_cells:
                municipalities.append(desc)
            else:
                a["areas"].append((f"cell:{c}", desc, f"WARNCELLID={c}"))
        if len(municipalities) == 1:  # name the polygon after its only municipality
            a["areas"] = [(k, municipalities[0] if nm is None else nm, c) for k, nm, c in a["areas"]]
        return a

    # ---------- MeteoAlarm (Europe) ----------
    def _ma_geocodes(self):
        """EMMA_ID geometry + aliases from MeteoAlarm's public geocode files, simplified and cached; checked weekly."""
        if self.ma_geo and time.time() - self.ma_geo_checked < GEOMETRY_MAX_AGE_S:
            return
        cached = self._read_cache("meteoalarm_geocodes.json")
        fresh = cached and time.time() - cached.get("t", 0) < GEOMETRY_MAX_AGE_S
        if not fresh:
            try:
                names = [x["name"] for x in self._json(MA_DOCS_TREE) if x.get("type") == "blob"]
                gj = sorted(n for n in names if re.fullmatch(r"MeteoAlarm_Geocodes_[\d_]+\.json", n))[-1]
                al = sorted(n for n in names if re.fullmatch(r"MeteoAlarm_Geocodes-Aliases_[\d_]+\.csv", n))[-1]
                if cached and cached.get("src") == [gj, al]:
                    cached["t"] = time.time()
                    self._write_cache("meteoalarm_geocodes.json", cached)
                else:
                    self.log(f"warnings: loading MeteoAlarm geocodes {gj}")
                    cached = self._ma_build_geocodes(gj, al)
                    self._write_cache("meteoalarm_geocodes.json", cached)
            except Exception as e:
                self.log(f"warnings: MeteoAlarm geocodes failed: {e}")
        if not cached:
            raise RuntimeError("no MeteoAlarm geocode geometry yet")
        if not self.ma_geo or cached.get("t", 0) > self.ma_geo_checked:
            geo = {k: make_shape(v["p"], presimplified=True) for k, v in cached["codes"].items()}
            with self.lock:
                self.ma_geo = {k: s for k, s in geo.items() if s}
                self.ma_alias = cached["alias"]
        self.ma_geo_checked = time.time()

    def _ma_build_geocodes(self, gj_name, alias_name):
        raw = self.fetch(MA_DOCS_RAW.format(gj_name), timeout=300)
        feats = json.loads(raw).get("features", [])
        del raw
        codes = {}
        for f in feats:
            p = f.get("properties") or {}
            s = make_shape(geojson_polys(f.get("geometry")))
            if s and p.get("code"):
                codes[p["code"]] = {"n": p.get("name"), "c": p.get("country"), "p": s.compact()}
        del feats
        alias = {}
        text = self.fetch(MA_DOCS_RAW.format(alias_name), timeout=120).decode("utf-8-sig")
        for r in csv.DictReader(io.StringIO(text)):
            code, a, typ = (r.get("CODE") or "").strip(), (r.get("ALIAS_CODE") or "").strip(), (r.get("ALIAS_TYPE") or "").strip()
            if code in codes and a:
                alias[f"{typ}:{a}"] = code
                alias.setdefault(a, code)
        return {"t": time.time(), "src": [gj_name, alias_name], "codes": codes, "alias": alias}

    def _ma_countries(self):
        if time.time() - self.ma_countries_checked < 86400:
            return
        self.ma_countries_checked = time.time()
        try:
            found = sorted(set(re.findall(r"meteoalarm-legacy-atom-([a-z-]+)", self.fetch(MA_INDEX, timeout=60).decode("utf-8", "replace"))))
            if len(found) >= 20:
                self.ma_countries = found
        except Exception as e:
            self.log(f"warnings: MeteoAlarm country list failed: {e}")

    def refresh_meteoalarm(self):
        self._ma_geocodes()
        self._ma_countries()
        for c in self.ma_countries:
            if c in self.ma_skip:
                continue
            try:
                self.refresh_meteoalarm_country(c)
            except Exception as e:
                self.log(f"warnings: MeteoAlarm {c} failed: {e}")

    def refresh_meteoalarm_country(self, country):
        data = self._json(MA_FEED.format(country), timeout=90)
        now = now_ms()
        old, parsed, refs = self.ma_parsed.get(country, {}), {}, set()
        for w in data.get("warnings") or []:
            al = w.get("alert") or {}
            aid = al.get("identifier")
            if not aid or al.get("status") != "Actual" or al.get("msgType") == "Cancel":
                continue
            refs |= refs_of(al.get("references"))
            a = old.get(aid) or self._parse_ma(al, country)
            if a:
                parsed[aid] = a
        self.ma_parsed[country] = parsed
        alerts = {k: a for k, a in parsed.items() if k not in refs and (a["expires"] is None or a["expires"] > now)}

        def shape_of(a, key):
            return a["own"].get(key) or self.ma_geo.get(key[5:]) if key.startswith("emma:") else a["own"].get(key)
        self.states[f"meteoalarm:{country}"] = build_state("meteoalarm", alerts, shape_of)

    def _parse_ma(self, al, country):
        infos = al.get("info") or []
        if isinstance(infos, dict):
            infos = [infos]
        if not infos:
            return None
        i0 = next((i for i in infos if str(i.get("language", "")).lower().startswith("en")), infos[0])
        params = {p.get("valueName"): p.get("value") for p in i0.get("parameter") or [] if isinstance(p, dict)}
        try:
            level = int(str(params.get("awareness_level", "")).split(";")[0])
        except ValueError:
            level = SEVERITY_MA.get(i0.get("severity"), 2)
        if level <= 1:  # green: no warning
            return None
        try:
            kind = MA_TYPES.get(int(str(params.get("awareness_type", "")).split(";")[0]), "other")
        except ValueError:
            kind = classify(i0.get("event"))
        aid = al["identifier"]
        a = self._alert("meteoalarm", aid, country=country, sent=iso_ms(al.get("sent")),
                        onset=iso_ms(i0.get("onset")) or iso_ms(i0.get("effective")) or iso_ms(al.get("sent")),
                        expires=iso_ms(i0.get("expires")), level=min(level, 4), event=kind,
                        sender=i0.get("senderName") or al.get("sender") or "", web=i0.get("web"),
                        infos=[(str(i.get("language", "")), i.get("event") or "", i.get("headline") or "", str(i.get("description") or "").strip())
                               for i in infos])
        for n, ar in enumerate(i0.get("area") or []):
            desc = ar.get("areaDesc")
            polys = ar.get("polygon")
            if polys:
                s = cap_shape([polys] if isinstance(polys, str) else polys)
                key = f"{aid}#{n}"
                if s:
                    a["own"][key] = s
                a["areas"].append((key, desc, f"polygon {country}"))
                continue
            gcs = [(g.get("valueName"), str(g.get("value"))) for g in ar.get("geocode") or []]
            emma = next((v for k, v in gcs if k == "EMMA_ID" and v in self.ma_geo), None) \
                or next((self.ma_alias[f"{k}:{v}"] for k, v in gcs if f"{k}:{v}" in self.ma_alias), None) \
                or next((self.ma_alias[v] for k, v in gcs if v in self.ma_alias), None)
            a["areas"].append((f"emma:{emma}" if emma else None, desc,
                               ";".join(f"{k}={v}" for k, v in gcs) or f"no geocode ({country})"))
        return a

    # ---------- NWS (USA) ----------
    def _nws_load_zones(self):
        """Zone geometry cache: shapes in memory as numpy rings only (no nested lists), names / fetch times aside."""
        if self.nws_zones is None:
            zones = (self._read_cache("nws_zones.json") or {}).get("zones", {})
            self.nws_shapes = {k: s for k, s in ((k, make_shape(v["p"], presimplified=True) if v.get("p") else None)
                                                 for k, v in zones.items()) if s}
            self.nws_zones = {k: {"n": v.get("n"), "t": v.get("t", 0)} for k, v in zones.items()}

    def _nws_save_zones(self):
        self._write_cache("nws_zones.json", {"zones": {
            k: {**v, "p": self.nws_shapes[k].compact() if k in self.nws_shapes else None} for k, v in self.nws_zones.items()}})

    def refresh_nws(self):
        self._nws_load_zones()
        d = self._json(NWS_ALERTS, timeout=90)
        now = now_ms()
        alerts, refs, wanted = {}, set(), {}
        for f in d.get("features") or []:
            p = f.get("properties") or {}
            if p.get("status") != "Actual" or p.get("messageType") == "Cancel":
                continue
            for r in p.get("references") or []:
                if isinstance(r, dict) and r.get("identifier"):
                    refs.add(r["identifier"])
            onset = iso_ms(p.get("onset")) or iso_ms(p.get("effective")) or iso_ms(p.get("sent"))
            expires = iso_ms(p.get("ends")) or iso_ms(p.get("expires"))
            if expires is not None and expires <= now:
                continue
            name = p.get("event") or ""
            aid = p.get("id") or f.get("id")
            sev = p.get("severity")
            low = name.lower()
            if sev == "Extreme":
                level = 4
            elif "warning" in low or "emergency" in low:
                level = 3
            elif "advisory" in low or "watch" in low:
                level = 2
            else:
                level = 1
            kind = classify(name)
            if kind == "other":  # e.g. "Special Weather Statement": what it is about is in its headline / text
                kind = classify(" ".join((p.get("parameters") or {}).get("NWSheadline") or []) + " " + str(p.get("description") or "")[:200])
            a = self._alert("nws", aid, country="US", sent=iso_ms(p.get("sent")), onset=onset, expires=expires,
                            level=level, event=kind, sender=p.get("senderName") or "NWS", web=p.get("web"),
                            infos=[(p.get("language") or "en-US", name, p.get("headline") or name, str(p.get("description") or "").strip())])
            s = make_shape(geojson_polys(f.get("geometry"))) if f.get("geometry") else None
            if s:
                a["own"][f"{aid}#0"] = s
                a["areas"].append((f"{aid}#0", short(p.get("areaDesc"), 120), "polygon"))
            else:
                zones = [u for u in p.get("affectedZones") or [] if str(u).startswith(NWS_ZONE_PREFIX)]
                if not zones:  # fall back to the UGC codes (SSZnnn forecast zone, SSCnnn county)
                    for u in (p.get("geocode") or {}).get("UGC") or []:
                        zones.append(f"{NWS_ZONE_PREFIX}{'county' if u[2:3] == 'C' else 'forecast'}/{u}")
                for u in zones:
                    zid = u.rstrip("/").rsplit("/", 1)[1]
                    wanted[zid] = u
                    a["areas"].append((f"zone:{zid}", None, zid))
            alerts[aid] = a
        alerts = {k: a for k, a in alerts.items() if k not in refs}
        self._nws_publish(alerts)
        # missing zone geometry: fetch it now (politely, capped), publishing progress every 50 zones
        missing = [(z, u) for z, u in wanted.items()
                   if (z not in self.nws_zones or time.time() - self.nws_zones[z].get("t", 0) > ZONE_MAX_AGE_S)
                   and time.time() - self.nws_failed.get(z, 0) > 1800]  # failed: retry after 30 min
        for n, (zid, url) in enumerate(missing[:ZONE_FETCH_MAX]):
            try:
                zf = self._json(url, timeout=60)
                s = make_shape(geojson_polys(zf.get("geometry")))
                self.nws_zones[zid] = {"n": (zf.get("properties") or {}).get("name"), "t": time.time()}
                if s:
                    self.nws_shapes[zid] = s
            except Exception as e:
                self.nws_failed[zid] = time.time()
                self.log(f"warnings: NWS zone {zid} failed: {e}")
            time.sleep(ZONE_PAUSE_S)
            if n % 50 == 49:
                self._nws_publish(alerts)
                self._nws_save_zones()
        if missing:
            self._nws_save_zones()
            self._nws_publish(alerts)

    def _nws_publish(self, alerts):
        # zone names become the area names once the zone is known
        for a in alerts.values():
            a["areas"] = [(k, (self.nws_zones.get(k[5:]) or {}).get("n") if k.startswith("zone:") and nm is None else nm, c)
                          for k, nm, c in a["areas"]]

        def shape_of(a, key):
            return self.nws_shapes.get(key[5:]) if key.startswith("zone:") else a["own"].get(key)
        self.states["nws"] = build_state("nws", alerts, shape_of, {"zones_cached": len(self.nws_shapes)})

    # ---------- ECCC (Canada) ----------
    def refresh_eccc(self):
        now = now_ms()
        alerts, offset = {}, 0
        for _ in range(40):
            d = self._json(ECCC_ALERTS.format(ECCC_PAGE, offset), timeout=90)
            feats = d.get("features") or []
            for f in feats:
                p = f.get("properties") or {}
                if str(p.get("status_en", "")).lower() in ("ended", "cancelled", "canceled"):
                    continue
                expires = iso_ms(p.get("event_end_datetime")) or iso_ms(p.get("expiration_datetime"))
                if expires is not None and expires <= now:
                    continue
                typ = str(p.get("alert_type") or "").lower()
                level = COLOUR.get(str(p.get("risk_colour_en") or "").lower()) or \
                    {"statement": 1, "advisory": 2, "watch": 2, "warning": 3}.get(typ, 2)
                aid = str(f.get("id") or p.get("id"))
                name_en, name_fr = p.get("alert_name_en") or "", p.get("alert_name_fr") or ""
                a = self._alert("eccc", aid, country="CA", sent=iso_ms(p.get("publication_datetime")),
                                onset=iso_ms(p.get("validity_datetime")) or iso_ms(p.get("publication_datetime")),
                                expires=expires, level=level, event=classify(name_en) if classify(name_en) != "other"
                                else classify(str(p.get("alert_text_en") or "")[:200]),  # e.g. special weather statement
                                sender="Environment and Climate Change Canada", web="https://weather.gc.ca/warnings/index_e.html",
                                infos=[("en", name_en, name_en[:1].upper() + name_en[1:], str(p.get("alert_text_en") or "").strip()),
                                       ("fr", name_fr, name_fr[:1].upper() + name_fr[1:], str(p.get("alert_text_fr") or "").strip())])
                a["fr_area"] = p.get("feature_name_fr")
                s = make_shape(geojson_polys(f.get("geometry")))
                if s:
                    a["own"][f"{aid}#0"] = s
                a["areas"].append((f"{aid}#0", p.get("feature_name_en"), f"region {p.get('feature_id')}"))
                alerts[aid] = a
            if len(feats) < ECCC_PAGE:
                break
            offset += ECCC_PAGE

        def shape_of(a, key):
            return a["own"].get(key)
        self.states["eccc"] = build_state("eccc", alerts, shape_of)


# ---------- command line test ----------
def _sample_points(w, per_state=2):
    """A few points inside currently warned areas (to try --at with)."""
    out = []
    for name, st in sorted(w.states.items()):
        n = 0
        for key, links in st.links.items():
            s = st.shapes[key]
            _, o, _ = max(s.polys, key=lambda p: (p[0][2] - p[0][0]) * (p[0][3] - p[0][1]))
            c = o[:-1].mean(0)
            for lon, lat in [c] + [(o[i] + c) / 2 for i in range(0, len(o) - 1, max(1, len(o) // 8))]:
                if s.contains(float(lon), float(lat)):
                    a = st.alerts[links[0][0]]
                    out.append((name, round(float(lat), 3), round(float(lon), 3), a["event"], a["level"], links[0][1]))
                    n += 1
                    break
            if n >= per_state:
                break
    return out


if __name__ == "__main__":
    import argparse
    import tempfile
    import urllib.request

    ap = argparse.ArgumentParser(description="Fetch official warnings once and query points.")
    ap.add_argument("--once", required=True, choices=sorted(REGIONS))
    ap.add_argument("--at", action="append", default=[], help="LAT,LON (repeatable)")
    ap.add_argument("--lang", default="de")
    ap.add_argument("--cache", default=None, help="cache dir (default: a new temporary directory)")
    ap.add_argument("--samples", action="store_true", help="print points inside warned areas")
    ap.add_argument("--countries", default=None, help="MeteoAlarm: only these feed slugs (comma separated)")
    args = ap.parse_args()

    def fetch(url, timeout=40, tries=3):  # like rainy_server.fetch: neutral User-Agent, nothing identifying
        req = urllib.request.Request(url, headers={"User-Agent": "Mozilla/5.0"})
        for attempt in range(tries):
            try:
                with urllib.request.urlopen(req, timeout=timeout) as r:
                    return r.read()
            except OSError:
                if attempt == tries - 1:
                    raise
                time.sleep(2 * (attempt + 1))

    def log(msg):
        print(msg, flush=True)

    cache = args.cache or tempfile.mkdtemp(prefix="rainy-warnings-")
    w = Warnings(fetch, log, cache, regions=(args.once,))
    if args.countries:
        w.ma_countries_checked = time.time()
        w.ma_countries = args.countries.split(",")
    for src in w.sources:
        t0 = time.time()
        try:
            w.refresh(src)
        except Exception as e:
            log(f"{src}: FAILED {e!r}")
        log(f"{src}: refreshed in {time.time() - t0:.1f} s")
    tot = {"alerts": 0, "with_geometry": 0, "areas": 0, "resolved": 0, "unresolved": 0}
    print(f"\n{'state':32s} {'alerts':>6s} {'w/geo':>6s} {'areas':>6s} {'resolved':>8s} {'unres.':>6s}  unresolved examples")
    for name, s in w.stats().items():
        for k in tot:
            tot[k] += s.get(k, 0)
        print(f"{name:32s} {s['alerts']:6d} {s['with_geometry']:6d} {s['areas']:6d} {s['resolved']:8d} {s['unresolved']:6d}  "
              f"{', '.join(map(str, s['unresolved_examples']))[:90]}")
    print(f"{'TOTAL':32s} {tot['alerts']:6d} {tot['with_geometry']:6d} {tot['areas']:6d} {tot['resolved']:8d} {tot['unresolved']:6d}")
    if args.samples:
        print("\nsample points inside warned areas:")
        for row in _sample_points(w):
            print("  ", row)
    for spec in args.at:
        lat, lon = map(float, spec.split(","))
        t0 = time.perf_counter()
        res = w.at(lat, lon, lang=args.lang)
        dt = (time.perf_counter() - t0) * 1000
        print(f"\nat {lat},{lon} ({dt:.1f} ms): {len(res)} warning(s)")
        for r in res:
            print(f"  [{r['source']}] L{r['level']} {r['event']:12s} {r['headline'][:70]!r} area={r['area']!r} lang={r['language']}")
            print(f"        {datetime.fromtimestamp((r['onset'] or 0) / 1000, timezone.utc):%d.%m %H:%M}Z -> "
                  f"{datetime.fromtimestamp(r['expires'] / 1000, timezone.utc):%d.%m %H:%M}Z" if r['expires'] else "open end",
                  f"| {r['sender']} | {r['description'][:110]!r}")
    if args.at:
        pts = [tuple(map(float, s.split(","))) for s in args.at]
        t0 = time.perf_counter()
        for i in range(200):
            w.at(*pts[i % len(pts)])
        print(f"\n200 lookups: {(time.perf_counter() - t0) * 1000 / 200:.2f} ms each")
    gj = w.areas_geojson()
    print(f"areas_geojson: {len(gj['features'])} features, {len(json.dumps(gj)) / 1e6:.1f} MB")
