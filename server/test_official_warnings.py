"""Offline tests for official_warnings.py (no network): python -m unittest test_official_warnings  (in server/)."""
import io
import json
import math
import tempfile
import time
import unittest
import zipfile
from datetime import datetime, timedelta, timezone

import numpy as np

import official_warnings as ow


def iso(dt):
    return dt.strftime("%Y-%m-%dT%H:%M:%S+00:00")


NOW = datetime.now(timezone.utc).replace(microsecond=0)
SQUARE = [[0, 0], [1, 0], [1, 1], [0, 1], [0, 0]]


def cap(lonlat):
    return " ".join(f"{lat},{lon}" for lon, lat in lonlat)


class FakeFetch:
    def __init__(self, routes):
        self.routes, self.calls = routes, []

    def __call__(self, url, timeout=40, tries=3):
        self.calls.append(url)
        for prefix, body in self.routes.items():
            if url.startswith(prefix):
                return body() if callable(body) else body
        raise OSError(f"404 {url}")


class Geometry(unittest.TestCase):
    def test_iso_ms(self):
        self.assertEqual(ow.iso_ms("2026-11-16T03:00:00+01:00"), ow.iso_ms("2026-11-16T02:00:00Z"))
        self.assertEqual(ow.iso_ms("2026-11-16T02:00:00.000Z"), 1794794400000)
        self.assertIsNone(ow.iso_ms("None"))
        self.assertIsNone(ow.iso_ms(""))

    def test_simplify(self):
        t = np.linspace(0, 2 * math.pi, 2001)
        circle = np.c_[np.cos(t), np.sin(t)]  # radius 1 deg, 2001 points
        s = ow.simplify(circle, 0.005)
        self.assertLess(len(s), 60)
        self.assertTrue(np.array_equal(s[0], s[-1]))
        self.assertEqual(len(ow.simplify(SQUARE)), 5)
        tiny = [[0, 0], [0.001, 0], [0.001, 0.001], [0, 0.001], [0, 0]]
        self.assertEqual(len(ow.simplify(tiny)), 5)  # kept as its bounding box, not dropped

    def test_shape_holes_and_islands(self):
        outer = [[0, 0], [10, 0], [10, 10], [0, 10], [0, 0]]
        hole = [[3, 3], [7, 3], [7, 7], [3, 7], [3, 3]]
        island = [[4, 4], [6, 4], [6, 6], [4, 6], [4, 4]]
        s = ow.make_shape([[outer, hole], [island]])
        self.assertTrue(s.contains(1, 1))
        self.assertFalse(s.contains(3.5, 3.5))   # in the hole
        self.assertTrue(s.contains(5, 5))        # on the island inside the hole
        self.assertFalse(s.contains(11, 5))
        gj = s.geojson()
        self.assertEqual(gj["type"], "MultiPolygon")
        self.assertEqual(len(gj["coordinates"]), 2)

    def test_cap_shape_exclude(self):
        s = ow.cap_shape([cap([[0, 0], [10, 0], [10, 10], [0, 10], [0, 0]])],
                         [cap([[3, 3], [7, 3], [7, 7], [3, 7], [3, 3]])])
        self.assertTrue(s.contains(1, 1))
        self.assertFalse(s.contains(5, 5))

    def test_classify(self):
        cases = {"Tornado Warning": "thunderstorm", "Severe Thunderstorm Watch": "thunderstorm", "Flash Flood Warning": "flood",
                 "Coastal Flood Advisory": "coastal", "Small Craft Advisory": "coastal", "Winter Storm Watch": "snow-ice",
                 "Gale Warning": "wind", "Storm Warning": "wind", "Wind Chill Advisory": "cold", "Freeze Watch": "cold",
                 "Excessive Heat Warning": "heat", "Dense Fog Advisory": "fog", "Red Flag Warning": "fire",
                 "fog advisory": "fog", "freezing drizzle advisory": "snow-ice", "blizzard warning": "snow-ice",
                 "rainfall warning": "rain", "arctic outflow warning": "cold", "Special Weather Statement": "other",
                 "Please take notice of this advice": "other"}  # 'notice' must not match 'ice'
        for name, want in cases.items():
            self.assertEqual(ow.classify(name), want, name)

    def test_pick_info(self):
        infos = [("de-DE", "a", "A", ""), ("en-GB", "b", "B", ""), ("fr", "c", "C", "")]
        self.assertEqual(ow.pick_info(infos, "de")[0], "de-DE")
        self.assertEqual(ow.pick_info(infos, "fr-CH")[0], "fr")
        self.assertEqual(ow.pick_info(infos, "it")[0], "en-GB")
        self.assertEqual(ow.pick_info([("sr", "x", "X", "")], "de")[0], "sr")


def dwd_zip(alerts):
    buf = io.BytesIO()
    with zipfile.ZipFile(buf, "w") as z:
        for name, xml in alerts.items():
            z.writestr(name, xml)
    return buf.getvalue()


def dwd_xml(ident, event, severity, group, onset, expires, polygon=None, exclude=None, cells=(), refs=""):
    area = ""
    if polygon:
        ex = f"<geocode><valueName>EXCLUDE_POLYGON</valueName><value>{exclude}</value></geocode>" if exclude else ""
        area += f"<area><areaDesc>polygonal event area</areaDesc><polygon>{polygon}</polygon>{ex}</area>"
    for cid, desc in cells:
        area += f"<area><areaDesc>{desc}</areaDesc><geocode><valueName>WARNCELLID</valueName><value>{cid}</value></geocode></area>"
    info = lambda lang, ev: (f"<info><language>{lang}</language><event>{ev}</event><severity>{severity}</severity>"
                             f"<eventCode><valueName>GROUP</valueName><value>{group}</value></eventCode>"
                             f"<onset>{iso(onset)}</onset><expires>{iso(expires)}</expires><senderName>Deutscher Wetterdienst</senderName>"
                             f"<headline>Amtliche WARNUNG vor {ev}</headline><description>Text {lang}.</description>{area}</info>")
    return (f'<?xml version="1.0" encoding="UTF-8"?><alert xmlns="urn:oasis:names:tc:emergency:cap:1.2">'
            f"<identifier>{ident}</identifier><sent>{iso(NOW)}</sent><status>Actual</status><msgType>Alert</msgType>"
            f"<references>{refs}</references>{info('de-DE', event)}{info('en', event.lower())}</alert>")


class Sources(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()

    def tearDown(self):
        self.tmp.cleanup()

    def make(self, routes, regions):
        self.logs = []
        return ow.Warnings(FakeFetch(routes), self.logs.append, self.tmp.name, regions=regions)

    def test_dwd(self):
        land = cap([[8, 50], [10, 50], [10, 52], [8, 52], [8, 50]])
        hole = cap([[8.8, 50.8], [9.2, 50.8], [9.2, 51.2], [8.8, 51.2], [8.8, 50.8]])
        alerts = {
            "a.xml": dwd_xml("A", "STURMBÖEN", "Moderate", "WIND", NOW - timedelta(hours=1), NOW + timedelta(hours=5),
                             land, hole, cells=[("809372117", "Gemeinde X"), ("501000005", "Nordfriesische Küste")]),
            "b.xml": dwd_xml("B", "GEWITTER", "Minor", "THUNDERSTORM", NOW + timedelta(hours=30), NOW + timedelta(hours=32), land),
            "c.xml": dwd_xml("C", "FROST", "Minor", "FROST", NOW - timedelta(hours=5), NOW - timedelta(hours=1), land),
            "d.xml": dwd_xml("D", "DAUERREGEN", "Severe", "RAIN", NOW - timedelta(hours=1), NOW + timedelta(hours=5), land),
            "e.xml": dwd_xml("E", "DAUERREGEN", "Severe", "RAIN", NOW - timedelta(hours=1), NOW + timedelta(hours=5), land,
                             refs="opendata@dwd.de,D,2026-11-16T12:00:00+01:00"),
        }
        coast = {"features": [{"properties": {"WARNCELLID": 501000005, "NAME": "Nordfriesische Küste"},
                               "geometry": {"type": "Polygon", "coordinates": [[[7, 54], [9, 54], [9, 55], [7, 55], [7, 54]]]}}]}
        w = self.make({ow.DWD_CAP: dwd_zip(alerts), ow.DWD_WFS: json.dumps(coast).encode()}, ("de",))
        w.refresh("dwd")
        res = w.at(51.5, 9.5, lang="en")
        self.assertEqual([(r["id"], r["level"], r["event"]) for r in res], [("E", 4, "rain"), ("A", 3, "wind")])
        self.assertEqual((res[1]["language"], res[1]["text"], res[1]["description"]), ("en", "Text en.", "Text en."))
        self.assertEqual(res[1]["area"], "Gemeinde X")   # the polygon is named after its only municipality
        self.assertEqual([r["id"] for r in w.at(51.0, 9.0)], ["E"])  # A has an EXCLUDE_POLYGON there, E not
        self.assertEqual([r["id"] for r in w.at(54.5, 8.0)], ["A"])  # coast cell via WFS geometry
        self.assertEqual(w.stats()["dwd"]["unresolved"], 0)
        fc = w.areas_geojson((7.5, 49.5, 10.5, 52.5))
        self.assertEqual(len(fc["features"]), 2)            # A's polygon and E's polygon (coast is outside the bbox)
        self.assertEqual(max(f["properties"]["level"] for f in fc["features"]), 4)

    def test_meteoalarm(self):
        def alert(ident, level, typ, geocodes=None, polygon=None, refs=None, hours=(-1, 6)):
            area = {"areaDesc": f"Area {ident}"}
            if geocodes:
                area["geocode"] = [{"valueName": k, "value": v} for k, v in geocodes]
            if polygon:
                area["polygon"] = [polygon]
            info = lambda lang: {"language": lang, "event": f"ev-{lang}", "headline": f"H-{lang}", "description": "D",
                                 "onset": iso(NOW + timedelta(hours=hours[0])), "expires": iso(NOW + timedelta(hours=hours[1])),
                                 "senderName": "MeteoSwiss", "area": [area],
                                 "parameter": [{"valueName": "awareness_level", "value": f"{level}; x; y"},
                                               {"valueName": "awareness_type", "value": f"{typ}; z"}]}
            a = {"identifier": ident, "status": "Actual", "msgType": "Alert", "sent": iso(NOW), "info": [info("de-CH"), info("en")]}
            if refs:
                a["references"] = refs
            return {"alert": a}
        feed = {"warnings": [
            alert("G", 1, 1, geocodes=[("EMMA_ID", "XX001")]),                       # green: dropped
            alert("Y", 2, 3, geocodes=[("NUTS3", "XX11")]),                          # via alias
            alert("O", 3, 10, polygon=cap([[0, 0], [1, 0], [1, 1], [0, 1], [0, 0]])),
            alert("OLD", 3, 10, geocodes=[("EMMA_ID", "XX001")]),
            alert("NEW", 4, 10, geocodes=[("EMMA_ID", "XX001")], refs="s,OLD,2026-11-16T00:00:00Z"),
            alert("U", 2, 1, geocodes=[("EMMA_ID", "ZZ999")]),                       # unknown geocode
        ]}
        w = self.make({ow.MA_FEED.format("switzerland"): json.dumps(feed).encode()}, ("eu",))
        w.ma_geo = {"XX001": ow.make_shape([[SQUARE]])}
        w.ma_alias = {"NUTS3:XX11": "XX001", "XX11": "XX001"}
        w.refresh_meteoalarm_country("switzerland")
        res = w.at(0.5, 0.5, lang="de")
        self.assertEqual([(r["id"], r["level"], r["event"]) for r in res],
                         [("NEW", 4, "rain"), ("O", 3, "rain"), ("Y", 2, "thunderstorm")])
        self.assertEqual(res[0]["language"], "de-CH")
        st = w.stats()["meteoalarm:switzerland"]
        self.assertEqual((st["alerts"], st["resolved"], st["unresolved"]), (4, 3, 1))
        self.assertIn("EMMA_ID=ZZ999", st["unresolved_examples"])
        self.assertIn("germany", w.ma_skip)

    def test_nws_and_eccc(self):
        poly = {"type": "Polygon", "coordinates": [[[-100, 40], [-99, 40], [-99, 41], [-100, 41], [-100, 40]]]}
        zone = {"type": "Feature", "properties": {"name": "Zone One"},
                "geometry": {"type": "GeometryCollection", "geometries": [
                    {"type": "MultiPolygon", "coordinates": [[[[-90, 30], [-89, 30], [-89, 31], [-90, 31], [-90, 30]]]]}]}}

        def feat(i, event, severity, geometry=None, zones=()):
            return {"id": i, "geometry": geometry, "properties": {
                "id": i, "event": event, "severity": severity, "status": "Actual", "messageType": "Alert",
                "sent": iso(NOW), "onset": iso(NOW), "ends": iso(NOW + timedelta(hours=3)), "expires": iso(NOW + timedelta(hours=1)),
                "headline": event, "description": "x", "senderName": "NWS Test", "areaDesc": "somewhere",
                "affectedZones": [f"{ow.NWS_ZONE_PREFIX}forecast/{z}" for z in zones], "references": []}}
        nws = {"features": [feat("t", "Tornado Warning", "Extreme", poly), feat("s", "Winter Storm Watch", "Severe", zones=["XXZ001"]),
                            feat("m", "Special Weather Statement", "Moderate", zones=["XXZ001"])]}
        eccc = {"features": [{"id": "c1", "geometry": {"type": "Polygon", "coordinates": [[[-80, 50], [-79, 50], [-79, 51], [-80, 51], [-80, 50]]]},
                              "properties": {"alert_type": "warning", "risk_colour_en": "orange", "alert_name_en": "rainfall warning",
                                             "alert_name_fr": "avertissement de pluie", "alert_text_en": "Rain.", "alert_text_fr": "Pluie.",
                                             "status_en": "issued", "publication_datetime": iso(NOW), "validity_datetime": iso(NOW),
                                             "event_end_datetime": iso(NOW + timedelta(hours=6)), "feature_name_en": "Region",
                                             "feature_name_fr": "Région"}},
                             {"id": "c2", "geometry": None, "properties": {"alert_type": "advisory", "status_en": "ended"}}]}
        w = self.make({ow.NWS_ALERTS: json.dumps(nws).encode(), ow.NWS_ZONE_PREFIX: json.dumps(zone).encode(),
                       "https://api.weather.gc.ca/": json.dumps(eccc).encode()}, ("na",))
        ow.ZONE_PAUSE_S = 0
        w.refresh("nws")
        w.refresh("eccc")
        self.assertEqual([(r["level"], r["event"]) for r in w.at(40.5, -99.5)], [(4, "thunderstorm")])
        res = w.at(30.5, -89.5)
        self.assertEqual([(r["level"], r["event"], r["area"]) for r in res], [(2, "snow-ice", "Zone One"), (1, "other", "Zone One")])
        self.assertEqual(res[0]["expires"], ow.iso_ms(iso(NOW + timedelta(hours=3))))  # 'ends', not the message expiry
        r = w.at(50.5, -79.5, lang="fr")
        self.assertEqual((r[0]["level"], r[0]["event"], r[0]["headline"], r[0]["area"]), (3, "rain", "Avertissement de pluie", "Région"))
        # the zone geometry is cached on disk and reused by a new instance without fetching it again
        w2 = self.make({ow.NWS_ALERTS: json.dumps(nws).encode()}, ("na",))
        w2.refresh("nws")
        self.assertEqual(len(w2.at(30.5, -89.5)), 2)

    def test_lookup_speed(self):
        shapes, links, alerts = {}, {}, {}
        rng = np.random.default_rng(1)
        for i in range(3000):  # 3000 random 0.5 deg blobs of 200 points over Europe
            x, y = rng.uniform(-10, 30), rng.uniform(36, 70)
            t = np.linspace(0, 2 * math.pi, 200)
            ring = np.c_[x + 0.5 * np.cos(t) * (1 + 0.2 * np.sin(5 * t)), y + 0.5 * np.sin(t)]
            shapes[f"k{i}"] = ow.make_shape([[ring]])
            links[f"k{i}"] = [(f"a{i}", "area")]
            alerts[f"a{i}"] = ow.Warnings._alert("x", f"a{i}", onset=0, expires=None, infos=[("en", "e", "h", "d")])
        w = ow.Warnings(lambda *a, **k: b"", print, self.tmp.name, regions=("de",))
        w.states["x"] = ow.State("x", alerts, links, shapes)
        pts = rng.uniform([36, -10], [70, 30], (1000, 2))
        t0 = time.perf_counter()
        n = sum(len(w.at(lat, lon)) for lat, lon in pts)
        dt = time.perf_counter() - t0
        self.assertGreater(n, 0)
        self.assertLess(dt, 2.0, f"{dt / len(pts) * 1000:.2f} ms per lookup")


if __name__ == "__main__":
    unittest.main()
