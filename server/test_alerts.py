"""Scenario tests for the alert rules and the DWD thunderstorm / lightning feed (python server/test_alerts.py)."""
import io
import json
import os
import sys
import time
import types

import numpy as np
from PIL import Image

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import alerts  # noqa: E402
import thunder  # noqa: E402
from alerts import AlertEngine  # noqa: E402

MIN = 60_000
NOW = 1_800_000_000_000 - (1_800_000_000_000 % (5 * MIN))  # a fixed time, 5 min aligned
CFG = {"lead": 20, "early": True, "quiet": False, "tz": "Europe/Berlin"}


def series(rain_from=None, rain_to=None, rate=1.0, share=0.9, wet_obs=()):
    """12 observed frames (last hour) + 24 forecast frames; rain in the forecast from rain_from to rain_to minutes."""
    out = []
    for k in range(-11, 1):
        wet = k * 5 in wet_obs
        out.append((NOW + k * 5 * MIN, False, 0.9 if wet else 0.0, 1.0 if wet else 0.0))
    for k in range(1, 25):
        m = k * 5
        wet = rain_from is not None and rain_from <= m < (rain_to or 999)
        out.append((NOW + m * MIN, True, share if wet else 0.0, rate if wet else 0.0))
    return out


def run(name, s, st=None, cfg=CFG, now=NOW, expect=None, prob=False):
    st = {} if st is None else st
    kind, info = AlertEngine.evaluate(s, now, cfg, st, prob=prob)
    msg = AlertEngine.message(kind, info, "Hamburg", cfg["tz"]) if kind else None
    ok = "ok  " if kind == expect else "FAIL"
    print(f"{ok} {name}: {kind} {msg or ''}")
    return st, kind == expect


results = []
results.append(run("dry, no rain coming", series(), expect=None)[1])
results.append(run("rain in 60 min (light) -> nothing yet", series(60, 100), expect=None)[1])
results.append(run("rain in 20 min -> main alert", series(20, 60, rate=3), expect="main")[1])
results.append(run("heavy rain in 60 min, certain -> early", series(60, 90, rate=12), expect="early")[1])
results.append(run("heavy rain in 60 min, uncertain -> nothing", series(60, 90, rate=12, share=0.55), expect=None)[1])
results.append(run("short drizzle 5 min -> nothing", series(20, 25), expect=None)[1])
results.append(run("weak share (shower beside the place) -> nothing", series(20, 60, share=0.3), expect=None)[1])
results.append(run("already raining -> nothing", series(0, 60, wet_obs=(0, -5)), expect=None)[1])
results.append(run("rained 30 min ago (cooldown) -> nothing", series(20, 60, wet_obs=(-30,)), expect=None)[1])
# same event: main alert only once, then an upgrade when it turns heavy
st, ok = run("event: main alert", series(20, 60, rate=3), expect="main")
results.append(ok)
results.append(run("event: 5 min later, same rain -> no repeat", series(15, 55, rate=3), st=st, now=NOW + 5 * MIN, expect=None)[1])
results.append(run("event: now heavy -> one upgrade", series(10, 50, rate=12), st=st, now=NOW + 10 * MIN, expect="upgrade")[1])
results.append(run("event: still heavy -> no second upgrade", series(5, 45, rate=12), st=st, now=NOW + 15 * MIN, expect=None)[1])
# daily limit
st = {"day": None}
for i in range(4):
    st.pop("event_ts", None)
    run(f"limit: alert {i + 1}", series(20, 60), st=st, expect="main")
results.append(run("limit: 5th alert today -> nothing", series(20, 60), st={**st, "event_ts": 0}, expect=None)[1])
# quiet hours (NOW is 08:00 UTC = midnight in Los Angeles)
results.append(run("quiet hours at night -> nothing", series(20, 60), cfg={**CFG, "quiet": True, "tz": "America/Los_Angeles"}, expect=None)[1])
# own quiet hours (NOW = 08:00 UTC)
Q = {**CFG, "quiet": True, "tz": "UTC"}
results.append(run("quiet 07:30-09:00 at 08:00 -> nothing", series(20, 60), cfg={**Q, "quietFrom": 450, "quietTo": 540}, expect=None)[1])
results.append(run("quiet 22:00-07:00 at 08:00 -> alert", series(20, 60), cfg={**Q, "quietFrom": 1320, "quietTo": 420}, expect="main")[1])
results.append(run("quiet 23:00-08:30 (over midnight) at 08:00 -> nothing", series(20, 60), cfg={**Q, "quietFrom": 1380, "quietTo": 510}, expect=None)[1])
results.append(run("quiet 08:00-08:00 (empty window) -> alert", series(20, 60), cfg={**Q, "quietFrom": 480, "quietTo": 480}, expect="main")[1])
results.append(run("quiet hours off -> alert", series(20, 60), cfg={**Q, "quiet": False, "quietFrom": 450, "quietTo": 540}, expect="main")[1])

# probabilistic trigger (RAIN_ALERT_PROB, off by default): the same rules, the start from the calibrated probability
print("\nprobabilistic trigger (P_ALERT %.2f)" % alerts.P_ALERT)
results.append(alerts.PROB_ON == bool(os.environ.get("RAIN_ALERT_PROB")))  # off unless switched on
results.append(run("P: dry, no rain coming", series(), expect=None, prob=True)[1])
results.append(run("P: certain rain in 15 min -> main", series(15, 60, rate=3, share=1.0), expect="main", prob=True)[1])
k, info = AlertEngine.evaluate(series(15, 60, rate=3, share=1.0), NOW, CFG, {}, prob=True)
t, b = AlertEngine.message(k, info, "Hamburg", "Europe/Berlin")
results.append(info["prob"] >= 65 and info["prob"] % 5 == 0 and "%" not in t + b)  # no percentage in the text
results.append("%" not in "".join(AlertEngine.message(k, info, "Hamburg", "Europe/Berlin", "en")))
print(f"     text: {t} / {b}")
results.append(run("P: marginal share 0.55 in 20 min -> nothing", series(20, 60, share=0.55), expect=None, prob=True)[1])
results.append(run("   (the same with the old rule -> main)", series(20, 60, share=0.55), expect="main")[1])
results.append(run("P: weak share (shower beside the place) -> nothing", series(20, 60, share=0.3), expect=None, prob=True)[1])
results.append(run("P: short drizzle 5 min -> nothing", series(20, 25, share=1.0), expect=None, prob=True)[1])
results.append(run("P: already raining -> nothing", series(0, 60, share=1.0, wet_obs=(0, -5)), expect=None, prob=True)[1])
results.append(run("P: rained 30 min ago (cooldown) -> nothing", series(15, 60, share=1.0, wet_obs=(-30,)), expect=None, prob=True)[1])
results.append(run("P: heavy certain rain in ~55 min -> early", series(50, 120, rate=12, share=1.0), expect="early", prob=True)[1])
results.append(run("P: certain light rain in 60 min -> nothing yet", series(60, 120, share=1.0), expect=None, prob=True)[1])
st, ok = run("P: event: main alert", series(15, 60, rate=3, share=1.0), expect="main", prob=True)
results.append(ok)
results.append(run("P: event: 5 min later -> no repeat", series(10, 55, rate=3, share=1.0), st=st, now=NOW + 5 * MIN, expect=None, prob=True)[1])
# the probability itself: higher with more of the circle wet, lower with the lead, 0 far from any rain, never 1
P = alerts.rain_probability([[1.0] * 24, [0.6] * 24, [0.0] * 24], [0.0] * 3, [list(range(5, 125, 5))] * 3)
results.append(bool(P[0, 3] > P[1, 3] > 0 and P[0, 3] > P[0, 20] and P[2].max() == 0 and P.max() < 1))
# env values: broken ones fall back to the defaults
results.append(alerts._env_coef("1,2", alerts.PROB_COEF_DE) == alerts.PROB_COEF_DE and alerts._env_float("2", 0.65) == 0.65
               and alerts._env_coef(",".join(["0.5"] * 9), None) == (0.5,) * 9 and alerts._env_float("0.7", 0.65) == 0.7)

# DWD thunderstorm warnings (server-v1.79): one push per place and storm, not one per NowCastMIX update (new ID every
# 5 min); a higher level notifies again; thunder pushes count towards MAX_PER_DAY
print("\nthunderstorm pushes")


class FakeThunder:
    def __init__(self):
        self.ws = []

    def at(self, lat, lon, now_ms=None):
        return self.ws


def warn(i, sev, now):
    return {"id": f"nc{i}", "group": "Gewitter", "severity": sev, "level": thunder.LEVEL[sev], "onset": now - 5 * MIN,
            "expires": now + 55 * MIN, "rings": []}


def thunder_run(name, eng, st, sev, now, cfg=CFG, expect=True):
    eng.thunder.ws = [warn(now // MIN, sev, now)] if sev else []
    msg = AlertEngine.thunder_alert(eng, {"name": "Hamburg", "lat": 53.55, "lon": 10.0}, now, cfg, st)
    print(f"{'ok  ' if bool(msg) == expect else 'FAIL'} {name}: {msg and msg['title']}")
    return bool(msg) == expect


eng, st = types.SimpleNamespace(thunder=FakeThunder()), {"thunder_sent": ["old"]}
results.append(thunder_run("T: first warning -> push", eng, st, "Moderate", NOW))
results.append("thunder_sent" not in st and st["count"] == 1)
results.append(all([thunder_run(f"T: +{k * 5} min, new ID, same level -> nothing", eng, st, "Moderate", NOW + k * 5 * MIN,
                                expect=False) for k in range(1, 12)]))
results.append(thunder_run("T: +60 min, lower level -> nothing", eng, st, "Minor", NOW + 60 * MIN, expect=False))
results.append(thunder_run("T: +65 min, higher level -> push", eng, st, "Severe", NOW + 65 * MIN))
results.append(thunder_run("T: +70 min, back to Moderate -> nothing", eng, st, "Moderate", NOW + 70 * MIN, expect=False))
results.append(thunder_run("T: +150 min, still Severe (window from the last push) -> nothing", eng, st, "Severe",
                           NOW + 150 * MIN, expect=False))
results.append(thunder_run("T: +160 min, still warned, window over -> push", eng, st, "Moderate", NOW + 160 * MIN))
results.append(st["count"] == 3)
# several warnings at once: the highest level is the one notified
eng2, st2 = types.SimpleNamespace(thunder=FakeThunder()), {}
eng2.thunder.ws = [warn(1, "Minor", NOW), warn(2, "Severe", NOW)]
m = AlertEngine.thunder_alert(eng2, {"name": "Hamburg", "lat": 53.55, "lon": 10.0}, NOW, CFG, st2)
results.append(bool(m) and "Schweres" in m["title"] and st2["thunder_last"]["rank"] == 3)
# daily limit: thunder pushes use it up, and over the limit only severe ones come through
st3 = {}
for i in range(4):
    st3.pop("event_ts", None)
    run(f"T limit: rain alert {i + 1}", series(20, 60), st=st3, expect="main")
results.append(thunder_run("T limit: Moderate over the daily limit -> nothing", eng, st3, "Moderate", NOW, expect=False))
results.append(thunder_run("T limit: Severe over the daily limit -> push", eng, st3, "Severe", NOW))
st4 = {}
for i in range(3):
    thunder_run(f"T limit: thunder push {i + 1}", eng, st4, "Minor", NOW + i * 100 * MIN)
results.append(st4["count"] == 3
               and run("T limit: 1 rain alert left", series(20, 60), st=st4, expect="main")[1]
               and run("T limit: then none", series(20, 60), st={**st4, "event_ts": 0}, expect=None)[1])
# quiet hours (NOW = 08:00 UTC): Minor / Moderate wait, Severe comes through
QT = {**CFG, "quiet": True, "tz": "UTC", "quietFrom": 450, "quietTo": 540}
results.append(thunder_run("T quiet: Moderate -> nothing", eng, {}, "Moderate", NOW, cfg=QT, expect=False))
results.append(thunder_run("T quiet: Extreme -> push", eng, {}, "Extreme", NOW, cfg=QT))
results.append(thunder_run("T: thunder off -> nothing", eng, {}, "Severe", NOW, cfg={**CFG, "thunder": False}, expect=False))

# the 'remind me' button label comes with the payload, in the device's language
pe = types.SimpleNamespace(kind_at=None, message=AlertEngine.message)
k, info = AlertEngine.evaluate(series(45, 90, rate=3), NOW, {**CFG, "lead": 45}, {})
pl = {"name": "Hamburg", "lat": 53.55, "lon": 10.0}
d_de = AlertEngine.payload(pe, "s1", {**CFG, "api": "/api"}, k, info, pl, "k")
d_en = AlertEngine.payload(pe, "s1", {**CFG, "api": "/api", "lang": "en"}, k, info, pl, "k")
print(f"     remind button: {d_de.get('remindLabel')} / {d_en.get('remindLabel')}")
results.append(d_de.get("remindLabel") == f"In {d_de.get('remind')} Min. erinnern"
               and d_en.get("remindLabel") == f"Remind me in {d_en.get('remind')} min")

# thunder.py: the WFS mixes lon/lat and lat/lon per feature; each one is put in lon/lat or dropped
print("\nthunder.py: axis order, expiry, lightning age")
now_s = int(time.time())


def iso(t):
    return time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime(t))


def sq(lon, lat):
    return [[lon - .1, lat - .1], [lon + .1, lat - .1], [lon + .1, lat + .1], [lon - .1, lat + .1], [lon - .1, lat - .1]]


def feat(i, ring, ex, multi=False):
    return {"type": "Feature", "geometry": {"type": "MultiPolygon" if multi else "Polygon",
                                            "coordinates": [[ring]] if multi else [ring]},
            "properties": {"ID_ALERT": i, "EC_GROUP": "Gewitter", "SEVERITY": "Moderate", "ONSET": iso(now_s - 300),
                           "EXPIRES": iso(now_s + ex)}}


swap = lambda ring: [[b, a] for a, b in ring]
wfs = {"features": [feat("lonlat", sq(5.09, 47.34), 3600),         # as the WFS answered with our filter (20.11.2026)
                    feat("latlon", swap(sq(10.0, 53.55)), 3600),  # as some stored features come
                    feat("far", sq(100.0, 10.0), 3600),          # neither order inside the DWD area: dropped
                    feat("multi", swap(sq(13.4, 52.5)), 3600, multi=True),
                    feat("soon_expired", sq(8.0, 50.0), 2)]}
logs = []
th = thunder.Thunder(lambda url, timeout=0: json.dumps(wfs).encode(), logs.append)
th.refresh_warnings()
results.append(sorted(w["id"] for w in th.warnings) == ["latlon", "lonlat", "multi", "soon_expired"] and len(logs) == 1)
results.append(len(th.at(47.34, 5.09)) == 1 and len(th.at(5.09, 47.34)) == 0 and len(th.at(53.55, 10.0)) == 1
               and len(th.at(52.5, 13.4)) == 1)
gj = json.loads(th.geojson)["features"]
pts = np.concatenate([np.asarray(f["geometry"]["coordinates"], np.float64).reshape(-1, 2) for f in gj])
results.append(len(gj) == 4 and bool((pts[:, 0] < 30).all() and (pts[:, 1] > 40).all()))  # GeoJSON: lon first
print(f"     map areas, first point: {[np.asarray(f['geometry']['coordinates']).reshape(-1, 2)[0].round(2).tolist() for f in gj]}")
time.sleep(2.5)  # "soon_expired" expires: no longer served, without a new fetch
results.append(len(json.loads(th.geojson)["features"]) == 3 and not th.at(50.0, 8.0))
th.geojson = b'{"type":"FeatureCollection","features":[]}'  # devtools reset
results.append(th.geojson == b'{"type":"FeatureCollection","features":[]}')
# lightning: the data time from the WMS capabilities; stale data is not served
cap = b'<Layer><Dimension name="time" default="%s" units="ISO8601">...</Dimension></Layer>'
png = io.BytesIO()
img = np.zeros((thunder.LIGHTNING_SIZE[1], thunder.LIGHTNING_SIZE[0], 4), np.uint8)
img[400:404, 300:304, 3] = 255
Image.fromarray(img).save(png, "PNG")
urls = []
for age_min, served in ((20, True), (40, False)):
    t_data = iso((now_s - age_min * 60) // 300 * 300)
    th2 = thunder.Thunder(lambda url, timeout=0: (urls.append(url), cap % t_data.encode() if "GetCapabilities" in url
                                                  else png.getvalue())[1], print)
    th2.refresh_lightning()
    ok = th2._lightning[0] == thunder.iso_ms(t_data) and len(th2._lightning[1]) == 1 and bool(th2.lightning) == served
    print(f"{'ok  ' if ok else 'FAIL'} lightning {age_min} min old -> served: {bool(th2.lightning)}")
    results.append(ok)
n = len(urls)
th2.refresh_lightning()  # same time step again: only the capabilities are asked
results.append(len(urls) == n + 1 and any("&time=" in u for u in urls if "GetMap" in u))

print(f"\n{sum(results)}/{len(results)} passed")
sys.exit(0 if all(results) else 1)
