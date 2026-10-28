"""Scenario tests for the alert rules (python server/test_alerts.py)."""
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
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


def run(name, s, st=None, cfg=CFG, now=NOW, expect=None):
    st = {} if st is None else st
    kind, info = AlertEngine.evaluate(s, now, cfg, st)
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
print(f"\n{sum(results)}/{len(results)} passed")
