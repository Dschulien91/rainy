"""Our Europe rain forecast against a national weather service's nowcast, measured at rain gauges.

Second benchmark next to WetterOnline (whose map can only be read as "rain or not" from screenshots, up to +90 min):
GeoSphere Austria's INCA nowcast (open data, CC BY 4.0, dataset nowcast-v1-15min-1km: precipitation per 15 min on a
1 km grid for the next 3 h, a new run every 15 min, Austria and its surroundings 45.5-49.5 N / 8.1-17.7 E) - exact
numbers from an official forecaster, free to use, read through its public API with the neutral user agent.
Truth: GeoSphere's own TAWES gauges (~280 stations, 10-minute sums stamped at the END of the 10 min: "Niederschlag
der letzten 10 Minuten"), independent of our radar data.

Every half hour (2 min before :00 / :30 = T0) both forecasts are taken as a user would get them at that moment - each
with the age its data has then - for the four half hours after T0, at every gauge inside our radar coverage:
  inca  the newest INCA run: its two 15-min sums of each half hour
  own   our forecast as served (3 x 3 cells around the gauge, like /api/rain/point): mean rate x 0.5 h
  pers  the radar image at that moment, held ("nothing moves")
When the gauges have reported the two hours, one row per variant and half hour goes to verify_log/inca_<day>.jsonl
(only gauges where all three forecasts and the truth exist: the same cases): hits/misses/false alarms for "wet"
(>= 0.1 mm per 30 min) and "rain" (>= 0.5 mm per 30 min = 1 mm/h), and the amounts. Summed up by score_inca.py.
The forecasts waiting for the gauges are kept on disk (fcverify.Pending).
"""
import json
import os
import time
from datetime import datetime, timezone

import numpy as np

import monitor
from fcverify import LOG_DIR, Pending

API = "https://dataset.api.hub.geosphere.at/v1"   # 5 requests/s and 240/h per address; this module uses ~8/h
HALF_MS = 30 * 60_000
LEADS = (30, 60, 90, 120)       # end of each half hour after T0, minutes
TAKE_BEFORE_S = 120             # the forecasts are taken this long before T0
RETRY_S = 90                    # a take that fails is tried again for this long (both forecasts are still taken before T0)
CHUNK = 140                     # gauges per forecast request
WAIT_MS = 25 * 60_000           # the gauges' last 10 min are there by then
GIVE_UP_MS = 8 * 3600_000
WET, RAIN = 0.1, 0.5            # mm per 30 min


def _ms(iso):
    return int(datetime.fromisoformat(iso).timestamp() * 1000)


class IncaBench:
    def __init__(self, fetch, log, region, merc):
        self.fetch, self.log, self.region, self.merc = fetch, log, region, merc
        self.pending = Pending("inca")
        self.stations, self.listed_day = [], None  # (id, lat, lon, row, col)
        self.done = None

    # ---------- inputs ----------
    def _station_list(self):
        meta = json.loads(self.fetch(f"{API}/station/current/tawes-v1-10min/metadata", timeout=60))
        r = self.region
        out = []
        for s in meta.get("stations", []):
            if not s.get("is_active"):
                continue
            col, row = (int(round(float(v))) for v in r.to_pixels(*self.merc(float(s["lon"]), float(s["lat"]))))
            if 1 <= row < r.NY - 1 and 1 <= col < r.NX - 1:
                out.append((str(s["id"]), float(s["lat"]), float(s["lon"]), row, col))
        return out

    def _inca(self, t0):
        """(4, n) mm per half hour from the newest INCA run, NaN where a step is missing; and that run's time."""
        out = np.full((len(LEADS), len(self.stations)), np.nan, np.float32)
        ref = None
        for i in range(0, len(self.stations), CHUNK):
            part = self.stations[i:i + CHUNK]
            q = "&".join(f"lat_lon={lat:.4f},{lon:.4f}" for _, lat, lon, _, _ in part)
            # one short try: when the service hangs (11.11. 12:30-14 UTC it answered nothing for 120 s x 3 tries per
            # chunk), the take is tried again by loop() until RETRY_UNTIL_S after T0 instead of blocking for minutes
            d = json.loads(self.fetch(f"{API}/timeseries/forecast/nowcast-v1-15min-1km?parameters=rr&output_format=geojson&{q}",
                                      timeout=40, tries=1))
            ref = _ms(d["reference_time"])
            ts = {_ms(t): k for k, t in enumerate(d["timestamps"])}
            for j, f in enumerate(d["features"]):  # in the order asked
                v = f["properties"]["parameters"]["rr"]["data"]
                for a, lead in enumerate(LEADS):
                    end = t0 + lead * 60_000
                    two = [v[ts[t]] if t in ts else None for t in (end - 15 * 60_000, end)]
                    if None not in two:
                        out[a, i + j] = two[0] + two[1]
            time.sleep(1)
        return out, ref

    def _ours(self, t0):
        """(4, n) mm per half hour of our forecast, (n,) of the newest radar image held, and its time."""
        r = self.region
        with r.lock:
            ref, frames = r.ref, {**r.obs, **r.fc}
        if not ref:
            return None, None, None

        def at(g):  # 3 x 3 mean rate at each gauge, NaN outside the radar coverage
            v = np.full(len(self.stations), np.nan, np.float32)
            for k, (_, _, _, row, col) in enumerate(self.stations):
                c = np.asarray(g[row - 1:row + 2, col - 1:col + 2], np.float32)
                if (c >= 0).all():
                    v[k] = c.mean()
            return v
        own = np.full((len(LEADS), len(self.stations)), np.nan, np.float32)
        for a, lead in enumerate(LEADS):
            end = t0 + lead * 60_000
            steps = [t for t in frames if end - HALF_MS < t <= end]
            if len(steps) >= 4:  # 5-minute frames: 6 per half hour (the last one or two may lie beyond our 2 h)
                own[a] = np.mean([at(frames[t]) for t in steps], axis=0) * 0.5
        return own, at(frames[ref]) * 0.5, ref

    def take(self, t0):
        if time.strftime("%Y%m%d") != self.listed_day or not self.stations:
            self.stations, self.listed_day = self._station_list(), time.strftime("%Y%m%d")
        if not self.stations:
            return
        own, pers, ref = self._ours(t0)
        if own is None:
            return
        inca, inca_ref = self._inca(t0)
        self.pending.put(("fc", t0, 0), ids=np.array([s[0] for s in self.stations], dtype=str), inca=inca, own=own,
                         pers=pers, age=np.array([(t0 - ref) / 60_000, (t0 - inca_ref) / 60_000], np.float32))

    # ---------- scoring ----------
    def _truth(self, t0, ids):
        """{station id: [mm in each half hour after t0]} for the gauges that reported every 10 min of the two hours."""
        start, end = t0 + 10 * 60_000, t0 + LEADS[-1] * 60_000
        iso = lambda ms: datetime.fromtimestamp(ms / 1000, timezone.utc).strftime("%Y-%m-%dT%H:%M")
        d = json.loads(self.fetch(f"{API}/station/historical/tawes-v1-10min?parameters=RR&station_ids={','.join(ids)}"
                                  f"&start={iso(start)}&end={iso(end)}", timeout=120))
        ts = [_ms(t) for t in d["timestamps"]]
        out = {}
        for f in d["features"]:
            v = dict(zip(ts, f["properties"]["parameters"]["RR"]["data"]))
            sums = []
            for lead in LEADS:
                three = [v.get(t0 + lead * 60_000 - k * 600_000) for k in (2, 1, 0)]
                sums.append(None if any(x is None or x < 0 for x in three) else float(sum(three)))
            if None not in sums:
                out[str(f["properties"]["station"])] = sums
        return out

    def score(self):
        now = time.time() * 1000
        lines = []
        for key in self.pending:
            t0 = key[1]
            if now < t0 + LEADS[-1] * 60_000 + WAIT_MS:
                continue
            z = self.pending.get(key)
            if z is None or now - t0 > GIVE_UP_MS:
                self.pending.drop(key)
                continue
            ids = [str(s) for s in z["ids"]]
            truth = self._truth(t0, ids)  # raises when the service is down: the forecast stays pending
            self.pending.drop(key)
            obs = np.array([truth.get(s, [np.nan] * len(LEADS)) for s in ids], np.float32).T  # (4, n)
            for a, lead in enumerate(LEADS):
                ok = np.isfinite(obs[a]) & np.isfinite(z["inca"][a]) & np.isfinite(z["own"][a]) & np.isfinite(z["pers"])
                if ok.sum() < 20:
                    continue
                o = obs[a][ok]
                for name, f in (("inca", z["inca"][a][ok]), ("own", z["own"][a][ok]), ("pers", z["pers"][ok])):
                    row = {"variant": name, "t0": int(t0), "lead": lead, "valid": int(t0 + lead * 60_000), "v": monitor.TAG,
                           "n": int(ok.sum()), "age": round(float(z["age"][1 if name == "inca" else 0]), 1),
                           "sum": [round(float(f.sum()), 2), round(float(o.sum()), 2)]}
                    for label, th in (("wet", WET), ("rain", RAIN)):
                        fw, ow = f >= th, o >= th
                        row[label] = [int((fw & ow).sum()), int((ow & ~fw).sum()), int((fw & ~ow).sum())]
                    lines.append(row)
        if lines:
            day = datetime.fromtimestamp(lines[0]["valid"] / 1000, timezone.utc).strftime("%Y%m%d")
            with open(os.path.join(LOG_DIR, f"inca_{day}.jsonl"), "a") as fh:
                for row in lines:
                    fh.write(json.dumps(row) + "\n")
        return len(lines)

    def loop(self):
        last_score = 0.0
        while True:
            try:
                t0 = (int(time.time() * 1000) // HALF_MS + 1) * HALF_MS  # the next :00 / :30
                left = t0 / 1000 - time.time()
                if t0 != self.done and left <= TAKE_BEFORE_S:
                    try:
                        self.take(t0)
                        self.done = t0
                    except Exception:
                        if left < TAKE_BEFORE_S - RETRY_S:  # no answer: give this half hour up
                            self.done = t0
                        raise
                if time.time() - last_score > 600:
                    last_score = time.time()
                    n = self.score()
                    if n:
                        self.log(f"[{datetime.now():%H:%M:%S}] inca benchmark: {n} lines")
            except Exception as e:
                self.log(f"inca benchmark failed: {e}")
            time.sleep(20)
