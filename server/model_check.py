"""Which weather model is best for what the app shows beyond the 2 h nowcast: the 24 h strip (hourly), the 3-day strip
(4-hour blocks) and the 7-day strip (one entry per day from 10-20 h local; app v1.64).

User, 27.11.2026: "wir müssen alle Daten, die wir anzeigen - drei Tage, vierundzwanzig Stunden, die hundertzwanzig
Minuten - durchmessen ... es gibt ja mehrere Quellen ... immer wieder prüfen, welche Quelle wohl am besten ist, und die
nehmen wir dann". The 120 minutes are scored since November (score_same.py, score.py, check_*_gauges.py); this module
records the rest. It runs in the OPERA instance (api-eu) for Europe and North America:

  ObsLog   every 10 min the METAR cache of aviationweather.gov (worldwide, open, the same file northamerica.py reads):
           temperature, precipitation / thunder in the present or recent weather, cloud cover of a fixed station
           sample -> verify_log/models/obs/<YYYYMMDD>.jsonl. Stations: one per 1 degree cell (Europe) / 2 degree cell
           (North America), picked once from the reports of the first 3 hours (the cache holds only the newest
           report of each station, ~1 h) -> verify_log/models/stations.json.
  FcLog    4x a day (02:40, 08:40, 14:40, 20:40 UTC) every candidate model's hourly forecast for the next 170 h at those
           stations from our self-hosted Open-Meteo (models not on its disk come from Open-Meteo's open-data mirror
           through it - nobody learns who asks) -> verify_log/models/fc/<region>_<issue ms>.npz (21 days).

The app's own choice ("app": the regional model of omModel() in index.html, its gaps from ICON) is put together from
the logged models when scoring (check_models.py). Candidates: what our Open-Meteo answers for 7 days (tested
27.11.2026: ICON, ECMWF IFS, ECMWF AIFS, GEM, UKMO global (6.4 days), GFS on our disk for North America, NBM for the
US) plus the regional models the app already uses in their own areas.
"""
import gzip
import json
import os
import re
import threading
import time
from datetime import datetime, timezone

import numpy as np

METAR_URL = "https://aviationweather.gov/data/cache/metars.cache.csv.gz"
HOURS = 170
KEEP_DAYS = 21
ISSUE_HOURS_UTC = (2, 8, 14, 20)   # at :40
CHUNK = 50
# models not on our disk come from the open-data mirror: cold, ~2-4 s per station (27.11.2026: GEM seamless 196 s,
# NBM 100 s for 50 stations; ECMWF / GEM global / UKMO 0.2-0.5 s once read)
MODEL_MAX_S = 25 * 60
# regions: (lat0, lat1, lon0, lon1), sample cell in degrees, ICAO prefixes kept
REGIONS = {
    "eu": ((34.0, 72.0, -25.0, 45.0), 1.0, ("E", "L", "BI", "UK", "UM")),
    "na": ((14.0, 72.0, -170.0, -50.0), 2.0, ("K", "C", "PA")),
}
SKIP_PREFIX = ("LL",)   # Israel: south of the Europe box mostly, not the app's area
MODELS = {
    "eu": ["icon_seamless", "ecmwf_ifs025", "ecmwf_aifs025_single", "gem_global", "ukmo_seamless",
           # the app's regional models, asked only for the stations in their app areas
           "metno_nordic", "ukmo_uk_deterministic_2km", "italia_meteo_arpae_icon_2i"],
    "na": ["icon_seamless", "gfs_seamless", "ecmwf_ifs025", "ecmwf_aifs025_single", "gem_global", "ncep_nbm_conus"],
}
REGIONAL = {"metno_nordic", "ukmo_uk_deterministic_2km", "italia_meteo_arpae_icon_2i"}


def asked(model, s):
    """Is model asked for at station s ([id, lat, lon])? A location outside a model's domain fails the whole request."""
    if model in REGIONAL:
        return om_model(s[1], s[2]) == model
    if model == "ncep_nbm_conus":
        return s[0].startswith("K")
    return True
VARS = ("temperature_2m", "precipitation", "cloud_cover", "weather_code")
SKY_COVER = {"CLR": 0.0, "SKC": 0.0, "NCD": 0.0, "NSC": 0.0, "CAVOK": 0.0, "FEW": 0.19, "SCT": 0.44, "BKN": 0.75,
             "OVC": 1.0, "OVX": 1.0, "VV": 1.0}
# precipitation or thunder at the station (not "VC" = in the vicinity); RE.. = recent weather (since the last report)
WX_WET = re.compile(r"^[-+]?(SH|TS|FZ|BC)?(TS|DZ|RA|SN|SG|PL|GR|GS|UP)")
RE_WET = re.compile(r"^RE(SH|TS|FZ)?(TS|DZ|RA|SN|SG|PL|GR|GS|UP)")


def om_model(lat, lon):
    """The app's model choice (index.html omModel, v1.36)."""
    if ((lat >= 55.3 and 4.5 <= lon < 32 and not (lat < 57.8 and lon < 12.7 and not (lat > 56.3 and lon > 11.8)))
            or (53.9 <= lat < 55.3 and 22.9 <= lon < 27)):
        return "metno_nordic"
    if 49.8 < lat < 61 and -11 < lon < 1.8 and not (lat < 50.2 and lon > 0):
        return "ukmo_uk_deterministic_2km"
    if (35.3 < lat < 43.75 and 7.9 < lon < 18.6 and not (lat > 41.3 and lon < 9.8) and not (lat > 42.3 and lon > 15.5)
            and not (lat < 38.5 and lon < 11.5)):
        return "italia_meteo_arpae_icon_2i"
    return "icon_seamless"


def region_of(sid, lat, lon):
    if sid.startswith(SKIP_PREFIX):
        return None
    for name, ((a, b, c, d), _, prefixes) in REGIONS.items():
        if a <= lat <= b and c <= lon <= d and sid.startswith(prefixes):
            return name
    return None


def parse_metar_csv(raw):
    """(station, lat, lon, time ms, temp C or None, wet now 0/1, wet recently 0/1, cloud cover 0..1 or None)."""
    import csv
    rows = list(csv.reader(gzip.decompress(raw).decode("utf-8", "replace").splitlines()))
    if not rows or rows[0][0] != "raw_text":
        return
    head = rows[0]
    col = {name: i for i, name in enumerate(head) if name not in ("sky_cover", "cloud_base_ft_agl")}
    sky = [i for i, name in enumerate(head) if name == "sky_cover"]
    for r in rows[1:]:
        try:
            sid = r[col["station_id"]]
            lat, lon = float(r[col["latitude"]]), float(r[col["longitude"]])
            t = int(datetime.strptime(r[col["observation_time"]][:19], "%Y-%m-%dT%H:%M:%S")
                    .replace(tzinfo=timezone.utc).timestamp() * 1000)
        except (ValueError, IndexError, KeyError):
            continue
        try:
            tc = float(r[col["temp_c"]])
        except (ValueError, IndexError):
            tc = None
        body = r[0].split(" RMK ")[0]
        tokens = body.split()
        wx = r[col["wx_string"]].split() if col.get("wx_string") is not None and col["wx_string"] < len(r) else []
        wet = int(any(WX_WET.match(w) for w in wx))
        recent = int(any(RE_WET.match(w) for w in tokens))
        covers = [SKY_COVER[r[i]] for i in sky if i < len(r) and r[i] in SKY_COVER]
        if not covers and re.search(r" (CLR|SKC|CAVOK|NCD|NSC) ", f" {body} "):
            covers = [0.0]
        yield sid, lat, lon, t, tc, wet, recent, (max(covers) if covers else None)


class ModelCheck:
    def __init__(self, fetch, log, log_dir, om_url):
        self.fetch, self.log, self.om = fetch, log, om_url.rstrip("/")
        self.dir = os.path.join(log_dir, "models")
        for sub in ("obs", "fc"):
            os.makedirs(os.path.join(self.dir, sub), exist_ok=True)
        self.st_path = os.path.join(self.dir, "stations.json")
        self.seen = set()
        self.cand, self.cand_since = {}, time.time()   # reports seen before the stations are picked

    # ---------- stations ----------
    def stations(self):
        """{region: [[id, lat, lon], ...]} (order fixed once picked)"""
        try:
            with open(self.st_path) as f:
                return json.load(f)
        except (OSError, ValueError):
            return {}

    def _pick(self, reports):
        """One station per sample cell and region: the one nearest the cell centre (stable for the same reports)."""
        best = {}
        for sid, lat, lon, *_ in reports:
            reg = region_of(sid, lat, lon)
            if reg is None or not re.fullmatch(r"[A-Z]{4}", sid):
                continue
            cell = REGIONS[reg][1]
            key = (reg, int((lat + 90) // cell), int((lon + 180) // cell))
            cy, cx = (key[1] + 0.5) * cell - 90, (key[2] + 0.5) * cell - 180
            d = (lat - cy) ** 2 + (lon - cx) ** 2
            if key not in best or d < best[key][0]:
                best[key] = (d, [sid, round(lat, 4), round(lon, 4)])
        out = {}
        for (reg, *_), (_, s) in sorted(best.items()):
            out.setdefault(reg, []).append(s)
        return out

    # ---------- observations ----------
    def poll_obs(self):
        reports = list(parse_metar_csv(self.fetch(METAR_URL, timeout=60)))
        st = self.stations()
        if not st:
            for r in reports:
                self.cand[r[0]] = r
            if time.time() - self.cand_since < 3 * 3600:
                return 0
            st = self._pick(list(self.cand.values()))
            self.cand = {}
            with open(self.st_path + ".tmp", "w") as f:
                json.dump(st, f)
            os.replace(self.st_path + ".tmp", self.st_path)
            self.log("models: stations picked - " + ", ".join(f"{k} {len(v)}" for k, v in st.items()))
        want = {s[0] for v in st.values() for s in v}
        lines = {}
        for sid, lat, lon, t, tc, wet, recent, cover in reports:
            if sid not in want or (sid, t) in self.seen:
                continue
            self.seen.add((sid, t))
            day = time.strftime("%Y%m%d", time.gmtime(t / 1000))
            lines.setdefault(day, []).append(json.dumps(
                {"id": sid, "t": t, "tc": tc, "w": wet, "re": recent, "n": cover}, separators=(",", ":")))
        for day, ls in lines.items():
            with open(os.path.join(self.dir, "obs", f"{day}.jsonl"), "a") as f:
                f.write("\n".join(ls) + "\n")
        if len(self.seen) > 100_000:
            cut = time.time() * 1000 - 12 * 3600_000
            self.seen = {k for k in self.seen if k[1] > cut}
        return sum(len(v) for v in lines.values())

    def obs_loop(self):
        while True:
            try:
                self.poll_obs()
            except Exception as e:
                self.log(f"models: METAR failed: {e}")
            time.sleep(600)

    # ---------- forecasts ----------
    def _ask(self, model, part):
        q = (f"{self.om}/v1/forecast?latitude={','.join(f'{s[1]:.2f}' for s in part)}"
             f"&longitude={','.join(f'{s[2]:.2f}' for s in part)}&hourly={','.join(VARS)}"
             f"&forecast_hours={HOURS}&timezone=auto&timeformat=unixtime&models={model}")
        d = json.loads(self.fetch(q, timeout=300))
        return d if isinstance(d, list) else [d]

    def log_issue(self, issue_ms):
        st = self.stations()
        for reg, stations in st.items():
            n = len(stations)
            out, t_ref, offs = {}, None, np.zeros(n, np.int32)
            for model in MODELS[reg]:
                idx = [i for i, s in enumerate(stations) if asked(model, s)]
                if not idx:
                    continue
                arr = {v: np.full((n, HOURS), np.nan, np.float32) for v in VARS}
                t0 = time.time()
                ok, errs = 0, []
                for c in range(0, len(idx), CHUNK):
                    part_i = idx[c:c + CHUNK]
                    if time.time() - t0 > MODEL_MAX_S:  # a slow remote model must not hold up the others
                        errs.append(f"stopped after {MODEL_MAX_S} s")
                        break
                    try:
                        res = self._ask(model, [stations[i] for i in part_i])
                    except Exception as e:  # leave the chunk empty
                        errs.append(str(e)[:80])
                        continue
                    for i, x in zip(part_i, res):
                        h = x.get("hourly") or {}
                        tt = h.get("time") or []
                        if not tt:
                            continue
                        if t_ref is None:
                            t_ref = int(tt[0])
                        k0 = (int(tt[0]) - t_ref) // 3600
                        offs[i] = x.get("utc_offset_seconds", 0)
                        for v in VARS:
                            vals = np.array([np.nan if y is None else y for y in h.get(v, [])], np.float32)
                            a, b = max(0, k0), min(HOURS, k0 + len(vals))
                            if b > a:
                                arr[v][i, a:b] = vals[a - k0:b - k0]
                        ok += 1
                out[model] = arr
                self.log(f"models: {reg} {model} {ok}/{len(idx)} stations in {time.time() - t0:.0f} s"
                         + (f" ({len(errs)} requests failed: {errs[0]})" if errs else ""))
            if t_ref is None:
                continue
            z = {"t0": np.int64(t_ref * 1000), "issue": np.int64(issue_ms), "ids": np.array([s[0] for s in stations]),
                 "offset": offs}
            for model, arr in out.items():
                z[f"{model}.temp"] = arr["temperature_2m"].astype(np.float16)
                z[f"{model}.precip"] = arr["precipitation"].astype(np.float16)
                z[f"{model}.cover"] = np.where(np.isnan(arr["cloud_cover"]), 255, arr["cloud_cover"]).astype(np.uint8)
                z[f"{model}.code"] = np.where(np.isnan(arr["weather_code"]), 255, arr["weather_code"]).astype(np.uint8)
            path = os.path.join(self.dir, "fc", f"{reg}_{issue_ms}.npz")
            np.savez_compressed(path + ".tmp.npz", **z)
            os.replace(path + ".tmp.npz", path)
        old = time.time() - KEEP_DAYS * 86400
        for sub in ("fc", "obs"):
            d = os.path.join(self.dir, sub)
            for fn in os.listdir(d):
                if os.path.getmtime(os.path.join(d, fn)) < old:
                    os.remove(os.path.join(d, fn))

    def fc_loop(self):
        while not self.stations():  # picked from the first 3 hours of METAR reports
            time.sleep(600)
        while True:
            now = time.time()
            nxt = min(((now // 86400) * 86400 + h * 3600 + 2400 + day * 86400)
                      for h in ISSUE_HOURS_UTC for day in (0, 1) if (now // 86400) * 86400 + h * 3600 + 2400 + day * 86400 > now)
            last = nxt - 6 * 3600
            # at start: catch up the issue time just passed (a restart must not lose it), otherwise wait for the next
            mark = os.path.join(self.dir, "fc", ".last")
            try:
                done = float(open(mark).read())
            except (OSError, ValueError):
                done = 0
            if done < last and now - last < 3 * 3600:
                target = last
            else:
                time.sleep(max(1, nxt - now))
                target = nxt
            try:
                self.log_issue(int(target // 3600 * 3600 * 1000))
                with open(mark, "w") as f:
                    f.write(str(target))
            except Exception as e:
                self.log(f"models: forecast log failed: {e}")

    def start(self):
        threading.Thread(target=self.obs_loop, daemon=True).start()
        threading.Thread(target=self.fc_loop, daemon=True).start()
