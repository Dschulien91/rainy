"""SYNOP weather reports of all of Europe as verification truth (16.11.2026; user: "in so ziemlich allen europaeischen
laendern ... auf augenhoehe mit den wettbewerbern, vor allem die 120 minuten vorhersage").

Source: DWD open data, weather/weather_reports/synoptic/international/ - the GTS SYNOP bulletins DWD receives, in BUFR,
a new file every ~5 minutes, the last ~24 h kept (no account). One hour of files holds ~7000 reports worldwide, ~2000
of them in Europe. Read with ecCodes (pip "eccodes", wheel with the library).

Per report we keep (verify_log/synop/<YYYYMMDD>.jsonl, one line per station and time; stations.json):
  n    total cloud cover 0..1 (observer / ceilometer; "sky obscured" = 1) -> the station truth of sunny in every
       European country (SynopFeed.sky, the interface of sunny_de.SunStations; the DWD / Austrian / Swiss sunshine
       stations stay the precise truth in those three countries)
  rr   [[period h, mm], ...] precipitation amounts with their period (1 h at most stations every hour, 6 / 12 h at the
       main hours) -> hourly rain gauges in countries without an open 10-minute network (France, Spain, Poland, the
       Benelux, the Balkans, Scandinavia, the Baltics ...): network "sy" of check_national_gauges.py
  ss   [[period h, minutes], ...] sunshine duration (1 h where reported) -> a later, sharper sunny truth
Ids are the WMO station numbers; country = the WMO block / index range (COUNTRIES).
"""
import json
import os
import re
import threading
import time
from datetime import datetime, timezone

BASE = "https://opendata.dwd.de/weather/weather_reports/synoptic/international/"
BOX = (34.0, 72.0, -25.0, 45.0)   # lat min, lat max, lon min, lon max: Europe as rainy / sunny show it
KEEP_DAYS = 30
# WMO index ranges -> country (or region) code; coarse where WMO blocks mix small countries
COUNTRIES = [(1001, 1499, "no"), (2000, 2699, "se"), (2700, 2999, "fi"), (3001, 3949, "gb"), (3950, 3999, "ie"),
             (4000, 4199, "is"), (6000, 6019, "fo"), (6020, 6199, "dk"), (6200, 6399, "nl"), (6400, 6499, "be"),
             (6580, 6599, "lu"), (6600, 6799, "ch"), (7000, 7999, "fr"), (8000, 8499, "es"), (8500, 8599, "pt"),
             (10000, 10999, "de"), (11000, 11399, "at"), (11400, 11799, "cz"), (11800, 11999, "sk"),
             (12000, 12699, "pl"), (12700, 12999, "hu"), (13000, 13999, "rs"), (14000, 14099, "si"),
             (14100, 14499, "hr"), (14500, 14699, "ba"), (15000, 15499, "ro"), (15500, 15999, "bg"),
             (16000, 16559, "it"), (16560, 16599, "mt"), (16600, 16799, "gr"), (17000, 17599, "tr"),
             (17600, 17699, "cy"), (26000, 26299, "ee"), (26300, 26499, "lv"), (26500, 26699, "lt"),
             (26700, 26999, "by"), (33000, 33299, "by"), (33300, 33699, "ua"), (33700, 33899, "md"),
             (33900, 34999, "ua")]
NAMES = {"no": "Norway", "se": "Sweden", "fi": "Finland", "gb": "United Kingdom", "ie": "Ireland", "is": "Iceland",
         "fo": "Faroe Islands", "dk": "Denmark", "nl": "Netherlands", "be": "Belgium", "lu": "Luxembourg",
         "ch": "Switzerland", "fr": "France", "es": "Spain", "pt": "Portugal", "de": "Germany", "at": "Austria",
         "cz": "Czechia", "sk": "Slovakia", "pl": "Poland", "hu": "Hungary", "rs": "Serbia / Montenegro / N. Macedonia",
         "si": "Slovenia", "hr": "Croatia", "ba": "Bosnia and Herzegovina", "ro": "Romania", "bg": "Bulgaria",
         "it": "Italy", "mt": "Malta", "gr": "Greece", "tr": "Turkey", "cy": "Cyprus", "ee": "Estonia", "lv": "Latvia",
         "lt": "Lithuania", "by": "Belarus", "ua": "Ukraine", "md": "Moldova"}
WANT = {"blockNumber", "stationNumber", "latitude", "longitude", "year", "month", "day", "hour", "minute",
        "cloudCoverTotal", "timePeriod", "totalPrecipitationOrTotalWaterEquivalent", "totalSunshine"}


def country(wmo):
    for lo, hi, c in COUNTRIES:
        if lo <= wmo <= hi:
            return c
    return None


def _ok(v):
    return v is not None and abs(v) < 1e9 and v != 2147483647


def messages(data):
    """The BUFR messages of a GTS bulletin file (WMO headers between them)."""
    for m in re.finditer(rb"BUFR", data):
        s = m.start()
        n = int.from_bytes(data[s + 4:s + 7], "big")
        msg = data[s:s + n]
        if len(msg) == n and msg[-4:] == b"7777":
            yield msg


def decode(data):
    """[(wmo, lat, lon, time ms, cover 0..1 or None, rr [[h, mm]], ss [[h, min]])] of one bulletin file."""
    import eccodes
    out = []
    for msg in messages(data):
        try:
            h = eccodes.codes_new_from_message(msg)
        except Exception:
            continue
        try:
            eccodes.codes_set(h, "unpack", 1)
            n = eccodes.codes_get(h, "numberOfSubsets")
            comp = eccodes.codes_get(h, "compressedData") == 1
            subs = [{"rr": [], "ss": []} for _ in range(n)]
            period = [None] * n
            cur = 0
            it = eccodes.codes_bufr_keys_iterator_new(h)
            while eccodes.codes_bufr_keys_iterator_next(it):
                key = eccodes.codes_bufr_keys_iterator_get_name(it)
                base = key.split("#")[-1]
                if base not in WANT:
                    continue
                try:
                    vals = list(eccodes.codes_get_array(h, key))
                except Exception:
                    continue
                if not comp:  # uncompressed: the keys run subset by subset, ranked across the message
                    if base == "blockNumber":
                        m = re.match(r"#(\d+)#", key)
                        cur = min(n - 1, int(m.group(1)) - 1) if m else 0
                    targets = [(cur, vals[0] if vals else None)]
                else:
                    targets = [(i, vals[i] if len(vals) == n else vals[0]) for i in range(n)] if vals else []
                for i, v in targets:
                    s = subs[i]
                    if base == "timePeriod":
                        period[i] = v if _ok(v) else None
                    elif base == "totalPrecipitationOrTotalWaterEquivalent":
                        if _ok(v) and period[i] is not None:
                            s["rr"].append([int(period[i]), round(max(0.0, float(v)), 2)])
                    elif base == "totalSunshine":
                        if _ok(v) and period[i] is not None:
                            s["ss"].append([int(period[i]), int(v)])
                    elif base not in s:  # the first occurrence (the station header)
                        s[base] = v
            eccodes.codes_bufr_keys_iterator_delete(it)
        except Exception:
            continue
        finally:
            eccodes.codes_release(h)
        for s in subs:
            try:
                if not all(_ok(s.get(k)) for k in ("blockNumber", "stationNumber", "latitude", "longitude", "year",
                                                    "month", "day", "hour")):
                    continue
                wmo = int(s["blockNumber"]) * 1000 + int(s["stationNumber"])
                t = datetime(int(s["year"]), int(s["month"]), int(s["day"]), int(s["hour"]),
                             int(s["minute"]) if _ok(s.get("minute")) else 0, tzinfo=timezone.utc)
                n_ = s.get("cloudCoverTotal")
                cover = (1.0 if n_ > 100 else n_ / 100.0) if _ok(n_) else None
                out.append((wmo, float(s["latitude"]), float(s["longitude"]), int(t.timestamp() * 1000), cover,
                            s["rr"], s["ss"]))
            except (ValueError, TypeError, OverflowError):
                continue
    return out


class SynopFeed:
    """Reads the new bulletin files every 5 minutes (the first pass reads the ~24 h DWD keeps) and keeps the European
    reports on disk; .sky = the cloud cover as sunny's station truth (northamerica.MetarSky)."""

    def __init__(self, fetch, log, log_dir):
        import northamerica
        self.fetch, self.log = fetch, log
        self.dir = os.path.join(log_dir, "synop")
        os.makedirs(self.dir, exist_ok=True)
        self.st_path = os.path.join(self.dir, "stations.json")
        self.read = set()
        self.sky = northamerica.MetarSky()
        self.lock = threading.Lock()

    def stations(self):
        """{wmo as str: [lat, lon, country, has hourly precipitation]}"""
        try:
            with open(self.st_path) as f:
                return json.load(f)
        except (OSError, ValueError):
            return {}

    def poll(self):
        text = self.fetch(BASE, timeout=60).decode("latin-1")
        names = sorted(set(re.findall(r'href="(Z__C_EDZW_\d{14}[^"]+\.bin)"', text)))
        todo = [n for n in names if n not in self.read]
        if not todo:
            return 0
        st = self.stations()
        lines, n_rep = {}, 0
        for name in todo:
            try:
                data = self.fetch(BASE + name, timeout=60)
            except Exception:
                continue
            self.read.add(name)
            for wmo, lat, lon, t, cover, rr, ss in decode(data):
                c = country(wmo)
                if c is None or not (BOX[0] <= lat <= BOX[1] and BOX[2] <= lon <= BOX[3]):
                    continue
                sid = str(wmo)
                hourly = any(p == -1 for p, _ in rr)
                if sid not in st:
                    st[sid] = [round(lat, 4), round(lon, 4), c, hourly]
                elif hourly and not st[sid][3]:
                    st[sid][3] = True
                if cover is not None:
                    self.sky.add(sid, lat, lon, t, cover)
                day = time.strftime("%Y%m%d", time.gmtime(t / 1000))
                lines.setdefault(day, []).append(json.dumps({"id": sid, "t": t, "n": cover, "rr": rr, "ss": ss}))
                n_rep += 1
        with self.lock:
            tmp = self.st_path + ".tmp"
            with open(tmp, "w") as f:
                json.dump(st, f)
            os.replace(tmp, self.st_path)
            for day, ls in lines.items():
                with open(os.path.join(self.dir, f"{day}.jsonl"), "a") as f:
                    f.write("\n".join(ls) + "\n")
        self.read = {n for n in self.read if n in set(names)}
        old = time.time() - KEEP_DAYS * 86400
        for fp in os.listdir(self.dir):
            p = os.path.join(self.dir, fp)
            if fp.endswith(".jsonl") and os.path.getmtime(p) < old:
                os.remove(p)
        return n_rep

    def loop(self):
        while True:
            t0 = time.time()
            try:
                n = self.poll()
                if n:
                    self.log(f"synop: {n} European reports in {time.time() - t0:.0f} s")
            except Exception as e:
                self.log(f"synop failed: {e}")
            time.sleep(300)


def hourly_rain(log_dir, ids, since, until):
    """{wmo: {end time ms: mm in the hour}} of the stations' 1-hour precipitation amounts (a station may send a
    report twice - the later line wins; equal anyway)."""
    want, out = set(ids), {}
    d = os.path.join(log_dir, "synop")
    for day in range(int(since // 86400_000), int(until // 86400_000) + 1):
        fp = os.path.join(d, time.strftime("%Y%m%d", time.gmtime(day * 86400)) + ".jsonl")
        try:
            f = open(fp)
        except OSError:
            continue
        with f:
            for line in f:
                try:
                    r = json.loads(line)
                except ValueError:
                    continue
                if r["id"] in want and since <= r["t"] <= until:
                    for p, mm in r.get("rr", []):
                        if p == -1:
                            out.setdefault(r["id"], {})[r["t"]] = mm
    return out
