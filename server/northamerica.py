"""North America for the rain radar: NOAA MRMS (Multi-Radar Multi-Sensor), the US national radar mosaic, which also
takes in the Canadian radars near the border (14.11.2026: 84 % of 49-52 N covered, 42 % of 52-55 N).

Source: s3://noaa-mrms-pds (NOAA open data, no account), product CONUS/PrecipRate_00.00 - surface precipitation rate
(mm/h), 0.01 deg (~1 km), a new file every 2 minutes, ~2-5 min after the measurement. GRIB2, data packed as a 16-bit
PNG (data representation template 5.41): value = (reference + raw * 2^E) / 10^D; -3 = outside the radars' reach.
Read here without a GRIB library (the image has none): only the sections this one product needs.

The instance (compose service api-na, behind /na/, env NA=1) runs the Europe edition's whole pipeline - nowcast,
tiles, timeline, verification against its own later images, warm start - with this region in place of OPERA. MRMS
writes even minutes: the 5-minute step T takes the file of T, else T+1 min, else T-1 min (a one-minute shift does not
matter for 5-minute rain motion).
"""
import gzip
import io
import re
import struct
import time
import urllib.request
from datetime import datetime, timedelta, timezone

import numpy as np
from PIL import Image

BUCKET = "https://noaa-mrms-pds.s3.amazonaws.com"
PRODUCT = "CONUS/PrecipRate_00.00"
LAT1, LON0, DEG0 = 54.995, -129.995, 0.01   # centre of MRMS's first cell (north-west), cell size
NX0, NY0 = 7000, 3500


def decode(raw):
    """MRMS GRIB2 (PNG packed) -> float32 mm/h on the 0.01 deg grid, < 0 = no data."""
    if raw[:2] == b"\x1f\x8b":
        raw = gzip.decompress(raw)
    if raw[:4] != b"GRIB" or raw[7] != 2:
        raise ValueError("not GRIB2")
    i, ref, E, D, png, ni, nj = 16, None, 0, 0, None, None, None
    while i < len(raw) - 4 and raw[i:i + 4] != b"7777":
        ln, num = struct.unpack(">IB", raw[i:i + 5])
        sec = raw[i:i + ln]
        if num == 3:
            ni, nj = struct.unpack(">II", sec[30:38])
        elif num == 5:
            if struct.unpack(">H", sec[9:11])[0] != 41:
                raise ValueError("not PNG packed")
            ref = struct.unpack(">f", sec[11:15])[0]
            E, D = struct.unpack(">hh", sec[15:19])
        elif num == 7:
            png = sec[5:]
        i += ln
    if png is None or ref is None or (ni, nj) != (NX0, NY0):
        raise ValueError(f"unexpected MRMS file (grid {ni} x {nj})")
    a = np.asarray(Image.open(io.BytesIO(png))).astype(np.float32)
    return (ref + a * 2.0 ** E) / 10.0 ** D


class MrmsListing:
    """5-minute steps -> the URL of the MRMS file used for it."""

    def __init__(self, fetch):
        self.fetch = fetch

    def times(self):
        now = datetime.now(timezone.utc)
        days = [now] + ([now - timedelta(days=1)] if now.hour < 3 else [])
        files = {}
        for day in days:
            token = None
            while True:
                url = f"{BUCKET}/?list-type=2&prefix={PRODUCT}/{day:%Y%m%d}/&max-keys=1000" + (
                    f"&continuation-token={urllib.request.quote(token)}" if token else "")
                xml = self.fetch(url).decode()
                for key in re.findall(r"<Key>([^<]+\.grib2\.gz)</Key>", xml):
                    m = re.search(r"_(\d{8})-(\d{6})\.grib2\.gz$", key)
                    if m:
                        t = int(datetime.strptime(m.group(1) + m.group(2), "%Y%m%d%H%M%S").replace(tzinfo=timezone.utc).timestamp() * 1000)
                        files[t // 60_000 * 60_000] = f"{BUCKET}/{key}"
                token = re.search(r"<NextContinuationToken>([^<]+)</NextContinuationToken>", xml)
                if not token:
                    break
                token = token.group(1)
        out = {}
        step = 5 * 60_000
        for t in range(min(files, default=0) // step * step, max(files, default=0) + 1, step):
            for dt in (0, 60_000, -60_000):
                if t + dt in files:
                    out[t] = files[t + dt]
                    break
        return out


def block_mean(a, k, nodata):
    """k x k block mean of a (< 0 = no data), no data where most of the block has none."""
    h, w = a.shape[0] // k * k, a.shape[1] // k * k
    ok = a[:h, :w] >= 0
    cov = ok.reshape(h // k, k, w // k, k).mean(axis=(1, 3))
    tot = np.where(ok, a[:h, :w], 0).reshape(h // k, k, w // k, k).mean(axis=(1, 3))
    return np.where(cov >= 0.5, tot / np.maximum(cov, 1e-6), nodata).astype(np.float32)


def merc_to_lonlat(x, y):
    r = 6378137.0
    return np.degrees(np.asarray(x) / r), np.degrees(np.arctan(np.sinh(np.asarray(y) / r)))


if __name__ == "__main__":  # quick check: newest file, decode time, coverage
    get = lambda u: urllib.request.urlopen(urllib.request.Request(u, headers={"User-Agent": "Mozilla/5.0"}), timeout=60).read()
    t = MrmsListing(get).times()
    ts = max(t)
    t0 = time.time()
    v = decode(get(t[ts]))
    print(len(t), "steps, newest", datetime.fromtimestamp(ts / 1000, timezone.utc), f"decoded in {time.time() - t0:.1f}s,",
          f"covered {np.mean(v >= 0):.2f}, wet {np.mean(v >= 0.1):.3f}, max {v.max():.0f} mm/h")


# ---------- rain gauges: US airport weather reports (METAR) ----------
METAR_URL = "https://aviationweather.gov/data/cache/metars.cache.csv.gz"


def metar_rows(raw):
    """(station, lat, lon, end of the hour in ms, mm) for every ROUTINE report of an automated station with a
    precipitation sensor ("AO2") in North America. The hourly report's P group is the precipitation since the last
    routine report (hundredths of an inch); without it the sensor measured nothing ("PNO" = sensor out: skipped).
    The cache holds each station's newest report only: the server reads it every 10 minutes and keeps them."""
    import csv
    text = gzip.decompress(raw).decode("utf-8", "replace").splitlines()
    rows = list(csv.reader(text))
    if not rows or rows[0][0] != "raw_text":
        return
    col = {name: i for i, name in enumerate(rows[0]) if name not in ("sky_cover", "cloud_base_ft_agl")}
    for r in rows[1:]:
        try:
            raw_text, sid, obs = r[0], r[col["station_id"]], r[col["observation_time"]]
            lat, lon = float(r[col["latitude"]]), float(r[col["longitude"]])
        except (ValueError, IndexError, KeyError):
            continue
        if r[col["metar_type"]] != "METAR" or " AO2" not in raw_text or " PNO" in raw_text:
            continue
        if not (20 <= lat <= 56 and -130 <= lon <= -60):
            continue
        m = re.search(r" P(\d{4})\b", raw_text.split(" RMK ", 1)[1] if " RMK " in raw_text else "")
        mm = int(m.group(1)) * 0.254 if m else 0.0
        t = int(datetime.strptime(obs[:19], "%Y-%m-%dT%H:%M:%S").replace(tzinfo=timezone.utc).timestamp() * 1000)
        yield sid, lat, lon, t, mm


# ---------- rain gauges: Canada (ECCC SWOB, 16.11.2026) ----------
# Canadian METARs carry no precipitation amount. ECCC's surface weather observations (SWOB) do: the MSC GeoMet OGC API
# serves every station's hourly record (open, no account); only the federal networks are kept (MSC, NAV CANADA, DND -
# the provincial partner networks are "all rights reserved"). pcpn_amt_pst1hr (all precipitation, water equivalent),
# else rnfl_amt_pst1hr (rain); the hour ends at date_tm (full hour).
SWOB_URL = ("https://api.weather.gc.ca/collections/swob-realtime/items?f=json&limit=3000&datetime={t}"
            "&properties=msc_id-value,stn_id-value,date_tm-value,pcpn_amt_pst1hr,rnfl_amt_pst1hr,data_pvdr-value")
SWOB_PROVIDERS = {"MSC", "NAV CANADA", "DND"}


def swob_rows(raw):
    """(station "CA-<id>", lat, lon, end of the hour ms, mm) of one hour's SWOB records."""
    import json
    for f in json.loads(raw).get("features", []):
        p = f.get("properties", {})
        sid = p.get("msc_id-value") or p.get("stn_id-value")  # MSC's own stations have only the MSC id
        if p.get("data_pvdr-value") not in SWOB_PROVIDERS or not sid:
            continue
        mm = p.get("pcpn_amt_pst1hr")
        mm = p.get("rnfl_amt_pst1hr") if mm is None else mm
        try:
            lon, lat = (float(v) for v in f["geometry"]["coordinates"][:2])
            t = int(datetime.strptime(p["date_tm-value"][:19], "%Y-%m-%dT%H:%M:%S").replace(tzinfo=timezone.utc).timestamp() * 1000)
        except (KeyError, TypeError, ValueError):
            continue
        if mm is None or mm < 0 or t % 3600_000:
            continue
        yield "CA-" + sid, lat, lon, t, float(mm)


class MetarGauges:
    """Keeps the hourly precipitation of the US METAR stations: <log_dir>/metar/<YYYYMMDD>.jsonl (one line per
    station hour: id, end time ms, mm) and <log_dir>/metar/stations.json (id, lat, lon; order never changes).
    Since 16.11.2026 also Canada's federal stations (SWOB, ids "CA-...", country "ca"; poll_canada)."""

    def __init__(self, fetch, log_dir, log):
        import os
        self.fetch, self.log = fetch, log
        self.dir = os.path.join(log_dir, "metar")
        os.makedirs(self.dir, exist_ok=True)
        self.st_path = os.path.join(self.dir, "stations.json")
        self.seen = set()
        self.ca_done = set()   # hours of SWOB already read
        self.sky = MetarSky()   # cloud cover, the station truth of sunny for the Americas (sunny_na.py)

    def stations(self):
        import json
        try:
            with open(self.st_path) as f:
                return [tuple(s) for s in json.load(f)]
        except (OSError, ValueError):
            return []

    def poll(self):
        import json
        import os
        st = self.stations()
        known = {s[1] for s in st}
        new_st, lines = [], {}
        raw = self.fetch(METAR_URL, timeout=60)
        for sid, lat, lon, t, cover in metar_sky(raw):
            self.sky.add(sid, lat, lon, t, cover)
        for sid, lat, lon, t, mm in metar_rows(raw):
            if sid not in known:
                new_st.append(("us", sid, round(lat, 4), round(lon, 4)))
                known.add(sid)
            if (sid, t) in self.seen:
                continue
            self.seen.add((sid, t))
            day = datetime.fromtimestamp(t / 1000, timezone.utc).strftime("%Y%m%d")
            lines.setdefault(day, []).append(json.dumps({"id": sid, "t": t, "mm": round(mm, 2)}))
        if new_st:
            with open(self.st_path + ".tmp", "w") as f:
                json.dump(st + new_st, f)
            os.replace(self.st_path + ".tmp", self.st_path)
        for day, ls in lines.items():
            with open(os.path.join(self.dir, f"{day}.jsonl"), "a") as f:
                f.write("\n".join(ls) + "\n")
        if len(self.seen) > 200_000:
            cut = time.time() * 1000 - 6 * 3600_000
            self.seen = {k for k in self.seen if k[1] > cut}
        return sum(len(v) for v in lines.values()), len(new_st)

    def _append(self, rows, country):
        """rows (sid, lat, lon, t, mm) -> the station list and the day files; returns (new hours, new stations)."""
        import json
        import os
        st = self.stations()
        known = {s[1] for s in st}
        new_st, lines = [], {}
        for sid, lat, lon, t, mm in rows:
            if sid not in known:
                new_st.append((country, sid, round(lat, 4), round(lon, 4)))
                known.add(sid)
            if (sid, t) in self.seen:
                continue
            self.seen.add((sid, t))
            day = datetime.fromtimestamp(t / 1000, timezone.utc).strftime("%Y%m%d")
            lines.setdefault(day, []).append(json.dumps({"id": sid, "t": t, "mm": round(mm, 2)}))
        if new_st:
            with open(self.st_path + ".tmp", "w") as f:
                json.dump(st + new_st, f)
            os.replace(self.st_path + ".tmp", self.st_path)
        for day, ls in lines.items():
            with open(os.path.join(self.dir, f"{day}.jsonl"), "a") as f:
                f.write("\n".join(ls) + "\n")
        return sum(len(v) for v in lines.values()), len(new_st)

    def poll_canada(self):
        """The last three full hours of SWOB not read yet (records arrive up to ~30 min after the hour)."""
        now = int(time.time() * 1000)
        n = s = 0
        for back in (3, 2, 1):
            t = (now - 40 * 60_000) // 3600_000 * 3600_000 - (back - 1) * 3600_000
            if t in self.ca_done:
                continue
            stamp = datetime.fromtimestamp(t / 1000, timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")
            a, b = self._append(list(swob_rows(self.fetch(SWOB_URL.format(t=stamp), timeout=120))), "ca")
            self.ca_done.add(t)
            n, s = n + a, s + b
        self.ca_done = {t for t in self.ca_done if t > now - 6 * 3600_000}
        return n, s

    def loop(self):
        while True:
            try:
                n, s = self.poll()
                if n:
                    self.log(f"metar: {n} station hours, {s} new stations")
            except Exception as e:
                self.log(f"metar failed: {e}")
            try:
                n, s = self.poll_canada()
                if n:
                    self.log(f"swob (Canada): {n} station hours, {s} new stations")
            except Exception as e:
                self.log(f"swob failed: {e}")
            time.sleep(600)


# ---------- cloud cover for sunny: the METAR sky condition ----------
SKY_COVER = {"CLR": 0.0, "SKC": 0.0, "NCD": 0.0, "NSC": 0.0, "CAVOK": 0.0, "FEW": 0.19, "SCT": 0.44, "BKN": 0.75,
             "OVC": 1.0, "OVX": 1.0, "VV": 1.0}


def metar_sky(raw):
    """(station, lat, lon, observation time ms, cloud cover 0..1) of every report in North America (US and Canada): the
    largest of the reported layers (FEW 1-2/8, SCT 3-4/8, BKN 5-7/8, OVC 8/8). Caveat: automated stations look up
    to 12,000 ft only - "CLR" there can hide high cloud the satellite sees."""
    import csv
    rows = list(csv.reader(gzip.decompress(raw).decode("utf-8", "replace").splitlines()))
    if not rows or rows[0][0] != "raw_text":
        return
    head = rows[0]
    col = {name: i for i, name in enumerate(head) if name not in ("sky_cover", "cloud_base_ft_agl")}
    sky = [i for i, name in enumerate(head) if name == "sky_cover"]
    for r in rows[1:]:
        try:
            lat, lon = float(r[col["latitude"]]), float(r[col["longitude"]])
        except (ValueError, IndexError):
            continue
        if not (12 <= lat <= 60 and -135 <= lon <= -50):
            continue
        covers = [SKY_COVER[r[i]] for i in sky if i < len(r) and r[i] in SKY_COVER]
        if not covers:  # a clear sky is often only in the raw text ("CLR", "SKC", "CAVOK", "NCD", "NSC")
            text = " " + r[0].split(" RMK ")[0] + " "
            if not re.search(r" (CLR|SKC|CAVOK|NCD|NSC) ", text):
                continue
            covers = [0.0]
        t = int(datetime.strptime(r[col["observation_time"]][:19], "%Y-%m-%dT%H:%M:%S").replace(tzinfo=timezone.utc).timestamp() * 1000)
        yield r[col["station_id"]], lat, lon, t, max(covers)


class MetarSky:
    """The METAR cloud cover as sunny's station truth (the interface of sunny_de.SunStations): truth(end) = {station:
    1 - cover} of the last report in the 40 minutes before end - compared with the forecast sun share by day (a
    rough truth: a cover of 1 - share, not measured sunshine; sunny_de.SunVerify, score_sun.py --prefix na).
    Filled by MetarGauges.poll every 10 minutes; nothing of its own to run."""

    def __init__(self):
        import threading
        self.lock = threading.Lock()
        self.stations = {}   # id -> (lat, lon)
        self.obs = {}        # id -> {time ms: cover}

    def add(self, sid, lat, lon, t, cover):
        with self.lock:
            self.stations[sid] = (lat, lon)
            d = self.obs.setdefault(sid, {})
            d[t] = cover
            if len(d) > 40:
                for k in sorted(d)[:-30]:
                    del d[k]

    def truth(self, end_ts):
        out = {}
        with self.lock:
            for sid, d in self.obs.items():
                ts = [t for t in d if end_ts - 40 * 60_000 < t <= end_ts]
                if ts:
                    out[sid] = 1.0 - d[max(ts)]
        return out

    def newest(self):
        with self.lock:
            ends = [max(d) for d in self.obs.values() if d]
        return int(np.median(ends)) if ends else 0

    @property
    def values(self):  # SunnyDE.health counts the stations by it (as SunStations.values)
        return self.obs

    def loop(self):
        pass
