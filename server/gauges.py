"""Gauge-nudged drizzle: light rain that the DWD gauges measure but the radar misses.

Shallow drizzle (low stratus, upslope drizzle in the Erzgebirge) forms below the lowest radar beam. On 30.10. the
Dresden and Erzgebirge gauges measured 0.2-0.8 mm/h for hours while the radar - and RADOLAN RW, which the DWD adjusts
to the same gauges - showed nothing. WetterOnline showed it.

Source: the DWD 10-minute gauge files (open data, ~1370 stations, ~0.5 KB each, 20-30 min behind). A station counts
as wet when its last two 10-min values are both > 0 and together >= 0.07 mm (~0.2 mm/h): one bucket tip, dew or fog
(0.01 mm) never qualifies.
Each value must also carry the gauge's own "precipitation yes" indicator and a duration; faulty gauges report
amounts without them. Backtest 30.10. 00-07 UTC: only Saxon stations qualified (1-5 at a time), exactly where
WetterOnline showed drizzle; none of the trace stations in the north. Where the radar is already wet within
NEAR_KM, the radar wins. The drizzle is painted as a small disc around the gauge (no motion: this kind of drizzle
stays put) and fades out over the first FADE_MIN of the forecast.
"""
import io
import threading
import time
import zipfile
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime, timezone

import numpy as np

CDC = "https://opendata.dwd.de/climate_environment/CDC/observations_germany/climate/10_minutes/precipitation/now/"
# Polling: ~1350 files from the same DWD host as the radar. Single files now and then take 15 s, and many at once
# got connections reset. So a slow sweep over all stations (2 at a time, about once an hour) finds new drizzle, and
# a second loop re-reads only the damp stations every 10 min.
EVERY_S = 600
SWEEP_EVERY_S = 3600
WORKERS = 2
MAX_AGE_MIN = 70     # newest value older than this (station late or broken): ignore the station
MIN_SUM_MM = 0.07    # last two 10-min values together (~0.2 mm/h)
RADIUS_KM = 6.0      # drizzle disc around a wet gauge
NEAR_KM = 5.0        # radar wet (>= RADAR_WET mm/h) this close to the gauge: the radar already shows it
RADAR_WET = 0.1
MAX_MMH = 1.0        # drizzle, not a downpour: cap what one gauge can paint
MAX_10MIN_MM = 0.3   # more than this in 10 min with a dry radar is a gauge fault, not drizzle (Moos 30.10.: 7 mm/h)
MAX_HEIGHT_M = 1000  # summit gauges (Fichtelberg, Brocken) collect fog inside the cloud: not drizzle around them
FADE_MIN = 60


class GaugeDrizzle:
    def __init__(self, fetch, log, de, merc):
        self.fetch, self.log, self.de, self.merc = fetch, log, de, merc
        self.lock = threading.Lock()
        self.stations = []   # (id, lat, lon)
        self.latest = {}     # id -> (end time of the newest value in ms, [previous, newest] mm)
        self.pool = ThreadPoolExecutor(WORKERS)

    def _station_list(self):
        text = self.fetch(f"{CDC}zehn_now_rr_Beschreibung_Stationen.txt", timeout=60).decode("latin-1")
        recent = time.strftime("%Y%m%d", time.gmtime(time.time() - 3 * 86400))
        out = []
        for line in text.splitlines()[2:]:
            p = line.split()
            if len(p) >= 6 and p[0].isdigit() and p[2] >= recent and float(p[3]) <= MAX_HEIGHT_M:
                out.append((p[0], float(p[4]), float(p[5])))
        return out

    def _last_two(self, sid):
        """(end time of the newest value in ms, [previous, newest] mm) or None."""
        try:
            z = zipfile.ZipFile(io.BytesIO(self.fetch(f"{CDC}10minutenwerte_nieder_{sid}_now.zip", timeout=30, tries=1)))
            rows = z.read(z.namelist()[0]).decode("latin-1").splitlines()
        except Exception:
            return None
        head = [h.strip() for h in rows[0].split(";")]
        k, dur, ind = head.index("RWS_10"), head.index("RWS_DAU_10"), head.index("RWS_IND_10")
        vals = []
        for r in rows[-2:]:
            p = r.split(";")
            mm = float(p[k])
            if mm < 0 or (mm > 0 and (float(p[dur]) <= 0 or int(p[ind]) != 1)):  # missing, or no "it rained" flag
                return None
            vals.append((p[1].strip(), mm))
        if len(vals) < 2:
            return None
        ts = int(datetime.strptime(vals[-1][0], "%Y%m%d%H%M").replace(tzinfo=timezone.utc).timestamp() * 1000)
        return ts, [v for _, v in vals]

    def _read(self, sids):
        got = dict(zip(sids, self.pool.map(self._last_two, sids)))
        with self.lock:
            for sid, v in got.items():
                if v is None:
                    self.latest.pop(sid, None)
                else:
                    self.latest[sid] = v

    def damp(self):
        """Stations with any precipitation in their last two values."""
        with self.lock:
            return [sid for sid, (_, v) in self.latest.items() if max(v) > 0]

    def wet(self):
        """[(id, lat, lon, mm/h)] of the gauges that measure drizzle now."""
        now = time.time() * 1000
        where = {sid: (lat, lon) for sid, lat, lon in self.stations}
        out = []
        with self.lock:
            items = list(self.latest.items())
        for sid, (ts, (a, b)) in items:
            if sid in where and now - ts <= MAX_AGE_MIN * 60_000 and a > 0 and b > 0 and a + b >= MIN_SUM_MM and max(a, b) <= MAX_10MIN_MM:
                out.append((sid, *where[sid], min((a + b) * 3.0, MAX_MMH)))
        return out

    def apply(self, grids, ref):
        """Add the gauge drizzle to the radar analysis (ref) and the forecast frames (faded), in place."""
        wet = self.wet()
        if not wet or ref not in grids:
            return []
        km = self.de.KM_PER_PX
        near, rad = int(round(NEAR_KM / km)), int(round(RADIUS_KM / km))
        analysis = grids[ref]
        used = []
        for sid, lat, lon, mmh in wet:
            col, row = (int(round(float(v))) for v in self.de.to_pixels(*self.merc(lon, lat)))
            r0, r1, c0, c1 = max(row - rad, 0), min(row + rad + 1, analysis.shape[0]), max(col - rad, 0), min(col + rad + 1, analysis.shape[1])
            if r0 >= r1 or c0 >= c1:
                continue
            if float(analysis[max(row - near, 0):row + near + 1, max(col - near, 0):col + near + 1].max()) >= RADAR_WET:
                continue  # the radar sees it
            yy, xx = np.mgrid[r0:r1, c0:c1]
            d = np.hypot(yy - row, xx - col) / rad
            # flat top, soft edge: reads as a small drizzle area, not a dot
            disc = (mmh * np.clip((1.0 - d) / 0.4, 0.0, 1.0)).astype(np.float32)
            for ts, g in grids.items():
                if ts < ref:
                    continue
                f = 1.0 if ts == ref else max(0.0, 1.0 - (ts - ref) / (FADE_MIN * 60_000))
                if f > 0:
                    g[r0:r1, c0:c1] = np.maximum(g[r0:r1, c0:c1], (disc * f).astype(g.dtype))
            used.append(sid)
        return used

    def _say(self, what, started):
        wet = self.wet()
        self.log(f"[{datetime.now():%H:%M:%S}] gauges: {what} in {time.time() - started:.0f}s, {len(self.damp())} damp, "
                 f"drizzle at {', '.join(s for s, *_ in wet) or '-'}")

    def sweep_loop(self):
        while True:
            started = time.time()
            try:
                self.stations = self._station_list()
                self._read([s[0] for s in self.stations])
                self._say(f"all {len(self.stations)} stations", started)
            except Exception as e:
                self.log(f"gauges: {e}")
            time.sleep(max(60.0, SWEEP_EVERY_S - (time.time() - started)))

    def loop(self):
        threading.Thread(target=self.sweep_loop, daemon=True).start()
        while True:
            time.sleep(EVERY_S)
            started = time.time()
            try:
                damp = self.damp()
                if damp:
                    self._read(damp)
                    self._say(f"{len(damp)} damp stations", started)
            except Exception as e:
                self.log(f"gauges: {e}")
