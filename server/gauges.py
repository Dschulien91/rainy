"""Gauge-nudged drizzle: light rain that the DWD gauges measure but the radar misses.

Shallow drizzle (low stratus, upslope drizzle in the Erzgebirge) forms below the lowest radar beam. On 30.10. the
Dresden and Erzgebirge gauges measured 0.2-0.8 mm/h for hours while the radar - and RADOLAN RW, which the DWD adjusts
to the same gauges - showed nothing. WetterOnline showed it.

Source: the DWD 10-minute gauge files (open data, ~1370 stations, ~0.5 KB each, 20-30 min behind). A station counts
as wet when its last two 10-min values are both > 0 and together >= 0.07 mm (~0.2 mm/h): one bucket tip, dew or fog
(0.01 mm) never qualifies.
A value must not contradict the gauge's own "precipitation" indicator or duration, where the station has them.
Backtest 30.10. 00-07 UTC: only Saxon stations qualified (1-5 at a time), exactly where
WetterOnline showed drizzle; none of the trace stations in the north. Where the radar is already wet within
NEAR_KM, the radar wins. The drizzle is painted as a soft, ragged light patch around the gauge (no motion: this kind of drizzle
stays put) and fades out over the first FADE_MIN of the forecast.

The other way round (veto): weak radar echoes that do not reach the ground. Radar beams over the Alps or under dry
low-level air see precipitation aloft that evaporates on the way down (Berchtesgaden/Hallein 04.11.: 0.1-0.3 mm/h on
the radar for 1.5 h, all 16 gauges there 0.0; MV and Altmark bands 03.11.: 0 of 1348 gauges damp). WetterOnline
does not show these. Weak echoes (< VETO_MAX mm/h) are removed where at least VETO_MIN_DRY gauges within VETO_KM
measured exactly 0 in both of their last values and no gauge there is damp - but only where the radar already showed
the echo during the time those values cover. The gauges report 20-110 min late, so without this a front that has just
arrived was erased (06.11.: Black Forest edge, gauge 00757 wet, map dry): they cannot know about rain after their last value.
"""
import io
import json
import os
import re
import threading
import time
import zipfile
from concurrent.futures import ThreadPoolExecutor, TimeoutError, as_completed
from datetime import datetime, timezone

import cv2
import numpy as np

CDC = "https://opendata.dwd.de/climate_environment/CDC/observations_germany/climate/10_minutes/precipitation/now/"
# Polling: ~1350 files from the same DWD host as the radar. Fixed sweeps every 15 min (~9 requests/s) ended in
# "connection refused" (05.11.: 1133 stations unanswered, 7 s per file). DWD rewrites all files together, so now one
# listing request every LIST_EVERY_S shows whether anything changed, and only changed files are read: at most
# PACE_S apart per worker, stopping after MAX_ERRORS failures in a row - the radar from that host matters more.
FAILED = object()      # _last_two: the download failed (unlike None: the file holds no usable values)
LIST_EVERY_S = 300
WORKERS = 2
PACE_S = 0.7           # ~3 requests/s, a full round in ~8 min
READ_DEADLINE_S = 1200
MAX_ERRORS = 10
MAX_AGE_MIN = 70     # newest value older than this (station late or broken): ignore the station
MIN_SUM_MM = 0.07    # last two 10-min values together (~0.2 mm/h)
RADIUS_KM = 6.0      # drizzle disc around a wet gauge
NEAR_KM = 5.0        # radar wet (>= RADAR_WET mm/h) this close to the gauge: the radar already shows it
RADAR_WET = 0.1
MAX_MMH = 1.0        # drizzle, not a downpour: cap what one gauge can paint
SOFT_MAX_MMH = 0.25  # drawn as the lightest classes (a veil, not a solid patch)
MAX_10MIN_MM = 0.3   # more than this in 10 min with a dry radar is a gauge fault, not drizzle (Moos 30.10.: 7 mm/h)
MAX_HEIGHT_M = 1000  # summit gauges (Fichtelberg, Brocken) collect fog inside the cloud: not drizzle around them
FADE_MIN = 60
VETO_KM = 15.0
VETO_MIN_DRY = 2
VETO_MAX = 0.5       # mm/h: only weak echoes; real showers are stronger and the gauges would show them soon
VETO_HISTORY_MS = 3 * 3600_000  # radar wet masks kept to compare with the gauges' time
VETO_MAX_AGE_MIN = 150  # the "now" files are 20-110 min behind (04.11. morning: 105 min); dry air changes slowly


class GaugeDrizzle:
    def __init__(self, fetch, log, de, merc, state_dir=None):
        self.fetch, self.log, self.de, self.merc = fetch, log, de, merc
        self.lock = threading.Lock()
        self.stations = []   # (id, lat, lon)
        self.errors = 0
        self.radar_wet = {}  # ts -> packed mask where the radar showed rain (slightly widened)
        self.latest = {}     # id -> (end time of the newest value in ms, [previous, newest] mm)
        # state_dir: the gauge values and the radar wet masks survive a restart there. Without them the first
        # ~10 min (gauges re-read) up to ~2 h (radar history for the veto) after a restart had no corrections.
        self.state_dir = state_dir
        self._load_state()

    def _load_state(self):
        if not self.state_dir:
            return
        try:
            with open(os.path.join(self.state_dir, "gauges_latest.json")) as f:
                self.latest = {sid: (int(ts), list(v)) for sid, (ts, v) in json.load(f).items()}
            with open(os.path.join(self.state_dir, "gauges_stations.json")) as f:
                self.stations = [tuple(st) for st in json.load(f)]
        except (OSError, ValueError):
            pass
        wet_dir = os.path.join(self.state_dir, "radar_wet")
        cutoff = time.time() * 1000 - VETO_HISTORY_MS
        for name in os.listdir(wet_dir) if os.path.isdir(wet_dir) else []:
            try:
                ts = int(name.split(".")[0])
                if ts >= cutoff:
                    self.radar_wet[ts] = np.load(os.path.join(wet_dir, name))
            except (OSError, ValueError):
                pass

    def _save_latest(self):
        if not self.state_dir:
            return
        try:
            os.makedirs(self.state_dir, exist_ok=True)
            with self.lock:
                data = dict(self.latest)
            tmp = os.path.join(self.state_dir, "gauges_latest.json.tmp")
            with open(tmp, "w") as f:
                json.dump(data, f)
            os.replace(tmp, os.path.join(self.state_dir, "gauges_latest.json"))
            with open(os.path.join(self.state_dir, "gauges_stations.json"), "w") as f:
                json.dump(self.stations, f)
        except OSError as e:
            self.log(f"gauges: state not saved ({e})")

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
        """(end time of the newest value in ms, [previous, newest] mm), None (no usable values) or FAILED (network)."""
        if self.errors >= MAX_ERRORS:
            return FAILED
        time.sleep(PACE_S)
        try:
            z = zipfile.ZipFile(io.BytesIO(self.fetch(f"{CDC}10minutenwerte_nieder_{sid}_now.zip", timeout=30, tries=1)))
            rows = z.read(z.namelist()[0]).decode("latin-1").splitlines()
            self.errors = 0
        except Exception:
            self.errors += 1
            return FAILED
        head = [h.strip() for h in rows[0].split(";")]
        k, dur, ind = head.index("RWS_10"), head.index("RWS_DAU_10"), head.index("RWS_IND_10")
        vals = []
        for r in rows[1:][-2:]:  # a file with a single value (new station, new day) must not hand on its header
            p = r.split(";")
            try:
                mm = float(p[k])
            except (ValueError, IndexError):
                return None
            # missing value, or the gauge itself says "no rain"; -999 = this station has no such sensor (many have none,
            # e.g. Zinnwald, Moos): its amount counts, MAX_10MIN_MM catches faults. Dropping them hid real rain from veto()
            if mm < 0 or (mm > 0 and (float(p[dur]) == 0 or int(p[ind]) == 0)):
                return None
            vals.append((p[1].strip(), mm))
        if len(vals) < 2:
            return None
        ts = int(datetime.strptime(vals[-1][0], "%Y%m%d%H%M").replace(tzinfo=timezone.utc).timestamp() * 1000)
        return ts, [v for _, v in vals]

    def _read(self, sids):
        """Read these stations; returns those that gave an answer. A failed one keeps its last values and is retried."""
        # a fresh pool per read with an overall deadline: on 04.11. one download hung for hours (no answer, no
        # timeout) and blocked every later sweep; now a stuck download is left behind and the next round runs
        self.errors = 0
        pool = ThreadPoolExecutor(WORKERS)
        futs = {pool.submit(self._last_two, sid): sid for sid in sids}
        got = {}
        try:
            for f in as_completed(futs, timeout=READ_DEADLINE_S):
                got[futs[f]] = f.result()
        except TimeoutError:
            pass
        pool.shutdown(wait=False, cancel_futures=True)
        got = {sid: v for sid, v in got.items() if v is not FAILED}
        with self.lock:
            for sid, v in got.items():
                if v is None:
                    self.latest.pop(sid, None)
                else:
                    self.latest[sid] = v
        return set(got)

    def _changed(self):
        """{station id: time stamp in the DWD listing} of our stations."""
        text = self.fetch(CDC, timeout=60).decode("latin-1")
        ids = {s[0] for s in self.stations}
        return {sid: stamp for sid, stamp in re.findall(r'nieder_(\d{5})_now\.zip</a>\s+(\S+ \S+)', text) if sid in ids}

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

    def _remember(self, ts, grid):
        wet = cv2.dilate((np.asarray(grid, np.float32) >= 0.05).astype(np.uint8), np.ones((7, 7), np.uint8))
        self.radar_wet[ts] = np.packbits(wet)
        wet_dir = os.path.join(self.state_dir, "radar_wet") if self.state_dir else None
        if wet_dir:
            try:
                os.makedirs(wet_dir, exist_ok=True)
                np.save(os.path.join(wet_dir, f"{ts}.npy"), self.radar_wet[ts])
            except OSError:
                pass
        for t in [t for t in self.radar_wet if t < ts - VETO_HISTORY_MS]:
            del self.radar_wet[t]
            if wet_dir:
                try:
                    os.remove(os.path.join(wet_dir, f"{t}.npy"))
                except OSError:
                    pass

    def veto(self, grids, ref, raw, motion=None):
        """Remove weak echoes over dry gauges from every frame (past, analysis, forecast), in place.
        raw: the measured frames as the radar sent them (the radar history the gauges are compared with).
        motion: (dx, dy) radar pixels per minute: the vetoed area moves with the rain in the forecast frames. Before
        (10.11.) it stayed where it was, the forecast's rain moved through it and the map showed lanes cut through
        the rain bands ("roads"); ~8 % of the wet area is vetoed. Without motion: no veto beyond the analysis.
        Returns the number of pixels removed from the analysis (ref)."""
        if ref not in grids:
            return 0
        for t, g in raw.items():
            if t not in self.radar_wet or t == ref:
                self._remember(t, g)
        now = time.time() * 1000
        where = {sid: (lat, lon) for sid, lat, lon in self.stations}
        with self.lock:
            items = list(self.latest.items())
        dry = np.zeros(grids[ref].shape, np.float32)
        damp = np.zeros(grids[ref].shape, np.uint8)
        for sid, (ts, vals) in items:
            if sid not in where or now - ts > VETO_MAX_AGE_MIN * 60_000:
                continue
            col, row = (int(round(float(v))) for v in self.de.to_pixels(*self.merc(where[sid][1], where[sid][0])))
            if 0 <= row < dry.shape[0] and 0 <= col < dry.shape[1]:
                if max(vals) > 0:
                    damp[row, col] = 1
                else:
                    dry[row, col] += 1
        fresh = [ts for sid, (ts, _) in items if sid in where and now - ts <= VETO_MAX_AGE_MIN * 60_000]
        if not dry.any() or not fresh:
            return 0
        # the radar during the gauges' last interval; unknown (gauges too old, just started): no veto
        seen = self.radar_wet.get(int(np.median(fresh)) // 300_000 * 300_000 - 600_000)
        if seen is None:
            return 0
        was_wet = np.unpackbits(seen, count=dry.size).reshape(dry.shape).astype(bool)
        r = int(round(VETO_KM / self.de.KM_PER_PX))
        disc = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (2 * r + 1, 2 * r + 1))
        n_dry = cv2.filter2D(dry, -1, disc.astype(np.float32), borderType=cv2.BORDER_CONSTANT)
        any_damp = cv2.dilate(damp, disc) > 0
        area = (n_dry >= VETO_MIN_DRY - 0.5) & ~any_damp & was_wet
        removed = 0
        for ts, g in grids.items():
            a = area
            if ts > ref:  # forecast: the area travels with the rain
                if motion is None:
                    continue
                lead = (ts - ref) / 60_000
                shift = np.float32([[1, 0, motion[0] * lead], [0, 1, motion[1] * lead]])
                a = cv2.warpAffine(area.astype(np.uint8), shift, (area.shape[1], area.shape[0]), flags=cv2.INTER_NEAREST) > 0
            weak = a & (np.asarray(g, np.float32) < VETO_MAX)
            if ts == ref:
                removed = int((weak & (np.asarray(g, np.float32) > 0)).sum())
            g[weak] = 0
        return removed

    def apply(self, grids, ref):
        """Add the gauge drizzle to the measured frames and the analysis (ref) at full strength and to the forecast
        frames (faded), in place. Whether the radar already sees it is decided on the analysis, for all frames alike."""
        wet = self.wet()
        if not wet or ref not in grids:
            return []
        km = self.de.KM_PER_PX
        near, rad = int(round(NEAR_KM / km)), int(round(RADIUS_KM / km))
        rad_soft = int(round(RADIUS_KM * 1.4 / km))  # the soft patch's tail reaches beyond RADIUS_KM
        analysis = grids[ref]
        used = []
        for sid, lat, lon, mmh in wet:
            col, row = (int(round(float(v))) for v in self.de.to_pixels(*self.merc(lon, lat)))
            r0, r1, c0, c1 = max(row - rad, 0), min(row + rad + 1, analysis.shape[0]), max(col - rad, 0), min(col + rad + 1, analysis.shape[1])
            if r0 >= r1 or c0 >= c1:
                continue
            if float(analysis[max(row - near, 0):row + near + 1, max(col - near, 0):col + near + 1].max()) >= RADAR_WET:
                continue  # the radar sees it
            # a soft, ragged light patch, not a flat disc: perfect circles looked artificial on the map (10.11., variant B
            # of three shown to the user). The outline comes from smooth noise seeded by the station, so it stays the
            # same from frame to frame and run to run.
            r0, r1 = max(row - rad_soft, 0), min(row + rad_soft + 1, analysis.shape[0])
            c0, c1 = max(col - rad_soft, 0), min(col + rad_soft + 1, analysis.shape[1])
            yy, xx = np.mgrid[r0:r1, c0:c1]
            d = np.hypot(yy - row, xx - col) / rad
            noise = cv2.GaussianBlur(np.random.default_rng(int(sid) if sid.isdigit() else 0).standard_normal(
                (2 * rad_soft + 1, 2 * rad_soft + 1)).astype(np.float32), (0, 0), 2.5)
            noise = (noise / (np.abs(noise).max() + 1e-6))[r0 - row + rad_soft:r1 - row + rad_soft, c0 - col + rad_soft:c1 - col + rad_soft]
            disc = (min(mmh, SOFT_MAX_MMH) * np.exp(-2.2 * (d * (1.0 + 0.45 * noise)) ** 2)).astype(np.float32)
            for ts, g in grids.items():
                f = 1.0 if ts <= ref else max(0.0, 1.0 - (ts - ref) / (FADE_MIN * 60_000))
                if f > 0:
                    g[r0:r1, c0:c1] = np.maximum(g[r0:r1, c0:c1], (disc * f).astype(g.dtype))
            used.append(sid)
        return used

    def _say(self, what, started):
        wet = self.wet()
        self.log(f"[{datetime.now():%H:%M:%S}] gauges: {what} in {time.time() - started:.0f}s, {len(self.damp())} damp, "
                 f"drizzle at {', '.join(s for s, *_ in wet) or '-'}")

    def loop(self):
        seen, listed_day = {}, None
        while True:
            started = time.time()
            try:
                if time.strftime("%Y%m%d") != listed_day:  # stations come and go
                    self.stations, listed_day = self._station_list(), time.strftime("%Y%m%d")
                stamps = self._changed()
                todo = [sid for sid, stamp in stamps.items() if seen.get(sid) != stamp]
                if todo:
                    done = self._read(todo)
                    seen.update({sid: stamps[sid] for sid in done})
                    self._save_latest()
                    self._say(f"{len(done)} of {len(todo)} new files", started)
            except Exception as e:
                self.log(f"gauges: {e}")
            time.sleep(max(30.0, LIST_EVERY_S - (time.time() - started)))
