"""Verification recorder: keeps what we forecast and what really happened, for server/score.py.

Forecasts: whenever a new rain or cloud forecast exists, the series the app would show is sampled at DWD station
locations (same averaging as /api/rain/point and /api/cloud/point) and saved with the time it was issued, so every
forecast can later be scored at its real lead time. Small maps at +0/+60/+120 min are kept for visual checks.
Truth: DWD 10-minute station observations (open data, ~30 min behind): sunshine duration at ~70 stations and
rain gauges at a thinned set of ~300 stations. The "now" files only hold the current day, so they are fetched
regularly and merged into one file per day.
"""
import io
import json
import re
import os
import time
import zipfile
from datetime import datetime, timezone

import cv2
import numpy as np

LOG_DIR = os.environ.get("VERIFY_LOG_DIR", os.path.join(os.path.dirname(os.path.abspath(__file__)), "verify_log"))
KEEP_S = 21 * 86400
CDC = "https://opendata.dwd.de/climate_environment/CDC/observations_germany/climate/10_minutes"
TRUTH = {  # kind: (folder, station list, file prefix, value column)
    "sun": ("solar", "zehn_now_sd_Beschreibung_Stationen.txt", "10minutenwerte_SOLAR_", "SD_10"),
    "rain": ("precipitation", "zehn_now_rr_Beschreibung_Stationen.txt", "10minutenwerte_nieder_", "RWS_10"),
}
RAIN_CELL = (0.35, 0.5)          # thin the ~1400 rain gauges to one per cell (deg lat, lon): ~300 stations
TRUTH_EVERY_S = 3600
RAIN_EVERY_MS = 15 * 60_000      # rain forecasts are recorded at most every 15 min (enough for statistics)
MAP_BOX = (2.0, 46.0, 19.0, 56.0)  # lon/lat box of the small verification maps (Germany and neighbours)
MAP_W, MAP_H = 340, 300          # ~4 km pixels
MAP_LEADS = (0, 60, 120)


def day_str(ts_ms):
    return datetime.fromtimestamp(ts_ms / 1000, timezone.utc).strftime("%Y%m%d")


class Recorder:
    def __init__(self, fetch, log, mosaic, clouds, merc):
        self.fetch, self.log, self.mosaic, self.clouds, self.merc = fetch, log, mosaic, clouds, merc
        self.stations = {}           # kind -> {"id": [...], "lat": array, "lon": array}
        self.cloud_ref, self.rain_key, self.rain_at = None, None, 0
        self.truth_at = 0
        lon = np.linspace(MAP_BOX[0], MAP_BOX[2], MAP_W)
        lat = np.linspace(MAP_BOX[3], MAP_BOX[1], MAP_H)
        self.map_xy = merc(*np.meshgrid(lon, lat))

    # ---------- stations ----------
    def _station_list(self, kind):
        folder, listing, _, _ = TRUTH[kind]
        text = self.fetch(f"{CDC}/{folder}/now/{listing}", timeout=60).decode("latin-1")
        recent = time.strftime("%Y%m%d", time.gmtime(time.time() - 3 * 86400))
        out = []
        for line in text.splitlines()[2:]:
            p = line.split()
            if len(p) >= 6 and p[0].isdigit() and p[2] >= recent:
                out.append((p[0], float(p[4]), float(p[5])))
        if kind == "rain":  # one gauge per cell, evenly spread
            cells = {}
            for s in out:
                cells.setdefault((int(s[1] // RAIN_CELL[0]), int(s[2] // RAIN_CELL[1])), s)
            out = list(cells.values())
        return {"id": [s[0] for s in out], "lat": np.array([s[1] for s in out]), "lon": np.array([s[2] for s in out])}

    def _ensure_stations(self):
        for kind in TRUTH:
            if kind not in self.stations:
                self.stations[kind] = self._station_list(kind)
                self.log(f"recorder: {len(self.stations[kind]['id'])} {kind} stations")

    # ---------- truth ----------
    def fetch_truth(self):
        os.makedirs(LOG_DIR, exist_ok=True)
        for kind, (folder, _, prefix, col) in TRUTH.items():
            new = {}
            for sid in self.stations[kind]["id"]:
                try:
                    z = zipfile.ZipFile(io.BytesIO(self.fetch(f"{CDC}/{folder}/now/{prefix}{sid}_now.zip", timeout=60)))
                    rows = z.read(z.namelist()[0]).decode("latin-1").splitlines()
                except Exception:
                    continue
                head = [h.strip() for h in rows[0].split(";")]
                k = head.index(col)
                for r in rows[1:]:
                    p = r.split(";")
                    v = float(p[k])
                    if v > -999:
                        ts = int(datetime.strptime(p[1].strip(), "%Y%m%d%H%M").replace(tzinfo=timezone.utc).timestamp() * 1000)
                        if ts < (time.time() - 3 * 86400) * 1000:  # a few "now" files carry stray old rows
                            continue
                        new.setdefault(day_str(ts), {}).setdefault(sid, {})[str(ts)] = v
            for day, data in new.items():
                fp = os.path.join(LOG_DIR, f"truth_{kind}_{day}.json")
                old = json.load(open(fp)) if os.path.exists(fp) else {}
                for sid, vals in data.items():
                    old.setdefault(sid, {}).update(vals)
                with open(fp + ".tmp", "w") as f:
                    json.dump(old, f)
                os.replace(fp + ".tmp", fp)
        self.truth_at = time.time()

    # ---------- forecasts ----------
    def _station_xy(self, kind):
        s = self.stations[kind]
        return self.merc(s["lon"], s["lat"])

    def record_clouds(self):
        ref = self.clouds.ref
        frames, _ = self.clouds.frames()
        if ref is None or ref == self.cloud_ref or not any(f["fc"] for f in frames):
            return
        issued = int(time.time() * 1000)
        X, Y = self._station_xy("sun")
        col, row = (v.astype(np.float32).reshape(1, -1) for v in self.clouds._pixels(X, Y))
        mcol, mrow = (v.astype(np.float32) for v in self.clouds._pixels(*self.map_xy))
        ts, fc, cloud, model, raw, maps = [], [], [], [], [], {}
        for f in frames:
            g = self.clouds.grid(f["ts"])
            if g is None:
                continue
            mean = cv2.blur(g, (5, 5))  # same ~12 km mean as /api/cloud/point
            ts.append(f["ts"]); fc.append(f["fc"])
            cloud.append(cv2.remap(mean, col, row, cv2.INTER_LINEAR).ravel() / 255.0)
            m = self.clouds.model_field(f["ts"])
            model.append(cv2.remap(m, col, row, cv2.INTER_LINEAR).ravel() if m is not None else np.full(len(X), np.nan))
            r = self.clouds.raw_grid(f["ts"])
            raw.append(cv2.remap(cv2.blur(r, (5, 5)), col, row, cv2.INTER_LINEAR).ravel().astype(np.float32)
                       if r is not None else np.full(len(X), np.nan))
            for lead in MAP_LEADS:
                if f["ts"] == min((x["ts"] for x in frames), key=lambda t: abs(t - (ref + lead * 60_000))):
                    maps[f"map{lead}"] = cv2.remap(g, mcol, mrow, cv2.INTER_LINEAR)
        np.savez_compressed(os.path.join(LOG_DIR, f"fc_clouds_{ref}.npz"), issued=issued, ref=ref, ts=np.array(ts),
                            ids=np.array(self.stations["sun"]["id"]), lat=self.stations["sun"]["lat"], lon=self.stations["sun"]["lon"],
                            fc=np.array(fc), cloud=np.array(cloud, np.float32), model=np.array(model, np.float32),
                            raw=np.array(raw, np.float32), **maps)
        self.cloud_ref = ref

    def record_rain(self):
        key = self.mosaic.key()
        now = int(time.time() * 1000)
        if key == self.rain_key or now - self.rain_at < RAIN_EVERY_MS:
            return
        X, Y = self._station_xy("rain")
        s = self.stations["rain"]
        d = np.array([-1500.0, 0.0, 1500.0])
        dx, dy = np.meshgrid(d, d)
        k = (1 / np.cos(np.radians(s["lat"])))[:, None]
        PX, PY = X[:, None] + dx.ravel()[None] * k, Y[:, None] + dy.ravel()[None] * k  # 3x3 like /api/rain/point
        frames = self.mosaic.frames()
        ts, fc, rate, opera, noblend, maps = [], [], [], [], [], {}
        eu = self.mosaic.regions["eu"]
        raw = getattr(eu, "fc_raw", None) or {}
        ec, er = (((np.asarray(v, np.float64) + 0.5) / 3 - 0.5).astype(np.float32) for v in eu.to_pixels(PX, PY))
        ref = max((f["ts"] for f in frames if not f["fc"]), default=now)
        for f in frames:
            r, _ = self.mosaic.rain(f["ts"], PX, PY)
            ts.append(f["ts"]); fc.append(f["fc"]); rate.append(r.mean(axis=1))
            o, _ = self.mosaic.rain(f["ts"], PX, PY, use_de=False)  # shadow: Europe composite only
            opera.append(o.mean(axis=1))
            g = raw.get(f["ts"]) if f["fc"] else None  # shadow 2: OPERA nowcast without the model's new cells
            noblend.append(o.mean(axis=1) if g is None else
                           cv2.remap(np.maximum(g.astype(np.float32), 0), ec, er, cv2.INTER_LINEAR).mean(axis=1))
            if any(f["ts"] == ref + lead * 60_000 for lead in MAP_LEADS):
                m, _ = self.mosaic.rain(f["ts"], *self.map_xy)
                maps[f"map{(f['ts'] - ref) // 60_000}"] = np.clip(m * 10, 0, 65535).astype(np.uint16)  # 0.1 mm/h
        np.savez_compressed(os.path.join(LOG_DIR, f"fc_rain_{ref}_{now}.npz"), issued=now, ref=ref, ts=np.array(ts),
                            ids=np.array(s["id"]), lat=s["lat"], lon=s["lon"],
                            fc=np.array(fc), rate=np.array(rate, np.float32), opera=np.array(opera, np.float32),
                            noblend=np.array(noblend, np.float32), **maps)
        self.rain_key, self.rain_at = key, now

    def archive_cells(self):
        """Keep DWD thunderstorm cell (KONRAD3D) and mesocyclone files that contain anything: there is no format
        description and no cells in autumn, so real examples are collected for building the cell display."""
        base = "https://opendata.dwd.de/weather/radar"
        out = os.path.join(LOG_DIR, "cells")
        for folder, pattern, empty in (("konrad3d", r"KONRAD3D_\d{8}T\d{6}\.xml", "<feature"), ("mesocyclones", r"meso_\d{8}_\d{4}\.xml", "<meso")):
            try:
                names = sorted(set(re.findall(pattern, self.fetch(f"{base}/{folder}/", timeout=60).decode("latin-1"))))[-3:]
                for n in names:
                    fp = os.path.join(out, n)
                    if os.path.exists(fp):
                        continue
                    data = self.fetch(f"{base}/{folder}/{n}", timeout=60)
                    if empty.encode() in data or (folder == "mesocyclones" and len(data) > 400):
                        os.makedirs(out, exist_ok=True)
                        with open(fp, "wb") as f:
                            f.write(data)
                        self.log(f"recorder: archived {n}")
            except Exception as e:
                self.log(f"recorder: cells {folder}: {e}")

    def _cleanup(self):
        for f in os.listdir(LOG_DIR):
            fp = os.path.join(LOG_DIR, f)
            if f.startswith(("fc_", "truth_")) and time.time() - os.path.getmtime(fp) > KEEP_S:
                os.remove(fp)

    def loop(self):
        os.makedirs(LOG_DIR, exist_ok=True)
        while True:
            try:
                self._ensure_stations()
                self.record_clouds()
                self.record_rain()
                self.archive_cells()
                if time.time() - self.truth_at > TRUTH_EVERY_S:
                    self.fetch_truth()
                    self._cleanup()
            except Exception as e:
                self.log(f"recorder: {e}")
            time.sleep(60)
