"""Forecast verification on the full radar grid: every forecast against the DWD radar image that arrives later.

For each new DWD run the forecast frames at LEADS are kept (downsampled to ~3 km). When the radar analysis for a
frame's valid time has arrived, both are compared inside the area the map shows (Germany + border zone): a forecast
pixel is a hit when real rain lies within TOL_PX (~9 km), so small position errors are forgiven. Persistence (the
analysis at issue time, i.e. "it stays as it is") is scored alongside as the baseline any forecast must beat.
Both forecasts are scored: DWD's own nowcast ("dwd") and ours ("own", rainy_server.OwnNowcast), same runs, same rules.

One JSON line per (run, lead) in verify_log/fcgrid_<YYYYMMDD>.jsonl; server/score_grid.py sums them up. No downloads,
no screenshots: thousands of comparisons per rainy day instead of a few.
"""
import json
import os
import threading
import time
from datetime import datetime, timezone

import cv2
import numpy as np

LOG_DIR = os.environ.get("VERIFY_LOG_DIR", os.path.join(os.path.dirname(os.path.abspath(__file__)), "verify_log"))
LEADS = (15, 30, 60, 90, 120)  # minutes; the same steps as the WetterOnline comparison, plus our +120
DOWN = 3                       # ~3 km pixels
TOL_PX = 3                     # ~9 km position tolerance
THRESHOLDS = (0.1, 1.0)        # mm/h: "wet" and "real rain"
KEEP_MS = 3 * 3600_000
OBS_DIR = os.path.join(LOG_DIR, "obs3km")
OBS_KEEP_DAYS = 14


def _down(g, shape):
    return cv2.resize(np.asarray(g, np.float32), shape, interpolation=cv2.INTER_AREA)


def _counts(fc, obs, mask):
    """hits, misses, false alarms with TOL_PX tolerance in both directions."""
    fc, obs = fc & mask, obs & mask
    near_obs = cv2.dilate(obs.astype(np.uint8), np.ones((2 * TOL_PX + 1,) * 2, np.uint8)) > 0
    near_fc = cv2.dilate(fc.astype(np.uint8), np.ones((2 * TOL_PX + 1,) * 2, np.uint8)) > 0
    return int((fc & near_obs).sum()), int((obs & ~near_fc).sum()), int((fc & ~near_obs).sum())


class GridVerify:
    def __init__(self, de, log):
        self.de, self.log = de, log
        self.lock = threading.Lock()
        self.pending = {}  # (variant, ref, lead) -> forecast grid (down)
        self.issued = {}   # ref -> analysis at issue time (down), for persistence
        self.done_ref, self.done_own = None, None
        self.shape = (de.nx // DOWN, de.ny // DOWN)  # cv2 (w, h)
        self.mask = None   # area the map shows, downsampled
        self.saved = set() # analyses already on disk (_save_obs)

    def step(self):
        de = self.de
        if getattr(de, "frozen", False) or not de.ref:
            return
        with de.lock:
            ref, obs, fc = de.ref, dict(de.obs), dict(de.fc)
            own_ref, own = de.own_ref, dict(de.fc_own)
            shadow = {k: dict(v) for k, v in getattr(de, "fc_shadow", {}).items()}
        if ref != self.done_ref and ref in obs:
            self.mask = _down(de.inside, self.shape) > 0.5  # Germany + border zone (radar range known after 1st run)
            self.issued[ref] = _down(obs[ref], self.shape)
            for lead in LEADS:
                g = fc.get(ref + lead * 60_000)
                if g is not None:
                    self.pending[("dwd", ref, lead)] = _down(g, self.shape)
            self.done_ref = ref
        if own_ref is not None and own_ref != self.done_own:
            for lead in LEADS:
                for name, grids in [("own", own)] + list(shadow.items()):  # shadows: variants being tested
                    g = grids.get(own_ref + lead * 60_000)
                    if g is not None:
                        self.pending[(name, own_ref, lead)] = _down(g, self.shape)
            self.done_own = own_ref
        lines = []
        for (variant, r, lead), g in list(self.pending.items()):
            valid = r + lead * 60_000
            if valid not in obs:
                if ref - r > KEEP_MS:
                    del self.pending[(variant, r, lead)]
                continue
            a = _down(obs[valid], self.shape)
            p = self.issued.get(r)
            row = {"variant": variant, "ref": r, "lead": lead, "valid": valid,
                   "sum": [round(float(g[self.mask].sum()), 1), round(float(a[self.mask].sum()), 1)]}  # rain amount: forecast, real
            for th in THRESHOLDS:
                o = a >= th
                row[f"obs{th}"] = int((o & self.mask).sum())
                row[f"fc{th}"] = list(_counts(g >= th, o, self.mask))
                if p is not None and variant == "dwd":  # persistence once per run
                    row[f"pers{th}"] = list(_counts(p >= th, o, self.mask))
            lines.append(row)
            del self.pending[(variant, r, lead)]
        for r in [r for r in self.issued if ref - r > KEEP_MS]:
            del self.issued[r]
        self._save_obs(obs)
        if lines:
            os.makedirs(LOG_DIR, exist_ok=True)
            day = datetime.fromtimestamp(lines[0]["valid"] / 1000, timezone.utc).strftime("%Y%m%d")
            with open(os.path.join(LOG_DIR, f"fcgrid_{day}.jsonl"), "a") as f:
                for row in lines:
                    f.write(json.dumps(row) + "\n")

    def _save_obs(self, obs):
        """Every radar analysis at ~3 km on disk (OBS_DIR/<UTC YYYYmmddHHMM>.npz, mm/h, float16), plus the scored area
        (mask.npy) and the grid (grid.json): the truth for forecasts made elsewhere (tools/wo_verify.py scores
        WetterOnline's map against it). Kept OBS_KEEP_DAYS."""
        if self.mask is None:
            return
        os.makedirs(OBS_DIR, exist_ok=True)
        if not os.path.exists(os.path.join(OBS_DIR, "mask.npy")):
            np.save(os.path.join(OBS_DIR, "mask.npy"), self.mask)
            with open(os.path.join(OBS_DIR, "grid.json"), "w") as f:  # Web Mercator, one cell = DOWN source cells
                json.dump({"bbox": list(self.de.BBOX), "cell": self.de.CELL * DOWN, "shape": list(self.mask.shape)}, f)
        for t, g in obs.items():
            name = datetime.fromtimestamp(t / 1000, timezone.utc).strftime("%Y%m%d%H%M")
            if name in self.saved:
                continue
            path = os.path.join(OBS_DIR, name + ".npz")
            if not os.path.exists(path):
                np.savez_compressed(path, rate=_down(g, self.shape).astype(np.float16))
            self.saved.add(name)
        if len(self.saved) > 400:  # once in a while: forget old names, delete old files
            cutoff = (datetime.now(timezone.utc).timestamp() - OBS_KEEP_DAYS * 86400)
            for p in os.listdir(OBS_DIR):
                if p.endswith(".npz") and datetime.strptime(p[:12], "%Y%m%d%H%M").replace(
                        tzinfo=timezone.utc).timestamp() < cutoff:
                    os.remove(os.path.join(OBS_DIR, p))
            self.saved = {n for n in self.saved if datetime.strptime(n, "%Y%m%d%H%M").replace(
                tzinfo=timezone.utc).timestamp() >= cutoff}

    def loop(self):
        while True:
            try:
                self.step()
            except Exception as e:
                self.log(f"fcverify: {e}")
            time.sleep(60)


def wo_eu_box():
    """lon0, lat0, lon1, lat1 of WetterOnline's Europe view (tools/wo_capture.py MAP_EU: centre 48.5 N 11.0 E, zoom 5,
    the 640 x 650 px frame with its centre at pixel (320, 345)): the area tools/wo_verify_eu.py scores, so that our rows
    for area "wo" cover exactly the same ground."""
    import math
    n = 256 * 2 ** 5
    cx = (11.0 + 180) / 360 * n
    cy = (1 - math.log(math.tan(math.radians(48.5)) + 1 / math.cos(math.radians(48.5))) / math.pi) / 2 * n
    lon = lambda x: x / n * 360 - 180
    lat = lambda y: math.degrees(math.atan(math.sinh(math.pi * (1 - 2 * y / n))))
    return lon(cx - 320), lat(cy + 305), lon(cx + 320), lat(cy - 345)


class RegionVerify:
    """The same verification for a radar region without an official nowcast (the Europe test instance, OPERA):
    every run's forecasts against the region's later radar image, inside the radar coverage at both times, on a grid of
    ~4 km (region grid / down). Variants: "own" (as served: S-PROG + the weather model's new showers in the 2nd hour),
    "sprog" (S-PROG alone, Region.fc_raw), "sprog0" (S-PROG before the advection blend of 10.11., Region.fc_pure) and
    "pers" (the analysis at issue time: the baseline to beat).
    Rows as fcgrid (score_same.py --log eugrid): verify_log/eugrid_<YYYYMMDD>.jsonl. Every analysis is saved on that
    grid to verify_log/obs_eu/ (truth for tools/wo_verify_eu.py, kept OBS_KEEP_DAYS) with grid.json (the region's
    projection and cell size). Each row is written twice: area "all" (the whole radar coverage) and area "wo" (only
    WetterOnline's Europe view, wo_eu_box: the same ground as their map for a fair comparison)."""

    def __init__(self, region, log, prefix="eugrid", obs_dir="obs_eu"):
        self.region, self.log, self.prefix = region, log, prefix
        self.down = getattr(region, "RAW_DOWN", 2)
        self.obs_dir = os.path.join(LOG_DIR, obs_dir)
        self.pending = {}  # (variant, ref, lead) -> (forecast grid, coverage at issue time)
        self.done_ref = None
        self.saved = set()
        self.areas = None  # name -> mask on the verification grid (without the radar coverage)

    def _areas(self, shape):
        h, w = shape
        cols, rows = np.meshgrid((np.arange(w) + 0.5) * self.down - 0.5, (np.arange(h) + 0.5) * self.down - 0.5)
        lon, lat = (np.asarray(v).reshape(h, w) for v in self.region.pixel_lonlat(cols.ravel(), rows.ravel()))
        x0, y0, x1, y1 = wo_eu_box()
        return {"all": np.ones((h, w), bool), "wo": (lon >= x0) & (lon <= x1) & (lat >= y0) & (lat <= y1)}

    def _grid(self, g):
        """(rate, covered) on the verification grid; NODATA (< 0) = outside the radar range."""
        g = np.asarray(g, np.float32)
        h, w = g.shape[0] // self.down, g.shape[1] // self.down
        cov = _down((g >= 0).astype(np.float32), (w, h)) > 0.5
        return _down(np.where(g < 0, 0, g), (w, h)), cov

    def step(self):
        r = self.region
        with r.lock:
            ref, obs = r.ref, dict(r.obs)
            fc, raw = dict(r.fc), dict(getattr(r, "fc_raw", None) or {})
            pure = dict(getattr(r, "fc_pure", None) or {})
        if not ref or ref not in obs:
            return
        if ref != self.done_ref:
            a0, cov0 = self._grid(obs[ref])
            for lead in LEADS:
                t = ref + lead * 60_000
                self.pending[("pers", ref, lead)] = (a0, cov0)
                if t in fc:
                    self.pending[("own", ref, lead)] = (self._grid(fc[t])[0], cov0)
                for name, src in (("sprog", raw), ("sprog0", pure)):
                    if t in src:
                        g = np.asarray(src[t], np.float32)
                        if g.shape != a0.shape:
                            g = cv2.resize(g, (a0.shape[1], a0.shape[0]), interpolation=cv2.INTER_AREA)
                        self.pending[(name, ref, lead)] = (np.where(g < 0, 0, g), cov0)
            self.done_ref = ref
        lines, truth = [], {}
        for (variant, rr, lead), (g, cov0) in list(self.pending.items()):
            valid = rr + lead * 60_000
            if valid not in obs:
                if ref - rr > KEEP_MS:
                    del self.pending[(variant, rr, lead)]
                continue
            if valid not in truth:
                truth[valid] = self._grid(obs[valid])
            a, cov = truth[valid]
            if self.areas is None:
                self.areas = self._areas(a.shape)
            for area, amask in self.areas.items():
                mask = cov0 & cov & amask
                row = {"variant": variant, "area": area, "ref": rr, "lead": lead, "valid": valid,
                       "sum": [round(float(g[mask].sum()), 1), round(float(a[mask].sum()), 1)]}
                for th in THRESHOLDS:
                    o = a >= th
                    row[f"obs{th}"] = int((o & mask).sum())
                    row[f"fc{th}"] = list(_counts(g >= th, o, mask))
                lines.append(row)
            del self.pending[(variant, rr, lead)]
        self._save_obs(obs)
        if lines:
            os.makedirs(LOG_DIR, exist_ok=True)
            day = datetime.fromtimestamp(lines[0]["valid"] / 1000, timezone.utc).strftime("%Y%m%d")
            with open(os.path.join(LOG_DIR, f"{self.prefix}_{day}.jsonl"), "a") as f:
                for row in lines:
                    f.write(json.dumps(row) + "\n")

    def _save_obs(self, obs):
        os.makedirs(self.obs_dir, exist_ok=True)
        meta = os.path.join(self.obs_dir, "grid.json")
        if not os.path.exists(meta):
            r = self.region
            with open(meta, "w") as f:
                json.dump({"proj": getattr(r, "LAEA", None), "cell": r.CELL * self.down, "origin": "top left of the "
                           "region grid (x = col * cell, y = -row * cell)"}, f)
        for t, g in obs.items():
            name = datetime.fromtimestamp(t / 1000, timezone.utc).strftime("%Y%m%d%H%M")
            if name in self.saved:
                continue
            path = os.path.join(self.obs_dir, name + ".npz")
            if not os.path.exists(path):
                a, cov = self._grid(g)
                np.savez_compressed(path, rate=np.where(cov, a, -1).astype(np.float16))
            self.saved.add(name)
        if len(self.saved) > 400:
            cutoff = datetime.now(timezone.utc).timestamp() - OBS_KEEP_DAYS * 86400
            for p in os.listdir(self.obs_dir):
                if p.endswith(".npz") and datetime.strptime(p[:12], "%Y%m%d%H%M").replace(
                        tzinfo=timezone.utc).timestamp() < cutoff:
                    os.remove(os.path.join(self.obs_dir, p))
            self.saved = {n for n in self.saved if datetime.strptime(n, "%Y%m%d%H%M").replace(
                tzinfo=timezone.utc).timestamp() >= cutoff}

    def loop(self):
        while True:
            try:
                self.step()
            except Exception as e:
                self.log(f"euverify: {e}")
            time.sleep(60)
