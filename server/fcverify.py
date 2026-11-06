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
        if lines:
            os.makedirs(LOG_DIR, exist_ok=True)
            day = datetime.fromtimestamp(lines[0]["valid"] / 1000, timezone.utc).strftime("%Y%m%d")
            with open(os.path.join(LOG_DIR, f"fcgrid_{day}.jsonl"), "a") as f:
                for row in lines:
                    f.write(json.dumps(row) + "\n")

    def loop(self):
        while True:
            try:
                self.step()
            except Exception as e:
                self.log(f"fcverify: {e}")
            time.sleep(60)
