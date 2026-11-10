"""What a user would notice, measured by the server itself - so that it does not take a user to find it.

Until 10.11. only forecast skill was measured (fcverify.py, sunny_de.SunVerify). Everything the user reported in the
days before was outside those numbers: a break between the last measured image and the first forecast step, a jump of
the sunny map at sunset (70 % -> 96 % cloudy in 30 min), an area flipping sunny - overcast - sunny, and - unnoticed by
anyone for 5 h - the Europe instance being killed every 10 min (out of memory) and a model shadow failing every minute.
Three records, all in verify_log, summed up by score_jumps.py and daily_report.py:

  timeline_<day>.jsonl  one row per run and product: how much the map changes from each step of the timeline to the
                        next (measured frames, the step at "now", every forecast step). A step that changes far more
                        than its neighbours is a jump a user sees when moving the timeline.
                          rain: "area" = wet cells (>= 0.1 mm/h, ~3 km), "chg" = share of the wet area that differs
                          sun:  "area" = sunny share of the map (displayed value >= 0.5), "chg" = mean |change| of the
                                displayed value (0 = overcast .. 1 = clear)
  starts_<day>.jsonl    one row per server start (edition, version): a crash loop shows as dozens of starts
  health_<day>.jsonl    every 5 min: age of the newest rain and sunny image, memory, and the error messages logged
                        since the last row (counted by text with the numbers removed)
"""
import json
import os
import re
import threading
import time
from collections import Counter
from datetime import datetime, timezone

import numpy as np

LOG_DIR = os.environ.get("VERIFY_LOG_DIR", os.path.join(os.path.dirname(os.path.abspath(__file__)), "verify_log"))
EDITION, TAG = "", ""   # set by start()
HEALTH_EVERY_S = 300
ERROR_WORDS = re.compile(r"fail|error|refused|traceback|exception|timed out|not saved", re.I)
_lock = threading.Lock()
_errors = Counter()


def _append(name, row, ts_ms=None, log_dir=None):
    day = datetime.fromtimestamp((ts_ms or time.time() * 1000) / 1000, timezone.utc).strftime("%Y%m%d")
    d = log_dir or LOG_DIR
    with _lock:
        os.makedirs(d, exist_ok=True)
        with open(os.path.join(d, f"{name}_{day}.jsonl"), "a") as f:
            f.write(json.dumps(row) + "\n")


def note(msg):
    """Called with every log line: error messages are counted for the health rows."""
    if ERROR_WORDS.search(msg):
        text = re.sub(r"^\[[\d:]+\]\s*", "", msg.strip().splitlines()[0])
        with _lock:
            _errors[re.sub(r"\d+(\.\d+)?", "N", text)[:140]] += 1


def timeline(kind, product, ref, offsets, frames, mask, n_obs, log_dir=None):
    """One timeline row (see the module text). frames: the run's maps in timeline order (rain: mm/h; sun: displayed
    value 0..1, NaN = unknown), offsets: their time in minutes relative to ref, n_obs: how many of them are measured."""
    area, chg, prev = [], [None], None
    for g in frames:
        if kind == "rain":
            cur = (g >= 0.1) & mask
            a = int(cur.sum())
            if prev is not None:
                chg.append(round(float((cur ^ prev[0]).sum()) / max(a, prev[1], 1), 4))
            prev = (cur, a)
            area.append(a)
        else:
            ok = mask & np.isfinite(g)
            n = int(ok.sum())
            area.append(round(float((g[ok] >= 0.5).mean()), 4) if n else None)
            if prev is not None:
                both = ok & prev[1]
                chg.append(round(float(np.abs(g[both] - prev[0][both]).mean()), 4) if both.any() else None)
            prev = (g, ok)
    _append("timeline", {"kind": kind, "product": product, "ref": int(ref), "v": TAG, "t": [int(o) for o in offsets],
                         "n_obs": int(n_obs), "area": area, "chg": chg}, ref, log_dir)


def start(edition, version, status):
    """status(): {"rss": MB, "rain_age": minutes since the newest radar image, "sun_age": ... satellite image}."""
    global EDITION, TAG
    EDITION, TAG = edition, version
    _append("starts", {"ts": int(time.time() * 1000), "edition": edition, "v": version})

    def loop():
        while True:
            time.sleep(HEALTH_EVERY_S)
            try:
                with _lock:
                    errors = dict(_errors)
                    _errors.clear()
                _append("health", {"ts": int(time.time() * 1000), "edition": edition, "v": version, **status(),
                                   "errors": errors})
            except Exception:
                pass
    threading.Thread(target=loop, daemon=True).start()
