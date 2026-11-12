"""Tell the operator at once when users see a broken map - instead of the next daily run finding it.

12.11.2026: the Europe radar had no new image for 22 h (the provider had moved to its backup bucket) and the map was
hatched all that time; the health rows showed it, but nobody reads them between the daily runs.

Runs in the Germany instance (the one with the push subscriptions). Every 5 minutes it reads the newest health row of
each instance from the shared verify_log (monitor.py writes one every 5 min) and the server starts, and finds:
  - an instance without a health row for 15 min (not running or hanging)
  - a rain radar image older than 25 min, or none (normal: 5-10 min)
  - a sunny image older than 50 min, or none (normal: 20-28 min)
  - more than 3 server starts in 30 min (crash loop)
A problem that is still there at the next check (so not the 1-2 minutes after a deploy) is sent as a push message to
the operator devices: the push subscriptions of the accounts listed in DATA_DIR/operators.txt (user ids, one per
line; without the file: account 1, the first one created). Sent again every 6 h while it lasts, and once more when it
is over. Nothing between 23:00 and 07:00 German time - it is sent at 07:00 if it still holds.
Every message is also written to verify_log/ops_<day>.jsonl (read by the daily run).
"""
import glob
import json
import os
import re
import time
from datetime import datetime
from zoneinfo import ZoneInfo

import monitor

EVERY_S = 300
NO_ROW_S = 15 * 60
RAIN_MAX_MIN, SUN_MAX_MIN = 25, 50
STARTS_MAX, STARTS_WINDOW_S = 3, 30 * 60
AGAIN_S = 6 * 3600
NAMES = {"de": "Deutschland", "eu": "Europa"}


def _rows(name, days=2):
    out = []
    for path in sorted(glob.glob(os.path.join(monitor.LOG_DIR, f"{name}_*.jsonl")))[-days:]:
        try:
            with open(path) as f:
                for line in f:
                    try:
                        out.append(json.loads(line))
                    except ValueError:
                        pass
        except OSError:
            pass
    return out


def problems(now=None, editions=("de", "eu")):
    """{key: German text} of what is wrong right now."""
    now = now or time.time()
    out = {}
    last = {}
    for r in _rows("health"):
        last[r.get("edition")] = r
    for ed in editions:
        name, r = NAMES[ed], last.get(ed)
        if r is None or now - r["ts"] / 1000 > NO_ROW_S:
            out[f"{ed}:down"] = f"{name}: Server meldet sich nicht"
            continue
        for key, label, limit in (("rain_age", "Regenradar", RAIN_MAX_MIN), ("sun_age", "sunny", SUN_MAX_MIN)):
            age = r.get(key)
            if age is None:
                out[f"{ed}:{key}"] = f"{name}: {label} hat keine Bilder"
            elif age > limit:
                out[f"{ed}:{key}"] = f"{name}: {label} seit {int(age)} Min. ohne neues Bild"
    starts = {}
    for r in _rows("starts"):
        if now - r["ts"] / 1000 < STARTS_WINDOW_S:
            starts[r.get("edition")] = starts.get(r.get("edition"), 0) + 1
    for ed, n in starts.items():
        if n > STARTS_MAX and ed in NAMES:
            out[f"{ed}:starts"] = f"{NAMES[ed]}: Server startet immer wieder neu ({n}x in 30 Min.)"
    return out


class OpsWatch:
    def __init__(self, alerts, data_dir, log):
        self.alerts, self.dir, self.log = alerts, data_dir, log
        self.seen = {}        # key -> text, at the previous check
        self.sent = {}        # key -> time of the last message
        self.no_device = 0.0

    def operators(self):
        try:
            with open(os.path.join(self.dir, "operators.txt")) as f:
                return {s.strip() for s in f if s.strip()}
        except OSError:
            return {"1"}

    def push(self, title, body):
        monitor._append("ops", {"ts": int(time.time() * 1000), "title": title, "body": body})
        ops = self.operators()
        with self.alerts.lock:
            sids = [sid for sid, e in self.alerts.subs.items() if str(e.get("user")) in ops]
        if not sids and time.time() - self.no_device > 6 * 3600:
            self.no_device = time.time()
            self.log(f"ops: no operator device with alerts switched on - not sent: {title} | {body}")
        for sid in sids:
            self.alerts.send(sid, {"title": title, "body": body, "tag": "ops"})
        return len(sids)

    def check(self, now=None):
        now = now or time.time()
        cur = problems(now)
        hour = datetime.fromtimestamp(now, ZoneInfo("Europe/Berlin")).hour
        if 7 <= hour < 23:
            due = [k for k in cur if k in self.seen and now - self.sent.get(k, 0) > AGAIN_S]
            if due:
                self.push("rainy: Störung", " · ".join(cur[k] for k in due))
                for k in due:
                    self.sent[k] = now
            gone = [k for k in self.sent if k not in cur]
            if gone:
                self.push("rainy: wieder in Ordnung", " · ".join(re.split(r" seit | hat keine | meldet sich | startet ", self.seen.get(k, k))[0] for k in gone))
                for k in gone:
                    del self.sent[k]
        self.seen = {**{k: v for k, v in self.seen.items() if k in self.sent}, **cur}

    def loop(self):
        time.sleep(EVERY_S)  # the first health rows after a start
        while True:
            try:
                self.check()
            except Exception as e:
                self.log(f"ops watch failed: {e}")
            time.sleep(EVERY_S)
