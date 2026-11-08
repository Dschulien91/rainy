"""Rain alerts ("Regenwarnung") for subscribed places, sent as Web Push notifications.

Rules (agreed design):
  - main alert about LEAD minutes (5-90, default 45) before rain starts at a place
  - early alert ~60 min ahead only for heavy rain (>= 8 mm/h), only when the forecast is very certain and only with
    a main warning time below 45 min (otherwise it adds nothing)
  - only on the change from dry to wet: no alert while it rains, and after rain a cooldown of COOLDOWN_H dry hours
  - at most MAX_PER_DAY alerts per place and day, quiet hours (optional, default 22:00-07:00, own start and end time;
    severe thunderstorm warnings still come through), one upgrade if it gets heavier
  - "rain at the place" is judged over a ~5 km circle (share of wet pixels), which removes most false alarms
    from showers that pass just beside the place
Accounts (v0.76): alerts need a login (server/auth.py). Every subscription is one device of a user: it stores the
user id and the hash of the device's session. The settings (places, own location on/off, lead time, heads-up,
thunder, quiet hours) belong to the account and are the same on all of the user's devices (apply_user, called when
the app saves the synced user data); only the push address and the device's last seen location ("own location")
differ per device. A subscription ends with the device's session (logout, expiry), with the account, and when the
user switches all alerts off.
Old anonymous subscriptions (before v0.76, no "user"): they keep working until their push address expires (the push
service answers 404/410) or the device unsubscribes; no new anonymous ones are created (subscribe needs a user).
Privacy: the server stores the push endpoint (from the browser's push service), place names and coordinates rounded
to ~1 km, the settings and the account's user id. No device identifiers.
"""
import json
import math
import os
import secrets
import threading
import time
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime
from zoneinfo import ZoneInfo

import numpy as np

WET = 0.3                 # mm/h counted as rain
P_START = 0.5             # share of the 5 km circle that must be wet for "rain here"
P_EARLY = 0.7             # stricter for the 60 min heads-up
MIN_STEPS = 2             # rain must last at least 2 steps (10 min) to be worth an alert
COOLDOWN_H = 2.0
MAX_PER_DAY = 4
RADIUS_KM = 5.0
LEVELS = ((8.0, "starker Regen", "Starker Regen"), (2.0, "mäßiger Regen", "Mäßiger Regen"), (0.0, "leichter Regen", "Leichter Regen"))
ICON = {"Starker Regen": "⛈", "Mäßiger Regen": "🌧", "Leichter Regen": "🌦"}

# Push texts per language (the device's app language, stored as cfg["lang"]; German is the default and fallback).
# Level names are keyed by the German title used internally (LEVELS / ICON).
TEXT = {
    "de": {"levels": {"Starker Regen": "Starker Regen", "Mäßiger Regen": "Mäßiger Regen", "Leichter Regen": "Leichter Regen"},
           "kinds": {"snow": "Schnee", "sleet": "Schneeregen", "freezing": "Glatteisregen", "graupel": "Graupel", "hail": "Hagel"},
           "rain_word": "Regen", "your_place": "dein Ort", "long": "länger anhaltend", "minutes": "etwa {m} Min. lang",
           "chance": " Wahrscheinlichkeit ca. {p} %.",
           "early": ("{icon} {lv} in ca. 1 Std. – {place}", "Voraussichtlich ab ca. {when}, {dur}.{chance} Wir melden uns kurz vorher noch einmal."),
           "upgrade": ("⛈ Jetzt starker Regen erwartet – {place}", "Ab ca. {when}, {dur}.{chance}"),
           "reminder": ("⏰ {lv} in {mins} Min. – {place}", "Erinnerung: ab ca. {when}, {dur}.{chance}"),
           "main": ("{icon} {lv} in {mins} Min. – {place}", "Ab ca. {when}, {dur}.{chance}"),
           "raining": ("🌧 Jetzt regnet es – {place}", "Erinnerung von vorhin."),
           "allclear": ("🌤 Entwarnung – {place}", "Der angekündigte Niederschlag kommt voraussichtlich doch nicht (nächste 2 Std.)."),
           "thunder_now": "Jetzt bis ca. {to}", "thunder_from": "Ab ca. {fr} bis ca. {to}",
           "thunder_body": "{when}. Warnung des Deutschen Wetterdienstes."},
    "en": {"levels": {"Starker Regen": "Heavy rain", "Mäßiger Regen": "Moderate rain", "Leichter Regen": "Light rain"},
           "kinds": {"snow": "snow", "sleet": "sleet", "freezing": "freezing rain", "graupel": "graupel", "hail": "hail"},
           "rain_word": "rain", "your_place": "your place", "long": "lasting a while", "minutes": "for about {m} min",
           "chance": " Probability about {p} %.",
           "early": ("{icon} {lv} in about 1 h – {place}", "Expected from about {when}, {dur}.{chance} We will remind you shortly before."),
           "upgrade": ("⛈ Heavy rain now expected – {place}", "From about {when}, {dur}.{chance}"),
           "reminder": ("⏰ {lv} in {mins} min – {place}", "Reminder: from about {when}, {dur}.{chance}"),
           "main": ("{icon} {lv} in {mins} min – {place}", "From about {when}, {dur}.{chance}"),
           "raining": ("🌧 It is raining now – {place}", "Reminder from earlier."),
           "allclear": ("🌤 All clear – {place}", "The expected precipitation is unlikely to come after all (next 2 h)."),
           "thunder_now": "Now until about {to}", "thunder_from": "From about {fr} until about {to}",
           "thunder_body": "{when}. Warning from the German Weather Service (DWD)."},
}


def text(cfg):
    return TEXT.get((cfg or {}).get("lang"), TEXT["de"])
MAX_PLACES = 20
# Share of alerts that were right (rain at the place within +-15 min of the predicted start), by lead time and
# predicted peak: server/verify_alerts.py --truth radar (first 1.5 days of recorded forecasts, provisional; rerun
# with more rainy days). (lead up to min, [light < 1 mm/h, moderate, heavy >= 4 mm/h]) - shown as "ca. X %".
HIT_RATE = ((30, (30, 60, 70)), (45, (20, 60, 70)), (60, (20, 45, 55)), (90, (15, 40, 50)), (999, (15, 25, 35)))
MAX_REMINDERS = 10
QUIET_FROM, QUIET_TO = 22 * 60, 7 * 60  # default quiet hours (minutes after local midnight)
SWEEP_S = 600  # how often subscriptions of ended sessions are removed
# Sending: pywebpush encrypts each message (~3 ms CPU) and waits for the push service (~50-150 ms), so messages go
# out in parallel. server/bench_push.py on 16 cores: 1 worker 12/s, 16 workers 188/s, 64 workers 395/s (CPU-bound).
# On the 4-vCPU production server 16 workers and at most 50 messages/s leave room for the map: 1000 alerts in ~20 s.
PUSH_WORKERS = 16
PUSH_PER_S = 50


def probability(lead_min, peak):
    """Rounded alert probability in percent for a lead time and peak rate."""
    k = 0 if peak < 1.0 else 1 if peak < 4.0 else 2
    return next(r[k] for lim, r in HIT_RATE if lead_min <= lim)


def remind_minutes(lead_min):
    """'Remind me again' delay: about a third of the remaining time (90 min -> 30, 30 -> 10), at least 5 min,
    and at least 5 min before the rain; None when there is too little time left."""
    if lead_min < 15:
        return None
    return int(max(5, min(30, round(lead_min / 3 / 5) * 5, lead_min - 5)))


def minutes(v, default):
    """'HH:MM' or minutes after midnight -> minutes (0-1439, 5 min steps), else default."""
    try:
        if isinstance(v, str):
            h, m = v.split(":")
            v = int(h) * 60 + int(m)
        v = int(v)
        return v if 0 <= v < 1440 and v % 5 == 0 else default
    except (ValueError, TypeError):
        return default


def quiet_now(cfg, local):
    """Inside the quiet hours (local datetime); the window may cross midnight (22:00-07:00)."""
    if not cfg.get("quiet", True):
        return False
    start, end = cfg.get("quietFrom", QUIET_FROM), cfg.get("quietTo", QUIET_TO)
    m = local.hour * 60 + local.minute
    if start == end:
        return False
    return start <= m < end if start < end else (m >= start or m < end)


def shared_settings(d):
    """Validated account settings: (places, cfg part). Same rules for the subscribe call and the synced user data."""
    places = []
    for p in (d.get("places") or [])[:MAX_PLACES]:
        places.append({"name": str(p.get("name") or "")[:60], "lat": round(float(p["lat"]), 2), "lon": round(float(p["lon"]), 2)})
    lead = int(d.get("lead", 45))
    cfg = {"lead": lead if 5 <= lead <= 90 and lead % 5 == 0 else 45, "early": bool(d.get("early", True)),
           "quiet": bool(d.get("quiet", True)), "quietFrom": minutes(d.get("quietFrom"), QUIET_FROM),
           "quietTo": minutes(d.get("quietTo"), QUIET_TO), "thunder": bool(d.get("thunder", True)),
           "own": bool(d.get("own", False))}
    return places, cfg


def level(rate):
    return next(l for l in LEVELS if rate >= l[0])


class AlertEngine:
    def __init__(self, data_dir, regions, log, ua, subject, order=("eu", "us"), thunder=None, kind_at=None):
        self.dir, self.regions, self.log, self.ua, self.subject = data_dir, regions, log, ua, subject
        self.thunder = thunder  # thunder.Thunder (DWD thunderstorm warnings) or None
        self.kind_at = kind_at  # (lat, lon, ts) -> "rain" | "snow" | "sleet" | "freezing" | "graupel" | "hail"
        self.dev, self.outbox = False, None  # development: "dev:" subscriptions go to this list instead of a push service
        self.pool = ThreadPoolExecutor(PUSH_WORKERS)
        self.rate_lock, self.next_slot = threading.Lock(), 0.0  # at most PUSH_PER_S messages per second
        self.order = order  # radar sources tried for a place, in this order
        os.makedirs(data_dir, exist_ok=True)
        self.path = os.path.join(data_dir, "subscriptions.json")
        self.lock = threading.Lock()
        try:
            with open(self.path, encoding="utf-8") as f:
                self.subs = json.load(f)
        except (OSError, ValueError):
            self.subs = {}
        self.vapid, self.public_key = self._vapid()
        self.session_alive = None  # token hash -> bool (auth.py), to drop subscriptions of ended sessions
        self.last_sweep = 0.0

    # ---------- keys and storage ----------
    def _vapid(self):
        try:
            from py_vapid import Vapid01
            from py_vapid.utils import b64urlencode
            from cryptography.hazmat.primitives import serialization
        except ImportError:
            return None, None
        pem = os.path.join(self.dir, "vapid_private.pem")
        if os.path.exists(pem):
            v = Vapid01.from_file(pem)
        else:
            v = Vapid01()
            v.generate_keys()
            v.save_key(pem)
        raw = v.public_key.public_bytes(serialization.Encoding.X962, serialization.PublicFormat.UncompressedPoint)
        return v, b64urlencode(raw)

    def _save(self):
        tmp = self.path + ".tmp"
        with open(tmp, "w", encoding="utf-8") as f:
            json.dump(self.subs, f)
        os.replace(tmp, self.path)

    @staticmethod
    def _effective(e):
        """Places checked for a subscription: the device's own location (if switched on) + the account's places."""
        own = e.get("ownLoc") if e["cfg"].get("own") else None
        return ([{"name": e.get("ownName") or "", **own}] if own else []) + e.get("base", [])

    @staticmethod
    def _keep_state(e):
        old = e.get("state", {})
        e["state"] = {f"{p['lat']},{p['lon']}": old.get(f"{p['lat']},{p['lon']}", {}) for p in e["places"]}

    def subscribe(self, body, user, sess):
        """Create or update the subscription of one device of a logged-in user (user id, session token hash).
        body: {id?, subscription, settings: {places, own, lead, early, quiet, quietFrom, quietTo, thunder},
        ownLoc?: {lat, lon}, ownName?, tz, api}"""
        if user is None:
            raise PermissionError("login required")  # no new anonymous subscriptions (see the module docstring)
        sub = body.get("subscription") or {}
        dev = self.dev and str(sub.get("endpoint", "")).startswith("dev:")  # devtools.py test subscription
        if not (str(sub.get("endpoint", "")).startswith("https://") or dev) or not sub.get("keys"):
            raise ValueError("invalid push subscription")
        base, shared = shared_settings(body.get("settings") or {})
        own = body.get("ownLoc")
        own = {"lat": round(float(own["lat"]), 2), "lon": round(float(own["lon"]), 2)} if isinstance(own, dict) else None
        try:
            ZoneInfo(str(body.get("tz")))
            tz = str(body.get("tz"))
        except Exception:
            tz = "UTC"
        api = str(body.get("api") or "")
        lang = str(body.get("lang") or "de")
        cfg = {**shared, "tz": tz, "lang": lang if lang in TEXT else "de",
               "api": api if api.startswith(("https://", "http://localhost", "http://127.0.0.1")) else ""}
        with self.lock:
            old = self.subs.get(body.get("id"))
            # an old anonymous subscription of this device is taken over by the account; never another user's
            sid = body["id"] if old is not None and old.get("user") in (None, user) else secrets.token_urlsafe(16)
            old = self.subs.get(sid, {})
            e = {"sub": {"endpoint": sub["endpoint"], "keys": sub["keys"]}, "user": user, "sess": sess, "base": base,
                 "ownLoc": own, "ownName": str(body.get("ownName") or "")[:60], "cfg": cfg,
                 "state": old.get("state", {}), "reminders": old.get("reminders", [])}
            e["places"] = self._effective(e)
            self._keep_state(e)
            self.subs[sid] = e
            self._apply(user, base, shared, skip=sid)  # the other devices of the user get the same settings
            self._save()
        return sid

    def _apply(self, user, base, shared, skip=None):
        for sid, e in self.subs.items():
            if e.get("user") != user or sid == skip:
                continue
            e["base"], e["cfg"] = base, {**e["cfg"], **shared}
            e["places"] = self._effective(e)
            self._keep_state(e)

    def apply_user(self, user, settings):
        """The account's alert settings changed (synced user data saved by any device): all devices follow."""
        if not isinstance(settings, dict):
            return
        try:
            base, shared = shared_settings(settings)
        except (KeyError, TypeError, ValueError, AttributeError):
            return
        with self.lock:
            if any(e.get("user") == user for e in self.subs.values()):
                self._apply(user, base, shared)
                self._save()

    def remove(self, match):
        """Remove the subscriptions for which match(entry) is true (logout, account deleted, alerts off)."""
        with self.lock:
            gone = [sid for sid, e in self.subs.items() if match(e)]
            for sid in gone:
                del self.subs[sid]
            if gone:
                self._save()
        return len(gone)

    def remove_user(self, user):
        return self.remove(lambda e: e.get("user") == user)

    def remove_session(self, sess):
        return self.remove(lambda e: e.get("sess") == sess)

    def sweep(self):
        """Subscriptions of sessions that ended without a logout (expired, deleted on the server) go as well."""
        if self.session_alive is None or time.time() - self.last_sweep < SWEEP_S:
            return
        self.last_sweep = time.time()
        with self.lock:
            sessions = {e["sess"] for e in self.subs.values() if e.get("sess")}
        dead = {x for x in sessions if not self.session_alive(x)}
        if dead:
            n = self.remove(lambda e: e.get("sess") in dead)
            self.log(f"alerts: {n} subscriptions of ended sessions removed")

    def remind(self, sid, body):
        """'Remind me again in N min' from the notification button: one more look at the place later."""
        minutes = int(body.get("minutes", 10))
        with self.lock:
            e = self.subs.get(sid)
            if e is None or not 5 <= minutes <= 60:
                return False
            p = {"name": str(body.get("name") or "")[:60], "lat": round(float(body["lat"]), 2), "lon": round(float(body["lon"]), 2)}
            e.setdefault("reminders", []).append({**p, "due": int(time.time() * 1000) + minutes * 60_000})
            del e["reminders"][:-MAX_REMINDERS]
            self._save()
        return True

    def unsubscribe(self, sid):
        with self.lock:
            if self.subs.pop(sid, None) is not None:
                self._save()

    # ---------- forecast at a place ----------
    def series(self, lat, lon, ll_to_3857):
        """Rain series around a place: [(ts, is_forecast, wet share of the 5 km circle, rate at the place)] or None."""
        x, y = ll_to_3857.transform(lon, lat)
        for name in self.order:
            region = self.regions.get(name)
            if region is None:
                continue
            col, row = (int(round(float(v))) for v in region.to_pixels(x, y))
            frames, ref = region.frames()
            if not frames or ref is None:
                continue
            g0 = region.grid(frames[-1]["ts"])
            ny, nx = g0.shape
            r = max(1, int(round(RADIUS_KM / region.KM_PER_PX)))
            if not (r <= col < nx - r and r <= row < ny - r):
                continue
            yy, xx = np.mgrid[-r:r + 1, -r:r + 1]
            disc = (xx ** 2 + yy ** 2) <= r * r
            if (g0[row - r:row + r + 1, col - r:col + r + 1][disc] < 0).mean() > 0.5:
                continue  # mostly outside radar coverage
            out = []
            for fr in frames:
                g = region.grid(fr["ts"])
                if g is None:
                    continue
                win = g[row - r:row + r + 1, col - r:col + r + 1].astype(np.float32)[disc]
                win = win[win >= 0]
                if not len(win):
                    continue
                cell = g[row - 1:row + 2, col - 1:col + 2].astype(np.float32)
                cell = cell[cell >= 0]
                rate = max(float(cell.mean()) if len(cell) else 0.0, float(np.percentile(win, 75)))
                out.append((fr["ts"], fr["fc"], float((win >= WET).mean()), rate))
            return out
        return None

    @staticmethod
    def evaluate(series, now_ms, cfg, st):
        """Decide what to send for one place. Returns (kind, info) or (None, info); updates the state dict st."""
        obs = [s for s in series if not s[1]]
        fc = [s for s in series if s[1]]
        info = {}
        # remember when it last rained here (observed)
        for ts, _, p, _ in obs:
            if p >= P_START * 0.6:
                st["last_wet"] = max(st.get("last_wet", 0), ts)
        raining_now = bool(obs) and obs[-1][2] >= P_START * 0.6
        # first forecast step where rain sets in and lasts at least MIN_STEPS
        start = next((i for i in range(len(fc) - MIN_STEPS + 1) if all(fc[i + k][2] >= P_START for k in range(MIN_STEPS))), None)
        if start is None:
            return None, {"raining_now": raining_now}
        end = next((i for i in range(start, len(fc)) if fc[i][2] < P_START * 0.6), len(fc))
        event = fc[start:end]
        peak = max(s[3] for s in event)
        lead = (fc[start][0] - now_ms) / 60_000
        lv = level(peak)
        info = {"start_ts": fc[start][0], "lead_min": round(lead), "minutes": (end - start) * 5, "open_end": end == len(fc),
                "prob": probability(lead, peak),
                "peak": round(peak, 1), "level": lv[2], "certainty": round(min(s[2] for s in fc[start:start + MIN_STEPS]), 2),
                "raining_now": raining_now}
        if raining_now:
            return None, info
        same_event = abs(st.get("event_ts", 0) - fc[start][0]) <= 30 * 60_000
        dry_long = now_ms - st.get("last_wet", 0) >= COOLDOWN_H * 3600_000
        tz = ZoneInfo(cfg.get("tz", "UTC"))
        local = datetime.fromtimestamp(now_ms / 1000, tz)
        day = local.strftime("%Y-%m-%d")
        if st.get("day") != day:
            st["day"], st["count"] = day, 0
        if st["count"] >= MAX_PER_DAY or quiet_now(cfg, local):
            return None, info
        kind = None
        if same_event and st.get("sent_level") and peak >= 8.0 and st.get("sent_peak", 0) < 8.0 and not st.get("upgraded") and lead <= cfg["lead"] + 10:
            kind = "upgrade"
        elif not same_event or not st.get("main"):
            if (not same_event and not dry_long):
                return None, info
            if lead <= cfg["lead"] + 2.5:
                kind = "main"
            elif cfg.get("early", True) and cfg["lead"] < 45 and peak >= 8.0 and 45 <= lead <= 65 and info["certainty"] >= P_EARLY and not (same_event and st.get("early")):
                kind = "early"
        if kind:
            if not same_event:
                st.update({"event_ts": fc[start][0], "main": False, "early": False, "upgraded": False})
            st[kind if kind != "upgrade" else "upgraded"] = True
            st["sent_level"], st["sent_peak"] = lv[2], max(peak, st.get("sent_peak", 0) if same_event else 0)
            st["count"] += 1
        return kind, info

    @staticmethod
    def message(kind, info, place, tz, lang="de"):
        T = TEXT.get(lang, TEXT["de"])
        when = datetime.fromtimestamp(info["start_ts"] / 1000, ZoneInfo(tz)).strftime("%H:%M")
        icon = {"snow": "🌨", "sleet": "🌨", "freezing": "🧊", "graupel": "🌨", "hail": "⛈"}.get(info.get("ptype"), ICON[info["level"]])
        lv = T["levels"].get(info["level"], info["level"])
        dur = T["long"] if info["open_end"] else T["minutes"].format(m=max(10, info["minutes"]))
        mins = max(1, info["lead_min"])
        word = T["kinds"].get(info.get("ptype"))
        if word:  # same strength levels, other precipitation: "Starker Schnee" / "Heavy snow", ...
            lv = lv.replace(T["rain_word"].capitalize(), word.capitalize()).replace(T["rain_word"], word)
        chance = T["chance"].format(p=info["prob"]) if info.get("prob") else ""
        title_f, body_f = T[kind if kind in ("early", "upgrade", "reminder") else "main"]
        fill = {"icon": icon, "lv": lv, "place": place, "when": when, "dur": dur, "chance": chance, "mins": mins}
        return title_f.format(**fill), body_f.format(**fill)

    # ---------- sending ----------
    def _wait_slot(self):
        with self.rate_lock:
            now = time.monotonic()
            slot = max(now, self.next_slot)
            self.next_slot = slot + 1.0 / PUSH_PER_S
        if slot > now:
            time.sleep(slot - now)

    def send_many(self, batch):
        """Send [(urgency, sid, payload)] in parallel, most urgent first (shortest time until the weather), rate-limited."""
        for _, sid, payload in sorted(batch, key=lambda b: b[0]):
            self.pool.submit(lambda s=sid, p=payload: (self._wait_slot(), self.send(s, p)))

    def send(self, sid, payload):
        try:
            from pywebpush import WebPushException, webpush
        except ImportError:
            return False
        entry = self.subs.get(sid)
        if entry and self.dev and entry["sub"]["endpoint"].startswith("dev:"):
            self.outbox.append({"sid": sid, **payload})
            return True
        if not entry or not self.vapid:
            return False
        try:
            webpush(entry["sub"], json.dumps(payload), vapid_private_key=self.vapid,
                    vapid_claims={"sub": self.subject}, headers={"User-Agent": self.ua}, ttl=900)
            return True
        except WebPushException as e:
            if e.response is not None and e.response.status_code in (404, 410):  # subscription gone
                self.unsubscribe(sid)
            self.log(f"push failed: {e}")
            return False
        except Exception as e:  # broken keys etc.: never let one subscription stop the others
            self.log(f"push failed: {e}")
            return False

    def run_once(self, ll_to_3857):
        now = int(time.time() * 1000)
        with self.lock:
            items = [(sid, json.loads(json.dumps(e))) for sid, e in self.subs.items()]
        changed = False
        batch = []  # (urgency in minutes, sid, payload): sent after the round, most urgent first
        for sid, e in items:
            for p in e["places"]:
                key = f"{p['lat']},{p['lon']}"
                st = e["state"].setdefault(key, {})
                before = json.dumps(st, sort_keys=True)
                thunder = self.thunder_alert(p, now, e["cfg"], st)
                if thunder:
                    batch.append((-1, sid, {**thunder, "tag": f"t{key}", "lat": p["lat"], "lon": p["lon"], "name": p["name"]}))
                    self.log(f"alert thunder: {thunder['title']}")
                    changed = True
                s = self.series(p["lat"], p["lon"], ll_to_3857)
                if not s:
                    continue
                kind, info = self.evaluate(s, now, e["cfg"], st)
                if kind:
                    batch.append((info.get("lead_min", 0), sid, self.payload(sid, e["cfg"], kind, info, p, key)))
                    self.log(f"alert {kind}: {info.get('level')} in {info.get('lead_min')} min")
                if json.dumps(st, sort_keys=True) != before:
                    changed = True
            with self.lock:
                if sid in self.subs:
                    self.subs[sid]["state"] = e["state"]
            for rem in [r for r in e.get("reminders", []) if r["due"] <= now]:
                s = self.series(rem["lat"], rem["lon"], ll_to_3857)
                info = self.evaluate(s, now, e["cfg"], {})[1] if s else {}
                key = f"{rem['lat']},{rem['lon']}"
                if info.get("start_ts") and not info.get("raining_now"):
                    batch.append((info.get("lead_min", 0), sid, self.payload(sid, e["cfg"], "reminder", info, rem, key)))
                elif info.get("raining_now"):
                    T = text(e["cfg"])
                    batch.append((0, sid, {"title": T["raining"][0].format(place=rem["name"] or T["your_place"]),
                                           "body": T["raining"][1], "tag": key}))
                else:
                    T = text(e["cfg"])
                    batch.append((60, sid, {"title": T["allclear"][0].format(place=rem["name"] or T["your_place"]),
                                            "body": T["allclear"][1], "tag": key}))
                changed = True
            with self.lock:
                if sid in self.subs:
                    self.subs[sid]["reminders"] = [r for r in self.subs[sid].get("reminders", []) if r["due"] > now]
        if changed:
            with self.lock:
                self._save()
        if self.dev:  # dry-run outbox: synchronous, so tests see the messages at once
            for _, sid, payload in sorted(batch, key=lambda b: b[0]):
                self.send(sid, payload)
        else:
            self.send_many(batch)

    def payload(self, sid, cfg, kind, info, place, key):
        """Push payload: text, the place, and the data for the 'remind me' button (service worker)."""
        if self.kind_at:  # snow, sleet, ... instead of "Regen"
            try:
                info["ptype"] = self.kind_at(place["lat"], place["lon"], info["start_ts"])
            except Exception:
                pass
        T = text(cfg)
        title, body = self.message(kind, info, place["name"] or T["your_place"], cfg["tz"], cfg.get("lang", "de"))
        d = {"title": title, "body": body, "tag": key, "lat": place["lat"], "lon": place["lon"], "name": place["name"]}
        wait = remind_minutes(info.get("lead_min", 0))
        if wait and cfg.get("api"):
            d.update({"id": sid, "api": cfg["api"], "remind": wait})
        return d

    def thunder_alert(self, place, now_ms, cfg, st):
        """One push per DWD thunderstorm warning covering the place (active now or within the hour)."""
        if not self.thunder or not cfg.get("thunder", True):
            return None
        sent = st.setdefault("thunder_sent", [])
        for w in self.thunder.at(place["lat"], place["lon"], now_ms):
            if w["group"] != "Gewitter" or w["id"] in sent:
                continue
            tz = ZoneInfo(cfg.get("tz", "UTC"))
            local = datetime.fromtimestamp(now_ms / 1000, tz)
            if quiet_now(cfg, local) and w["severity"] in ("Minor", "Moderate"):
                continue  # in the quiet hours only severe thunderstorms wake people up
            sent.append(w["id"])
            del sent[:-30]
            hm = lambda t: datetime.fromtimestamp(t / 1000, tz).strftime("%H:%M")
            T = text(cfg)  # the warning level itself comes from the DWD in German
            when = (T["thunder_now"].format(to=hm(w["expires"])) if w["onset"] <= now_ms
                    else T["thunder_from"].format(fr=hm(w["onset"]), to=hm(w["expires"])))
            return {"title": f"⚡ {w['level']} – {place['name'] or T['your_place']}", "body": T["thunder_body"].format(when=when)}
        return None

    def loop(self, ll_to_3857, every_s=60):
        while True:
            try:
                self.sweep()
                self.run_once(ll_to_3857)
            except Exception as e:
                self.log(f"alerts failed: {e}")
            time.sleep(every_s)
