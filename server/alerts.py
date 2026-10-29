"""Rain alerts ("Regenwarnung") for subscribed places, sent as Web Push notifications.

Rules (agreed design):
  - main alert about LEAD minutes (5-90, default 45) before rain starts at a place
  - early alert ~60 min ahead only for heavy rain (>= 8 mm/h), only when the forecast is very certain and only with
    a main warning time below 45 min (otherwise it adds nothing)
  - only on the change from dry to wet: no alert while it rains, and after rain a cooldown of COOLDOWN_H dry hours
  - at most MAX_PER_DAY alerts per place and day, quiet hours 22-7 (optional), one upgrade if it gets heavier
  - "rain at the place" is judged over a ~5 km circle (share of wet pixels), which removes most false alarms
    from showers that pass just beside the place
Privacy: the server stores only the push endpoint (from the browser's push service), place names and coordinates
rounded to ~1 km, and the settings. No account, no device or person identifiers.
"""
import json
import math
import os
import secrets
import threading
import time
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
MAX_PLACES = 20


def level(rate):
    return next(l for l in LEVELS if rate >= l[0])


class AlertEngine:
    def __init__(self, data_dir, regions, log, ua, subject, order=("eu", "us"), thunder=None, kind_at=None):
        self.dir, self.regions, self.log, self.ua, self.subject = data_dir, regions, log, ua, subject
        self.thunder = thunder  # thunder.Thunder (DWD thunderstorm warnings) or None
        self.kind_at = kind_at  # (lat, lon, ts) -> "rain" | "snow" | "sleet" | "freezing" | "graupel" | "hail"
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

    def subscribe(self, body):
        sub = body.get("subscription") or {}
        if not str(sub.get("endpoint", "")).startswith("https://") or not sub.get("keys"):
            raise ValueError("invalid push subscription")
        places = []
        for p in (body.get("places") or [])[:MAX_PLACES]:
            places.append({"name": str(p.get("name") or "")[:60], "lat": round(float(p["lat"]), 2), "lon": round(float(p["lon"]), 2)})
        try:
            ZoneInfo(str(body.get("tz")))
            tz = str(body.get("tz"))
        except Exception:
            tz = "UTC"
        lead = int(body.get("lead", 45))
        cfg = {"lead": lead if 5 <= lead <= 90 and lead % 5 == 0 else 45,
               "early": bool(body.get("early", True)), "quiet": bool(body.get("quiet", True)), "tz": tz,
               "thunder": bool(body.get("thunder", True))}
        with self.lock:
            sid = body.get("id") if body.get("id") in self.subs else secrets.token_urlsafe(16)
            old = self.subs.get(sid, {}).get("state", {})
            state = {f"{p['lat']},{p['lon']}": old.get(f"{p['lat']},{p['lon']}", {}) for p in places}
            self.subs[sid] = {"sub": {"endpoint": sub["endpoint"], "keys": sub["keys"]}, "places": places, "cfg": cfg, "state": state}
            self._save()
        return sid

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
        if st["count"] >= MAX_PER_DAY or (cfg.get("quiet", True) and (local.hour >= 22 or local.hour < 7)):
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
    def message(kind, info, place, tz):
        when = datetime.fromtimestamp(info["start_ts"] / 1000, ZoneInfo(tz)).strftime("%H:%M")
        lv = info["level"]
        dur = "länger anhaltend" if info["open_end"] else f"etwa {max(10, info['minutes'])} Min. lang"
        mins = max(1, info["lead_min"])
        word = {"snow": "Schnee", "sleet": "Schneeregen", "freezing": "Glatteisregen", "graupel": "Graupel", "hail": "Hagel"}.get(info.get("ptype"))
        if word:  # same strength levels, other precipitation: "Starker Schnee", "Leichter Schneeregen", ...
            lv = lv.replace("Regen", word).replace("regen", word.lower())
        if kind == "early":
            title = f"{ICON[lv]} {lv} in ca. 1 Std. – {place}"
            body = f"Voraussichtlich ab ca. {when}, {dur}. Wir melden uns kurz vorher noch einmal."
        elif kind == "upgrade":
            title = f"⛈ Jetzt starker Regen erwartet – {place}"
            body = f"Ab ca. {when}, {dur}."
        else:
            title = f"{ICON[lv]} {lv} in {mins} Min. – {place}"
            body = f"Ab ca. {when}, {dur}."
        return title, body

    # ---------- sending ----------
    def send(self, sid, payload):
        try:
            from pywebpush import WebPushException, webpush
        except ImportError:
            return False
        entry = self.subs.get(sid)
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
        for sid, e in items:
            for p in e["places"]:
                key = f"{p['lat']},{p['lon']}"
                st = e["state"].setdefault(key, {})
                before = json.dumps(st, sort_keys=True)
                thunder = self.thunder_alert(p, now, e["cfg"], st)
                if thunder:
                    self.send(sid, {**thunder, "tag": f"t{key}", "lat": p["lat"], "lon": p["lon"], "name": p["name"]})
                    self.log(f"alert thunder: {thunder['title']}")
                    changed = True
                s = self.series(p["lat"], p["lon"], ll_to_3857)
                if not s:
                    continue
                kind, info = self.evaluate(s, now, e["cfg"], st)
                if kind:
                    if self.kind_at:  # snow, sleet, ... instead of "Regen"
                        try:
                            info["ptype"] = self.kind_at(p["lat"], p["lon"], info["start_ts"])
                        except Exception:
                            pass
                    title, body = self.message(kind, info, p["name"] or "dein Ort", e["cfg"]["tz"])
                    self.send(sid, {"title": title, "body": body, "tag": key, "lat": p["lat"], "lon": p["lon"], "name": p["name"]})
                    self.log(f"alert {kind}: {info.get('level')} in {info.get('lead_min')} min")
                if json.dumps(st, sort_keys=True) != before:
                    changed = True
            with self.lock:
                if sid in self.subs:
                    self.subs[sid]["state"] = e["state"]
        if changed:
            with self.lock:
                self._save()

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
            if cfg.get("quiet", True) and (local.hour >= 22 or local.hour < 7) and w["severity"] in ("Minor", "Moderate"):
                continue  # at night only severe thunderstorms wake people up
            sent.append(w["id"])
            del sent[:-30]
            hm = lambda t: datetime.fromtimestamp(t / 1000, tz).strftime("%H:%M")
            when = f"Jetzt bis ca. {hm(w['expires'])}" if w["onset"] <= now_ms else f"Ab ca. {hm(w['onset'])} bis ca. {hm(w['expires'])}"
            return {"title": f"⚡ {w['level']} – {place['name'] or 'dein Ort'}", "body": f"{when}. Warnung des Deutschen Wetterdienstes."}
        return None

    def loop(self, ll_to_3857, every_s=60):
        while True:
            try:
                self.run_once(ll_to_3857)
            except Exception as e:
                self.log(f"alerts failed: {e}")
            time.sleep(every_s)
