"""Official warnings against our own data: does our radar (and our 0-120 min forecast) show what the weather services
warn about - and where it does not, who is wrong? Rain gauges are the referee.

User, 18.11.2026: the app no longer shows the official warnings; they are used to CHECK OUR OWN DATA instead - our
rain clouds should already show severe weather, and where they do not, find out whether our data or the warning is
wrong.

Two parts:

1. Recorder (inside the instances, started by rainy_server.warnings_start; light: a few KB per cycle, no new
   downloads - it reads what the instance already holds)
     <VERIFY_LOG_DIR>/warnarchive/<source>_<YYYYMMDD>.jsonl   (day = when the line was written, UTC)
       {"k": "w", ...}  each distinct precipitation-related warning ONCE (events rain, thunderstorm, snow-ice, flood;
                        wind / fog / heat ... are not kept): source, id, short hash "h", country, event, level, names
                        (de / en), sent, onset, expires, first seen, English (else first) text, geometry keys "g"
       {"k": "g", ...}  each distinct area geometry ONCE (content hash "gk"; [[outer, hole, ...], ...] lon/lat, 3 dec.)
       {"k": "s", ...}  every cycle (10 min): the hashes present in the feed now - a warning that disappears before
                        its expiry (cancelled, replaced by an update with a new id) ends there
       {"k": "v", ...}  DWD nowcast warnings only: a new geometry / expiry of the same warning (they move with the cell)
     sources: de instance "dwd" (DWD CAP), "dwdnow" (DWD NowCastMIX Autowarn thunderstorm / heavy rain, thunder.py) and
              "lightning" ({"k": "l", "t", "pts": [[lat, lon], ...]} MTG flash clusters, every 5 min);
              eu instance "meteoalarm"; na instance (VERIFY_LOG_DIR = verify_log/na) "nws", "eccc".
     <VERIFY_LOG_DIR>/warnfc_<de|eu|na>/<ref ms>.npz   our forecast of every run at :00/:15/:30/:45, on the grid of
       the analysis archive (obs3km / obs_eu, the same transform as fcverify): r30, r60, r120 = rate (mm/h) at
       that lead; a60 / a120 = rain (mm) of the hours (ref, ref+60] and (ref+60, ref+120]. uint8 code
       32 * log2(1 + v) (~2 % steps; < 0.05 -> 0). Kept FC_KEEP_DAYS; Europe on a rainy day ~0.3 MB per run
       (~30 MB a day), Germany ~30 KB.

2. Check (daily batch, outside the instances - ~0.3-0.6 GB for a Europe / North America day; run it in a throwaway
   container with -m 1g, see "Run" below). Per domain and UTC day:
     de  DWD radar analysis (verify_log/obs3km, ~4.5 km, 5 min) vs DWD CAP + DWD nowcast warnings + MTG lightning,
         gauges: DWD 10-minute gauges (gaugetruth "de:*")
     eu  OPERA analysis (verify_log/obs_eu, 4 km, 5 min) vs MeteoAlarm (all countries but Germany - Germany is "de"),
         gauges: the Europe gauge archive (verify_log/gaugetruth, ~6000 gauges of 24 networks)
     eun the same against our own Europe picture (OPERA + national radars, verify_log/eun/obs_eu; no forecast archive)
     na  MRMS analysis (verify_log/na/obs_eu, 0.04 deg) vs NWS + ECCC, gauges: verify_log/na/gaugetruth (US ASOS
         5-minute reports, Canadian SWOB minute records)
   For every warning area (polygon, onset, effective end, level, event) and every 15-min step from 60 min before
   onset to 30 min after the end: max / 95th percentile / "robust max" (max of the 3x3 median - one hot pixel of
   clutter does not count) of the rain rate and of the sliding 1-hour rain inside the area; for criteria over 6, 12,
   24 h the same for rain sums over hour-aligned windows; our forecast valid at that step issued 30 / 60 / 120 min
   before (rate, and the 1-hour rain where the forecast archive has it); lightning clusters (Germany).
   Rows -> verify_log/warncheck_<YYYYMMDD>.jsonl ("step", "warn" per warning, "own" per unwarned case).

Classes (ratio = the best criterion's robust max / its threshold):
   hit                 ratio >= HIT (0.7: radar rates are uncertain by tens of percent), or lightning in a warned
                       thunderstorm area (Germany)
   partial             PARTIAL (0.4) <= ratio < HIT
   miss_radar_low      ratio < PARTIAL, but the gauges in / near the area (2 cells around it) measured >= HIT x the
                       threshold, or >= PARTIAL x and at least twice what our radar shows at the same gauges:
                       OUR DATA IS WRONG (under-reads)
   miss_not_borne      ratio < PARTIAL and the gauges stayed low too: the warning was not borne out (or precautionary,
                       or the rain fell between the gauges - see the gauge count)
   miss_no_gauges      ratio < PARTIAL, no gauge with data in / near the area
   not checked         criteria not about precipitation amounts (river floods, glaze, drifting snow, pre-information
                       and level-1 statements, NWS watches) or the area is outside our radar
   own_real / own_artefact / own_unverified
                       our analysis exceeds OWN_T mm in 1 h (15 = DWD "Starkregen"; North America 25, ~flash-flood
                       guidance) inside a warning service's land area with no precipitation warning of any level within
                       ~9 km and +-60 min: real when a gauge within ~9 km measured >= 0.5 x OWN_T in 1 h or >= half of
                       what our radar shows at that gauge; artefact when the peak pixel had >= 5 mm/h in >= 25 % of the
                       day's images (static clutter) or gauges measured < 20 % of our radar at the gauge; else
                       unverified
Flood warnings: the rain of the 3 h before onset counts too (PRE_FLOOD - the rain that caused the flood); river
floods days after the rain remain out of reach.
Updates: a warning updated with a new id (same source, name, level, areas, windows touching) is one warning.
Lead time (d), for warnings whose criterion is a 1-hour amount or a rate (not for 6-24 h amounts: a 2 h nowcast
cannot show them): t_obs = the first step at which our analysis reached HIT x the threshold; "shown at L" = our forecast
issued L min before (t_obs +- 15 min) reached HIT x the threshold too (the 1-hour forecast rain a60 / a120 where kept,
else the forecast rate as the proxy). Also reported: the official lead (t_obs - the warning's sent time).

Thresholds (sources; where a service's criteria are qualitative or vary by region an approximate proxy is used,
and every "warn" row says which: "thr" = dwd | text | table | proxy):
   DWD (Germany): www.dwd.de/DE/wetter/warnungen_aktuell/kriterien/warnkriterien.html (read 18.11.2026)
     Starkregen 15-25 l/m2 in 1 h or 20-35 in 6 h; Heftiger Starkregen 25-40 / 1 h or 35-60 / 6 h; Extrem heftiger
     Starkregen > 40 / 1 h or > 60 / 6 h. Dauerregen 25-40 / 12 h, 30-50 / 24 h (48 / 72 h not checked); Ergiebiger
     Dauerregen 40-70 / 12 h, 50-80 / 24 h; Extrem ergiebiger > 70 / 12 h, > 80 / 24 h. Schneefall (Flachland)
     5-10 cm / 6 h, 10-15 / 12 h; Starker Schneefall 10-20 / 6 h, 15-25 / 12 h; Extrem starker > 20 / 6 h, > 25 / 12 h;
     Leichter Schneefall: any snow (proxy 1 mm / 6 h). Snow as water: 1 cm ~ 1 mm (10:1 - a proxy, and radar and
     gauges both under-read snow). Gewitter: "elektrische Entladung" -> lightning, else a rate proxy; thunderstorms
     "mit (heftigem / extrem heftigem) Starkregen" use the Starkregen amounts.
     NowCastMIX (Autowarn) Starkregen: Moderate 15 mm / 1 h, Severe 25, Extreme 40 (same scale).
   Thunderstorm rate proxy (all services): robust max rate >= 10 / 15 / 25 mm/h for level 2 / 3 / 4 (a thunderstorm
     core is >= ~40 dBZ, Marshall-Palmer ~12 mm/h; strong / severe storms carry Starkregen rates) - a proxy.
   MeteoAlarm: the awareness levels are defined qualitatively (meteoalarm.org, "awareness levels") and thresholds are
     national and regional. Amounts given in the warning text are used first ("One-hour accumulated precipitation:
     15 mm" AEMET; "from 10 to 30 mm in 12 hours" HNMS; "from 25 to 50 mm/24h" ARSO; lower bound = threshold);
     otherwise the table MA_RAIN, roughly the band of the DWD and AEMET (Meteoalerta plan) criteria:
     yellow 15 / 1 h, 25 / 6 h, 30 / 12 h, 40 / 24 h; orange 25, 40, 50, 70; red 40, 60, 80, 100. Snow (events naming
     snow) MA_SNOW as water. Floods (hydrological), avalanches, ice: not checked.
   NWS: NWSI 10-511 / 10-513. Flash Flood Warning: no fixed amount (flash flood guidance, typically 1-3 in / 1-3 h)
     -> proxy 25 mm / 1 h or 50 mm / 3 h; Flood Advisory for urban / small streams -> proxy 12 mm / 1 h; Severe
     Thunderstorm / Tornado Warning: hail / wind / rotation, not rain -> rate proxy; Winter Storm Warning 6 in / 12 h
     (~15 mm water), Winter Weather Advisory ~2-5 in (proxy 5 mm / 12 h), Lake Effect Snow Warning as Winter Storm,
     Ice Storm Warning >= 1/4 in ice (proxy 6 mm / 12 h); "rainfall rates of X inches per hour" and "snow
     accumulations of X inches" in the text take precedence. Watches, statements, river floods: not checked.
   ECCC: public alerting criteria (canada.ca, "Criteria for public weather alerts"): rainfall warning 50 mm / 24 h
     (25 mm / 1 h in summer in places), snowfall warning 15 cm / 12 h (varies), freezing rain warning (proxy 2 mm /
     6 h), severe thunderstorm warning (rate proxy, criterion 50 mm / h of rain or hail / wind); text amounts
     ("rainfall amounts of 50 to 70 mm", "snowfall amounts of 15 to 25 cm") take precedence.

Run (daily, on the host, e.g. cron 40 2 * * *; the image already holds this file once deployed; ~1-2 min, < 0.4 GB):
   docker run --rm -m 1g --entrypoint python -v deploy_verify-log:/app/verify_log -v deploy_eu-data:/data/eu:ro        -v deploy_na-data:/data/na:ro -w /app deploy-api-eu warn_check.py check [--day YYYYMMDD] [--days 3]        [--domains de,eu,eun,na] [--extra /arch]
   (default: yesterday; writes verify_log/warncheck_<day>.jsonl - a re-run replaces that day's rows of the domain -
   and prints the summary; "summary --days 7" prints it again from the files)
   Backfill: "backfill-dwd" (DWD's open CAP directory keeps ~2 days of complete snapshots), "backfill-nws --since
   2026-11-14T00:00:00Z" (api.weather.gov keeps the alerts of the last days). MeteoAlarm, ECCC and the DWD nowcast
   warnings have no open history: only what the Recorder archives.
   Before the Recorder runs in the instances: "collect --out DIR --cache DIR" runs it outside them (its own anonymous
   downloads of the warnings; our forecasts taken from fcverify's pending files, rates only) into DIR with the same
   layout (DIR/warnarchive, DIR/warnfc_de, DIR/na/...); "check --extra DIR" reads it next to the volume.
"""
import argparse
import glob
import hashlib
import json
import math
import os
import re
import sys
import threading
import time
import urllib.request
from collections import deque
from datetime import datetime, timedelta, timezone

import cv2
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

LOG_DIR = os.environ.get("VERIFY_LOG_DIR", os.path.join(os.path.dirname(os.path.abspath(__file__)), "verify_log"))
ARCH = "warnarchive"
PRECIP_EVENTS = ("rain", "thunderstorm", "snow-ice", "flood")
FC_LEADS = (30, 60, 120)           # minutes: rates kept per run
FC_EVERY_MS = 15 * 60_000          # one run per quarter hour is archived
FC_KEEP_DAYS = 14
CYCLE_S = 600                      # heartbeat of the warning feeds
KNOWN_DAYS = 3                     # warnings / geometries written in the last days are not written again
STEP_MS = 15 * 60_000
HIT, PARTIAL = 0.7, 0.4
PRE_FLOOD = 3 * 3600_000           # flood warnings: the rain of the 3 h before onset counts (it caused the flood)
OWN_T = {"de": 15.0, "eu": 15.0, "eun": 15.0, "na": 25.0}
TS_RATE = {2: 10.0, 3: 15.0, 4: 25.0}
GAUGE_DIL = 2                      # cells around an area / case in which gauges count (~8-9 km)
HOURLY_NETS = ("sy", "pt")         # gauge networks with hourly amounts in the 10-minute archive
UA = "Mozilla/5.0"

# ---------- thresholds ----------
DWD_NAMES = (  # (German CAP event name contains, [(hours, mm)]) - most specific first; see the module docstring
    ("EXTREM HEFTIGE", [(1, 40), (6, 60)]),     # "EXTREM HEFTIGER STARKREGEN", "... mit EXTREM HEFTIGEM STARKREGEN"
    ("HEFTIGE", [(1, 25), (6, 35)]),
    ("STARKREGEN", [(1, 15), (6, 20)]),
    ("EXTREM ERGIEBIGER DAUERREGEN", [(12, 70), (24, 80)]),
    ("ERGIEBIGER DAUERREGEN", [(12, 40), (24, 50)]),
    ("DAUERREGEN", [(12, 25), (24, 30)]),
    ("EXTREM STARKER SCHNEEFALL", [(6, 20), (12, 25)]),
    ("STARKER SCHNEEFALL", [(6, 10), (12, 15)]),
    ("LEICHTER SCHNEEFALL", [(6, 1)]),
    ("SCHNEEFALL", [(6, 5), (12, 10)]),
)
MA_RAIN = {2: [(1, 15), (6, 25), (12, 30), (24, 40)], 3: [(1, 25), (6, 40), (12, 50), (24, 70)],
           4: [(1, 40), (6, 60), (12, 80), (24, 100)]}
MA_SNOW = {2: [(12, 5), (24, 8)], 3: [(12, 15), (24, 20)], 4: [(12, 25), (24, 35)]}
NWS_NAMES = {"flash flood warning": [(1, 25), (3, 50)], "winter storm warning": [(12, 15)],
             "lake effect snow warning": [(12, 15)], "winter weather advisory": [(12, 5)],
             "ice storm warning": [(12, 6)], "heavy snow warning": [(12, 15)]}
NWS_RATE = ("severe thunderstorm warning", "tornado warning")
ECCC_NAMES = (("rainfall warning", [(24, 50)]), ("snowfall warning", [(12, 15)]), ("winter storm warning", [(12, 15)]),
              ("freezing rain warning", [(6, 2)]), ("snow squall warning", [(1, 2)]))

_NUM = r"(\d+(?:[.,]\d+)?)"
_WORDS = {"one": 1, "two": 2, "three": 3, "six": 6, "twelve": 12, "a": 1, "an": 1}
RX_ES = re.compile(r"\b(one|\d+)[\s-]*hours?\s+accumulated\s+precipitation\s*:\s*" + _NUM + r"\s*mm", re.I)
RX_MM = re.compile(r"(?:from\s+|between\s+|von\s+|of\s+|up\s+to\s+)?" + _NUM + r"\s*(?:(?:-|–|to|and|bis)\s*" + _NUM +
                   r"\s*)?(?:mm|l/m²|l/m2|litres?\s+per\s+square\s+met(?:re|er))\s*(?:/\s*|in\s+|within\s+|per\s+|over\s+)"
                   r"(?:the\s+(?:next|following)\s+|an?\s+)?(one|two|three|six|twelve|\d+)?\s*-?\s*(hours?|hrs?|h\b|stunden?|day)",
                   re.I)
RX_INCH_RATE = re.compile(r"rain(?:fall)?\s+rates?\s+(?:of\s+|up\s+to\s+|around\s+)?" + _NUM +
                          r"(?:\s*(?:to|-)\s*" + _NUM + r")?\s*inch(?:es)?\s+(?:per|an)\s+hour", re.I)
RX_SNOW = re.compile(r"snow(?:fall)?\s+(?:accumulations?|amounts?|totals?)\s+(?:of\s+|up\s+to\s+|near\s+|around\s+)?" + _NUM +
                     r"(?:\s*(?:to|-)\s*" + _NUM + r")?\s*(inch(?:es)?|cm|centimet)", re.I)
RX_RAIN_TOTAL = re.compile(r"rain(?:fall)?\s+(?:amounts?|totals?|accumulations?)\s+(?:of\s+|up\s+to\s+|near\s+|around\s+)?" +
                           _NUM + r"(?:\s*(?:to|-)\s*" + _NUM + r")?\s*mm", re.I)


def _f(s):
    return float(str(s).replace(",", "."))


def text_criteria(text):
    """Amounts stated in a warning text -> [(hours, mm)] (lower bound of a range = the threshold), [] if none."""
    t = " ".join(str(text or "").split())
    out = []
    for m in RX_ES.finditer(t):
        out.append((_WORDS.get(m.group(1).lower(), None) or int(m.group(1)), _f(m.group(2))))
    for m in RX_MM.finditer(t):
        unit = m.group(4).lower()
        h = 24 if unit == "day" else _WORDS.get((m.group(3) or "").lower()) or (int(m.group(3)) if m.group(3) else 1)
        out.append((h, _f(m.group(1))))
    for m in RX_INCH_RATE.finditer(t):
        out.append((1, round(_f(m.group(1)) * 25.4, 1)))
    for m in RX_SNOW.finditer(t):
        v = _f(m.group(1))
        out.append((12, round(v * 2.54 if m.group(3).lower().startswith("inch") else v, 1)))  # cm snow ~ mm water
    if not out:
        for m in RX_RAIN_TOTAL.finditer(t):
            out.append((24, _f(m.group(1))))
    return sorted({(int(h), float(v)) for h, v in out if 1 <= h <= 24 and 1 <= v <= 300})


def criteria(w):
    """-> {"kind": "acc", "crit": [(hours, mm)], "thr": source} | {"kind": "rate", "T": mm/h, "thr": ...} | None
    (with w["why"] set: not about precipitation amounts)."""
    src, ev, lv = w["src"], w["ev"], w.get("lv", 2)
    name = (w.get("name_de") or w.get("name") or "").upper()
    en = (w.get("name_en") or w.get("name") or "").lower()
    if lv <= 1:
        return None
    if src == "dwd":
        if name.startswith("VORABINFORMATION"):
            return None
        for key, crit in DWD_NAMES:
            if key in name and (ev != "thunderstorm" or "STARKREGEN" in name):
                return {"kind": "acc", "crit": crit, "thr": "dwd"}
        if ev == "thunderstorm":
            return {"kind": "rate", "T": TS_RATE.get(lv, 10.0), "thr": "proxy", "lightning": True}
        return None
    if src == "dwdnow":
        if w.get("group") == "Starkregen":
            return {"kind": "acc", "crit": [(1, {"Severe": 25, "Extreme": 40}.get(w.get("sev"), 15))], "thr": "dwd"}
        return {"kind": "rate", "T": TS_RATE.get(lv, 10.0), "thr": "proxy", "lightning": True}
    text = w.get("txt") or ""
    if src == "meteoalarm":
        if ev == "rain":
            c = text_criteria(text)
            return {"kind": "acc", "crit": c, "thr": "text"} if c else {"kind": "acc", "crit": MA_RAIN[min(lv, 4)], "thr": "table"}
        if ev == "thunderstorm":
            return {"kind": "rate", "T": TS_RATE.get(lv, 10.0), "thr": "proxy"}
        if ev == "snow-ice" and "snow" in en and not re.search(r"blow|drift", en):
            c = [x for x in text_criteria(text) if x[0] >= 6]
            return {"kind": "acc", "crit": c, "thr": "text"} if c else {"kind": "acc", "crit": MA_SNOW[min(lv, 4)], "thr": "table"}
        return None
    if src == "nws":
        if en in NWS_RATE:
            return {"kind": "rate", "T": TS_RATE.get(lv, 15.0), "thr": "proxy"}
        if en == "flood advisory" and not re.search(r"small stream|urban|heavy rain|rainfall", text, re.I):
            return None
        base = NWS_NAMES.get(en) or ([(1, 12)] if en == "flood advisory" else None)
        if base is None:
            return None
        c = text_criteria(text)
        c = [x for x in c if (x[0] >= 6) == (base[0][0] >= 6)]
        return {"kind": "acc", "crit": c, "thr": "text"} if c else {"kind": "acc", "crit": base, "thr": "proxy"}
    if src == "eccc":
        if "thunderstorm" in en and "warning" in en:
            return {"kind": "rate", "T": TS_RATE.get(lv, 15.0), "thr": "proxy"}
        for key, base in ECCC_NAMES:
            if key in en:
                c = [x for x in text_criteria(text) if (x[0] >= 6) == (base[0][0] >= 6)]
                return {"kind": "acc", "crit": c, "thr": "text"} if c else {"kind": "acc", "crit": base, "thr": "proxy"}
        return None
    return None


# ---------- small helpers ----------
def now_ms():
    return int(time.time() * 1000)


def day_of(ms):
    return datetime.fromtimestamp(ms / 1000, timezone.utc).strftime("%Y%m%d")


def day_ms(day):
    return int(datetime.strptime(day, "%Y%m%d").replace(tzinfo=timezone.utc).timestamp() * 1000)


def hstr(ms):
    return datetime.fromtimestamp(ms / 1000, timezone.utc).strftime("%d.%m %H:%M") if ms else "-"


def short_hash(s):
    return hashlib.sha1(s.encode("utf-8")).hexdigest()[:12]


def fetch(url, timeout=60, tries=3):
    """Anonymous: a neutral User-Agent, nothing identifying."""
    req = urllib.request.Request(url, headers={"User-Agent": UA})
    for attempt in range(tries):
        try:
            with urllib.request.urlopen(req, timeout=timeout) as r:
                return r.read()
        except OSError:
            if attempt == tries - 1:
                raise
            time.sleep(3 * (attempt + 1))


def compact_polys(shape):
    """official_warnings.Shape -> [[outer, hole, ...], ...] lon/lat lists, 3 decimals (~100 m)."""
    r = lambda a: np.round(np.asarray(a, np.float64), 3).tolist()
    return [[r(o)] + [r(h) for h in hs] for _, o, hs in shape.polys]


def pick(infos, lang):
    for i in infos or ():
        if str(i[0]).lower().startswith(lang):
            return i
    return None


# ---------- part 1: recorder ----------
class RegionFc:
    """Our forecast of a live region, on the analysis-archive grid (same transforms as fcverify)."""

    def __init__(self, region, name):
        self.r, self.name = region, name
        self.de = name == "de"

    def latest(self):
        r = self.r
        with r.lock:
            if self.de:
                return r.own_ref, dict(r.fc_own)
            return r.ref, dict(r.fc)

    def to_grid(self, g):
        if self.de:
            import fcverify
            return fcverify._down(g, (self.r.nx // fcverify.DOWN, self.r.ny // fcverify.DOWN))
        g = np.asarray(g, np.float32)
        d = getattr(self.r, "RAW_DOWN", 2)
        h, w = g.shape[0] // d, g.shape[1] // d
        return cv2.resize(np.where(g < 0, 0, g), (w, h), interpolation=cv2.INTER_AREA)


class PendingFc:
    """The same products from fcverify's pending files (<pending dir>/own~<ref>~<lead>.npz, already on the archive
    grid; they live until the lead has passed): rates only, no hourly sums. For running the Recorder outside the
    instances (collect)."""

    def __init__(self, pending_dir, name):
        self.dir, self.name = pending_dir, name

    def runs(self):
        out = {}
        for fp in glob.glob(os.path.join(self.dir, "own~*~*.npz")):
            try:
                _, ref, lead = os.path.basename(fp)[:-4].split("~")
                ref, lead = int(ref), int(lead)
            except ValueError:
                continue
            if ref % FC_EVERY_MS == 0 and lead in FC_LEADS:
                out.setdefault(ref, {})[lead] = fp
        return out


def _q(g):
    """mm/h or mm -> uint8 code 32 * log2(1 + v) (~2 % steps, 0 below 0.05, up to ~250): a fifth of float16 on disk."""
    g = np.asarray(g, np.float32)
    return np.where(g >= 0.05, np.clip(np.round(32.0 * np.log2(1.0 + np.maximum(g, 0))), 1, 255), 0).astype(np.uint8)


def _unq(q):
    q = np.asarray(q)
    return (np.exp2(q.astype(np.float32) / 32.0) - 1.0) if q.dtype == np.uint8 else q.astype(np.float32)


class Recorder:
    def __init__(self, log, warnings=None, thunder=None, fc=None, name="de", skip=(), log_dir=None, cycle_s=CYCLE_S):
        self.log, self.warnings, self.thunder, self.fc, self.name = log, warnings, thunder, fc, name
        self.skip = set(skip)
        self.root = log_dir or LOG_DIR
        self.dir = os.path.join(self.root, ARCH)
        self.fc_dir = os.path.join(self.root, f"warnfc_{name}")
        self.cycle_s = cycle_s
        self.known, self.geo_known, self.versions = set(), set(), {}
        self.last_cycle, self.last_lightning, self.fc_done = 0.0, None, set()
        self._load_known()

    def _load_known(self):
        for fp in glob.glob(os.path.join(self.dir, "*.jsonl")):
            try:
                if (datetime.now(timezone.utc) - datetime.strptime(fp[-14:-6], "%Y%m%d").replace(tzinfo=timezone.utc)).days > KNOWN_DAYS:
                    continue
                with open(fp, encoding="utf-8") as f:
                    for line in f:
                        if line.startswith('{"k": "w"') or line.startswith('{"k": "g"') or line.startswith('{"k": "v"'):
                            d = json.loads(line)
                            if d["k"] == "w":
                                self.known.add(d["h"])
                            elif d["k"] == "g":
                                self.geo_known.add(d["gk"])
                            else:
                                self.versions[d["h"]] = (d.get("g"), d.get("ex"))
            except (OSError, ValueError, KeyError):
                continue

    def _write(self, source, rows, t=None):
        if not rows:
            return
        os.makedirs(self.dir, exist_ok=True)
        with open(os.path.join(self.dir, f"{source}_{day_of(t or now_ms())}.jsonl"), "a", encoding="utf-8") as f:
            for r in rows:
                f.write(json.dumps(r, ensure_ascii=False, separators=(", ", ": ")) + "\n")

    def _geo(self, polys, rows):
        gk = short_hash(json.dumps(polys))
        if gk not in self.geo_known:
            self.geo_known.add(gk)
            rows.append({"k": "g", "gk": gk, "p": polys})
        return gk

    # warnings of official_warnings.Warnings: one State per source (dwd, nws, eccc) or country (meteoalarm:<country>)
    def archive_states(self, states, t=None, extra=None, heartbeat=True):
        t = t or now_ms()
        by_src, present = {}, {}
        for st in list(states.values()):
            shapes = st.shapes
            for aid, a in list(st.alerts.items()):
                src = a["source"]
                if src in self.skip or a["event"] not in PRECIP_EVENTS:
                    continue
                h = short_hash(f"{src}|{aid}")
                present.setdefault(src, []).append(h)
                if h in self.known:
                    continue
                rows = by_src.setdefault(src, [])
                gks = []
                for key, _, _ in a["areas"]:
                    s = shapes.get(key) if key else None
                    if s is not None:
                        gks.append(self._geo(compact_polys(s), rows))
                de, en = pick(a["infos"], "de"), pick(a["infos"], "en")
                first = (en or (a["infos"][0] if a["infos"] else ("", "", "", "")))
                row = {"k": "w", "h": h, "src": src, "id": aid, "c": a.get("country"), "ev": a["event"], "lv": a["level"],
                       "name": first[1], "name_de": de[1] if de else None, "name_en": en[1] if en else None,
                       "sent": a.get("sent"), "on": a.get("onset"), "ex": a.get("expires"), "seen": t,
                       "txt": " ".join(str(first[3] or first[2] or "").split())[:600], "g": gks}
                if extra and aid in extra:
                    row.update(extra[aid])
                rows.append(row)
                self.known.add(h)
        for src in set(by_src) | set(present):
            rows = by_src.get(src, [])
            if heartbeat:
                rows.append({"k": "s", "t": t, "src": src, "h": sorted(set(present.get(src, [])))})
            self._write(src, rows, t)

    def archive_lightning(self):
        with self.thunder.lock:
            li = self.thunder.lightning
        if li and li[0] != self.last_lightning:
            self.last_lightning = li[0]
            self._write("lightning", [{"k": "l", "t": li[0], "pts": li[1]}], li[0])

    def archive_thunder(self, t=None):
        t = t or now_ms()
        th = self.thunder
        with th.lock:
            ws = list(th.warnings)
        rows, present = [], []
        for w in ws:
            rings = [np.asarray(r, np.float64) for r in w["rings"] if len(r) >= 4]
            if not rings:
                continue
            # thunder.py asks the DWD WFS 2.0 for EPSG:4326, which answers in lat/lon axis order: put lon first
            if float(np.mean([r[:, 0].mean() for r in rings])) > 40.0:
                rings = [r[:, ::-1] for r in rings]
            polys = [[np.round(r, 3).tolist()] for r in rings]
            h = short_hash(f"dwdnow|{w['id']}|{w['group']}")
            present.append(h)
            gk = self._geo(polys, rows)
            lv = {"Minor": 2, "Moderate": 3, "Severe": 4, "Extreme": 4}.get(w.get("severity"), 2)
            if h not in self.known:
                self.known.add(h)
                rows.append({"k": "w", "h": h, "src": "dwdnow", "id": w["id"], "c": "DE",
                             "ev": "thunderstorm" if w["group"] == "Gewitter" else "rain", "lv": lv,
                             "name": w.get("level"), "group": w["group"], "sev": w.get("severity"), "sent": None,
                             "on": w["onset"], "ex": w["expires"], "seen": t, "txt": "", "g": [gk]})
                self.versions[h] = (gk, w["expires"])
            elif self.versions.get(h) != (gk, w["expires"]):
                self.versions[h] = (gk, w["expires"])
                rows.append({"k": "v", "h": h, "t": t, "g": gk, "ex": w["expires"], "lv": lv})
        rows.append({"k": "s", "t": t, "src": "dwdnow", "h": sorted(set(present))})
        self._write("dwdnow", rows, t)

    # forecasts
    def archive_fc(self):
        fc = self.fc
        if fc is None:
            return
        os.makedirs(self.fc_dir, exist_ok=True)
        if isinstance(fc, PendingFc):
            for ref, leads in sorted(fc.runs().items()):
                path = os.path.join(self.fc_dir, f"{ref}.npz")
                if ref in self.fc_done or os.path.exists(path) or len(leads) < len(FC_LEADS):
                    continue
                try:
                    out = {f"r{lead}": _q(np.load(fp)["g"]) for lead, fp in leads.items()}
                except (OSError, ValueError, KeyError):
                    continue
                self._save_fc(path, out)
                self.fc_done.add(ref)
        else:
            ref, frames = fc.latest()
            if ref is None or ref % FC_EVERY_MS or ref in self.fc_done:
                return
            path = os.path.join(self.fc_dir, f"{ref}.npz")
            if os.path.exists(path):
                self.fc_done.add(ref)
                return
            times = sorted(t for t in frames if ref < t <= ref + 120 * 60_000)
            if len(times) < 12:
                return
            out, acc = {}, {60: None, 120: None}
            prev = ref
            for t in times:
                g = fc.to_grid(frames[t]).astype(np.float32)
                k = 60 if t <= ref + 60 * 60_000 else 120
                add = np.maximum(g, 0) * ((t - prev) / 3600_000)
                acc[k] = add if acc[k] is None else acc[k] + add
                prev = t
                lead = (t - ref) // 60_000
                if lead in FC_LEADS:
                    out[f"r{lead}"] = _q(g)
                del g
            for k, a in acc.items():
                if a is not None:
                    out[f"a{k}"] = _q(a)
            self._save_fc(path, out)
            self.fc_done.add(ref)
        if len(self.fc_done) > 500:
            cutoff = now_ms() - FC_KEEP_DAYS * 86400_000
            for fp in glob.glob(os.path.join(self.fc_dir, "*.npz")):
                try:
                    if int(os.path.basename(fp)[:-4]) < cutoff:
                        os.remove(fp)
                except ValueError:
                    pass
            self.fc_done = {r for r in self.fc_done if r >= cutoff}

    @staticmethod
    def _save_fc(path, out):
        with open(path + ".tmp", "wb") as f:
            np.savez_compressed(f, **out)
        os.replace(path + ".tmp", path)

    def step(self):
        if time.time() - self.last_cycle >= self.cycle_s:
            self.last_cycle = time.time()
            if self.warnings is not None and self.warnings.states:
                self.archive_states(self.warnings.states)
            if self.thunder is not None:
                self.archive_thunder()
        if self.thunder is not None:
            self.archive_lightning()
        self.archive_fc()

    def loop(self):
        time.sleep(120)  # let the warning feeds and the first forecast arrive
        while True:
            try:
                self.step()
            except Exception as e:
                self.log(f"warn_check: {e!r}")
            time.sleep(60)


# ---------- backfill ----------
DWD_CAP_DIR = "https://opendata.dwd.de/weather/alerts/cap/COMMUNEUNION_DWD_STAT/"


def backfill_dwd(out_root, cache_dir, since_ms=0, every_min=30, log=print):
    """DWD's open CAP directory keeps ~2 days of complete status snapshots: one every `every_min` -> the archive
    (as if the Recorder had run then). German-language files (all the check needs)."""
    import official_warnings as ow
    names = sorted(set(re.findall(r"Z_CAP_C_EDZW_(\d{14})_PVW_STATUS_PREMIUMDWD_COMMUNEUNION_DE\.zip",
                                  fetch(DWD_CAP_DIR, timeout=120).decode("utf-8", "replace"))))
    w = ow.Warnings(fetch, log, cache_dir, sources=["dwd"])
    w._dwd_cells()
    rec = Recorder(log, name="de", log_dir=out_root)
    last = 0
    import io
    import zipfile
    for stamp in names:
        t = int(datetime.strptime(stamp, "%Y%m%d%H%M%S").replace(tzinfo=timezone.utc).timestamp() * 1000)
        if t < since_ms or t - last < every_min * 60_000 - 90_000:
            continue
        try:
            z = zipfile.ZipFile(io.BytesIO(fetch(DWD_CAP_DIR + f"Z_CAP_C_EDZW_{stamp}_PVW_STATUS_PREMIUMDWD_COMMUNEUNION_DE.zip")))
        except Exception as e:
            log(f"  {stamp}: {e}")
            continue
        alerts, refs = {}, set()
        for n in z.namelist():
            try:
                a = w._parse_dwd(z.read(n), t)
            except Exception:
                continue
            if a:
                refs |= a.pop("refs")
                alerts[a["id"]] = a
        alerts = {k: a for k, a in alerts.items() if k not in refs}

        def shape_of(a, key):
            if key in a["own"]:
                return a["own"][key]
            c = w.dwd_cells.get(key.split(":", 1)[1]) if key.startswith("cell:") else None
            return c[0] if c else None
        rec.archive_states({"dwd": ow.build_state("dwd", alerts, shape_of)}, t=t)
        last = t
        log(f"  dwd {stamp}: {len(alerts)} warnings")


NWS_HIST_EVENTS = ("Flash Flood Warning", "Flood Advisory", "Severe Thunderstorm Warning", "Tornado Warning",
                   "Winter Storm Warning", "Winter Weather Advisory", "Lake Effect Snow Warning", "Ice Storm Warning")


def backfill_nws(out_root, cache_dir, since, until=None, log=print):
    """api.weather.gov keeps the alerts of the last days (start / end query): the checkable kinds -> the archive.
    Their effective end: the 'sent' time of a later message that references them (update, cancel), else ends /
    expires. Zone geometry from the instance's cache (copy it into cache_dir), missing zones fetched (capped)."""
    import urllib.parse
    import official_warnings as ow
    w = ow.Warnings(fetch, log, cache_dir, sources=["nws"])
    w._nws_load_zones()
    until = until or datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")
    url = ("https://api.weather.gov/alerts?" + urllib.parse.urlencode(
        {"start": since, "end": until, "event": ",".join(NWS_HIST_EVENTS), "limit": 500}))
    feats = []
    for _ in range(40):
        d = json.loads(fetch(url, timeout=120))
        feats += d.get("features") or []
        nxt = (d.get("pagination") or {}).get("next")
        if not nxt or not d.get("features"):
            break
        url = nxt
        time.sleep(1)
    later = {}  # referenced id -> earliest sent of a message referencing it
    for f in feats:
        p = f.get("properties") or {}
        for r in p.get("references") or []:
            if isinstance(r, dict) and r.get("identifier"):
                s = ow.iso_ms(p.get("sent"))
                if s:
                    later[r["identifier"]] = min(later.get(r["identifier"], s), s)
    alerts, extra, fetched = {}, {}, 0
    for f in feats:
        p = f.get("properties") or {}
        if p.get("status") != "Actual" or p.get("messageType") == "Cancel":
            continue
        aid = p.get("id") or f.get("id")
        name = p.get("event") or ""
        low = name.lower()
        level = 4 if p.get("severity") == "Extreme" else 3 if ("warning" in low or "emergency" in low) else 2
        a = ow.Warnings._alert("nws", aid, country="US", sent=ow.iso_ms(p.get("sent")),
                               onset=ow.iso_ms(p.get("onset")) or ow.iso_ms(p.get("effective")) or ow.iso_ms(p.get("sent")),
                               expires=ow.iso_ms(p.get("ends")) or ow.iso_ms(p.get("expires")), level=level,
                               event=ow.classify(name), sender=p.get("senderName") or "NWS",
                               infos=[(p.get("language") or "en-US", name, p.get("headline") or name, str(p.get("description") or ""))])
        s = ow.make_shape(ow.geojson_polys(f.get("geometry"))) if f.get("geometry") else None
        if s:
            a["own"][f"{aid}#0"] = s
            a["areas"].append((f"{aid}#0", None, "polygon"))
        else:
            for u in p.get("affectedZones") or []:
                zid = str(u).rstrip("/").rsplit("/", 1)[1]
                if zid not in w.nws_shapes and fetched < 300:
                    try:
                        zs = ow.make_shape(ow.geojson_polys(json.loads(fetch(u, timeout=60)).get("geometry")))
                        if zs:
                            w.nws_shapes[zid] = zs
                    except Exception as e:
                        log(f"  zone {zid}: {e}")
                    fetched += 1
                    time.sleep(0.25)
                a["areas"].append((f"zone:{zid}", None, zid))
        alerts[aid] = a
        if aid in later:
            extra[aid] = {"end_eff": later[aid]}

    def shape_of(a, key):
        return w.nws_shapes.get(key[5:]) if key.startswith("zone:") else a["own"].get(key)
    st = ow.build_state("nws", alerts, shape_of)
    rec = Recorder(log, name="na", log_dir=out_root)
    # write each alert with its own sent time as "seen" (no heartbeats: the history has none)
    for aid, a in sorted(alerts.items(), key=lambda kv: kv[1]["sent"] or 0):
        one = ow.State("nws", {aid: a}, {k: v for k, v in st.links.items() if any(x == aid for x, _ in v)}, st.shapes)
        rec.archive_states({"x": one}, t=a["sent"] or now_ms(), extra=extra, heartbeat=False)
    rec._write("nws", [{"k": "c", "src": "nws", "t0": ow.iso_ms(since), "t1": ow.iso_ms(until),
                        "events": list(NWS_HIST_EVENTS)}], now_ms())
    log(f"nws backfill: {len(alerts)} alerts ({len(feats)} messages), {fetched} zones fetched")


# ---------- part 2: the check ----------
WORLD = 20037508.342789244


class Grid:
    """The analysis archive of a domain: shape, lon/lat -> (col, row), (col, row) -> lon/lat, frame files."""

    def __init__(self, dom, root):
        self.dom = dom
        self.kind = "eu" if dom == "eun" else dom
        if dom == "de":
            self.obs_dir = os.path.join(root, "obs3km")
            meta = json.load(open(os.path.join(self.obs_dir, "grid.json")))
            self.bbox, self.cell = meta["bbox"], float(meta["cell"])
            self.shape = tuple(meta["shape"])
            self.static = np.load(os.path.join(self.obs_dir, "mask.npy")).astype(bool)
        elif self.kind == "eu":   # eun: the same grid, our own Europe picture (OPERA + national radars)
            self.obs_dir = os.path.join(root, "eun", "obs_eu") if dom == "eun" else os.path.join(root, "obs_eu")
            meta = json.load(open(os.path.join(self.obs_dir, "grid.json")))
            from pyproj import Transformer
            self.fwd = Transformer.from_crs("EPSG:4326", meta["proj"], always_xy=True)
            self.inv = Transformer.from_crs(meta["proj"], "EPSG:4326", always_xy=True)
            self.cell = float(meta["cell"])
        else:
            import northamerica as na
            self.obs_dir = os.path.join(root, "na", "obs_eu")
            self.deg = na.DEG0 * 4                      # MRMS 0.01 deg x 2 (region) x 2 (verification grid)
            self.lat_top, self.lon_left = na.LAT1 + na.DEG0 / 2, na.LON0 - na.DEG0 / 2
        if self.kind != "de":
            f = sorted(glob.glob(os.path.join(self.obs_dir, "2*.npz")))
            self.shape = np.load(f[-1])["rate"].shape if f else (0, 0)
            self.static = np.ones(self.shape, bool)

    def to_px(self, lon, lat):
        lon, lat = np.asarray(lon, np.float64), np.asarray(lat, np.float64)
        if self.kind == "de":
            x = lon * WORLD / 180.0
            y = np.log(np.tan((90.0 + np.clip(lat, -85, 85)) * np.pi / 360.0)) * WORLD / np.pi
            return (x - self.bbox[0]) / self.cell - 0.5, (self.bbox[3] - y) / self.cell - 0.5
        if self.kind == "eu":
            x, y = self.fwd.transform(lon, lat)
            return np.asarray(x) / self.cell - 0.5, -np.asarray(y) / self.cell - 0.5
        return (lon - self.lon_left) / self.deg - 0.5, (self.lat_top - lat) / self.deg - 0.5

    def to_lonlat(self, col, row):
        col, row = np.asarray(col, np.float64), np.asarray(row, np.float64)
        if self.kind == "de":
            x, y = self.bbox[0] + (col + 0.5) * self.cell, self.bbox[3] - (row + 0.5) * self.cell
            return x / WORLD * 180.0, np.degrees(2 * np.arctan(np.exp(y / WORLD * np.pi)) - np.pi / 2)
        if self.kind == "eu":
            return self.inv.transform((col + 0.5) * self.cell, -(row + 0.5) * self.cell)
        return self.lon_left + (col + 0.5) * self.deg, self.lat_top - (row + 0.5) * self.deg

    def frames(self, t0, t1):
        out = []
        for fp in sorted(glob.glob(os.path.join(self.obs_dir, "2*.npz"))):
            try:
                t = day_ms(os.path.basename(fp)[:8]) + int(os.path.basename(fp)[8:10]) * 3600_000 + int(os.path.basename(fp)[10:12]) * 60_000
            except ValueError:
                continue
            if t0 < t <= t1:
                out.append((t, fp))
        return out

    def rasterize(self, polys, value=1, out=None):
        m = np.zeros(self.shape, np.uint8) if out is None else out
        h, w = self.shape
        for rings in polys:
            for k, ring in enumerate(rings):
                a = np.asarray(ring, np.float64)
                if a.ndim != 2 or len(a) < 3:
                    continue
                c, r = self.to_px(a[:, 0], a[:, 1])
                if c.max() < -1 or r.max() < -1 or c.min() > w or r.min() > h:
                    continue
                pts = np.round(np.stack([np.clip(c, -10, w + 10), np.clip(r, -10, h + 10)], 1) * 8).astype(np.int32)
                cv2.fillPoly(m, [pts], 0 if k else value, lineType=cv2.LINE_8, shift=3)
                if k == 0:  # an area smaller than a cell still gets the cell it lies in
                    ci, ri = int(round(float(c.mean()))), int(round(float(r.mean())))
                    if 0 <= ci < w and 0 <= ri < h and not m[max(0, ri - 1):ri + 2, max(0, ci - 1):ci + 2].any():
                        m[ri, ci] = value
        return m


US_STATES = set("AL AZ AR CA CO CT DE DC FL GA ID IL IN IA KS KY LA ME MD MA MI MN MS MO MT NE NV NH NJ NM NY NC ND OH "
                "OK OR PA RI SC SD TN TX UT VT VA WA WV WI WY".split())
DOMAINS = {  # own = the sources that must have been archived for an "our-only" case (no warning) to count
    "de": {"sources": ("dwd", "dwdnow"), "own": ("dwd", "dwdnow"), "arch": "", "fc": "warnfc_de", "gauges": "", "skip_c": ()},
    "eu": {"sources": ("meteoalarm",), "own": ("meteoalarm",), "arch": "", "fc": "warnfc_eu", "gauges": "", "skip_c": ("DE",)},
    "eun": {"sources": ("meteoalarm",), "own": ("meteoalarm",), "arch": "", "fc": "eun/warnfc_eun", "gauges": "", "skip_c": ("DE",)},
    "na": {"sources": ("nws", "eccc"), "own": ("nws",), "arch": "na", "fc": "na/warnfc_na", "gauges": "na", "skip_c": ()}}


def load_archive(dirs, sources, day, back=4, ahead=1):
    W, G, last, hbs, V, spans = {}, {}, {}, {}, {}, []
    d0 = datetime.strptime(day, "%Y%m%d")
    days = [(d0 + timedelta(days=k)).strftime("%Y%m%d") for k in range(-back, ahead + 1)]
    for d in dirs:
        for src in sources:
            for dd in days:
                fp = os.path.join(d, f"{src}_{dd}.jsonl")
                if not os.path.exists(fp):
                    continue
                with open(fp, encoding="utf-8") as f:
                    for line in f:
                        try:
                            r = json.loads(line)
                        except ValueError:
                            continue
                        k = r.get("k")
                        if k == "w":
                            W.setdefault(r["h"], r)
                        elif k == "g":
                            G[r["gk"]] = r["p"]
                        elif k == "s":
                            hbs.setdefault(r["src"], set()).add(r["t"])
                            for h in r["h"]:
                                last[h] = max(last.get(h, 0), r["t"])
                        elif k == "v":
                            V.setdefault(r["h"], []).append(r)
                        elif k == "c":
                            spans.append((r["src"], r["t0"], r["t1"]))
    hb_sorted = {s: np.array(sorted(v), np.int64) for s, v in hbs.items()}
    for h, w in W.items():
        on = w.get("on") or w.get("sent") or w["seen"]
        ex = w.get("ex") or on + 24 * 3600_000
        vs = sorted(V.get(h, []), key=lambda r: r["t"])
        if vs and vs[-1].get("ex"):
            ex = vs[-1]["ex"]
        end = ex
        ls = last.get(h)
        hb = hb_sorted.get(w["src"])
        if ls is not None and hb is not None:
            i = np.searchsorted(hb, ls, side="right")
            if i < len(hb) and hb[i] < ex:
                end = max(on, (ls + int(hb[i])) // 2)  # gone from the feed before its expiry
        if w.get("end_eff"):
            end = min(end, w["end_eff"])
        w["_on"], w["_end"], w["_vers"] = on, end, vs
    # superseded before it began (an update moved / replaced it): never in force
    W = {h: w for h, w in W.items() if w["_end"] > w["_on"]}
    # one event, several messages: updates of a warning carry a new id (DWD, MeteoAlarm, NWS); same source, name,
    # level and areas with windows that touch (<= 30 min gap) are one warning (the first message's id and sent time)
    groups = {}
    for h, w in W.items():
        groups.setdefault((w["src"], w.get("name_de") or w.get("name"), w.get("lv"), tuple(sorted(w.get("g", [])))), []).append(w)
    merged = {}
    for ws in groups.values():
        ws.sort(key=lambda w: w["_on"])
        cur = None
        for w in ws:
            if cur is not None and not cur["_vers"] and not w["_vers"] and w["_on"] <= cur["_end"] + 30 * 60_000:
                cur["_end"] = max(cur["_end"], w["_end"])
                cur["_msgs"] = cur.get("_msgs", 1) + 1
                continue
            cur = w
            merged[w["h"]] = w
    W = merged
    def archived(t, need):
        """Was the archive of every source in `need` running at t (a heartbeat within 20 min, or a backfilled span)?"""
        for src in need:
            beats = hb_sorted.get(src, np.zeros(0, np.int64))
            i = np.searchsorted(beats, t)
            near = any(0 <= j < len(beats) and abs(int(beats[j]) - t) <= 20 * 60_000 for j in (i - 1, i))
            if not near and not any(s == src and a <= t <= b for s, a, b in spans):
                return False
        return True
    return W, G, archived


def load_lightning(dirs, day):
    pts = []
    d0 = datetime.strptime(day, "%Y%m%d")
    for d in dirs:
        for k in (-1, 0, 1):
            fp = os.path.join(d, f"lightning_{(d0 + timedelta(days=k)).strftime('%Y%m%d')}.jsonl")
            if os.path.exists(fp):
                with open(fp) as f:
                    for line in f:
                        try:
                            r = json.loads(line)
                            pts.append((r["t"], r["pts"]))
                        except (ValueError, KeyError):
                            pass
    return sorted(pts)


def load_gauges(root, dom, grid, day):
    """-> ids, nets, flat pixel index, (n, 288) mm per 10 min ending at day-1 00:00 + k * 10 min (NaN unknown)."""
    sub = DOMAINS[dom]["gauges"]  # every network counts (in Germany's border zone the neighbours' gauges too)
    st_path = os.path.join(root, "na", "metar", "stations.json") if dom == "na" else os.path.join(root, "eun", "natobs", "stations.json")
    try:
        pos = {f"{s[0]}:{s[1]}": (s[2], s[3]) for s in json.load(open(st_path))}
    except (OSError, ValueError):
        pos = {}
    d0 = day_ms(day)
    series = {}
    for k, dms in enumerate((d0 - 86400_000, d0)):
        fp = os.path.join(root, sub, "gaugetruth", day_of(dms) + ".npz")
        try:
            z = np.load(fp)
        except (OSError, ValueError):
            continue
        for gid, row in zip(z["ids"], z["mm"]):
            gid = str(gid)
            series.setdefault(gid, np.full(288, np.nan, np.float32))[k * 144:(k + 1) * 144] = row
    ids = [g for g in series if g in pos]
    if not ids:
        return [], [], np.zeros(0, np.int64), np.zeros((0, 288), np.float32)
    lat = np.array([pos[g][0] for g in ids])
    lon = np.array([pos[g][1] for g in ids])
    c, r = grid.to_px(lon, lat)
    c, r = np.round(c).astype(int), np.round(r).astype(int)
    h, w = grid.shape
    ok = (c >= 0) & (c < w) & (r >= 0) & (r < h)
    ids = [g for g, o in zip(ids, ok) if o]
    return (ids, [g.split(":")[0] for g in ids], (r[ok] * w + c[ok]).astype(np.int64),
            np.stack([series[g] for g in ids]).astype(np.float32))


def gauge_window_max(mm, nets, idx, D, a, b, base):
    """Per gauge in idx: the largest D-hour sum with its end in (a, b] (10-minute steps), None where unknown."""
    out = []
    n = D * 6
    for i in idx:
        s = mm[i]
        hourly = nets[i] in HOURLY_NETS
        best = None
        k1 = int((b - base) // 600_000)
        k0 = max(n - 1, int((a - base) // 600_000) + 1)
        for k in range(k0, min(k1, 287) + 1):
            win = s[k - n + 1:k + 1]
            have = np.isfinite(win)
            need = D if hourly else int(0.8 * n)
            if have.sum() >= need:
                v = float(np.nansum(win))
                best = v if best is None or v > best else best
        out.append(best)
    return out


def service_mask(dom, grid, geocodes, nws_zones):
    """Where a service warns (land): country index raster + names. de: Germany (germany.json); eu: MeteoAlarm
    geocode areas (the instance's cache); na: the NWS zones the instance has seen (partial)."""
    lab = np.zeros(grid.shape, np.uint8)
    names = [""]
    if dom == "de":
        g = json.load(open(os.path.join(os.path.dirname(os.path.abspath(__file__)), "germany.json")))
        grid.rasterize(g["coordinates"], 1, lab)
        names.append("DE")
    elif dom in ("eu", "eun") and geocodes and os.path.exists(geocodes):
        codes = json.load(open(geocodes))["codes"]
        for v in codes.values():
            c = (v.get("c") or "").upper()
            if c == "DE":
                continue
            if c not in names:
                names.append(c)
            grid.rasterize(v["p"], names.index(c), lab)
    elif dom == "na" and nws_zones and os.path.exists(nws_zones):
        names.append("US")
        for zid, v in json.load(open(nws_zones)).get("zones", {}).items():
            if v.get("p") and zid[:2] in US_STATES and zid[2:3] in ("Z", "C"):   # land zones / counties, not marine
                grid.rasterize(v["p"], 1, lab)
    return lab, names


def make_area(grid, polys, kernel):
    """A warned area on the grid, kept small: flat indices of its cells ("idx"), of the cells within GAUGE_DIL
    ("didx"), its bounding box with the mask ("box": row0, col0, uint8 sub-mask) and its middle (lon, lat)."""
    m = grid.rasterize(polys)
    rows, cols = np.nonzero(m)
    if not len(rows):
        return None
    h, w = grid.shape
    p = GAUGE_DIL + 1
    y0, y1, x0, x1 = max(0, rows.min() - p), min(h, rows.max() + p + 1), max(0, cols.min() - p), min(w, cols.max() + p + 1)
    sub = (m[y0:y1, x0:x1] > 0).astype(np.uint8)
    del m
    dil = cv2.dilate(sub, kernel)
    rr, cc = np.nonzero(sub)
    dr, dc = np.nonzero(dil)
    lon, lat = grid.to_lonlat(float(cols.mean()), float(rows.mean()))
    return {"idx": ((rr + y0) * w + cc + x0).astype(np.int64), "didx": ((dr + y0) * w + dc + x0).astype(np.int64),
            "box": (y0, x0, sub), "ll": (round(float(lon), 2), round(float(lat), 2))}


def fc_load(dirs, ref, L):
    """Our forecast of run `ref` at lead L: (rate grid, 1-hour rain grid or None), None when not archived."""
    for d in dirs:
        fp = os.path.join(d, f"{ref}.npz")
        if os.path.exists(fp):
            try:
                z = np.load(fp)
                if f"r{L}" not in z.files:
                    return None
                a = f"a{L}" if L in (60, 120) and f"a{L}" in z.files else None
                return _unq(z[f"r{L}"]), (_unq(z[a]) if a else None)
            except (OSError, ValueError):
                return None
    return None


def _stats(v, rob):
    """[max, 95th percentile, robust max (max of the 3x3 median)] of the values inside an area."""
    if v.size == 0:
        return [0.0, 0.0, 0.0]
    return [round(float(v.max()), 2), round(float(np.percentile(v, 95)), 2), round(float(rob.max()), 2)]


def km_per_px(grid):
    if grid.dom == "na":
        return 4.0                        # 0.04 deg: 4.4 km north-south, ~3.3 km east-west at 40 N
    return grid.cell / 1000.0 * (0.63 if grid.dom == "de" else 1.0)   # Web Mercator at ~51 N / LAEA


def check_domain(dom, day, root, extra=(), geocodes=None, nws_zones=None, log=print):
    cfg = DOMAINS[dom]
    grid = Grid(dom, root)
    if not grid.shape[0]:
        log(f"{dom}: no analysis archive")
        return []
    h_, w_ = grid.shape
    t_day0 = day_ms(day)
    t_day1 = t_day0 + 86400_000
    arch_dirs = [os.path.join(root, cfg["arch"], ARCH)] + [os.path.join(e, cfg["arch"], ARCH) for e in extra]
    fc_dirs = [os.path.join(root, cfg["fc"])] + [os.path.join(e, cfg["fc"]) for e in extra]
    W, G, archived = load_archive(arch_dirs, cfg["sources"], day)
    frames = grid.frames(t_day0 - 24 * 3600_000, t_day1)
    if not frames:
        log(f"{dom} {day}: no radar frames")
        return []
    t_last = frames[-1][0]
    kernel = np.ones((2 * GAUGE_DIL + 1,) * 2, np.uint8)
    gids, gnets, gflat, gmm = load_gauges(root, dom, grid, day)
    gbase = t_day0 - 86400_000
    # warnings touching the day; "cover" = every precipitation warning with an area (for the our-only test),
    # "warns" = those with checkable criteria
    warns, cover = [], []
    for h, w in W.items():
        if w["_end"] <= t_day0 - 24 * 3600_000 or w["_on"] >= t_day1 + 3600_000 or (w.get("c") or "").upper() in cfg["skip_c"]:
            continue
        polys = [p for gk in w.get("g", []) for p in G.get(gk, [])]
        if not polys:
            continue
        ar = make_area(grid, polys, kernel)
        if ar is None:
            continue
        w["_area"] = ar
        w["_vareas"] = [(r["t"], va) for r in w.get("_vers", []) if r.get("g") in G
                        for va in [make_area(grid, G[r["g"]], kernel)] if va is not None]
        cover.append(w)
        crit = criteria(w)
        if crit is None:
            continue
        # floods follow the rain: the rain of the PRE_FLOOD before a flood warning's onset counts for it
        w.update(_crit=crit, _idx=ar["idx"], _steps=[], _lt=0, _acc={}, _pre=PRE_FLOOD if w["ev"] == "flood" else 0,
                 _gi=np.flatnonzero(np.isin(gflat, ar["didx"])) if len(gflat) else np.zeros(0, np.int64))
        warns.append(w)
    lpx = []
    if dom == "de":
        for t, pts in load_lightning(arch_dirs, day):
            if pts:
                a = np.asarray(pts, np.float64)
                c, r = grid.to_px(a[:, 1], a[:, 0])
                c, r = np.round(c).astype(int), np.round(r).astype(int)
                ok = (c >= 0) & (c < w_) & (r >= 0) & (r < h_)
                lpx.append((t, r[ok] * w_ + c[ok]))
    lab, cnames = service_mask(dom, grid, geocodes, nws_zones)
    log(f"{dom} {day}: {len(frames)} frames, {len(W)} archived warnings, {len(cover)} with an area on the grid, "
        f"{len(warns)} checkable, {len(gids)} gauges, {len(lpx)} lightning images")

    def area_at(w, t):
        """The warned area at time t (DWD nowcast warnings move: their newest version before t)."""
        ar = w["_area"]
        for tv, va in w["_vareas"]:
            if tv <= t:
                ar = va
        return ar

    ring = deque()
    hours, cur_hour, cur, cur_n = {}, None, None, 0
    cnt5, n_day = np.zeros(grid.shape, np.uint16), 0
    rg_t, rg = [], []
    rows, own_cases, warned_cache = [], [], {}
    T1, kmpx = OWN_T[dom], km_per_px(grid)
    for t, fp in frames:
        try:
            r = np.load(fp)["rate"].astype(np.float32)
        except (OSError, ValueError, KeyError):
            continue
        cov = (r >= 0) & grid.static
        r = np.where(cov, np.nan_to_num(r, nan=0.0), 0.0).astype(np.float32)
        hk = ((t - 1) // 3600_000 + 1) * 3600_000   # the frame belongs to the hour (hk - 1 h, hk]
        if hk != cur_hour:
            if cur is not None and cur_n >= 6:
                hours[cur_hour] = (cur * (12.0 / cur_n)).astype(np.float16)
            cur_hour, cur, cur_n = hk, np.zeros(grid.shape, np.float32), 0
        cur += r * (5.0 / 60.0)
        cur_n += 1
        ring.append((t, r))
        while ring[0][0] <= t - 3600_000:
            ring.popleft()
        rg_t.append(t)
        rg.append(np.where(cov.ravel()[gflat], r.ravel()[gflat], np.nan) if len(gflat) else np.zeros(0, np.float32))
        in_day = t > t_day0
        if in_day:
            n_day += 1
            cnt5 += (r >= 5.0)
        if t % STEP_MS or len(ring) < 9:
            continue
        act = [w for w in warns if w["_on"] - w["_pre"] - 3600_000 <= t <= w["_end"] + 1800_000]
        if not in_day and not act:
            continue
        # ---- one 15-min step ----
        s1 = sum(x[1] for x in ring) * (5.0 / 60.0) * (12.0 / len(ring))   # sliding 1-hour rain (mm)
        s1m = cv2.medianBlur(s1.astype(np.float32), 3)
        rm = cv2.medianBlur(r, 3)
        s1f, s1mf, rf, rmf, covf = s1.ravel(), s1m.ravel(), r.ravel(), rm.ravel(), cov.ravel()
        fcs = {}
        if act:
            for L in (30, 60, 120):
                f = fc_load(fc_dirs, t - L * 60_000, L)
                if f is not None:
                    g, a = f
                    fcs[L] = (g.ravel(), cv2.medianBlur(g, 3).ravel(),
                              None if a is None else a.ravel(), None if a is None else cv2.medianBlur(a, 3).ravel())
        for w in act:
            ar = area_at(w, t)
            idx = ar["idx"]
            c = idx[covf[idx]]
            row = {"k": "step", "dom": dom, "h": w["h"], "t": t, "cov": round(len(c) / max(len(idx), 1), 2),
                   "r": _stats(rf[c], rmf[c]), "s1": _stats(s1f[c], s1mf[c])}
            fr = {}
            for L, (g, gm, a, am) in fcs.items():
                e = {"r": _stats(g[idx], gm[idx])}
                if a is not None:
                    e["a"] = _stats(a[idx], am[idx])
                fr[str(L)] = e
            if fr:
                row["fc"] = fr
            if lpx:
                n = sum(int(np.isin(p, ar["didx"]).sum()) for tl, p in lpx if t - STEP_MS < tl <= t)
                if n:
                    row["lt"] = n
                    if w["_on"] <= t <= w["_end"]:
                        w["_lt"] += n
            w["_steps"].append(row)
            if in_day:
                rows.append(row)
        if not in_day:
            continue
        # ---- our-only: >= OWN_T mm in 1 h on a service's land, no precipitation warning (any level) nearby ----
        if not archived(t, cfg["own"]):   # a warning archive missing at that time: "no warning" would mean nothing
            continue
        big = (s1m >= T1) & cov & (lab > 0)
        if not big.any():
            continue
        key = tuple(sorted(w["h"] for w in cover if w["_on"] - 3600_000 <= t <= w["_end"] + 3600_000))
        if key not in warned_cache:
            u = np.zeros(grid.shape, np.uint8)
            for w in cover:
                if w["_on"] - 3600_000 <= t <= w["_end"] + 3600_000:
                    ar = area_at(w, t)
                    y0, x0, sub = ar["box"]
                    u[y0:y0 + sub.shape[0], x0:x0 + sub.shape[1]] |= sub
            warned_cache = {key: cv2.dilate(u, kernel) > 0}
        warned = warned_cache[key]
        n, labels, st, _ = cv2.connectedComponentsWithStats(big.astype(np.uint8), connectivity=8)
        for k in range(1, n):
            x, y, bw, bh = (int(v) for v in st[k, :4])
            comp = labels[y:y + bh, x:x + bw] == k
            if (comp & warned[y:y + bh, x:x + bw]).any():
                continue
            rr, cc = np.nonzero(comp)
            cf = (rr + y) * w_ + (cc + x)
            pk = int(cf[np.argmax(s1mf[cf])])
            pr, pc = divmod(pk, w_)
            case = next((oc for oc in own_cases if t - oc["t1"] <= 45 * 60_000 and
                         math.hypot(oc["pr"] - pr, oc["pc"] - pc) * kmpx <= 25), None)
            if case is None:
                case = {"t0": t, "t1": t, "peak": 0.0, "pr": pr, "pc": pc, "pix": set(), "c": cnames[int(lab[pr, pc])]}
                own_cases.append(case)
            case["t1"] = t
            if s1mf[pk] > case["peak"]:
                lon, lat = grid.to_lonlat(pc, pr)
                case.update(peak=float(s1mf[pk]), pr=pr, pc=pc, lat=float(lat), lon=float(lon), tpk=t, rmax=float(rf[cf].max()))
            if len(case["pix"]) < 20000:
                case["pix"].update(cf.tolist())
    if cur is not None and cur_n >= 6:
        hours[cur_hour] = (cur * (12.0 / cur_n)).astype(np.float16)
    del ring, cur
    rg = np.stack(rg) if rg and len(gflat) else np.zeros((len(rg_t), 0), np.float32)
    rg_t = np.array(rg_t, np.int64)

    def radar_at_gauge(gi, D, a, b):
        """Largest D-hour radar sum at gauge gi with its end in (a, b] (10-minute steps); None: radar blind there."""
        best = None
        col = rg[:, gi]
        for e in range(int(a // 600_000 + 1) * 600_000, int(b) + 1, 600_000):
            v = col[(rg_t > e - D * 3600_000) & (rg_t <= e)]
            k = int(np.isfinite(v).sum())
            if k >= 0.75 * D * 12:
                x = float(np.nansum(v)) * (5.0 / 60.0) * (D * 12.0 / k)
                best = x if best is None or x > best else best
        return best

    def radar_rate_at_gauge(gi, a, b):
        v = rg[(rg_t > a) & (rg_t <= b), gi]
        return float(np.nanmax(v)) if np.isfinite(v).any() else None

    # ---- the warnings evaluated today: their (effective) window ends today, or is still open at the newest image ----
    ev_warns = [w for w in warns if t_day0 <= min(w["_end"], t_last) < t_day1]
    # 6-24 h amounts: running sums over hour-aligned windows, one pass per duration for all warnings
    needs = {}
    for w in ev_warns:
        if w["_crit"]["kind"] == "acc":
            for D, _ in w["_crit"]["crit"]:
                if D >= 2:
                    for he in hours:
                        if w["_on"] - w["_pre"] < he <= w["_end"] + 3600_000:
                            needs.setdefault((D, he), []).append(w)
    for D in sorted({d for d, _ in needs}):
        run, q = None, deque()
        for he in range(min(hours), max(hours) + 1, 3600_000):
            if he not in hours:
                run, q = None, deque()
                continue
            run = hours[he].astype(np.float32) if run is None else run + hours[he]
            q.append(he)
            if len(q) > D:
                run -= hours[q.popleft()]
            if len(q) == D and (D, he) in needs:
                med = cv2.medianBlur(np.ascontiguousarray(run, np.float32), 3).ravel()
                rf = run.ravel()
                for w in needs[(D, he)]:
                    st = _stats(rf[w["_idx"]], med[w["_idx"]])
                    old = w["_acc"].get(D, [0.0, 0.0, 0.0])
                    w["_acc"][D] = [max(a, b) for a, b in zip(old, st)]
    out = []
    for w in ev_warns:
        on, end = w["_on"], w["_end"]
        crit = w["_crit"]
        steps = [s for s in w["_steps"] if on - w["_pre"] <= s["t"] <= end + 1800_000]
        if not steps:
            continue
        res = {"k": "warn", "dom": dom, "h": w["h"], "src": w["src"], "id": w["id"], "c": (w.get("c") or "").upper()[:16],
               "ev": w["ev"], "lv": w["lv"], "name": w.get("name_de") or w.get("name"), "on": on, "end": end,
               "sent": w.get("sent") or w.get("seen"), "open": end > t_last, "thr": crit["thr"], "px": int(len(w["_idx"])),
               "lon": w["_area"]["ll"][0], "lat": w["_area"]["ll"][1],
               "cov": round(float(np.mean([s["cov"] for s in steps])), 2)}
        ratio, t_obs = -1.0, None
        if crit["kind"] == "rate":
            T = crit["T"]
            res["crit"] = [["rate", T]]
            res["obs"] = {"rate": [max(s["r"][k] for s in steps) for k in range(3)]}
            ratio = res["obs"]["rate"][2] / T
            t_obs = next((s["t"] for s in steps if s["r"][2] >= HIT * T), None)
            if w["_lt"]:
                res["lightning"] = w["_lt"]
        else:
            res["crit"] = [list(c) for c in crit["crit"]]
            res["obs"] = {}
            for D, T in crit["crit"]:
                if D == 1:
                    vals = [max(s["s1"][k] for s in steps) for k in range(3)]
                    t_obs = t_obs or next((s["t"] for s in steps if s["s1"][2] >= HIT * T), None)
                else:
                    vals = w["_acc"].get(D, [0.0, 0.0, 0.0])
                res["obs"][f"{D}h"] = vals
                ratio = max(ratio, vals[2] / T)
        res["ratio"] = round(ratio, 2)
        cls = "hit" if (crit.get("lightning") and w["_lt"] > 0) or ratio >= HIT else "partial" if ratio >= PARTIAL else "miss"
        if res["open"] and cls != "hit":
            cls = "open"      # still in force at the newest image: judged on the day it ends
        if w.get("_msgs"):
            res["msgs"] = w["_msgs"]
        # gauges in / near the area (logged for every warning: radar vs gauges in warned areas)
        gres = {"n": 0}
        gi = list(w["_gi"])
        if gi:
            best_g = -1.0
            for D, T in (crit["crit"] if crit["kind"] == "acc" else [("rate", crit["T"])]):
                if D == "rate":   # the gauges' highest 10-minute amount as a rate
                    k0, k1 = int((on - gbase) // 600_000) + 1, int((end - gbase) // 600_000) + 1
                    vals = [float(np.nanmax(gmm[i, k0:k1 + 1]) * 6) if np.isfinite(gmm[i, k0:k1 + 1]).any() else None for i in gi]
                    rat = lambda i: radar_rate_at_gauge(i, on, end)
                else:
                    vals = gauge_window_max(gmm, gnets, gi, D, on - w["_pre"], end + 3600_000, gbase)
                    rat = lambda i, D=D: radar_at_gauge(i, D, on - w["_pre"], end + 3600_000)
                have = [(i, v) for i, v in zip(gi, vals) if v is not None]
                gres["n"] = max(gres["n"], len(have))
                if not have:
                    continue
                i_best, v_best = max(have, key=lambda x: x[1])
                if v_best / T > best_g:
                    best_g = v_best / T
                    ra = rat(i_best)
                    gres.update(D=D, T=T, gmax=round(v_best, 1), radar_at=None if ra is None else round(ra, 1),
                                gauge=gids[i_best], ratio=round(best_g, 2))
                pairs = [(v, rat(i)) for i, v in have[:60]]
                pairs = [(v, x) for v, x in pairs if x is not None]
                if pairs:  # gauges, radar at the same gauges (where the radar sees them)
                    gres[f"sum_{D}"] = [round(sum(v for v, _ in pairs), 1), round(sum(x for _, x in pairs), 1), len(pairs)]
        res["gauges"] = gres
        if cls == "miss":
            g = gres.get("ratio")
            if g is None:
                cls = "miss_no_gauges"
            elif g >= HIT or (g >= PARTIAL and gres["gmax"] >= 2 * max(gres["radar_at"] or 0.0, 0.1)):
                cls = "miss_radar_low"
            else:
                cls = "miss_not_borne"
        res["class"] = cls
        # lead time (d): 1-hour amounts and rates only
        T_lead = crit["T"] if crit["kind"] == "rate" else next((T for D, T in crit["crit"] if D == 1), None)
        if T_lead is not None and t_obs is not None:
            res["t_obs"] = t_obs
            res["obs_vs_onset_min"] = (t_obs - on) // 60_000
            if res["sent"]:
                res["official_lead_min"] = (t_obs - res["sent"]) // 60_000
            near = [s for s in w["_steps"] if abs(s["t"] - t_obs) <= STEP_MS]
            obs_val = max((s["r"][2] if crit["kind"] == "rate" else s["s1"][2]) for s in near)
            lead = {}
            for L in ("30", "60", "120"):
                vals = [((s.get("fc") or {})[L]["a"][2] if crit["kind"] == "acc" and "a" in s["fc"][L] else s["fc"][L]["r"][2])
                        for s in near if L in (s.get("fc") or {})]
                if vals:
                    lead[L] = [bool(max(vals) >= HIT * T_lead), round(max(vals) / max(obs_val, 0.1), 2)]
            res["lead"] = lead
            shown = [int(L) for L, v in lead.items() if v[0]]
            res["lead_min"] = max(shown) if shown else (0 if lead else None)
        out.append(res)
    # ---- our-only cases: gauges and clutter decide ----
    for case in own_cases:
        pix = np.zeros(h_ * w_, np.uint8)
        pix[np.fromiter(case["pix"], np.int64)] = 1
        dil = cv2.dilate(pix.reshape(grid.shape), kernel).ravel() > 0
        gi = list(np.flatnonzero(dil[gflat])) if len(gflat) else []
        a, b = case["t0"] - 3600_000, case["t1"] + 1800_000
        have = [(i, v) for i, v in zip(gi, gauge_window_max(gmm, gnets, gi, 1, a, b, gbase)) if v is not None] if gi else []
        clutter = float(cnt5[case["pr"], case["pc"]]) / max(n_day, 1)
        res = {"k": "own", "dom": dom, "c": case["c"], "t0": case["t0"], "t1": case["t1"], "tpk": case.get("tpk"),
               "peak1h": round(case["peak"], 1), "rmax": round(case.get("rmax", 0), 1), "lat": round(case["lat"], 3),
               "lon": round(case["lon"], 3), "px": len(case["pix"]), "clutter": round(clutter, 2), "gauges": {"n": len(have)}}
        cls = "own_unverified"
        if have:
            i_best, v_best = max(have, key=lambda x: x[1])
            r_at = radar_at_gauge(i_best, 1, a, b) or 0.0
            res["gauges"].update(gmax=round(v_best, 1), radar_at=round(r_at, 1), gauge=gids[i_best])
            if v_best >= 0.5 * T1 or (r_at >= 5 and v_best >= 0.5 * r_at):
                cls = "own_real"
            elif r_at >= 5 and v_best < 0.2 * r_at:
                cls = "own_artefact"
        if clutter >= 0.25 and cls != "own_real":
            cls = "own_artefact"
        res["class"] = cls
        out.append(res)
    return rows + out


def write_day(rows, dom, day, root):
    fp = os.path.join(root, f"warncheck_{day}.jsonl")
    keep = []
    if os.path.exists(fp):
        with open(fp) as f:
            keep = [line for line in f if f'"dom": "{dom}"' not in line]
    with open(fp + ".tmp", "w") as f:
        f.writelines(keep)
        for r in rows:
            f.write(json.dumps(r, separators=(", ", ": ")) + "\n")
    os.replace(fp + ".tmp", fp)


def summary(root, days, out=print):
    W, O = [], []
    for day in days:
        fp = os.path.join(root, f"warncheck_{day}.jsonl")
        if not os.path.exists(fp):
            continue
        with open(fp) as f:
            for line in f:
                if '"k": "warn"' in line:
                    W.append(json.loads(line))
                elif '"k": "own"' in line:
                    O.append(json.loads(line))
    seen, uniq = set(), []
    for r in sorted(W, key=lambda r: (r["open"], -r["end"])):   # a warning evaluated on two days: the closed one
        if (r["dom"], r["h"]) not in seen:
            seen.add((r["dom"], r["h"]))
            uniq.append(r)
    groups = {}
    for r in uniq:
        groups.setdefault((r["dom"], r["c"] or "?", r["ev"], r["lv"]), []).append(r)
    out(f"\nOfficial warnings vs our radar ({', '.join(days)}); hit >= {HIT} x threshold, partial >= {PARTIAL}")
    out(f"{'dom':3s} {'country':14s} {'event':12s} {'lv':>2s} {'n':>4s} {'hit':>4s} {'part':>4s} {'lowR':>4s} {'notB':>4s} "
        f"{'noG':>4s} {'open':>4s}  {'lead n':>6s} {'@30':>4s} {'@60':>4s} {'@120':>4s} {'fc/obs60':>8s} {'offic.':>6s}")
    for key in sorted(groups):
        g = groups[key]
        cnt = lambda c: sum(1 for r in g if r["class"] == c)
        le = [r for r in g if r.get("lead")]
        pct = lambda L: (f"{100 * sum(1 for r in le if r['lead'].get(L, [False])[0]) / max(1, sum(1 for r in le if L in r['lead'])):3.0f}%"
                         if any(L in r["lead"] for r in le) else "   -")
        fo = [r["lead"]["60"][1] for r in le if "60" in r["lead"]]
        off = [r["official_lead_min"] for r in le if r.get("official_lead_min") is not None]
        out(f"{key[0]:3s} {key[1][:14]:14s} {key[2]:12s} {key[3]:2d} {len(g):4d} {cnt('hit'):4d} {cnt('partial'):4d} "
            f"{cnt('miss_radar_low'):4d} {cnt('miss_not_borne'):4d} {cnt('miss_no_gauges'):4d} {cnt('open'):4d}  {len(le):6d} {pct('30'):>4s} "
            f"{pct('60'):>4s} {pct('120'):>4s} {np.median(fo) if fo else float('nan'):8.2f} "
            f"{(str(int(np.median(off))) + 'm') if off else '-':>6s}")
    out("  lowR = miss, gauges show it (OUR RADAR LOW); notB = miss, gauges low too (not borne out); noG = miss, no gauge;"
        " lead: share of hits whose forecast issued 30/60/120 min before already showed it; fc/obs60: forecast/obs at +60 (median);"
        " offic.: median minutes from the warning's issue to our radar seeing it")
    og = {}
    for r in O:
        og.setdefault((r["dom"], r["c"] or "?"), []).append(r)
    if og:
        out(f"\nOur radar >= OWN_T mm in 1 h with no precipitation warning nearby (OWN_T = {OWN_T})")
        out(f"{'dom':3s} {'country':8s} {'cases':>5s} {'real':>5s} {'artef.':>6s} {'unver.':>6s} {'max mm/1h':>9s}")
        for key in sorted(og):
            g = og[key]
            cnt = lambda c: sum(1 for r in g if r["class"] == c)
            out(f"{key[0]:3s} {key[1][:8]:8s} {len(g):5d} {cnt('own_real'):5d} {cnt('own_artefact'):6d} {cnt('own_unverified'):6d} "
                f"{max(r['peak1h'] for r in g):9.1f}")
    return uniq, O


def cases(uniq, O, out=print, n=12):
    """The individual cases worth a look: misses with gauges, our-only cases."""
    out("\nMisses (our radar far below the threshold):")
    for r in sorted((r for r in uniq if r["class"].startswith("miss")), key=lambda r: r["ratio"])[:n]:
        g = r.get("gauges", {})
        out(f"  {r['class']:15s} {r['dom']} {r['c'][:12]:12s} {r.get('lat')},{r.get('lon')} {r['ev']:12s} L{r['lv']} {str(r['name'])[:32]:32s} {hstr(r['on'])}-{hstr(r['end'])}"
            f" crit {r['crit']} obs {r['obs']} ratio {r['ratio']} gauges n={g.get('n')} max={g.get('gmax')} radar@g={g.get('radar_at')} {g.get('gauge', '')}")
    out("\nOur-only cases:")
    for r in sorted(O, key=lambda r: -r["peak1h"])[:n]:
        g = r.get("gauges", {})
        out(f"  {r['class']:15s} {r['dom']} {r['c']:4s} {r['lat']:.2f} {r['lon']:.2f} {hstr(r['t0'])}-{hstr(r['t1'])[6:]} peak {r['peak1h']} mm/1h"
            f" (rate {r['rmax']}) px {r['px']} clutter {r['clutter']} gauges n={g.get('n')} max={g.get('gmax')} radar@g={g.get('radar_at')} {g.get('gauge', '')}")


# ---------- the collector: the Recorder outside the instances ----------
def collect(out, root, cache, interval=300, log=print):
    import shutil
    import official_warnings as ow
    import thunder as th
    ow.REFRESH_S = max(ow.REFRESH_S, 600)
    th.REFRESH_S = 300
    for sub in ("eu", "na"):
        os.makedirs(os.path.join(cache, sub), exist_ok=True)
    for src, dst in (("/data/eu/warnings/meteoalarm_geocodes.json", "eu"), ("/data/eu/warnings/dwd_cells.json", "eu"),
                     ("/data/na/warnings/nws_zones.json", "na")):
        if os.path.exists(src) and not os.path.exists(os.path.join(cache, dst, os.path.basename(src))):
            shutil.copy(src, os.path.join(cache, dst))
    w_eu = ow.Warnings(fetch, log, os.path.join(cache, "eu"), sources=["dwd", "meteoalarm"])
    w_na = ow.Warnings(fetch, log, os.path.join(cache, "na"), sources=["nws", "eccc"])
    t = th.Thunder(fetch, log)
    for obj in (w_eu, w_na, t):
        threading.Thread(target=obj.loop, daemon=True).start()
    recs = [Recorder(log, w_eu, t, PendingFc(os.path.join(root, "pending", "fcgrid"), "de"), "de", log_dir=out, cycle_s=interval),
            Recorder(log, None, None, PendingFc(os.path.join(root, "pending", "eugrid"), "eu"), "eu", log_dir=out),
            Recorder(log, w_na, None, PendingFc(os.path.join(root, "na", "pending", "eugrid"), "na"), "na",
                     log_dir=os.path.join(out, "na"), cycle_s=interval)]
    log(f"collect: writing to {out}")
    time.sleep(90)
    while True:
        for rec in recs:
            try:
                rec.step()
            except Exception as e:
                log(f"collect: {rec.name}: {e!r}")
        time.sleep(60)


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    sub = ap.add_subparsers(dest="cmd", required=True)
    c = sub.add_parser("check")
    c.add_argument("--day", default=None, help="UTC day YYYYMMDD (default: yesterday)")
    c.add_argument("--days", type=int, default=1, help="this many days ending with --day")
    c.add_argument("--domains", default="de,eu,eun,na")
    c.add_argument("--root", default=LOG_DIR)
    c.add_argument("--extra", action="append", default=[], help="another archive root (collect --out)")
    c.add_argument("--geocodes", default="/data/eu/warnings/meteoalarm_geocodes.json")
    c.add_argument("--nws-zones", default="/data/na/warnings/nws_zones.json")
    c.add_argument("--out", default=None, help="where warncheck_<day>.jsonl goes (default: --root)")
    s = sub.add_parser("summary")
    s.add_argument("--day", default=None)
    s.add_argument("--days", type=int, default=7)
    s.add_argument("--root", default=LOG_DIR)
    b1 = sub.add_parser("backfill-dwd")
    b1.add_argument("--out", required=True)
    b1.add_argument("--cache", required=True)
    b1.add_argument("--every", type=int, default=30)
    b2 = sub.add_parser("backfill-nws")
    b2.add_argument("--out", required=True)
    b2.add_argument("--cache", required=True)
    b2.add_argument("--since", required=True, help="ISO time, e.g. 2026-11-14T00:00:00Z")
    co = sub.add_parser("collect")
    co.add_argument("--out", required=True)
    co.add_argument("--root", default=LOG_DIR)
    co.add_argument("--cache", required=True)
    a = ap.parse_args()
    log = lambda m: print(m, flush=True)
    if a.cmd in ("check", "summary"):
        last = a.day or (datetime.now(timezone.utc) - timedelta(days=1)).strftime("%Y%m%d")
        days = [(datetime.strptime(last, "%Y%m%d") - timedelta(days=k)).strftime("%Y%m%d") for k in range(a.days - 1, -1, -1)]
    if a.cmd == "check":
        out = a.out or a.root
        for day in days:
            for dom in a.domains.split(","):
                t0 = time.time()
                rows = check_domain(dom, day, a.root, a.extra, a.geocodes, a.nws_zones, log)
                write_day(rows, dom, day, out)
                log(f"  {dom} {day}: {sum(1 for r in rows if r['k'] == 'warn')} warnings, "
                    f"{sum(1 for r in rows if r['k'] == 'own')} our-only cases, {len(rows)} rows, {time.time() - t0:.0f} s")
        uniq, O = summary(out, days, log)
        cases(uniq, O, log)
        import resource
        log(f"peak memory {resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024:.0f} MB")
    elif a.cmd == "summary":
        uniq, O = summary(a.root, days, log)
        cases(uniq, O, log)
    elif a.cmd == "backfill-dwd":
        backfill_dwd(a.out, a.cache, every_min=a.every, log=log)
    elif a.cmd == "backfill-nws":
        backfill_nws(a.out, a.cache, a.since, log=log)
    elif a.cmd == "collect":
        collect(a.out, a.root, a.cache, log=log)


if __name__ == "__main__":
    main()
