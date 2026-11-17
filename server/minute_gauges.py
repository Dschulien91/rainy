"""North America: rain gauges at 10-minute resolution - the truth for rain alerts and for the forecast's timing.

The hourly METAR / SWOB amounts (northamerica.MetarGauges) cannot score a rain start in a +-15 min window or a
forecast at +30 min on 10-minute pairs. Two open, keyless sources give 10-minute sums at ~1100 of the instance's
gauges - the SAME stations and ids as <log_dir>/metar/stations.json, so natobs / fcgauge / fcalert need no change:

  USA     the ASOS stations' 5-minute reports (NWS "high-frequency METAR", distributed by MADIS) as the Iowa
          Environmental Mesonet archives them: mesonet.agron.iastate.edu/cgi-bin/request/asos.py, report_type=1
          (5-minute) and 3 (routine), all stations in one request (no station list: <= 24 h per request). Free, no
          account; the service has a 1 s per-IP throttle and asks for one request per hour for all stations - this
          loader makes two (5-minute + routine reports) once per hour, ~1 MB. p01i of a 5-minute report = the
          precipitation since the station's last ROUTINE report (inches, 0.01 steps; "M" = no P group = nothing yet);
          the routine report (~:51-:56) carries the hour's total and starts the next count. -> a cumulative curve
          per station (only between two routine reports 40-80 min apart, never decreasing, ending at the routine
          total; the open hour after the newest routine report while it is monotonic) -> 10-minute sums ending at
          :00, :10, ... The curve at such a time is the report there, else linear between the two reports around it
          when they are <= 20 min apart (exact in dry spells; in rain it spreads the amount over those minutes - many
          stations send no 5-minute report at :50 / :55, next to the routine one). ~740 ASOS stations of the METAR
          list (Alaska, Hawaii are outside the MRMS box), steps of 0.254 mm. The reports reach the IEM ~15-20 min
          after their time.
          Stations whose newest METAR has no working sensor ("PNO", or no "AO2") are skipped (MetarGauges.precip_ok).
          Also the reports' present weather ("wxcodes": RA, DZ, SN, ... from the ASOS precipitation identifier,
          which reacts before the first 0.01 inch tip) per 10 minutes.
  Canada  ECCC SWOB minute records of MSC's automatic stations (MSC GeoMet OGC API, collection swob-realtime, the
          "...-AUTO-minute-swob" records; Environment and Climate Change Canada Data Servers End-use Licence: open,
          with attribution): pcpn_amt_pst10mts = precipitation of the 10 minutes ending at the record time (mm,
          weighing gauges, 0.1 mm), read at :00, :10, ... ~330 stations, a few minutes behind, the API keeps 30 days.
          One request per 10-minute stamp, read when it is 12 min old and once more after 70 min (late records).

Archive (the format of check_national_gauges.archive_truth, so its readers work unchanged):
  <log_dir>/gaugetruth/<YYYYMMDD>.npz   ids "us:KDSM" / "ca:CA-71xxx" (network:id as in stations.json),
      mm  float32 (n, 144): mm in the 10 minutes ending 00:00, 00:10, ... UTC (NaN = unknown)
      wx  uint8 (n, 144): 1 = a 5-minute report in those 10 minutes had precipitation as present weather, 0 = reports
          without, 255 = no report (and all of Canada)
  kept 60 days (check_national_gauges.TRUTH_DAYS).

Server: rainy_server starts MinuteGauges(...).loop() in the North America instance (env NA_MINUTE_GAUGES=0: off).
Readers: check_na_gauges.py (10-minute tables), verify_alerts_eu.py and score_alert_prob.py (alerts in North
America), score_competitors.py --na (ECCC on 10-minute pairs).

Backfill / test without the server (writes <dir>/gaugetruth; --stations: a metar/stations.json):
  python minute_gauges.py --backfill HOURS --dir <log_dir> [--stations <log_dir>/metar/stations.json] [--no-ca]
"""
import calendar
import json
import os
import sys
import time

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

TEN = 600_000
DAY = 86400_000
IEM_ASOS = "https://mesonet.agron.iastate.edu/cgi-bin/request/asos.py"
SWOB_MINUTE_URL = ("https://api.weather.gc.ca/collections/swob-realtime/items?f=json&limit=3000&datetime={t}"
                   "&properties=msc_id-value,stn_id-value,date_tm-value,pcpn_amt_pst10mts,data_pvdr-value")
SWOB_PROVIDERS = {"MSC", "NAV CANADA", "DND"}   # as northamerica.SWOB_PROVIDERS (partner networks: rights reserved)
WX_PRECIP = ("DZ", "RA", "SN", "SG", "PL", "GR", "GS", "UP")
INCH = 25.4
GAP = 20 * 60_000          # USA: the curve between two reports at most this far apart is interpolated
KEEP_DAYS = 60


def _iso(ms):
    return time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime(ms / 1000))


def wx_precip(codes):
    """Present weather with precipitation at the station: RA, -SHRA, FZDZ, SN, ... (not VCSH = in the vicinity, not
    BLSN / DRSN = snow blown up from the ground)."""
    for tok in codes.split():
        tok = tok.lstrip("+-")
        if tok.startswith(("VC", "BL", "DR")):
            continue
        if any(p in tok for p in WX_PRECIP):
            return True
    return False


# ---------- USA: the IEM's copy of the ASOS 5-minute reports ----------
def iem_rows(raw):
    """(IEM station id, report time ms, p01i inches (no P group = 0), wxcodes text) of an asos.py CSV answer."""
    import csv
    rows = csv.reader(raw.decode("utf-8", "replace").splitlines())
    head = next(rows, None)
    if not head or head[:2] != ["station", "valid"]:
        raise ValueError(f"unexpected answer: {raw[:200]!r}")
    i_p = head.index("p01i")
    i_w = head.index("wxcodes") if "wxcodes" in head else None
    days = {}
    for r in rows:
        if len(r) < len(head):
            continue
        v = r[1]                                   # "2026-11-17 08:35" (tz=UTC)
        d = days.get(v[:10])
        if d is None:
            try:
                d = days[v[:10]] = calendar.timegm(time.strptime(v[:10], "%Y-%m-%d")) * 1000
            except ValueError:
                continue
        try:
            t = d + (int(v[11:13]) * 60 + int(v[14:16])) * 60_000
        except ValueError:
            continue
        p = r[i_p]
        try:
            inch = 0.0 if p in ("M", "") else float(p)
        except ValueError:
            continue
        if inch < 0.005:                           # "trace" (0.0001) and nothing: 0, the count steps by 0.01
            inch = 0.0
        w = r[i_w] if i_w is not None else ""
        yield r[0], t, inch, ("" if w == "M" else w)


def us_ten_minute(hf_raw, routine_raw, ok):
    """5-minute and routine reports (iem_rows answers) -> {"us:<ICAO>": {end of the 10 minutes ms: (mm or NaN,
    wx 0 / 1 / 255)}} for the stations ok(icao) accepts. See the module docstring for the rules."""
    rt, hf = {}, {}
    for sid, t, inch, _ in iem_rows(routine_raw):
        rt.setdefault(sid, {})[t] = inch
    for sid, t, inch, w in iem_rows(hf_raw):
        hf.setdefault(sid, {})[t] = (inch, w)
    out = {}
    for sid, reps in hf.items():
        icao = "K" + sid if len(sid) == 3 else sid
        if not ok(icao):
            continue
        routine = sorted(rt.get(sid, {}).items())
        times = sorted(t for t in reps if t not in rt.get(sid, {}))   # a 5-min report at a routine minute: ambiguous
        cum = {}                                   # time -> (chain, inches since the chain's start)
        chain, base, prev_end = 0, 0.0, None

        def segment(r0, r1, p1):
            """The reports inside (r0, r1) as inches since r0; None when the count is not plausible."""
            inside = [(t, reps[t][0]) for t in times if r0 < t < (r1 if r1 is not None else r0 + 70 * 60_000)]
            vals = [0.0] + [v for _, v in inside] + ([p1] if r1 is not None else [])
            if any(b < a - 1e-6 for a, b in zip(vals, vals[1:])):
                return None
            return inside

        for k, (r0, _) in enumerate(routine):
            nxt = routine[k + 1] if k + 1 < len(routine) else None
            if nxt is not None and not 40 * 60_000 <= nxt[0] - r0 <= 80 * 60_000:
                prev_end = None
                continue
            inside = segment(r0, *(nxt if nxt else (None, None)))
            if inside is None:
                prev_end = None
                continue
            if prev_end != r0:
                chain, base = chain + 1, 0.0
            cum[r0] = (chain, base)
            for t, v in inside:
                cum[t] = (chain, base + v)
            if nxt is None:
                break
            base += nxt[1]
            cum[nxt[0]] = (chain, base)
            prev_end = nxt[0]
        # the curve at :00, :10, ...: the report there, else linear between the reports around it when they are at
        # most GAP apart (exact when nothing fell between them; in rain it spreads the amount, as the Europe archive
        # does with 15-minute sums - many stations send no 5-minute report at :50 / :55, next to the routine one)
        ser = {}
        for c in sorted({c for c, _ in cum.values()}):
            pts = sorted((t, v) for t, (cc, v) in cum.items() if cc == c)
            tt = np.array([p[0] for p in pts], np.int64)
            vv = np.array([p[1] for p in pts])
            grid = np.arange(-(-tt[0] // TEN) * TEN, tt[-1] + 1, TEN, dtype=np.int64)
            if len(grid) < 2:
                continue
            j = np.searchsorted(tt, grid)              # tt[j - 1] < T <= tt[j]
            exact = tt[np.minimum(j, len(tt) - 1)] == grid
            good = exact | ((j > 0) & ((tt[np.minimum(j, len(tt) - 1)] - tt[np.maximum(j - 1, 0)]) <= GAP))
            cg = np.interp(grid, tt, vv)
            for k in np.flatnonzero(good[1:] & good[:-1]):
                ser[int(grid[k + 1])] = [round(float(cg[k + 1] - cg[k]) * INCH, 2), 255]
        for t in times:                            # present weather of the reports in (T - 10, T]
            e = -(-t // TEN) * TEN
            x = ser.setdefault(e, [float("nan"), 255])
            x[1] = max(x[1] if x[1] != 255 else 0, int(wx_precip(reps[t][1])))
        if ser:
            out[f"us:{icao}"] = {t: tuple(v) for t, v in ser.items()}
    return out


# ---------- Canada: SWOB minute records ----------
def swob_minute_rows(raw):
    """("CA-<id>", end of the 10 minutes ms, mm) of one stamp's SWOB records (only the minute records carry
    pcpn_amt_pst10mts)."""
    for f in json.loads(raw).get("features", []):
        p = f.get("properties", {})
        sid = p.get("msc_id-value") or p.get("stn_id-value")
        mm = p.get("pcpn_amt_pst10mts")
        if p.get("data_pvdr-value") not in SWOB_PROVIDERS or not sid or mm is None:
            continue
        try:
            t = calendar.timegm(time.strptime(p["date_tm-value"][:19], "%Y-%m-%dT%H:%M:%S")) * 1000
            mm = float(mm)
        except (KeyError, TypeError, ValueError):
            continue
        if t % TEN == 0 and mm >= 0:
            yield "CA-" + sid, t, mm


# ---------- archive ----------
def save(log_dir, values):
    """{key: {t: (mm, wx)}} -> merged into <log_dir>/gaugetruth/<UTC day>.npz (known values never become unknown).
    Returns the number of values written."""
    d = os.path.join(log_dir, "gaugetruth")
    os.makedirs(d, exist_ok=True)
    by_day = {}
    for key, ser in values.items():
        for t, (mm, wx) in ser.items():
            by_day.setdefault(t // DAY, []).append((key, (t % DAY) // TEN, mm, wx))
    n = 0
    for day, items in by_day.items():
        fp = os.path.join(d, time.strftime("%Y%m%d", time.gmtime(day * 86400)) + ".npz")
        try:
            z = np.load(fp)
            ids = [str(x) for x in z["ids"]]
            mm = z["mm"].astype(np.float32)
            wx = z["wx"].astype(np.uint8) if "wx" in z.files else np.full(mm.shape, 255, np.uint8)
        except (OSError, ValueError, KeyError):
            ids, mm, wx = [], np.zeros((0, 144), np.float32), np.zeros((0, 144), np.uint8)
        index = {k: i for i, k in enumerate(ids)}
        new = sorted({it[0] for it in items} - set(index))
        if new:
            index.update({k: len(ids) + i for i, k in enumerate(new)})
            ids += new
            mm = np.concatenate([mm, np.full((len(new), 144), np.nan, np.float32)])
            wx = np.concatenate([wx, np.full((len(new), 144), 255, np.uint8)])
        for key, b, v, w in items:
            i = index[key]
            if v == v:                     # not NaN
                mm[i, b] = v
                n += 1
            if w != 255:
                wx[i, b] = w
        tmp = fp[:-4] + ".tmp.npz"
        np.savez_compressed(tmp, ids=np.array(ids), mm=mm, wx=wx)
        os.replace(tmp, fp)
    old = time.time() - KEEP_DAYS * 86400
    for name in os.listdir(d):
        if name.endswith(".npz") and os.path.getmtime(os.path.join(d, name)) < old:
            os.remove(os.path.join(d, name))
    return n


# ---------- the loader ----------
class MinuteGauges:
    US_EVERY_S = 3600          # the IEM's advice: one request per hour for all stations
    US_AT_MIN = 25             # ... at :25 (the 5-minute reports of :00-:10 have arrived)
    US_HOURS = 3               # per request: the last 3 hours (late reports, the hour's routine report)
    CA_FIRST_S, CA_AGAIN_S = 12 * 60, 70 * 60
    CA_HOURS = 3
    PAUSE_S = 1.5              # between two requests to the same service

    def __init__(self, fetch, log_dir, log, stations, precip_ok=None):
        """stations: callable -> [(network, id, lat, lon)] (metar/stations.json); precip_ok: {ICAO: bool} kept by
        MetarGauges (None: every listed station counts)."""
        self.fetch, self.log_dir, self.log, self.stations, self.precip_ok = fetch, log_dir, log, stations, precip_ok
        self.us_at = 0.0
        self.ca_reads = {}     # stamp ms -> reads with records so far
        self.ca_empty = {}     # stamp ms -> answers without records (3: given up)
        self.ca_count = [0, 0]  # values, wet values since the last log line
        self.ca_failed_at = 0.0

    def _ok_us(self):
        known = {s[1] for s in self.stations() if s[0] == "us"}
        pok = self.precip_ok if self.precip_ok is not None else {}
        return lambda icao: icao in known and pok.get(icao, True)

    def poll_us(self, span):
        """span: "hours=3" or "sts=...&ets=..." (<= 24 h). Returns (gauges, values, wet values)."""
        hf = self.fetch(f"{IEM_ASOS}?data=p01i&data=wxcodes&report_type=1&tz=UTC&{span}", timeout=180)
        time.sleep(self.PAUSE_S)
        rt = self.fetch(f"{IEM_ASOS}?data=p01i&report_type=3&tz=UTC&{span}", timeout=180)
        vals = us_ten_minute(hf, rt, self._ok_us())
        n = save(self.log_dir, vals)
        wet = sum(1 for s in vals.values() for mm, _ in s.values() if mm >= 0.05)
        return len(vals), n, wet

    def poll_canada(self, hours=None, now=None):
        """The 10-minute stamps of the last hours that are due (first read / second read). Returns values written."""
        now = now or time.time() * 1000
        known = {s[1] for s in self.stations() if s[0] == "ca"}
        n = 0
        last = int(now - self.CA_FIRST_S * 1000) // TEN * TEN
        for t in range(last - int((hours or self.CA_HOURS) * 3600_000) // TEN * TEN, last + 1, TEN):
            k = self.ca_reads.get(t, 0)
            if k >= 2 or (k == 1 and now - t < self.CA_AGAIN_S * 1000):
                continue
            rows = [(sid, tt, mm) for sid, tt, mm in swob_minute_rows(self.fetch(SWOB_MINUTE_URL.format(t=_iso(t)), timeout=120))
                    if tt == t and sid in known]
            if rows:
                self.ca_reads[t] = k + 1
            else:                                     # nothing yet (an API gap): again next time, 3 times at most
                self.ca_empty[t] = self.ca_empty.get(t, 0) + 1
                if self.ca_empty[t] >= 3:
                    self.ca_reads[t] = 2
            got = save(self.log_dir, {f"ca:{sid}": {t: (mm, 255)} for sid, _, mm in rows})
            n += got
            self.ca_count[0] += got
            self.ca_count[1] += sum(mm >= 0.05 for _, _, mm in rows)
            time.sleep(self.PAUSE_S)
        self.ca_reads = {t: k for t, k in self.ca_reads.items() if t > now - 2 * DAY}
        self.ca_empty = {t: k for t, k in self.ca_empty.items() if t > now - 2 * DAY}
        return n

    def loop(self):
        time.sleep(240)        # MetarGauges reads the METAR cache first (precip_ok)
        first = True
        while True:
            try:
                self.poll_canada(6 if first else None)
            except Exception as e:
                if time.time() - self.ca_failed_at > 3600:   # an outage: one line per hour, retried every 2 min
                    self.ca_failed_at = time.time()
                    self.log(f"minute gauges (Canada) failed: {e}")
            now = time.time()
            if first or (now - self.us_at >= self.US_EVERY_S - 300 and now % 3600 >= self.US_AT_MIN * 60):
                self.us_at = now
                try:
                    g, n, wet = self.poll_us(f"hours={6 if first else self.US_HOURS}")
                    self.log(f"minute gauges: USA {g} gauges, {n} 10-min values ({wet} wet), Canada {self.ca_count[0]} "
                             f"values ({self.ca_count[1]} wet) since the last line, {time.time() - now:.0f} s")
                    self.ca_count = [0, 0]
                except Exception as e:
                    self.us_at = now - self.US_EVERY_S + 900     # again in ~15 min
                    self.log(f"minute gauges (USA) failed: {e}")
            first = False
            time.sleep(120)


# ---------- backfill / test ----------
def _test_fetch(url, timeout=60, tries=3):
    import urllib.request
    req = urllib.request.Request(url, headers={"User-Agent": "Mozilla/5.0"})
    for attempt in range(tries):
        try:
            with urllib.request.urlopen(req, timeout=timeout) as r:
                return r.read()
        except OSError:
            if attempt == tries - 1:
                raise
            time.sleep(10 * (attempt + 1))


def main():
    args = sys.argv[1:]
    opt = lambda n, d=None: args[args.index(n) + 1] if n in args else d
    if "--backfill" not in args:
        sys.exit(__doc__)
    hours = float(opt("--backfill"))
    log_dir = opt("--dir", ".")
    st_path = opt("--stations", os.path.join(log_dir, "metar", "stations.json"))
    with open(st_path) as f:
        st = [tuple(s) for s in json.load(f)]
    import northamerica
    try:
        pok = northamerica.precip_status(_test_fetch(northamerica.METAR_URL))
    except Exception as e:
        print(f"(METAR cache: {e}; every listed station counts)")
        pok = None
    m = MinuteGauges(_test_fetch, log_dir, print, lambda: st, pok)
    now = int(time.time() * 1000)
    t0 = time.time()
    # USA: 24-hour pieces (the IEM's limit without a station list), overlapping by 2 h for the routine reports
    end = now
    while end > now - hours * 3600_000:
        start = max(end - 24 * 3600_000, now - int(hours * 3600_000) - 2 * 3600_000)
        g, n, wet = m.poll_us(f"sts={_iso(start)[:16]}Z&ets={_iso(end)[:16]}Z")
        print(f"USA {_iso(start)} - {_iso(end)}: {g} gauges, {n} values, {wet} wet ({time.time() - t0:.0f} s)")
        end = start + 2 * 3600_000
        if start <= now - hours * 3600_000 - 2 * 3600_000:
            break
        time.sleep(m.PAUSE_S)
    if "--no-ca" not in args:
        n = m.poll_canada(hours, now)
        print(f"Canada: {n} values, {m.ca_count[1]} wet, {len(m.ca_reads)} stamps ({time.time() - t0:.0f} s)")


if __name__ == "__main__":
    main()
