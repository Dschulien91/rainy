"""Official open nowcasts ("competitors") at our rain gauges, recorded next to our own forecast (16.11.2026; user:
"auf Augenhoehe mit den Wettbewerbern, vor allem die 120 Minuten Vorhersage").

Wherever a national weather service publishes its own radar nowcast as open data, this module records that nowcast
at the gauges of check_national_gauges.py (Europe) / northamerica.MetarGauges (North America), once per hour (CHMI, KNMI
and ECCC twice), as a user would have got it: the provider's run at a full hour, its forecast for +30 / +60 / +90 / +120 min.
score_competitors.py joins these rows with OUR forecast of the same run time at the same gauges (verify_log/fcgauge,
the instances' record_fc) and the gauge truth (verify_log/gaugetruth, metar/) and prints the scores per country.

Providers (all read anonymously through the server's fetch(); no account, no key - KNMI: its public anonymous key):
  metno   MET Norway radar nowcast (the one Yr shows; Nordic composite NO/SE/FI + DK/Baltic edges), CC BY 4.0.
          THREDDS radarnowcasting/yrwms-nordic...nordiclcc-1000.<YYYYMMDDTHHMM00Z>.nc: CF NetCDF4, lwe_precipitation_rate
          (mm/h, instantaneous), Lambert conformal 1 km 1694 x 2134, 24 steps 0..+115 min every 5 min, a new run every
          5 min, ~5-6 min after its time, kept ~2 days. The ~85 MB files are NOT downloaded: only boxes around the gauges
          are read through OPeNDAP (.dods, binary), grouped in 100-km tiles, two requests per tile (steps 6/12/18 =
          +30/+60/+90 and step 23 = +115, recorded as "+120" with dt 115). NCSS subsetting answers 503 (off).
  incabe  RMI Belgium INCA-BE (BE / NL / LU / N France / W Germany), WCS on opendata.meteo.be, GeoTIFF float64
          EPSG:4326 756 x 453 (0.011885 deg), every 10 min from the analysis to +3 h 50, ONLY the current run (served
          ~3 min after its time, replaced by the next run 10 min later) -> taken within minutes. -1 = dry (there is no
          "outside the radar" value: the whole rectangle has numbers). Deflate-compressed TIFF
          (geotiff:compression=Deflate: 63 KB instead of 7.3 MB). Unit undocumented: taken as mm per 10 min (see
          INCA_BE_FACTOR; rows carry "scale"). Licence: the catalogue
          record RMI_DATASET_INCA says CC BY 4.0 but also "This product is not publicly available yet" - unconfirmed.
  chmi    CHMI COTREC extrapolation of the 2-km pseudo-CAPPI (Czechia + surroundings), tar of six ODIM HDF5 files
          (+10 .. +60 min), DBZH uint8 (gain 0.5, offset -32, 255 = no data, 0 = undetect), Mercator 598 x 378
          (1555.7 m at the equator = ~1 km), a run every 5 min, its name = the END of the 5-min radar interval,
          on the server within ~1 min.
          Rate by Marshall-Palmer Z = 200 R^1.6 (CHMI's radar_description_en.pdf: the pseudo-CAPPI "after conversion
          using the Marshall-Palmer Z-R relation, is operationally used" for surface rain intensity), echoes below 7 dBZ
          (= 0.1 mm/h; CHMI's own "precipitation reaches the ground" threshold in the same document) count as dry.
  meander HungaroMet MEANDER nowcast (Hungary + borders), odp.met.hu, zipped NetCDF3 classic per variable and step
          (read with the small reader below: h5py cannot read NetCDF3), hourly runs, 15-min steps to +2 h, published
          ~10 min after the hour, kept ~5-6 h. sumRadPrec = radar-based precipitation ACCUMULATED since the run start
          (mm; monotonic, all zero at +0) -> mean rate of the 30 min centred on each lead (+120: the last 15 min).
          cloudines (okta) is recorded too (var "cloud", 0..1) at the SYNOP stations - for a later sunny benchmark.
  eccc    ECCC North American radar extrapolation (pySTEPS-based, 1 km), WMS GetFeatureInfo per point on
          geo.weather.gc.ca, layer Radar_1km_RainPrecipRate-Extrapolation (mm/h, the rain assumption everywhere;
          "Undetected" = 0, "Search returned no results" = outside), runs every 6 min to +72 min, ~20 min after their
          time, kept 3 h. One request per point and time: a fixed, spread-out sample of up to SAMPLE_PER_COUNTRY US
          and Canadian gauges (per 1-degree cell a gauge with 10-minute truth first, minute_gauges.py), the :00 run
          for +30 and the :30 run for +60 (both valid at :30, where the METAR / SWOB hours are centred), <= 4
          requests/s. The run is chosen with the WMS dimension parameter DIM_REFERENCE_TIME (17.11.2026): GeoMet
          IGNORES a plain "reference_time=" and answers from the NEWEST run, checking "time" against that run's
          range - rows before then (no "ref" field) hold the newest run at reading time (~:12-:18 for the "+30" of
          :00, i.e. a lead of ~15 min, ~45 min for the "+60") and failed with "time outside valid hours" whenever a
          run was read after a newer one covered its valid time (restarts, retries). score_competitors.py skips them.
          The snow layer (Radar_1km_SnowPrecipRate-Extrapolation) is in cm/h of
          SNOW depth and returns values wherever there are echoes (same reflectivity, snow Z-S), not a phase-resolved
          water equivalent -> not used.
  knmi    KNMI precipitation nowcast (operational pySTEPS, deterministic, from the real-time 5-min accumulation
          RTCOR-5m; NL + BE / NW Germany within the Dutch / Belgian / German radars' reach), KNMI Data Platform open
          data API, dataset radar_forecast version 2.0 (files RAD_NL25_RAC_FM_<YYYYMMDDHHMM>.h5, ~0.7 MB, a run every
          5 min, published ~3-4 min after its time). Licence CC BY 4.0, attribution: "Source: KNMI (Royal Netherlands
          Meteorological Institute)". KNMI HDF5: image1 .. image25, image_datetime_valid = run + 0 .. +120 min, each
          the precipitation SUM of the 5 min ending at that time (uint16, image<N>/calibration "GEO=0.01*PV+0" mm;
          65534 = missing, 65535 = outside the image) -> x 12 = the mean rate (mm/h) of the 5 min before the lead.
          Grid 700 x 765 cells of 1 km, polar stereographic (geographic/map_projection proj4 string, in km), cell
          (row, col) has its outer corner at x = (col + geo_column_offset) * geo_pixel_size_x, y = (row +
          geo_row_offset) * geo_pixel_size_y. Also +0 is recorded (KNMI's own radar picture, not scored as a
          forecast). The API needs a key: KNMI's PUBLISHED anonymous key (KNMI_ANON_KEY, shared by all unregistered
          users, rate-limited, replaced periodically - env KNMI_KEY overrides it); no account. Per take two requests:
          the API's temporary download URL of the run's file, then the file (S3). At most one take per 15 min (runs
          :00 and :30); a 429 (the shared 50/min are often gone for a minute) is retried twice 30 s later, then
          429 / 403 back off for 15 min .. 2 h (logged).

Each provider value at a gauge is the mean over a box of +-FOOT_KM around it ("v"; our fcgauge value is the 5 x 5 mean
of the 2-km grid = ~10 x 10 km) and the cell under the gauge ("pt"). ECCC has only the point.
Rows: <log_dir>/bench/<provider>_<YYYYMMDD of the run>.jsonl, ONE LINE PER RUN (the gauge ids once, not per lead):
  {"p": provider, "run": run ms, "got": ms when read, "fp": box half-width km, "leads": [30, 60, ...] nominal minutes,
   "dt": [30, 60, ...] the provider's actual step, "ids": ["network:id", ...], "v": [[mm/h or null per id] per lead],
   "pt": [[...] per lead]}   (+ "var": "cloud" for MEANDER's cloud cover, values 0..1; KNMI: lead 0 too, and "pub" =
   ms when KNMI published the file)

Server: CompetitorBench(fetch, log, log_dir, stations, na=False).loop() in its own thread - one download at a time,
every request with a timeout, every take with a time budget, a failing provider is retried within its window and
never blocks the others for longer than its budget.
Test without the server:  python bench_competitors.py --once metno|incabe|chmi|knmi|meander|eccc [--stations f.json] [--dir d]
"""
import calendar
import hashlib
import io
import json
import os
import re
import struct
import sys
import tarfile
import time
import urllib.error
import zipfile
from datetime import datetime, timezone

import numpy as np

LEADS = (30, 60, 90, 120)
FOOT_KM = 5.0          # half-width of the box averaged around a gauge (ours: 5 x 5 cells of 2 km)
KEEP_DAYS = 60
H_MS = 3600_000
MIN_MS = 60_000


class NotReady(Exception):
    """The run is not (yet) available: try again later within the window."""


class Gone(Exception):
    """The run can no longer be read (replaced): give it up."""


class OverBudget(Exception):
    """A take ran longer than its provider's BUDGET_S."""


def _iso(ms, fmt="%Y-%m-%dT%H:%M:%SZ"):
    return datetime.fromtimestamp(ms / 1000, timezone.utc).strftime(fmt)


def _ms_iso(text):
    return int(datetime.strptime(text[:19], "%Y-%m-%dT%H:%M:%S").replace(tzinfo=timezone.utc).timestamp() * 1000)


def _key(s):
    return f"{s[0]}:{s[1]}"


def _round(a):
    return [None if not np.isfinite(x) else round(float(x), 2) for x in a]


def sample(grid, fr, fc, hr, hc):
    """grid (ny, nx) float32, NaN = no data; fr / fc fractional CENTRE indices of the gauges (row 0 = first row).
    -> (box mean over +-hr rows / +-hc cols, the nearest cell); NaN outside, or where any box cell has no data."""
    ny, nx = grid.shape
    r, c = np.round(fr).astype(int), np.round(fc).astype(int)
    v = np.full(len(r), np.nan, np.float32)
    p = np.full(len(r), np.nan, np.float32)
    for k in range(len(r)):
        if 0 <= r[k] < ny and 0 <= c[k] < nx:
            p[k] = grid[r[k], c[k]]
            if hr <= r[k] < ny - hr and hc <= c[k] < nx - hc:
                b = grid[r[k] - hr:r[k] + hr + 1, c[k] - hc:c[k] + hc + 1]
                if np.isfinite(b).all():
                    v[k] = b.mean()
    return v, p


def mp_rate(dbz, dry_dbz=7.0):
    """Marshall-Palmer Z = 200 R^1.6 (mm/h); NaN stays NaN; below dry_dbz = 0."""
    with np.errstate(invalid="ignore"):
        r = (10 ** (dbz / 10.0) / 200.0) ** (1 / 1.6)
        return np.where(np.isnan(dbz), np.nan, np.where(dbz >= dry_dbz, r, 0.0)).astype(np.float32)


# ---------- a minimal NetCDF3 classic reader (MEANDER; the container has h5py, which reads only NetCDF4) ----------
_NC_TYPES = {1: ">i1", 2: "S1", 3: ">i2", 4: ">i4", 5: ">f4", 6: ">f8"}


def nc3_read(b):
    """(global attributes, {variable: (array, attributes)}) of a NetCDF3 classic / 64-bit-offset file in memory.
    Record variables: the first record only (MEANDER files hold one)."""
    if b[:3] != b"CDF" or b[3] not in (1, 2):
        raise ValueError("not a NetCDF3 classic file")
    off64 = b[3] == 2
    pos = [8]  # magic + numrecs

    def u32():
        v = struct.unpack(">I", b[pos[0]:pos[0] + 4])[0]
        pos[0] += 4
        return v

    def name():
        n = u32()
        s = b[pos[0]:pos[0] + n].decode("utf-8", "replace")
        pos[0] += (n + 3) // 4 * 4
        return s

    def attrs():
        tag, n = u32(), u32()
        out = {}
        for _ in range(n if tag == 12 else 0):
            k, t, m = name(), u32(), u32()
            size = np.dtype(_NC_TYPES[t]).itemsize * m
            raw = b[pos[0]:pos[0] + size]
            pos[0] += (size + 3) // 4 * 4
            out[k] = raw.decode("utf-8", "replace") if t == 2 else np.frombuffer(raw, _NC_TYPES[t]).copy()
            if t != 2 and m == 1:
                out[k] = out[k][0]
        return out

    tag, n = u32(), u32()
    dims = []
    for _ in range(n if tag == 10 else 0):
        dims.append((name(), u32()))
    gatt = attrs()
    tag, n = u32(), u32()
    var = {}
    for _ in range(n if tag == 11 else 0):
        vname = name()
        ids = [u32() for _ in range(u32())]
        vatt = attrs()
        t, _vsize = u32(), u32()
        begin = struct.unpack(">Q", b[pos[0]:pos[0] + 8])[0] if off64 else u32()
        if off64:
            pos[0] += 8
        shape = [dims[i][1] or 1 for i in ids]  # an unlimited dimension: one record
        count = int(np.prod(shape)) if shape else 1
        dt = np.dtype(_NC_TYPES[t])
        a = np.frombuffer(b[begin:begin + count * dt.itemsize], dt).reshape(shape)
        var[vname] = (a, vatt)
    return gatt, var


# ---------- providers ----------
class Provider:
    name = ""
    OFFSETS = (0,)            # runs taken: minutes past the hour
    START_MIN, END_MIN = 10, 50   # window after the run time in which it is taken
    RETRY_S = 120
    BUDGET_S = 300
    na = False

    def __init__(self, fetch, log):
        self.fetch, self.log = fetch, log
        self.done = set()
        self.next_try = 0.0
        self.deadline = None
        self.requests = 0

    def runs(self, now_ms):
        """Run times whose window is open now, newest first."""
        out = []
        for h in range(int(now_ms // H_MS) - 3, int(now_ms // H_MS) + 1):
            for m in self.OFFSETS:
                r = h * H_MS + m * MIN_MS
                if r + self.START_MIN * MIN_MS <= now_ms < r + self.END_MIN * MIN_MS:
                    out.append(r)
        return sorted(out, reverse=True)

    def due(self, now_ms):
        if time.time() < self.next_try:
            return None
        return next((r for r in self.runs(now_ms) if r not in self.done), None)

    def get(self, url, timeout=40, tries=1):
        if self.deadline and time.time() > self.deadline:
            raise OverBudget(f"{self.name}: take ran over its {self.BUDGET_S} s budget")
        self.requests += 1
        return self.fetch(url, timeout=timeout, tries=tries)

    def select(self, stations):
        """The gauges inside the provider's domain: list of station tuples."""
        return list(stations)

    def take(self, run, stations):
        raise NotImplementedError

    def row(self, run, st, leads, dt, v, pt, fp=FOOT_KM, **extra):
        return {"p": self.name, "run": int(run), "got": int(time.time() * 1000), "fp": fp, "leads": list(leads),
                "dt": list(dt), "ids": [_key(s) for s in st], "v": [_round(x) for x in v],
                "pt": [_round(x) for x in pt], **extra}


class MetNo(Provider):
    """MET Norway's Nordic radar nowcast through OPeNDAP (see the module docstring)."""
    name = "metno"
    BASE = "https://thredds.met.no/thredds/dodsC/radarnowcasting/"
    FILE = "yrwms-nordic.mos.pcappi-0-rr.noclass-clfilter-novpr-clcorr-block.nordiclcc-1000.{t}.nc"
    PROJ = "+proj=lcc +lat_0=63 +lon_0=15 +lat_1=63 +lat_2=63 +no_defs +R=6.371e+06"
    START_MIN, END_MIN = 36, 58     # after INCA-BE / CHMI / MEANDER; the files stay for days
    BUDGET_S = 600
    TILE = 100                      # km (= cells): gauges are read in boxes of at most this size
    FILL = 1e30
    STEPS = ((6, 30, 30), (12, 60, 60), (18, 90, 90), (23, 120, 115))   # (index, nominal lead, actual minutes)

    def __init__(self, fetch, log):
        super().__init__(fetch, log)
        import pyproj
        self.proj = pyproj.Proj(self.PROJ)
        self.mask, self.mask_day = None, None   # coarse coverage (every 10th cell), refreshed daily

    def url(self, run):
        return self.BASE + self.FILE.format(t=_iso(run, "%Y%m%dT%H%M00Z"))

    @staticmethod
    def _q(s):
        return s.replace("[", "%5B").replace("]", "%5D")

    @staticmethod
    def dods(raw):
        """The float32 values of a one-variable .dods answer (big-endian, after the DDS text)."""
        i = raw.index(b"\nData:\n") + 7
        n = struct.unpack(">I", raw[i:i + 4])[0]
        return np.frombuffer(raw[i + 8:i + 8 + 4 * n], ">f4").astype(np.float32)

    def axes(self, run):
        try:
            text = self.get(self._q(self.url(run) + ".ascii?time,Xc[0:1:1],Yc[0:1:1],forecast_reference_time"),
                            timeout=60).decode("utf-8", "replace")
        except urllib.error.HTTPError as e:
            raise NotReady(f"no file yet ({e.code})")
        except OSError as e:
            raise NotReady(str(e))
        nums = lambda label: [float(x) for x in re.search(label + r"\[\d+\]\s*\n([^\n]+)", text).group(1).split(",")]
        t, x, y = nums(r"\ntime"), nums(r"\nXc"), nums(r"\nYc")
        if int(t[0]) * 1000 != run or len(t) < 24 or t[1] - t[0] != 300:
            raise Gone(f"unexpected time axis {t[:3]}")
        return x[0], x[1] - x[0], y[0], y[1] - y[0]

    def select(self, stations):
        x, y = self.proj(np.array([s[3] for s in stations]), np.array([s[2] for s in stations]))
        self.xy = dict(zip((_key(s) for s in stations), zip(x, y)))
        ok = (x > -796000) & (x < 897000) & (y > -1008000) & (y < 1125000)
        return [s for s, o in zip(stations, ok) if o]

    def coverage(self, run, x0, dx, y0, dy, st):
        """Gauges whose 10-km neighbourhood is inside the radar coverage (a coarse read of step 0, once a day)."""
        day = run // 86400_000
        if self.mask is None or self.mask_day != day:
            raw = self.get(self._q(self.url(run) + ".dods?lwe_precipitation_rate.lwe_precipitation_rate"
                                   "[0][0:10:2133][0:10:1693]"), timeout=120)
            self.mask, self.mask_day = (self.dods(raw) < self.FILL).reshape(214, 170), day
        out = []
        for s in st:
            x, y = self.xy[_key(s)]
            r, c = int(round((y - y0) / dy / 10)), int(round((x - x0) / dx / 10))
            if 0 <= r < 214 and 0 <= c < 170 and self.mask[max(0, r - 1):r + 2, max(0, c - 1):c + 2].any():
                out.append(s)
        return out

    def take(self, run, stations):
        x0, dx, y0, dy = self.axes(run)
        st = self.coverage(run, x0, dx, y0, dy, self.select(stations))
        if not st:
            return []
        fr = np.array([(self.xy[_key(s)][1] - y0) / dy for s in st])
        fc = np.array([(self.xy[_key(s)][0] - x0) / dx for s in st])
        h = int(round(FOOT_KM / (abs(dx) / 1000)))
        v = np.full((len(self.STEPS), len(st)), np.nan, np.float32)
        p = v.copy()
        tiles = {}
        for k in range(len(st)):
            tiles.setdefault((int(fr[k] // self.TILE), int(fc[k] // self.TILE)), []).append(k)
        for ks in tiles.values():
            r0 = max(0, int(round(fr[ks].min())) - h)
            r1 = min(2133, int(round(fr[ks].max())) + h)
            c0 = max(0, int(round(fc[ks].min())) - h)
            c1 = min(1693, int(round(fc[ks].max())) + h)
            for times, which in (("[6:6:18]", (0, 1, 2)), ("[23:1:23]", (3,))):
                raw = self.get(self._q(self.url(run) + f".dods?lwe_precipitation_rate.lwe_precipitation_rate"
                                       f"{times}[{r0}:1:{r1}][{c0}:1:{c1}]"), timeout=90, tries=2)
                a = self.dods(raw).reshape(len(which), r1 - r0 + 1, c1 - c0 + 1)
                a = np.where(a < self.FILL, a, np.nan)
                for j, i in enumerate(which):
                    v[i, ks], p[i, ks] = sample(a[j], fr[ks] - r0, fc[ks] - c0, h, h)
                time.sleep(0.3)
        return [self.row(run, st, [s[1] for s in self.STEPS], [s[2] for s in self.STEPS], v, p)]


# INCA-BE's precipitation unit is not documented anywhere we could find (DescribeCoverage carries GeoServer's
# placeholder uom "W.m-2.Sr-1", the catalogue record none). Two hints, contradicting each other (16.11.2026):
#  - the layer's legend (WMS GetLegendGraphic, application/json) has class limits 0.1 / 0.3 / 1 / 3 / 10 / 30
#    ("Very light" ... "Downpour/hail") - a typical mm/h scale;
#  - the analyses of 15:20 and 15:40 UTC against 159 + 155 DWD 10-minute gauges inside the domain (light rain, ~0.3
#    mm/h, 90-95 gauges wet, correlation 0.5-0.6): INCA / gauge = 0.40 if the values are mm per 10 min, 0.07 if
#    they are mm/h - an analysis that ingests gauges does not show 1/14 of their rain; the values are also quantised
#    at 0.01 (fine for 10-min amounts) and wet pixels have a median of 0.05.
# -> taken as mm per 10 min (x 6 = mm/h). Each row carries "scale" (this factor): score_competitors.py --incabe-scale
#    re-scales old rows if a rainy week's amount ratio says otherwise (x ~6 against the gauges = the values were mm/h).
INCA_BE_FACTOR = 6.0   # value -> mm/h


class IncaBe(Provider):
    name = "incabe"
    WCS = "https://opendata.meteo.be/geoserver/ows?service=WCS&version=2.0.1"
    COV = "inca__Precipitation"
    START_MIN, END_MIN = 3, 16      # served ~3 min after its time, replaced by the next one 10 min later
    RETRY_S = 40
    BUDGET_S = 180

    def describe(self):
        x = self.get(f"{self.WCS}&request=DescribeCoverage&coverageId={self.COV}", timeout=40).decode("utf-8", "replace")
        begin = _ms_iso(re.search(r"<gml:beginPosition>([^<]+)</gml:beginPosition>", x).group(1))
        lat0, lon0 = (float(v) for v in re.search(r"<gml:origin>.*?<gml:pos>([^<]+)</gml:pos>", x, re.S).group(1).split())
        vec = [[float(v) for v in m.split()] for m in re.findall(r"<gml:offsetVector[^>]*>([^<]+)</gml:offsetVector>", x)]
        dlon = next(a[1] for a in vec if a[1]) if vec else None
        dlat = next(a[0] for a in vec if a[0]) if vec else None
        return begin, lat0, lon0, dlat, dlon

    def select(self, stations):
        return [s for s in stations if 47.89 <= s[2] <= 53.27 and 0.03 <= s[3] <= 9.0]

    def take(self, run, stations):
        begin, lat0, lon0, dlat, dlon = self.describe()
        if begin < run:
            raise NotReady(f"newest run {_iso(begin)}")
        if begin > run:
            raise Gone(f"already replaced by {_iso(begin)}")
        st = self.select(stations)
        if not st:
            return []
        import cv2
        lat, lon = np.array([s[2] for s in st]), np.array([s[3] for s in st])
        fr, fc = (lat - lat0) / dlat - 0.5, (lon - lon0) / dlon - 0.5     # the origin is the outer corner
        hr = int(round(FOOT_KM / (abs(dlat) * 111.2)))
        hc = int(round(FOOT_KM / (abs(dlon) * 111.2 * np.cos(np.radians(51.0)))))
        v = np.full((len(LEADS), len(st)), np.nan, np.float32)
        p = v.copy()
        for i, lead in enumerate(LEADS):
            t = _iso(run + lead * MIN_MS, "%Y-%m-%dT%H:%M:00.000Z")
            raw = self.get(f"{self.WCS}&request=GetCoverage&coverageId={self.COV}&format=image/tiff"
                           f"&subset=time(%22{t}%22)&geotiff:compression=Deflate", timeout=60, tries=2)
            g = cv2.imdecode(np.frombuffer(raw, np.uint8), cv2.IMREAD_UNCHANGED)
            if g is None or g.ndim != 2:
                raise ValueError(f"unreadable GeoTIFF for {t} ({len(raw)} bytes)")
            g = g.astype(np.float32)
            g = np.where(g < 0, 0.0, g) * INCA_BE_FACTOR      # -1 = dry
            v[i], p[i] = sample(g, fr, fc, hr, hc)
            time.sleep(0.5)
        if self.describe()[0] != run:   # replaced while we read: the steps may mix two runs
            raise Gone("replaced during the take")
        return [self.row(run, st, LEADS, LEADS, v, p, scale=INCA_BE_FACTOR)]


class Chmi(Provider):
    name = "chmi"
    BASE = "https://opendata.chmi.cz/meteorology/weather/radar/composite/fct_pseudocappi2km/hdf5/"
    OFFSETS = (0, 30)
    START_MIN, END_MIN = 2, 40
    RETRY_S = 90
    BUDGET_S = 120
    STEPS = ((30, 30), (60, 60))

    def select(self, stations):
        return [s for s in stations if 48.05 <= s[2] <= 51.45 and 11.27 <= s[3] <= 19.62]

    def take(self, run, stations):
        import h5py
        import pyproj
        st = self.select(stations)
        if not st:
            return []
        try:
            raw = self.get(f"{self.BASE}T_PANV23_C_OKPR_{_iso(run, '%Y%m%d.%H%M')}.ft60s10.tar", timeout=60, tries=2)
        except urllib.error.HTTPError as e:
            raise NotReady(f"no file yet ({e.code})")
        tar = tarfile.open(fileobj=io.BytesIO(raw))
        files = {m.name: m for m in tar.getmembers() if m.isfile()}
        lat, lon = np.array([s[2] for s in st]), np.array([s[3] for s in st])
        v = np.full((len(self.STEPS), len(st)), np.nan, np.float32)
        p = v.copy()
        for i, (lead, _) in enumerate(self.STEPS):
            name = next((n for n in files if n.endswith(f"_ft{lead}.hdf")), None)
            if name is None:
                continue
            with h5py.File(io.BytesIO(tar.extractfile(files[name]).read()), "r") as f:
                what = f["what"].attrs
                if f"{what['date'].decode()}{what['time'].decode()}" != _iso(run, "%Y%m%d%H%M%S"):
                    raise Gone("run time inside the file differs")
                w = f["where"].attrs
                proj = pyproj.Proj(w["projdef"].decode())
                ux, uy = proj(float(w["UL_lon"]), float(w["UL_lat"]))
                x, y = proj(lon, lat)
                fc = (x - ux) / float(w["xscale"]) - 0.5      # ODIM corners are the outer corners
                fr = (uy - y) / float(w["yscale"]) - 0.5
                dw = f["dataset1/data1/what"].attrs
                d = f["dataset1/data1/data"][()].astype(np.float32)
                dbz = d * float(dw["gain"]) + float(dw["offset"])
                dbz[d == float(dw["nodata"])] = np.nan
                dbz[d == float(dw["undetect"])] = -99.0
                cell_km = float(w["xscale"]) / 1000 * np.cos(np.radians(49.8))
                h = int(round(FOOT_KM / cell_km))
                v[i], p[i] = sample(mp_rate(dbz), fr, fc, h, h)
        return [self.row(run, st, [s[0] for s in self.STEPS], [s[1] for s in self.STEPS], v, p)]


# KNMI's anonymous Open Data API key as PUBLISHED on https://developer.dataplatform.knmi.nl/open-data-api ("Anonymous
# key": shared by all unregistered users, 50 requests/min + 3000/h for everybody together; read 16.11.2026, valid till
# 1.8.2027 - KNMI replaces it periodically: copy the new one from that page, or set env KNMI_KEY). Not ours, no account.
KNMI_ANON_KEY = ("eyJvcmciOiI1ZTU1NGUxOTI3NGE5NjAwMDEyYTNlYjEiLCJpZCI6IjUzYTg1ZDBhMmQ5YzRkYzJiYWNlNzQ4NTQ2Zjk4ODExIiwiaCI6"
                 "Im11cm11cjEyOCJ9")


class Knmi(Provider):
    """KNMI's pySTEPS precipitation nowcast for the Netherlands (see the module docstring)."""
    name = "knmi"
    API = "https://api.dataplatform.knmi.nl/open-data/v1/datasets/radar_forecast/versions/2.0/files/"
    FILE = "RAD_NL25_RAC_FM_{t}.h5"
    OFFSETS = (0, 30)
    START_MIN, END_MIN = 5, 50      # published ~3-4 min after the run, kept for days
    RETRY_S = 900                   # the anonymous key's quota is shared: never more than one take per 15 min
    MIN_GAP_S = 900
    BACKOFF_S = (900, 1800, 3600, 7200)   # after 429 / 403 answers in a row
    QUICK_RETRIES = 2               # 429: retried 30 s later within the take, then BACKOFF_S
    BUDGET_S = 180
    LEADS = (0,) + LEADS            # +0 = KNMI's radar picture at the run time (not a forecast; for the amounts)

    def __init__(self, fetch, log):
        super().__init__(fetch, log)
        self.key = os.environ.get("KNMI_KEY") or KNMI_ANON_KEY
        self.last_api = 0.0
        self.hold_until, self.strikes = 0.0, 0

    def due(self, now_ms):
        if time.time() < max(self.hold_until, self.last_api + self.MIN_GAP_S):
            return None
        return super().due(now_ms)

    def api(self, url, timeout=40):
        """One Open Data API request with the key - the same neutral User-Agent as the server's fetch(), the key in
        the Authorization header (KNMI's public anonymous key), nothing else."""
        import urllib.request
        if self.deadline and time.time() > self.deadline:
            raise OverBudget(f"{self.name}: take ran over its {self.BUDGET_S} s budget")
        self.requests += 1
        self.last_api = time.time()
        req = urllib.request.Request(url, headers={"User-Agent": "Mozilla/5.0", "Authorization": self.key})
        for attempt in range(1 + self.QUICK_RETRIES):
            try:
                with urllib.request.urlopen(req, timeout=timeout) as r:
                    out = r.read()
                break
            except urllib.error.HTTPError as e:
                # 429 mostly = the shared 50 requests/min are gone for this minute (seen 16.11.2026: a 429, 25 s
                # later 200 with 2500 of the hour's 3000 left) -> two quiet retries 30 s apart, then the long back-off
                if e.code == 429 and attempt < self.QUICK_RETRIES and time.time() + 35 < (self.deadline or 1e18):
                    time.sleep(30)
                    continue
                self._refused(e)
        self.strikes = 0
        return out

    def _refused(self, e):
        """A final HTTP error of the API: 429 / 403 -> back off (15 min .. 2 h) and log it; others are raised."""
        if e.code in (403, 429):
            wait = self.BACKOFF_S[min(self.strikes, len(self.BACKOFF_S) - 1)]
            self.strikes += 1
            self.hold_until = time.time() + wait
            why = "shared quota used up" if e.code == 429 else "key expired? set KNMI_KEY"
            self.log(f"competitor knmi: HTTP {e.code} from the Open Data API ({why}), backing off {wait // 60} min")
            raise NotReady(f"HTTP {e.code}")
        raise e

    def take(self, run, stations):
        import h5py
        import pyproj
        name = self.FILE.format(t=_iso(run, "%Y%m%d%H%M"))
        try:
            meta = json.loads(self.api(self.API + name + "/url"))
        except urllib.error.HTTPError as e:
            if e.code == 404:
                raise NotReady("no file yet (404)")
            raise
        raw = self.get(meta["temporaryDownloadUrl"], timeout=60, tries=2)   # S3, signed URL: no key, no API quota
        pub = _ms_iso(meta["lastModified"]) if meta.get("lastModified") else None
        with h5py.File(io.BytesIO(raw), "r") as f:
            start = f["overview"].attrs["product_datetime_start"].decode()
            if datetime.strptime(start[:17], "%d-%b-%Y;%H:%M").replace(tzinfo=timezone.utc).timestamp() * 1000 != run:
                raise Gone(f"run time inside the file is {start}")
            g = f["geographic"].attrs
            px, py = float(g["geo_pixel_size_x"][0]), float(g["geo_pixel_size_y"][0])
            x0, y0 = float(g["geo_column_offset"][0]) * px, float(g["geo_row_offset"][0]) * py   # outer UL corner
            nx, ny = int(g["geo_number_columns"][0]), int(g["geo_number_rows"][0])
            proj = pyproj.Proj(f["geographic/map_projection"].attrs["projection_proj4_params"].decode())
            x, y = proj(np.array([s[3] for s in stations]), np.array([s[2] for s in stations]))
            fc, fr = (x - x0) / px - 0.5, (y - y0) / py - 0.5     # fractional centre indices
            inside = (fc >= 0) & (fc <= nx - 1) & (fr >= 0) & (fr <= ny - 1)
            st, fc, fr = [s for s, o in zip(stations, inside) if o], fc[inside], fr[inside]
            if not st:
                return []
            h = int(round(FOOT_KM / abs(px)))
            v = np.full((len(self.LEADS), len(st)), np.nan, np.float32)
            p = v.copy()
            for i, lead in enumerate(self.LEADS):
                grp = f[f"image{lead // 5 + 1}"]
                valid = datetime.strptime(grp.attrs["image_datetime_valid"].decode()[:17], "%d-%b-%Y;%H:%M")
                if valid.replace(tzinfo=timezone.utc).timestamp() * 1000 != run + lead * MIN_MS:
                    raise Gone(f"image{lead // 5 + 1} is valid at {valid}, expected +{lead}")
                cal = grp["calibration"].attrs
                m = re.fullmatch(r"GEO=([-+.\deE]+)\*PV\+?([-+.\deE]+)", cal["calibration_formulas"].decode().strip())
                if not m:
                    raise ValueError(f"unknown calibration {cal['calibration_formulas']}")
                d = grp["image_data"][()]
                a = (d.astype(np.float32) * float(m.group(1)) + float(m.group(2))) * 12.0   # mm / 5 min -> mm/h
                a[(d == int(cal["calibration_missing_data"][0])) | (d == int(cal["calibration_out_of_image"][0]))] = np.nan
                v[i], p[i] = sample(a, fr, fc, h, h)
        ok = np.isfinite(p[0])          # inside the radar coverage at the run time
        st = [s for s, o in zip(st, ok) if o]
        if not st:
            return []
        return [self.row(run, st, self.LEADS, self.LEADS, v[:, ok], p[:, ok], pub=pub)]


class Meander(Provider):
    name = "meander"
    BASE = "https://odp.met.hu/weather/nwp/MEANDER/nc/"
    START_MIN, END_MIN = 11, 90
    RETRY_S = 120
    BUDGET_S = 180
    # nominal lead -> (start, end) minutes of the accumulation difference = the 30 min centred on the lead
    WINDOWS = {30: (15, 45), 60: (45, 75), 90: (75, 105), 120: (105, 120)}

    def select(self, stations):
        return [s for s in stations if 45.66 <= s[2] <= 48.79 and 15.71 <= s[3] <= 23.22]

    def _file(self, run, var, minutes):
        hh = _iso(run, "%H")
        name = f"MEANDER-{var}-{_iso(run, '%Y%m%d_%H00')}+0{minutes // 60:02d}{minutes % 60:02d}.nc.zip"
        try:
            raw = self.get(f"{self.BASE}{hh}/{name}", timeout=60, tries=2)
        except urllib.error.HTTPError as e:
            raise NotReady(f"{name}: {e.code}")
        z = zipfile.ZipFile(io.BytesIO(raw))
        gatt, var_ = nc3_read(z.read(z.namelist()[0]))
        a, att = var_[var]
        a = a.reshape(int(gatt["Ny"]), int(gatt["Nx"])).astype(np.float32)
        fill = att.get("_FillValue")
        if fill is not None:
            a[np.isclose(a, float(fill))] = np.nan
        return a, gatt

    def take(self, run, stations):
        st = self.select(stations)
        if not st:
            return []
        acc, gatt = {}, None
        for m in sorted({x for w in self.WINDOWS.values() for x in w}):
            acc[m], gatt = self._file(run, "sumRadPrec", m)
            time.sleep(0.3)
        lat, lon = np.array([s[2] for s in st]), np.array([s[3] for s in st])
        fr = (float(gatt["La1"]) - lat) / float(gatt["Dy"])
        fc = (lon - float(gatt["Lo1"])) / float(gatt["Dx"])
        hr = int(round(FOOT_KM / (float(gatt["Dy"]) * 111.2)))
        hc = int(round(FOOT_KM / (float(gatt["Dx"]) * 111.2 * np.cos(np.radians(47.2)))))
        v = np.full((len(LEADS), len(st)), np.nan, np.float32)
        p = v.copy()
        for i, lead in enumerate(LEADS):
            a, b = self.WINDOWS[lead]
            rate = np.maximum(acc[b] - acc[a], 0.0) * 60.0 / (b - a)
            v[i], p[i] = sample(rate, fr, fc, hr, hc)
        rows = [self.row(run, st, LEADS, LEADS, v, p)]
        # cloud cover at the SYNOP stations (sunny truth everywhere in Europe): instantaneous at each lead, 0..1
        sy = [s for s in st if s[0] == "sy"]
        if sy:
            try:
                lat, lon = np.array([s[2] for s in sy]), np.array([s[3] for s in sy])
                fr = (float(gatt["La1"]) - lat) / float(gatt["Dy"])
                fc = (lon - float(gatt["Lo1"])) / float(gatt["Dx"])
                cv = np.full((len(LEADS), len(sy)), np.nan, np.float32)
                cp = cv.copy()
                for i, lead in enumerate(LEADS):
                    a, _ = self._file(run, "cloudines", lead)
                    cv[i], cp[i] = sample(np.clip(a / 8.0, 0, 1), fr, fc, hr, hc)
                    time.sleep(0.3)
                rows.append(self.row(run, sy, LEADS, LEADS, cv, cp, var="cloud"))
            except Exception as e:
                self.log(f"competitor meander cloud cover failed: {e}")
        return rows


class Eccc(Provider):
    """ECCC's radar extrapolation, one GetFeatureInfo per sampled gauge (see the module docstring)."""
    name = "eccc"
    na = True
    WMS = "https://geo.weather.gc.ca/geomet?service=WMS&version=1.3.0"
    LAYER = "Radar_1km_RainPrecipRate-Extrapolation"
    OFFSETS = (0, 30)
    LEAD_OF = {0: 30, 30: 60}       # the :00 run for +30, the :30 run for +60 -> both valid at :30
    START_MIN, END_MIN = 24, 150    # ~20 min behind, kept 3 h
    RETRY_S = 180
    BUDGET_S = 1200                 # ~1.1 s per point (0.25 s pause + the answer): ~7.5 min for 400 points
    SAMPLE_PER_COUNTRY = 200
    PAUSE_S = 0.25
    MAX_ERRORS = 10                 # consecutive failures that end the round
    prefer = None                   # callable -> {"network:id"} gauges with 10-minute truth (CompetitorBench, NA)

    def select(self, stations):
        """A fixed, spread-out sample: the stations ordered by a hash of their id (stable when the list grows), the
        first one per 1 x 1 degree cell - a gauge with 10-minute truth first (17.11.2026: scorable on 10-minute
        pairs) - then the first SAMPLE_PER_COUNTRY of those per country."""
        try:
            pref = set(self.prefer()) if self.prefer else set()
        except Exception:
            pref = set()
        out = []
        for country in ("us", "ca"):
            cand = sorted((s for s in stations if s[0] == country),
                          key=lambda s: (_key(s) not in pref, hashlib.md5(s[1].encode()).hexdigest()))
            cells, pick = set(), []
            for s in cand:
                cell = (int(s[2] // 1), int(s[3] // 1))
                if cell not in cells:
                    cells.add(cell)
                    pick.append(s)
            pick.sort(key=lambda s: hashlib.md5(s[1].encode()).hexdigest())
            out += pick[:self.SAMPLE_PER_COUNTRY]
        return out

    def served_runs(self):
        """(oldest, newest) run time the WMS serves (ms; 0, 0 when the capabilities have no reference_time)."""
        x = self.get(f"{self.WMS}&request=GetCapabilities&layer={self.LAYER}", timeout=60).decode("utf-8", "replace")
        m = re.search(r'<Dimension name="reference_time"[^>]*>([^<]+)</Dimension>', x)
        if not m:
            return 0, 0
        parts = m.group(1).split("/")
        return _ms_iso(parts[0]), _ms_iso(parts[1])

    def value(self, lat, lon, valid, run):
        # DIM_REFERENCE_TIME, not reference_time: GeoMet ignores the latter and answers from the newest run (module
        # docstring); checked 17.11.2026 - with DIM_ the runs differ at the same valid time, +72 min is the last step
        d = 0.005
        url = (f"{self.WMS}&request=GetFeatureInfo&layers={self.LAYER}&query_layers={self.LAYER}&crs=EPSG:4326"
               f"&bbox={lat - d:.4f},{lon - d:.4f},{lat + d:.4f},{lon + d:.4f}&width=3&height=3&i=1&j=1"
               f"&info_format=text/plain&time={_iso(valid)}&DIM_REFERENCE_TIME={_iso(run)}")
        text = self.get(url, timeout=30).decode("utf-8", "replace")
        if "Search returned no results" in text:
            return np.nan
        m = re.search(r"value_0 = '([^']*)'", text)
        if not m:
            raise ValueError(text[:200].replace("\n", " "))
        cls = re.search(r"class = '([^']*)'", text)
        if cls and cls.group(1).strip().lower() == "undetected":
            return 0.0
        return float(m.group(1))

    def take(self, run, stations):
        oldest, newest = self.served_runs()
        if newest < run:
            raise NotReady("run not served yet")
        if oldest > run:
            raise Gone("run no longer served")
        st = self.select(stations)
        if not st:
            return []
        lead = self.LEAD_OF[int(run % H_MS // MIN_MS)]
        vals = np.full(len(st), np.nan, np.float32)
        errors = done = 0
        for k, s in enumerate(st):
            try:
                vals[k] = self.value(s[2], s[3], run + lead * MIN_MS, run)
                errors, done = 0, done + 1
            except OverBudget:
                raise
            except Exception as e:
                errors += 1
                if errors >= self.MAX_ERRORS:
                    if done < len(st) // 2:
                        raise RuntimeError(f"{errors} errors in a row after {done} points: {e}")
                    break
            time.sleep(self.PAUSE_S)
        # "ref": the run was chosen with DIM_REFERENCE_TIME (rows without it are the newest run, not this one)
        return [self.row(run, st, [lead], [lead], [vals], [vals], fp=0, ref="dim")]


EUROPE = (IncaBe, Chmi, Knmi, Meander, MetNo)   # priority order: the shortest windows first
NORTH_AMERICA = (Eccc,)


class CompetitorBench:
    def __init__(self, fetch, log, log_dir, stations, na=False, only=None):
        """stations: the gauge list [(network, id, lat, lon)] or a callable returning it (read again every hour)."""
        self.fetch, self.log, self.na = fetch, log, na
        self.dir = os.path.join(log_dir, "bench")
        self._stations, self._st, self._st_at = stations, [], 0.0
        self.providers = [cls(fetch, log) for cls in (NORTH_AMERICA if na else EUROPE) if only in (None, cls.name)]
        self.truth_dir, self._minute, self._minute_at = os.path.join(log_dir, "gaugetruth"), set(), 0.0
        for p in self.providers:
            if p.na:
                p.prefer = self.minute_keys

    def minute_keys(self):
        """North America: the gauges in the 10-minute archive of the last two UTC days (minute_gauges.py), hourly."""
        if time.time() - self._minute_at > 3600:
            self._minute_at = time.time()
            keys = set()
            for back in (0, 1):
                fp = os.path.join(self.truth_dir, time.strftime("%Y%m%d", time.gmtime(time.time() - back * 86400)) + ".npz")
                try:
                    keys |= {str(x) for x in np.load(fp)["ids"]}
                except (OSError, ValueError, KeyError):
                    pass
            self._minute = keys or self._minute
        return self._minute

    def stations(self):
        if callable(self._stations):
            if not self._st or time.time() - self._st_at > 3600:
                try:
                    self._st, self._st_at = [tuple(s) for s in self._stations()], time.time()
                except Exception as e:
                    self.log(f"competitor bench: gauge list failed: {e}")
            return self._st
        return [tuple(s) for s in self._stations]

    def write(self, rows):
        if not rows:
            return
        os.makedirs(self.dir, exist_ok=True)
        for r in rows:
            with open(os.path.join(self.dir, f"{r['p']}_{_iso(r['run'], '%Y%m%d')}.jsonl"), "a") as f:
                f.write(json.dumps(r, separators=(",", ":")) + "\n")

    def prune(self):
        old = time.time() - KEEP_DAYS * 86400
        try:
            for fp in os.listdir(self.dir):
                if fp.endswith(".jsonl") and os.path.getmtime(os.path.join(self.dir, fp)) < old:
                    os.remove(os.path.join(self.dir, fp))
        except OSError:
            pass

    def take(self, p, run):
        p.deadline, p.requests = time.time() + p.BUDGET_S, 0
        try:
            return p.take(run, self.stations())
        finally:
            p.deadline = None

    def loop(self):
        time.sleep(90)
        last_prune = 0.0
        while True:
            for p in self.providers:
                try:
                    run = p.due(time.time() * 1000)
                    if run is None:
                        continue
                    t0 = time.time()
                    try:
                        rows = self.take(p, run)
                    except NotReady:
                        p.next_try = time.time() + p.RETRY_S
                        continue
                    except Gone as e:
                        p.done.add(run)
                        self.log(f"competitor {p.name} {_iso(run, '%H:%M')}: given up ({e})")
                        continue
                    p.done.add(run)
                    self.write(rows)
                    rr = next((r for r in rows if r.get("var") is None), None)
                    if rr:
                        v = np.array([x for lead in rr["v"] for x in lead if x is not None], np.float32)
                        self.log(f"[{datetime.now():%H:%M:%S}] competitor {p.name} {_iso(run, '%H:%M')}: "
                                 f"{len(rr['ids'])} gauges, {int((v >= 0.1).sum())} wet values, {p.requests} requests, "
                                 f"{time.time() - t0:.0f} s")
                except Exception as e:
                    p.next_try = time.time() + p.RETRY_S
                    self.log(f"competitor {p.name} failed: {e}")
                finally:
                    p.done = {r for r in p.done if r > time.time() * 1000 - 6 * H_MS}
            if time.time() - last_prune > 86400:
                last_prune = time.time()
                self.prune()
            time.sleep(20)


# ---------- test without the server ----------
FAKE_EUROPE = [
    ("sy", "no-01492", 59.9423, 10.7200), ("sy", "no-01317", 60.3830, 5.3330), ("sy", "no-01384", 60.2000, 11.0800),
    ("sy", "se-02485", 59.3500, 18.0500), ("se", "71420", 57.7200, 11.9900), ("se", "53430", 55.5700, 13.0600),
    ("fi", "100971", 60.1750, 24.9440), ("fi", "101104", 61.4600, 23.7600), ("sy", "dk-06180", 55.6140, 12.6460),
    ("sy", "ee-26038", 59.3980, 24.6030), ("sy", "be-06447", 50.7970, 4.3580), ("sy", "be-06407", 51.2000, 2.8700),
    ("sy", "nl-06260", 52.1000, 5.1800), ("sy", "lu-06590", 49.6200, 6.2000), ("de", "01078", 51.2960, 6.7690),
    ("de", "05906", 49.5063, 8.5584), ("cz", "0-203-0-11518", 50.1000, 14.2600), ("cz", "0-203-0-11723", 49.1500, 16.6900),
    ("sy", "cz-11782", 49.6900, 18.1100), ("hu", "12843", 47.4300, 19.1800), ("hu", "12982", 46.2500, 20.0900),
    ("sy", "hu-12882", 47.4900, 21.6100), ("sy", "hu-12843", 47.4300, 19.1800)]
FAKE_NA = [
    ("ca", "CA-YYZ", 43.68, -79.63), ("ca", "CA-YUL", 45.47, -73.74), ("ca", "CA-YVR", 49.19, -123.18),
    ("ca", "CA-YHZ", 44.88, -63.51), ("ca", "CA-YYC", 51.11, -114.02), ("us", "KSEA", 47.44, -122.31),
    ("us", "KSFO", 37.62, -122.37), ("us", "KBOI", 43.56, -116.22), ("us", "KMSY", 29.99, -90.25),
    ("us", "KJFK", 40.64, -73.78), ("us", "KSJC", 37.36, -121.93)]


def _test_fetch(url, timeout=40, tries=3):
    """The server's fetch without the server: the same neutral User-Agent ("Mozilla/5.0"), nothing else."""
    import urllib.request
    req = urllib.request.Request(url, headers={"User-Agent": "Mozilla/5.0"})
    for attempt in range(tries):
        try:
            with urllib.request.urlopen(req, timeout=timeout) as r:
                return r.read()
        except urllib.error.HTTPError:
            raise
        except OSError:
            if attempt == tries - 1:
                raise
            time.sleep(5 * (attempt + 1))


def main():
    args = sys.argv[1:]
    opt = lambda n, d=None: args[args.index(n) + 1] if n in args else d
    name = opt("--once")
    classes = {c.name: c for c in EUROPE + NORTH_AMERICA}
    if name not in classes:
        sys.exit(f"usage: python bench_competitors.py --once {'|'.join(classes)} [--stations f.json] [--dir out] "
                 "[--run YYYYMMDDHHMM]")
    cls = classes[name]
    if opt("--stations"):
        with open(opt("--stations")) as f:
            st = [tuple(s) for s in json.load(f)]
    else:
        st = FAKE_NA if cls.na else FAKE_EUROPE
    bench = CompetitorBench(_test_fetch, print, opt("--dir", "."), st, na=cls.na, only=name)
    p = bench.providers[0]
    now = time.time() * 1000
    # the newest run whose window has started (any age the provider still serves), newest first
    cands = sorted({h * H_MS + m * MIN_MS for h in range(int(now // H_MS) - 2, int(now // H_MS) + 1) for m in cls.OFFSETS
                    if h * H_MS + m * MIN_MS + cls.START_MIN * MIN_MS <= now}, reverse=True)
    if opt("--run"):
        cands = [calendar.timegm(time.strptime(opt("--run"), "%Y%m%d%H%M")) * 1000]
    elif name == "incabe":   # only its current run exists (every 10 min): test with that one
        cands = [p.describe()[0]]
    t0 = time.time()
    for run in cands:
        try:
            rows = bench.take(p, run)
            break
        except (NotReady, Gone) as e:
            print(f"run {_iso(run)}: {type(e).__name__}: {e}")
    else:
        sys.exit("no run available")
    print(f"{name}: run {_iso(run)}, {p.requests} requests, {time.time() - t0:.1f} s, "
          f"{len(st)} gauges given, {len(rows[0]['ids']) if rows else 0} inside")
    for r in rows:
        print(f"  var {r.get('var', 'rr')}: latency (read - run) {(r['got'] - r['run']) / 60000:.0f} min, box +-{r['fp']} km")
        for i, lead in enumerate(r["leads"]):
            for kind in ("v", "pt"):
                a = np.array([x if x is not None else np.nan for x in r[kind][i]], np.float32)
                ok = np.isfinite(a)
                rng = f"{np.nanmin(a):.2f}..{np.nanmax(a):.2f}, mean {np.nanmean(a):.3f}" if ok.any() else "-"
                print(f"    +{lead:3d} (step {r['dt'][i]:3d}) {kind:2s}: {ok.sum():3d} values, {(a[ok] > 0).sum():3d} > 0, "
                      f"{(a[ok] >= 0.1).sum():3d} >= 0.1, range {rng}")
        wet = [(r["ids"][k], [r["v"][i][k] for i in range(len(r["leads"]))]) for k in range(len(r["ids"]))
               if any((r["v"][i][k] or 0) > 0 for i in range(len(r["leads"])))]
        for k, vals in wet[:8]:
            print(f"      {k}: {vals}")
    if "--dir" in args:
        bench.write(rows)
        print(f"written to {bench.dir}")


if __name__ == "__main__":
    main()
