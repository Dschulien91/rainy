"""Sunshine now and for the next 2 h over Germany ("sunny" in the Germany edition), from DWD satellite data.

Observation: DWD's satellite direct radiation, opendata.dwd.de/weather/satellite/radiation/sid/SIDhr<stamp>EUv4.nc.bz2
(SPECMAGIC_now from Meteosat/MTG, every 10 min, ~17-20 min behind, netCDF3, regular lat/lon grid 0.015 deg = ~1 x 1.7 km,
35.5-59.8 N / 13 W-26 E, int16 W/m2, fill -1, bz2 ~3 MB by day). It holds SID (direct irradiance on the horizontal) and
DNI (direct normal irradiance); the WMO defines sunshine as DNI > 120 W/m2, so DNI is used (SID = DNI * sin(elevation):
the same clear-sky index either way).
Low sun and night: below DAY_ELEV the radiation retrieval says nothing about clouds, so DWD's infrared cloud top height
(opendata.dwd.de/weather/satellite/clouds/CTH/CTHin<stamp>nEU.nc.bz2, MTG FCI IR10.5, every 10 min, 0.03 deg, netCDF3,
metres, works day and night) stands in: low "tops" are the (cold) ground or low fog, high tops are cloud. It keeps the
map's clouds at night and lets the forecast know the clouds that drift into the morning sun. Its "opa" (opacity) field
is 100 % almost everywhere in November and told sunny from cloudy pixels worse (80 % vs 90 % agreement with DNI by day).
Not used: SIS/SISin (global radiation: diffuse light, not sunshine), CAL (cloud albedo: SID already contains it),
SIDin...DEv3 (the same data on a coarse 0.05 deg grid), TS (thunderstorm severity, 800 MB HDF5) and SISfc - an hourly
global radiation forecast for Germany (0.05 deg, +18 h, issued hourly; see server/replay_sun.py for its score): hourly
global radiation cannot tell 10-minute sunshine.

Method (separates the weather from the sun's geometry):
  1. clear-sky index k = DNI / DNI_clear(elevation) (simple clear-sky beam model, air mass after Kasten & Young)
  2. motion: dense optical flow (OpenCV Farneback, as server/clouds.py) between the k fields ~30 min apart, smoothed
  3. extrapolation of k along the motion in 10 min steps to +150 min, blurred with lead time (small clouds are not
     predictable for long)
  4. sunshine at each forecast time from the advected k and that time's sun: DNI = k * DNI_clear(elevation(t)),
     soft threshold around 120 W/m2 -> "sun share" 0..1 (fraction of the 10 min with sunshine); none below SUN_ELEV
Shadows (scored, not served): "pers" (the analysis stays), "kpers" (k stays, only the sun moves), "model" (fading into
the weather model's sunshine in the second hour), "nwp" (weather model only: Open-Meteo minutely_15 sunshine_duration,
self-hosted, best match = ICON-D2 in Germany), "lowf03" / "lowf10" (the other low-sun clear beam factor, see
LOW_F), "metfog" (Europe: IR k capped around airports reporting fog, see METAR_URL). The served forecast changes only
when a shadow wins (server/score_sun.py).
Replay 07.+08.11. (server/replay_sun.py, 94 runs, daytime): station MAE +60/+120 sat 0.089/0.124, persistence
0.106/0.145, kpers 0.102/0.139, model blend 0.089/0.175, model alone 0.230/0.241 (far too sunny, bias +0.2); grid CSI
+60/+120 sat 0.87/0.77, persistence 0.83/0.71. So the model is not mixed in (to be re-checked with the live logs).

Verification (like server/fcverify.py for rain): every run's leads LEADS are compared with the later SID analysis on
a ~3 km grid inside the shown area (sunny = share >= 0.5, ~9 km tolerance) -> verify_log/sungrid_<YYYYMMDD>.jsonl, and
with DWD's 10-minute station sunshine duration (SD_10, ~70 stations, CDC open data "now" files) ->
verify_log/sunstation_<YYYYMMDD>.jsonl. Daytime only (sun >= SCORE_ELEV at the valid time).

Served by server/rainy_server.py in the Germany edition through the same API as the full edition's clouds.py, so the
app's sunny mode works unchanged:
  GET /api/cloud/frames          -> {"frames": [{"ts", "fc"}], "ref": ms}
  GET /api/cloud/point?lat=&lon= -> {"series": [{"ts", "fc", "cloud": 0..1, "sun": 0..1}], "ref": ms}  ([] outside)
  GET /tiles/cloud/<key>/<ts>/<z>/<x>/<y>.png -> grey = cloud value, alpha = inside the shown area
"cloud" is encoded so that the app's sunShare(cloud) = 1 - (cloud - 0.05) / 0.8 gives the sun share back by day; "sun"
is the sun share itself (0 below SUN_ELEV). /api/health reports "sunny": frames, ref, age, stations.
Env: SUNNY_DE=0 disables it; VERIFY_LOG_DIR as for fcverify.
Memory (measured, 4 h replay): 3 h of observed shares + map values, the k fields of the last 40 min, one forecast
(float16 / uint8 on a ~1.9 km grid, 631 x 593) and the pending verification grids: ~50 MB; ~150 MB transient while a
file is read (bz2 -> 17 MB netCDF). CPU: ~1.3 s per 10-min slot on the x86 dev machine (unpack, regrid, IR, forecast,
shadows); downloads ~3 MB (SID) + ~0.9 MB (CTH, only when the sun is low somewhere) per 10 min, stations ~70 small files
per ~30 min. No lock is shared with the rain processing.
"""
import bz2
import csv
import gzip
import io
import json
import math
import os
import re
import threading
import time
import urllib.parse
import urllib.request
import zipfile
from collections import OrderedDict
from concurrent.futures import ThreadPoolExecutor, TimeoutError, as_completed
from datetime import datetime, timezone

import cv2
import numpy as np
from PIL import Image
from scipy.io import netcdf_file

import gauges
import monitor
import tileimg
from clouds import sun_elevation
from fcverify import Pending
from mosaic import GermanyDwd, WORLD, merc

SID_URL = "https://opendata.dwd.de/weather/satellite/radiation/sid/"
CTH_URL = "https://opendata.dwd.de/weather/satellite/clouds/CTH/"
SOLAR = "https://opendata.dwd.de/climate_environment/CDC/observations_germany/climate/10_minutes/solar/now/"
LOG_DIR = os.environ.get("VERIFY_LOG_DIR", os.path.join(os.path.dirname(os.path.abspath(__file__)), "verify_log"))

STEP_MS = 600_000
CELL = 2 * GermanyDwd.CELL     # 3000 m Web Mercator = ~1.9 km on the ground (the satellite has ~1 x 1.7 km)
OBS_KEEP_MS = 3 * 3600_000     # observed frames kept (the app shows the last hour; verification needs 2 h)
K_KEEP_MS = 40 * 60_000        # k fields kept for the motion
FC_LEADS = range(10, 151, 10)  # minutes after the newest image (it is ~20 min old): covers "now + 2 h"
POLL_S = 120                   # new files every 10 min
CTH_WAIT_AREA = 0.5            # night / twilight: a slot waits for its own CTH file (SunnyDE._wait_cth) ...
CTH_WAIT_S = 15 * 60           # ... at most this long after its SIDhr file appeared
# DWD's SIDhr (DNI) files stopped at 07:00Z on 24.11.2026 while the CTH (IR cloud top) files kept coming: sunny Germany
# AND Europe froze on the 07:00Z picture for hours (the app showed a "forecast" that ended in the past). When the
# newest CTH slot is IR_FALLBACK_MS newer than the newest SIDhr slot, the newer slots are made without the DNI: the
# weather model's clear-sky index stands in for it by day and the CTH cloud tops at low sun and night (blended between
# SUN_ELEV and DAY_ELEV as usual). The cloud tops alone were useless by day: 24.11.2026 09:50-10:10Z at 261 DE/AT/CH
# sunshine stations (87 % sunny under thin high cloud) IR-only MAE 0.55-0.62 (bias -0.6), the model 0.13. Such slots
# are never a grid truth (obs_share) and a SIDhr file that arrives later replaces them. SUNNY_IR_FALLBACK=0: off.
IR_FALLBACK = os.environ.get("SUNNY_IR_FALLBACK", "1") != "0"
IR_FALLBACK_MS = 30 * 60_000
# sun geometry
SUN_ELEV = 3.0                 # below: no sunshine (the beam is too weak and the retrieval unreliable)
DAY_ELEV = 8.0                 # above: k from DNI alone; between SUN_ELEV and DAY_ELEV it fades into the IR cloud top height
DNI_SUN = 120.0                # WMO sunshine threshold, W/m2
# The map shows clouds vs clear sky, not the sunshine that is left of a low sun: below DISP_ELEV its value is the share
# the sky's k would give at DISP_ELEV. Before (share at the real sun, and k at 15 deg only below DAY_ELEV) every clear
# area turned grey in the late afternoon (no beam > 120 W/m2 below ~8 deg) and yellow again at ~5 deg - Hannover 09.11.
# showed "sunny, overcast, sunny" while the sky stayed clear (WetterOnline's cloud map, DWD station Hannover 0/8).
# The sunshine itself (point "sun", the hours, the verification) still uses the real sun.
DISP_ELEV = 15.0
CS_A = 0.72                    # clear-sky beam transmittance per air mass (fitted to the clearest DWD pixels, 08.11.)
K_MAX = 1.3
# soft threshold: share = clip((DNI - (DNI_SUN - RAMP)) / (2 RAMP)); against SD_10 (replay_sun.py --calib, 08.11., 2467
# station values): RAMP 40 + 3 x 3 mean MAE 0.072, hit 0.929 (RAMP 20-80 within 0.001, 200: MAE 0.22)
RAMP = 40.0
# IR cloud top height (m) -> k: clear below CTH0, cloud above CTH1. Fitted against daytime DNI on 08.11. (scratch replay,
# 320k pixels, server/replay_sun.py --calib): 90 % agreement on sunny / not sunny; recheck in summer (warm ground, other heights for clear sky)
CTH0, CTH1 = 3000.0, 4000.0
# nowcast
FLOW_DOWN = 3                  # optical flow on a ~6 km grid: large-scale motion only
FLOW_SIGMA = 10.0              # flow smoothing (down pixels, ~55 km)
BLUR = 1.2                     # blur sigma (px) at 10 min lead, grows with sqrt(lead)
# ... but only from SHARP_MIN on in full: before 10.11. the first step (+10) was blurred at once and small clouds
# vanished exactly at "now" (a visible break). Now the blur ramps up over the first hour; the old way is the scored
# shadow "satold".
SHARP_MIN = 60
# model shadow (Open-Meteo)
MODEL_DEG, MODEL_BATCH, MODEL_REFRESH_S = 0.1, 2000, 30 * 60
MODEL_START, MODEL_END, MODEL_END_W = 60, 120, 0.5   # satellite weight 1 until +60, 0.5 at +120 (shadow "model")
# verification
LEADS = (10, 30, 60, 90, 120)
VERIFY_DOWN = 1.5              # verification grid = sunny grid / 1.5 = ~2.8 km (the rain verification's ~3 km)
TOL_PX = 3                     # ~9 km position tolerance
SCORE_ELEV = 5.0               # daytime only
MIN_PX = 300                   # fewer daytime pixels: no grid line
SUNNY = 0.5                    # "sunny" = share >= 0.5
STATION_WAIT_MS = 4 * 3600_000 # station values are 20-110 min behind; give up on a forecast after this
STATION_LAG_MS = 10 * 60_000   # SD_10 stamped T covers (T-10, T]; the satellite slot S scans Germany ~S+5..S+9
# Bad slots: now and then one SIDhr file shows sun almost everywhere (08.11. 14:50Z: DNI > 120 W/m2 on 91 % of Germany,
# 15 % ten minutes before and 29 % after; the stations saw no such thing). A slot whose daytime sun share differs from
# the previous image by more than JUMP_MAX on average is skipped; after JUMP_ACCEPT skipped slots in a row it counts.
JUMP_MAX = 0.35
JUMP_ACCEPT = 2
TRUTH_KEEP_MS = 8 * 3600_000
# Low sun: CS_A is fitted to the clearest pixels of an image, but at a low sun the beam through the usual haze is far
# weaker. Where the IR cloud tops said "clear" (07.-09.11., 4 M pixels at sunrise/sunset) the beam was sunny in only
# 26-40 % at 3-9 deg, 55 % at 9-12, 87 % at 12-16, 97 % above 16; k (old model) median 0.2-0.35 below 12 deg, 0.9 above
# 16. So sunrise runs (k from IR = 1) forecast sun for ~5x the area that got it (09.11. morning). The clear beam is
# scaled down by LOW_F at <= LOW_E0, fading out at LOW_E1. The analysis is unaffected (share = f(k * clear_dni) = f(DNI)).
# Germany: switched off (LOW_F_DE = 1, server-v1.79, 20.11.2026). k = DNI / clear_dni was ~3x too high at 3-10 deg,
# and the map (k shown at DISP_ELEV) cleared up every afternoon ~30 min before IR took over and closed again (Germany
# 20.11. clear share 13:30 .28, 14:00 .36, 14:30 .40, 15:00 .19; with 1: .25/.30/.29/.14); the forecast lost the sun of
# a setting sun (live 14.-20.11., valid 13-16 UTC, +60: sunny stations forecast 659 vs observed 988, bias -0.086). The
# satellite's own clear beam at a low sun is not weaker than CS_A says: the 99th percentile of DNI per 2 deg (19./20.11.,
# all of DWD's area) is 1.1-1.3x clear_dni(1) at 2-12 deg in the morning and 0.8-1.0x in the afternoon. Replay 19.+20.11.
# (replay_sun.py, 158 slots, ICON-D2), DWD stations MAE / bias / hit (sunny fc:obs) at +60 and +120, 0.3 -> 1:
#   07-09 UTC  +60 .077/-.033/.927 -> .055/-.016/.950   +120 .065/-.057/.942 -> .052/-.021/.956
#   09-13 UTC  +60 .098/+.028/.903 -> .092/+.015/.909   +120 .131/+.035/.873 -> .109/-.004/.896
#   13-16 UTC  +60 .115/-.083/.893 (x0.28) -> .130/-.028/.873 (x0.83)   +120 .122/-.083/.887 -> .126/-.031/.886
#   all day    +60 .098 -> .094, +120 .116 -> .102; grid CSI +60/+120 .715/.554 -> .755/.604 (13-16 UTC .586/.435 ->
#   .710/.614, sunny area x0.63/0.50 -> 0.89/0.88; 07-09 UTC .332/.152 -> .578/.461)
# (the afternoon's slightly lower station MAE with 0.3 is the hedging of a forecast that drops the sun).
# Europe and the Americas keep 0.3: in the Alps the low sun is behind the mountains, and 0.3 stands in for that.
# Europe replay 20.11. (ICON-EU, DWD + 265 GeoSphere TAWES stations), MAE at +60 / +120, 0.3 -> 1:
#   Austria  07-09 UTC .185/.136 -> .178/.187   13-17 UTC .155/.149 -> .226/.248 (bias -.047 -> +.069)
#   Germany  07-09 UTC .138/.115 -> .095/.088   13-17 UTC .146/.160 -> .140/.158
#   grid CSI +60 07-09 .761 -> .872, 13-16 .833 -> .902, but sunny area x1.10-1.14 in the afternoon
# A clear IR sky scaled down instead (IR_F 0.3 with LOW_F 1) helped the Austrian mornings (+60 .145) but lost in
# Germany (+60 07-09 .066 vs .055) and on the grid. Each instance scores the other value as the shadow
# "lowf03" / "lowf10" (low_f_shadow).
LOW_E0, LOW_E1 = 10.0, 18.0
LOW_F = float(os.environ.get("SUNNY_LOW_F", "0.3"))  # Europe, the Americas, tools (env only for replays)
LOW_F_DE = float(os.environ.get("SUNNY_LOW_F", "1.0"))  # the Germany instance (SunnyDE without an area)
# IR "clear" is a hazy clear: the IR k (k_from_cth, after the fog caps) times IR_F, i.e. a clear IR sky gives the beam
# clear_dni * IR_F (env only for replays, server/replay_sun.py --ir-f; see LOW_F)
IR_F = float(os.environ.get("SUNNY_IR_F", "1.0"))
# Fog and low stratus: their tops are about as warm as the ground, so the IR cloud top height calls them "clear" (CTH
# near 0) and sunny showed full sunshine under fog at sunrise (Stuttgart 09.11.: fog, visibility ~1 km, gauges 0 sun;
# ICON-D2 had 90-100 % low cloud). Where k comes from IR (low sun, night) it is capped by the weather model's low cloud
# cover: k <= 1 - LOWCC_W * low cover. Env only for replays (0 = without).
LOWCC_W = float(os.environ.get("SUNNY_LOWCC_W", "1.0"))
# ... but ICON-EU (the Europe instance's model) misses shallow valley fog: Po valley 20.11.2026 night, METAR LIML / LIMC
# 0500 FG, LIPE 0200 FG VV001, LIPX 0400 FG VV001, ICON-EU low cloud 0-51 % (ICON-D2 100 % at Bologna only), the map
# clear (cloud 0.05). Airport reports - aviationweather.gov's global METAR file, the one the North America instance
# reads - of fog (FG / FZFG, visibility < 1 km) or a ceiling / vertical visibility at or below 300 ft cap the IR k
# around the station: 0 within METAR_FOG_R0 km, no cap beyond METAR_FOG_R1 km (server-v1.79, Europe instance only).
# Scored as the shadow "metfog" (the forecast from the capped k; equal to "sat" when no fog is reported); the served
# map and forecast change only with SUNNY_METAR_FOG=1 (after the shadow has won on foggy mornings).
METAR_URL = "https://aviationweather.gov/data/cache/metars.cache.csv.gz"
METAR_BOX = (34.0, 72.0, -25.0, 45.0)    # lat min, lat max, lon min, lon max
METAR_FOG_R0, METAR_FOG_R1 = 10.0, 20.0  # km
METAR_FOG_AGE_MS = 75 * 60_000           # a report counts for the slots up to this far from its time
METAR_EVERY_S = 600
METAR_FOG_SERVE = os.environ.get("SUNNY_METAR_FOG", "0") == "1"
# Weather-model blend beyond +30 min (scored shadow "satnwp", 17.11.2026): the advected satellite loses skill with the
# lead, the weather model does not, but the raw model is far too sunny (live 7 days: station bias +0.16, grid sunny area
# x1.7) - so the "model" shadows mixed in that bias. Here the model's clear-sky index (Open-Meteo minutely_15
# direct_normal_irradiance / clear_dni) is first quantile-mapped onto the satellite's k at ref, over the whole area
# (the same mapping for every lead), and then mixed into the served share: (1 - w) * sat + w * model, w = 0 until
# NWP_START min, linear to NWP_W120 at +120. NWP_LAG_MS: the satellite slot S shows the sky at ~S+7 min.
# Offline (server/tune_sun.py, 08./15./16.11., 162 runs, ICON-D2 15-min runs as old as they were then, model sampled
# like the live 0.1 deg points), station MAE / hit and grid MAE at +60/+90/+120:
#   sat (served)   .128/.155/.168  hit .874/.851/.841   grid .118/.143/.163  sunny area x1.02/1.04/1.06
#   model raw      .233/.236/.240  hit .763/.760/.754   grid .217/.221/.226  x1.67-1.72 (the live "nwp")
#   model qm alone .140/.152/.161  hit .863/.850/.842   grid .149/.157/.166
#   blend 30/0.55  .124/.143/.152  hit .878/.856/.859   grid .120/.142/.158  x1.02/1.04/1.01
# Weights: w 0.1-0.8 at each lead and five schedules; stations want more model (best w ~0.5-0.6 at +120), the grid less
# (~0.3); 30/0.55 is the best of both. Blending in k instead of the share was as good on the grid but x1.16 too sunny;
# the model's Lagrangian trend added to the advected k (+0.5 / +1.0) was worse (+120 station MAE .176 / .201). Other
# motion fields (flow smoothing 5 / 20, finer flow grid, DIS, mean of 3 pairs, 60 min pair, semi-Lagrangian
# trajectories) all scored within +-0.003 of the served one - the motion is not what limits +60..+120.
# SUNNY_NWP=1 serves it (the map's k is blended alike); the plain advection then stays scored as "satadv".
NWP_START = float(os.environ.get("SUNNY_NWP_START", "30"))
NWP_W120 = float(os.environ.get("SUNNY_NWP_W120", "0.55"))
NWP_LAG_MS = 7 * 60_000
SUNNY_NWP = os.environ.get("SUNNY_NWP", "0") == "1"
# The model's skill today: the corrected model's sun share at ref against the analysis (daylight pixels, mean abs.
# error, median 0.14 in the test). The weight is scaled by clip(1 - (error - NWP_SKILL_A) / NWP_SKILL_B, 0, 1): full
# below 0.10, none above 0.20. Offline that turned the blend's one bad day (08.11.: +120 station MAE .121 -> .133) into
# no harm (.121) and was better on all leads (+60/+90/+120 station .124/.143/.148 hit .877/.855/.874, grid
# .117/.138/.152, sunny area x1.02/1.03/0.96). With ICON-EU (the Europe instance's model; archived runs 3-6 h old)
# the ungated blend was worse on 08.11. (+120 .128 -> .197), gated it helped overall (+120 station .177 -> .161, hit
# .832 -> .867, grid .168 -> .160). ICON-D2-RUC (hourly, not on the local Open-Meteo) was a little better than ICON-D2.
NWP_SKILL_A, NWP_SKILL_B = 0.10, 0.10
# shadow "satkp" (22.11.2026, see SunnyDE.forecast): share of the stationary k in the satellite part, 0 until
# KP_START min, linear to KP_W120 at +120
KP_START = float(os.environ.get("SUNNY_KP_START", "30"))
KP_W120 = float(os.environ.get("SUNNY_KP_W120", "0.5"))


def nwp_weight(lead):
    """Weather-model weight in the "satnwp" blend at a lead (minutes)."""
    return 0.0 if lead <= NWP_START else NWP_W120 * min(1.0, (lead - NWP_START) / max(1e-6, 120.0 - NWP_START))


def quantile_map(src, ref, n=50_000):
    """A function x -> ref's value at x's mid-rank in src (src, ref: 1-d samples of the same pixels), or None when
    there are too few. Mid-ranks keep ties (a model's many k = 0 pixels) at the middle of their range."""
    if len(src) < 500:
        return None
    sel = np.linspace(0, len(src) - 1, min(n, len(src))).astype(np.int64)
    s, r = np.sort(np.asarray(src, np.float32)[sel]), np.sort(np.asarray(ref, np.float32)[sel])
    m = len(s)

    def f(x):
        x = np.asarray(x, np.float32)
        p = np.searchsorted(s, x, side="left") + np.searchsorted(s, x, side="right")
        return np.where(np.isfinite(x), r[np.clip(p // 2, 0, m - 1)], np.nan).astype(np.float32)
    return f


def clear_dni(elev, low_f=None):
    """Clear-sky direct normal irradiance (W/m2) for a solar elevation in degrees (array): Meinel-type
    transmittance CS_A ** (AM ** 0.678) with the Kasten & Young air mass, lowered at a low sun by low_f (default LOW_F).
    Only its shape over elevation matters for the forecast (the index k is divided by it and multiplied back with it at
    the forecast time)."""
    low_f = LOW_F if low_f is None else low_f
    e = np.maximum(np.asarray(elev, np.float32), 0.5)
    am = 1.0 / (np.sin(np.radians(e)) + 0.50572 * (e + 6.07995) ** -1.6364)
    low = low_f + (1.0 - low_f) * np.clip((e - LOW_E0) / (LOW_E1 - LOW_E0), 0.0, 1.0)
    return (1361.0 * CS_A ** (am ** 0.678) * low).astype(np.float32)


def share_from_k(k, elev, low_f=None):
    """Sun share 0..1 from the clear-sky index and the solar elevation (NaN where k is unknown)."""
    s = np.clip((k * clear_dni(elev, low_f) - (DNI_SUN - RAMP)) / (2 * RAMP), 0.0, 1.0)
    return np.where(elev < SUN_ELEV, np.where(np.isfinite(k), 0.0, np.nan), s).astype(np.float32)


def metar_fog(raw, box=METAR_BOX):
    """{station: (lat, lon, observation time ms, fog)} of the METARs inside box in aviationweather.gov's cache file
    (gzip csv). fog: visibility below 1 km (FG / FZFG when the visibility is missing; not MIFG / BCFG / PRFG / VCFG:
    shallow, patches, nearby; "AUTO 9999 FG" neither), or a BKN / OVC layer or a vertical visibility at or below 300 ft."""
    rows = list(csv.reader(gzip.decompress(raw).decode("utf-8", "replace").splitlines()))
    if not rows or rows[0][0] != "raw_text":
        return {}
    col = {name: i for i, name in enumerate(rows[0]) if name not in ("sky_cover", "cloud_base_ft_agl")}
    out = {}
    for r in rows[1:]:
        try:
            lat, lon = float(r[col["latitude"]]), float(r[col["longitude"]])
            if not (box[0] <= lat <= box[1] and box[2] <= lon <= box[3]):
                continue
            t = int(datetime.strptime(r[col["observation_time"]][:19], "%Y-%m-%dT%H:%M:%S")
                    .replace(tzinfo=timezone.utc).timestamp() * 1000)
            vis = r[col["visibility_statute_mi"]].rstrip("+")
            vis = float(vis) if vis else None
        except (ValueError, IndexError, KeyError):
            continue
        text = " " + re.split(r" (?:RMK|TEMPO|BECMG|NOSIG)\b", r[0])[0] + " "  # the observation, not its trend
        fog = ((vis is not None and vis < 0.62)
               or (vis is None and re.search(r" [+-]?(FZ)?FG ", text) is not None)
               or any(int(h) <= 3 for h in re.findall(r" (?:BKN|OVC|VV)(\d{3})", text)))
        sid = r[col["station_id"]]
        if sid not in out or t > out[sid][2]:
            out[sid] = (lat, lon, t, fog)
    return out


def k_from_cth(cth):
    """Clear-sky index estimated from the IR cloud top height (m)."""
    return np.clip((CTH1 - cth) / (CTH1 - CTH0), 0.0, 1.0).astype(np.float32)


def read_nc(data, names):
    """(lat, lon, {name: float32 grid with NaN for fill values}) of a (bz2) netCDF3 file with a regular lat/lon grid."""
    if data[:3] == b"BZh":
        data = bz2.decompress(data)
    f = netcdf_file(io.BytesIO(data), "r", mmap=False)
    lat, lon = np.array(f.variables["lat"][:], np.float64), np.array(f.variables["lon"][:], np.float64)
    out = {}
    for n in names:
        v = f.variables[n]
        a = np.array(v[0] if len(v.shape) == 3 else v[:], np.float32)
        fill = getattr(v, "_FillValue", None)
        if fill is not None:
            a[a == float(fill)] = np.nan
        out[n] = a
    f.close()
    return lat, lon, out


def stamp_ts(stamp):
    return int(datetime.strptime(stamp, "%Y%m%d%H%M").replace(tzinfo=timezone.utc).timestamp() * 1000)


def ts_stamp(ts):
    return datetime.fromtimestamp(ts / 1000, timezone.utc).strftime("%Y%m%d%H%M")


KEEP_RUNS = 3  # runs (10 min apart) whose map frames stay available for tile keys (the app reloads every 5 min)

# sunny for all of Europe (the Europe test instance, server EDITION=eu): DWD's satellite files already cover
# 35.5-59.8 N, 13 W-26 E; the grid is 4 km Web Mercator (~2.6 km on the ground at 50 N: 4x the Germany area's pixels
# for ~11x its area), the weather model (fog cap, shadows) on a 0.25 deg grid from ICON-EU only (on the server's disk).
# Verified against the later analysis on the whole area and at the DWD stations (Germany) - files eusungrid_* and
# eusunstation_* next to the Germany edition's.
# Europe up to 66.8 N and from 25 W (15.11.2026, user: Norway, Sweden, Iceland were missing): DWD's SID / CTH end at
# 59.8 N / 13 W (CTH: 63 N / 20 W); beyond them EUMETSAT's cloud mask fills in (clm_cloud). Further north Meteosat's
# mask has no data (67 N: 40 %, 68 N: 0) - that would need polar-orbiting satellites.
_ex, _ey = merc(np.array([-25.0, 32.0]), np.array([35.6, 66.8]))
EUROPE = {"bbox": (float(_ex[0]), float(_ey[0]), float(_ex[1]), float(_ey[1])), "cell": 4000.0,
          "model": {"lats": (36.0, 59.5), "lons": (-12.0, 25.5), "deg": 0.25, "models": "icon_eu"}, "prefix": "eu",
          "clm": True}
CLM_WMS = "https://view.eumetsat.int/geoserver/wms"   # EUMETView, public (no account), PNG maps of MSG products
CLM_BOX, CLM_DEG = (-25.0, 35.5, 32.0, 67.5), 0.04     # lon0, lat0, lon1, lat1


def clm_cloud(fetch, ts):
    """EUMETSAT's MSG cloud mask (every 15 min, ~30 min late) as the cloud fraction on a regular lat/lon grid:
    (lat ascending, lon, float32 1 = cloud / 0 = clear / NaN = no data), or None. The newest image at or before ts."""
    lon0, lat0, lon1, lat1 = CLM_BOX
    w, h = int(round((lon1 - lon0) / CLM_DEG)), int(round((lat1 - lat0) / CLM_DEG))
    for back in (15, 30):
        t = ts // 900_000 * 900_000 - back * 60_000
        iso = time.strftime("%Y-%m-%dT%H:%M:00.000Z", time.gmtime(t / 1000))
        url = (f"{CLM_WMS}?service=WMS&version=1.3.0&request=GetMap&layers=msg_fes:clm&styles=&crs=EPSG:4326"
               f"&bbox={lat0},{lon0},{lat1},{lon1}&width={w}&height={h}&format=image/png&time={iso}")
        try:
            a = np.asarray(Image.open(io.BytesIO(fetch(url, timeout=25))).convert("RGBA"))
        except Exception:
            continue
        data = a[..., 3] > 128
        if data.mean() < 0.2:
            continue
        cloud = data & (a[..., 0] > 200) & (a[..., 1] > 200) & (a[..., 2] > 200)   # white = cloud; green / blue = clear
        frac = np.where(data, cloud.astype(np.float32), np.nan)[::-1]
        return lat0 + (np.arange(h) + 0.5) * CLM_DEG, lon0 + (np.arange(w) + 0.5) * CLM_DEG, frac
    return None


class SunGrid:
    """The sunny grid: a Web Mercator box (default GermanyDwd's at twice its cell size; EUROPE: see there)."""

    def __init__(self, bbox=GermanyDwd.BBOX, cell=CELL):
        self.BBOX, self.cell = bbox, cell
        x0, y0, x1, y1 = self.BBOX
        self.nx, self.ny = int(round((x1 - x0) / cell)), int(round((y1 - y0) / cell))
        xs = x0 + (np.arange(self.nx) + 0.5) * cell
        ys = y1 - (np.arange(self.ny) + 0.5) * cell
        lon = xs / WORLD * 180.0
        lat = np.degrees(2 * np.arctan(np.exp(ys / WORLD * np.pi)) - np.pi / 2)
        self.lat, self.lon = (a.astype(np.float32) for a in np.meshgrid(lat, lon, indexing="ij"))
        self.gx, self.gy = np.meshgrid(np.arange(self.nx, dtype=np.float32), np.arange(self.ny, dtype=np.float32))
        self.maps = {}  # source grid -> (crop rows, crop cols, remap tables)

    def to_pixels(self, x, y):
        x0, _, _, y1 = self.BBOX
        return (np.asarray(x) - x0) / self.cell - 0.5, (y1 - np.asarray(y)) / self.cell - 0.5

    def elev(self, ts):
        return sun_elevation(ts, self.lat, self.lon).astype(np.float32)

    def regrid(self, lat, lon, a, sigma=(0.0, 0.0)):
        """A field on a regular lat/lon grid onto the sunny grid (bilinear; optional blur in source pixels first, so a
        finer source is averaged rather than sampled)."""
        key = (lat[0], lat[-1], len(lat), lon[0], lon[-1], len(lon))
        if key not in self.maps:
            dlat, dlon = (lat[-1] - lat[0]) / (len(lat) - 1), (lon[-1] - lon[0]) / (len(lon) - 1)
            fy, fx = (self.lat - lat[0]) / dlat, (self.lon - lon[0]) / dlon
            r0, r1 = max(int(np.floor(fy.min())) - 3, 0), min(int(np.ceil(fy.max())) + 4, len(lat))
            c0, c1 = max(int(np.floor(fx.min())) - 3, 0), min(int(np.ceil(fx.max())) + 4, len(lon))
            self.maps[key] = (slice(r0, r1), slice(c0, c1), (fx - c0).astype(np.float32), (fy - r0).astype(np.float32))
        rows, cols, mx, my = self.maps[key]
        crop = np.ascontiguousarray(a[rows, cols], np.float32)
        valid = np.isfinite(crop)
        filled = np.where(valid, crop, 0.0).astype(np.float32)
        w = valid.astype(np.float32)
        if sigma[0] or sigma[1]:
            filled = cv2.GaussianBlur(filled, (0, 0), sigmaX=sigma[0] or 0.01, sigmaY=sigma[1] or 0.01)
            w = cv2.GaussianBlur(w, (0, 0), sigmaX=sigma[0] or 0.01, sigmaY=sigma[1] or 0.01)
        v = cv2.remap(filled, mx, my, cv2.INTER_LINEAR, borderMode=cv2.BORDER_CONSTANT, borderValue=0)
        n = cv2.remap(w, mx, my, cv2.INTER_LINEAR, borderMode=cv2.BORDER_CONSTANT, borderValue=0)
        return np.where(n > 0.5, v / np.maximum(n, 1e-6), np.nan).astype(np.float32)


def fill_nan(a, sigma=15.0):
    """Unknown pixels from their surroundings (normalised blur), what stays unknown -> 0.5."""
    ok = np.isfinite(a)
    if ok.all():
        return a
    w = cv2.GaussianBlur(ok.astype(np.float32), (0, 0), sigma)
    v = cv2.GaussianBlur(np.where(ok, a, 0).astype(np.float32), (0, 0), sigma)
    return np.where(ok, a, np.where(w > 1e-3, v / np.maximum(w, 1e-3), 0.5)).astype(np.float32)


class SunModel:
    """Weather model sunshine share (Open-Meteo minutely_15 sunshine_duration / 900 s) on a 0.1 deg grid over the
    shown area, for the "model" and "nwp" shadows."""

    def __init__(self, grid, url, available, log, lats=(46.0, 56.0), lons=(2.0, 19.0), deg=MODEL_DEG, models=None):
        self.grid, self.url, self.available, self.log = grid, url, available, log
        self.deg, self.models = deg, models
        self.lock = threading.Lock()
        self.times = self.share = None   # (T,) ms at the middle of each 15 min interval; (T, nlat, nlon)
        self.low_times = self.low = None  # hourly low cloud cover 0..1 (T, nlat, nlon), for fog under "clear" IR
        self.dni = None                   # direct normal irradiance W/m2 (T, nlat, nlon), same times as share (NWP blend)
        self.at = 0.0
        self.lats = np.arange(lats[0], lats[1] + 1e-6, deg)
        self.lons = np.arange(lons[0], lons[1] + 1e-6, deg)
        self.points = None
        self.maps = (((grid.lon - self.lons[0]) / deg).astype(np.float32),
                     ((grid.lat - self.lats[0]) / deg).astype(np.float32))
        # server-v2.06: the grid's pixels outside the model's box get NaN, not the box's edge values: replicated, they
        # drew straight stripes into the sunny map west of 12 W and north of 59.5 N (Europe, 26.11.2026)
        self.covers = ((self.maps[0] >= 0) & (self.maps[0] <= len(self.lons) - 1)
                       & (self.maps[1] >= 0) & (self.maps[1] <= len(self.lats) - 1))

    def refresh(self, inside, past_steps=4, forecast_steps=14):
        if not self.available() or time.time() - self.at < MODEL_REFRESH_S:
            return 0
        la, lo = (v.ravel() for v in np.meshgrid(self.lats, self.lons, indexing="ij"))
        if self.points is None:  # only where the map shows sunny (plus a margin)
            near = cv2.dilate((inside > 0).astype(np.uint8), np.ones((15, 15), np.uint8))
            col, row = (np.round(v).astype(int) for v in self.grid.to_pixels(*merc(lo, la)))
            ok = (col >= 0) & (col < self.grid.nx) & (row >= 0) & (row < self.grid.ny)
            ok[ok] = near[row[ok], col[ok]] > 0
            self.points = np.flatnonzero(ok)
        la, lo = la[self.points], lo[self.points]
        times, vals, low_times, low = None, None, None, None
        for i in range(0, len(la), MODEL_BATCH):
            body = urllib.parse.urlencode({
                "latitude": ",".join(f"{v:.2f}" for v in la[i:i + MODEL_BATCH]),
                "longitude": ",".join(f"{v:.2f}" for v in lo[i:i + MODEL_BATCH]),
                "minutely_15": "sunshine_duration,direct_normal_irradiance", "past_minutely_15": str(past_steps),
                "forecast_minutely_15": str(forecast_steps),
                "hourly": "cloud_cover_low", "past_hours": str(past_steps // 4 + 2),
                "forecast_hours": str(forecast_steps // 4 + 2), "timeformat": "unixtime",
                **({"models": self.models} if self.models else {})}).encode()
            req = urllib.request.Request(f"{self.url}/v1/forecast", data=body,
                                         headers={"Content-Type": "application/x-www-form-urlencoded"})
            with urllib.request.urlopen(req, timeout=600) as r:
                d = json.loads(r.read())
            d = d if isinstance(d, list) else [d]
            if times is None:
                times = np.array(d[0]["minutely_15"]["time"], np.int64) * 1000 - 450_000  # value = previous 15 min
                vals = np.zeros((len(times), len(la)), np.float32)
                dni = np.full((len(times), len(la)), np.nan, np.float32)
            if low_times is None:
                low_times = np.array(d[0]["hourly"]["time"], np.int64) * 1000  # instantaneous value at the full hour
                low = np.full((len(low_times), len(la)), np.nan, np.float32)
            for j, e in enumerate(d):
                vals[:, i + j] = [min(1.0, (v or 0.0) / 900.0) for v in e["minutely_15"]["sunshine_duration"]]
                if e["minutely_15"].get("direct_normal_irradiance"):
                    dni[:, i + j] = [np.nan if v is None else v for v in e["minutely_15"]["direct_normal_irradiance"]]
                low[:, i + j] = [np.nan if v is None else v / 100.0 for v in e["hourly"]["cloud_cover_low"]]
        share = np.full((len(times), len(self.lats) * len(self.lons)), np.nan, np.float32)
        share[:, self.points] = vals
        share = share.reshape(len(times), len(self.lats), len(self.lons))
        dni_g = np.full((len(times), len(self.lats) * len(self.lons)), np.nan, np.float32)
        dni_g[:, self.points] = dni
        dni_g = dni_g.reshape(len(times), len(self.lats), len(self.lons))
        low_g = np.full((len(low_times), len(self.lats) * len(self.lons)), np.nan, np.float32)
        low_g[:, self.points] = low
        low_g = low_g.reshape(len(low_times), len(self.lats), len(self.lons))
        with self.lock:
            self.times, self.share, self.dni = times, share, dni_g
            self.low_times, self.low = low_times, low_g
        self.at = time.time()
        return len(la)

    def field(self, ts, what="share"):
        """Model sunshine share ("share"), low cloud cover ("low", 0..1) or direct normal irradiance ("dni", W/m2) on
        the sunny grid at ts (NaN outside the model's box); None outside its times."""
        with self.lock:
            times, share = {"share": (self.times, self.share), "low": (self.low_times, self.low),
                            "dni": (self.times, self.dni)}[what]
        if share is None or not times[0] <= ts <= times[-1]:
            return None
        k = min(int(np.searchsorted(times, ts, side="right")) - 1, len(times) - 2)
        f = (ts - times[k]) / (times[k + 1] - times[k])
        m = ((1 - f) * share[k] + f * share[k + 1]).astype(np.float32)
        v = cv2.remap(fill_nan(m, 3.0), *self.maps, cv2.INTER_LINEAR, borderMode=cv2.BORDER_REPLICATE)
        return np.where(self.covers, v, np.nan).astype(np.float32)


class SunStations:
    """DWD 10-minute station sunshine duration (SD_10, hours per 10 min -> share = SD_10 * 6), read politely like
    gauges.GaugeDrizzle: one listing request every gauges.LIST_EVERY_S shows which files changed, only those are read,
    gauges.PACE_S apart per worker, stopping after gauges.MAX_ERRORS failures in a row, with an overall deadline."""

    def __init__(self, fetch, log):
        self.fetch, self.log = fetch, log
        self.lock = threading.Lock()
        self.stations = {}  # id -> (lat, lon)
        self.values = {}    # id -> {end time ms: share}
        self.errors = 0
        self.seen, self.listed_day = {}, None

    def _station_list(self):
        text = self.fetch(f"{SOLAR}zehn_now_sd_Beschreibung_Stationen.txt", timeout=60).decode("latin-1")
        recent = time.strftime("%Y%m%d", time.gmtime(time.time() - 3 * 86400))
        out = {}
        for line in text.splitlines()[2:]:
            p = line.split()
            if len(p) >= 6 and p[0].isdigit() and p[2] >= recent:
                out[p[0]] = (float(p[4]), float(p[5]))
        return out

    def _read_one(self, sid):
        if self.errors >= gauges.MAX_ERRORS:
            return None
        time.sleep(gauges.PACE_S)
        try:
            z = zipfile.ZipFile(io.BytesIO(self.fetch(f"{SOLAR}10minutenwerte_SOLAR_{sid}_now.zip", timeout=30, tries=1)))
            rows = z.read(z.namelist()[0]).decode("latin-1").splitlines()
            self.errors = 0
        except Exception:
            self.errors += 1
            return None
        k = [h.strip() for h in rows[0].split(";")].index("SD_10")
        vals = {}
        for r in rows[1:]:
            p = r.split(";")
            try:
                v = float(p[k])
                if v >= 0:
                    vals[stamp_ts(p[1].strip())] = min(1.0, v * 6.0)
            except (ValueError, IndexError):
                continue
        return vals

    def poll(self):
        if time.strftime("%Y%m%d") != self.listed_day:
            self.stations, self.listed_day = self._station_list(), time.strftime("%Y%m%d")
        text = self.fetch(SOLAR, timeout=60).decode("latin-1")
        stamps = {sid: s for sid, s in re.findall(r'SOLAR_(\d{5})_now\.zip</a>\s+(\S+ \S+)', text) if sid in self.stations}
        todo = [sid for sid, s in stamps.items() if self.seen.get(sid) != s]
        if not todo:
            return 0
        self.errors = 0
        pool = ThreadPoolExecutor(gauges.WORKERS)
        futs = {pool.submit(self._read_one, sid): sid for sid in todo}
        got = {}
        try:
            for f in as_completed(futs, timeout=gauges.READ_DEADLINE_S):
                if f.result() is not None:
                    got[futs[f]] = f.result()
        except TimeoutError:
            pass
        pool.shutdown(wait=False, cancel_futures=True)
        cut = time.time() * 1000 - TRUTH_KEEP_MS
        with self.lock:
            for sid, vals in got.items():
                d = self.values.setdefault(sid, {})
                d.update({t: v for t, v in vals.items() if t >= cut})
                for t in [t for t in d if t < cut]:
                    del d[t]
        self.seen.update({sid: stamps[sid] for sid in got})
        return len(got)

    def truth(self, end_ts):
        """{id: share} of the stations that reported the 10 min ending at end_ts."""
        with self.lock:
            return {sid: v[end_ts] for sid, v in self.values.items() if end_ts in v}

    def newest(self):
        with self.lock:
            ends = [max(v) for v in self.values.values() if v]
        return int(np.median(ends)) if ends else 0

    def loop(self):
        while True:
            started = time.time()
            try:
                n = self.poll()
                if n:
                    self.log(f"[{datetime.now():%H:%M:%S}] sunny: {n} station files in {time.time() - started:.0f}s")
            except Exception as e:
                self.log(f"sunny stations: {e}")
            time.sleep(max(30.0, gauges.LIST_EVERY_S - (time.time() - started)))


GEOSPHERE = "https://dataset.api.hub.geosphere.at/v1/station/current/tawes-v1-10min"
SMN_ITEMS = "https://data.geo.admin.ch/api/stac/v1/collections/ch.meteoschweiz.ogd-smn/items?limit=100"
# Swiss stations are one file each (~395 SwissMetNet items): read every 5 min day and night that was ~46k requests a
# day for values that change every 10 min and are 0 at night. Now (server-v1.79, 20.11.2026) every CH_EVERY_S, only
# while the sun is above CH_SUN_ELEV somewhere in CH_BOX (Switzerland + Austria), and never again on a day for a
# station whose file has no sunshine column. The values only feed the verification (station truth, daytime only).
CH_EVERY_S = 570               # ~10 min (a little less, so that a 300 s polling loop takes every second round)
CH_BOX = (45.8, 49.1, 5.9, 17.2)  # lat min, lat max, lon min, lon max
CH_SUN_ELEV = -2.0             # deg: a small margin before sunrise / after sunset
NETWORK_LIVE_MS = 2 * 3600_000 # EuSunStations.newest: a network whose newest value is this far behind is skipped


class EuSunStations(SunStations):
    """The Europe instance's station truth: the DWD stations plus Austria (GeoSphere TAWES, open data CC BY 4.0: "SO",
    sunshine seconds in the last 10 min) and Switzerland (MeteoSwiss SwissMetNet open data, source MeteoSwiss:
    "sre000z0", sunshine minutes in 10 min). Ids "AT:<id>" / "CH:<abbr>". The timestamps are taken as the END of the
    10 min like DWD's SD_10 (to be checked against the satellite analysis once a few days exist)."""

    def __init__(self, fetch, log):
        super().__init__(fetch, log)
        self.at_ids, self.ch_items, self.extra_day = [], {}, None
        self.ch_at, self.ch_nosun = 0.0, set()  # last Swiss round (time.time()); stations without sunshine today

    @staticmethod
    def _ch_daylight(ts_ms):
        """True while the sun is above CH_SUN_ELEV somewhere in CH_BOX."""
        lat0, lat1, lon0, lon1 = CH_BOX
        la, lo = np.meshgrid(np.linspace(lat0, lat1, 3), np.linspace(lon0, lon1, 5), indexing="ij")
        return float(np.max(sun_elevation(ts_ms, la, lo))) > CH_SUN_ELEV

    def _extra_lists(self):
        d = json.loads(self.fetch(f"{GEOSPHERE}/metadata", timeout=60))
        at = {f"AT:{x['id']}": (float(x["lat"]), float(x["lon"])) for x in d.get("stations", []) if x.get("is_active")}
        ch, url = {}, SMN_ITEMS
        while url:
            j = json.loads(self.fetch(url, timeout=60))
            for f in j.get("features", []):
                lon, lat = f["geometry"]["coordinates"][:2]
                href = next((a["href"] for k, a in f["assets"].items() if k.endswith("_t_now.csv")), None)
                if href:
                    ch[f"CH:{f['id']}"] = (float(lat), float(lon), href)
            url = next((l["href"] for l in j.get("links", []) if l.get("rel") == "next"), None)
        return at, ch

    def _poll_extra(self):
        if time.strftime("%Y%m%d") != self.extra_day:
            at, ch = self._extra_lists()
            self.at_ids, self.ch_items, self.extra_day = [k[3:] for k in at], ch, time.strftime("%Y%m%d")
            self.ch_nosun = set()
            with self.lock:
                self.stations.update(at)
                self.stations.update({k: v[:2] for k, v in ch.items()})
        got = {}
        # Austria: the current 10 min of every station in one request
        d = json.loads(self.fetch(f"{GEOSPHERE}?parameters=SO&station_ids={','.join(self.at_ids)}", timeout=60))
        # stations that are behind make the answer list several times; each data array is aligned with them
        ts = [int(datetime.fromisoformat(x).timestamp() * 1000) for x in d.get("timestamps", [])]
        for f in d.get("features", []):
            vals = {t: min(1.0, v / 600.0) for t, v in zip(ts, f["properties"]["parameters"]["SO"]["data"])
                    if v is not None and v >= 0}
            if vals:
                got[f"AT:{f['properties']['station']}"] = vals
        # Switzerland: one small file per station (today's 10-min values), see CH_EVERY_S
        now = time.time()
        ch_due = now - self.ch_at >= CH_EVERY_S and self._ch_daylight(now * 1000)
        if ch_due:
            self.ch_at = now
        for sid, (_, _, href) in (self.ch_items.items() if ch_due else ()):
            if sid in self.ch_nosun:
                continue
            time.sleep(0.2)
            try:
                rows = self.fetch(href, timeout=30, tries=1).decode("latin-1").splitlines()
            except Exception:
                continue
            head = rows[0].split(";")
            if "sre000z0" not in head:
                if len(head) > 3:  # a real header without sunshine (not an error page)
                    self.ch_nosun.add(sid)
                continue
            k = head.index("sre000z0")
            vals = {}
            for r in rows[-30:]:  # the last ~5 h
                p = r.split(";")
                try:
                    vals[int(datetime.strptime(p[1], "%d.%m.%Y %H:%M").replace(tzinfo=timezone.utc).timestamp() * 1000)] =                         min(1.0, float(p[k]) / 10.0)
                except (ValueError, IndexError):
                    continue
            if vals:
                got[sid] = vals
        cut = time.time() * 1000 - TRUTH_KEEP_MS
        with self.lock:
            for sid, vals in got.items():
                dd = self.values.setdefault(sid, {})
                dd.update({tt: v for tt, v in vals.items() if tt >= cut})
                for tt in [tt for tt in dd if tt < cut]:
                    del dd[tt]
        return len(got)

    def poll(self):
        n = super().poll()
        try:
            n += self._poll_extra()
        except Exception as e:
            self.log(f"sunny stations AT/CH: {e}")
        return n

    @staticmethod
    def network(sid):
        return sid[:2] if sid[:3] in ("AT:", "CH:") else "DE"

    def newest(self):
        """The newest 10-min end that EVERY live network has reached (min over DE / AT / CH of each one's median newest
        end; a network with nothing in NETWORK_LIVE_MS does not hold the others back). Until 27.11.2026 this was the
        median over all stations: Austria's ~265 stations (GeoSphere, ~10 min late) outnumber the ~60 DWD ones (files
        ~20-40 min late) and the Swiss ones (read every ~10 min), so a run was scored before DWD / MeteoSwiss had its
        end - the Europe station rows were Austria alone (26.11.: Austria overcast all day -> MAE 0.002)."""
        with self.lock:
            ends = {}
            for sid, v in self.values.items():
                if v:
                    ends.setdefault(self.network(sid), []).append(max(v))
        if not ends:
            return 0
        top = max(max(e) for e in ends.values())
        live = [int(np.median(e)) for e in ends.values() if top - max(e) <= NETWORK_LIVE_MS]
        return min(live)


class SunVerify:
    """Scores every forecast run against the later SID analysis (grid) and the DWD stations (point).
    Also at the stations: the analysis itself, variant "ana" at lead 0 (how well the satellite image says "sun or not"
    where a gauge measures it - the ceiling for every forecast; rows of their own in score_sun.py).
    The forecasts waiting for their truth are kept on disk (fcverify.Pending) and a run is taken only once, also across
    restarts (last_ref): before 10.11. a restart lost them, and the last hour's runs were made again from the reloaded
    images (the first of them without motion) and replaced the real rows. log_dir given (replays): memory only."""

    def __init__(self, sun, stations, log, log_dir=None, prefix="", stations_only=False, group=None,
                 lag_ms=STATION_LAG_MS):
        """stations_only: a second station truth for the same runs (synop.SynopFeed.sky for all of Europe, 16.11.2026)
        - its own station rows (<prefix>sunstation_*), no grid rows and no timeline (the main verifier writes those).
        group: station id -> group name (e.g. the country); station rows then also hold "by" = {group: [n, sum |error|,
        sum error, hits]} so score_sun.py --by can tell where a variant wins or loses (19.11.2026: against SYNOP cover
        persistence beat every forecast Europe-wide, and the all-stations mean cannot say where).
        lag_ms: the station truth for valid time v is stations.truth(v + lag_ms) (DWD SD_10: 10 min; SYNOP hourly
        sunshine duration, synop.HourlySun: 30 min = the forecast at the middle of the hour the minutes are summed over)."""
        self.sun, self.stations, self.log, self.prefix = sun, stations, log, prefix
        self.lag_ms = lag_ms
        self.group = group
        self.stations_only = stations_only
        self._pix, self._pix_n = None, -1
        self.persist = log_dir is None
        self.log_dir = log_dir or LOG_DIR
        g = sun.grid
        self.shape = (int(round(g.nx / VERIFY_DOWN)), int(round(g.ny / VERIFY_DOWN)))  # cv2 (w, h)
        # (variant, ref, lead) -> g: share on the verification grid (uint8, x 250, 255 = unknown)
        self.pending = Pending(prefix + "sungrid" if self.persist else None)
        # (variant, ref, lead) -> sid, v: station ids and the forecast share there
        self.spending = Pending(prefix + "sunstation" if self.persist else None)
        self.done_ref = None
        self.last_file = os.path.join(self.log_dir, "pending", prefix + "sun_last_ref.txt")
        try:
            with open(self.last_file) as f:
                self.last_ref = int(f.read().strip()) if self.persist else 0
        except (OSError, ValueError):
            self.last_ref = 0

    def _where(self):
        return self.stations.stations if self.stations else {}

    def _pixels(self):
        """Station pixels on the sunny grid: the instance's own stations, or this verifier's other station set."""
        if self.stations is self.sun.stations:
            return self.sun.station_pixels()
        where = self._where()
        if len(where) != self._pix_n:
            pix = {}
            for sid, (lat, lon) in list(where.items()):
                col, row = (int(round(float(v))) for v in self.sun.grid.to_pixels(*merc(lon, lat)))
                if 1 <= col < self.sun.grid.nx - 1 and 1 <= row < self.sun.grid.ny - 1:
                    pix[sid] = (row, col)
            self._pix, self._pix_n = pix, len(where)
        return self._pix

    def _down(self, a):
        ok = np.isfinite(a)
        v = cv2.resize(np.where(ok, a, 0).astype(np.float32), self.shape, interpolation=cv2.INTER_AREA)
        w = cv2.resize(ok.astype(np.float32), self.shape, interpolation=cv2.INTER_AREA)
        return np.where(w > 0.5, v / np.maximum(w, 1e-6), np.nan).astype(np.float32)

    @staticmethod
    def _counts(fc, obs, mask):
        fc, obs = fc & mask, obs & mask
        k = np.ones((2 * TOL_PX + 1,) * 2, np.uint8)
        near_obs = cv2.dilate(obs.astype(np.uint8), k) > 0
        near_fc = cv2.dilate(fc.astype(np.uint8), k) > 0
        return int((fc & near_obs).sum()), int((obs & ~near_fc).sum()), int((fc & ~near_obs).sum())

    def take(self, ref, variants):
        """Keep the leads of a new run: variants = {name: {valid ts: share grid}}."""
        if ref == self.done_ref or ref <= self.last_ref:
            return
        sts = self._pixels()

        def at_stations(key, g):
            if sts:
                vals = {sid: v for sid, v in self.sun.at_pixels(g, sts).items() if np.isfinite(v)}
                self.spending.put(key, sid=np.array(list(vals), dtype=str), v=np.array(list(vals.values()), np.float32))
        for name, grids in variants.items():
            for lead in LEADS:
                g = grids.get(ref + lead * 60_000)
                if g is None:
                    continue
                if not self.stations_only:
                    d = self._down(g)
                    self.pending.put((name, ref, lead), g=np.where(np.isfinite(d), np.round(d * 250), 255).astype(np.uint8))
                at_stations((name, ref, lead), g)
        a = self.sun.obs_share(ref)
        if a is not None:
            at_stations(("ana", ref, 0), a)
        self.done_ref = ref
        if self.persist:
            self.last_ref = ref
            try:
                with open(self.last_file, "w") as f:
                    f.write(str(ref))
            except OSError:
                pass
        if not self.stations_only:
            self._timeline(ref)

    def _timeline(self, ref):
        """The run's map frames (last hour + forecast) as the app draws them, step by step: monitor.timeline."""
        try:
            with self.sun.lock:
                disp = {t: d for t, d in self.sun.disp.items() if t >= ref - 3600_000}
            times = sorted(disp)
            # the displayed sun share (SunnyDE._display), every 2nd cell
            share = lambda d: np.where(d[::2, ::2] == 255, np.nan, 1.0 - (d[::2, ::2].astype(np.float32) / 254 - 0.05) / 0.8)
            monitor.timeline("sun", self.prefix or "de", ref, [(t - ref) // 60_000 for t in times],
                             (share(disp[t]) for t in times), self.sun.inside()[::2, ::2] > 0.5,
                             sum(t <= ref for t in times), log_dir=self.log_dir)
        except Exception as e:
            self.log(f"sunny timeline failed: {e}")

    def _write(self, kind, lines):
        if not lines:
            return
        os.makedirs(self.log_dir, exist_ok=True)
        by_day = {}
        for row in lines:
            by_day.setdefault(datetime.fromtimestamp(row["valid"] / 1000, timezone.utc).strftime("%Y%m%d"), []).append(row)
        for day, rows in by_day.items():
            with open(os.path.join(self.log_dir, f"{kind}_{day}.jsonl"), "a") as f:
                for row in rows:
                    f.write(json.dumps(row) + "\n")

    def step(self, now_ref):
        """Score what can be scored; now_ref = the newest analysis time (the server's clock for this purpose)."""
        sun, lines, slines = self.sun, [], []
        mask_area = self._down(sun.inside()) > 0.5
        for key in self.pending:
            variant, r, lead = key
            valid = r + lead * 60_000
            a = sun.obs_share(valid)
            if a is None:
                if now_ref - r > OBS_KEEP_MS:
                    del self.pending[key]
                continue
            z = self.pending.get(key)
            del self.pending[key]
            if z is None or z["g"].shape != mask_area.shape:
                continue
            a = self._down(a)
            g = np.where(z["g"] == 255, np.nan, z["g"] / 250.0).astype(np.float32)
            e = self._down(sun.elev(valid))
            mask = mask_area & (e >= SCORE_ELEV) & np.isfinite(a) & np.isfinite(g)
            n = int(mask.sum())
            if n < MIN_PX:
                continue
            d = g[mask] - a[mask]
            lines.append({"variant": variant, "ref": r, "lead": lead, "valid": valid, "v": monitor.TAG, "n": n,
                          "obs": int((a[mask] >= SUNNY).sum()), "fcn": int((g[mask] >= SUNNY).sum()),
                          "fc": list(self._counts(g >= SUNNY, a >= SUNNY, mask)),
                          "bias": round(float(d.mean()), 4), "mae": round(float(np.abs(d).mean()), 4)})
        newest = self.stations.newest() if self.stations else 0
        for key in self.spending:
            variant, r, lead = key
            valid = r + lead * 60_000
            end = valid + self.lag_ms
            if newest < end:
                if now_ref - r > STATION_WAIT_MS:
                    del self.spending[key]
                continue
            z = self.spending.get(key)
            del self.spending[key]
            if z is None:
                continue
            fc = dict(zip((str(x) for x in z["sid"]), (float(x) for x in z["v"])))
            truth = self.stations.truth(end)
            where = self._where()
            keys = [s for s in fc if s in truth and s in where and sun_elevation(valid, where[s][0], where[s][1]) >= SCORE_ELEV]
            if len(keys) < 5:
                continue
            f, o = np.array([(fc[s], truth[s]) for s in keys], np.float32).T
            row = {"variant": variant, "ref": r, "lead": lead, "valid": valid, "v": monitor.TAG, "n": len(keys),
                   "mae": round(float(np.abs(f - o).mean()), 4), "bias": round(float((f - o).mean()), 4),
                   "hit": round(float(((f >= SUNNY) == (o >= SUNNY)).mean()), 4),
                   "obs": int((o >= SUNNY).sum()), "fcn": int((f >= SUNNY).sum())}
            # broken cloud at the run's start (the satellite analysis at the station between 0.15 and 0.85 sun share):
            # scored apart, for Schaefchenwolken & co. (15.11.2026; score_sun.py --broken)
            a0 = sun.obs_share(r)
            if a0 is not None:
                ana = sun.at_pixels(a0, self._pixels())
                b = np.array([0.15 <= ana.get(s, np.nan) <= 0.85 for s in keys])
                if b.any():
                    row.update({"nb": int(b.sum()), "mae_b": round(float(np.abs(f - o)[b].mean()), 4),
                                "bias_b": round(float((f - o)[b].mean()), 4),
                                "hit_b": round(float(((f >= SUNNY) == (o >= SUNNY))[b].mean()), 4)})
            if self.group:
                by = {}
                for s, fv, ov in zip(keys, f, o):
                    g = self.group(s)
                    if g is None:
                        continue
                    a = by.setdefault(g, [0, 0.0, 0.0, 0])
                    a[0] += 1; a[1] += abs(float(fv - ov)); a[2] += float(fv - ov); a[3] += int((fv >= SUNNY) == (ov >= SUNNY))
                row["by"] = {g: [a[0], round(a[1], 3), round(a[2], 3), a[3]] for g, a in by.items()}
            slines.append(row)
        self._write(self.prefix + "sungrid", lines)
        self._write(self.prefix + "sunstation", slines)
        return len(lines), len(slines)


class SunnyDE:
    name = "sunny"

    def __init__(self, fetch, log, de, openmeteo_url=None, openmeteo_available=lambda: False, stations=True, log_dir=None,
                 area=None):
        """de: the Germany radar (its shown area is sunny's); area: EUROPE (then de may be None: the whole grid)."""
        self.fetch, self.log, self.de, self.area = fetch, log, de, area
        self.grid = SunGrid(area["bbox"], area["cell"]) if area else SunGrid()
        self.lock = threading.Lock()
        self.obs, self.fc, self.ref = {}, {}, None  # ts -> share grid (float16, NaN = unknown); ref = newest image
        self.disp = {}                              # ts -> cloud value for the map (uint8, 255 = unknown)
        # the map frames of the last KEEP_RUNS runs: a tile URL names its run (key = ref) and is cached as immutable, so
        # it must show that run's frames - drawing the current ones under an older key mixed runs in one timeline
        self.disp_snaps = OrderedDict()             # ref -> {ts: cloud value}
        self.k = {}                                 # ts -> clear-sky index (float16), last K_KEEP_MS only
        self.shift_ms = 0                           # DEV replay: frames relabelled by this, the sun stays at the real time
        self.frozen = False
        self.stations = ((EuSunStations if area else SunStations)(fetch, log)) if stations else None
        self.model = (SunModel(self.grid, openmeteo_url, openmeteo_available, log, **(area or {}).get("model", {}))
                      if openmeteo_url else None)
        self.verify = SunVerify(self, self.stations, log, log_dir, prefix=(area or {}).get("prefix", ""))
        self._inside = None
        self.cycle_s = None
        self._st_pix = None
        self.skipped, self.skips_in_row = set(), 0  # slots rejected as implausible (JUMP_MAX)
        self._clm_cache = {}                        # 15-min slot -> EUMETSAT cloud mask on the grid (clm_cloud)
        self.extra_verify = []                      # SunVerify(stations_only=True): other station truths (SYNOP)
        self._sid_seen = {}                         # slot -> time.time() its SIDhr file was first listed (_wait_cth)
        self.ir_only = set()                        # slots made from the CTH file alone (DWD's SIDhr late, IR_FALLBACK)
        self._metar = {}                            # METAR station -> (lat, lon, time ms, fog), Europe only
        self._k_fog = None                          # (slot, k with the METAR fog cap) of the last ingest, or None
        self._k_low = None                          # (slot, k with low_f_shadow) of the last ingest, or None
        # the low-sun clear beam factor (see LOW_F) served here, and the other one, scored as a shadow
        self.low_f = LOW_F if area else LOW_F_DE
        self.low_f_shadow = 1.0 if self.low_f < 1.0 else 0.3

    # ---------- helpers ----------
    def inside(self):
        """Area the map shows (Germany + border zone, from the rain radar's mask) on the sunny grid; Europe: all of it."""
        if self.de is None:
            if self._inside is None:
                self._inside = np.ones((self.grid.ny, self.grid.nx), np.float32)
            return self._inside
        if self._inside is None or getattr(self, "_inside_from", None) is not self.de.inside:
            self._inside = cv2.resize(self.de.inside, (self.grid.nx, self.grid.ny), interpolation=cv2.INTER_AREA)
            self._inside_from = self.de.inside
        return self._inside

    def elev(self, ts):
        return self.grid.elev(ts - self.shift_ms)

    def station_coords(self):
        return self.stations.stations if self.stations else {}

    def station_pixels(self):
        where = self.station_coords()
        if not where:
            return {}
        if self._st_pix is None or len(self._st_pix) != len(where):
            pix = {}
            for sid, (lat, lon) in where.items():
                col, row = (int(round(float(v))) for v in self.grid.to_pixels(*merc(lon, lat)))
                if 1 <= col < self.grid.nx - 1 and 1 <= row < self.grid.ny - 1:
                    pix[sid] = (row, col)
            self._st_pix = pix
        return self._st_pix

    @staticmethod
    def at_pixels(g, pix):
        """3 x 3 mean (~6 km, like /api/cloud/point) of a grid at station pixels."""
        out = {}
        for sid, (row, col) in pix.items():
            c = g[row - 1:row + 2, col - 1:col + 2].astype(np.float32)
            out[sid] = float(np.nanmean(c)) if np.isfinite(c).any() else float("nan")
        return out

    def obs_share(self, ts):
        with self.lock:
            # a slot from the IR cloud tops alone (IR_FALLBACK) is shown, but never the truth the forecasts are scored on
            g = None if ts in self.ir_only else self.obs.get(ts)
        return None if g is None else g.astype(np.float32)

    # ---------- observation ----------
    def ingest(self, ts, sid_data, cth_data=None):
        """One satellite slot: SIDhr file bytes (+ CTH file bytes) -> k, share and map value at ts.
        sid_data None (IR_FALLBACK, DWD's SIDhr late): the weather model's k in place of the DNI's."""
        e = self.elev(ts)
        if sid_data is None:
            mk = self._model_k(ts)
            k_dni = mk if mk is not None else np.full(e.shape, np.nan, np.float32)
            k_dni_o = None
        else:
            lat, lon, v = read_nc(sid_data, ["DNI"])
            dni = self.grid.regrid(lat, lon, v["DNI"], sigma=(0.8, 0.5))
            del v
            k_dni = np.where(e >= SUN_ELEV, np.clip(dni / clear_dni(e, self.low_f), 0, K_MAX), np.nan).astype(np.float32)
            # the same with the other low-sun factor, for the shadow "lowf03" / "lowf10" (see LOW_F)
            k_dni_o = (np.where(e >= SUN_ELEV, np.clip(dni / clear_dni(e, self.low_f_shadow), 0, K_MAX), np.nan)
                       .astype(np.float32) if self.low_f_shadow != self.low_f else None)
        k_ir = None
        if cth_data is not None:
            lat, lon, c = read_nc(cth_data, ["CTH"])
            k_ir = k_from_cth(self.grid.regrid(lat, lon, c["CTH"]))
            # the file's outermost rows / columns read 0 (= no cloud): along its north edge at 63 N that drew a straight
            # cloud-free line across Norway and Finland (15.11.2026) - a 0.3 deg margin is left to the next source
            m = 0.3
            edge = ((self.grid.lat > lat.max() - m) | (self.grid.lat < lat.min() + m)
                    | (self.grid.lon > lon.max() - m) | (self.grid.lon < lon.min() + m))
            k_ir = np.where(edge, np.nan, k_ir).astype(np.float32)
            low = self.model.field(ts, "low") if self.model and LOWCC_W > 0 else None
            if low is not None:  # fog / low stratus the IR cannot tell from the ground
                # night and rising sun only: replay 08./09.11. vs the DWD stations, morning MAE +60/+90/+120
                # .102/.110/.124 -> .078/.084/.086, midday unchanged, but evening +10 .099 -> .127 (sunset: no fog cap)
                fog_time = (e < 0) | (self.elev(ts + 1_800_000) > e) | (sid_data is None)  # IR alone: no DNI sees the fog
                cap = np.clip(1.0 - LOWCC_W * np.nan_to_num(low), 0.0, 1.0)  # no cap outside the model's box
                k_ir = np.where(fog_time, np.minimum(k_ir, cap), k_ir).astype(np.float32)
        k_ir_o = k_ir  # the low-sun shadow keeps the IR k as it was
        if k_ir is not None and IR_F != 1.0:
            k_ir = (k_ir * IR_F).astype(np.float32)
        if k_ir is None:
            k, k_o = k_dni, k_dni_o
        else:  # DNI by day, IR cloud tops at low sun and night, blended in between; either one fills in for the other
            w = np.clip((e - SUN_ELEV) / (DAY_ELEV - SUN_ELEV), 0.0, 1.0)
            kd = np.where(np.isfinite(k_dni), k_dni, k_ir)
            ki = np.where(np.isfinite(k_ir), k_ir, kd)
            k = (w * kd + (1 - w) * ki).astype(np.float32)
            k_o = None
            if k_dni_o is not None:
                kd_o = np.where(np.isfinite(k_dni_o), k_dni_o, k_ir_o)
                k_o = (w * kd_o + (1 - w) * np.where(np.isfinite(k_ir_o), k_ir_o, kd_o)).astype(np.float32)
        k_fog = None
        cap = self._metar_cap(ts) if k_ir is not None else None
        if cap is not None:  # METAR fog (see METAR_URL): the IR part capped; by day the DNI sees the fog itself
            k_fog = (w * kd + (1 - w) * np.minimum(ki, cap)).astype(np.float32)
            if METAR_FOG_SERVE:
                k, k_fog = k_fog, None
        if (self.area or {}).get("clm") and not np.isfinite(k).all():  # beyond DWD's area: EUMETSAT's cloud mask
            t_clm, got = self._clm_cache.get("now", (0, None))
            if got is not None and abs(ts - t_clm) <= 60 * 60_000:  # the newest one the background thread has
                k = np.where(np.isfinite(k), k, 1.0 - 0.92 * got).astype(np.float32)
                if k_fog is not None:
                    k_fog = np.where(np.isfinite(k_fog), k_fog, 1.0 - 0.92 * got).astype(np.float32)
                if k_o is not None:
                    k_o = np.where(np.isfinite(k_o), k_o, 1.0 - 0.92 * got).astype(np.float32)
        self._k_fog = (ts, k_fog) if k_fog is not None else None  # for the shadow "metfog" (add_slot -> forecast)
        self._k_low = (ts, k_o) if k_o is not None else None      # for the low-sun shadow
        share = share_from_k(k, e, self.low_f)
        return k, share, self._display(k, e)

    def _metar_cap(self, ts):
        """Cap for the IR k from the METAR stations that report fog near the slot ts (1 = none, 0 at the station:
        METAR_FOG_R0 / R1), or None when none does."""
        t = ts - self.shift_ms
        with self.lock:
            pts = [(la, lo) for la, lo, tr, fog in self._metar.values() if fog and abs(tr - t) <= METAR_FOG_AGE_MS]
        if not pts:
            return None
        g = self.grid
        cap = np.ones((g.ny, g.nx), np.float32)
        la, lo = np.array(pts, np.float64).T
        cols, rows = g.to_pixels(*merc(lo, la))
        for lat, c, r in zip(la, cols, rows):
            km = g.cell / 1000.0 * math.cos(math.radians(lat))  # Web Mercator cell -> km on the ground
            rad = int(math.ceil(METAR_FOG_R1 / km)) + 1
            r0, r1, c0, c1 = max(int(r) - rad, 0), min(int(r) + rad + 1, g.ny), max(int(c) - rad, 0), min(int(c) + rad + 1, g.nx)
            if r0 >= r1 or c0 >= c1:
                continue
            yy, xx = np.mgrid[r0:r1, c0:c1]
            d = np.hypot(xx - c, yy - r) * km
            cap[r0:r1, c0:c1] = np.minimum(cap[r0:r1, c0:c1], np.clip((d - METAR_FOG_R0) / (METAR_FOG_R1 - METAR_FOG_R0), 0, 1))
        return cap if cap.min() < 1.0 else None

    def metar_loop(self):
        """The European METARs (fog) every METAR_EVERY_S, in the background (never in the slot loop)."""
        while True:
            try:
                got = metar_fog(self.fetch(METAR_URL, timeout=60))
                cut = time.time() * 1000 - 3 * 3600_000
                with self.lock:
                    for sid, rep_ in got.items():
                        if sid not in self._metar or rep_[2] >= self._metar[sid][2]:
                            self._metar[sid] = rep_
                    self._metar = {sid: v for sid, v in self._metar.items() if v[2] >= cut}
            except Exception as e:
                self.log(f"sunny: METAR failed: {e}")
            time.sleep(METAR_EVERY_S)

    def _display(self, k, e):
        """Cloud value for the map: the sun share k gives with the sun at least DISP_ELEV high (= the share by day, the
        clouds at a low sun and at night: no fading to grey towards sunset, no seam at the terminator); encoded so that
        the app's sunShare() returns that share."""
        s = np.nan_to_num(share_from_k(k, np.maximum(e, DISP_ELEV), self.low_f))
        v = np.clip(0.05 + 0.8 * (1.0 - s), 0, 1) * 254
        return np.where(np.isfinite(k), v, 255).astype(np.uint8)

    # ---------- forecast ----------
    def _motion(self, k0, k1, dt_min):
        img = lambda k: (np.clip(1.0 - fill_nan(k), 0, 1) * 255).astype(np.uint8)
        s = (self.grid.nx // FLOW_DOWN, self.grid.ny // FLOW_DOWN)
        a = cv2.resize(img(k0), s, interpolation=cv2.INTER_AREA)
        b = cv2.resize(img(k1), s, interpolation=cv2.INTER_AREA)
        flow = cv2.calcOpticalFlowFarneback(a, b, None, 0.5, 4, 15, 4, 7, 1.5, 0)
        flow = cv2.GaussianBlur(flow, (0, 0), FLOW_SIGMA)
        return cv2.resize(flow, (self.grid.nx, self.grid.ny), interpolation=cv2.INTER_LINEAR) * (FLOW_DOWN / dt_min)

    def forecast(self, ref):
        """Forecast shares {valid ts: grid} for the served variant and the shadows, from the k fields up to ref."""
        with self.lock:
            k1 = self.k.get(ref)
            prev = next(((self.k[ref - m * 60_000], m) for m in (30, 20, 40, 10) if ref - m * 60_000 in self.k), None)
            obs = self.obs.get(ref)
        if k1 is None:
            return None
        k1 = k1.astype(np.float32)
        flow = self._motion(prev[0].astype(np.float32), k1, prev[1]) if prev else None
        base = fill_nan(k1)
        known = np.isfinite(k1).astype(np.float32)
        kf = getattr(self, "_kfog_ref", None)
        base_f = fill_nan(kf[1]) if kf is not None and kf[0] == ref else None  # shadow "metfog" (METAR_URL)
        ko = getattr(self, "_klow_ref", None)
        base_o = fill_nan(ko[1]) if ko is not None and ko[0] == ref else None  # low-sun shadow (low_f_shadow)
        lowf = f"lowf{int(round(10 * self.low_f_shadow)):02d}"
        out = {"sat": {}, "satold": {}, "sharp": {}, "kpers": {}, "pers": {}, "model": {}, "nwp": {}, "shblur": {},
               "model90": {}, "satnwp": {}, "satadv": {}, "metfog": {}, lowf: {}, "satkp": {}}
        disp = {}
        qm, skill = self._nwp_qm(ref, k1)  # the model's k -> the satellite's distribution at ref; weight factor
        for lead in FC_LEADS:
            ts = ref + lead * 60_000
            e = self.elev(ts)
            if flow is not None:
                mx, my = self.grid.gx - flow[..., 0] * lead, self.grid.gy - flow[..., 1] * lead
                kk = cv2.remap(base, mx, my, cv2.INTER_LINEAR, borderMode=cv2.BORDER_REPLICATE)
                kn = cv2.remap(known, mx, my, cv2.INTER_LINEAR, borderMode=cv2.BORDER_REPLICATE)
            else:
                kk, kn = base.copy(), known
            kold = ksharp = None
            if lead in LEADS:  # shadow "sharp": moved, never blurred (15.11.2026, user: "Schaefchenwolken can stay all
                # day, but our forecast blurs small clouds away" - does keeping the pattern score better at the stations,
                # above all in broken cloud? score_sun.py sat sharp ... --broken)
                ksharp = np.where(kn > 0.5, kk, np.nan).astype(np.float32)
            if lead in LEADS:  # the old blur, for the shadow "satold"
                kold = np.where(kn > 0.5, cv2.GaussianBlur(kk, (0, 0), BLUR * math.sqrt(lead / 10)), np.nan).astype(np.float32)
            sig = max(0.3, BLUR * math.sqrt(lead / 10) * min(1.0, lead / SHARP_MIN))
            kk = cv2.GaussianBlur(kk, (0, 0), sig)
            kk = np.where(kn > 0.5, kk, np.nan).astype(np.float32)
            s = share_from_k(kk, e, self.low_f)
            out["sat"][ts] = s
            disp[ts] = self._display(kk, e)
            if SUNNY_NWP and lead in LEADS:
                out["satadv"][ts] = s  # the plain advection, scored while the blend is served
            w = nwp_weight(lead) * skill
            mq = None
            if qm is not None and (lead in LEADS or (SUNNY_NWP and w > 0)):
                mk = self._model_k(ts) if w > 0 else None
                mq = qm(mk) if mk is not None else None
                if w <= 0:
                    out["satnwp"][ts] = s
                elif mq is not None:
                    ok = np.isfinite(mq) & np.isfinite(s)
                    sn = np.where(ok, (1 - w) * s + w * share_from_k(mq, e, self.low_f), s).astype(np.float32)
                    if lead in LEADS:
                        out["satnwp"][ts] = sn
                    if SUNNY_NWP:  # served: the share as blended, the map from the k blended alike
                        out["sat"][ts] = sn
                        disp[ts] = self._display(np.where(ok, (1 - w) * kk + w * mq, kk).astype(np.float32), e)
            if lead in LEADS:  # shadows only where they are scored
                # shadow "satkp" (22.11.2026): as served, but the satellite part mixes the moved k with the k that stays
                # where it is (share KP_W120 at +120, from KP_START) - in daytime showers the clouds grow and decay in
                # place instead of moving, and the stationary k beat the moved one at the German stations at +90/+120
                # on 21.11. (MAE .249 vs .257) and 22.11. (.146 vs .154); over the week before the motion won slightly
                a = KP_W120 * min(1.0, max(0.0, (lead - KP_START) / (120.0 - KP_START)))
                if a > 0:
                    ks = cv2.GaussianBlur(base, (0, 0), sig)
                    km = np.where(kn > 0.5, (1 - a) * kk + a * ks, np.nan).astype(np.float32)
                    skp = share_from_k(km, e, self.low_f)
                else:
                    skp = s
                if SUNNY_NWP and w > 0 and mq is not None:  # the served model blend on top, as for "sat"
                    okp = np.isfinite(mq) & np.isfinite(skp)
                    skp = np.where(okp, (1 - w) * skp + w * share_from_k(mq, e, self.low_f), skp).astype(np.float32)
                out["satkp"][ts] = skp
                if base_f is None and (self.area or {}).get("clm"):  # Europe, no fog reported: the plain advection
                    out["metfog"][ts] = s
                elif base_f is not None:
                    kf_ = cv2.remap(base_f, mx, my, cv2.INTER_LINEAR, borderMode=cv2.BORDER_REPLICATE) if flow is not None else base_f
                    kf_ = np.where(kn > 0.5, cv2.GaussianBlur(kf_, (0, 0), sig), np.nan).astype(np.float32)
                    out["metfog"][ts] = share_from_k(kf_, e, self.low_f)
                if base_o is not None:  # the plain advection with the other low-sun factor (Germany: as served before v1.79)
                    ko_ = cv2.remap(base_o, mx, my, cv2.INTER_LINEAR, borderMode=cv2.BORDER_REPLICATE) if flow is not None else base_o
                    ko_ = np.where(kn > 0.5, cv2.GaussianBlur(ko_, (0, 0), sig), np.nan).astype(np.float32)
                    out[lowf][ts] = share_from_k(ko_, e, self.low_f_shadow)
                out["satold"][ts] = share_from_k(kold, e, self.low_f)
                out["sharp"][ts] = share_from_k(ksharp, e, self.low_f)
                # shadow "shblur" (16.11.2026): the served "sat" blurs k and then turns it into a sun share - the share
                # is not linear in k, so broken fields lose sun with the lead (Germany grid sunny area x0.90 at +120,
                # station bias -0.07). Here the SHARE is blurred instead: the local sun fraction stays as moved.
                sh = out["sharp"][ts]
                out["shblur"][ts] = np.where(np.isfinite(sh), cv2.GaussianBlur(fill_nan(sh), (0, 0), sig), np.nan).astype(np.float32)
                out["kpers"][ts] = share_from_k(k1, e, self.low_f)
                if obs is not None:
                    out["pers"][ts] = obs.astype(np.float32)
                m = self.model.field(ts) if self.model else None
                if m is not None:
                    m = np.where(e >= SUN_ELEV, m, 0.0).astype(np.float32)
                    w = 1.0 if lead <= MODEL_START else 1.0 - min(1.0, (lead - MODEL_START) / (MODEL_END - MODEL_START)) * (1 - MODEL_END_W)
                    out["model"][ts] = np.where(np.isfinite(s), w * s + (1 - w) * m, np.nan).astype(np.float32)
                    out["nwp"][ts] = np.where(np.isfinite(s), m, np.nan).astype(np.float32)
                    # shadow "model90" (16.11.2026): "model" fixed the stations at +120 but was x1.32 too sunny on the
                    # grid - the model only from +90, weight 0.25 at +120
                    w9 = 1.0 - max(0.0, min(1.0, (lead - 90) / 30)) * 0.25
                    out["model90"][ts] = np.where(np.isfinite(s), w9 * s + (1 - w9) * m, np.nan).astype(np.float32)
        return out, disp

    def _model_k(self, ts):
        """The weather model's clear-sky index for the satellite slot ts on the sunny grid (NaN at night), or None."""
        d = self.model.field(ts + NWP_LAG_MS, "dni") if self.model else None
        if d is None:
            return None
        e = self.elev(ts + NWP_LAG_MS)
        return np.where(e >= SUN_ELEV, np.clip(d / clear_dni(e, self.low_f), 0, K_MAX), np.nan).astype(np.float32)

    def _nwp_qm(self, ref, k1):
        """(quantile mapping of the model's k onto the satellite's k at ref over the shown area's daylight pixels, the
        model's weight factor from its error at ref), or (None, 0) without a usable model."""
        if not self.model or self.model.dni is None or not np.isfinite(self.model.dni).any():
            return None, 0.0
        m0 = self._model_k(ref)
        if m0 is None:
            return None, 0.0
        e0 = self.elev(ref)
        msk = np.isfinite(m0) & np.isfinite(k1) & (e0 >= DAY_ELEV) & (self.inside() > 0.5)
        qm = quantile_map(m0[msk], k1[msk])
        if qm is None:
            return None, 0.0
        s0, a0 = share_from_k(qm(m0), e0, self.low_f), share_from_k(k1, e0, self.low_f)
        sel = np.isfinite(s0) & np.isfinite(a0) & (e0 >= SCORE_ELEV) & (self.inside() > 0.5)
        if sel.sum() < 500:
            return None, 0.0
        err = float(np.abs(s0[sel] - a0[sel]).mean())
        return qm, float(np.clip(1.0 - (err - NWP_SKILL_A) / NWP_SKILL_B, 0.0, 1.0))

    # ---------- update ----------
    def _jump(self, ts, share, back=(1, 2)):
        """Mean daytime |share - an earlier image's share| inside the shown area (the first of `back` steps that exists),
        or None without such an image. Negative `back` steps look at later images."""
        with self.lock:
            prev = next((self.obs[ts - b * STEP_MS] for b in back if ts - b * STEP_MS in self.obs), None)
        if prev is None:
            return None
        m = (self.inside() > 0.5) & (self.elev(ts) >= SCORE_ELEV) & np.isfinite(share) & np.isfinite(prev.astype(np.float32))
        return float(np.abs(share[m] - prev.astype(np.float32)[m]).mean()) if m.sum() >= 500 else None

    def _spike(self, ts, share):
        """True when the image at ts differs by more than JUMP_MAX from the images before and after it (each one that
        exists, at least one): a bad SIDhr file in the history. 26.11.2026: DWD's 08:30Z file (0.8 MB instead of ~3 MB)
        had DNI > 120 W/m2 on 87 % of its area (24 % ten minutes before and after); add_slot skipped it, but the
        restart at 09:14 forgot that, took it again as a late slot - late slots were never checked - and the timeline
        showed a cloud-free Europe at 08:30 (Munich, Milan, Vienna: no sun, full sun, no sun)."""
        jumps = [j for j in (self._jump(ts, share, back=(1, 2)), self._jump(ts, share, back=(-1, -2))) if j is not None]
        return bool(jumps) and min(jumps) > JUMP_MAX

    def _drop_slot(self, bad, why):
        with self.lock:
            for d in (self.obs, self.k, self.disp):
                d.pop(bad, None)
        for d in (self.verify.pending, self.verify.spending, *(v.spending for v in self.extra_verify)):
            for key in [key for key in d if key[1] == bad]:
                del d[key]
        self.skipped.add(bad)
        self.log(f"sunny: slot {ts_stamp(bad)} dropped ({why})")

    def _drop_spikes(self):
        """Drops the images of the history that are spikes (_spike), e.g. one taken before the check existed."""
        with self.lock:
            ts_all = sorted(self.obs)
        for t in ts_all[1:-1]:
            with self.lock:
                g = self.obs.get(t)
            if g is not None and self._spike(t, g.astype(np.float32)):
                self._drop_slot(t, "its sun share differs from the images before and after it by more than "
                                   f"{JUMP_MAX} on average")

    def add_slot(self, ts, sid_data, cth_data=None):
        """Ingest one slot as the newest image, make the forecast, hand it to the verification. False: slot skipped."""
        k, share, disp = self.ingest(ts, sid_data, cth_data)
        self._kfog_ref, self._klow_ref = self._k_fog, self._k_low
        jump = self._jump(ts, share)
        if jump is not None and jump > JUMP_MAX and ts - STEP_MS in self.obs:
            before = self._jump(ts, share, back=(2,))
            if before is not None and before <= JUMP_MAX:  # the previous image was the odd one out: drop it
                self._drop_slot(ts - STEP_MS, f"sun share jumped by {jump:.2f} on average and back")
                jump = None
        if jump is not None and jump > JUMP_MAX and self.skips_in_row < JUMP_ACCEPT:
            self.skipped.add(ts)
            self.skips_in_row += 1
            self.log(f"sunny: slot {ts_stamp(ts)} skipped (sun share jumps by {jump:.2f} on average)")
            return False
        self.skips_in_row = 0
        with self.lock:
            self.k[ts] = k.astype(np.float16)
            self.obs[ts] = share.astype(np.float16)
            self.disp[ts] = disp
            for t in [t for t in self.k if t < ts - K_KEEP_MS]:
                del self.k[t]
            for t in [t for t in self.obs if t < ts - OBS_KEEP_MS]:
                del self.obs[t]
                self.disp.pop(t, None)
            newest = ts >= (self.ref or 0)
        if not newest:
            return True
        res = self.forecast(ts)
        if res is None:
            return True
        variants, disp = res
        with self.lock:
            self.fc = {t: g.astype(np.float16) for t, g in variants["sat"].items()}
            for t, d in disp.items():
                if t not in self.obs:
                    self.disp[t] = d
            for t in [t for t in self.disp if t not in self.obs and t not in self.fc]:
                del self.disp[t]
            self.ref = ts
            self.disp_snaps[ts] = dict(self.disp)
            while len(self.disp_snaps) > KEEP_RUNS:
                self.disp_snaps.popitem(last=False)
        self.verify.take(ts, {n: v for n, v in variants.items() if v})
        for ev in self.extra_verify:
            try:
                ev.take(ts, {n: v for n, v in variants.items() if v})
            except Exception as e:
                self.log(f"sunny extra verify ({ev.prefix}) failed: {e}")
        threading.Thread(target=self.save_state, daemon=True).start()
        return True

    # ---------- warm start (14.11.2026: after every restart of the instance the clouds were missing for minutes -
    # the first run needs the last hour of satellite slots again; user: "Wolken erscheinen erst deutlich später") ----
    def _state_path(self):
        d = os.path.join(os.environ.get("VERIFY_LOG_DIR", os.path.dirname(os.path.abspath(__file__))), "state")
        return d, os.path.join(d, f"sunny{(self.area or {}).get('prefix', '') or 'de'}.npz")

    def save_state(self):
        if self.shift_ms or self.frozen or not self.ref or not hasattr(self, "_save_lock"):
            return
        if not self._save_lock.acquire(blocking=False):
            return
        try:
            with self.lock:
                parts = {f"obs_{t}": g for t, g in self.obs.items()}
                parts.update({f"fc_{t}": g for t, g in self.fc.items()})
                parts.update({f"disp_{t}": g for t, g in self.disp.items()})
                parts.update({f"k_{t}": g for t, g in self.k.items()})
                ref = self.ref
            d, path = self._state_path()
            os.makedirs(d, exist_ok=True)
            tmp = path[:-4] + ".tmp.npz"
            np.savez_compressed(tmp, ref=np.array([ref], np.int64), skipped=np.array(sorted(self.skipped), np.int64), **parts)
            os.replace(tmp, path)
        except Exception as e:
            self.log(f"sunny: state not saved: {e}")
        finally:
            self._save_lock.release()

    def load_state(self):
        """Continue with the run from before a restart when it is at most 30 min old. True when taken over."""
        self._save_lock = threading.Lock()
        try:
            z = np.load(self._state_path()[1])
            ref = int(z["ref"][0])
            if time.time() * 1000 - ref > 30 * 60_000:
                return False
            got = {"obs": {}, "fc": {}, "disp": {}, "k": {}}
            for key in z.files:
                kind, _, t = key.partition("_")
                if kind in got:
                    got[kind][int(t)] = z[key]
            if ref not in got["obs"] or not got["fc"] or got["obs"][ref].shape != (self.grid.ny, self.grid.nx):
                return False  # missing, or saved on another grid (the area grew)
            with self.lock:
                self.obs, self.fc, self.disp, self.k, self.ref = got["obs"], got["fc"], got["disp"], got["k"], ref
                self.disp_snaps[ref] = dict(self.disp)
            # the slots skipped as implausible stay skipped (before 26.11.2026 a restart forgot them, see _spike)
            self.skipped |= {int(t) for t in z["skipped"]} if "skipped" in z.files else set()
            self._drop_spikes()
            self.log(f"sunny: continuing with the run of {ts_stamp(ref)} from before the restart "
                     f"({len(got['obs'])} images, {len(got['fc'])} forecast steps)")
            return True
        except FileNotFoundError:
            return False
        except Exception as e:
            self.log(f"sunny: state not read: {e}")
            return False

    def _listing(self, url, pattern):
        text = self.fetch(url, timeout=60).decode("latin-1")
        return {stamp_ts(s): name for name, s in ((m.group(0), m.group(1)) for m in re.finditer(pattern, text))}

    def _wait_cth(self, t, e):
        """True: leave slot t for a later round because its own CTH file is not out yet. At night the SIDhr file comes
        ~6 min before the CTHin file of the same slot (20.11.2026: SIDhr 18:00 at 18:16:33, CTHin 18:00 at 18:22:16):
        taken at once, the slot was paired with the CTH of 10 min before and the map lagged one IR image all night.
        Waited for only while at least CTH_WAIT_AREA of the shown area is below DAY_ELEV (by day the DNI decides and
        nothing is held up), and at most CTH_WAIT_S after the SIDhr file was first seen (server-v1.79)."""
        ins = self.inside() > 0.5
        if not ins.any() or float((e[ins] < DAY_ELEV).mean()) < CTH_WAIT_AREA:
            return False
        return time.time() - self._sid_seen.setdefault(t, time.time()) < CTH_WAIT_S

    def update(self):
        if self.frozen:
            return 0
        sids = self._listing(SID_URL, r"SIDhr(\d{12})EUv4\.nc\.bz2")
        if not sids:
            return 0
        newest = max(sids)
        self.skipped = {t for t in self.skipped if t > newest - OBS_KEEP_MS}
        # new slots, plus late ones up to K_KEEP_MS before the newest image (not the whole history after a start)
        # (a slot made from the cloud tops alone, IR_FALLBACK, is taken again once its SIDhr file is there)
        want = [t for t in sorted(sids) if t > newest - OBS_KEEP_MS and (t not in self.obs or t in self.ir_only)
                and t not in self.skipped and t > (self.ref or 0) - K_KEEP_MS]
        if not want:
            return self._ir_fallback(newest) if IR_FALLBACK else 0
        if self.ref is None:  # first start: the last hour (the app's history and the motion), not all 3 h
            want = want[-7:]
        cths = None
        n = 0
        self._sid_seen = {t: s for t, s in self._sid_seen.items() if t > newest - OBS_KEEP_MS}
        for t in want:
            e = self.elev(t)
            cth = None
            if float(e.min()) < DAY_ELEV:  # IR needed somewhere (low sun, night)
                try:
                    cths = cths if cths is not None else self._listing(CTH_URL, r"CTHin(\d{12})nEU\.nc\.bz2")
                    if t not in cths and self._wait_cth(t, e):
                        break  # this slot (and the newer ones) in a later round
                    near = [c for c in cths if abs(c - t) <= 10 * 60_000]
                    if near:
                        c = min(near, key=lambda c: abs(c - t))
                        cth = self.fetch(CTH_URL + cths[c], timeout=120)
                except Exception as ex:
                    self.log(f"sunny: CTH failed: {ex}")
            data = self.fetch(SID_URL + sids[t], timeout=120)
            if t < (self.ref or 0):  # an older slot that arrived late: history only, no forecast from it
                k, share, disp = self.ingest(t, data, cth)
                if self._spike(t, share):
                    self.skipped.add(t)
                    self.log(f"sunny: late slot {ts_stamp(t)} skipped (sun share differs from its neighbours by "
                             f"more than {JUMP_MAX} on average)")
                    continue
                with self.lock:
                    self.obs[t], self.disp[t] = share.astype(np.float16), disp
                    if t >= self.ref - K_KEEP_MS:
                        self.k[t] = k.astype(np.float16)
            else:
                self.add_slot(t, data, cth)
            self.ir_only.discard(t)  # the DNI slot replaces one made from the cloud tops alone
            n += 1
        if IR_FALLBACK:
            n += self._ir_fallback(newest)
        return n

    def _ir_fallback(self, newest_sid):
        """DWD's SIDhr files late (24.11.2026: none after 07:00Z for hours) while the CTH files keep coming: the slots
        after the newest SIDhr one are made without the DNI (model by day, IR cloud tops at low sun / night, see
        IR_FALLBACK), so the map and the forecast go on instead of freezing. Returns the number of slots added."""
        now = time.time() * 1000
        if now - newest_sid < IR_FALLBACK_MS + 20 * 60_000:  # a SIDhr file normally comes ~16 min after its slot
            return 0
        cths = self._listing(CTH_URL, r"CTHin(\d{12})nEU\.nc\.bz2")
        want = [t for t in sorted(cths) if t > newest_sid and t not in self.obs and t not in self.skipped
                and t > (self.ref or 0) and t > max(cths) - OBS_KEEP_MS]
        if not cths or max(cths) - newest_sid < IR_FALLBACK_MS or not want:
            return 0
        n = 0
        for t in want[-7:]:
            cth = self.fetch(CTH_URL + cths[t], timeout=120)
            self.ir_only.add(t)
            if self.add_slot(t, None, cth):
                n += 1
            else:
                self.ir_only.discard(t)
        self.ir_only = {t for t in self.ir_only if t > now - OBS_KEEP_MS}
        if n:
            self.log(f"sunny: DWD's SIDhr files late (newest {ts_stamp(newest_sid)}), {n} slot(s) up to "
                     f"{ts_stamp(want[-1])} without the DNI (model by day, IR cloud tops at low sun)")
        return n

    def verify_step(self):
        if not self.ref:
            return 0, 0
        g, s = self.verify.step(self.ref)
        for ev in self.extra_verify:
            try:
                s += ev.step(self.ref)[1]
            except Exception as e:
                self.log(f"sunny extra verify ({ev.prefix}) failed: {e}")
        return g, s

    def clm_loop(self):
        """EUMETSAT's cloud mask for the area beyond DWD's data, fetched in the background every 10 minutes - never in
        the slot loop: on 15.11.2026 EUMETView answered HTTP 500 after 20 s, and fetching it per slot (up to four
        tries) held sunny for all of Europe up for 16 minutes per update."""
        while True:
            try:
                now = int(time.time() * 1000)
                c = clm_cloud(self.fetch, now)
                if c is not None:
                    grid = self.grid.regrid(c[0], c[1], c[2], sigma=(0.7, 0.7))
                    self._clm_cache = {"now": (now // 900_000 * 900_000 - 15 * 60_000, grid)}
            except Exception as e:
                self.log(f"sunny: cloud mask failed: {e}")
            time.sleep(600)

    def loop(self):
        self.load_state()
        if (self.area or {}).get("clm"):
            threading.Thread(target=self.clm_loop, daemon=True).start()
            threading.Thread(target=self.metar_loop, daemon=True).start()
        if self.stations:
            threading.Thread(target=self.stations.loop, daemon=True).start()
        if self.model:
            threading.Thread(target=self.model_loop, daemon=True).start()
        while True:
            started = time.time()
            try:
                n = self.update()
                if n:
                    self.cycle_s = round(time.time() - started, 1)
                    self.log(f"[{datetime.now():%H:%M:%S}] sunny: +{n} images, latest "
                             f"{datetime.fromtimestamp(self.ref / 1000, timezone.utc):%H:%M}Z, {self.cycle_s}s")
            except Exception as e:  # keep serving the last good data
                self.log(f"sunny update failed: {e}")
            try:
                g, s = self.verify_step()
                if g or s:
                    self.log(f"[{datetime.now():%H:%M:%S}] sunny verify: {g} grid, {s} station lines")
            except Exception as e:
                self.log(f"sunny verify failed: {e}")
            time.sleep(POLL_S)

    def model_loop(self):
        while True:
            try:
                n = self.model.refresh(self.inside())
                if n:
                    self.log(f"[{datetime.now():%H:%M:%S}] sunny: model sunshine for {n} points")
            except Exception as e:
                self.log(f"sunny model failed: {e}")
            time.sleep(60)

    # ---------- serving (same API as clouds.CloudNowcast) ----------
    def frames(self):
        with self.lock:
            return ([{"ts": t, "fc": False} for t in sorted(self.obs)] +
                    [{"ts": t, "fc": True} for t in sorted(self.fc) if t not in self.obs]), self.ref

    def health(self):
        frames, ref = self.frames()
        age = round((time.time() * 1000 - ref) / 60_000) if ref else None
        return {"frames": len(frames), "obs": sum(not f["fc"] for f in frames), "ref": ref, "age_min": age,
                "cycle_s": self.cycle_s, "stations": len(self.stations.values) if self.stations else 0,
                "model": bool(self.model and self.model.share is not None), "replay": bool(self.shift_ms),
                "skipped": len(self.skipped), "ir_only": len(self.ir_only)}

    def has_run(self, run):
        with self.lock:
            return run in self.disp_snaps

    def _frame(self, ts, run):
        with self.lock:
            g = self.disp_snaps.get(run, {}).get(ts)
            return g if g is not None else self.disp.get(ts)

    SMOOTH_CELLS = 0.8  # Gaussian sigma in grid cells for the map tiles (_grey_alpha)

    def _grey_alpha(self, g, bbox, w, h, col_row=None):
        minx, miny, maxx, maxy = bbox
        if col_row is None:
            X, Y = np.meshgrid(minx + (np.arange(w) + 0.5) / w * (maxx - minx), maxy - (np.arange(h) + 0.5) / h * (maxy - miny))
            col_row = tuple(v.astype(np.float32) for v in self.grid.to_pixels(X, Y))
        col, row = col_row
        # server-v2.08: smooth upsampling - the grid smoothed by SMOOTH_CELLS, then bicubic. Bilinear made every cloud
        # outline a staircase of grid cells with straight sides and right-angled corners when zoomed in (user,
        # 26.11.2026: "eckige Kanten", 4 km cells at 50 N: ~40 px each at zoom 10). Only the crop the tile needs is
        # smoothed; its margin (> 3 sigma) keeps neighbouring tiles identical along their common edge.
        pad = int(3 * self.SMOOTH_CELLS) + 3
        c0, r0 = max(int(np.floor(col.min())) - pad, 0), max(int(np.floor(row.min())) - pad, 0)
        c1, r1 = min(int(np.ceil(col.max())) + pad + 1, g.shape[1]), min(int(np.ceil(row.max())) + pad + 1, g.shape[0])
        if c0 >= c1 or r0 >= r1:  # the tile lies outside the grid
            z = np.zeros(col.shape, np.uint8)
            return z, z, col_row
        sub = g[r0:r1, c0:c1]
        ok = (sub < 255).astype(np.float32)
        blur = lambda a: cv2.GaussianBlur(a, (0, 0), self.SMOOTH_CELLS, borderType=cv2.BORDER_CONSTANT)
        vs, ks = blur(np.where(sub < 255, sub, 0).astype(np.float32)), blur(ok)
        cc, rr = col - np.float32(c0), row - np.float32(r0)
        v = cv2.remap(vs, cc, rr, cv2.INTER_CUBIC, borderMode=cv2.BORDER_CONSTANT, borderValue=0)
        kn = np.clip(cv2.remap(ks, cc, rr, cv2.INTER_CUBIC, borderMode=cv2.BORDER_CONSTANT, borderValue=0), 0, 1)
        ins = cv2.remap(self.inside(), col, row, cv2.INTER_LINEAR, borderMode=cv2.BORDER_CONSTANT, borderValue=0)
        grey = np.clip(v / np.maximum(kn, 1e-3), 0, 255).astype(np.uint8)
        alpha = ((ins > 0.5) & (kn > 0.5)).astype(np.uint8) * 255
        return grey, alpha, col_row

    def image(self, ts, bbox, w, h, run=None, fmt="png"):
        """Cloud value as a grey + alpha PNG or lossless WebP (fmt, tileimg.py; alpha 0 outside the shown area or where
        unknown); None: unknown frame. run: the run a tile key names; its frames while it is kept, else (or for a frame
        it lacks) the current ones."""
        g = self._frame(ts, run)
        if g is None:
            return None
        grey, alpha, _ = self._grey_alpha(g, bbox, w, h)
        return tileimg.encode(Image.fromarray(np.dstack([grey, alpha]), "LA"), fmt, "cloud", 3)

    ATLAS_COLS = 8

    def image_atlas(self, t0, n, bbox, w, h, run=None, fmt="png", margin=0):
        """The frames t0, t0 + 10 min, ... (n of them) of one map tile in ONE grey + alpha PNG, ATLAS_COLS cells of
        w x h per row (a frame the run lacks stays transparent). The app loads one of these per tile position for the
        whole sunny timeline instead of one tile per time step (15.11.2026: ~250 requests, clouds came in late).
        margin (server-v2.02): each cell is the tile plus this many pixels of its neighbours on every side
        (w + 2 margin wide), so the app can smooth and interpolate across tile edges: without them the smoothed
        clouds showed the tile squares (user 25.11.2026: "a big grey area is made of squares with hard edges")."""
        if margin:
            minx, miny, maxx, maxy = bbox
            dx, dy = (maxx - minx) / w * margin, (maxy - miny) / h * margin
            bbox, w, h = (minx - dx, miny - dy, maxx + dx, maxy + dy), w + 2 * margin, h + 2 * margin
        rows = -(-n // self.ATLAS_COLS)  # only as many columns as the frames need (server-v1.69; cell i stays at
        out = np.zeros((rows * h, min(n, self.ATLAS_COLS) * w, 2), np.uint8)  # (i // ATLAS_COLS, i % ATLAS_COLS))
        col_row = None
        for i in range(n):
            g = self._frame(t0 + i * STEP_MS, run)
            if g is None:
                continue
            grey, alpha, col_row = self._grey_alpha(g, bbox, w, h, col_row)
            r, c = divmod(i, self.ATLAS_COLS)
            out[r * h:(r + 1) * h, c * w:(c + 1) * w, 0], out[r * h:(r + 1) * h, c * w:(c + 1) * w, 1] = grey, alpha
        return tileimg.encode(Image.fromarray(out, "LA"), fmt, "cloud", 3)

    def point(self, lat, lon):
        """Series at a place (3 x 3 mean, ~6 km), or [] outside the shown area."""
        col, row = (int(round(float(v))) for v in self.grid.to_pixels(*merc(lon, lat)))
        if not (1 <= col < self.grid.nx - 1 and 1 <= row < self.grid.ny - 1) or self.inside()[row, col] <= 0.5:
            return []
        frames, _ = self.frames()
        out = []
        with self.lock:
            for f in frames:
                g = self.obs.get(f["ts"]) if not f["fc"] else self.fc.get(f["ts"])
                d = self.disp.get(f["ts"])
                if g is None or d is None:
                    continue
                c = d[row - 1:row + 2, col - 1:col + 2]
                c = c[c < 255]
                s = g[row - 1:row + 2, col - 1:col + 2].astype(np.float32)
                if not c.size:
                    continue
                out.append({"ts": f["ts"], "fc": f["fc"], "cloud": round(float(c.mean()) / 254, 3),
                            "sun": round(float(np.nanmean(s)), 3) if np.isfinite(s).any() else 0.0})
        return out

    # ---------- development ----------
    def dev_replay(self, at, folder):
        """DEV only: replay archived files (folder with SIDhr / CTHin files) up to the UTC slot `at` as if it were now:
        frames are relabelled to the current time, the sun geometry stays at the archived time."""
        at_ts = stamp_ts(at)
        now_slot = int(time.time() * 1000) // STEP_MS * STEP_MS - 20 * 60_000
        shift = now_slot - at_ts
        self.frozen = True
        with self.lock:
            self.obs, self.fc, self.disp, self.k, self.ref = {}, {}, {}, {}, None
            self.shift_ms = shift
        n = 0
        for t in range(at_ts - 18 * STEP_MS, at_ts + 1, STEP_MS):
            fp = os.path.join(folder, f"SIDhr{ts_stamp(t)}EUv4.nc.bz2")
            if not os.path.exists(fp):
                continue
            cp = os.path.join(folder, f"CTHin{ts_stamp(t)}nEU.nc.bz2")
            cth = open(cp, "rb").read() if os.path.exists(cp) else None
            self.add_slot(t + shift, open(fp, "rb").read(), cth)
            n += 1
        self.verify.pending.clear()
        self.verify.spending.clear()
        return n

    def dev_reset(self):
        with self.lock:
            self.obs, self.fc, self.disp, self.k, self.ref = {}, {}, {}, {}, None
            self.shift_ms = 0
        self.frozen = False
