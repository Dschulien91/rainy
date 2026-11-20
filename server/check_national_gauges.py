"""Is our own Europe picture (OPERA + national layers) closer to the rain gauges than OPERA alone?

The third instance (EU_NATIONAL=1) notes for every image it composes, at every gauge of the lists below, what OPERA
alone showed there and what the composed picture shows (5 x 5 cells around the gauge, mm/h):
verify_log/eun/natobs/<ts>.npz with "opera" and "nat" in the order of natobs/stations.json (record(), called by
rainy_server.EuropeNational.load), and "who": which layers count at the gauge in that image (bit k = layer k of
"layers"), and "each": what every layer alone shows at the gauge. This script reads the gauges for those times and
prints both pictures against them - per country (10-minute pairs and sums over full hours), and per country and
combination of layers, e.g. "it" against "it+ch", there with every layer on its own ("only ch") - so a layer that
hurts outside its own country shows up, as Italy's did in Carinthia on 13.11.2026:

  docker exec -w /app deploy-api-eun-1 python check_national_gauges.py [--hours 24]

Gauges (independent of all radar data), each list open data without an account:
  de  DWD 10-minute gauges (check_rain_gauges.stations(), one per ~25 km cell)
  at  GeoSphere Austria TAWES 10-minute gauges (CC BY) - the Swiss and German layers reach into Austria
  ch  MeteoSwiss automatic stations and precipitation stations, 10-minute sums (CC BY). Only the file "t_now" is
      read (today since 00:00 UTC; the next one holds the whole year), so Swiss pairs exist for today's images only.
  it  ARPA Lombardia rain gauges (dati.lombardia.it, 10-minute or 1-minute sums, times in UTC+1)
  iv  ARPA Veneto rain gauges (api.arpa.veneto.it, 5-minute sums of the last ~2.5 days per gauge, times in UTC+1)
  ie  ARPAE Emilia-Romagna (dati-simc.arpae.it, one file with the last hours of all stations, CC BY), 15-MINUTE
      sums: spread evenly over their three 5-minute bins and summed into the 10 minutes ending at :00, :10, ...
      (exact over full hours, slightly smoothed in the 10-minute pairs)
      More Italian regions are on the list (EVALUATION.md 8b).
  cz  CHMI open data, 10-minute sums (element SRA10M) of the automatic stations, one file per station and day
  se  SMHI metobs (opendata-download-metobs.smhi.se, CC BY 4.0), parameter 14 = 15-MINUTE sums (spread as "ie"),
      ~125 stations; published hourly, so 30-90 min behind. One bulk file for the last hour, else per station
      "latest-day" (4 workers).
  fi  FMI open WFS (opendata.fmi.fi, CC BY 4.0), ri_10min = mean precipitation intensity of the 10 minutes ending at
      the time (mm/h, / 6 = mm), ~115 weather stations in one request, a few minutes behind.
  en  England, Environment Agency flood-monitoring API (environment.data.gov.uk, Open Government Licence v3), ~950
      tipping-bucket gauges, 15-MINUTE totals (spread as "ie"), 15-60 min behind (many gauges send hourly). Read
      newest first, 10000 readings per page (~2.7 h of all gauges), pages in parallel (4 at most).
  hu  HungaroMet open data (odp.met.hu, free with the source named: HungaroMet ODP terms of use), automatic
      stations, 10-minute sums (r), one small zipped CSV per 10 minutes (the last ~24 h kept), ~260 gauges, ~20 min
      behind.
  pl  Poland, IMGW telemetry: only the newest 10-minute sum per gauge -> polled (poll_latest_networks)
  pt  Portugal, IPMA: every station's last 24 h, hourly amounts
  sy  European SYNOP stations, hourly amounts
  Added 16.11.2026 (all keyless; time stamps are the END of the interval unless noted, converted to UTC):
  sc  Scotland, SEPA KiWIS (timeseries.sepa.org.uk, Open Government Licence), ~300 gauges, 15-MINUTE totals (spread
      as "ie"), stamps in UTC, ~15 min behind; one range request for all gauges (getTimeseriesValues by ts_path)
  be  Belgium, RMI AWS (opendata.meteo.be, open data), 14 stations, 10-minute sums, UTC, ~10 min behind
  bw  Belgium, Wallonia SPW hydrometrie KiWIS, ~90 gauges, 5-MINUTE sums (paired into 10 min, as "iv"), stamps with
      their offset (+01:00), ~10 min behind; one range request
  bf  Belgium, Flanders VMM waterinfo KiWIS, ~47 gauges, 15-minute totals (spread as "ie"), +01:00, ~15 min behind
  gl  Spain, MeteoGalicia (servizos.meteogalicia.gal), ~160 stations, 10-minute sums, UTC, ~10 min behind: only the
      newest value per station -> polled (poll_latest_networks)
  ct  Spain, Meteocat XEMA via the Catalan open data portal (Socrata nzvn-apee, Meteocat terms: free with the source
      named), ~185 stations, 30-MINUTE sums stamped at the START of the half hour in UTC (dataset description), ~30-60
      min behind; spread evenly over the three 10-minute steps (only the hourly tables are sharp)
  bz  Italy, South Tyrol (Province of Bolzano meteo API, CC0), 60 gauges, 5-minute sums, stamps "CET"/"CEST", ~10 min
      behind; one request per gauge
  tn  Italy, Trentino (dati.meteotrentino.it), ~100 gauges, 15-minute sums of yesterday and today in LOCAL time
      without zone (the offset is read from the newest stamp against the clock); one request per gauge
  si  Slovenia, ARSO automatic stations (meteo.arso.gov.si, one file of all stations' newest values), ~100 gauges,
      10-minute sums (some 30-minute), UTC -> polled (poll_latest_networks)
  dk  Denmark, DMI metObs API (opendataapi.dmi.dk, keyless since 2025, CC BY 4.0), ~105 gauges, precip_past10min,
      UTC, a few minutes behind; one range request
  Not used: Slovakia (SHMU's open server has 1-minute gauge data but no station positions), Norway, the Netherlands
  (identifying header or key), Liguria (OMIRL: the meaning and time base of its station values are not documented).
"Shadow" layers (national.Layer.SHADOW: loaded, noted here, not in the picture) get their own tables: where such a
candidate has data, OPERA alone and the own picture against "only <shadow>" - the evidence for promoting it.
New lists are appended to stations.json (the order of the older files stays valid; they are shorter).
Compared per image time T: the radar value at T against the gauge's mean rate over T-10 .. T+10 min (two 10-minute
sums, as in check_opera_gauges.py). Printed: the mean, the rain amount ratio radar/gauge and CSI / POD / FAR for
"wet" at 0.1, 0.3 and 1 mm/h (hours: 0.2, 1, 3 mm) - for OPERA alone and for the composed picture, on the same pairs.
"""
import glob
import io
import json
import os
import sys
import time
import urllib.parse
import zipfile
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime, timezone

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

LOG_DIR = os.environ.get("VERIFY_LOG_DIR", os.path.join(os.path.dirname(os.path.abspath(__file__)), "verify_log"))
OBS_DIR = os.path.join(LOG_DIR, "natobs")
FC_DIR = os.path.join(LOG_DIR, "fcgauge")   # forecasts at the gauges, per run (check_forecast_gauges.py)
FC_LEADS = (30, 60, 90, 120)                # minutes
KEEP_DAYS = 14
# the volume's root (the own-picture instance logs into <root>/eun): what both Europe instances share lives here
ROOT = os.path.dirname(LOG_DIR.rstrip("/")) if os.path.basename(LOG_DIR.rstrip("/")) == "eun" else LOG_DIR
TRUTH_DIR = os.path.join(ROOT, "gaugetruth")  # the gauges' 10-minute sums, one file per UTC day (archive_truth)
TRUTH_DAYS = 60
TAWES = "https://dataset.api.hub.geosphere.at/v1"
SWISS = "https://data.geo.admin.ch"
SWISS_SETS = {"smn": "ogd-smn", "pre": "ogd-smn-precip"}  # id prefix -> collection (ch.meteoschweiz.<...>)
LOMBARDY = "https://www.dati.lombardia.it/resource"
VENETO = "https://api.arpa.veneto.it/REST/v1"
CZECH = "https://opendata.chmi.cz/meteorology/climate/now"
EMILIA = "https://dati-simc.arpae.it/opendata/osservati/meteo/realtime/realtime.jsonl"
SWEDEN = "https://opendata-download-metobs.smhi.se/api/version/1.0/parameter/14"
FINLAND = ("https://opendata.fmi.fi/wfs?service=WFS&version=2.0.0&request=getFeature"
           "&storedquery_id=fmi::observations::weather::multipointcoverage&bbox=19,59,32,71"
           "&parameters=ri_10min&timestep=10&maxlocations=1000")
ENGLAND = "https://environment.data.gov.uk/flood-monitoring"
ENGLAND_MEASURE = "-rainfall-tipping_bucket_raingauge-t-15_min-mm"
HUNGARY = "https://odp.met.hu/weather/weather_reports/synoptic/hungary/10_minutes/csv"
POLAND = "https://danepubliczne.imgw.pl/api/data/meteo"   # IMGW telemetry: the newest 10-min sum per station only
PORTUGAL = "https://api.ipma.pt/open-data/observation/meteorology/stations"  # IPMA: every station's last 24 h, hourly
# KiWIS servers (base, datasource, time series path); their stamps are the ends of the intervals (checked 16.11.2026
# against the co-located RMI gauges: Uccle 5-min Wallonia vs RMI 10-min r = 0.999 at no shift, Flanders likewise)
SCOTLAND = ("https://timeseries.sepa.org.uk/KiWIS/KiWIS", 0, "1/*/RE/15m.Total")
WALLONIA = ("https://hydrometrie.wallonie.be/services/KiWIS/KiWIS", 0, "DGH/*/Precip/5m.CmdTotal.P")
FLANDERS = ("https://download.waterinfo.be/tsmdownload/KiWIS/KiWIS", 1, "AOW_PLUVIO/*/N/15m.Total")
BELGIUM = "https://opendata.meteo.be/service/aws/ows?service=WFS&version=2.0.0&request=GetFeature&outputFormat=csv"
GALICIA = "https://servizos.meteogalicia.gal/mgrss/observacion"
CATALONIA = "https://analisi.transparenciacatalunya.cat/resource"
SOUTH_TYROL = "https://geoservices.buergernetz.bz.it/services/meteo/v1"
TRENTINO = "https://dati.meteotrentino.it/service.asmx"
SLOVENIA = "https://meteo.arso.gov.si/uploads/probase/www/observ/surface/text/sl/observationAms_si_latest.xml"
DENMARK = "https://opendataapi.dmi.dk/v2/metObs/collections/observation/items?parameterId=precip_past10min"
NAMES = {"de": "Germany (DWD gauges)", "at": "Austria (TAWES gauges)", "ch": "Switzerland (MeteoSwiss gauges)",
         "it": "Italy, Lombardy (ARPA Lombardia gauges)", "iv": "Italy, Veneto (ARPAV gauges)",
         "cz": "Czechia (CHMI gauges)", "ie": "Italy, Emilia-Romagna (ARPAE gauges)",
         "se": "Sweden (SMHI gauges)", "fi": "Finland (FMI gauges)", "en": "England (Environment Agency gauges)",
         "hu": "Hungary (HungaroMet gauges)",
         "pl": "Poland (IMGW gauges)", "pt": "Portugal (IPMA stations, hourly)",
         "sy": "Europe, SYNOP stations (hourly amounts only; per country in check_forecast_gauges.py)",
         "sc": "Scotland (SEPA gauges)", "be": "Belgium (RMI stations)", "bw": "Belgium, Wallonia (SPW gauges)",
         "bf": "Belgium, Flanders (VMM gauges)", "gl": "Spain, Galicia (MeteoGalicia stations)",
         "ct": "Spain, Catalonia (Meteocat stations, 30-min)", "bz": "Italy, South Tyrol (Province of Bolzano gauges)",
         "tn": "Italy, Trentino (Meteotrentino gauges)", "si": "Slovenia (ARSO stations)", "dk": "Denmark (DMI gauges)"}


# ---------- gauge lists ----------
def _list_de(fetch):
    import check_rain_gauges as crg
    return [("de", sid, lat, lon) for sid, lat, lon in crg.stations()]


def _list_at(fetch):
    meta = json.loads(fetch(f"{TAWES}/station/current/tawes-v1-10min/metadata", timeout=60))
    return [("at", str(s["id"]), float(s["lat"]), float(s["lon"])) for s in meta.get("stations", []) if s.get("is_active")]


def _list_ch(fetch):
    out = []
    for prefix, coll in SWISS_SETS.items():
        url = f"{SWISS}/api/stac/v1/collections/ch.meteoschweiz.{coll}/items?limit=100"
        while url:
            d = json.loads(fetch(url, timeout=60))
            for f in d["features"]:
                lon, lat = f["geometry"]["coordinates"][:2]
                out.append(("ch", f"{prefix}:{f['id']}", float(lat), float(lon)))
            url = next((l["href"] for l in d.get("links", []) if l.get("rel") == "next"), None)
    return out


def _list_it(fetch):
    q = urllib.parse.urlencode({"tipologia": "Precipitazione", "storico": "N", "$limit": 5000, "$select": "idsensore,lat,lng"})
    return [("it", str(s["idsensore"]), float(s["lat"]), float(s["lng"]))
            for s in json.loads(fetch(f"{LOMBARDY}/nf78-nj6b.json?{q}", timeout=60)) if s.get("lat") and s.get("lng")]


def _list_iv(fetch):
    d = json.loads(fetch(f"{VENETO}/meteo_meteogrammi?rete=MGRAMMI&coordcd=23&orario=0", timeout=60))["data"]
    return [("iv", str(s["codseqst"]), float(s["latitudine"]), float(s["longitudine"])) for s in d]


def _czech_day(fetch, folder):
    import re
    return max(re.findall(r"-(\d{8})\.json", fetch(f"{CZECH}/{folder}/", timeout=60).decode("utf-8", "replace")))


def _list_cz(fetch):
    """Stations that measure SRA10M (meta2) and have a file today, with their position (meta1: WSI, .., lon, lat)."""
    import re
    day = _czech_day(fetch, "metadata")
    rows = lambda name: json.loads(fetch(f"{CZECH}/metadata/{name}-{day}.json", timeout=60))["data"]["data"]["values"]
    rain = {r[1] for r in rows("meta2") if r[2] == "SRA10M"}
    today = set(re.findall(r"10m-([0-9-]+)-\d{8}\.json", fetch(f"{CZECH}/data/", timeout=60).decode("utf-8", "replace")))
    pos = {r[0]: (float(r[4]), float(r[3])) for r in rows("meta1")  # the list also holds foreign SYNOP stations
           if r[0] in rain and r[0] in today and 12.0 < r[3] < 19.0 and 48.5 < r[4] < 51.1}
    return [("cz", sid, lat, lon) for sid, (lat, lon) in sorted(pos.items())]


def _emilia_rows(raw):
    """(id, lat, lon, end of the 15 minutes in ms, mm) for every 15-minute precipitation value in ARPAE's file
    (one JSON object per line and station time; B13011 with timerange [1, 0, 900] = sum of the 900 s before)."""
    for line in raw.splitlines():
        if b'"B13011"' not in line:
            continue
        try:
            d = json.loads(line)
        except ValueError:
            continue
        for part in d.get("data", []):
            if part.get("timerange") == [1, 0, 900] and "B13011" in part.get("vars", {}):
                v = part["vars"]["B13011"].get("v")
                if v is not None and v >= 0:
                    yield (f"{d['lon']}_{d['lat']}", d["lat"] / 1e5, d["lon"] / 1e5,
                           _ms(d["date"][:19], "%Y-%m-%dT%H:%M:%S"), float(v))


def _list_ie(fetch):
    pos = {sid: (lat, lon) for sid, lat, lon, _, _ in _emilia_rows(fetch(EMILIA, timeout=180))}
    return [("ie", sid, lat, lon) for sid, (lat, lon) in sorted(pos.items())]


def _list_se(fetch):
    """The stations of SMHI's bulk file that reported in the last week."""
    d = json.loads(fetch(f"{SWEDEN}/station-set/all/period/latest-hour/data.json", timeout=60))
    week = time.time() * 1000 - 7 * 86400_000
    return [("se", str(s["key"]), float(s["latitude"]), float(s["longitude"])) for s in d.get("station", [])
            if (s.get("to") or 0) > week and s.get("latitude") is not None]


def _fmi_rows(raw):
    """(fmisid, lat, lon, end of the 10 minutes in ms, mm) from FMI's multipointcoverage answer (positions
    "lat lon epoch" and values in the same order; the station points carry the fmisid)."""
    import re
    x = raw.decode("utf-8", "replace")
    sid = {(a, b): i for i, a, b in re.findall(r'gml:id="point-(\d+)".*?<gml:pos>\s*(\S+)\s+(\S+)\s*</gml:pos>', x, re.S)}
    pos = re.search(r"<gmlcov:positions>(.*?)</gmlcov:positions>", x, re.S)
    val = re.search(r"<gml:doubleOrNilReasonTupleList>(.*?)</gml:doubleOrNilReasonTupleList>", x, re.S)
    if not pos or not val:
        return
    pos, val = pos.group(1).split(), val.group(1).split()
    for i in range(min(len(pos) // 3, len(val))):
        lat, lon, t = pos[3 * i:3 * i + 3]
        try:
            v = float(val[i])
        except ValueError:
            continue
        if v == v and v >= 0 and (lat, lon) in sid:  # NaN = no value
            yield sid[(lat, lon)], float(lat), float(lon), int(t) * 1000, v / 6


def _fmi_get(since, until, fetch=None):
    iso = lambda ms: datetime.fromtimestamp(ms / 1000, timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")
    since = max(since, until - 160 * 3600_000)  # FMI answers at most 168 hours of 10-minute data
    return (fetch or _get)(f"{FINLAND}&starttime={iso(since)}&endtime={iso(until)}", timeout=120)


def _list_fi(fetch):
    now = int(time.time() * 1000)
    pos = {sid: (lat, lon) for sid, lat, lon, _, _ in _fmi_rows(_fmi_get(now - 3 * 3600_000, now, fetch))}
    return [("fi", sid, lat, lon) for sid, (lat, lon) in sorted(pos.items())]


def _list_en(fetch):
    """Environment Agency rainfall stations with a 15-minute tipping-bucket total (id = station reference)."""
    d = json.loads(fetch(f"{ENGLAND}/id/stations?parameter=rainfall&_limit=10000", timeout=120))
    out = {}
    for s in d.get("items", []):
        ms = s.get("measures") or []
        ms = ms if isinstance(ms, list) else [ms]
        ref = s.get("stationReference") or s.get("notation")
        if (ref and ref not in out and isinstance(s.get("lat"), float) and isinstance(s.get("long"), float)
                and any(str(m.get("@id", "")).endswith(f"/{ref}{ENGLAND_MEASURE}") for m in ms if isinstance(m, dict))):
            out[ref] = (s["lat"], s["long"])
    return [("en", ref, lat, lon) for ref, (lat, lon) in sorted(out.items())]


def _hungary_rows(raw):
    """(station, lat, lon, end of the 10 minutes in ms, mm) of one HungaroMet file (a zipped ";" CSV; Time = the
    end of the 10 minutes in UTC, r = 10-minute sum, -999 = no value)."""
    z = zipfile.ZipFile(io.BytesIO(raw))
    lines = z.read(z.namelist()[0]).decode("utf-8", "replace").splitlines()
    k = {h.strip(): i for i, h in enumerate(lines[0].split(";"))}
    for line in lines[1:]:
        q = [x.strip() for x in line.split(";")]
        try:
            v = float(q[k["r"]])
            if v >= 0:
                yield q[k["StationNumber"]], float(q[k["Latitude"]]), float(q[k["Longitude"]]), _ms(q[k["Time"]], "%Y%m%d%H%M"), v
        except (ValueError, IndexError, KeyError):
            continue


def _hungary_files(fetch, since, until):
    """The file names made from `since` until an hour after `until` (late reports come in later files)."""
    import re
    names = sorted(set(re.findall(r"HABP_10M_SYNOP_(\d{14})\.csv\.zip", fetch(f"{HUNGARY}/", timeout=60).decode("latin-1"))))
    return [f"HABP_10M_SYNOP_{n}.csv.zip" for n in names if since <= _ms(n[:12], "%Y%m%d%H%M") <= until + 3600_000]


def _list_hu(fetch):
    now = int(time.time() * 1000)
    pos = {}
    for name in _hungary_files(fetch, now - 3600_000, now)[-6:]:
        for sid, lat, lon, _, _ in _hungary_rows(fetch(f"{HUNGARY}/{name}", timeout=60)):
            pos[sid] = (lat, lon)
    return [("hu", sid, lat, lon) for sid, (lat, lon) in sorted(pos.items())]


def _list_pl(fetch):
    return [("pl", str(x["kod_stacji"]), float(x["lat"]), float(x["lon"])) for x in json.loads(fetch(POLAND, timeout=60))
            if x.get("opad_10min_data") and x.get("lat") and x.get("lon")]


def _list_pt(fetch):
    d = json.loads(fetch(f"{PORTUGAL}/stations.json", timeout=60))
    return [("pt", str(f["properties"]["idEstacao"]), float(f["geometry"]["coordinates"][1]),
             float(f["geometry"]["coordinates"][0])) for f in d if 36.5 <= f["geometry"]["coordinates"][1] <= 42.5]


_WRITTEN = {}  # network -> {(id, t, mm)} this process already archived (a poll repeats values until they change)


def _archive_latest(net, rows):
    """rows (id, end of the 10 minutes in ms, mm) -> <ROOT>/synop/<net>_<UTC day>.jsonl (one JSON line per value;
    the reader takes the last line of a gauge and time). synop.SynopFeed removes the files after its KEEP_DAYS."""
    seen = _WRITTEN.setdefault(net, set())
    now = time.time() * 1000
    old = now - 2 * 86400_000  # dead stations keep their last value for years: not kept
    if len(seen) > 300_000:
        seen -= {k for k in seen if k[1] < old}
    lines = {}
    for sid, t, mm in rows:
        if t % 600_000 == 0 and old <= t <= now + 600_000 and mm >= 0 and (sid, t, mm) not in seen:
            seen.add((sid, t, mm))
            lines.setdefault(time.strftime("%Y%m%d", time.gmtime(t / 1000)), []).append(
                json.dumps({"id": sid, "t": t, "mm": mm}))
    d = os.path.join(ROOT, "synop")
    os.makedirs(d, exist_ok=True)
    for day, ls in lines.items():
        with open(os.path.join(d, f"{net}_{day}.jsonl"), "a") as f:
            f.write("\n".join(sorted(ls)) + "\n")
    return sum(len(v) for v in lines.values())


def poll_poland(fetch):
    """IMGW serves only each gauge's newest 10-minute sum (times in UTC): polled every 5 minutes
    (poll_latest_networks) into <ROOT>/synop/pl_<day>.jsonl - gauges_pl reads those."""
    rows = []
    for x in json.loads(fetch(POLAND, timeout=60)):
        try:
            rows.append((str(x["kod_stacji"]), _ms(x["opad_10min_data"], "%Y-%m-%d %H:%M:%S"), float(x["opad_10min"])))
        except (KeyError, TypeError, ValueError):
            continue
    return _archive_latest("pl", rows)


def _galicia_latest(fetch):
    """MeteoGalicia's newest 10-minute rain sum per station (PP_SUM_1.5m; instanteLecturaUTC = the END of the 10
    minutes in UTC: on 16.11.2026 15:35 UTC stations already reported 15:30). Only values marked valid (code 1)."""
    raw = fetch(f"{GALICIA}/ultimos10minEstacionsMeteo.action?idParam=PP_SUM_1.5m", timeout=60)
    for x in json.loads(raw.decode("latin-1"))["listUltimos10min"]:
        for m in x.get("listaMedidas") or []:
            try:
                if m["codigoParametro"] == "PP_SUM_1.5m" and m.get("lnCodigoValidacion") == 1:
                    yield str(x["idEstacion"]), _ms(x["instanteLecturaUTC"][:19], "%Y-%m-%dT%H:%M:%S"), float(m["valor"])
            except (KeyError, TypeError, ValueError):
                continue


def poll_galicia(fetch):
    return _archive_latest("gl", list(_galicia_latest(fetch)))


def _list_gl(fetch):
    have = {sid for sid, _, _ in _galicia_latest(fetch)}
    d = json.loads(fetch(f"{GALICIA}/listaEstacionsMeteo.action", timeout=60).decode("latin-1"))
    return sorted(("gl", str(s["idEstacion"]), float(s["lat"]), float(s["lon"])) for s in d["listaEstacionsMeteo"]
                  if str(s["idEstacion"]) in have and s.get("lat") and s.get("lon"))


def _slovenia_rows(raw):
    """(id, lat, lon, start ms, end ms, mm) of ARSO's file of all automatic stations' newest values (validStart /
    validEnd in UTC, rr_val = the precipitation sum of that interval: 10 minutes, at some stations 30)."""
    import re
    x = raw.decode("utf-8", "replace")
    for block in re.findall(r"<metData>(.*?)</metData>", x, re.S):
        f = {k: v.strip() for k, v in re.findall(r"<(domain_meteosiId|domain_lat|domain_lon|validStart|validEnd|rr_val)>"
                                                 r"([^<]*)</\1>", block)}
        try:
            yield (f["domain_meteosiId"].strip("_"), float(f["domain_lat"]), float(f["domain_lon"]),
                   _ms(f["validStart"], "%d.%m.%Y %H:%M UTC"), _ms(f["validEnd"], "%d.%m.%Y %H:%M UTC"), float(f["rr_val"]))
        except (KeyError, ValueError):
            continue


_NEWEST = {}  # network -> the newest end time a poll found (ms)


def poll_slovenia(fetch):
    """The file (~0.8 MB) is fetched only when a newer 10 minutes should be out (13 min after the newest one seen);
    a 30-minute sum is spread evenly over its three 10-minute steps."""
    now = time.time() * 1000
    if now < _NEWEST.get("si", 0) + 13 * 60_000:
        return 0
    rows = []
    for sid, _, _, t0, t1, mm in _slovenia_rows(fetch(SLOVENIA, timeout=60)):
        n = (t1 - t0) // 600_000
        if t1 % 600_000 == 0 and n in (1, 3) and mm >= 0:
            rows += [(sid, t1 - k * 600_000, mm / n) for k in range(n)]
            _NEWEST["si"] = max(_NEWEST.get("si", 0), min(t1, now))
    return _archive_latest("si", rows)


def _list_si(fetch):
    return sorted({("si", sid, lat, lon) for sid, lat, lon, _, _, _ in _slovenia_rows(fetch(SLOVENIA, timeout=60))})


def poll_latest_networks(fetch, log=print):
    """The networks that serve only each gauge's newest value, kept in <ROOT>/synop/<net>_<day>.jsonl. Called every 5
    minutes by the OPERA instance; one request per network (Slovenia's larger file every ~10 minutes)."""
    n = {}
    for net, fn in (("pl", poll_poland), ("gl", poll_galicia), ("si", poll_slovenia)):
        try:
            n[net] = fn(fetch)
        except Exception as e:
            log(f"{net} gauges (latest values) failed: {e}")
    return n


# ---------- the KiWIS servers (Scotland, Wallonia, Flanders): one request for all gauges ----------
def _iso_ms(text):
    """ms of an ISO time with its zone ("Z", "+01:00"); without one it is taken as UTC."""
    d = datetime.fromisoformat(text.strip().replace("Z", "+00:00"))
    return int((d if d.tzinfo else d.replace(tzinfo=timezone.utc)).timestamp() * 1000)


def _utc_iso(ms):
    return datetime.fromtimestamp(ms / 1000, timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")


def _kiwis(src, request, fetch=None, **q):
    base, ds, path = src
    q = urllib.parse.urlencode({"service": "kisters", "type": "queryServices", "datasource": ds, "format": "json",
                                "request": request, "ts_path": path, **q}, safe="/*,:")
    return json.loads((fetch or _get)(f"{base}?{q}", timeout=180))


def _kiwis_list(net, src, fetch):
    """Every time series of the path with a value in the last week; id = its ts_id (the range request answers by it)."""
    week = time.time() * 1000 - 7 * 86400_000
    out = {}
    for s in _kiwis(src, "getTimeseriesValueLayer", fetch, metadata="true"):
        try:
            lat, lon = float(s["station_latitude"]), float(s["station_longitude"])
            if s.get("timestamp") and _iso_ms(s["timestamp"]) > week and lat and lon:
                out[str(s["ts_id"])] = (lat, lon)
        except (KeyError, TypeError, ValueError):
            continue
    return [(net, sid, lat, lon) for sid, (lat, lon) in sorted(out.items())]


def _kiwis_values(src, ids, since, until, label):
    """{ts_id: {end of the interval ms: mm}} of the listed series from `since` to `until` (one request)."""
    if not ids:
        return {}
    try:
        d = _kiwis(src, "getTimeseriesValues", None, **{"from": _utc_iso(since), "to": _utc_iso(until),
                                                       "returnfields": "Timestamp,Value"})
    except Exception as e:
        print(f"  ({label} gauges: {e})")
        return {}
    want, out = set(ids), {}
    for s in d:
        sid = str(s.get("ts_id"))
        if sid not in want:
            continue
        for row in s.get("data") or []:
            try:
                v = row[1]
                if isinstance(v, (int, float)) and v >= 0:
                    out.setdefault(sid, {})[_iso_ms(row[0])] = float(v)
            except (IndexError, TypeError, ValueError):
                continue
    return out


def _list_sc(fetch):
    return _kiwis_list("sc", SCOTLAND, fetch)


def _list_bw(fetch):
    return _kiwis_list("bw", WALLONIA, fetch)


def _list_bf(fetch):
    return _kiwis_list("bf", FLANDERS, fetch)


def _belgium(fetch, typename, cql=None, props=None):
    """Rows of RMI's WFS as CSV (the 10-minute layer must be filtered: unfiltered it answers 502)."""
    import csv
    url = f"{BELGIUM}&typenames={typename}"
    if props:
        url += f"&propertyName={props}"
    if cql:
        url += "&CQL_FILTER=" + urllib.parse.quote(cql)
    return list(csv.DictReader(io.StringIO((fetch or _get)(url, timeout=120).decode("utf-8", "replace"))))


def _list_be(fetch):
    """RMI's automatic stations in operation; the_geom is "POINT (lat lon)"."""
    import re
    out = []
    for r in _belgium(fetch, "aws:aws_station"):
        p = re.findall(r"-?\d+(?:\.\d+)?", r.get("the_geom", ""))
        if not r.get("date_end") and len(p) == 2:
            out.append(("be", r["code"], float(p[0]), float(p[1])))
    return sorted(out)


def _denmark(since, until, fetch=None):
    """(station, lat, lon, end of the 10 minutes ms, mm) of DMI's precip_past10min (observed = the end, UTC), Denmark only
    (the API also serves Greenland and the Faroes)."""
    url, n = f"{DENMARK}&datetime={_utc_iso(since)}/{_utc_iso(until)}&limit=300000", 0
    while url and n < 20:
        d = json.loads((fetch or _get)(url, timeout=180))
        feats = d.get("features", [])
        for f in feats:
            try:
                p, (lon, lat) = f["properties"], f["geometry"]["coordinates"][:2]
                if 54.4 < lat < 58 and 7.8 < lon < 15.4 and p["value"] is not None and p["value"] >= 0:
                    yield p["stationId"], float(lat), float(lon), _iso_ms(p["observed"]), float(p["value"])
            except (KeyError, TypeError, ValueError):
                continue
        url = next((l["href"] for l in d.get("links", []) if l.get("rel") == "next"), None) if len(feats) >= 300000 else None
        n += 1


def _list_dk(fetch):
    now = int(time.time() * 1000)
    pos = {sid: (lat, lon) for sid, lat, lon, _, _ in _denmark(now - 3 * 3600_000, now, fetch)}
    return [("dk", sid, lat, lon) for sid, (lat, lon) in sorted(pos.items())]


def _catalonia(fetch, where, select):
    """Rows of a Meteocat dataset on the Catalan open data portal (Socrata), 50000 per page."""
    out, k = [], 0
    while k < 20:
        q = urllib.parse.urlencode({"$where": where, "$select": select, "$order": ":id", "$limit": 50000,
                                    "$offset": k * 50000})
        rows = json.loads((fetch or _get)(f"{CATALONIA}/nzvn-apee.json?{q}", timeout=180))
        out += rows
        if len(rows) < 50000:
            break
        k += 1
    return out


def _list_ct(fetch):
    """The XEMA stations in operation (yqwd-vj5e) that sent precipitation (variable 35) in the last 6 hours."""
    since = time.strftime("%Y-%m-%dT%H:%M:%S", time.gmtime(time.time() - 6 * 3600))
    have = {r["codi_estacio"] for r in _catalonia(fetch, f"codi_variable='35' AND data_lectura >= '{since}'", "codi_estacio")}
    st = json.loads(fetch(f"{CATALONIA}/yqwd-vj5e.json?$limit=5000", timeout=60))
    return sorted(("ct", s["codi_estacio"], float(s["latitud"]), float(s["longitud"])) for s in st
                  if s.get("codi_estacio") in have and s.get("latitud") and s.get("longitud"))


def _list_bz(fetch):
    """South Tyrol: the stations with a precipitation sensor (type N); the station file carries LAT / LONG."""
    rain = {s["SCODE"] for s in json.loads(fetch(f"{SOUTH_TYROL}/sensors", timeout=60)) if s.get("TYPE") == "N"}
    out = []
    for f in json.loads(fetch(f"{SOUTH_TYROL}/stations", timeout=60))["features"]:
        p = f["properties"]
        if p.get("SCODE") in rain and p.get("LAT") and p.get("LONG"):
            out.append(("bz", p["SCODE"], float(p["LAT"]), float(p["LONG"])))
    return sorted(set(out))


def _trentino_rows(raw):
    """[(local time as if UTC, ms; mm)] of one Meteotrentino station file: 15-minute sums of yesterday and today."""
    import re
    x = raw.decode("utf-8", "replace")
    out = []
    for t, v in re.findall(r"<precipitazione[^>]*>\s*<data>([^<]+)</data>\s*<pioggia>([^<]*)</pioggia>", x):
        try:
            if float(v) >= 0:
                out.append((_ms(t[:19], "%Y-%m-%dT%H:%M:%S"), float(v)))
        except ValueError:
            continue
    return out


def _trentino_station(sid, fetch=None):
    try:
        return _trentino_rows((fetch or _get)(f"{TRENTINO}/ultimiDatiStazione?codice={sid}", timeout=60))
    except Exception:
        return []


def _list_tn(fetch):
    """Meteotrentino's stations in operation (no end date) whose file holds precipitation (one request each)."""
    import re
    x = fetch(f"{TRENTINO}/listaStazioni", timeout=60).decode("utf-8", "replace")
    cand = []
    for a in re.findall(r"<anagrafica>(.*?)</anagrafica>", x, re.S):
        f = dict(re.findall(r"<(codice|latitudine|longitudine)>([^<]*)</", a))
        if "<fine />" in a and f.get("codice") and f.get("latitudine") and f.get("longitudine"):
            cand.append((f["codice"], float(f["latitudine"]), float(f["longitudine"])))
    with ThreadPoolExecutor(3) as ex:
        has = list(ex.map(lambda c: bool(_trentino_station(c[0], fetch)), cand))
    return sorted(("tn", sid, lat, lon) for (sid, lat, lon), h in zip(cand, has) if h)


def _list_sy(fetch):
    """The European SYNOP stations that report a 1-hour precipitation amount (synop.SynopFeed in the OPERA
    instance; not Germany - the DWD gauges are denser). Id "<country>-<WMO number>"."""
    try:
        with open(os.path.join(ROOT, "synop", "stations.json")) as f:
            st = json.load(f)
    except (OSError, ValueError):
        return []
    return [("sy", f"{v[2]}-{wmo}", v[0], v[1]) for wmo, v in sorted(st.items()) if v[3] and v[2] != "de"]


LISTS = {"pl": _list_pl, "pt": _list_pt, "de": _list_de, "at": _list_at, "ch": _list_ch, "it": _list_it, "iv": _list_iv, "cz": _list_cz, "ie": _list_ie,
         "se": _list_se, "fi": _list_fi, "en": _list_en, "hu": _list_hu, "sy": _list_sy,
         # 16.11.2026 - new networks are only ever APPENDED to natobs/stations.json (station_list), in this order
         "sc": _list_sc, "be": _list_be, "bw": _list_bw, "bf": _list_bf, "gl": _list_gl, "ct": _list_ct,
         "bz": _list_bz, "tn": _list_tn, "si": _list_si, "dk": _list_dk}


# ---------- recording (inside the server) ----------
def station_list(fetch, log):
    """[(country, id, lat, lon)]; kept in natobs/stations.json so every file refers to the same order. A country
    that is not in the file yet is appended (at the next server start when its list could not be read) - the gauges
    already in the file never move (natobs, fcgauge and fcalert store arrays in this order)."""
    path = os.path.join(OBS_DIR, "stations.json")
    try:
        with open(path) as f:
            out = [tuple(s) for s in json.load(f)]
    except (OSError, ValueError):
        out = []
    have, n = {s[0] for s in out}, len(out)
    for c, fn in LISTS.items():
        if c not in have:
            try:
                out += fn(fetch)
            except Exception as e:
                log(f"natobs: gauge list {c} failed: {e}")
    if len(out) > n:
        os.makedirs(OBS_DIR, exist_ok=True)
        with open(path, "w") as f:
            json.dump(out, f)
    return out


def pixels(region, stations, merc):
    col, row = region.to_pixels(*merc(np.array([s[3] for s in stations]), np.array([s[2] for s in stations])))
    col, row = np.round(col).astype(int), np.round(row).astype(int)
    ok = (row >= 2) & (row < region.NY - 2) & (col >= 2) & (col < region.NX - 2)
    return col, row, ok


def at_gauges(grid, px, half=2):
    col, row, ok = px
    out = np.full(len(col), np.nan, np.float32)
    for k in np.flatnonzero(ok):
        c = grid[row[k] - half:row[k] + half + 1, col[k] - half:col[k] + half + 1]
        if (c >= 0).all():
            out[k] = c.mean()
    return out


def record(ts, opera, nat, who=None, layers=(), each=None, shadow=()):
    os.makedirs(OBS_DIR, exist_ok=True)
    extra = {} if who is None else {"who": who.astype(np.uint8), "layers": np.array(list(layers))}
    if each is not None:
        extra["each"] = each.astype(np.float16)
        extra["shadow"] = np.array(list(shadow), dtype="U8")
    np.savez_compressed(os.path.join(OBS_DIR, f"{int(ts)}.npz"), opera=opera.astype(np.float16), nat=nat.astype(np.float16), **extra)
    old = time.time() * 1000 - KEEP_DAYS * 86400_000
    for fp in glob.glob(os.path.join(OBS_DIR, "*.npz")):
        if int(os.path.basename(fp)[:-4]) < old:
            os.remove(fp)


FCA_DIR = os.path.join(LOG_DIR, "fcalert")  # the alerts' input at the gauges, per run (verify_alerts_eu.py)


def record_alert(ref, share, peak):
    """share: (1 + forecast steps, gauges) uint8 wet share of the alert circle x250 (255 = no data), row 0 = the
    picture at ref, then every 5-minute step; peak: the circle's highest rate (mm/h), same shape."""
    os.makedirs(FCA_DIR, exist_ok=True)
    np.savez_compressed(os.path.join(FCA_DIR, f"{int(ref)}.npz"), share=share, peak=peak)
    old = time.time() * 1000 - KEEP_DAYS * 86400_000
    for fp in glob.glob(os.path.join(FCA_DIR, "*.npz")):
        if int(os.path.basename(fp)[:-4]) < old:
            os.remove(fp)


def record_fc(ref, values, more=None):
    """values: (1 + len(FC_LEADS), gauges) - the picture at the run's time `ref`, then the forecast per lead.
    more: {variant: (len(FC_LEADS), gauges)} - other forecasts of the same run ("pure" = S-PROG alone, "cells" = with
    the weather model's new showers)."""
    os.makedirs(FC_DIR, exist_ok=True)
    np.savez_compressed(os.path.join(FC_DIR, f"{int(ref)}.npz"), fc=values.astype(np.float16),
                        **{k: v.astype(np.float16) for k, v in (more or {}).items()})
    old = time.time() * 1000 - KEEP_DAYS * 86400_000
    for fp in glob.glob(os.path.join(FC_DIR, "*.npz")):
        if int(os.path.basename(fp)[:-4]) < old:
            os.remove(fp)


# ---------- scoring: {gauge id: {ts: mm in the 10 min ending at ts}} per country ----------
def _get(url, timeout=120):
    import urllib.request
    with urllib.request.urlopen(urllib.request.Request(url, headers={"User-Agent": "Mozilla/5.0"}), timeout=timeout) as r:
        return r.read()


def _ms(text, fmt):
    return int(datetime.strptime(text, fmt).replace(tzinfo=timezone.utc).timestamp() * 1000)


def gauge_de(sid, since):
    """As check_rain_gauges.gauge, but reads only the rows since `since` (that one parses ~500 days per gauge:
    6 CPU-minutes for 300 gauges) and the "recent" file only when `since` is before today."""
    import check_rain_gauges as crg
    stamp = time.strftime("%Y%m%d%H%M", time.gmtime(since / 1000))
    today = time.strftime("%Y%m%d", time.gmtime()) + "0000"
    vals = {}
    for kind, name in (("recent", "akt"), ("now", "now")):
        if kind == "recent" and stamp >= today:
            continue
        try:
            z = zipfile.ZipFile(io.BytesIO(crg.get(f"{crg.CDC}/{kind}/10minutenwerte_nieder_{sid}_{name}.zip")))
        except Exception:
            continue
        rows = z.read(z.namelist()[0]).decode("latin-1").splitlines()
        k = [h.strip() for h in rows[0].split(";")].index("RWS_10")
        for r in reversed(rows[1:]):
            q = r.split(";")
            if len(q) <= k:
                continue
            d = q[1].strip()
            if d < stamp:
                break
            if float(q[k]) > -999:
                vals[_ms(d, "%Y%m%d%H%M")] = float(q[k])
    return vals


def gauges_de(ids, since, until):
    with ThreadPoolExecutor(8) as ex:
        return dict(zip(ids, ex.map(lambda s: gauge_de(s, since), ids)))


def gauges_at(ids, since, until):
    iso = lambda ms: datetime.fromtimestamp(ms / 1000, timezone.utc).strftime("%Y-%m-%dT%H:%M")
    out = {}
    for i in range(0, len(ids), 60):
        part = ids[i:i + 60]
        try:
            d = json.loads(_get(f"{TAWES}/station/historical/tawes-v1-10min?parameters=RR&station_ids={','.join(part)}"
                                f"&start={iso(since)}&end={iso(until)}"))
        except Exception as e:
            print(f"  (Austrian gauges {i}-{i + len(part)}: {e})")
            continue
        ts = [int(datetime.fromisoformat(t).timestamp() * 1000) for t in d["timestamps"]]
        for f in d["features"]:
            v = f["properties"]["parameters"]["RR"]["data"]
            out[str(f["properties"]["station"])] = {t: x for t, x in zip(ts, v) if x is not None and x >= 0}
        time.sleep(1)
    return out


def gauge_ch(sid, since):
    prefix, stn = sid.split(":")
    coll = SWISS_SETS[prefix]
    try:
        rows = _get(f"{SWISS}/ch.meteoschweiz.{coll}/{stn}/{coll}_{stn}_t_now.csv", timeout=60).decode("latin-1").splitlines()
        k = rows[0].split(";").index("rre150z0")
    except Exception:
        return {}
    vals = {}
    for r in rows[1:]:
        q = r.split(";")
        if len(q) > k and q[k] not in ("", "-"):
            try:
                vals[_ms(q[1], "%d.%m.%Y %H:%M")] = float(q[k])
            except ValueError:
                pass
    return vals


def gauges_ch(ids, since, until):
    with ThreadPoolExecutor(8) as ex:
        return dict(zip(ids, ex.map(lambda s: gauge_ch(s, since), ids)))


def gauges_it(ids, since, until):
    """ARPA Lombardia: rows (sensor, time in UTC+1, mm since the previous row, state "VA" = valid); the 1-minute
    gauges are summed into the 10 minutes ending at :00, :10, ..."""
    local = lambda ms: datetime.fromtimestamp(ms / 1000 + 3600, timezone.utc).strftime("%Y-%m-%dT%H:%M:%S")
    out = {}
    for i in range(0, len(ids), 40):
        part = ids[i:i + 40]
        where = (f"data > '{local(since)}' AND data <= '{local(until)}' AND idsensore in("
                 + ",".join(f"'{s}'" for s in part) + ")")
        q = urllib.parse.urlencode({"$where": where, "$limit": 50000, "$select": "idsensore,data,valore,stato"})
        try:
            rows = json.loads(_get(f"{LOMBARDY}/647i-nhxk.json?{q}"))
        except Exception as e:
            print(f"  (Lombardy gauges {i}-{i + len(part)}: {e})")
            continue
        for r in rows:
            v = float(r["valore"])
            if r.get("stato") != "VA" or v < 0:
                continue
            t = _ms(r["data"][:19], "%Y-%m-%dT%H:%M:%S") - 3600_000
            t = -(-t // 600_000) * 600_000  # the end of its 10 minutes
            g = out.setdefault(str(r["idsensore"]), {})
            g[t] = g.get(t, 0.0) + v
        time.sleep(1)
    return out


def gauge_iv(sid, since):
    try:
        rows = json.loads(_get(f"{VENETO}/meteo_meteogrammi_tabella?codseqst={sid}", timeout=60))["data"]
    except Exception:
        return {}
    five = {}
    for r in rows:
        if r["tipo"] == "PREC" and r["valore"] not in (None, ""):
            five[_ms(r["dataora"][:19], "%Y-%m-%dT%H:%M:%S") - 3600_000] = float(r["valore"])
    return {t: v + five[t - 300_000] for t, v in five.items() if t % 600_000 == 0 and t - 300_000 in five and t >= since}


def gauges_iv(ids, since, until):
    with ThreadPoolExecutor(4) as ex:
        return dict(zip(ids, ex.map(lambda s: gauge_iv(s, since), ids)))


def gauge_cz(sid, since):
    vals = {}
    day = datetime.fromtimestamp(since / 1000, timezone.utc).date()
    while day <= datetime.now(timezone.utc).date():
        try:
            d = json.loads(_get(f"{CZECH}/data/10m-{sid}-{day:%Y%m%d}.json", timeout=60))["data"]["data"]
            k = {h: i for i, h in enumerate(d["header"].split(","))}
            for r in d["values"]:
                if r[k["ELEMENT"]] == "SRA10M" and r[k["VAL"]] is not None and r[k["VAL"]] >= 0:
                    vals[_ms(r[k["DT"]][:19], "%Y-%m-%dT%H:%M:%S")] = float(r[k["VAL"]])
        except Exception:
            pass
        day = datetime.fromordinal(day.toordinal() + 1).date()
    return vals


def gauges_cz(ids, since, until):
    with ThreadPoolExecutor(8) as ex:
        return dict(zip(ids, ex.map(lambda s: gauge_cz(s, since), ids)))


def gauges_ie(ids, since, until):
    try:
        raw = _get(EMILIA, timeout=180)
    except Exception as e:
        print(f"  (Emilia-Romagna gauges: {e})")
        return {}
    want, five = set(ids), {}
    for sid, _, _, t, v in _emilia_rows(raw):
        if sid in want and since - 900_000 <= t <= until + 900_000:
            for k in range(3):
                five.setdefault(sid, {})[t - k * 300_000] = v / 3
    return {sid: {t: v + f[t - 300_000] for t, v in f.items() if t % 600_000 == 0 and t - 300_000 in f}
            for sid, f in five.items()}


def _spread15(fifteen):
    """{id: {end of 15 minutes: mm}} -> {id: {ts: mm in the 10 minutes ending at ts}}: every sum spread evenly over
    its three 5-minute bins, as in gauges_ie (exact over full hours)."""
    out = {}
    for sid, ser in fifteen.items():
        five = {}
        for t, v in ser.items():
            for k in range(3):
                five[t - k * 300_000] = v / 3
        g = {t: v + five[t - 300_000] for t, v in five.items() if t % 600_000 == 0 and t - 300_000 in five}
        if g:
            out[sid] = g
    return out


def _sweden_values(stations):
    """{station: {end of 15 minutes: mm}} of SMHI station entries; quality G (checked) or Y (suspect/unchecked)."""
    out = {}
    for key, values in stations:
        for r in values or []:
            try:
                v = float(r["value"])
            except (KeyError, TypeError, ValueError):
                continue
            if v >= 0 and r.get("quality") in ("G", "Y"):
                out.setdefault(str(key), {})[int(r["date"])] = v
    return out


def gauge_se(sid):
    try:
        d = json.loads(_get(f"{SWEDEN}/station/{sid}/period/latest-day/data.json", timeout=30))
        return _sweden_values([(sid, d.get("value"))]).get(sid, {})
    except Exception:
        return {}


def gauges_se(ids, since, until):
    """The last hour from the one bulk file; a longer look-back from each station's last 24 hours (4 workers)."""
    want = set(ids)
    try:
        if since >= time.time() * 1000 - 3600_000:
            d = json.loads(_get(f"{SWEDEN}/station-set/all/period/latest-hour/data.json", timeout=60))
            fifteen = _sweden_values((s["key"], s.get("value")) for s in d.get("station", []) if str(s["key"]) in want)
        else:
            with ThreadPoolExecutor(4) as ex:
                fifteen = dict(zip(ids, ex.map(gauge_se, ids)))
    except Exception as e:
        print(f"  (Swedish gauges: {e})")
        return {}
    return _spread15({sid: {t: v for t, v in f.items() if since - 900_000 <= t <= until + 900_000}
                      for sid, f in fifteen.items()})


def gauges_fi(ids, since, until):
    try:
        raw = _fmi_get(since - 600_000, until)
    except Exception as e:
        print(f"  (Finnish gauges: {e})")
        return {}
    want, out = set(ids), {}
    for sid, _, _, t, v in _fmi_rows(raw):
        if sid in want and since <= t <= until:
            out.setdefault(sid, {})[t] = v
    return out


def gauges_en(ids, since, until):
    """All rainfall readings of the days in question, newest first, page by page (4 in parallel) until the pages
    reach `since`; only the 15-minute tipping-bucket totals of the listed stations are kept."""
    day = lambda ms: datetime.fromtimestamp(ms / 1000, timezone.utc).strftime("%Y-%m-%d")
    lo = since - 900_000
    url = (f"{ENGLAND}/data/readings?parameter=rainfall&startdate={day(lo)}&enddate={day(until)}&_sorted&_limit=10000"
           "&_offset=")
    want, fifteen, t0 = set(ids), {}, time.time()

    def page(k):
        try:
            return json.loads(_get(f"{url}{k * 10000}", timeout=90)).get("items", [])
        except Exception as e:
            print(f"  (English gauges, page {k}: {e})")
            return None

    k, n = 0, min(4, 1 + int((until - lo) / 3600_000 * 0.38))  # ~3800 readings an hour
    while k < 40 and time.time() - t0 < 50:
        done = False
        with ThreadPoolExecutor(4) as ex:
            pages = list(ex.map(page, range(k, k + n)))
        for items in pages:
            if items is None:
                continue
            for r in items:
                m = str(r.get("measure", "")).rsplit("/", 1)[-1]
                v = r.get("value")
                if not m.endswith(ENGLAND_MEASURE) or not isinstance(v, (int, float)) or v < 0:
                    continue
                sid = m[:-len(ENGLAND_MEASURE)]
                try:
                    t = _ms(r["dateTime"][:19], "%Y-%m-%dT%H:%M:%S")
                except (KeyError, ValueError):
                    continue
                if sid in want and lo <= t <= until + 900_000:
                    fifteen.setdefault(sid, {})[t] = float(v)
            if len(items) < 10000 or (items and items[-1].get("dateTime", "9")[:19] < datetime.fromtimestamp(
                    lo / 1000, timezone.utc).strftime("%Y-%m-%dT%H:%M:%S")):
                done = True
        if done or all(p is None for p in pages):
            break
        k, n = k + n, 4
    return _spread15(fifteen)


def gauges_hu(ids, since, until):
    """One small file per 10 minutes (4 workers); a later file may repeat or complete an earlier time."""
    try:
        names = _hungary_files(_get, since, until)
    except Exception as e:
        print(f"  (Hungarian gauges: {e})")
        return {}

    def rows(name):
        try:
            return list(_hungary_rows(_get(f"{HUNGARY}/{name}", timeout=30)))
        except Exception:
            return []

    want, out = set(ids), {}
    with ThreadPoolExecutor(4) as ex:
        for part in ex.map(rows, names):
            for sid, _, _, t, v in part:
                if sid in want and since <= t <= until:
                    out.setdefault(sid, {})[t] = v
    return out


def _read_latest(net, ids, since, until):
    """What poll_latest_networks kept for this network (<ROOT>/synop/<net>_<day>.jsonl; the last line wins)."""
    want, out = set(ids), {}
    for day in range(int(since // 86400_000), int(until // 86400_000) + 1):
        fp = os.path.join(ROOT, "synop", f"{net}_{time.strftime('%Y%m%d', time.gmtime(day * 86400))}.jsonl")
        if not os.path.exists(fp):
            continue
        with open(fp) as f:
            for line in f:
                try:
                    r = json.loads(line)
                except ValueError:
                    continue
                if r["id"] in want and since <= r["t"] <= until:
                    out.setdefault(r["id"], {})[r["t"]] = r["mm"]
    return out


def gauges_pl(ids, since, until):
    return _read_latest("pl", ids, since, until)


def gauges_gl(ids, since, until):
    return _read_latest("gl", ids, since, until)


def gauges_si(ids, since, until):
    return _read_latest("si", ids, since, until)


def _pairs5(five):
    """{end of 5 minutes: mm} -> {end of 10 minutes (:00, :10, ...): mm} where both halves are known."""
    return {t: v + five[t - 300_000] for t, v in five.items() if t % 600_000 == 0 and t - 300_000 in five}


def gauges_sc(ids, since, until):
    """SEPA's 15-minute totals (stamps = the end, UTC), spread as "ie"."""
    return _spread15(_kiwis_values(SCOTLAND, ids, since - 900_000, until + 900_000, "Scottish"))


def gauges_bf(ids, since, until):
    """VMM's 15-minute totals (stamps = the end, +01:00), spread as "ie"."""
    return _spread15(_kiwis_values(FLANDERS, ids, since - 900_000, until + 900_000, "Flemish"))


def gauges_bw(ids, since, until):
    """SPW's 5-minute sums (stamps = the end, +01:00; 0.1 mm tips) paired into 10 minutes."""
    out = {sid: _pairs5(f) for sid, f in _kiwis_values(WALLONIA, ids, since - 600_000, until, "Walloon").items()}
    return {sid: g for sid, g in out.items() if g}


def gauges_be(ids, since, until):
    """RMI's 10-minute sums (timestamp = the end of the 10 minutes, UTC without a zone: on 16.11.2026 15:29 UTC the
    newest was 15:20, from 8 of 14 stations)."""
    if not ids:
        return {}
    iso = lambda ms: datetime.fromtimestamp(ms / 1000, timezone.utc).strftime("%Y-%m-%dT%H:%M:%S")
    try:
        rows = _belgium(None, "aws:aws_10min", f"timestamp > '{iso(since - 600_000)}' AND timestamp <= '{iso(until)}'",
                        "code,timestamp,precip_quantity")
    except Exception as e:
        print(f"  (Belgian RMI gauges: {e})")
        return {}
    want, out = set(ids), {}
    for r in rows:
        try:
            if r["code"] in want and r.get("precip_quantity") not in (None, "") and float(r["precip_quantity"]) >= 0:
                out.setdefault(r["code"], {})[_ms(r["timestamp"][:19], "%Y-%m-%dT%H:%M:%S")] = float(r["precip_quantity"])
        except (KeyError, ValueError):
            continue
    return out


def gauges_dk(ids, since, until):
    if not ids:
        return {}
    try:
        rows = list(_denmark(since - 600_000, until))
    except Exception as e:
        print(f"  (Danish gauges: {e})")
        return {}
    want, out = set(ids), {}
    for sid, _, _, t, v in rows:
        if sid in want and since <= t <= until:
            out.setdefault(sid, {})[t] = v
    return out


def gauges_ct(ids, since, until):
    """Meteocat's half-hour sums: data_lectura is the START of the half hour in UTC ("records labelled at the front,
    time in UT" - the dataset's column description); base "HO" = a whole hour. Spread evenly over the 10-minute steps."""
    if not ids:
        return {}
    iso = lambda ms: datetime.fromtimestamp(ms / 1000, timezone.utc).strftime("%Y-%m-%dT%H:%M:%S")
    try:
        rows = _catalonia(None, f"codi_variable='35' AND data_lectura >= '{iso(since - 3600_000)}' AND "
                                f"data_lectura <= '{iso(until)}'", "codi_estacio,data_lectura,valor_lectura,codi_base")
    except Exception as e:
        print(f"  (Catalan gauges: {e})")
        return {}
    want, out = set(ids), {}
    for r in rows:
        try:
            n = 6 if r.get("codi_base") == "HO" else 3
            t1, v = _ms(r["data_lectura"][:19], "%Y-%m-%dT%H:%M:%S") + n * 600_000, float(r["valor_lectura"])
        except (KeyError, TypeError, ValueError):
            continue
        if r["codi_estacio"] in want and v >= 0 and t1 % 600_000 == 0:
            for k in range(n):
                out.setdefault(r["codi_estacio"], {})[t1 - k * 600_000] = v / n
    return out


def gauge_bz(sid, since, until):
    """One South Tyrol gauge: 5-minute sums, stamped "...CET" / "...CEST" (local time; date_from / date_to are local
    too, so the window asked for is widened by 2 hours)."""
    loc = lambda ms: time.strftime("%Y%m%d%H%M", time.gmtime(ms / 1000))
    try:
        d = json.loads(_get(f"{SOUTH_TYROL}/timeseries?station_code={sid}&sensor_code=N&output_format=JSON"
                            f"&date_from={loc(since - 2 * 3600_000)}&date_to={loc(until + 3 * 3600_000)}", timeout=60))
    except Exception:
        return {}
    five = {}
    for r in d if isinstance(d, list) else []:
        try:
            s, v = r["DATE"], float(r["VALUE"])
            off = {"CET": 1, "CEST": 2}[s[19:]]
            t = _ms(s[:19], "%Y-%m-%dT%H:%M:%S") - off * 3600_000
        except (KeyError, TypeError, ValueError):
            continue
        if v >= 0 and since - 600_000 <= t <= until:
            five[t] = v
    return _pairs5(five)


def gauges_bz(ids, since, until):
    with ThreadPoolExecutor(3) as ex:
        return {sid: g for sid, g in zip(ids, ex.map(lambda s: gauge_bz(s, since, until), ids)) if g}


def _eu_summer(ms):
    """Central European summer time at this UTC instant (last Sunday of March 01 UTC .. last Sunday of October)."""
    import calendar
    y = time.gmtime(ms / 1000).tm_year
    last_sun = lambda m: max(d for d in range(25, 32) if calendar.weekday(y, m, d) == 6)
    a = calendar.timegm((y, 3, last_sun(3), 1, 0, 0)) * 1000
    b = calendar.timegm((y, 10, last_sun(10), 1, 0, 0)) * 1000
    return a <= ms < b


def gauges_tn(ids, since, until):
    """Meteotrentino: 15-minute sums (stamp = the end; 0.2 mm tips) in local time WITHOUT a zone. Whether that is
    UTC+1 all year (solar time, as Lombardy and Veneto) or Italian civil time is read from the data: the newest stamp
    of all gauges against the clock (~15-30 min behind: UTC+1 -> 30-45 min ahead, UTC+2 -> 90-105 min ahead).
    One request per gauge (3 workers): each answers yesterday and today."""
    with ThreadPoolExecutor(3) as ex:
        raw = dict(zip(ids, ex.map(_trentino_station, ids)))
    newest = max((t for rows in raw.values() for t, _ in rows), default=None)
    if newest is None:
        return {}
    ahead = (newest - time.time() * 1000) / 3600_000
    civil = 1.25 < ahead < 2.5  # the stamps run on summer time -> Italian civil time (CET / CEST by date)
    fifteen = {}
    for sid, rows in raw.items():
        for t, v in rows:
            off = 2 if civil and _eu_summer(t - 2 * 3600_000) else 1
            u = t - off * 3600_000
            if since - 900_000 <= u <= until + 900_000:
                fifteen.setdefault(sid, {})[u] = v
    return _spread15(fifteen)


def gauges_pt(ids, since, until):
    """IPMA's hourly precipitation ("precAcumulada", the hour ending at the stamp, UTC), spread over its six 10-minute
    steps - only the hourly tables are meaningful."""
    d = json.loads(_get(f"{PORTUGAL}/observations.json", timeout=120))
    want, out = set(ids), {}
    for stamp, sts in d.items():
        t = _ms(stamp, "%Y-%m-%dT%H:%M")
        if not since - 3600_000 <= t <= until:
            continue
        for sid, v in (sts or {}).items():
            mm = (v or {}).get("precAcumulada")
            if sid in want and mm is not None and mm >= 0:
                for k in range(6):
                    out.setdefault(sid, {})[t - k * 600_000] = mm / 6
    return out


def gauges_sy(ids, since, until):
    """SYNOP 1-hour amounts, spread evenly over the hour's six 10-minute steps - only the HOURLY tables are
    meaningful for this network (the 10-minute ones see a smooth hour)."""
    import synop
    by_wmo = {sid.split("-", 1)[1]: sid for sid in ids}
    out = {}
    for wmo, ser in synop.hourly_rain(ROOT, list(by_wmo), since - 3600_000, until).items():
        g = {}
        for t, mm in ser.items():
            if t % 600_000 == 0:
                for k in range(6):
                    g[t - k * 600_000] = mm / 6
        out[by_wmo[wmo]] = g
    return out


GAUGES = {"pl": gauges_pl, "pt": gauges_pt, "de": gauges_de, "at": gauges_at, "ch": gauges_ch, "it": gauges_it, "iv": gauges_iv, "cz": gauges_cz,
          "ie": gauges_ie, "se": gauges_se, "fi": gauges_fi, "en": gauges_en, "hu": gauges_hu, "sy": gauges_sy,
          "sc": gauges_sc, "be": gauges_be, "bw": gauges_bw, "bf": gauges_bf, "gl": gauges_gl, "ct": gauges_ct,
          "bz": gauges_bz, "tn": gauges_tn, "si": gauges_si, "dk": gauges_dk}


# ---------- the gauge archive: the sources keep hours to days, a model is built on weeks ----------
def stations():
    """The gauge list both Europe instances use (kept by the own-picture instance)."""
    try:
        with open(os.path.join(ROOT, "eun", "natobs", "stations.json")) as f:
            return [tuple(s) for s in json.load(f)]
    except (OSError, ValueError):
        return []


def _day_file(day):
    return os.path.join(TRUTH_DIR, time.strftime("%Y%m%d", time.gmtime(day * 86400)) + ".npz")


def _day_load(day, rejected=None):
    """{"country:id": float32[144]} (mm in the 10 minutes ending at 00:00, 00:10, ... UTC; NaN = not known).
    Implausible values (implausible()) read as NaN; rejected: a dict that collects {key: largest dropped mm}."""
    try:
        z = np.load(_day_file(day))
        ids, mm = [str(x) for x in z["ids"]], z["mm"].astype(np.float32)
    except (OSError, ValueError, KeyError):
        return {}
    _drop_implausible(ids, mm, rejected)
    return dict(zip(ids, mm))


# ---------- plausibility of the gauge values (server-v1.79, 20.11.2026) ----------
# 20.11.: one England gauge (en:1552, radar dry) reported 580 and 1161 mm in 10 minutes - 97 % of all English gauge
# rain in the gauge adjustment's 12 h window: England's factor went from x1.0 to the x3.0 cap for 12 h; a Swedish
# gauge (se:22228110) 120 mm six times in a row. No check existed. Archive 13.-20.11. (6200 gauges): apart from
# these, the largest 10-minute values were 21 mm (dk:05140, radar dry) and 12 mm (en:52148, radar 11 mm/h: real).
# Dropped where the archive is written and read (archive_truth, _day_load: the gauge adjustment and every score).
# The archive holds 10-minute sums for every network; longer sums were spread evenly (_spread15, hourly / 6, ...):
NATIVE_MIN = {"ie": 15, "se": 15, "en": 15, "sc": 15, "bf": 15, "tn": 15, "ct": 30, "pt": 60, "sy": 60}
MAX_10MIN_MM = 30.0   # per 10 min; over a native interval of m minutes 30 * sqrt(m / 10) mm (world records ~ D^0.5)
REPEAT_N, REPEAT_MM = 4, 2.0  # the same value >= 2 mm in >= 4 native intervals in a row: a stuck gauge (13.-20.11.:
                              # real rain repeated <= 1.4 mm 4 times in the 10-minute networks)
# a spike: >= 3 mm in 10 min while >= 3 gauges within 20 km all read < 5 % of it within +-1 native interval.
# 13.-20.11.: 16 values at 9 gauges, 15 of them with the radar dry there (< 0.3 mm/h, one shower in Galicia, radar
# 6 mm/h); at 10 % also 6 more showers (radar 1-4 mm/h, Galicia / Wallonia / Austria) would have gone
NB_KM, NB_N, NB_MIN_MM, NB_FRAC = 20.0, 3, 3.0, 0.05
_COORDS = {}


def _coords():
    """{"net:id": (lat, lon)} of stations(), re-read when the list changes."""
    fp = os.path.join(ROOT, "eun", "natobs", "stations.json")
    try:
        m = os.path.getmtime(fp)
    except OSError:
        return {}
    if _COORDS.get("mtime") != m:
        _COORDS.clear()
        _COORDS.update(mtime=m, xy={f"{s[0]}:{s[1]}": (float(s[2]), float(s[3])) for s in stations()})
    return _COORDS["xy"]


def implausible(ids, mm):
    """ids: "net:id" per row, mm: float (n, 144) 10-minute sums of one UTC day -> bool (n, 144): values to drop
    (too large for the network's interval, a stuck repeat, or a spike that no gauge around it sees)."""
    mm = np.asarray(mm, np.float32)
    bad = np.zeros(mm.shape, bool)
    if not len(ids):
        return bad
    nat = np.array([NATIVE_MIN.get(k.split(":", 1)[0], 10) for k in ids], np.float32)
    v = np.where(np.isfinite(mm), mm, -1.0)
    bad |= v > (MAX_10MIN_MM / np.sqrt(nat / 10))[:, None]
    # stuck repeats
    same = (v[:, 1:] == v[:, :-1]) & (v[:, 1:] >= REPEAT_MM)
    for i in np.flatnonzero(same.any(1)):
        need = int(np.ceil(REPEAT_N * nat[i] / 10))
        j = 0
        while j < 143:
            if same[i, j]:
                e = j
                while e < 143 and same[i, e]:
                    e += 1
                if e - j + 1 >= need:  # steps j .. e hold the same value
                    bad[i, j:e + 1] = True
                j = e
            j += 1
    # spikes against the neighbours (only where positions are known and >= NB_N neighbours report)
    cand = np.argwhere((v >= NB_MIN_MM) & ~bad)
    xy = _coords() if len(cand) else {}
    if len(cand) and xy:
        lat = np.array([xy.get(k, (np.nan, np.nan))[0] for k in ids])
        lon = np.array([xy.get(k, (np.nan, np.nan))[1] for k in ids])
        for i, j in cand:
            if not np.isfinite(lat[i]):
                continue
            d = np.hypot((lat - lat[i]) * 111.2, (lon - lon[i]) * 111.2 * np.cos(np.radians(lat[i])))
            nb = np.flatnonzero(d <= NB_KM)
            nb = nb[nb != i]
            w = max(1, int(round(nat[i] / 10)))
            win = mm[nb, max(0, j - w):j + w + 1]
            known = np.isfinite(win).any(1)
            if known.sum() >= NB_N and np.nanmax(win[known]) < NB_FRAC * v[i, j]:
                bad[i, j] = True
    # a 15-minute sum lands unevenly on two 10-minute steps (1/3, 2/3): drop the steps next to a bad one too
    odd = (nat % 10 != 0)[:, None]
    bad |= odd & (np.pad(bad[:, 1:], ((0, 0), (0, 1))) | np.pad(bad[:, :-1], ((0, 0), (1, 0))))
    return bad


def _drop_implausible(ids, mm, rejected=None):
    """Set implausible values of a day (implausible()) to NaN in place; rejected collects {key: largest mm}."""
    bad = implausible(ids, mm)
    for i in np.flatnonzero(bad.any(1)):
        if rejected is not None:
            rejected[ids[i]] = max(rejected.get(ids[i], 0.0), float(np.nanmax(mm[i][bad[i]])))
        mm[i][bad[i]] = np.nan
    return bad


def archive_truth(st, hours=6.0):
    """Read the last hours of every gauge network and keep them (server loop, hourly, OPERA instance). Returns the
    number of values now known for that time. A network that fails is skipped - the next hours read it again."""
    now = int(time.time() * 1000)
    since = now - int(hours * 3600_000)
    days, n = {}, 0
    for c, fn in GAUGES.items():
        try:
            got = fn([s[1] for s in st if s[0] == c], since, now)
        except Exception as e:
            print(f"  (gauge archive {c}: {e})")
            continue
        for sid, series in got.items():
            for t, v in series.items():
                if t % 600_000 == 0 and since <= t <= now and v is not None and np.isfinite(v):
                    day = t // 86400_000
                    if day not in days:
                        days[day] = _day_load(day)
                    row = days[day].setdefault(f"{c}:{sid}", np.full(144, np.nan, np.float32))
                    row[(t % 86400_000) // 600_000] = v
                    n += 1
    os.makedirs(TRUTH_DIR, exist_ok=True)
    rejected = {}
    for day, d in days.items():
        if d:
            ids = sorted(d)
            mm = np.stack([d[i] for i in ids]).astype(np.float32)
            _drop_implausible(ids, mm, rejected)  # never kept (server-v1.79)
            tmp = _day_file(day) + ".tmp.npz"
            np.savez_compressed(tmp, ids=np.array(ids), mm=mm)
            os.replace(tmp, _day_file(day))
    if rejected:  # one short line per hourly run
        top = sorted(rejected.items(), key=lambda x: -x[1])
        print(f"gauge archive: implausible values dropped at {len(top)} gauges: "
              + ", ".join(f"{k} {v:.1f}" for k, v in top[:8]) + (" ..." if len(top) > 8 else ""), flush=True)
    old = time.time() - TRUTH_DAYS * 86400
    for fp in glob.glob(os.path.join(TRUTH_DIR, "*.npz")):
        if os.path.getmtime(fp) < old:
            os.remove(fp)
    return n


def truth(st, since, until, live=True):
    """{country: {gauge id: {ts: mm in the 10 minutes ending at ts}}} for the scoring scripts: the archive, and on
    top of it (live) what the sources have right now - so old days stay scorable after the sources dropped them."""
    out = {c: {} for c in GAUGES}
    for day in range(int(since // 86400_000), int(until // 86400_000) + 1):
        for key, row in _day_load(day).items():
            c, sid = key.split(":", 1)
            ser = out.setdefault(c, {}).setdefault(sid, {})
            for k in np.flatnonzero(np.isfinite(row)):
                t = day * 86400_000 + int(k) * 600_000
                if since <= t <= until:
                    ser[t] = float(row[k])
    if live:
        for c, fn in GAUGES.items():
            ids = [s[1] for s in st if s[0] == c]
            try:
                for sid, series in fn(ids, since, until).items():
                    out[c].setdefault(sid, {}).update(series)
            except Exception as e:
                print(f"  ({c} gauges, live: {e})")
    return out


def table(name, g, cols, ths=(0.1, 0.3, 1.0), unit="mm/h"):
    """cols: {label: values on the same pairs as g}. A column with gaps (a layer without data there) is scored on
    its own pairs (their number is printed). "x" = rain amount against the gauges (not printed when they are dry)."""
    g = np.asarray(g, np.float32)
    if not len(g):
        print(f"\n{name}: no pairs")
        return
    print(f"\n{name}: {len(g)} pairs, gauge mean {g.mean():.3f} {unit}")
    for label, q in cols.items():
        q = np.asarray(q, np.float32)
        ok = np.isfinite(q)
        if not ok.any():
            continue
        gg, qq = g[ok], q[ok]
        cells = []
        for th in ths:
            wo, wf = gg >= th, qq >= th
            h, m, f = (wo & wf).sum(), (wo & ~wf).sum(), (~wo & wf).sum()
            cells.append(f"{th}: CSI {h / (h + m + f):.2f} POD {h / max(1, h + m):.2f} FAR {f / max(1, h + f):.2f}"
                         if h + m + f else f"{th}: nothing wet           ")
        ratio = f"x{qq.sum() / gg.sum():5.2f}" if gg.mean() >= 0.005 else "x   - "
        print(f"  {label:12s} mean {qq.mean():.3f} {ratio}   " + "   ".join(cells) + ("" if ok.all() else f"   ({ok.sum()} pairs)"))


def hourly(a):
    """Rows (gauge, time, values ...) -> one row per gauge and clock hour: the mean of its six 10-minute rows (hours
    with fewer are left out). A tipping gauge resolves 0.1-0.2 mm, so light rain can only be judged on such sums."""
    key = a[:, 0] * 1e7 + a[:, 1] // 3600_000
    order = np.argsort(key, kind="stable")
    a, key = a[order], key[order]
    _, start, count = np.unique(key, return_index=True, return_counts=True)
    s = start[count == 6]
    return a[s[:, None] + np.arange(6)].mean(axis=1)


def columns(a, combo=""):
    cols = {"OPERA alone": a[:, 3], "own picture": a[:, 4]}
    names = combo.split("+")
    if len(names) > 1:  # where layers overlap: each of them on its own
        cols.update({f"only {l}": a[:, 5 + i] for i, l in enumerate(names)})
    return cols


def main():
    hours = float(sys.argv[sys.argv.index("--hours") + 1]) if "--hours" in sys.argv else 24
    with open(os.path.join(OBS_DIR, "stations.json")) as f:
        st = [tuple(s) for s in json.load(f)]
    now = time.time() * 1000
    files = sorted(fp for fp in glob.glob(os.path.join(OBS_DIR, "*.npz"))
                   if now - hours * 3600_000 < int(os.path.basename(fp)[:-4]) < now - 45 * 60_000)
    files = [fp for fp in files if int(os.path.basename(fp)[:-4]) % 600_000 == 0]  # the gauges report every 10 min
    if not files:
        print("no recorded images yet")
        return
    t0, t1 = int(os.path.basename(files[0])[:-4]), int(os.path.basename(files[-1])[:-4])
    print(f"{len(files)} images {datetime.fromtimestamp(t0 / 1000, timezone.utc):%d.%m. %H:%M} - "
          f"{datetime.fromtimestamp(t1 / 1000, timezone.utc):%d.%m. %H:%M} UTC, {len(st)} gauges")
    truth_all = truth(st, t0 - 1200_000, t1 + 1200_000, live="--archive" not in sys.argv)
    # (country, the layers that count at the gauge) -> rows (gauge, time, gauge mm/h, OPERA, own, each of those layers alone)
    rows = {}
    shadows = {}  # (country, shadow layer) -> rows (gauge, time, gauge mm/h, OPERA, own, the shadow alone)
    for fp in files:
        t = int(os.path.basename(fp)[:-4])
        z = np.load(fp)
        o, n = z["opera"].astype(np.float32), z["nat"].astype(np.float32)
        who = z["who"] if "who" in z.files else None  # files from before server-v1.12 have none
        layers = [str(x) for x in z["layers"]] if who is not None else []
        each = z["each"].astype(np.float32) if "each" in z.files else None  # since server-v1.13
        shadow = {layers.index(str(x)) for x in z["shadow"]} if "shadow" in z.files else set()  # since server-v1.14
        for k, (c, sid, _, _) in enumerate(st[:len(o)]):  # files from before a list was added are shorter
            g = truth_all.get(c, {}).get(sid, {})
            v = [g.get(t), g.get(t + 600_000)]
            if None in v or not np.isfinite(o[k]) or not np.isfinite(n[k]):
                continue
            for b in shadow:
                if np.isfinite(each[b, k]):
                    shadows.setdefault((c, layers[b]), []).append((k, t, np.mean(v) * 6, o[k], n[k], each[b, k]))
            idx = [b for b in range(len(layers)) if who[k] >> b & 1 and b not in shadow]
            combo = "?" if who is None else "+".join(layers[b] for b in idx) or "no layer"
            alone = [each[b, k] if each is not None else np.nan for b in idx]
            rows.setdefault((c, combo), []).append((k, t, np.mean(v) * 6, o[k], n[k], *alone))
    rows = {key: np.array(r, np.float64) for key, r in rows.items()}
    country = {c: np.concatenate([r[:, :5] for (cc, _), r in rows.items() if cc == c] or [np.zeros((0, 5))]) for c in GAUGES}
    for c in GAUGES:
        table(NAMES[c], country[c][:, 2], columns(country[c]))
    print("\n--- sums over full hours (thresholds in mm; light rain is judged here, a gauge tips at 0.1-0.2 mm) ---")
    for c in GAUGES:
        h = hourly(country[c])
        table(NAMES[c] + ", hours", h[:, 2], columns(h), (0.2, 1.0, 3.0), "mm")
    print("\n--- by the layers that count at the gauge (at least half their weight; 50 pairs or more); "
          "\"only x\" = that layer on its own ---")
    for (c, combo), r in sorted(rows.items()):
        if len(r) >= 50 and combo != "?":
            table(f"{NAMES[c]} - {combo}", r[:, 2], columns(r, combo))
            h = hourly(r)
            if len(h) >= 20:
                table(f"{NAMES[c]} - {combo}, hours", h[:, 2], columns(h, combo), (0.2, 1.0, 3.0), "mm")


    print("\n--- shadow layers (candidates, not in the picture): where they have data, 50 pairs or more ---")
    for (c, l), r in sorted(shadows.items(), key=lambda x: (x[0][1], x[0][0])):
        r = np.array(r, np.float64)
        if len(r) >= 50:
            for a, ths, unit, tag in ((r, (0.1, 0.3, 1.0), "mm/h", ""), (hourly(r), (0.2, 1.0, 3.0), "mm", ", hours")):
                if len(a) >= 20:
                    table(f"shadow {l} - {NAMES[c]}{tag}", a[:, 2], {**columns(a), f"only {l}": a[:, 5]}, ths, unit)


if __name__ == "__main__":
    main()
