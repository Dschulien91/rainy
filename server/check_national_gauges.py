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
KEEP_DAYS = 4
TAWES = "https://dataset.api.hub.geosphere.at/v1"
SWISS = "https://data.geo.admin.ch"
SWISS_SETS = {"smn": "ogd-smn", "pre": "ogd-smn-precip"}  # id prefix -> collection (ch.meteoschweiz.<...>)
LOMBARDY = "https://www.dati.lombardia.it/resource"
VENETO = "https://api.arpa.veneto.it/REST/v1"
CZECH = "https://opendata.chmi.cz/meteorology/climate/now"
EMILIA = "https://dati-simc.arpae.it/opendata/osservati/meteo/realtime/realtime.jsonl"
NAMES = {"de": "Germany (DWD gauges)", "at": "Austria (TAWES gauges)", "ch": "Switzerland (MeteoSwiss gauges)",
         "it": "Italy, Lombardy (ARPA Lombardia gauges)", "iv": "Italy, Veneto (ARPAV gauges)",
         "cz": "Czechia (CHMI gauges)", "ie": "Italy, Emilia-Romagna (ARPAE gauges)"}


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


LISTS = {"de": _list_de, "at": _list_at, "ch": _list_ch, "it": _list_it, "iv": _list_iv, "cz": _list_cz, "ie": _list_ie}


# ---------- recording (inside the server) ----------
def station_list(fetch, log):
    """[(country, id, lat, lon)]; kept in natobs/stations.json so every file refers to the same order. A country
    that is not in the file yet is appended (at the next server start when its list could not be read)."""
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


def at_gauges(grid, px):
    col, row, ok = px
    out = np.full(len(col), np.nan, np.float32)
    for k in np.flatnonzero(ok):
        c = grid[row[k] - 2:row[k] + 3, col[k] - 2:col[k] + 3]
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


GAUGES = {"de": gauges_de, "at": gauges_at, "ch": gauges_ch, "it": gauges_it, "iv": gauges_iv, "cz": gauges_cz,
          "ie": gauges_ie}


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
    truth = {c: fn([s[1] for s in st if s[0] == c], t0 - 1200_000, t1 + 1200_000) for c, fn in GAUGES.items()}
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
            g = truth[c].get(sid, {})
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
