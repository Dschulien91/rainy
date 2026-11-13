"""Is our own Europe picture (OPERA + national layers) closer to the rain gauges than OPERA alone?

The third instance (EU_NATIONAL=1) notes for every image it composes, at every gauge of the lists below, what OPERA
alone showed there and what the composed picture shows (5 x 5 cells around the gauge, mm/h):
verify_log/eun/natobs/<ts>.npz with "opera" and "nat" in the order of natobs/stations.json (record(), called by
rainy_server.EuropeNational.load), and "who": which layers count at the gauge in that image (bit k = layer k of
"layers"). This script reads the gauges for those times and prints both against them - per country, and per country
and combination of layers, e.g. "it" against "it+ch" (so a layer that hurts outside its own country shows up, as
Italy's did in Carinthia on 13.11.2026):

  docker exec -w /app deploy-api-eun-1 python check_national_gauges.py [--hours 24]

Gauges (independent of all radar data), each list open data without an account:
  de  DWD 10-minute gauges (check_rain_gauges.stations(), one per ~25 km cell)
  at  GeoSphere Austria TAWES 10-minute gauges (CC BY) - the Swiss and German layers reach into Austria
  ch  MeteoSwiss automatic stations and precipitation stations, 10-minute sums (CC BY). Only the file "t_now" is
      read (today since 00:00 UTC; the next one holds the whole year), so Swiss pairs exist for today's images only.
  it  ARPA Lombardia rain gauges (dati.lombardia.it, 10-minute or 1-minute sums, times in UTC+1)
  iv  ARPA Veneto rain gauges (api.arpa.veneto.it, 5-minute sums of the last ~2.5 days per gauge, times in UTC+1)
      More Italian regions are on the list (EVALUATION.md 8b).
New lists are appended to stations.json (the order of the older files stays valid; they are shorter).
Compared per image time T: the radar value at T against the gauge's mean rate over T-10 .. T+10 min (two 10-minute
sums, as in check_opera_gauges.py). Printed per country: the rain amount ratio radar/gauge and CSI / POD / FAR for
"wet" at 0.1, 0.3 and 1 mm/h - for OPERA alone and for the composed picture, on the same pairs.
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
NAMES = {"de": "Germany (DWD gauges)", "at": "Austria (TAWES gauges)", "ch": "Switzerland (MeteoSwiss gauges)",
         "it": "Italy, Lombardy (ARPA Lombardia gauges)", "iv": "Italy, Veneto (ARPAV gauges)"}


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


LISTS = {"de": _list_de, "at": _list_at, "ch": _list_ch, "it": _list_it, "iv": _list_iv}


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


def record(ts, opera, nat, who=None, layers=()):
    os.makedirs(OBS_DIR, exist_ok=True)
    extra = {} if who is None else {"who": who.astype(np.uint8), "layers": np.array(list(layers))}
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


GAUGES = {"de": gauges_de, "at": gauges_at, "ch": gauges_ch, "it": gauges_it, "iv": gauges_iv}


def table(name, g, o, n):
    g, o, n = (np.array(a, np.float32) for a in (g, o, n))
    print(f"\n{name}: {len(g)} pairs, gauge mean {g.mean():.3f} mm/h" if len(g) else f"\n{name}: no pairs")
    if not len(g):
        return
    for label, q in (("OPERA alone", o), ("own picture", n)):
        cells = []
        for th in (0.1, 0.3, 1.0):
            wo, wf = g >= th, q >= th
            h, m, f = (wo & wf).sum(), (wo & ~wf).sum(), (~wo & wf).sum()
            cells.append(f"{th}: CSI {h / max(1, h + m + f):.2f} POD {h / max(1, h + m):.2f} FAR {f / max(1, h + f):.2f}")
        print(f"  {label:12s} amount x{q.sum() / max(g.sum(), 1e-9):.2f}   " + "   ".join(cells))


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
    pairs = {c: ([], [], []) for c in GAUGES}
    by_layer = {}  # (country, layer or "no layer") -> pairs
    for fp in files:
        t = int(os.path.basename(fp)[:-4])
        z = np.load(fp)
        o, n = z["opera"].astype(np.float32), z["nat"].astype(np.float32)
        who = z["who"] if "who" in z.files else None
        layers = [str(x) for x in z["layers"]] if who is not None else []
        for k, (c, sid, _, _) in enumerate(st[:len(o)]):  # files from before a list was added are shorter
            g = truth[c].get(sid, {})
            v = [g.get(t), g.get(t + 600_000)]
            if None in v or not np.isfinite(o[k]) or not np.isfinite(n[k]):
                continue
            row = (np.mean(v) * 6, o[k], n[k])
            keys = [pairs[c]]
            if who is not None:
                names = "+".join(l for b, l in enumerate(layers) if who[k] >> b & 1)
                if names:
                    keys.append(by_layer.setdefault((c, names), ([], [], [])))
            for p in keys:
                for a, x in zip(p, row):
                    a.append(x)
    for c in GAUGES:
        table(NAMES[c], *pairs[c])
    print("\n--- by the layers that count at the gauge (at least half their weight; 50 pairs or more) ---")
    for (c, l), p in sorted(by_layer.items()):
        if len(p[0]) >= 50:
            table(f"{NAMES[c]} - {l}", *p)


if __name__ == "__main__":
    main()
