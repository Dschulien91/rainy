"""Is our own Europe picture (OPERA + national layers) closer to the rain gauges than OPERA alone?

The third instance (EU_NATIONAL=1) notes for every image it composes, at every gauge of the lists below, what OPERA
alone showed there and what the composed picture shows (5 x 5 cells around the gauge, mm/h):
verify_log/eun/natobs/<ts>.npz with "opera" and "nat" in the order of natobs/stations.json (record(), called by
rainy_server.EuropeNational.load). This script reads the gauges for those times and prints both against them:

  docker exec -w /app deploy-api-eun-1 python check_national_gauges.py [--hours 24]

Gauges (independent of all radar data):
  de  DWD 10-minute gauges (check_rain_gauges.stations(), one per ~25 km cell), open data
  at  GeoSphere Austria TAWES 10-minute gauges, open data (CC BY) - the Italy and Switzerland layers reach into Austria
Compared per image time T: the radar value at T against the gauge's mean rate over T-10 .. T+10 min (two 10-minute
sums, as in check_opera_gauges.py). Printed per country: the rain amount ratio radar/gauge and CSI / POD / FAR for
"wet" at 0.1, 0.3 and 1 mm/h - for OPERA alone and for the composed picture, on the same pairs.
"""
import glob
import json
import os
import sys
import time
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime, timezone

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

LOG_DIR = os.environ.get("VERIFY_LOG_DIR", os.path.join(os.path.dirname(os.path.abspath(__file__)), "verify_log"))
OBS_DIR = os.path.join(LOG_DIR, "natobs")
KEEP_DAYS = 4
TAWES = "https://dataset.api.hub.geosphere.at/v1"


# ---------- recording (inside the server) ----------
def station_list(fetch, log):
    """[(country, id, lat, lon)]; kept in natobs/stations.json so every file refers to the same order."""
    path = os.path.join(OBS_DIR, "stations.json")
    try:
        with open(path) as f:
            return [tuple(s) for s in json.load(f)]
    except (OSError, ValueError):
        pass
    out = []
    try:
        import check_rain_gauges as crg
        out += [("de", sid, lat, lon) for sid, lat, lon in crg.stations()]
    except Exception as e:
        log(f"natobs: German gauge list failed: {e}")
    try:
        meta = json.loads(fetch(f"{TAWES}/station/current/tawes-v1-10min/metadata", timeout=60))
        out += [("at", str(s["id"]), float(s["lat"]), float(s["lon"])) for s in meta.get("stations", []) if s.get("is_active")]
    except Exception as e:
        log(f"natobs: Austrian gauge list failed: {e}")
    if len({c for c, *_ in out}) == 2:  # only a complete list is kept
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


def record(ts, opera, nat):
    os.makedirs(OBS_DIR, exist_ok=True)
    np.savez_compressed(os.path.join(OBS_DIR, f"{int(ts)}.npz"), opera=opera.astype(np.float16), nat=nat.astype(np.float16))
    old = time.time() * 1000 - KEEP_DAYS * 86400_000
    for fp in glob.glob(os.path.join(OBS_DIR, "*.npz")):
        if int(os.path.basename(fp)[:-4]) < old:
            os.remove(fp)


# ---------- scoring ----------
def _get(url, timeout=120):
    import urllib.request
    with urllib.request.urlopen(urllib.request.Request(url, headers={"User-Agent": "Mozilla/5.0"}), timeout=timeout) as r:
        return r.read()


def gauge_de(sid, since):
    """{ts: mm in the 10 min ending at ts}. As check_rain_gauges.gauge, but reads only the rows since `since` (that one
    parses ~500 days per gauge: 6 CPU-minutes for 300 gauges) and the "recent" file only when `since` is before today."""
    import io
    import zipfile
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
                vals[int(datetime.strptime(d, "%Y%m%d%H%M").replace(tzinfo=timezone.utc).timestamp() * 1000)] = float(q[k])
    return vals


def gauges_de(ids, since):
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
    truth = {"de": gauges_de([s[1] for s in st if s[0] == "de"], t0 - 3600_000),
             "at": gauges_at([s[1] for s in st if s[0] == "at"], t0 - 1200_000, t1 + 1200_000)}
    pairs = {"de": ([], [], []), "at": ([], [], [])}
    for fp in files:
        t = int(os.path.basename(fp)[:-4])
        z = np.load(fp)
        o, n = z["opera"].astype(np.float32), z["nat"].astype(np.float32)
        for k, (c, sid, _, _) in enumerate(st):
            g = truth[c].get(sid, {})
            v = [g.get(t), g.get(t + 600_000)]
            if None in v or not np.isfinite(o[k]) or not np.isfinite(n[k]):
                continue
            pairs[c][0].append(np.mean(v) * 6)
            pairs[c][1].append(o[k])
            pairs[c][2].append(n[k])
    table("Germany (DWD gauges)", *pairs["de"])
    table("Austria (TAWES gauges)", *pairs["at"])


if __name__ == "__main__":
    main()
