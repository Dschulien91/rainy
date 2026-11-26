"""Build DATA_DIR/areas.sqlite for server/areas.py (the blue dot's town as an area, app v1.63).

Source: Overture Maps "divisions / division_area" (OpenStreetMap boundaries, ODbL), the GeoParquet files of one
release downloaded once (public bucket, no account):
  https://overturemaps-us-west-2.s3.amazonaws.com/release/<release>/theme=divisions/type=division_area/part-*.parquet
(~4.6 GB for 2026-11-18.0). Kept: subtypes locality and localadmin inside the app's areas (Europe, North America), and
county / region areas up to MAX_KM2 (big towns are often no locality there: Munich and Cologne are counties, Berlin a
region - their localities are the boroughs),
simplified to SIMPLIFY degrees (~20 m), coordinates to 5 decimals, names: the primary one plus German and English.

Usage: python tools/city_areas.py <folder with the part-*.parquet files> <out.sqlite>
Then: scp to the server and `docker compose cp out.sqlite api:/app/data/areas.sqlite` (the push-data volume).
Needs duckdb (with its spatial extension).
"""
import json
import math
import os
import sqlite3
import sys
import time

import duckdb

SIMPLIFY = 0.0002
MAX_KM2 = 3000.0  # county / region: a town-sized one only
BOXES = [(-25, 34, 45, 72), (-170, 14, -50, 72)]  # Europe, North America (lon0, lat0, lon1, lat1)


def main(src, out):
    t = time.time()
    con = duckdb.connect()
    con.execute("INSTALL spatial; LOAD spatial;")
    where = " OR ".join(f"(bbox.xmin >= {a} AND bbox.xmax <= {c} AND bbox.ymin >= {b} AND bbox.ymax <= {d})"
                        for a, b, c, d in BOXES)
    cur = con.execute(f"""
        SELECT subtype, names.primary, names.common['de'], names.common['en'], country,
               bbox.xmin, bbox.xmax, bbox.ymin, bbox.ymax, ST_Area(geometry),
               ST_AsGeoJSON(ST_SimplifyPreserveTopology(geometry, {SIMPLIFY}))
        FROM read_parquet('{src.replace(os.sep, '/')}/part-*.parquet')
        WHERE subtype IN ('locality', 'localadmin', 'county', 'region') AND ({where})
          AND ST_GeometryType(geometry) IN ('POLYGON', 'MULTIPOLYGON')""")
    if os.path.exists(out):
        os.remove(out)
    db = sqlite3.connect(out)
    db.execute("CREATE TABLE area (id INTEGER PRIMARY KEY, subtype TEXT, name TEXT, names TEXT, country TEXT, km2 REAL, geom TEXT)")
    db.execute("CREATE VIRTUAL TABLE area_idx USING rtree(id, xmin, xmax, ymin, ymax)")
    n = 0
    while True:
        rows = cur.fetchmany(5000)
        if not rows:
            break
        for subtype, name, de, en, country, x0, x1, y0, y1, deg2, gj in rows:
            if not gj or not name:
                continue
            g = json.loads(gj)
            if g["type"] not in ("Polygon", "MultiPolygon"):
                continue
            rnd = lambda ring: [[round(x, 5), round(y, 5)] for x, y in ring]
            if g["type"] == "Polygon":
                g["coordinates"] = [rnd(r) for r in g["coordinates"]]
            else:
                g["coordinates"] = [[rnd(r) for r in p] for p in g["coordinates"]]
            km2 = deg2 * 111.32 ** 2 * math.cos(math.radians((y0 + y1) / 2))
            if subtype in ('county', 'region') and km2 > MAX_KM2:
                continue
            n += 1
            db.execute("INSERT INTO area VALUES (?, ?, ?, ?, ?, ?, ?)",
                       (n, subtype, name, json.dumps(sorted({v for v in (de, en) if v}), ensure_ascii=False), country,
                        round(km2, 3), json.dumps(g, separators=(",", ":"))))
            db.execute("INSERT INTO area_idx VALUES (?, ?, ?, ?, ?)", (n, x0, x1, y0, y1))
        db.commit()
        print(f"{n} areas, {time.time() - t:.0f} s", flush=True)
    db.execute("VACUUM")
    db.close()
    print(f"done: {n} areas, {os.path.getsize(out) / 1e6:.0f} MB, {time.time() - t:.0f} s")


if __name__ == "__main__":
    main(sys.argv[1], sys.argv[2])
