"""Offline: which static clutter spots would clutter.py mask, and is there real rain under them?

  docker exec -w /app deploy-api-eu-1 python check_clutter.py eu|eun|na [--hours 72] [--to YYYYMMDDHHMM]
      [--probe "lat,lon;lat,lon"]

Reads the verification archive of the domain (obs_eu, eun/obs_eu, na/obs_eu: the served picture on clutter.py's
counting grid, every 5 minutes), feeds the images of the window to the same decaying counters as the instance
(clutter.counts_of) and runs clutter.build_mask - the mask an instance bootstrapped at --to would build. Lists every
spot (test, lat lon, cells, wet share when the ring is dry and the number of such images, mean rate over the window
and around it), the gauges within 8 km of a masked cell (gauge sum over the window from gaugetruth vs the picture
there) and, for the record, the Austria box (light rain aloft over Carinthia / Styria is NOT clutter, 13.11.2026).
--to: end of the window (default: the newest image). ~2-4 s per 100 images.
"""
import argparse
import json
import os
import time
from datetime import datetime, timezone

import cv2
import numpy as np

import clutter
import warn_check as wc

ap = argparse.ArgumentParser()
ap.add_argument("dom", choices=("eu", "eun", "na"))
ap.add_argument("--hours", type=float, default=72)
ap.add_argument("--to", default="")
ap.add_argument("--probe", default="", help="lat,lon;lat,lon: the counters near these points")
ap.add_argument("--root", default=os.environ.get("VERIFY_LOG_DIR", "/app/verify_log"))
a = ap.parse_args()
root = a.root
if a.dom != "eu" and root.rstrip("/").endswith("/" + a.dom):  # VERIFY_LOG_DIR of the eun / na instance
    root = os.path.dirname(root.rstrip("/"))
grid = wc.Grid(a.dom, root)
t1 = (datetime.strptime(a.to, "%Y%m%d%H%M").replace(tzinfo=timezone.utc).timestamp() * 1000 if a.to else
      max(t for t, _ in grid.frames(0, 1e15)))
frames = grid.frames(t1 - a.hours * 3600_000, t1)
km = 4.0   # counting grid: 4 km (eu), 0.04 deg (na, ~3.6-4.4 km); EuropeOpera / NorthAmericaMrms KM_PER_PX x 2
t0 = time.time()
c = {k: np.zeros(grid.shape, np.float32) for k in clutter.KEYS}
plain = {k: np.zeros(grid.shape, np.uint32) for k in ("valid", "wet", "strong")}
dec = np.float32(1.0 - 1.0 / clutter.TAU_IMAGES)
n_eff = 0.0
for t, fp in frames:
    try:
        r = np.load(fp)["rate"].astype(np.float32)
    except Exception:
        continue
    for k, x in zip(clutter.KEYS, clutter.counts_of(r, km)):
        c[k] *= dec
        c[k] += x
    n_eff = n_eff * float(dec) + 1.0
    v = r >= 0
    plain["valid"] += v
    plain["wet"] += v & (r >= clutter.WET)
    plain["strong"] += v & (r >= 5.0)
n = len(frames)
mask, blobs = clutter.build_mask(c, km, n_eff)
print(f"{a.dom}: {n} images {datetime.fromtimestamp(frames[0][0] / 1000, timezone.utc):%d.%m. %H:%M} - "
      f"{datetime.fromtimestamp(frames[-1][0] / 1000, timezone.utc):%d.%m. %H:%MZ}, read in {time.time() - t0:.0f} s; "
      f"{len(blobs)} spots, {int(mask.sum())} cells masked (before growing)")
pv = np.maximum(plain["valid"], 1)
for b in sorted(blobs, key=lambda b: -b["mean"]):
    lon, lat = grid.to_lonlat(b["col"], b["row"])
    rr, cc = b["row"], b["col"]
    print(f"  {b['kind']:4s} {float(lat):7.2f} {float(lon):8.2f}  cells {b['px']:3d}  wet|dry ring {b['cond']:.2f} of "
          f"{b['ndry']:4d}  wet all {plain['wet'][rr, cc] / pv[rr, cc]:.2f}  mean {b['mean']:5.2f} mm/h (ring "
          f"{b['ring_mean']:.2f})  >=5 mm/h {plain['strong'][rr, cc] / pv[rr, cc]:.2f}")

# gauges near the mask
if a.dom == "na":
    st = [tuple(s) for s in json.load(open(os.path.join(root, "na", "metar", "stations.json")))]
    tdir = os.path.join(root, "na", "gaugetruth")
else:
    st = [tuple(s) for s in json.load(open(os.path.join(root, "eun", "natobs", "stations.json")))]
    tdir = os.path.join(root, "gaugetruth")
mean = c["rsum"] / np.maximum(c["valid"], 1e-6)
if mask.any() and st:
    near = cv2.dilate(mask.astype(np.uint8), clutter.disk(8.0 / km)) > 0
    lat = np.array([s[2] for s in st])
    lon = np.array([s[3] for s in st])
    cs, rs = grid.to_px(lon, lat)
    cs, rs = np.round(cs).astype(int), np.round(rs).astype(int)
    h, w = grid.shape
    ok = (cs >= 0) & (cs < w) & (rs >= 0) & (rs < h)
    hit = [k for k in np.flatnonzero(ok) if near[rs[k], cs[k]]]
    truth = {}
    for d in range(int((t1 - a.hours * 3600_000) // 86400_000), int(t1 // 86400_000) + 1):
        fp = os.path.join(tdir, time.strftime("%Y%m%d", time.gmtime(d * 86400)) + ".npz")
        if not os.path.exists(fp):
            continue
        z = np.load(fp)
        tt = d * 86400_000 + (np.arange(144) + 1) * 600_000
        sel = (tt > t1 - a.hours * 3600_000) & (tt <= t1)
        for gid, mm in zip(z["ids"], z["mm"]):
            tot = truth.setdefault(str(gid), [0.0, 0])
            tot[0] += float(np.nansum(mm[sel]))
            tot[1] += int(np.isfinite(mm[sel]).sum())
    print(f"gauges within 8 km of a masked cell: {len(hit)}")
    for k in hit:
        gid = f"{st[k][0]}:{st[k][1]}"
        g = truth.get(gid)
        print(f"  {gid:16s} {st[k][2]:7.3f} {st[k][3]:8.3f}  gauge {('%.1f mm (%d values)' % tuple(g)) if g else 'no data'}"
              f"  picture there ~{float(mean[rs[k], cs[k]]) * a.hours:.1f} mm, masked {bool(mask[rs[k], cs[k]])}")

if a.probe:   # why a known spot is (not) masked: the cell with the highest mean rate within 8 km of each point
    for p in a.probe.split(";"):
        la, lo = (float(x) for x in p.split(","))
        cc, rr = grid.to_px(lo, la)
        cc, rr = int(round(float(cc))), int(round(float(rr)))
        d = int(np.ceil(8 / km))
        if not (d <= rr < grid.shape[0] - d and d <= cc < grid.shape[1] - d):
            print(f"probe {la:.2f} {lo:.2f}: outside the grid")
            continue
        sub = mean[rr - d:rr + d + 1, cc - d:cc + d + 1]
        j = np.unravel_index(int(np.argmax(sub)), sub.shape)
        r_, c_ = rr - d + j[0], cc - d + j[1]
        k = clutter.ring_kernel(km)
        ring = cv2.filter2D(mean, -1, k)[r_, c_] / k.sum()
        print(f"probe {la:.2f} {lo:.2f}: wet all {plain['wet'][r_, c_] / pv[r_, c_]:.2f}, wet|dry ring "
              f"{c['dwet'][r_, c_] / max(c['dvalid'][r_, c_], 1e-6):.2f} of {c['dvalid'][r_, c_]:.0f}, mean "
              f"{mean[r_, c_]:.2f} mm/h (ring ~{ring:.2f}), masked {bool(mask[r_, c_])}")

if a.dom in ("eu", "eun"):
    lat_, lon_ = (46.3, 49.1), (9.5, 17.2)
    cs, rs = grid.to_px(np.array([lon_[0], lon_[1]]), np.array([lat_[1], lat_[0]]))
    sub = (slice(int(rs[0]), int(rs[1]) + 1), slice(int(cs[0]), int(cs[1]) + 1))
    f = (plain["wet"][sub] / pv[sub]).astype(np.float32)
    print(f"Austria box: max wet share {f.max():.2f}, cells wet in >= 40 %: {int((f >= 0.4).sum())}, "
          f"masked cells there: {int(mask[sub].sum())}")
