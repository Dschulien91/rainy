"""Our own Europe radar picture: the European composite (EUMETNET OPERA) with national radar products laid on top.

Why (user, 13.11.2026): OPERA has holes - measured on 13.11.: southern Italy 12 % covered, central Italy 79 %,
Sardinia 62 %, Greece 0 % - and it is one uniform 2 km product, while several weather services publish their own
national composite: finer, gauge-adjusted, with their own radars. The third instance of the server (EU_NATIONAL=1,
behind /eun/) serves this mosaic with the same nowcast, tiles and verification as the OPERA instance, so the two can
be compared number by number and switched in the app.

A layer = one national product: it polls its source, keeps the last ~80 min resampled to the host grid (OPERA's
grid at EU_KM) inside its own window, and says how much it is trusted per pixel:
  weight 1 inside its coverage, falling to 0 over FEATHER_KM towards the edge of its coverage - no hard seams where
  it meets OPERA; where OPERA has nothing, the layer counts alone.
The host region (rainy_server.EuropeNational) waits with a new time step until every layer that is alive has its
image for that step (the sources publish 5-12 min after the measurement), at most WAIT_MIN; a layer whose source is
down is skipped (OPERA alone there) and logged.

Layers:
  it  Italy - Dipartimento della Protezione Civile, Radar-DPC: SRI (surface rain intensity, mm/h), 1 km, every
      5 min, GeoTIFF, transverse Mercator (lat0 42, lon0 12.5, WGS84), no account. NOT gauge-adjusted (their hourly
      product SRT1 is exactly the sum of the SRI images T-60 .. T-5, measured 13.11.) and 1.4-2.8 x too wet at the
      gauges, most of all in light rain: see ItalyDpc.damp().
      Licence CC BY-SA ("citare la fonte Radar-DPC", derived works under the same licence).
      Used only over Italy, the seas around it and western Slovenia (ItalyDpc.AREA, WEST_SLOVENIA since 18.11.2026:
      OPERA has no radar that sees the west of Slovenia well): far from the Italian radars the product is much too
      wet - 13.11.2026, 33 Austrian gauges in Carinthia over 2.5 h: gauges 0.07 mm/h, OPERA 0.32, SRI 2.71 (wet at
      460 of 475 gauge readings, the gauges at 35); Klagenfurt 3.9 mm/h at 20 dBZ maximum reflectivity.
  de  Germany - Deutscher Wetterdienst, open data RV composite: the analysis of each 5-min run (mm per 5 min, 1 km,
      polar stereographic, ODIM HDF5 in a tar with the 24 forecast steps), x GAIN 1.5 as in the Germany edition
      (measured against ~300 DWD gauges, mosaic.GermanyDwd.GAIN_RV). Published ~5 min after the measurement.
  ch  Switzerland - MeteoSwiss open data (CC BY), product RZC: rain rate (mm/h) from the five Swiss radars with
      MeteoSwiss' own corrections for the Alps, 1 km, every 5 min, ODIM HDF5, Swiss oblique Mercator; one STAC item
      per day (data.geo.admin.ch, collection ch.meteoschweiz.ogd-radar-precip), files rzcYYDDDHHMM??.001.h5 stamped
      with the END of the 5 minutes, published ~4 min later. Used only within 160 km of a Swiss radar
      (SwissRzc.RANGE_KM): farther out it misses most of the rain.

Shadow layers (Layer.SHADOW): a candidate that is loaded and noted at the rain gauges (check_national_gauges.py,
tables "shadow <name>") but NOT laid into the picture. Every new country starts like this - Italy's product looked
plausible on the map and was 2.5x too wet at the gauges. A shadow is promoted (SHADOW = False, credit into both
impressum pages) when it beats OPERA at the gauges on several rainy days.
  cz  Czechia - CHMI open data (CC BY 4.0): composite of the two Czech radars, reflectivity at 2 km height (dBZ),
      ~1 km, every 5 min, ODIM HDF5, stamped with the end of the scan and published ~3 min later (much earlier than
      OPERA). Rain rate by Z = 200 R^1.6. OPERA has these radars too: what a national copy gains is open.
  dk  Denmark - DMI open data (CC BY 4.0): composite of the five Danish radars, DBZH, 500 m, every 5 min, ODIM HDF5,
      within 120 km of the radars (DenmarkDmi). Added 22.11.2026 for OPERA's clear-air / bird echoes (a ring over the
      Kattegat on 19.11. that DMI's and SMHI's own composites did not have).
"""
import io
import json
import os
import re
import tarfile
import threading
import time
import urllib.error
import urllib.request
from datetime import datetime, timedelta, timezone

import cv2
import h5py
import numpy as np
from PIL import Image, ImageDraw
from pyproj import Transformer

import mosaic

STEP_MS = 5 * 60_000
KEEP_MS = 80 * 60_000
FEATHER_KM = 30.0
WAIT_MIN = 20          # a time step is composed without a layer's image after this long
ALIVE_MIN = 30         # a layer that delivered within this time is waited for
NEAR_MS = 10 * 60_000  # without the exact image: the nearest one within this distance
# DROPPED-OUT RADAR (server-v1.96, the whole area v1.97, cleared as a whole v1.98): 25.11.2026 04:25 UTC Radar-DPC's SRI lost the rain of South Tyrol and Trentino at
# once - 13 boxes of 20 km around 46.46 N 11.21 E (the Monte Macaion radar) went from rain to dry in one image, while
# the 60 South Tyrol and 96 Trentino gauges kept measuring rain (wet at 12-15 / 33-42 of them, more than before) and
# OPERA sees nothing there either. The product reports a radar that is out as "no rain", not as "no data", so the map
# showed a large rain area vanishing (user, 25.11.: "a huge cloud over Bozen"). In the hours before, at most 2 boxes
# turned dry from one image to the next. Now: at least COLLAPSE_MIN boxes that turn from wet (> 30 % of the box) to
# dry (< 2 %) in one image are taken as a radar that dropped out; they stay "no data" in the picture (bridged with the
# forecast for a few steps, then hatched, as an OPERA radar that is out) until rain shows up in a good part of them again (the radar
# is back) or for DEAD_KEEP_MS at most.
COLLAPSE_KM = 20.0
COLLAPSE_MIN = 6
# server-v2.00: 24 h (was 6). Macaion was still out 5.5 h after it dropped (25.11.2026 09:30-09:40 UTC: 16-31 of 96
# Trentino gauges wet, no rain in the dead boxes); at 6 h its area would have shown "dry" again while it rained. A
# radar that is back clears the area as soon as it rains there; hatching a dry area for longer is the smaller harm.
DEAD_KEEP_MS = 24 * 3600_000


def dead_outline(shape, b, boxes):
    """The no-data mask of a dropped-out radar from its dead COLLAPSE_KM boxes, with a rounded outline (server-v2.01).
    The plain union of the boxes put a staircase of 20 km blocks on the map (25.11.2026, South Tyrol / Trentino: a
    hatched block with dead-straight east-west edges at 46.11 / 46.30 N and north-south ones at 11.37 / 11.89 E, and
    the rain of the neighbouring radars cut along them in every step, now and forecast). A radar's coverage is round:
    one-box notches are closed (a disc of 1.5 boxes), then the outline is rounded (blur of 2/3 of a box, cut at 0.4).
    v2.04: v2.01's disc of one box and blur of 1/3 box still left the 20 km steps visible along a diagonal (the
    Sardinian drop-out of 25.11. 09:55Z); offline on that day's dead boxes this one is a smooth footprint that grows
    the area by ~30 % (up to about a box beyond the union: hatched, i.e. no data - the safe side) and leaves ~3 % of
    it out (box corners)."""
    m = np.zeros(shape, np.uint8)
    for r, c in boxes:
        m[r * b:(r + 1) * b, c * b:(c + 1) * b] = 1
    if not m.any():
        return m.astype(bool)
    r = int(round(1.5 * b))
    k = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (2 * r + 1, 2 * r + 1))
    pad = 2 * r   # closing must not stick to the window's edge
    mp = cv2.morphologyEx(np.pad(m, pad), cv2.MORPH_CLOSE, k)[pad:-pad, pad:-pad]
    sm = cv2.GaussianBlur(mp.astype(np.float32), (0, 0), sigmaX=2.0 * b / 3.0, borderType=cv2.BORDER_REPLICATE)
    return sm >= 0.4


class Layer:
    name, credit = "", ""
    SHIFT_MS = 0        # an image stamped T belongs to the host's time step T + SHIFT_MS (see timing())
    PROJ = ""           # proj string of the product's grid
    X0, Y1, CELL, NX, NY = 0.0, 0.0, 1000.0, 0, 0   # top-left corner, cell size (m), size
    AREA = None         # [(lon, lat), ...]: the product is used only inside this polygon (None: everywhere it has data)
    RADARS, RANGE_KM = None, 0.0   # [(lat, lon), ...]: the product is used only within RANGE_KM of one of its radars
    SHADOW = False      # True: loaded and noted at the gauges, not laid into the picture (a candidate under test)

    def __init__(self, host, log, ua):
        self.host, self.log, self.ua = host, log, ua
        self.lock = threading.Lock()
        self.frames = {}        # ts -> (rate on the host window, float16, < 0 = no data; weight float16)
        self.dead = {}          # (box row, box col) -> step since when a dropped-out radar's box shows no rain
        self._load_dead()
        self.blind = {}         # ts -> bool mask on the host window (dead boxes in that image) or None
        self.first_poll_done = False
        self.failed = 0.0
        self.geometry()
        self._window()

    def geometry(self):
        """Products that describe their own grid set PROJ, X0, Y1, CELL, NX, NY here (from a first file)."""

    # ---------- geometry ----------
    def _window(self):
        """The part of the host grid this product covers, and where each of its pixels lies in the product."""
        to_ll = Transformer.from_crs(self.PROJ, "EPSG:4326", always_xy=True)
        xs = self.X0 + np.array([0, self.NX, self.NX, 0, self.NX / 2, self.NX / 2, 0, self.NX]) * self.CELL
        ys = self.Y1 - np.array([0, 0, self.NY, self.NY, 0, self.NY, self.NY / 2, self.NY / 2]) * self.CELL
        lon, lat = to_ll.transform(xs, ys)
        m = Transformer.from_crs("EPSG:4326", "EPSG:3857", always_xy=True)
        col, row = self.host.to_pixels(*m.transform(lon, lat))
        h = self.host
        c0, c1 = max(int(np.floor(col.min())), 0), min(int(np.ceil(col.max())) + 1, h.NX)
        r0, r1 = max(int(np.floor(row.min())), 0), min(int(np.ceil(row.max())) + 1, h.NY)
        self.win = (slice(r0, r1), slice(c0, c1))
        cc, rr = np.meshgrid(np.arange(c0, c1), np.arange(r0, r1))
        lon, lat = h.pixel_lonlat(cc, rr)
        x, y = Transformer.from_crs("EPSG:4326", self.PROJ, always_xy=True).transform(lon, lat)
        self.map_col = ((np.asarray(x) - self.X0) / self.CELL - 0.5).astype(np.float32)
        self.map_row = ((self.Y1 - np.asarray(y)) / self.CELL - 0.5).astype(np.float32)
        self.k = max(1, int(round(h.KM_PER_PX * 1000 / self.CELL)))  # product pixels per host pixel
        self.area = None
        if self.AREA:
            x, y = Transformer.from_crs("EPSG:4326", self.PROJ, always_xy=True).transform(*zip(*self.AREA))
            im = Image.new("L", (self.NX, self.NY), 0)
            ImageDraw.Draw(im).polygon([((a - self.X0) / self.CELL, (self.Y1 - b) / self.CELL) for a, b in zip(x, y)], fill=1)
            self.area = np.asarray(im).astype(bool)
        if self.RADARS:
            x, y = Transformer.from_crs("EPSG:4326", self.PROJ, always_xy=True).transform(
                [b for _, b in self.RADARS], [a for a, _ in self.RADARS])
            xs = self.X0 + (np.arange(self.NX) + 0.5) * self.CELL
            ys = self.Y1 - (np.arange(self.NY) + 0.5) * self.CELL
            near = np.zeros((self.NY, self.NX), bool)
            for a, b in zip(x, y):
                near |= (xs[None, :] - a) ** 2 + (ys[:, None] - b) ** 2 <= (self.RANGE_KM * 1000.0) ** 2
            self.area = near if self.area is None else self.area & near

    def to_host(self, rate):
        """Product grid (mm/h, < 0 = no data) -> (rate, weight) on the host window. Mean over the product pixels of a
        host cell (box filter before the bilinear lookup), no data where less than half of them have data."""
        if self.area is not None:
            rate = np.where(self.area, rate, -1.0)
        ok = (rate >= 0).astype(np.float32)
        val = np.where(rate >= 0, rate, 0).astype(np.float32)
        if self.k > 1:
            ok, val = cv2.blur(ok, (self.k, self.k)), cv2.blur(val, (self.k, self.k))
        kw = dict(interpolation=cv2.INTER_LINEAR, borderMode=cv2.BORDER_CONSTANT, borderValue=0)
        cov = cv2.remap(ok, self.map_col, self.map_row, **kw)
        v = cv2.remap(val, self.map_col, self.map_row, **kw)
        inside = cov >= 0.5
        out = np.where(inside, v / np.maximum(cov, 1e-6), -1.0).astype(np.float32)
        dist = cv2.distanceTransform(inside.astype(np.uint8), cv2.DIST_L2, 3) * self.host.KM_PER_PX
        return out.astype(np.float16), np.clip(dist / FEATHER_KM, 0, 1).astype(np.float16)

    # ---------- source (subclasses) ----------
    def newest(self):
        """Time (ms) of the newest image the source has."""
        raise NotImplementedError

    def load(self, ts):
        """The product grid (float32 mm/h, < 0 = no data) of the image at ts; raises when it does not exist."""
        raise NotImplementedError

    def _get(self, url, data=None, timeout=40):
        req = urllib.request.Request(url, data=data, headers={"User-Agent": self.ua, **(
            {"Content-Type": "application/json"} if data else {})})
        with urllib.request.urlopen(req, timeout=timeout) as r:
            return r.read()

    # ---------- store ----------
    def poll(self):
        newest = self.newest()
        now = time.time() * 1000
        want = [t for t in range(newest - KEEP_MS + STEP_MS, newest + 1, STEP_MS) if t not in self.frames and t > now - KEEP_MS]
        got = 0
        for t in sorted(want, reverse=True):  # the newest first: the map waits for it
            try:
                frame = self.to_host(self.load(t))
            except Exception:
                if t == newest:
                    raise
                frame = None  # an older step the source does not have (any more): not asked again
            with self.lock:
                if frame is not None:
                    self.frames[t] = frame
                    got += 1
                else:
                    self.frames.setdefault(t, None)  # not asked again
        with self.lock:
            for t in [t for t in self.frames if t < now - KEEP_MS]:
                del self.frames[t]
            for t in [t for t in self.blind if t not in self.frames]:
                del self.blind[t]
        if got:
            self.watch()
        return got

    def _boxes(self, rate):
        """Wet share (> 0.1 mm/h) and data share of the COLLAPSE_KM boxes of a host-window image."""
        b = max(1, int(round(COLLAPSE_KM / self.host.KM_PER_PX)))
        h, w = rate.shape[0] // b, rate.shape[1] // b
        a = np.asarray(rate[:h * b, :w * b], np.float32).reshape(h, b, w, b)
        return (a > 0.1).mean(axis=(1, 3)), (a >= 0).mean(axis=(1, 3)), b

    def watch(self):
        """Dropped-out radar (see COLLAPSE_MIN): the images not looked at yet, oldest first."""
        with self.lock:
            todo = sorted(t for t, f in self.frames.items() if f is not None and t not in self.blind)
        for t in todo:
            with self.lock:
                f = self.frames.get(t)
                prev = next((self.frames[t - k * STEP_MS] for k in (1, 2) if self.frames.get(t - k * STEP_MS) is not None), None)
            if f is None:
                continue
            wet, cov, b = self._boxes(f[0])
            before = dict(self.dead)
            self.dead = {k: v for k, v in self.dead.items() if k[0] < wet.shape[0] and k[1] < wet.shape[1]}
            # back: rain in a good part of the area again (v1.97: not box by box - a neighbouring radar's bit of rain
            # in one box at the edge of its range cleared that box, and Bozen showed "dry" in the middle of the hole)
            # server-v2.03: judged per drop-out (its boxes share one "since") and only from images after it. Before,
            # all dead boxes were judged together: at the restart of 25.11.2026 10:22Z the replay of the 09:05Z image
            # found the rain of a Sardinian band that dropped out only at 09:55Z inside "the dead area" and cleared
            # Macaion's area too, so Bozen showed "dry" again while that radar was still out.
            groups = {}
            for k, since in self.dead.items():
                groups.setdefault(since, []).append(k)
            for since, ks in groups.items():
                if t > since and (np.mean([wet[k] >= 0.1 for k in ks]) >= 0.3 or t - since > DEAD_KEEP_MS):
                    for k in ks:
                        del self.dead[k]
                    self.log(f"layer {self.name}: rain shows again where a radar had dropped out at "
                             f"{datetime.fromtimestamp(since / 1000, timezone.utc):%d.%m. %H:%MZ} ({len(ks)} boxes, "
                             f"{datetime.fromtimestamp(t / 1000, timezone.utc):%H:%MZ})")
            if prev is not None:
                wet0, cov0, _ = self._boxes(prev[0])
                gone = (wet0 > 0.3) & (wet < 0.02) & (cov >= 0.5) & (cov0 >= 0.5)
                if gone.sum() >= COLLAPSE_MIN:
                    r, c = np.nonzero(gone)
                    # the radar's whole area, not only where it rained just before: the dry boxes inside the hull of
                    # the vanished ones (25.11.: Bozen itself was dry at 04:20 and kept showing "dry" in the hole)
                    hull = np.zeros(wet.shape, np.uint8)
                    cv2.fillConvexPoly(hull, cv2.convexHull(np.stack([c, r], 1).astype(np.int32)), 1)
                    rr, cc = np.nonzero((hull > 0) & (wet < 0.1) & (cov >= 0.5))
                    for k in zip(rr.tolist(), cc.tolist()):
                        self.dead.setdefault(k, t)
                    lon, lat = self.host.pixel_lonlat(self.win[1].start + (c.mean() + 0.5) * b,
                                                      self.win[0].start + (r.mean() + 0.5) * b)
                    self.log(f"layer {self.name}: the rain of {len(r)} boxes of {COLLAPSE_KM:.0f} km ({len(rr)} with the dry "
                             f"ones between them) around "
                             f"{float(lat):.2f} N {float(lon):.2f} E vanished at once at "
                             f"{datetime.fromtimestamp(t / 1000, timezone.utc):%H:%MZ} (a radar dropped out) - "
                             f"no data there until rain shows again")
            if self.dead != before:
                self._save_dead()
            mask = None
            if any(since <= t for since in self.dead.values()):  # not the images before it dropped out
                mask = dead_outline(f[0].shape, b, [k for k, since in self.dead.items() if since <= t])
            with self.lock:
                self.blind[t] = mask

    # the dead boxes survive a restart (server-v1.99): the layer reloads only its last KEEP_MS, so the image before the
    # drop-out - and with it the drop-out - is gone after a restart 80 min later
    def _dead_file(self):
        return os.path.join(os.environ.get("VERIFY_LOG_DIR", "."), f"layer_dead_{self.name}.json")

    def _load_dead(self):
        try:
            with open(self._dead_file()) as f:
                rows = json.load(f)
            now = time.time() * 1000
            self.dead = {(int(r), int(c)): int(t) for r, c, t in rows if now - t < DEAD_KEEP_MS}
            if self.dead:
                self.log(f"layer {self.name}: {len(self.dead)} boxes of a dropped-out radar still without data (from before the restart)")
        except (OSError, ValueError, TypeError):
            self.dead = {}

    def _save_dead(self):
        try:
            with open(self._dead_file() + ".tmp", "w") as f:
                json.dump([[r, c, t] for (r, c), t in self.dead.items()], f)
            os.replace(self._dead_file() + ".tmp", self._dead_file())
        except OSError as e:
            self.log(f"layer {self.name}: dead boxes not saved ({e})")

    def blind_at(self, ts):
        """The dead-box mask of the image get(ts) answers with (None: none)."""
        ts -= self.SHIFT_MS
        with self.lock:
            ok = [t for t, f in self.frames.items() if f is not None and abs(t - ts) <= NEAR_MS]
            return self.blind.get(min(ok, key=lambda t: abs(t - ts))) if ok else None

    def loop(self):
        while True:
            try:
                n = self.poll()
                if n:
                    with self.lock:
                        last = max(t for t, f in self.frames.items() if f is not None)
                    self.log(f"[{datetime.now():%H:%M:%S}] layer {self.name}: +{n} images, latest "
                             f"{datetime.fromtimestamp(last / 1000, timezone.utc):%H:%MZ}")
            except Exception as e:
                if time.time() - self.failed > 600:
                    self.failed = time.time()
                    self.log(f"layer {self.name} failed: {e}")
            self.first_poll_done = True
            time.sleep(30)

    def newest_ts(self):
        with self.lock:
            return max((t for t, f in self.frames.items() if f is not None), default=None)

    def alive(self):
        t = self.newest_ts()
        return t is not None and time.time() * 1000 - t < ALIVE_MIN * 60_000

    def has(self, ts):
        with self.lock:
            return self.frames.get(ts - self.SHIFT_MS) is not None

    def get(self, ts):
        """(rate, weight) of the image at ts, else of the nearest one within NEAR_MS, else None. A shadow layer is
        not waited for, so it answers only with the exact image (an older one would be scored unfairly)."""
        ts -= self.SHIFT_MS
        with self.lock:
            if self.SHADOW:
                return self.frames.get(ts)
            ok = [t for t, f in self.frames.items() if f is not None and abs(t - ts) <= NEAR_MS]
            return self.frames[min(ok, key=lambda t: abs(t - ts))] if ok else None



def compose(grid, layers, ts):
    """Lay the layers' images for the step ts over `grid` (host grid, float32, < 0 = no data), in place; returns the
    names of the layers that had an image. Where layers overlap each counts with its weight (the one whose coverage
    edge is farther away counts more); what is left of the weight (1 - sum, near the edges) goes to OPERA, and where
    OPERA has nothing the layers count alone."""
    num = np.zeros(grid.shape, np.float32)
    den = np.zeros(grid.shape, np.float32)
    used = []
    for l in layers:
        f = None if l.SHADOW else l.get(ts)
        if f is None:
            continue
        rate, w = (a.astype(np.float32) for a in f)
        num[l.win] += w * np.maximum(rate, 0)
        den[l.win] += w
        used.append(l.name)
    if used:
        have, base = den > 0, grid >= 0
        mean = num / np.maximum(den, 1e-6)
        rest = np.clip(1 - den, 0, 1)
        grid[...] = np.where(have & base, (1 - rest) * mean + rest * grid, np.where(have, mean, grid))
    for l in layers:  # a dropped-out radar's area: no data, whatever OPERA or a neighbour shows there (COLLAPSE_MIN)
        m = None if l.SHADOW else l.blind_at(ts)
        if m is not None:
            grid[l.win][m] = -1.0
    return used


def cover_at(layers, ts, col, row):
    """Per point (host pixel col, row): bit k set where layer k counts with at least half its weight in the image
    for the step ts (the gauge check splits its tables by this)."""
    out = np.zeros(len(col), np.uint8)
    for k, l in enumerate(layers):
        f = l.get(ts)
        if f is None:
            continue
        w = f[1]
        r, c = row - l.win[0].start, col - l.win[1].start
        ok = (r >= 0) & (r < w.shape[0]) & (c >= 0) & (c < w.shape[1])
        out[ok] |= (w[r[ok], c[ok]] >= 0.5).astype(np.uint8) << k
    return out


def values_at(layers, ts, col, row):
    """Per layer and point (host pixel col, row): what that layer alone shows there in the image for the step ts
    (mean of the 5 x 5 cells around the point, as check_national_gauges.at_gauges); NaN without an image or data.
    The gauge check scores every layer on its own with this - inside and outside its country."""
    out = np.full((len(layers), len(col)), np.nan, np.float32)
    for k, l in enumerate(layers):
        f = l.get(ts)
        if f is None:
            continue
        rate = f[0]
        r, c = row - l.win[0].start, col - l.win[1].start
        for i in np.flatnonzero((r >= 2) & (r < rate.shape[0] - 2) & (c >= 2) & (c < rate.shape[1] - 2)):
            box = rate[r[i] - 2:r[i] + 3, c[i] - 2:c[i] + 3]
            if (box >= 0).all():
                out[k, i] = box.astype(np.float32).mean()
    return out


class ItalyDpc(Layer):
    name, credit = "it", "Radar-DPC (Dipartimento della Protezione Civile, CC BY-SA)"
    API = "https://radar-api.protezionecivile.it"
    PROJ = "+proj=tmerc +lat_0=42 +lon_0=12.5 +k=1 +x_0=0 +y_0=0 +ellps=WGS84 +units=m"
    X0, Y1, CELL, NX, NY = -600000.0, 650000.0, 1000.0, 1200, 1400
    PRODUCT = "SRI"
    # Italy's borders (to ~10 km, the layer fades out over FEATHER_KM inside them anyway) and the seas around it up to
    # about the middle towards the neighbours; Corsica is left to OPERA (French radar).
    AREA = [(7.53, 43.78), (7.70, 44.10), (7.20, 44.20), (6.90, 44.40), (6.98, 44.68), (6.72, 44.93), (6.63, 45.11),
            (6.90, 45.22), (7.10, 45.45), (6.85, 45.68), (6.86, 45.83), (7.04, 45.92), (7.17, 45.87), (7.66, 45.98),
            (7.87, 45.93), (8.12, 46.22), (8.38, 46.45), (8.45, 46.25), (8.60, 46.12), (8.85, 46.00), (8.95, 45.83),
            (9.08, 45.90), (9.02, 46.05), (9.17, 46.17), (9.28, 46.50), (9.45, 46.50), (9.58, 46.29), (9.95, 46.38),
            (10.10, 46.23), (10.17, 46.40), (10.08, 46.60), (10.45, 46.55), (10.47, 46.86), (10.75, 46.80),
            (11.05, 46.80), (11.15, 46.95), (11.50, 47.00), (12.13, 47.09), (12.22, 46.88), (12.40, 46.70),
            (12.75, 46.64), (13.20, 46.59), (13.72, 46.52)] + [
            # the Slovenian border (used without WEST_SLOVENIA)
            (13.66, 46.44), (13.40, 46.32), (13.48, 46.20), (13.52, 46.05), (13.64, 45.95), (13.60, 45.80),
            (13.80, 45.73), (13.92, 45.63), (13.72, 45.60)] + [
            (13.40, 45.50), (13.05, 45.10), (13.30, 44.50), (14.30, 43.60), (15.40, 43.00), (17.00, 42.10),
            (18.30, 41.20), (18.95, 40.30), (19.20, 39.40), (19.60, 34.50), (11.50, 34.50), (11.50, 37.30),
            (10.20, 37.90), (8.00, 38.20), (7.30, 38.40), (7.30, 41.20), (8.60, 41.33), (9.26, 41.31), (9.60, 41.36),
            (9.75, 41.60), (9.75, 42.90), (9.30, 43.35), (8.50, 43.50), (7.60, 43.60)]
    # WESTERN SLOVENIA (18.11.2026): OPERA has only the eastern Slovenian radar (Lisca; Pasja ravan has not been in
    # the composite's node list on any day 16.-18.11., nor in ARSO's own composite, which lists SI1 only) and no
    # Italian radar, so the west of Slovenia is seen from 100+ km away: 17.11., 09-20 UTC, Otlica gauge 54 mm, OPERA
    # 6.6, Nova Gorica 16 / 1.8. DPC's SRI holds the Friuli radar ~40 km from there. 16.-18.11., 19 ARSO gauges, hourly
    # sums CSI at 0.2 / 1 / 3 mm: OPERA 0.45/0.32/0.00 x0.38 -> with this area 0.45/0.49/0.19 x0.50 (the 5 western
    # gauges 0.46/0.28/0.00 x0.15 -> 0.46/0.50/0.28 x0.32; the SRI without damp() would give 0.67/0.73/0.58 x0.66 -
    # damp() reads NE Italy and Slovenia much too dry in these days' light rain: NE Veneto gauges raw x0.87, damped
    # x0.08). The area ends at 14.6 E, at 45.2 N in Istria and at the Austrian border (DPC is far too wet beyond the Karawanks: x100 at
    # Carinthian gauges). WEST_SLOVENIA = False restores the Italian border.
    WEST_SLOVENIA = True
    AREA_WEST_SLOVENIA = [(14.10, 46.47), (14.55, 46.40), (14.60, 46.38), (14.60, 45.50), (14.35, 45.35),
                          (13.90, 45.20), (13.50, 45.20)]   # with northern Istria: the coast keeps its full weight
    if WEST_SLOVENIA:
        AREA = AREA[:AREA.index((13.66, 46.44))] + AREA_WEST_SLOVENIA + AREA[AREA.index((13.72, 45.60)) + 1:]
    # The image stamped T is the scan T .. T+5 (DWD, MeteoSwiss and OPERA stamp the end): DPC's own hourly sum SRT1 at T
    # is exactly the sum of the SRI images T-60 .. T-5 (13.11.: ratio 1.00, correlation 1.00 over 93 000 wet cells),
    # the file is written ~7 min after its stamp, and timing() against OPERA is best at +5 (0.444/0.454/0.449 at
    # 0/+5/+10, slow rain).
    SHIFT_MS = STEP_MS
    # damp(): SRI raw reads far more than the rain gauges everywhere it was measured - 13.11.2026, hourly sums, 8 h:
    # 835 pairs at ARPA Veneto gauges (steady light rain, gauge mean 0.40 mm/h): raw = 2.8 x the gauges; 676 pairs at
    # ARPAE Emilia-Romagna gauges (heavy rain at the Apennines, 2.24 mm/h): raw = 1.44 x (other radars, other
    # calibration - one curve cannot be right for both). OPERA as the Europe instance shows it: 0.24 x / 0.10 x.
    # r * min(DAMP_C, DAMP_A r^DAMP_B) with (0.2, 0.8, 0.7) is the best of ~1900 curves for both regions together
    # (check_italy_gain.py; also with a taper above 6-24 mm/h, which changed nothing): amount x1.01 (Veneto x1.40,
    # Emilia x0.92), CSI at 0.2/1/3/8 mm per hour 0.67/0.72/0.62/0.25 (Veneto 0.51/0.59/0.42, Emilia 0.79/0.77/0.67/
    # 0.25); raw 0.63/0.69/0.61/0.17, OPERA 0.44/0.09/0.02/0.00. The first fit (Veneto alone, 4 h: 0.25, 0.4, no cap)
    # read 1-12 mm/h too low in Emilia (CSI at 3 mm 0.54) and passed everything above 32 mm/h unchanged without
    # one measurement up there. Still too wet at the top on that day: hours with raw 8-16 mm/h gave 5.7 mm at the
    # gauges (shown now 7.3), raw 16-32 gave 6.3 (shown 14.1, 14 pairs) - a taper x (12 / r)^0.5 above 12 mm/h has
    # the same CSI and a smaller mean error and is the candidate if that repeats. Never measured so far: showers /
    # thunderstorms (raw 5-min values above 38 mm/h did not occur) - refit with check_italy_gain.py on such a day
    # and when the daily Italy tables drift.
    DAMP_A, DAMP_B, DAMP_C = 0.2, 0.8, 0.7

    def newest(self):
        d = json.loads(self._get(f"{self.API}/findLastProductByType?type={self.PRODUCT}", timeout=30))
        return int(d["lastProducts"][0]["time"])

    def load(self, ts):
        d = json.loads(self._get(f"{self.API}/downloadProduct", timeout=30,
                                 data=json.dumps({"productType": self.PRODUCT, "productDate": int(ts)}).encode()))
        a = np.asarray(Image.open(io.BytesIO(self._get(d["url"], timeout=60))), np.float32)
        if a.shape != (self.NY, self.NX):
            raise ValueError(f"unexpected grid {a.shape}")
        return np.where(np.isfinite(a) & (a >= 0), self.damp(np.minimum(a, 300.0)), -1.0).astype(np.float32)

    @classmethod
    def damp(cls, r):
        """SRI (mm/h) -> rain rate: light rain strongly reduced (1 -> 0.2, 2 -> 0.7), x DAMP_C from 4.8 mm/h on."""
        return r * np.minimum(cls.DAMP_C, cls.DAMP_A * np.maximum(r, 1e-6) ** cls.DAMP_B)


def odim_grid(f):
    """(proj string, x0, y1, cell, nx, ny) of an ODIM HDF5 composite from its /where group."""
    w = f["where"].attrs
    proj = w["projdef"].decode() if isinstance(w["projdef"], bytes) else str(w["projdef"])
    x, y = Transformer.from_crs("EPSG:4326", proj, always_xy=True).transform(float(w["UL_lon"]), float(w["UL_lat"]))
    return proj, float(x), float(y), float(w["xscale"]), int(w["xsize"]), int(w["ysize"])


class GermanyRv(Layer):
    name, credit = "de", "Deutscher Wetterdienst (open data)"
    RV = "https://opendata.dwd.de/weather/radar/composite/rv/"
    GAIN = 1.5
    # BORDER_KM (server-v1.79, 20.11.2026; OFF unless DE_BORDER_KM is set, e.g. 30): the RV composite only within this
    # distance of Germany (germany.json) - weight 1 at the border, 0 at BORDER_KM (FEATHER_KM). Without it the layer
    # counts with full weight far into Alsace / Lorraine, Benelux, W Czechia, W Poland, N Austria, N Switzerland, where
    # it reads like this at the gauges. Offline (natobs 16.-20.11., the composite recomputed per gauge from OPERA and
    # each layer alone - 0.4 % from the recorded one; gauges where DE counts >= .5; unadjusted amount; hourly CSI at
    # 0.5 / 2 mm), today -> 30 km (OPERA alone): Austria north x1.70 .36/.21 -> x1.47 .38/.23 (x1.25 .41/.24), Czechia
    # x0.79 .53/.23 -> x0.88 .58/.24 (x0.91 .61/.24), Poland west x0.76 .47/.15 -> x0.87 .55/.19 (x0.83 .55/.18),
    # Belgium / Luxembourg / E Netherlands x0.89 .64/.46 -> x1.10 .68/.49 (x1.17 .67/.47), Denmark south x0.94 .43/.28
    # -> x0.89 .51/.17, Switzerland north x1.36 .53/.34 -> x1.44 .54/.36, Germany x1.15 .64/.41 -> x1.15 .64/.40.
    BORDER_KM = float(os.environ.get("DE_BORDER_KM", "0") or 0) or None

    def _window(self):
        super()._window()
        if not self.BORDER_KM:
            return
        with open(os.path.join(os.path.dirname(os.path.abspath(__file__)), "germany.json"), encoding="utf-8") as f:
            g = json.load(f)
        tr = Transformer.from_crs("EPSG:4326", self.PROJ, always_xy=True)
        im = Image.new("L", (self.NX, self.NY), 0)
        draw = ImageDraw.Draw(im)
        for poly in g["coordinates"]:
            x, y = tr.transform(*zip(*poly[0]))
            draw.polygon([((a - self.X0) / self.CELL, (self.Y1 - b) / self.CELL) for a, b in zip(x, y)], fill=1)
        out_km = cv2.distanceTransform(1 - np.asarray(im, np.uint8), cv2.DIST_L2, 5) * self.CELL / 1000.0
        near = out_km <= self.BORDER_KM
        self.area = near if self.area is None else self.area & near

    def _runs(self):
        names = re.findall(r"composite_rv_(\d{8}_\d{4})\.tar", self._get(self.RV, timeout=60).decode("latin-1"))
        return sorted({int(datetime.strptime(n, "%Y%m%d_%H%M").replace(tzinfo=timezone.utc).timestamp() * 1000) for n in names})

    def _analysis(self, ts):
        name = datetime.fromtimestamp(ts / 1000, timezone.utc).strftime("composite_rv_%Y%m%d_%H%M.tar")
        with tarfile.open(fileobj=io.BytesIO(self._get(self.RV + name, timeout=120))) as tar:
            m = next(m for m in tar.getmembers() if m.name.rsplit("_", 1)[1].startswith("000"))
            return h5py.File(io.BytesIO(tar.extractfile(m).read()), "r")

    def geometry(self):
        with self._analysis(self._runs()[-1]) as f:
            self.PROJ, self.X0, self.Y1, self.CELL, self.NX, self.NY = odim_grid(f)

    def newest(self):
        return self._runs()[-1]

    def load(self, ts):
        with self._analysis(ts) as f:
            w = {k: float(f["dataset1/data1/what"].attrs[k]) for k in ("gain", "offset", "nodata")}
            raw = f["dataset1/data1/data"][:]
        nod = raw == w["nodata"]
        mm5 = mosaic.despeckle(np.where(nod, 0.0, np.maximum(raw * w["gain"] + w["offset"], 0.0)).astype(np.float32))
        return np.where(nod, -1.0, mm5 * 12.0 * self.GAIN).astype(np.float32)


class SwissRzc(Layer):
    name, credit = "ch", "MeteoSwiss (open data, CC BY)"
    STAC = "https://data.geo.admin.ch/api/stac/v1/collections/ch.meteoschweiz.ogd-radar-precip/items"
    # The product reaches ~250 km beyond the radars (Albis, La Dole, Monte Lema, Plaine Morte, Weissfluhgipfel), but
    # far out and behind the Alps the beams pass above the rain: 13.11.2026, 6.5 h, Italian gauges 180-220 km from
    # the nearest Swiss radar (Veneto, 1703 pairs): gauges 0.153 mm/h, ch 0.021 (wet at 9 % of the gauge readings
    # >= 0.3 mm/h), Italy's layer 0.080 - and ch pulled the picture down there (x0.38 of the gauges where both
    # counted, x0.73 from Italy's layer alone). 140-180 km: dry that day, unmeasured. So: full weight up to
    # RANGE_KM - FEATHER_KM, nothing beyond RANGE_KM (check_swiss_range.py; widen only when gauges out there
    # say so).
    RADARS = [(47.284, 8.512), (46.425, 6.099), (46.041, 8.833), (46.371, 7.487), (46.835, 9.794)]
    RANGE_KM = 160.0

    def __init__(self, *a):
        self.files, self.listed = {}, 0.0   # ts -> url
        super().__init__(*a)

    def _list(self):
        if time.time() - self.listed < 45:
            return
        now = datetime.now(timezone.utc)
        days = [now] + ([now - timedelta(days=1)] if now.hour < 2 else [])
        for day in days:
            d = json.loads(self._get(f"{self.STAC}/{day:%Y%m%d}-ch", timeout=60))
            for k, a in d.get("assets", {}).items():
                m = re.match(r"rzc(\d{2})(\d{3})(\d{4})..\.001\.h5$", k)
                if m:
                    t = datetime.strptime(f"20{m.group(1)}{m.group(2)}{m.group(3)}", "%Y%j%H%M").replace(tzinfo=timezone.utc)
                    self.files[int(t.timestamp() * 1000)] = a["href"]
        old = time.time() * 1000 - 3 * KEEP_MS
        self.files = {t: u for t, u in self.files.items() if t > old}
        self.listed = time.time()

    def geometry(self):
        self._list()
        with h5py.File(io.BytesIO(self._get(self.files[max(self.files)], timeout=60)), "r") as f:
            self.PROJ, self.X0, self.Y1, self.CELL, self.NX, self.NY = odim_grid(f)

    def newest(self):
        self._list()
        return max(self.files)

    def load(self, ts):
        with h5py.File(io.BytesIO(self._get(self.files[ts], timeout=60)), "r") as f:
            a = np.asarray(f["dataset1/data1/data"][:], np.float32)
        return np.where(np.isfinite(a) & (a >= 0), np.minimum(a, 300.0), -1.0).astype(np.float32)


class CzechChmi(Layer):
    name, credit = "cz", "CHMI (open data, CC BY 4.0)"
    SHADOW = True
    BASE = "https://opendata.chmi.cz/meteorology/weather/radar/composite/pseudocappi2km/hdf5/"
    ZR_A, ZR_B = 200.0, 1.6

    def __init__(self, *a):
        self.latest, self.raw = None, {}
        super().__init__(*a)

    def _url(self, ts):
        return self.BASE + datetime.fromtimestamp(ts / 1000, timezone.utc).strftime("T_PANV23_C_OKPR_%Y%m%d%H%M00.hdf")

    def geometry(self):
        t = self.newest()
        with h5py.File(io.BytesIO(self.raw[t]), "r") as f:
            self.PROJ, self.X0, self.Y1, self.CELL, self.NX, self.NY = odim_grid(f)

    def _window(self):
        super()._window()
        # the grid is Mercator: its 1556 m cells are ~1 km on the ground at 50 N
        self.k = max(1, int(round(self.host.KM_PER_PX * 1000 / (self.CELL * 0.64))))

    def newest(self):
        """The folder lists a week of files (2000 names), so the last steps are asked for by name instead."""
        t = int(time.time() * 1000) // STEP_MS * STEP_MS
        stop = max(self.latest or 0, t - 6 * STEP_MS)
        while t > stop:
            try:
                self.raw = {t: self._get(self._url(t), timeout=30)}
                self.latest = t
                break
            except urllib.error.HTTPError as e:
                if e.code != 404:
                    raise
            t -= STEP_MS
        if self.latest is None:
            raise ValueError("no file in the last 30 min")
        return self.latest

    def load(self, ts):
        with h5py.File(io.BytesIO(self.raw.get(ts) or self._get(self._url(ts), timeout=30)), "r") as f:
            w = {k: float(f["dataset1/data1/what"].attrs[k]) for k in ("gain", "offset", "nodata", "undetect")}
            raw = f["dataset1/data1/data"][:]
        if raw.shape != (self.NY, self.NX):
            raise ValueError(f"unexpected grid {raw.shape}")
        rate = (10.0 ** ((raw * w["gain"] + w["offset"]) / 10.0) / self.ZR_A) ** (1.0 / self.ZR_B)
        rate = np.where(raw == w["undetect"], 0.0, np.minimum(rate, 300.0))
        return np.where(raw == w["nodata"], -1.0, rate).astype(np.float32)


class DenmarkDmi(Layer):
    """Denmark - DMI open data (CC BY 4.0, opendataapi.dmi.dk, no account): DMI's composite of its five radars, DBZH
    (dBZ = 0.5 raw - 32, Z = 200 R^1.6 as DMI's own how/zr), 500 m, polar stereographic, every 5 min, ODIM HDF5, one
    STAC item per image (dk.com.<YYYYmmddHHMM>.500_max.h5), published in pairs every 10 min, 8-13 min after
    its time (22.11.2026).
    Why (22.11.2026): on 19.11. ~07:55Z the map showed a thin ring of light "rain" over the Kattegat (centre ~56.2 N
    11.3 E, ~86 km radius, no radar there) - in the OPERA picture only: DMI's own composite and SMHI's (Swedish) one
    were dry there at 07:45-08:10Z. The national services' quality control removes such clear-air / bird echoes
    (an hour after sunrise) that OPERA keeps. Shadow first, like every new country (scored at the gauges).
    The images alternate: "doppler" scans reach ~120 km from the radars, "fullRange" scans (every 10 min) ~240 km.
    Used only within RANGE_KM = the doppler range, so the coverage is the same in every image (no flicker at the edge)."""
    name, credit = "dk", "DMI (open data, CC BY 4.0)"
    SHADOW = True
    STAC = "https://opendataapi.dmi.dk/v1/radardata/collections/composite/items"
    ZR_A, ZR_B = 200.0, 1.6
    RADARS = [(55.1127, 14.8875), (55.1731, 8.552), (55.8119, 10.5853), (57.4893, 10.1365), (55.3262, 12.4493)]
    RANGE_KM = 120.0

    def __init__(self, *a):
        self.files, self.listed = {}, 0.0   # ts -> url
        super().__init__(*a)

    def _list(self):
        if time.time() - self.listed < 45:
            return
        now = datetime.now(timezone.utc)
        since = now - timedelta(milliseconds=KEEP_MS + 15 * 60_000)
        d = json.loads(self._get(f"{self.STAC}?datetime={since:%Y-%m-%dT%H:%M:%SZ}/{now:%Y-%m-%dT%H:%M:%SZ}&limit=200",
                                 timeout=60))
        for f in d.get("features", []):
            if not f.get("id", "").endswith(".500_max.h5"):
                continue
            t = datetime.strptime(f["properties"]["datetime"], "%Y-%m-%dT%H:%M:%SZ").replace(tzinfo=timezone.utc)
            self.files[int(t.timestamp() * 1000)] = f["asset"]["data"]["href"]
        old = time.time() * 1000 - 3 * KEEP_MS
        self.files = {t: u for t, u in self.files.items() if t > old}
        self.listed = time.time()

    def geometry(self):
        """The file's /where has the corners and the cell size but no xsize / ysize: the grid size from the data."""
        self._list()
        with h5py.File(io.BytesIO(self._get(self.files[max(self.files)], timeout=60)), "r") as f:
            w = f["where"].attrs
            proj = w["projdef"].decode() if isinstance(w["projdef"], bytes) else str(w["projdef"])
            x, y = Transformer.from_crs("EPSG:4326", proj, always_xy=True).transform(
                float(np.ravel(w["UL_lon"])[0]), float(np.ravel(w["UL_lat"])[0]))
            self.PROJ, self.X0, self.Y1, self.CELL = proj, float(x), float(y), float(np.ravel(w["xscale"])[0])
            self.NY, self.NX = f["dataset1/data1/data"].shape

    def newest(self):
        self._list()
        return max(self.files)

    def load(self, ts):
        with h5py.File(io.BytesIO(self._get(self.files[ts], timeout=60)), "r") as f:
            w = {k: float(np.ravel(f["what"].attrs[k])[0]) for k in ("gain", "offset", "nodata", "undetect")}
            raw = f["dataset1/data1/data"][:]
        if raw.shape != (self.NY, self.NX):
            raise ValueError(f"unexpected grid {raw.shape}")
        rate = (10.0 ** ((raw * w["gain"] + w["offset"]) / 10.0) / self.ZR_A) ** (1.0 / self.ZR_B)
        rate = np.where(raw == w["undetect"], 0.0, np.minimum(rate, 300.0))
        return np.where(raw == w["nodata"], -1.0, rate).astype(np.float32)


LAYERS = (ItalyDpc, GermanyRv, SwissRzc, CzechChmi, DenmarkDmi)


def timing(host, layer, shifts=(-10, -5, 0, 5, 10)):
    """How well the layer's wet area matches the host's own source (OPERA alone) for each time shift in minutes:
    {shift: mean CSI of "wet >= 0.3 mm/h" over the host's images}. The best shift says which of the layer's time
    stamps belongs to which host step (the services stamp the start or the end of the measurement)."""
    base = super(type(host), host)
    times = base.list_times()
    out = {}
    grids = {t: base.load(times[t])[layer.win] for t in sorted(times)[-8:]}
    for sh in shifts:
        scores = []
        for t, g in grids.items():
            with layer.lock:
                f = layer.frames.get(t - sh * 60_000)
            if f is None:
                continue
            rate, w = (a.astype(np.float32) for a in f)
            both = (w >= 1) & (g >= 0)
            a, b = (rate >= 0.3) & both, (g >= 0.3) & both
            if (a | b).sum() > 200:
                scores.append((a & b).sum() / (a | b).sum())
        out[sh] = round(float(np.mean(scores)), 3) if scores else None
    return out


def start(host, log, ua):
    """Create the layers for the host region and start their polling threads."""
    layers = []
    for cls in LAYERS:
        try:
            layer = cls(host, log, ua)
            threading.Thread(target=layer.loop, daemon=True).start()
            layers.append(layer)
        except Exception as e:
            log(f"layer {cls.name} failed to start: {e}")
    return layers
