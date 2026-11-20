"""Static clutter mask: radar pixels that show rain in most images while the country around them does not.

Found by warn_check.py (18./19.11.2026): a fixed echo near Dublin (53.28 N 6.29 W) in OPERA, wet in ~80 % of the
images with peaks of 15 mm/h and no gauge under it; MRMS cells on the Florida Keys (24.98 N 80.58 W, 25.86 N
80.10 W). The median filter in EuropeOpera.load (single pixels) and mosaic.despeckle (isolated tiny echoes in one
image) do not see such spots: they are a few pixels wide and the same in every image. spokes.py removes lines towards
a radar, not spots.

Method (data-driven, nothing hand-placed). Every image of the instance's own picture is counted BEFORE the mask is
applied (a mask built from masked pictures would vanish the next day), on the verification grid (grid / COUNT_DOWN,
the resolution of the obs_eu archive), with counters that decay with a time constant of WINDOW_DAYS. Tests:
1. "spot": counted are only the images in which the ring RING_KM around the cell is dry (<= RING_DRY of its area
   wet); a cell that is wet in >= COND_MIN of those is clutter. Real rain is almost never wet in one spot while
   everything 20-40 km around it is dry, and if, it moves on; clutter stays. Connected areas > MAX_AREA_KM2: never.
   72 h to 19.11. (check_clutter.py): radar-site echoes in Norway, Sweden, Finland, Poland (Rzeszow: picture ~100 mm,
   gauge 12.7 mm), Malta, Romania; Mount Baker in MRMS.
2. "hot": the mean rate over the window >= HOT_MM_H and >= HOT_RATIO x the ring's mean rate, wet in >= HOT_WET
   of the images, connected area <= HOT_AREA_KM2. Dublin is not a spot but a ~1300 km2 area wet in 80 % of the images (the ring is never dry) with a
   core of 8-23 mm/h MEAN over three days (= 600-1600 mm) next to 0.5 mm/h around it - no rain does that.
3. "weak" (server-v1.79, 20.11.2026): wet in >= WEAK_COND of the dry-ring images AND a mean rate >= WEAK_RATIO x
   and a wet share >= WEAK_WET x the ring's - fixed echoes too faint for test 1 or 2 (wind farms: Blakliden 64.22 N
   18.48 E cond .35, mean .45 vs ring .04 mm/h, wet .51 vs .15; Pretul 47.56 N 15.74 E cond .38, 1.28 vs .12,
   .49 vs .14). Connected areas > MAX_AREA_KM2: never.
GROWING (server-v1.79, 20.11.2026): the tests find the core of a fixed echo, not all of it - Markbygden (65.35 N
20.76 E, ~20 x 20 km) was masked with 3 of its 20 cells (only those reached 3 mm/h; the 16 cells wet in >= .5 of the
dry-ring images were one area > MAX_AREA_KM2, and on 16.-19.11. no cell reached 3 mm/h: not masked at all), and eun's
x3 Sweden gauge factor then showed the rest at 9-19 mm/h. So test 1 takes areas up to SPOT_AREA_KM2 (what that adds
on 16.-20.11.: Markbygden, Rost radar 14 cells, the sea 20 km NW of the radar-site echo 59.85 N 5.08 E (wet .91,
cond .71), 47.46 N 24.62 E (wet .86, cond .69) - all wet in >= .64 of the dry-ring images), and every found spot is
grown into the connected cells that are wet in >= GROW_COND of the dry-ring images, or have a
mean rate >= GROW_MM_H and >= GROW_RATIO x the ring's (wet >= HOT_WET); a grown area > GROW_AREA_KM2 keeps its seed
only. Replay (30-min archive copy, 72 h windows to 16./17./18./19.11. 12 UTC, eu and eun): 49-82 -> 162-229 cells
(800-1300 -> 2600-3700 km2, all Europe; live counters 20.11.: eu 63 -> 198, eun 71 -> 208, na 1 -> 4: Mount Rainier
next to Mount Baker); gauge rain in or next to the new cells 0.00-0.03 % of
all gauge rain (at most 2 gauges: 17.5 mm next to a spot in the Bukk 48.10 N 20.50 E, picture 22.6, ring 16.3);
the next 24 h (out of sample), 2-8 gauges within 12 km of new cells: gauges 4.1 / 4.5 / 27.7 mm, picture with the
old mask 12.1 / 11.1 / 17.6, with the new 12.3 / 10.8 / 17.1 (eu). Alps: no new cell but Pretul; W Norway: only
around the radar-site echo 59.85 N 5.08 E (incl. the sea 20 km NW of it, wet 88-100 % of the day on 13.-15.11. while
Bergen was 6-60 %, dry gauge no-1418) and 58.55 N 6.8 E. CLUTTER_GROW=0 switches test 3, the larger spots and the
growing off (the server-v1.61 mask).
The mask is upsampled to the picture's grid and grown by GROW_KM; in each image its pixels get the mean of the valid,
unmasked pixels within FILL_KM: dry country -> 0, real rain passing over -> the surrounding rain, no hole.
Rebuilt every REBUILD_H hours; counters and mask in <log dir>/clutter/. A new instance bootstraps the counters from
the obs_eu archive (same grid) so it does not wait days. check_clutter.py runs the same tests on the archive offline.
"""
import glob
import os
import threading
import time
from datetime import datetime, timezone

import cv2
import numpy as np

WET = 0.1             # mm/h: "wet" for the counters
COUNT_DOWN = 2        # counters on the verification grid (eu: 4 km, na: 0.04 deg) - as the obs_eu archive
RING_KM = (20.0, 40.0)
# test 1, "spot"
COND_MIN = 0.50       # wet in >= half of the images in which the ring around it is dry
RING_DRY = 0.05       # "the ring is dry": <= 5 % of its valid area wet in that image
COARSE_KM = 8.0       # the ring is judged on an ~8 km grid (cheap)
MAX_AREA_KM2 = 150.0  # larger connected candidate areas are real rain, never masked
MIN_DRY = 0.10        # judged once the decayed dry-ring count is >= 0.1 x TAU_IMAGES (~7 h of dry ring)
# test 2, "hot"
HOT_MM_H = 3.0        # mean rate over the window (3 mm/h = 216 mm in 72 h)
HOT_RATIO = 8.0       # ... and >= 8 x the mean rate of the ring around it
HOT_WET = 0.25        # ... and wet in >= 25 % of the images: a fixed echo, not one extreme storm that stayed for hours
HOT_AREA_KM2 = 800.0
MIN_VALID = 0.5       # judged once the decayed valid count is >= 0.5 x TAU_IMAGES (~2 days of images)
# test 3, "weak", and growing the found spots (server-v1.79, 20.11.2026; see above)
GROW = os.environ.get("CLUTTER_GROW", "1") == "1"
WEAK_COND = 0.33
WEAK_RATIO = 8.0      # mean rate >= 8 x the ring's
WEAK_WET = 3.0        # wet share >= 3 x the ring's
GROW_COND = 0.25
GROW_MM_H = 1.0
GROW_RATIO = 4.0
GROW_AREA_KM2 = 800.0
SPOT_AREA_KM2 = 400.0  # test 1 up to this area (MAX_AREA_KM2 without GROW)
GROW_KM = 2.0
FILL_KM = 10.0
WINDOW_DAYS = 3
TAU_IMAGES = WINDOW_DAYS * 288   # decay time constant of the counters, in 5-minute images
REBUILD_H = 6
SAVE_EVERY = 12       # counters written every 12 images (~1 h)
DAY_MS = 86400_000
KEYS = ("valid", "wet", "rsum", "dvalid", "dwet")


def ring_kernel(km_per_px, ring_km=RING_KM):
    r0, r1 = ring_km[0] / km_per_px, ring_km[1] / km_per_px
    n = int(np.ceil(r1))
    y, x = np.mgrid[-n:n + 1, -n:n + 1]
    d = np.hypot(x, y)
    return ((d >= r0) & (d <= r1)).astype(np.float32)


def disk(radius_px):
    n = max(0, int(round(radius_px)))
    y, x = np.mgrid[-n:n + 1, -n:n + 1]
    return (np.hypot(x, y) <= radius_px + 0.01).astype(np.uint8)


def down(g, f):
    """The verification grid of fcverify.RegionVerify._grid / obs_eu: block mean, NODATA (-1) where < half covered."""
    g = np.asarray(g, np.float32)
    if f == 1:
        return g
    h, w = g.shape[0] // f, g.shape[1] // f
    cov = cv2.resize((g >= 0).astype(np.float32), (w, h), interpolation=cv2.INTER_AREA) > 0.5
    a = cv2.resize(np.where(g < 0, 0, g), (w, h), interpolation=cv2.INTER_AREA)
    return np.where(cov, a, -1.0).astype(np.float32)


def counts_of(r, km_per_px):
    """One image on the counting grid -> its increments (valid, wet, rate, dry-ring valid, dry-ring wet)."""
    v = r >= 0
    w = v & (r >= WET)
    f = max(1, int(round(COARSE_KM / km_per_px)))
    H, W = r.shape
    h, wd = H // f, W // f
    vc = v[:h * f, :wd * f].reshape(h, f, wd, f).mean(axis=(1, 3), dtype=np.float32)
    wc = w[:h * f, :wd * f].reshape(h, f, wd, f).mean(axis=(1, 3), dtype=np.float32)
    k = ring_kernel(km_per_px * f)
    num = cv2.filter2D(wc, -1, k, borderType=cv2.BORDER_CONSTANT)
    den = cv2.filter2D(vc, -1, k, borderType=cv2.BORDER_CONSTANT)
    dc = (den >= 0.5 * k.sum()) & (num <= RING_DRY * np.maximum(den, 1e-6))
    dry = np.zeros(r.shape, bool)
    dry[:h * f, :wd * f] = np.repeat(np.repeat(dc, f, axis=0), f, axis=1)
    return v, w, np.where(v, r, 0.0), v & dry, w & dry


def build_mask(c, km_per_px, n_eff):
    """c: counters {valid, rsum, dvalid, dwet} on the counting grid; n_eff: the number of images they hold (decayed:
    <= TAU_IMAGES). Returns (mask bool on the counting grid, not grown; blobs [{kind, row, col, px, ...}])."""
    valid = np.asarray(c["valid"], np.float32)
    dvalid = np.asarray(c["dvalid"], np.float32)
    ok_d = dvalid >= MIN_DRY * n_eff
    cond = np.where(ok_d, c["dwet"] / np.maximum(dvalid, 1e-6), 0.0).astype(np.float32)
    ok_v = valid >= MIN_VALID * n_eff
    mean = np.where(ok_v, c["rsum"] / np.maximum(valid, 1e-6), 0.0).astype(np.float32)
    wet = np.where(ok_v, c["wet"] / np.maximum(valid, 1e-6), 0.0).astype(np.float32)
    k = ring_kernel(km_per_px)
    num = cv2.filter2D(mean * ok_v, -1, k, borderType=cv2.BORDER_CONSTANT)
    den = cv2.filter2D(ok_v.astype(np.float32), -1, k, borderType=cv2.BORDER_CONSTANT)
    full = den >= 0.5 * k.sum()
    ring_mean = np.where(full, num / np.maximum(den, 1e-6), np.inf)
    tests = [("spot", ok_d & (cond >= COND_MIN), SPOT_AREA_KM2 if GROW else MAX_AREA_KM2),
             ("hot", ok_v & (mean >= HOT_MM_H) & (mean >= HOT_RATIO * ring_mean) & (wet >= HOT_WET), HOT_AREA_KM2)]
    if GROW:
        ring_wet = np.where(full, cv2.filter2D(wet * ok_v, -1, k, borderType=cv2.BORDER_CONSTANT) / np.maximum(den, 1e-6),
                            np.inf)
        tests.append(("weak", ok_d & (cond >= WEAK_COND) & (mean >= WEAK_RATIO * ring_mean) & (wet >= WEAK_WET * ring_wet),
                      MAX_AREA_KM2))
    px_km2 = km_per_px * km_per_px
    kinds = np.zeros(valid.shape, np.uint8)   # bit per test that found the cell
    for bit, (kind, cand, max_km2) in enumerate(tests):
        if not cand.any():
            continue
        n, lab, stats, _ = cv2.connectedComponentsWithStats(cand.astype(np.uint8), connectivity=8)
        keep = stats[:, cv2.CC_STAT_AREA] <= max_km2 / px_km2
        keep[0] = False
        kinds[keep[lab]] |= np.uint8(1 << bit)
    mask = kinds > 0
    if GROW and mask.any():   # each found spot -> the connected area of its fixed echo
        reg = mask | (ok_d & (cond >= GROW_COND)) | (ok_v & (mean >= GROW_MM_H) & (mean >= GROW_RATIO * ring_mean) &
                                                   (wet >= HOT_WET))
        n, lab, stats, _ = cv2.connectedComponentsWithStats(reg.astype(np.uint8), connectivity=8)
        ok = np.zeros(n, bool)
        ok[np.unique(lab[mask])] = True
        ok &= stats[:, cv2.CC_STAT_AREA] <= GROW_AREA_KM2 / px_km2
        ok[0] = False
        mask |= ok[lab]
    blobs = []
    n, lab, stats, _ = cv2.connectedComponentsWithStats(mask.astype(np.uint8), connectivity=8)
    for i in range(1, n):
        x0, y0, w, h = (int(stats[i, j]) for j in (cv2.CC_STAT_LEFT, cv2.CC_STAT_TOP, cv2.CC_STAT_WIDTH, cv2.CC_STAT_HEIGHT))
        sub = lab[y0:y0 + h, x0:x0 + w] == i
        bits = int(np.bitwise_or.reduce(kinds[y0:y0 + h, x0:x0 + w][sub]))
        rr, cc = np.nonzero(sub)
        j = int(np.argmax(mean[y0:y0 + h, x0:x0 + w][sub]))
        r_, c_ = y0 + int(rr[j]), x0 + int(cc[j])
        blobs.append({"kind": "+".join(t[0] for b, t in enumerate(tests) if bits >> b & 1), "row": r_, "col": c_,
                      "px": int(stats[i, cv2.CC_STAT_AREA]), "cond": round(float(cond[r_, c_]), 2),
                      "ndry": round(float(dvalid[r_, c_])), "mean": round(float(mean[r_, c_]), 2),
                      "ring_mean": round(float(min(ring_mean[r_, c_], 99)), 2)})
    return mask, blobs


def full_mask(mask, f, shape, km_per_px):
    """The counting grid's mask on the picture's grid, grown by GROW_KM."""
    m = np.zeros(shape, np.uint8)
    if f > 1:
        up = np.repeat(np.repeat(mask.astype(np.uint8), f, axis=0), f, axis=1)
        m[:up.shape[0], :up.shape[1]] = up[:shape[0], :shape[1]]
    else:
        m[:] = mask
    grow = GROW_KM / km_per_px
    if m.any() and grow >= 0.5:
        m = cv2.dilate(m, disk(grow))
    return m.astype(bool)


class Filler:
    """Replaces the masked pixels of an image by the mean of the valid, unmasked pixels within FILL_KM (per spot box)."""

    def __init__(self, mask, km_per_px):
        self.mask = mask
        self.boxes = []
        if mask is None or not mask.any():
            return
        pad = int(np.ceil(FILL_KM / km_per_px)) + 1
        self.k = disk(FILL_KM / km_per_px).astype(np.float32)
        n, lab, stats, _ = cv2.connectedComponentsWithStats(mask.astype(np.uint8), connectivity=8)
        H, W = mask.shape
        for i in range(1, n):
            x0, y0, w, h = (int(stats[i, j]) for j in (cv2.CC_STAT_LEFT, cv2.CC_STAT_TOP, cv2.CC_STAT_WIDTH, cv2.CC_STAT_HEIGHT))
            self.boxes.append((max(0, y0 - pad), min(H, y0 + h + pad), max(0, x0 - pad), min(W, x0 + w + pad)))

    def __call__(self, rate):
        if not self.boxes or rate.shape != self.mask.shape:
            return rate
        out = np.array(rate, np.float32, copy=True)
        for r0, r1, c0, c1 in self.boxes:
            a = out[r0:r1, c0:c1]
            m = self.mask[r0:r1, c0:c1]
            good = (~m) & (a >= 0)
            num = cv2.filter2D(np.where(good, a, 0).astype(np.float32), -1, self.k, borderType=cv2.BORDER_CONSTANT)
            den = cv2.filter2D(good.astype(np.float32), -1, self.k, borderType=cv2.BORDER_CONSTANT)
            # filter2D may go through a DFT for this kernel: its round-off left -1e-7 in dry country, which reads as
            # "no data" (< 0) - found in the server image's test, 19.11.
            fill = np.where(den > 0.5, np.maximum(num / np.maximum(den, 1e-6), 0.0), 0.0)
            sel = m & (a >= 0)   # outside the radar's coverage stays NODATA
            a[sel] = fill[sel]
        return out


class ClutterMask:
    """Per instance: counts every image, keeps the mask up to date, applies it (apply(ts, rate) -> rate).
    Counters: five float32 grids on the counting grid (~21 MB for Europe, ~31 MB for North America), decaying per
    image (time constant TAU_IMAGES) - the host's memory is tight."""

    def __init__(self, name, shape, km_per_px, state_dir, log, archive=None, lonlat=None):
        self.name, self.shape, self.km = name, tuple(shape), float(km_per_px)
        self.f = COUNT_DOWN
        self.cshape = (self.shape[0] // self.f, self.shape[1] // self.f)
        self.ckm = self.km * self.f
        self.dir = os.path.join(state_dir, "clutter")
        self.log, self.lonlat = log, lonlat
        self.lock = threading.Lock()
        self.save_lock = threading.Lock()
        self.c = None            # {KEYS: float32 grids}
        self.n_eff = 0.0         # decayed number of images in the counters
        self.n_img = 0
        self.seen = set()        # image times already counted (refilled steps, restarts)
        self.unsaved = 0
        self.filler = Filler(None, self.km)
        self.built_at = 0.0
        self.building = False
        self._load()
        if self.c is None and archive:
            self.building = True   # no rebuilds from half-filled counters while the archive is read
            threading.Thread(target=self._bootstrap, args=(archive,), daemon=True).start()
        elif self.c is not None:
            self._rebuild_async()

    # ---- counters ----
    def _load(self):
        try:
            fp = os.path.join(self.dir, "counts.npz")
            if os.path.exists(fp):
                z = np.load(fp)
                if tuple(z["valid"].shape) == self.cshape:
                    self.c = {k: z[k].astype(np.float32) for k in KEYS}
                    self.seen = set(int(t) for t in z["seen"])
                    self.n_eff, self.n_img = float(z["n_eff"]), int(z["n"])
        except Exception as e:
            self.log(f"{self.name}: clutter counters not read: {e}")
            self.c = None

    def _save(self):
        if not self.save_lock.acquire(blocking=False):   # the previous save is still writing
            return
        try:
            os.makedirs(self.dir, exist_ok=True)
            with self.lock:
                if self.c is None:
                    return
                c = {k: v.copy() for k, v in self.c.items()}
                seen, n_eff, n = np.array(sorted(self.seen), np.int64), self.n_eff, self.n_img
            tmp = os.path.join(self.dir, "tmp_counts.npz")
            np.savez_compressed(tmp, seen=seen, n_eff=n_eff, n=n, **c)
            os.replace(tmp, os.path.join(self.dir, "counts.npz"))
        except Exception as e:
            self.log(f"{self.name}: clutter counters not saved: {e}")
        finally:
            self.save_lock.release()

    def count(self, ts, r, autosave=True):
        """r: one image on the counting grid."""
        if tuple(r.shape) != self.cshape or ts in self.seen:
            return
        inc = counts_of(r, self.ckm)   # outside the lock
        save = False
        with self.lock:
            if ts in self.seen:
                return
            self.seen.add(ts)
            if len(self.seen) > 2 * TAU_IMAGES:
                cut = ts - WINDOW_DAYS * DAY_MS
                self.seen = {t for t in self.seen if t >= cut}
            if self.c is None:
                self.c = {k: np.zeros(self.cshape, np.float32) for k in KEYS}
            a = np.float32(1.0 - 1.0 / TAU_IMAGES)
            for k, x in zip(KEYS, inc):
                self.c[k] *= a
                self.c[k] += x
            self.n_eff = self.n_eff * float(a) + 1.0
            self.n_img += 1
            self.unsaved += 1
            if autosave and self.unsaved >= SAVE_EVERY:
                self.unsaved, save = 0, True
        if save:
            threading.Thread(target=self._save, daemon=True).start()

    def _bootstrap(self, obs_dir):
        """Counters from the verification archive (the counting grid, last WINDOW_DAYS days, oldest first)."""
        try:
            t0 = time.time()
            now = time.time() * 1000
            files = []
            for fp in sorted(glob.glob(os.path.join(obs_dir, "2*.npz"))):
                try:
                    t = int(datetime.strptime(os.path.basename(fp)[:12], "%Y%m%d%H%M").replace(
                        tzinfo=timezone.utc).timestamp() * 1000)
                except ValueError:
                    continue
                if now - t <= WINDOW_DAYS * DAY_MS:
                    files.append((t, fp))
            n = 0
            for t, fp in files:
                try:
                    r = np.load(fp)["rate"].astype(np.float32)
                except Exception:
                    continue
                if tuple(r.shape) == self.cshape:
                    self.count(t, r, autosave=False)   # saved once at the end
                    n += 1
                time.sleep(0.005)   # leave the CPU to the forecast
            self._save()
            self.log(f"{self.name}: clutter counters bootstrapped from {n} archived images in {time.time() - t0:.0f} s")
        except Exception as e:
            self.log(f"{self.name}: clutter bootstrap failed: {e}")
        self._rebuild()

    # ---- mask ----
    def _rebuild(self):
        self.built_at = time.time()
        try:
            with self.lock:
                if self.c is None:
                    return
                c = {k: v.copy() for k, v in self.c.items()}
                n_eff, n_img = self.n_eff, self.n_img
            m, blobs = build_mask(c, self.ckm, n_eff)
            del c
            mask = full_mask(m, self.f, self.shape, self.km)
            self.filler = Filler(mask, self.km)
            os.makedirs(self.dir, exist_ok=True)
            np.savez_compressed(os.path.join(self.dir, "mask.npz"), mask=m)
            spots = ", ".join(f"{b['kind']} {b['lat']:.2f} {b['lon']:.2f} ({b['px']})" for b in self._located(blobs)[:20])
            self.log(f"clutter mask {self.name}: {int(mask.sum())} pixels masked in {len(blobs)} spots, "
                     f"{n_img} images counted" + (f": {spots}" if spots else ""))
        except Exception as e:
            self.log(f"{self.name}: clutter mask not built: {e}")
        finally:
            self.building = False

    def _located(self, blobs):
        out = []
        for b in sorted(blobs, key=lambda b: -b["mean"]):
            if self.lonlat:   # counting-grid cell centre -> picture pixel -> lon / lat
                lon, lat = self.lonlat((b["col"] + 0.5) * self.f - 0.5, (b["row"] + 0.5) * self.f - 0.5)
            else:
                lon, lat = np.nan, np.nan
            out.append(dict(b, lat=float(lat), lon=float(lon)))
        return out

    def _rebuild_async(self):
        if self.building:
            return
        self.building = True
        threading.Thread(target=self._rebuild, daemon=True).start()

    def apply(self, ts, rate):
        """Counts the unmasked image, then returns it with the clutter replaced."""
        try:
            ts = int(ts)
            if ts not in self.seen:
                self.count(ts, down(rate, self.f))
            if time.time() - self.built_at > REBUILD_H * 3600:
                self._rebuild_async()
            return self.filler(rate)
        except Exception as e:
            self.log(f"{self.name}: clutter mask failed: {e}")
            return rate
