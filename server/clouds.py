"""Cloud map with a 0-3 h nowcast for Europe ("sunny" mode), built from the EUMETSAT MTG GeoColour image.

Observation: GeoColour (MTG FCI, every 10 min, ~30-35 min latency; true colour by day, infrared plus a fog/low-cloud
channel by night) for all of Europe in one Web Mercator image (~2.5 km pixels). Cloudiness per pixel is read from
the image: clouds are bright and grey/white, land, sea and city lights have at least one dark channel.
By day the brightness scale follows the sun's elevation, so broken cloud fields with sun between them are not
read as overcast when the sun is low.
Nowcast, per forecast step:
  1. dense motion field (optical flow between the images 30 min apart, smoothed), so clouds move and also
     stretch, turn and shear where the wind differs from place to place, instead of one global shift
  2. growth and decay along the motion from the weather model: where the model's cloud cover rises or falls
     along a cloud's path, the cloud thickens/forms or thins/dissolves (Lagrangian trend)
  3. small structures fade with lead time (they are not predictable for long)
  4. blend into the model's cloud cover: satellite only for the first hour, model only from 4 h
The model is the self-hosted Open-Meteo "best match" (ICON-D2, AROME, HARMONIE, UKV ... wherever available).
"""
import io
import json
import math
import os
import threading
import time
import urllib.parse
import urllib.request
from datetime import datetime, timezone

import cv2
import numpy as np
from PIL import Image

WMS = "https://view.eumetsat.int/geoserver/ows"
LAYER = "mtg_fd:rgb_geocolour"
BBOX = (-2783000.0, 4028802.0, 5009000.0, 11753184.0)  # 25 W - 45 E, 34 N - 72 N in EPSG:3857
N = 3100                                                 # 2.5 km Web Mercator pixels (~1.5 km on the ground)
OBS_KEEP = 7                                             # observed frames: the last hour in 10 min steps
# every observed image is also logged at 10 km for backtests of the nowcast (server/verify_clouds.py)
LOG_DIR = os.environ.get("CLOUD_LOG_DIR", os.path.join(os.path.dirname(os.path.abspath(__file__)), "cloud_log"))
LOG_DOWN, LOG_KEEP_S = 4, 72 * 3600
FC_STEP_MS = 15 * 60_000                                 # forecast frames on the app's 15 min timeline
FC_HORIZON_MS = 3 * 3600_000                             # beyond the newest image (it is ~35 min old already)
SAT_FULL_MIN, SAT_ZERO_MIN = 60, 240                     # satellite weight: 1 for the first hour, 0 from 4 h
TREND = 0.5                                              # share of the model's cloud trend that is applied
MODEL_DEG, MODEL_BATCH, MODEL_REFRESH_S = 0.25, 2000, 30 * 60
FLOW_DOWN = 4                                            # optical flow on a 10 km grid: large-scale motion only


def sun_elevation(ts, lat, lon):
    """Solar elevation in degrees (NOAA approximation, same as the app); lat/lon may be arrays."""
    rad, d = math.pi / 180, ts / 86400e3 - 10957.5  # days since J2000.0
    g, q = (357.529 + 0.98560028 * d) * rad, 280.459 + 0.98564736 * d
    lam = (q + 1.915 * math.sin(g) + 0.02 * math.sin(2 * g)) * rad
    eps = (23.439 - 0.00000036 * d) * rad
    dec = math.asin(math.sin(eps) * math.sin(lam))
    ra = math.atan2(math.cos(eps) * math.sin(lam), math.cos(lam))
    gmst = (18.697374558 + 24.06570982441908 * d) % 24
    ha = (gmst * 15 + lon) * rad - ra
    return np.degrees(np.arcsin(np.sin(lat * rad) * math.sin(dec) + np.cos(lat * rad) * math.cos(dec) * np.cos(ha)))


def cloudiness(rgb, elev):
    """0..1 from a GeoColour RGB image (same formula as the app's fallback path).

    By day the image is a true-colour photo whose brightness depends on the sun: with a low sun a thick deck is
    only ~190 of 255 and a broken (cumulus / altocumulus) field ~130, so a fixed scale turned broken fields with
    lots of sun into overcast. The scale ends at the brightness of a thick deck for the local sun elevation.
    By night (infrared + fog channel) the brightness does not depend on the sun."""
    m = rgb.min(axis=2).astype(np.float32)
    top = 120.0 + 120.0 * np.sqrt(np.clip(np.sin(np.radians(elev)), 0.0, 1.0))  # thick deck brightness by day
    day = np.clip((m - 85.0) / np.maximum(top - 85.0, 30.0), 0.0, 1.0)
    night = np.clip((m - 80.0) / 90.0, 0.0, 1.0)
    w = np.clip((elev + 3.0) / 9.0, 0.0, 1.0)  # twilight: GeoColour fades from the IR to the photo
    c = w * day + (1 - w) * night
    return cv2.GaussianBlur(c, (0, 0), 0.7)  # JPEG noise


def sat_weight(lead_min):
    if lead_min <= SAT_FULL_MIN:
        return 1.0
    return max(0.0, 1.0 - (lead_min - SAT_FULL_MIN) / (SAT_ZERO_MIN - SAT_FULL_MIN))


class CloudNowcast:
    name = "clouds"

    def __init__(self, fetch, log, openmeteo_url, openmeteo_available, ll_to_3857):
        self.fetch, self.log, self.om_url, self.om_ok, self.ll = fetch, log, openmeteo_url, openmeteo_available, ll_to_3857
        self.lock = threading.Lock()
        self.obs, self.fc, self.ref = {}, {}, None  # ts -> uint8 cloudiness grid
        self.model = None                            # (times ms, cover (T, nlat, nlon) 0..1)
        self.model_at = 0
        x0, y0, x1, y1 = BBOX
        # lat/lon of the model grid and where each (coarse) cloud pixel lies in it
        self.lats = np.arange(34.0, 72.0 + 1e-6, MODEL_DEG)
        self.lons = np.arange(-25.0, 45.0 + 1e-6, MODEL_DEG)
        s = N // FLOW_DOWN
        xs = x0 + (np.arange(s) + 0.5) / s * (x1 - x0)
        ys = y1 - (np.arange(s) + 0.5) / s * (y1 - y0)
        lon, _ = ll_to_3857.transform(xs, np.full(s, y0), direction="INVERSE")
        _, lat = ll_to_3857.transform(np.full(s, x0), ys, direction="INVERSE")
        mx = ((lon - self.lons[0]) / MODEL_DEG).astype(np.float32)
        my = ((lat - self.lats[0]) / MODEL_DEG).astype(np.float32)
        self.model_maps = np.meshgrid(mx, my)
        self.coarse_ll = tuple(np.meshgrid(lat, lon, indexing="ij"))  # (lat, lon) of the coarse pixels, for the sun
        self.gx, self.gy = np.meshgrid(np.arange(N, dtype=np.float32), np.arange(N, dtype=np.float32))

    # ---------- data ----------
    def frames(self):
        with self.lock:
            return ([{"ts": t, "fc": False} for t in sorted(self.obs)] +
                    [{"ts": t, "fc": True} for t in sorted(self.fc)]), self.ref

    def grid(self, ts):
        with self.lock:
            return self.obs.get(ts) if ts in self.obs else self.fc.get(ts)

    def _latest_time(self):
        xml = self.fetch(f"{WMS[:-4]}/{LAYER.replace(':', '/')}/ows?service=WMS&version=1.3.0&request=GetCapabilities").decode("utf-8", "ignore")
        i = xml.find('<Dimension name="time"')
        stamp = xml[i:xml.find("<", i + 5)].rsplit("/", 2)[-2]
        return int(datetime.strptime(stamp[:19], "%Y-%m-%dT%H:%M:%S").replace(tzinfo=timezone.utc).timestamp() * 1000)

    def _load(self, ts):
        x0, y0, x1, y1 = BBOX
        stamp = datetime.fromtimestamp(ts / 1000, timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")
        url = (f"{WMS}?service=WMS&version=1.1.1&request=GetMap&layers={LAYER}&styles=&srs=EPSG:3857"
               f"&bbox={x0},{y0},{x1},{y1}&width={N}&height={N}&format=image/jpeg&time={stamp}")
        rgb = np.asarray(Image.open(io.BytesIO(self.fetch(url, timeout=120))).convert("RGB"))
        return (cloudiness(rgb, cv2.resize(sun_elevation(ts, *self.coarse_ll), (N, N))) * 255).astype(np.uint8)

    def _log(self, ts, g):
        try:
            os.makedirs(LOG_DIR, exist_ok=True)
            small = cv2.resize(g, (N // LOG_DOWN, N // LOG_DOWN), interpolation=cv2.INTER_AREA)
            np.savez_compressed(os.path.join(LOG_DIR, f"obs_{ts}.npz"), cloud=small)
            for f in os.listdir(LOG_DIR):
                fp = os.path.join(LOG_DIR, f)
                if time.time() - os.path.getmtime(fp) > LOG_KEEP_S:
                    os.remove(fp)
        except OSError as e:
            self.log(f"cloud log failed: {e}")

    def refresh_model(self):
        """Hourly total cloud cover on a 0.25 deg grid over Europe (self-hosted Open-Meteo, ~43k points)."""
        if not self.om_ok() or time.time() - self.model_at < MODEL_REFRESH_S:
            return
        la, lo = (v.ravel() for v in np.meshgrid(self.lats, self.lons, indexing="ij"))
        times, cover = None, None
        for i in range(0, len(la), MODEL_BATCH):
            body = urllib.parse.urlencode({
                "latitude": ",".join(f"{v:.2f}" for v in la[i:i + MODEL_BATCH]),
                "longitude": ",".join(f"{v:.2f}" for v in lo[i:i + MODEL_BATCH]),
                "hourly": "cloud_cover", "past_hours": "2", "forecast_hours": "6", "timeformat": "unixtime"}).encode()
            req = urllib.request.Request(f"{self.om_url}/v1/forecast", data=body,
                                         headers={"Content-Type": "application/x-www-form-urlencoded"})
            with urllib.request.urlopen(req, timeout=600) as r:
                d = json.loads(r.read())
            d = d if isinstance(d, list) else [d]
            if times is None:
                times = np.array(d[0]["hourly"]["time"], np.int64) * 1000
                cover = np.zeros((len(times), len(la)), np.float32)
            for j, e in enumerate(d):
                cover[:, i + j] = [(v or 0.0) / 100.0 for v in e["hourly"]["cloud_cover"]]
        cover = cover.reshape(len(times), len(self.lats), len(self.lons))
        with self.lock:
            self.model = (times, cover)
        self.model_at = time.time()
        try:  # model snapshot for fair backtests (only data that existed at the forecast time)
            os.makedirs(LOG_DIR, exist_ok=True)
            np.savez_compressed(os.path.join(LOG_DIR, f"model_{int(self.model_at * 1000)}.npz"), times=times, cover=cover.astype(np.float16))
        except OSError as e:
            self.log(f"cloud model log failed: {e}")
        self.log(f"[{datetime.now():%H:%M:%S}] clouds: model cloud cover for {len(la)} points")

    def model_field(self, ts):
        """Model cloud cover 0..1 at time ts on the cloud grid, or None."""
        with self.lock:
            m = self.model
        if m is None or not m[0][0] <= ts <= m[0][-1]:
            return None
        times, cover = m
        k = min(int(np.searchsorted(times, ts, side="right")) - 1, len(times) - 2)
        f = (ts - times[k]) / (times[k + 1] - times[k])
        small = cv2.remap(((1 - f) * cover[k] + f * cover[k + 1]).astype(np.float32), *self.model_maps,
                          cv2.INTER_LINEAR, borderMode=cv2.BORDER_REPLICATE)
        return cv2.resize(small, (N, N), interpolation=cv2.INTER_LINEAR)

    # ---------- nowcast ----------
    def update(self):
        end = self._latest_time()
        want = [end - k * 600_000 for k in range(OBS_KEEP)]
        new = [t for t in want if t not in self.obs]
        for t in new:  # newest first: the map is usable as soon as the first image is in
            g = self._load(t)
            with self.lock:
                self.obs[t] = g
            self._log(t, g)
        with self.lock:
            for t in list(self.obs):
                if t not in want:
                    del self.obs[t]
        if not new:
            return 0
        prev_t = end - 30 * 60_000
        if prev_t in self.obs:
            self._forecast(self.obs[prev_t], self.obs[end], end)
        return len(new)

    def _forecast(self, prev, last, ref):
        s = (N // FLOW_DOWN, N // FLOW_DOWN)
        a = cv2.resize(prev, s, interpolation=cv2.INTER_AREA)
        b = cv2.resize(last, s, interpolation=cv2.INTER_AREA)
        flow = cv2.calcOpticalFlowFarneback(a, b, None, 0.5, 4, 21, 4, 7, 1.5, 0)
        flow = cv2.GaussianBlur(flow, (0, 0), 8)  # ~80 km: keeps the large-scale pattern, avoids streaks from noisy shear
        flow = cv2.resize(flow, (N, N), interpolation=cv2.INTER_LINEAR) * (FLOW_DOWN / 30.0)  # pixels per minute
        base = last.astype(np.float32) / 255.0
        m0 = self.model_field(ref)
        fc = {}
        first = (ref // FC_STEP_MS + 1) * FC_STEP_MS  # forecast frames at :00, :15, :30, :45
        for ts in range(first, ref + FC_HORIZON_MS + 1, FC_STEP_MS):
            lead = (ts - ref) / 60_000
            mx, my = self.gx - flow[..., 0] * lead, self.gy - flow[..., 1] * lead  # where the air came from
            c = cv2.remap(base, mx, my, cv2.INTER_LINEAR, borderMode=cv2.BORDER_REPLICATE)
            m1 = self.model_field(ts)
            if m0 is not None and m1 is not None:
                a0 = cv2.remap(m0, mx, my, cv2.INTER_LINEAR, borderMode=cv2.BORDER_REPLICATE)  # model at the origin
                a1 = a0 + TREND * (m1 - a0)
                thin = np.where(a0 > 0.05, a1 / np.maximum(a0, 1e-3), 1.0)
                form = np.where(a0 < 0.95, (a1 - a0) / np.maximum(1 - a0, 1e-3), 0.0)
                c = np.where(a1 < a0, c * thin, c + (1 - c) * form * 0.5)
            c = cv2.GaussianBlur(c, (0, 0), 0.6 * math.sqrt(lead / 10))  # small clouds are not predictable for long
            w = sat_weight(lead)
            if m1 is not None and w < 1:
                c = w * c + (1 - w) * m1
            fc[ts] = (np.clip(c, 0, 1) * 255).astype(np.uint8)
        with self.lock:
            self.fc, self.ref = fc, ref

    def model_loop(self):
        """Separate thread: a slow model download (e.g. a new run on a cold Open-Meteo cache) never holds up the
        satellite clouds; the nowcast uses whatever model data is there and is redone once new data arrives."""
        while True:
            had = self.model is not None
            try:
                self.refresh_model()
            except Exception as e:
                self.log(f"clouds model update failed: {e}")
            if not had and self.model is not None and self.ref is not None:
                with self.lock:
                    prev, last = self.obs.get(self.ref - 30 * 60_000), self.obs.get(self.ref)
                if prev is not None and last is not None:
                    self._forecast(prev, last, self.ref)
            time.sleep(60)

    def loop(self):
        threading.Thread(target=self.model_loop, daemon=True).start()
        while True:
            started = time.time()
            try:
                n = self.update()
                if n:
                    self.log(f"[{datetime.now():%H:%M:%S}] clouds: +{n} images, latest "
                             f"{datetime.fromtimestamp(self.ref / 1000, timezone.utc):%H:%M}Z, {time.time() - started:.0f}s"
                             if self.ref else f"[{datetime.now():%H:%M:%S}] clouds: +{n} images")
            except Exception as e:  # keep serving the last good data
                self.log(f"clouds update failed: {e}")
            time.sleep(60)

    # ---------- serving ----------
    def _pixels(self, X, Y):
        x0, y0, x1, y1 = BBOX
        return ((X - x0) / (x1 - x0) * N - 0.5).astype(np.float32), ((y1 - Y) / (y1 - y0) * N - 0.5).astype(np.float32)

    def image(self, ts, bbox, w, h):
        """Cloudiness as a grey + alpha PNG (alpha 0 outside the covered area); None for an unknown frame."""
        g = self.grid(ts)
        if g is None:
            return None
        minx, miny, maxx, maxy = bbox
        X, Y = np.meshgrid(minx + (np.arange(w) + 0.5) / w * (maxx - minx), maxy - (np.arange(h) + 0.5) / h * (maxy - miny))
        col, row = self._pixels(X, Y)
        v = cv2.remap(g, col, row, cv2.INTER_LINEAR, borderMode=cv2.BORDER_CONSTANT, borderValue=0)
        inside = ((col >= 0) & (col <= N - 1) & (row >= 0) & (row <= N - 1)).astype(np.uint8) * 255
        buf = io.BytesIO()
        Image.fromarray(np.dstack([v, inside]), "LA").save(buf, "PNG", compress_level=3)
        return buf.getvalue()

    def point(self, lat, lon):
        """Cloudiness series around a place (mean over ~3 km), or [] outside the covered area."""
        x, y = self.ll.transform(lon, lat)
        col, row = (int(round(float(v))) for v in self._pixels(np.float64(x), np.float64(y)))
        if not (2 <= col < N - 2 and 2 <= row < N - 2):
            return []
        frames, _ = self.frames()
        out = []
        for f in frames:
            g = self.grid(f["ts"])
            if g is not None:
                out.append({"ts": f["ts"], "fc": f["fc"], "cloud": round(float(g[row - 2:row + 3, col - 2:col + 3].mean()) / 255, 3)})
        return out
