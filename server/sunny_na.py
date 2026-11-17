"""sunny for the Americas from the US weather satellite GOES-East (GOES-19, 75.2 W), stage 1 of sunny worldwide
(user, 15.11.2026: "Wolken weltweit ... zumindest sunny").

Source: s3://noaa-goes19 (NOAA open data, no account), product ABI-L2-ACMF = the clear sky mask of the full disk, every
10 minutes, ~10 minutes after the scan starts, 2 km, NetCDF4 (read with h5py). Its "Cloud_Probabilities" (0..1) is
derived from the infrared bands as well, so it is the same by day and by night - unlike Europe's sunny, which reads
the sunshine from DWD's direct radiation by day and from cloud-top heights at night.

Everything else is SunnyDE (sunny_de.py): k = 1 - 0.95 x cloud probability stands in for the clear-sky index, and the
motion, the 2 h forecast, the map frames, the atlas tiles, the grid verification and the warm start run unchanged on an
America grid (Web Mercator, 6 km cells, 135 W - 50 W, 12 N - 60 N; GOES-East sees it all, the far north-west at a
flat angle). Verified at the stations against the METAR cloud cover (northamerica.MetarSky, rough: 1 - cover vs the sun share).
Not yet: GOES-West (Alaska, Hawaii, the Pacific coast at a better angle). The weather-model shadows only with
SUNNY_NA_MODEL=1.
"""
import io
import os
import re
import time
from datetime import datetime, timezone

import h5py
import numpy as np

import sunny_de
from sunny_de import merc, share_from_k

BUCKET = "https://noaa-goes19.s3.amazonaws.com"
PRODUCT = "ABI-L2-ACMF"
_x, _y = merc(np.array([-135.0, -50.0]), np.array([12.0, 60.0]))
AMERICA = {"bbox": (float(_x[0]), float(_y[0]), float(_x[1]), float(_y[1])), "cell": 6000.0, "prefix": "na"}


def goes_index(f, lat, lon):
    """(row, col) in the GOES fixed grid of file f for every lat / lon (degrees), -1 where the satellite does not see it
    (GOES-R Product User Guide, 4.2.8.1)."""
    p = f["goes_imager_projection"].attrs
    a, b = float(p["semi_major_axis"][0]), float(p["semi_minor_axis"][0])
    H = float(p["perspective_point_height"][0]) + a
    lon0 = np.radians(float(p["longitude_of_projection_origin"][0]))
    la, lo = np.radians(lat.astype(np.float64)), np.radians(lon.astype(np.float64))
    e2 = (a * a - b * b) / (a * a)
    lat_c = np.arctan((b * b) / (a * a) * np.tan(la))
    rc = b / np.sqrt(1 - e2 * np.cos(lat_c) ** 2)
    sx = H - rc * np.cos(lat_c) * np.cos(lo - lon0)
    sy = -rc * np.cos(lat_c) * np.sin(lo - lon0)
    sz = rc * np.sin(lat_c)
    seen = H * (H - sx) >= sy * sy + (a * a) / (b * b) * sz * sz
    x = np.arcsin(-sy / np.sqrt(sx * sx + sy * sy + sz * sz))
    y = np.arctan(sz / sx)
    ax, ay = f["x"].attrs, f["y"].attrs
    col = np.round((x - float(ax["add_offset"][0])) / float(ax["scale_factor"][0])).astype(np.int32)
    row = np.round((y - float(ay["add_offset"][0])) / float(ay["scale_factor"][0])).astype(np.int32)
    nx, ny = f["x"].shape[0], f["y"].shape[0]
    ok = seen & (col >= 1) & (col < nx - 1) & (row >= 1) & (row < ny - 1)
    return np.where(ok, row, -1), np.where(ok, col, -1)


class SunnyGoes(sunny_de.SunnyDE):
    def __init__(self, fetch, log, sky=None):
        """sky: northamerica.MetarSky - the METAR cloud cover as the station truth (filled by the instance's METAR loop)."""
        super().__init__(fetch, log, None, None, stations=False, area=AMERICA)
        self._idx = None
        # Weather model (self-hosted Open-Meteo, best match = HRRR over the USA) for the model shadows, above all
        # "satnwp" (sunny_de.NWP_*: quantile-mapped model blended in beyond +30 min). Off by default: ~65k points every
        # 30 min are a new load on the shared Open-Meteo, and offline (tune_sun.py --region na, 16.11., 47 runs, HRRR 15-min
        # runs as old as they were then) the HRRR did not fit the GOES cloud mask: corrected model share error at ref
        # 0.25-0.28 (ICON-D2 vs DWD SID: median 0.14), so the skill factor keeps its weight at 0 ("satnwp" = "sat");
        # ungated it was worse on the grid (+120 MAE .167 -> .199-.209). Only a station truth can say more.
        om = os.environ.get("OPENMETEO_URL") if os.environ.get("SUNNY_NA_MODEL", "0") == "1" else None
        if om:
            self.model = sunny_de.SunModel(self.grid, om, lambda: True, log, lats=(12.0, 60.0), lons=(-135.0, -50.0), deg=0.25)
        if sky is not None:
            self.stations = sky
            self.verify = sunny_de.SunVerify(self, sky, log, prefix="na")

    def _files(self):
        """{slot (scan start rounded down to 10 min, ms): key} of the last ~3 hours."""
        out = {}
        now = time.time()
        for h in (2, 1, 0):
            d = time.strftime("%Y/%j/%H", time.gmtime(now - h * 3600))
            xml = self.fetch(f"{BUCKET}/?list-type=2&prefix={PRODUCT}/{d}/", timeout=60).decode()
            for key in re.findall(r"<Key>([^<]+\.nc)</Key>", xml):
                m = re.search(r"_s(\d{4})(\d{3})(\d{2})(\d{2})", key)
                if m:
                    t = datetime.strptime("".join(m.groups()), "%Y%j%H%M").replace(tzinfo=timezone.utc)
                    out[int(t.timestamp() * 1000) // sunny_de.STEP_MS * sunny_de.STEP_MS] = key
        return out

    def cloud_probability(self, data):
        """The file's cloud probability on the sunny grid: mean of the 3 x 3 GOES pixels (~6 km) at each cell."""
        with h5py.File(io.BytesIO(data), "r") as f:
            if self._idx is None:
                self._idx = goes_index(f, self.grid.lat, self.grid.lon)
            v = f["Cloud_Probabilities"]
            raw = v[:]
            scale = float(v.attrs["scale_factor"][0])
        row, col = self._idx
        ok = row >= 0
        acc = np.zeros(row.shape, np.float32)
        cnt = np.zeros(row.shape, np.float32)
        for dr in (-1, 0, 1):
            for dc in (-1, 0, 1):
                s = raw[np.where(ok, row + dr, 0), np.where(ok, col + dc, 0)]
                good = ok & (s != 65535)
                acc += np.where(good, s * scale, 0)
                cnt += good
        del raw
        return np.where(cnt >= 5, acc / np.maximum(cnt, 1), np.nan).astype(np.float32)

    def ingest(self, ts, data, cth_data=None):
        p = self.cloud_probability(data)
        k = np.where(np.isfinite(p), 1.0 - 0.95 * np.clip(p, 0, 1), np.nan).astype(np.float32)
        e = self.elev(ts)
        return k, share_from_k(k, e), self._display(k, e)

    def update(self):
        if self.frozen:
            return 0
        files = self._files()
        if not files:
            return 0
        newest = max(files)
        self.skipped = {t for t in self.skipped if t > newest - sunny_de.OBS_KEEP_MS}
        want = [t for t in sorted(files) if t > newest - sunny_de.OBS_KEEP_MS and t not in self.obs
                and t not in self.skipped and t > (self.ref or 0) - sunny_de.K_KEEP_MS]
        if not want:
            return 0
        if self.ref is None:  # first start: the last hour (the app's history and the motion)
            want = want[-7:]
        n = 0
        for t in want:
            data = self.fetch(f"{BUCKET}/{files[t]}", timeout=180)
            if t < (self.ref or 0):  # late slot: history only
                k, share, disp = self.ingest(t, data)
                with self.lock:
                    self.obs[t], self.disp[t] = share.astype(np.float16), disp
                    if t >= self.ref - sunny_de.K_KEEP_MS:
                        self.k[t] = k.astype(np.float16)
            else:
                self.add_slot(t, data)
            n += 1
        return n
