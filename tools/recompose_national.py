"""Rebuild the national layers of the own Europe picture (eun, server/national.py) offline for one time step, to
trace a map artefact to its source (DWD layer, Swiss layer, their weights, or the mix).

It runs national.py's own GermanyRv and SwissRzc code (to_host: resampling, coverage, feather weights) on files
fetched anonymously from the sources (DWD open data RV, MeteoSwiss open data RZC; both keep about a day), mixes them
like national.compose (OPERA left out: it only counts where the layers' weights sum below 1, i.e. near their edges),
and draws a lat/lon box in the app's display classes (mosaic.class_q): composite, DWD alone, Swiss alone, DWD share.
22.11.2026: at 10:00Z the composite matched the live eun data tile almost texel for texel over the Black Forest
(0.85 = that hour's gauge adjust factor for Germany), so it is a faithful stand-in for the observed picture there.

Usage (repo root):
  python -I tools/recompose_national.py 20261122_0800 --box 7.7,47.25,8.3,47.55 --png C:/tmp/aarau.png [--adjust 0.85]
"""
import argparse
import io
import json
import os
import re
import sys
import tarfile
import urllib.request
from datetime import datetime

import cv2
import h5py
import numpy as np
from PIL import Image
from pyproj import Transformer

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "server"))
import national  # noqa: E402
from mosaic import class_q  # noqa: E402

UA = {"User-Agent": "Mozilla/5.0"}   # anonymous, neutral


def get(url):
    return urllib.request.urlopen(urllib.request.Request(url, headers=UA), timeout=120).read()


class Host:
    """The eun host grid (rainy_server.EuropeOpera with EU_KM = 2)."""
    LAEA = "+proj=laea +lat_0=55.0 +lon_0=10.0 +x_0=1950000.0 +y_0=-2100000.0 +units=m +ellps=WGS84"
    CELL, NY, NX, KM_PER_PX = 2000.0, 2200, 1900, 2.0

    def __init__(self):
        self.tr = Transformer.from_crs("EPSG:3857", self.LAEA, always_xy=True)
        self.to_ll = Transformer.from_crs(self.LAEA, "EPSG:4326", always_xy=True)

    def to_pixels(self, x, y):
        gx, gy = self.tr.transform(x, y)
        return np.asarray(gx) / self.CELL - 0.5, -np.asarray(gy) / self.CELL - 0.5

    def pixel_lonlat(self, c, r):
        return self.to_ll.transform((np.asarray(c) + 0.5) * self.CELL, -(np.asarray(r) + 0.5) * self.CELL)


def layers(stamp):
    tar = tarfile.open(fileobj=io.BytesIO(get(f"{national.GermanyRv.RV}composite_rv_{stamp}.tar")))
    m = next(m for m in tar.getmembers() if m.name.rsplit("_", 1)[1].startswith("000"))
    rv = tar.extractfile(m).read()
    t = datetime.strptime(stamp, "%Y%m%d_%H%M")
    d = json.loads(get(f"{national.SwissRzc.STAC}/{stamp[:8]}-ch"))
    rzc = get(d["assets"][next(k for k in d["assets"] if re.match(rf"rzc{t:%y%j%H%M}..\.001\.h5$", k))]["href"])

    class De(national.GermanyRv):
        def geometry(self):
            with h5py.File(io.BytesIO(rv), "r") as f:
                self.PROJ, self.X0, self.Y1, self.CELL, self.NX, self.NY = national.odim_grid(f)

        def _analysis(self, ts):
            return h5py.File(io.BytesIO(rv), "r")

    class Ch(national.SwissRzc):
        def __init__(self, *a):
            national.Layer.__init__(self, *a)

        def geometry(self):
            with h5py.File(io.BytesIO(rzc), "r") as f:
                self.PROJ, self.X0, self.Y1, self.CELL, self.NX, self.NY = national.odim_grid(f)

        def load(self, ts):
            with h5py.File(io.BytesIO(rzc), "r") as f:
                a = np.asarray(f["dataset1/data1/data"][:], np.float32)
            return np.where(np.isfinite(a) & (a >= 0), np.minimum(a, 300.0), -1.0).astype(np.float32)
    return De, Ch


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("stamp", help="UTC YYYYMMDD_HHMM (5-min step)")
    ap.add_argument("--box", default="7.7,47.25,8.3,47.55", help="lon0,lat0,lon1,lat1")
    ap.add_argument("--px", type=float, default=0.8, help="km per output pixel (~ zoom-8 data texel)")
    ap.add_argument("--adjust", type=float, default=1.0, help="gauge adjust factor of that hour (log: 'gauge adjust (nat)')")
    ap.add_argument("--png", required=True)
    ap.add_argument("--npz", default="")
    a = ap.parse_args()
    host = Host()
    full = {}
    for cls in layers(a.stamp):
        l = cls(host, print, UA["User-Agent"])
        rate, w = l.to_host(l.load(0))
        for name, v in (("rate", rate), ("w", w)):
            g = np.zeros((host.NY, host.NX), np.float32)
            g[l.win] = np.nan_to_num(v.astype(np.float32), nan=0.0)
            full[f"{l.name}_{name}"] = g
    dw, cw = full["de_w"], full["ch_w"]
    den = dw + cw
    comp = np.where(den > 0, (dw * np.maximum(full["de_rate"], 0) + cw * np.maximum(full["ch_rate"], 0)) / np.maximum(den, 1e-6), 0)
    lon0, lat0, lon1, lat1 = (float(v) for v in a.box.split(","))
    ny = max(2, int(round((lat1 - lat0) * 111.2 / a.px)))
    nx = max(2, int(round((lon1 - lon0) * 111.2 * np.cos(np.radians((lat0 + lat1) / 2)) / a.px)))
    lon, lat = np.meshgrid(np.linspace(lon0, lon1, nx), np.linspace(lat1, lat0, ny))
    gx, gy = Transformer.from_crs("EPSG:4326", Host.LAEA, always_xy=True).transform(lon, lat)
    mc, mr = (gx / Host.CELL - 0.5).astype(np.float32), (-gy / Host.CELL - 0.5).astype(np.float32)
    smp = lambda g: cv2.remap(g.astype(np.float32), mc, mr, cv2.INTER_LINEAR)
    show = lambda v: ((class_q(v * a.adjust).astype(np.int16) // 8) * 30).clip(0, 255).astype(np.uint8)
    panels = [show(smp(comp)), show(smp(full["de_rate"])), show(smp(full["ch_rate"])),
              (smp(np.where(den > 0, dw / np.maximum(den, 1e-6), 0)) * 255).astype(np.uint8)]
    sep = np.full((ny, 3), 255, np.uint8)
    im = np.concatenate([x for p in panels for x in (p, sep)][:-1], 1)
    Image.fromarray(im).resize((im.shape[1] * 3, im.shape[0] * 3), Image.NEAREST).save(a.png)
    if a.npz:
        np.savez_compressed(a.npz, comp=comp, **full)
    print(f"{a.png}: composite | DWD layer | Swiss layer | DWD share (white = DWD only)")


if __name__ == "__main__":
    main()
