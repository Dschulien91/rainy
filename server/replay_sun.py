"""Replay of the Germany sunny nowcast (server/sunny_de.py) on archived DWD satellite files, scored like the live server.

Runs the very same code (SunnyDE.add_slot, SunVerify) slot by slot on a folder of downloaded files, with the DWD
10-minute station sunshine (verify_log/truth_sun_<day>.json, written by server/recorder.py, or the CDC "now" files)
as the station truth. Writes sungrid_/sunstation_ lines to --out (not the live verify_log), summed up with
  python score_sun.py sat pers kpers --dir <out>
--calib instead fits the constants: the clear-sky beam (CS_A), the station time alignment, the soft threshold (RAMP),
the IR cloud top height mapping (CTH0/CTH1), and scores DWD's hourly SISfc global radiation forecast at the stations.

Files in --cache: SIDhr<YYYYmmddHHMM>EUv4.nc.bz2, CTHin<...>nEU.nc.bz2, SISfc<YYYYmmddHH>_fc+18h-DE.nc (opendata.dwd.de
weather/satellite/radiation/sid, clouds/CTH, radiation/sis; download them politely, one at a time).

  python replay_sun.py --cache DIR --from 202611080700 --to 202611081530 --out DIR [--model] [--calib]
"""
import argparse
import glob
import json
import os
import sys
import time
import urllib.request

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import sunny_de as sd  # noqa: E402
from clouds import sun_elevation  # noqa: E402
from mosaic import GermanyDwd, merc  # noqa: E402

HERE = os.path.dirname(os.path.abspath(__file__))


def get(url, tries=4):
    for k in range(tries):
        try:
            with urllib.request.urlopen(urllib.request.Request(url, headers={"User-Agent": "Mozilla/5.0"}), timeout=60) as r:
                return r.read()
        except OSError:
            if k == tries - 1:
                raise
            time.sleep(10 * (k + 1))


class ReplayStations:
    """Station truth from the recorder's daily files: {id: {end time ms: share}}; everything counts as arrived."""

    def __init__(self, truth_dir, days):
        text = get(f"{sd.SOLAR}zehn_now_sd_Beschreibung_Stationen.txt").decode("latin-1")
        self.stations = {}
        for line in text.splitlines()[2:]:
            p = line.split()
            if len(p) >= 6 and p[0].isdigit():
                self.stations[p[0]] = (float(p[4]), float(p[5]))
        self.values = {}
        for day in days:
            fp = os.path.join(truth_dir, f"truth_sun_{day}.json")
            if os.path.exists(fp):
                for sid, vals in json.load(open(fp)).items():
                    self.values.setdefault(sid, {}).update({int(t): min(1.0, v * 6.0) for t, v in vals.items() if v >= 0})
        self.values = {s: v for s, v in self.values.items() if s in self.stations}

    def truth(self, end_ts):
        return {sid: v[end_ts] for sid, v in self.values.items() if end_ts in v}

    def newest(self):
        return 1 << 62


def slots(a, b):
    return range(sd.stamp_ts(a), sd.stamp_ts(b) + 1, sd.STEP_MS)


def read(cache, prefix, t, suffix):
    fp = os.path.join(cache, f"{prefix}{sd.ts_stamp(t)}{suffix}")
    return open(fp, "rb").read() if os.path.exists(fp) else None


def replay(a):
    log = lambda m: print(m, flush=True)
    de = GermanyDwd(None, log)
    days = sorted({sd.ts_stamp(t)[:8] for t in slots(a.start, a.end)})
    st = ReplayStations(a.truth, days)
    om = os.environ.get("OPENMETEO_URL", "http://127.0.0.1:8080")
    sun = sd.SunnyDE(None, log, de, om if a.model else None, lambda: True, stations=False, log_dir=a.out)
    sun.stations = sun.verify.stations = st
    if a.model:  # one request for the whole replay period (past values of the runs at the time: optimistic)
        hours_back = (time.time() * 1000 - sd.stamp_ts(a.start)) / 3600_000 + 1
        sun.model.refresh(sun.inside(), past_steps=int(hours_back * 4) + 4, forecast_steps=16)
        log(f"model: {len(sun.model.times)} steps")
    os.makedirs(a.out, exist_ok=True)
    for fp in glob.glob(os.path.join(a.out, "sun*_*.jsonl")):
        os.remove(fp)
    t_cpu, n = 0.0, 0
    for t in slots(a.start, a.end):
        data = read(a.cache, "SIDhr", t, "EUv4.nc.bz2")
        if data is None:
            continue
        cth = read(a.cache, "CTHin", t, "nEU.nc.bz2") if float(sun.elev(t).min()) < sd.DAY_ELEV else None
        started = time.process_time()
        sun.add_slot(t, data, cth)
        t_cpu += time.process_time() - started
        n += 1
        g, s = sun.verify.step(t)
        log(f"{sd.ts_stamp(t)}: {g} grid, {s} station lines")
    for t in range(sd.stamp_ts(a.end) + sd.STEP_MS, sd.stamp_ts(a.end) + 3 * 3600_000, sd.STEP_MS):
        sun.verify.step(t)
    log(f"{n} slots, {t_cpu / max(n, 1):.2f} s CPU per slot (ingest + forecast + shadows)")


# ---------- calibration ----------
def calib(a):
    days = sorted({sd.ts_stamp(t)[:8] for t in slots(a.start, a.end)})
    st = ReplayStations(a.truth, days)
    grid = sd.SunGrid()
    pix = {}
    for sid, (lat, lon) in st.stations.items():
        col, row = (int(round(float(v))) for v in grid.to_pixels(*merc(lon, lat)))
        if 1 <= col < grid.nx - 1 and 1 <= row < grid.ny - 1 and 47 <= lat <= 55.2 and 5.8 <= lon <= 15.1:
            pix[sid] = (row, col)
    env, pairs, cth_k, rows = [], [], [], []
    for t in slots(a.start, a.end):
        data = read(a.cache, "SIDhr", t, "EUv4.nc.bz2")
        if data is None:
            continue
        lat, lon, v = sd.read_nc(data, ["DNI"])
        dni_full = v["DNI"]
        # clear-sky envelope: 99.5th percentile of DNI per elevation bin over the whole image
        la, lo = np.meshgrid(lat[::4], lon[::4], indexing="ij")
        e_full = sun_elevation(t, la, lo)
        d_full = dni_full[::4, ::4]
        for e0 in range(3, 40, 2):
            m = (e_full >= e0) & (e_full < e0 + 2) & np.isfinite(d_full)
            if m.sum() > 2000:
                env.append((e0 + 1, float(np.percentile(d_full[m], 99.5))))
        dni = grid.regrid(lat, lon, dni_full, sigma=(0.8, 0.5))
        e = grid.elev(t)
        cth = read(a.cache, "CTHin", t, "nEU.nc.bz2")
        if cth is not None:
            la2, lo2, c = sd.read_nc(cth, ["CTH"])
            cth_m = grid.regrid(la2, lo2, c["CTH"])
            m = (e >= 15) & np.isfinite(cth_m) & np.isfinite(dni)
            k = dni / sd.clear_dni(e)
            sel = np.flatnonzero(m.ravel())[::50]
            cth_k.extend(zip(cth_m.ravel()[sel], k.ravel()[sel]))
        rows.append((t, dni, e))
    # 1) clear-sky beam: CS_A that matches the envelope
    env = np.array(env)
    am = 1.0 / (np.sin(np.radians(env[:, 0])) + 0.50572 * (env[:, 0] + 6.07995) ** -1.6364)
    best = min((float(np.mean(np.log(1361.0 * A ** (am ** 0.678) / env[:, 1]) ** 2)), A) for A in np.arange(0.60, 0.90, 0.01))
    print(f"CS_A fit: {best[1]:.2f} (rms log error {best[0] ** 0.5:.3f}); envelope by elevation:",
          " ".join(f"{e:.0f}:{d:.0f}" for e, d in env[np.argsort(env[:, 0])][::max(1, len(env) // 12)]))
    # 2) time alignment and soft threshold at the stations (analysis vs SD_10)
    print("station MAE / hit rate of the analysis, by alignment (station interval end - slot) and RAMP:")
    for lag in (0, 10, 20):
        for ramp in (20, 40, 80, 120, 200):
            for win in (1, 3, 5):
                err, hit = [], []
                for t, dni, e in rows:
                    truth = st.truth(t + lag * 60_000)
                    s = np.clip((dni - (sd.DNI_SUN - ramp)) / (2 * ramp), 0, 1)
                    s = np.where(e < sd.SUN_ELEV, 0, s)
                    h = win // 2
                    for sid, (r, c) in pix.items():
                        if sid in truth and sun_elevation(t, *st.stations[sid]) >= sd.SCORE_ELEV:
                            f = float(np.nanmean(s[r - h:r + h + 1, c - h:c + h + 1]))
                            if np.isfinite(f):
                                err.append(abs(f - truth[sid]))
                                hit.append((f >= 0.5) == (truth[sid] >= 0.5))
                if err and (lag == 10 or (ramp == 80 and win == 3)):
                    print(f"  lag +{lag:2d} RAMP {ramp:3d} window {win}: MAE {np.mean(err):.3f}  hit {np.mean(hit):.3f}  n {len(err)}")
    # 3) IR cloud top height -> k
    if cth_k:
        ok = np.array(cth_k, np.float32)
        print("median daytime k (DNI) by IR cloud top height (km):", " ".join(
            f"{o0 / 1000:.0f}-{(o0 + 1000) / 1000:.0f}:{np.median(ok[(ok[:, 0] >= o0) & (ok[:, 0] < o0 + 1000), 1]):.2f}"
            for o0 in range(0, 12001, 1000) if ((ok[:, 0] >= o0) & (ok[:, 0] < o0 + 1000)).sum() > 50))
        for o0, o1 in ((2000, 4000), (2500, 4000), (3000, 4000), (3000, 4500), (3200, 3800)):
            kk = np.clip((o1 - ok[:, 0]) / (o1 - o0), 0, 1)
            print(f"  CTH {o0}-{o1}: MAE vs k {np.mean(np.abs(kk - np.clip(ok[:, 1], 0, 1))):.3f}, sunny agreement "
                  f"{np.mean((kk >= 0.5) == (ok[:, 1] >= 0.5)):.3f}")
    # 4) DWD SISfc (hourly global radiation forecast) at the stations, as a sun share via the global clear-sky index
    sisfc(a, st, grid, pix)


def sisfc(a, st, grid, pix):
    from scipy.io import netcdf_file
    res = {}
    for fp in sorted(glob.glob(os.path.join(a.cache, "SISfc*_fc+18h-DE.nc"))):
        run = sd.stamp_ts(os.path.basename(fp)[5:15] + "00")
        f = netcdf_file(fp, "r", mmap=False)
        lat, lon = np.array(f.variables["lat"][:], np.float64), np.array(f.variables["lon"][:], np.float64)
        tt = np.array(f.variables["time"][:], np.float64)
        base = os.path.basename(fp)[5:13]
        sis = np.array(f.variables["SIS"][:], np.float32)
        f.close()
        for i, h in enumerate(tt):
            valid = sd.stamp_ts(base + "0000") + int(round(h * 3600_000))
            lead = (valid - run) // 60_000
            if lead not in (0, 60, 120):
                continue
            g = grid.regrid(lat, lon, np.where(sis[i] < 0, np.nan, sis[i]))
            e = grid.elev(valid)
            ghi_cs = 1098.0 * np.sin(np.radians(np.maximum(e, 0.5))) * np.exp(-0.057 / np.sin(np.radians(np.maximum(e, 0.5))))
            kg = g / ghi_cs
            for share_of_k in ((0.55, 0.85),):
                s = np.where(e < sd.SUN_ELEV, 0, np.clip((kg - share_of_k[0]) / (share_of_k[1] - share_of_k[0]), 0, 1))
                for end in (valid, valid + 600_000):  # the hour's value vs the 10 min around it
                    truth = st.truth(end)
                    for sid, (r, c) in pix.items():
                        if sid in truth and sun_elevation(valid, *st.stations[sid]) >= sd.SCORE_ELEV:
                            fv = float(np.nanmean(s[r - 1:r + 2, c - 1:c + 2]))
                            if np.isfinite(fv):
                                res.setdefault(lead, []).append((fv, truth[sid]))
    for lead, p in sorted(res.items()):
        p = np.array(p)
        print(f"SISfc +{lead:3d} min: station MAE {np.mean(np.abs(p[:, 0] - p[:, 1])):.3f}, hit "
              f"{np.mean((p[:, 0] >= 0.5) == (p[:, 1] >= 0.5)):.3f}, n {len(p)}")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--cache", required=True)
    ap.add_argument("--from", dest="start", required=True)
    ap.add_argument("--to", dest="end", required=True)
    ap.add_argument("--out", default=os.path.join(HERE, "verify_log", "replay"))
    ap.add_argument("--truth", default=os.path.join(HERE, "verify_log"))
    ap.add_argument("--model", action="store_true", help="also the model shadows (self-hosted Open-Meteo, past values)")
    ap.add_argument("--calib", action="store_true")
    a = ap.parse_args()
    calib(a) if a.calib else replay(a)
