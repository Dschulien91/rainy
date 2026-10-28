"""Scorecard: how good were our 2 h rain and sun forecasts, measured against DWD stations.

Reads what server/recorder.py logged (forecasts as issued + DWD 10-minute station observations) and prints one
compact table per kind, by lead time from the moment the forecast was available:
  sun:  MAE of the sun share per 10 min (0..1, from sunshine duration) and hit rate of "sunny or not" (>= 0.5);
        only with the sun higher than 10 deg (sunshine recorders see little below)
  rain: CSI / POD / FAR of "wet" (gauge >= 0.05 mm per 10 min, forecast >= 0.3 mm/h) and of >= 1 mm/h
Every method is compared with simple baselines: "persist" (the value at issue time held) and, for the sun,
"model" (weather model cloud cover only). A forecast is only worth showing where it beats both.

  python score.py [--hours 48] [--sun-map 0.35,0.45] [--visual out.png]
"""
import argparse
import glob
import json
import math
import os
import sys
import time

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from clouds import sun_elevation  # noqa: E402
from recorder import LOG_DIR, MAP_BOX, MAP_H, MAP_W  # noqa: E402

BUCKETS = [("jetzt", -15, 15), ("+30", 15, 45), ("+60", 45, 75), ("+90", 75, 105), ("+120", 105, 135)]
TEN = 600_000


def load_truth(kind, since):
    out = {}
    for fp in sorted(glob.glob(os.path.join(LOG_DIR, f"truth_{kind}_*.json"))):
        for sid, vals in json.load(open(fp)).items():
            d = out.setdefault(sid, {})
            for t, v in vals.items():
                if int(t) >= since:
                    d[int(t)] = v
    return out


def records(kind, since):
    for fp in sorted(glob.glob(os.path.join(LOG_DIR, f"fc_{kind}_*.npz"))):
        try:
            r = dict(np.load(fp))
        except Exception:
            continue
        if int(r["issued"]) >= since:
            yield r


def bucket(lead_min):
    for name, a, b in BUCKETS:
        if a <= lead_min < b:
            return name
    return None


# ---------- sun ----------
def sun_share(c, a, b):
    return 1 - np.clip((c - a) / b, 0, 1)


def score_sun(hours, sun_map):
    now = time.time() * 1000
    truth = load_truth("sun", now - (hours + 3) * 3600_000)
    acc = {}  # (method, bucket) -> [abs errors], [hits]
    for r in records("clouds", now - hours * 3600_000):
        ts, issued = r["ts"].astype(np.float64), float(r["issued"])
        at_issue = np.array([np.interp(issued, ts, r["cloud"][:, j]) for j in range(len(r["ids"]))])
        for j, sid in enumerate(r["ids"]):
            obs = truth.get(str(sid))
            if not obs:
                continue
            for t, sd in obs.items():
                mid = t - TEN / 2
                b = bucket((mid - issued) / 60_000)
                if b is None or not ts[0] <= mid <= ts[-1]:
                    continue
                if sun_elevation(mid, float(r["lat"][j]), float(r["lon"][j])) < 10:
                    continue
                o = min(1.0, sd * 6)  # hours of sunshine in 10 min -> share
                c = np.interp(mid, ts, r["cloud"][:, j])
                preds = {"rainy": sun_share(c, *sun_map), "persist": sun_share(at_issue[j], *sun_map)}
                m = np.interp(mid, ts, r["model"][:, j])
                if not math.isnan(m):
                    preds["model"] = 1 - m
                for k, p in preds.items():
                    e = acc.setdefault((k, b), ([], [], []))
                    e[0].append(abs(p - o)); e[1].append((p >= 0.5) == (o >= 0.5)); e[2].append(p - o)
    print("SONNE  (MAE Sonnenanteil, Treffer sonnig/nicht, Bias)   n")
    for k in ("rainy", "persist", "model"):
        cells = []
        for name, _, _ in BUCKETS:
            e = acc.get((k, name))
            cells.append(f"{name}: {np.mean(e[0]):.2f} {100 * np.mean(e[1]):3.0f}% {np.mean(e[2]):+.2f}" if e else f"{name}: -")
        n = sum(len(acc[(k, nm)][0]) for nm, _, _ in BUCKETS if (k, nm) in acc)
        print(f"  {k:8s} " + " | ".join(cells) + f"   {n}")
    return acc


# ---------- rain ----------
def score_rain(hours):
    now = time.time() * 1000
    truth = load_truth("rain", now - (hours + 3) * 3600_000)
    acc = {}  # (method, bucket, threshold) -> [hits, misses, false alarms]
    for r in records("rain", now - hours * 3600_000):
        ts, issued = r["ts"].astype(np.float64), float(r["issued"])
        for j, sid in enumerate(r["ids"]):
            obs = truth.get(str(sid))
            if not obs:
                continue
            series = r["rate"][:, j]
            at_issue = np.interp(issued, ts, series)
            for t, mm in obs.items():
                b = bucket((t - TEN / 2 - issued) / 60_000)
                if b is None or not ts[0] <= t - TEN <= t <= ts[-1]:
                    continue
                fc = np.interp([t - 7.5 * 60_000, t - 2.5 * 60_000], ts, series).mean()
                for k, p in (("rainy", fc), ("persist", at_issue)):
                    for th, (to, tf) in (("nass", (0.05, 0.3)), (">=1mm/h", (1 / 6, 1.0))):
                        e = acc.setdefault((k, b, th), [0, 0, 0])
                        wo, wf = mm >= to, p >= tf
                        e[0] += wo and wf; e[1] += wo and not wf; e[2] += wf and not wo
    print("REGEN  (CSI / POD / FAR)")
    for th in ("nass", ">=1mm/h"):
        for k in ("rainy", "persist"):
            cells = []
            for name, _, _ in BUCKETS:
                h, m, f = acc.get((k, name, th), [0, 0, 0])
                cells.append(f"{name}: {h / max(1, h + m + f):.2f} {h / max(1, h + m):.2f} {f / max(1, h + f):.2f}")
            n = sum(sum(acc.get((k, nm, th), [0, 0, 0])[:2]) for nm, _, _ in BUCKETS)
            print(f"  {th:8s} {k:8s} " + " | ".join(cells) + f"   Regenfälle {n}")
    return acc


# ---------- visual ----------
def visual(out):
    """One small image: per row (clouds, rain) now / forecast +60 / reality +60 / forecast +120 / reality +120."""
    from PIL import Image, ImageDraw
    poly = json.load(open(os.path.join(os.path.dirname(os.path.abspath(__file__)), "germany.json")))
    rings = poly["coordinates"] if poly["type"] == "Polygon" else [p[0] for p in poly["coordinates"]]

    def border(img):
        d, (x0, y0, x1, y1) = ImageDraw.Draw(img), MAP_BOX
        ym = lambda lat: math.log(math.tan(math.radians(45 + lat / 2)))
        for ring in rings:
            pts = [((lo - x0) / (x1 - x0) * img.width, (ym(y1) - ym(la)) / (ym(y1) - ym(y0)) * img.height) for lo, la in ring]
            d.line(pts + pts[:1], fill=(240, 180, 40), width=1)
        return img

    def pick(kind):
        recs = sorted(records(kind, time.time() * 1000 - 30 * 3600_000), key=lambda r: int(r["ref"]))
        def near(t):  # record whose analysis is closest to t (within 7 min)
            best = min(recs, key=lambda x: abs(int(x["ref"]) - t))
            return best if abs(int(best["ref"]) - t) <= 7 * 60_000 else None
        for r in reversed(recs):
            later = [near(int(r["ref"]) + m * 60_000) for m in (60, 120)]
            if all(x is not None and "map0" in x for x in later) and "map120" in r:
                return [r["map0"], r["map60"], later[0]["map0"], r["map120"], later[1]["map0"]], int(r["ref"])
        return None, None

    def cloud_img(a):
        g = a.astype(np.float32) / 255
        rgb = np.stack([30 + 210 * g, 40 + 200 * g, 60 + 180 * g], -1)
        return Image.fromarray(rgb.astype(np.uint8))

    def rain_img(a):
        r = a.astype(np.float32) / 10
        v = np.clip(np.log1p(r * 3) / np.log1p(60), 0, 1)
        rgb = np.where((r >= 0.1)[..., None], np.stack([40 + 0 * v, 90 + 120 * v, 160 + 95 * v], -1), np.array([25, 28, 34]))
        return Image.fromarray(rgb.astype(np.uint8))

    rows = []
    for kind, fn in (("clouds", cloud_img), ("rain", rain_img)):
        maps, ref = pick(kind)
        if maps is None:
            print(f"visual: not enough {kind} records yet (needs forecasts 2 h old)")
            continue
        rows.append([border(fn(m).resize((MAP_W // 2, MAP_H // 2))) for m in maps])
        print(f"visual {kind}: issued {time.strftime('%d.%m. %H:%M', time.gmtime(ref / 1000))} UTC; "
              "columns: now | forecast +60 | reality +60 | forecast +120 | reality +120")
    if rows:
        w, h = MAP_W // 2, MAP_H // 2
        sheet = Image.new("RGB", (5 * w + 8, len(rows) * h + 2 * (len(rows) - 1)), (255, 255, 255))
        for i, row in enumerate(rows):
            for k, im in enumerate(row):
                sheet.paste(im, (k * (w + 2), i * (h + 2)))
        sheet.save(out)
        print(f"visual: {out}")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--hours", type=float, default=48)
    ap.add_argument("--sun-map", default="0.05,0.8")
    ap.add_argument("--visual")
    a = ap.parse_args()
    score_sun(a.hours, [float(v) for v in a.sun_map.split(",")])
    score_rain(a.hours)
    if a.visual:
        visual(a.visual)
