"""Scores the weather models against the METAR stations the way the app shows them (model_check.py logs both).

  python check_models.py [--days 14] [--region eu|na] [--by]    (in the OPERA instance: docker exec -w /app deploy-api-eu-1 ...)

Per model and view (app v1.64):
  24 h     the hourly slots 0-23 h ahead (Open-Meteo: temperature / cloud at H, precipitation and weather code of the
           hour after H = the app's slot "H"): temperature MAE (and bias), wet hours (>= 0.1 mm or a wet weather code)
           against precipitation or thunder in the METARs of that hour (present or recent weather): POD (wet hours
           caught), FAR (false wet hours), CSI; cloud MAE (0..1).
  3 Tage   4-hour blocks of 24-72 h (the app's 3-day blocks): mean temperature, block wet = any wet hour.
  7 Tage   the local days 1-6 ahead, slots 10-20 h (the app's 7-day entries): mean temperature, day wet = any wet slot,
           cloud = mean cover; --by adds the rows by lead day (where a forecast stops being useful).
"app" = the model the app shows at each station (omModel: the regional model, its gaps from ICON). A case counts for a
model when the truth, "app" and that model all exist (forecast means over the hours that have a measurement); "<"
marks the lowest temperature MAE among the models with at least 95 % of app's cases (UKMO ends at ~6.4 days).
Read it as: a model that beats "app" by a clear margin (temperature >= 0.1 C MAE, CSI >= 0.03) for 7+ days in a
region is the better source there - for that view.
"""
import glob
import json
import os
import sys
import time
from collections import defaultdict

import numpy as np

import model_check as mc

HERE = os.path.dirname(os.path.abspath(__file__))
LOG_DIR = os.environ.get("VERIFY_LOG_DIR", os.path.join(HERE, "verify_log"))
DIR = os.path.join(LOG_DIR, "models")
args = sys.argv[1:]
DAYS = float(args[args.index("--days") + 1]) if "--days" in args else 14.0
ONLY = args[args.index("--region") + 1] if "--region" in args else None
BY = "--by" in args
H = 3600_000
# station groups for the tables (ICAO prefixes)
GROUPS = {
    "eu": [("Deutschland", ("ED", "ET")), ("Österreich, Schweiz", ("LO", "LS")),
           ("Frankreich, Benelux", ("LF", "EB", "EH", "EL")), ("Großbritannien, Irland", ("EG", "EI")),
           ("Skandinavien, Island", ("EN", "ES", "EF", "EK", "BI")), ("Italien, Malta", ("LI", "LM")),
           ("Spanien, Portugal", ("LE", "LP")),
           ("Osteuropa, Baltikum", ("EP", "LK", "LZ", "LH", "EE", "EV", "EY", "UK", "UM", "LU")),
           ("Balkan, Griechenland, Türkei", ("LJ", "LD", "LQ", "LY", "LW", "LA", "LR", "LB", "LG", "LT", "LC"))],
    "na": [("USA Ost", ("K",)), ("USA West", ("K",)), ("Kanada", ("C",)), ("Alaska", ("PA",))],
}
VIEWS = (("24h", "24 Stunden (stündlich, 0-24 h)"), ("3d", "3 Tage (4-Stunden-Blöcke, 24-72 h)"),
         ("7d", "7 Tage (Tage 1-6 voraus, 10-20 Uhr Ortszeit)"))


def group_of(reg, sid, lon):
    for name, pre in GROUPS[reg]:
        if sid.startswith(pre):
            if reg == "na" and name.startswith("USA"):
                return "USA Ost" if lon > -100 else "USA West"
            return name
    return None


def load_obs(ids, t_start, n_hours):
    """Hourly truth grids (station x hour from t_start): temperature and cloud = the report nearest the full hour
    within 20 min; wet[h] = precipitation / thunder in (t + h - 1 h, t + h] (a report's present weather, or its
    recent weather, speaks for the hour it ends: reports up to 10 min after the full hour count for that hour)."""
    pos = {s: i for i, s in enumerate(ids)}
    n = len(ids)
    tc, cov = np.full((n, n_hours), np.nan, np.float32), np.full((n, n_hours), np.nan, np.float32)
    dist = np.full((n, n_hours), 1e9, np.float64)
    wet = np.full((n, n_hours), -1, np.int8)
    t_end = t_start + n_hours * H
    for day in range(int(t_start // 86400_000) - 1, int(t_end // 86400_000) + 1):
        fp = os.path.join(DIR, "obs", time.strftime("%Y%m%d", time.gmtime(day * 86400)) + ".jsonl")
        try:
            f = open(fp)
        except OSError:
            continue
        with f:
            for line in f:
                try:
                    r = json.loads(line)
                except ValueError:
                    continue
                i = pos.get(r["id"])
                if i is None:
                    continue
                t = r["t"]
                h = int((t - t_start + H // 2) // H)
                if 0 <= h < n_hours:
                    d = abs(t - (t_start + h * H))
                    if d <= 20 * 60_000 and d < dist[i, h]:
                        dist[i, h] = d
                        tc[i, h] = np.nan if r["tc"] is None else r["tc"]
                        cov[i, h] = np.nan if r["n"] is None else r["n"]
                e = int(-(-(t - 10 * 60_000 - t_start) // H))   # the hour ending at t_start + e h
                if 0 <= e < n_hours:
                    wet[i, e] = max(wet[i, e], 1 if (r["w"] or r["re"]) else 0)
    return tc, cov, wet


def wet_code(c):
    return ((c >= 51) & (c <= 67)) | ((c >= 71) & (c <= 86)) | ((c >= 95) & (c != 255))


class Acc:
    __slots__ = ("ae", "e", "n", "hit", "miss", "fa", "ce", "cn")

    def __init__(self):
        self.ae = self.e = self.ce = 0.0
        self.n = self.hit = self.miss = self.fa = self.cn = 0

    def add(self, ft, ot, fw, ow, fc, oc):
        self.ae += float(np.abs(ft - ot).sum())
        self.e += float((ft - ot).sum())
        self.n += len(ft)
        self.hit += int((fw & ow).sum())
        self.miss += int((~fw & ow).sum())
        self.fa += int((fw & ~ow).sum())
        ok = ~np.isnan(oc) & ~np.isnan(fc)
        self.ce += float(np.abs(fc[ok] - oc[ok]).sum())
        self.cn += int(ok.sum())

    def mae(self):
        return self.ae / max(1, self.n)

    def row(self):
        pod = self.hit / max(1, self.hit + self.miss)
        far = self.fa / max(1, self.hit + self.fa)
        csi = self.hit / max(1, self.hit + self.miss + self.fa)
        cl = f"{self.ce / self.cn:.2f}" if self.cn else "  - "
        return f"{self.mae():5.2f}  {self.e / max(1, self.n):+5.2f}   {pod:.2f} {far:.2f} {csi:.2f}   {cl}   {self.n}"


def cases(view, offs, t0, nh):
    """Slot index arrays per case: (n_st, n_cases, n_slots) with -1 for none, plus the lead day of each case."""
    n = len(offs)
    if view == "24h":
        return np.broadcast_to(np.arange(24)[None, :, None], (n, 24, 1)), [None] * 24, 1
    if view == "3d":
        return np.broadcast_to(np.arange(24, 72).reshape(12, 4)[None], (n, 12, 4)), [None] * 12, 3
    off = offs.astype(np.int64) * 1000
    day0 = (t0 + off) // 86400_000
    base = ((day0 * 86400_000 - off) - t0) // H          # index of local midnight of the issue day
    idx = base[:, None, None] + 24 * np.arange(1, 7)[None, :, None] + np.arange(10, 20)[None, None, :]
    idx = np.where((idx >= 0) & (idx < nh - 1), idx, -1)
    return idx, list(range(1, 7)), 7


def main():
    now = time.time() * 1000
    files = sorted(glob.glob(os.path.join(DIR, "fc", "*_*.npz")))
    files = [f for f in files if int(os.path.basename(f)[:-4].split("_")[1]) > now - DAYS * 86400_000]
    if not files:
        print("noch keine Vorhersagen gespeichert (model_check.py, seit server-v2.14)")
        return
    try:
        stations = json.load(open(os.path.join(DIR, "stations.json")))
    except (OSError, ValueError):
        stations = {}
    for reg in ("eu", "na"):
        if ONLY and reg != ONLY:
            continue
        rfiles = [f for f in files if os.path.basename(f).startswith(reg + "_")]
        st = {s[0]: s for s in stations.get(reg, [])}
        if not rfiles or not st:
            continue
        ids = list(st)
        t_start = min(int(np.load(f)["t0"]) for f in rfiles)
        n_hours = int((now - t_start) // H) + 2
        o_tc, o_cov, o_wet = load_obs(ids, t_start, n_hours)
        acc = defaultdict(Acc)   # (view, group, model[, lead day])
        models_seen = set()
        for f in rfiles:
            z = np.load(f)
            t0, zids, offs = int(z["t0"]), list(z["ids"]), z["offset"]
            models = sorted({k.split(".")[0] for k in z.files if "." in k})
            if "icon_seamless" not in models:
                continue
            nh = z["icon_seamless.temp"].shape[1]
            rows = np.array([ids.index(s) if s in st else -1 for s in zids])
            keep = rows >= 0
            data = {}
            for m in models:
                tp = z[f"{m}.temp"].astype(np.float32)
                pr = z[f"{m}.precip"].astype(np.float32)
                cd = z[f"{m}.code"].astype(np.int32)
                cv = z[f"{m}.cover"].astype(np.float32)
                cv[cv == 255] = np.nan
                data[m] = [tp, pr, cd, cv]
            app = [a.copy() for a in data["icon_seamless"]]
            for i, sid in enumerate(zids):
                s = st.get(sid)
                m = mc.om_model(s[1], s[2]) if s else "icon_seamless"
                if m != "icon_seamless" and m in data:
                    ok = ~np.isnan(data[m][0][i])
                    for v in range(4):
                        app[v][i] = np.where(ok, data[m][v][i], app[v][i])
            data["app"] = app
            names = ["app"] + [m for m in models if m not in mc.REGIONAL]
            models_seen.update(names)
            # the app's slot k: temperature / cloud at k, precipitation and code of the hour after (k + 1)
            fslots = {}
            for m in names:
                tp, pr, cd, cv = data[m]
                ft = tp[:, :-1]
                fw = (pr[:, 1:] >= 0.1) | wet_code(cd[:, 1:])
                fv = ~np.isnan(ft) & ~np.isnan(pr[:, 1:]) & (cd[:, 1:] != 255)
                fslots[m] = (ft[keep], fw[keep], cv[:, :-1][keep] / 100.0, fv[keep])
            r = rows[keep]
            col0 = int((t0 - t_start) // H)
            span = np.arange(nh - 1) + col0
            inside = (span >= 0) & (span < n_hours)
            ot = np.full((len(r), nh - 1), np.nan, np.float32)
            oc, ow = ot.copy(), np.full((len(r), nh - 1), -1, np.int8)
            ot[:, inside] = o_tc[r][:, span[inside]]
            oc[:, inside] = o_cov[r][:, span[inside]]
            we = span + 1   # wet of the hour ending at slot start + 1 h
            inside_w = (we >= 0) & (we < n_hours)
            ow[:, inside_w] = o_wet[r][:, we[inside_w]]
            tvalid = ~np.isnan(ot) & (ow >= 0) & ((t0 + np.arange(nh - 1) * H) < now - 2 * H)[None, :]
            grp = np.array([group_of(reg, ids[j], st[ids[j]][2]) or "" for j in r])
            for view, _ in VIEWS:
                idx, leads, need = cases(view, offs[keep], t0, nh)
                safe = np.where(idx >= 0, idx, 0)
                g = lambda a: np.take_along_axis(a[:, None, :].repeat(idx.shape[1], 1), safe, 2)
                tv = g(tvalid) & (idx >= 0)
                av = g(fslots["app"][3])
                for m in names:
                    ft, fw, fc, fv = (g(a) for a in fslots[m])
                    use = tv & av & fv
                    cnt = use.sum(2)
                    case = (cnt >= need) & (use | ~tv).all(2)   # every measured hour has this model and app
                    if not case.any():
                        continue
                    mean = lambda a: (np.where(use, a, 0).sum(2) / np.maximum(1, cnt))
                    f_t, o_t = mean(ft), mean(g(ot))
                    f_w, o_w = (fw & use).any(2), ((g(ow) == 1) & use).any(2)
                    ocv = g(oc)
                    cu = use & ~np.isnan(ocv)
                    f_c = np.where(cu, fc, 0).sum(2) / np.maximum(1, cu.sum(2))
                    o_c = np.where(cu, ocv, 0).sum(2) / np.maximum(1, cu.sum(2))
                    o_c[cu.sum(2) < np.maximum(1, cnt // 2)] = np.nan
                    for gname in set(grp):
                        sel = case & (grp == gname)[:, None]
                        if not gname or not sel.any():
                            continue
                        for key in ((view, gname, m), (view, "alle", m)):
                            acc[key].add(f_t[sel], o_t[sel], f_w[sel], o_w[sel], f_c[sel], o_c[sel])
                        if BY and view == "7d":
                            for c, dd in enumerate(leads):
                                s2 = sel[:, c]
                                if s2.any():
                                    acc[(view, "alle", m, dd)].add(f_t[:, c][s2], o_t[:, c][s2], f_w[:, c][s2],
                                                                   o_w[:, c][s2], f_c[:, c][s2], o_c[:, c][s2])
        title = {"eu": "Europa", "na": "Nordamerika"}[reg]
        print(f"=== {title}: {len(rfiles)} Vorhersage-Zeitpunkte, {len(st)} Stationen (METAR)")
        if not acc:
            print("noch nichts auszuwerten (die Messungen kommen erst nach den Vorhersagen)\n")
            continue
        order = ["app"] + sorted(m for m in models_seen if m != "app")
        for view, label in VIEWS:
            print(f"\n-- {label}")
            print(f"   {'Gebiet':30s} {'Modell':22s}  T-MAE  Bias    POD  FAR  CSI   Wolken  Fälle")
            for gname in ["alle"] + [g for g, _ in GROUPS[reg]]:
                rows_ = [(m, acc[(view, gname, m)]) for m in order if (view, gname, m) in acc]
                if not rows_:
                    continue
                n_app = acc[(view, gname, "app")].n if (view, gname, "app") in acc else 0
                fair = [x for x in rows_ if x[1].n >= 0.95 * n_app]
                best = min(fair, key=lambda x: x[1].mae())[0] if fair else None
                for m, a in rows_:
                    print(f"   {gname:30s} {m:22s}  {a.row()}{' <' if m == best and len(rows_) > 1 else ''}")
            if BY and view == "7d":
                print("   nach Vorlauf (alle Gebiete):")
                for dd in range(1, 7):
                    for m in order:
                        if (view, "alle", m, dd) in acc:
                            print(f"   Tag +{dd}  {m:22s}  {acc[(view, 'alle', m, dd)].row()}")
        print()


if __name__ == "__main__":
    main()
