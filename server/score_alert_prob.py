"""Probabilistic rain alerts (alerts.RAIN_ALERT_PROB): calibration and alert quality against rain gauges.

Compares the live rule (alerts.evaluate: >= P_START of the 5 km circle wet for MIN_STEPS steps) with the
probabilistic trigger (alerts.rain_probability: alert at the first forecast step with P >= threshold) on the same
recorded alert inputs, and checks how honest the probability is. Optionally refits the coefficients.

Data (written by the running instances, nothing is fetched):
  de       Germany instance: the recorder's verify_log/fc_rain_*.npz ("share" = wet share of the alert circle in
           DWD's forecast, which is what the Germany alerts use; since 12.11.2026) + truth_rain_*.json (DWD
           10-minute gauges).
  fcalert  Europe instances: <LOG_DIR>/fcalert/<ref>.npz (rainy_server.record_alert_inputs, since 16.11.2026) + the
           gauge archive (check_national_gauges), 10-minute networks only.
           North America (since 17.11.2026): <LOG_DIR>/fcalert + the 10-minute archive <LOG_DIR>/gaugetruth
           (minute_gauges.py: US ASOS 5-minute reports, Canadian SWOB minute records); only those ~1100 gauges,
           the others have hourly METAR / SWOB truth only (too coarse for a +-15 min window). A US interval with
           precipitation as present weather counts as wet too (the 0.01 inch gauge alone misses light rain for
           tens of minutes: verify_alerts_eu.py); --gauge-only: the gauge alone.
  Default: de when EDITION is de (or unset), fcalert otherwise.

Truth and scores (as verify_alerts.py / verify_alerts_eu.py, so the numbers compare):
  rain at a step   a gauge measures >= 0.05 mm (0.3 mm/h, = alerts.WET) in a 10-minute interval overlapping the
                   step's time +- 15 min (verify_alerts.window); unknown when no wet value and a gap
  eligible         the circle is dry now (share < P_START * 0.6, else "raining now", no alert) and the gauge
                   measured < 0.05 mm in the 10-minute sums ending issue - 20 .. issue (missing = dry)
  FAR              1 - share of alerts with rain in the window of the predicted start; alerts deduplicated per gauge
                   and lead bucket (a start within 30 min of the last counted one is the same event)
  POD              rain onsets (wet interval after >= 60 min known dry) for which some run issued l0 .. l1 min
                   before the onset alerted with a window containing it (only onsets with such a run)
  lead error       for hits: first wet interval after the issue (its middle) minus the predicted start, minutes;
                   median / median of the absolute value (10-minute gauges: +-5 min resolution)
  reliability      every eligible forecast step near rain (P > 0): predicted P vs how often it rained
  Brier            on the same rows; skill vs the lead-bucket climatology and vs the live rule read as a
                   probability (alerts.HIT_RATE where the rule's quantity >= P_START, else the rest's base rate)
Not simulated (as in the verify scripts): cooldown, MAX_PER_DAY, quiet hours, early / upgrade alerts.

  docker exec -w /app deploy-api-1 python score_alert_prob.py                 (Germany)
  docker exec -w /app deploy-api-eun-1 python score_alert_prob.py             (Europe, own picture; eu: OPERA)
  docker exec -w /app deploy-api-na-1 python score_alert_prob.py [--gauge-only] (North America, 10-minute gauges)
  options: --days N (newest days only; default all for de, 2 for fcalert), --thr 0.6,0.65,0.7, --fit
           (leave-one-day-out refit: held-out reliability and the RAIN_ALERT_COEF line for the instance),
           --all-nets (Europe: also SUSPECT_NETS), --source de|fcalert
Memory: about 0.6 GB per day of a Europe instance (~650k run x gauge samples), 0.3 GB for 3 weeks of Germany.
"""
import argparse
import glob
import json
import os
import sys
import time

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import alerts  # noqa: E402

TEN, HOUR, STEP = 600_000, 3600_000, 300_000
NS = 24                                        # forecast steps, +5 .. +120 min
BUCKETS = [(0, 15), (15, 30), (30, 60), (60, 120)]
WET10 = 0.05
CHUNK = 100_000
# Europe networks whose 10-minute truth disagrees with the radar far beyond the rest (16.11.2026, eun: gauge wet in
# the interval after a run whose circle was >= 50 % wet: en 37 %, ch 42 %, bf 32 %, at 21 %, sc 1 % vs 61-95 % for
# de / pl / cz / hu / fi / se / dk / bw / gl / be): tipping buckets of 0.2 mm, sums spread over 15 min, gaps. They
# are left out unless --all-nets; the "truth check" line is printed on every run to re-check this list.
SUSPECT_NETS = ("en", "sc", "at", "bf", "ch")


# ---------------- data ----------------
def log_dir():
    return os.environ.get("VERIFY_LOG_DIR", os.path.join(os.path.dirname(os.path.abspath(__file__)), "verify_log"))


def truth_matrix(n, t0, t1):
    return np.full((n, (t1 - t0) // TEN + 1), np.nan, np.float32)


def load_de(days):
    """Recorder files -> samples (one per run and gauge): S / R (N, 1 + NS) share and rate (column 0 = now),
    issue, g, t1 (first forecast step) and the truth matrix tm (gauges, 10-minute bins from t0)."""
    d = log_dir()
    since = (time.time() - days * 86400) * 1000 if days else 0
    recs = []
    for fp in sorted(glob.glob(os.path.join(d, "fc_rain_*.npz"))):
        try:
            if int(os.path.basename(fp)[:-4].split("_")[3]) < since:
                continue
            z = np.load(fp)
            if "share" not in z.files:
                continue
            recs.append({k: z[k] for k in ("issued", "ts", "ids", "fc", "share", "rate")})
        except Exception:
            continue                                # the newest file may still be being written
    if not recs:
        sys.exit(f"no recorder file with 'share' in {d}")
    ids = sorted({str(x) for r in recs for x in r["ids"]})
    gi = {s: k for k, s in enumerate(ids)}
    t0 = (min(int(r["issued"]) for r in recs) - 3 * HOUR) // HOUR * HOUR
    t1 = max(int(r["issued"]) for r in recs) + 4 * HOUR
    tm = truth_matrix(len(ids), t0, t1)
    lo_day, hi_day = time.strftime("%Y%m%d", time.gmtime(t0 / 1000)), time.strftime("%Y%m%d", time.gmtime(t1 / 1000))
    for fp in sorted(glob.glob(os.path.join(d, "truth_rain_*.json"))):
        if not lo_day <= fp[-13:-5] <= hi_day:
            continue
        with open(fp) as f:
            for sid, vals in json.load(f).items():
                k = gi.get(sid)
                if k is None:
                    continue
                for t, v in vals.items():
                    t = int(t)
                    if t % TEN == 0 and t0 <= t <= t1 and v is not None and v >= 0:
                        tm[k, (t - t0) // TEN] = v
    S, R, issue, g, first = [], [], [], [], []
    for r in recs:
        fc = r["fc"].astype(bool)
        t = r["ts"].astype(np.int64)
        oi = np.flatnonzero(~fc)
        if not len(oi):
            continue
        fi = np.flatnonzero(fc & (t > t[oi[-1]]))[:NS]
        if len(fi) < NS or np.any(np.diff(t[fi]) != STEP):
            continue
        cols = np.concatenate([[oi[-1]], fi])
        n = len(r["ids"])
        S.append(r["share"][cols].T.astype(np.float32))
        R.append(r["rate"][cols].T.astype(np.float32))
        g.append([gi[str(x)] for x in r["ids"]])
        issue.append(np.full(n, int(r["issued"]), np.int64))
        first.append(np.full(n, int(t[fi[0]]), np.int64))
    return {"S": np.concatenate(S), "R": np.concatenate(R), "issue": np.concatenate(issue),
            "g": np.concatenate(g).astype(np.int32), "t1": np.concatenate(first), "tm": tm, "t0": t0,
            "net": np.array(["de"] * len(ids))}


def load_fcalert(days, all_nets, wx=False):
    import check_national_gauges as cng
    d = cng.LOG_DIR
    na = os.path.basename(d.rstrip("/\\")) == "na"
    st = []
    if na:   # North America: the gauge list of northamerica.MetarGauges (fcalert's columns)
        try:
            with open(os.path.join(d, "metar", "stations.json")) as f:
                st = [tuple(s) for s in json.load(f)]
        except (OSError, ValueError):
            st = []
    else:
        st = cng.stations()
    if not st and not na:
        try:
            with open(os.path.join(cng.OBS_DIR, "stations.json")) as f:
                st = [tuple(s) for s in json.load(f)]
        except (OSError, ValueError):
            st = []
    if not st:
        sys.exit("no gauge list")
    fca = os.path.join(d, "fcalert")
    refs = sorted(int(os.path.basename(p)[:-4]) for p in glob.glob(os.path.join(fca, "*.npz")) if os.path.basename(p)[:-4].isdigit())
    refs = [r for r in refs if r >= (time.time() - days * 86400) * 1000]
    if not refs:
        sys.exit(f"no fcalert run in {fca} in the last {days:g} days")
    net = np.array([s[0] for s in st])
    use_g = ~np.isin(net, ("sy", "pt", "ct"))           # hourly networks (verify_alerts_eu.HOURLY_NETS)
    if not all_nets and not na:
        use_g &= ~np.isin(net, SUSPECT_NETS)
    t0 = (refs[0] - 3 * HOUR) // HOUR * HOUR
    t1 = refs[-1] + 4 * HOUR
    tm = truth_matrix(len(st), t0, t1)
    idx = {f"{s[0]}:{s[1]}": k for k, s in enumerate(st)}
    for day in range(int(t0 // 86400_000), int(t1 // 86400_000) + 1):
        wxd = {}
        if wx:   # North America: the 5-minute reports' present weather (minute_gauges.py "wx")
            try:
                z = np.load(cng._day_file(day))
                if "wx" in z.files:
                    wxd = dict(zip((str(x) for x in z["ids"]), z["wx"]))
            except (OSError, ValueError, KeyError):
                pass
        for key, row in cng._day_load(day).items():   # one day in memory at a time
            k = idx.get(key)
            if k is None:
                continue
            if key in wxd:
                row = np.where(wxd[key] == 1, np.fmax(row, WET10), row)
            tt = day * 86400_000 + np.arange(144, dtype=np.int64) * TEN
            ok = np.isfinite(row) & (tt >= t0) & (tt <= t1)
            tm[k, (tt[ok] - t0) // TEN] = row[ok]
    if na:   # only the gauges of the 10-minute archive
        use_g &= np.isfinite(tm).any(axis=1)
        if not use_g.any():
            sys.exit(f"North America: no 10-minute gauge truth in {cng.TRUTH_DIR} for these runs (minute_gauges.py)")
    S, R, issue, g = [], [], [], []
    for ref in refs:
        try:
            z = np.load(os.path.join(fca, f"{ref}.npz"))
            share, peak = z["share"], z["peak"]
        except Exception:
            continue
        n = min(share.shape[1], len(st))
        if share.shape[0] < NS + 1:
            continue
        gg = np.flatnonzero(use_g[:n] & (share[0, :n] != 255))
        raw = share[:NS + 1, gg].T
        sh = raw.astype(np.float32) / 250
        sh[raw == 255] = np.nan
        S.append(sh)
        R.append(peak[:NS + 1, gg].T.astype(np.float32))
        g.append(gg.astype(np.int32))
        issue.append(np.full(len(gg), ref, np.int64))
    issue = np.concatenate(issue)
    return {"S": np.concatenate(S), "R": np.concatenate(R), "issue": issue, "g": np.concatenate(g),
            "t1": issue + STEP, "tm": tm, "t0": t0, "net": net}


def add_truth(d):
    """Y (N, NS): rain in the window of each step (1 / 0 / -1 unknown); gdry: gauge dry at issue; onset: middle of
    the first wet interval after the issue (within 150 min) or -1; lead (N, NS) minutes."""
    tm, t0 = d["tm"], d["t0"]
    nb = tm.shape[1]
    g = d["g"]

    def at(b):
        ok = (b >= 0) & (b < nb)
        v = np.full(len(b), np.nan, np.float32)
        v[ok] = tm[g[ok], b[ok]]
        return v

    Y = np.empty((len(g), NS), np.int8)
    for i in range(NS):
        s = d["t1"] + i * STEP
        lo = (s - 900_000 + TEN - 1) // TEN * TEN
        wet = np.zeros(len(g), bool)
        known = np.ones(len(g), bool)
        for j in range(4):                              # window(): interval ends in [s - 15, s + 25) min
            e = lo + j * TEN
            inwin = e < s + 1500_000
            v = at(((e - t0) // TEN).astype(np.int64))
            wet |= inwin & (v >= WET10)
            known &= ~inwin | np.isfinite(v)
        Y[:, i] = np.where(wet, 1, np.where(known, 0, -1))
    d["Y"] = Y
    bi = (d["issue"] // TEN * TEN - t0) // TEN
    d["gdry"] = ~np.any([at(bi + j) >= WET10 for j in (-2, -1, 0)], axis=0)
    onset = np.full(len(g), -1, np.int64)
    for j in range(15, 0, -1):
        w = at(bi + j) >= WET10
        onset[w] = t0 + (bi[w] + j) * TEN - TEN // 2
    d["onset"] = onset
    d["lead"] = (((d["t1"] - d["issue"]) / 60_000)[:, None] + 5 * np.arange(NS)[None]).astype(np.float32)
    d["day"] = d["issue"] // 86400_000


def probabilities(d, coef=None, sel=None):
    """alerts.rain_probability for all (or the selected) samples, in chunks. -> (N, NS) float32"""
    idx = np.arange(len(d["S"])) if sel is None else np.flatnonzero(sel)
    P = np.zeros((len(d["S"]), NS), np.float32)
    for c in range(0, len(idx), CHUNK):
        k = idx[c:c + CHUNK]
        P[k] = alerts.rain_probability(d["S"][k, 1:], d["S"][k, 0], d["lead"][k], coef)
    return P


# ---------------- rules and scores ----------------
def rule_starts(d):
    F = np.nan_to_num(d["S"][:, 1:], nan=0.0)
    ok = F >= alerts.P_START
    for k in range(1, alerts.MIN_STEPS):
        ok[:, :-k] &= F[:, k:] >= alerts.P_START
        ok[:, -k:] = False
    ok &= d["lead"] >= 2.5
    return np.where(ok.any(axis=1), ok.argmax(axis=1), -1)


def prob_starts(P, lead, thr):
    ok = (P >= thr) & (lead >= 2.5)
    return np.where(ok.any(axis=1), ok.argmax(axis=1), -1)


def score(d, el, start, shown):
    """Per lead bucket: n alerts, FAR, lead error (median, median |.|), mean shown P, observed hit rate."""
    idx = np.flatnonzero(el & (start >= 0))
    k = start[idx]
    lead, y, s = d["lead"][idx, k], d["Y"][idx, k], d["t1"][idx] + k * STEP
    g, onset = d["g"][idx], d["onset"][idx]
    order = np.argsort(d["issue"][idx], kind="stable")
    out = []
    for l0, l1 in BUCKETS:
        last, n, h, errs, ps = {}, 0, 0, [], []
        for j in order:
            if not (l0 < lead[j] <= l1) or y[j] < 0:
                continue
            pv = last.get(g[j])
            if pv is not None and abs(s[j] - pv) <= 30 * 60_000:
                continue
            last[g[j]] = s[j]
            n += 1
            h += int(y[j])
            ps.append(shown[idx[j]])
            if y[j] == 1 and onset[j] > 0:
                errs.append((onset[j] - s[j]) / 60_000)
        out.append({"n": n, "hits": h, "far": 1 - h / n if n else np.nan, "shown": float(np.sum(ps)),
                    "err": (np.median(errs), np.median(np.abs(errs))) if errs else (np.nan, np.nan)})
    return out


def onsets(d):
    gauges = np.unique(d["g"])
    x = d["tm"][gauges]
    dry = (x < WET10).astype(np.int32)
    cs = np.concatenate([np.zeros((len(x), 1), np.int32), np.cumsum(dry, axis=1)], axis=1)
    dry6 = np.zeros_like(dry, bool)
    dry6[:, 6:] = (cs[:, 6:-1] - cs[:, :-7]) == 6
    lo, hi = d["issue"].min() + 120 * 60_000, d["issue"].max()
    out = [(int(gauges[r]), d["t0"] + int(b) * TEN) for r, b in np.argwhere((x >= WET10) & dry6)]
    return [(g, t) for g, t in out if lo <= t <= hi]


def pod(d, start, ons, by_g):
    out = []
    for l0, l1 in BUCKETS:
        n = c = 0
        for g, t in ons:
            o = t - TEN // 2
            js = [j for j in by_g.get(g, ()) if o - l1 * 60_000 <= d["issue"][j] <= o - max(l0, 2.5) * 60_000]
            if not js:
                continue
            n += 1
            c += any(start[j] >= 0 and -900_000 <= t - (d["t1"][j] + start[j] * STEP) < 1500_000 for j in js)
        out.append((c, n))
    return out


def fit_logistic(X, y, lam=1e-3, it=50):
    Xb = np.concatenate([np.ones((len(X), 1)), X], axis=1).astype(np.float64)
    beta = np.zeros(Xb.shape[1])
    reg = lam * len(y) * np.diag(np.r_[0, np.ones(len(beta) - 1)])
    for _ in range(it):
        p = 1 / (1 + np.exp(-(Xb @ beta)))
        step = np.linalg.solve((Xb * (p * (1 - p))[:, None]).T @ Xb + reg, Xb.T @ (p - y) + reg @ beta)
        beta -= step
        if np.abs(step).max() < 1e-8:
            break
    return beta


def fit_rows(d, cand_sel):
    """Features and labels of the candidate steps of the selected samples (fit input)."""
    Xs, ys = [], []
    idx = np.flatnonzero(cand_sel.any(axis=1))
    for c in range(0, len(idx), CHUNK):
        k = idx[c:c + CHUNK]
        X = alerts.prob_features(d["S"][k, 1:], d["S"][k, 0], d["lead"][k])
        m = cand_sel[k]
        Xs.append(X[m])
        ys.append(d["Y"][k][m].astype(np.float64))
    return np.concatenate(Xs), np.concatenate(ys)


def brier(p, y):
    return float(np.mean((p - y) ** 2))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--source", choices=("de", "fcalert"))
    ap.add_argument("--days", type=float, help="only the newest N days (default: all for de, 2 for fcalert)")
    ap.add_argument("--thr", default="0.5,0.55,0.6,0.65,0.7,0.75")
    ap.add_argument("--fit", action="store_true", help="leave-one-day-out refit and the RAIN_ALERT_COEF line")
    ap.add_argument("--all-nets", action="store_true", help="Europe: include SUSPECT_NETS")
    ap.add_argument("--gauge-only", action="store_true",
                    help="North America: the gauge alone (default: also precipitation as present weather = wet)")
    a = ap.parse_args()
    src = a.source or ("de" if os.environ.get("EDITION", "de") == "de" else "fcalert")
    d = load_de(a.days or 0) if src == "de" else load_fcalert(a.days or 2, a.all_nets, not a.gauge_only)
    add_truth(d)
    S, lead, Y = d["S"], d["lead"], d["Y"]
    el = np.isfinite(S[:, 0]) & (S[:, 0] < alerts.P_START * 0.6) & d["gdry"]
    days = sorted(set(d["day"].tolist()))
    print(f"source {src}; runs {len(set(d['issue'].tolist()))} "
          f"({time.strftime('%d.%m. %H:%M', time.gmtime(d['issue'].min() / 1000))} - "
          f"{time.strftime('%d.%m. %H:%M', time.gmtime(d['issue'].max() / 1000))} UTC, {len(days)} UTC days), "
          f"gauges {len(np.unique(d['g']))}, run x gauge samples {len(S)}, eligible (dry now) {int(el.sum())}")
    print(f"probabilistic trigger in this process: RAIN_ALERT_PROB {'on' if alerts.PROB_ON else 'off'}, "
          f"P_ALERT {alerts.P_ALERT:.2f}")
    if src == "fcalert":
        b = (d["issue"] // TEN * TEN + TEN - d["t0"]) // TEN
        v = d["tm"][d["g"], np.clip(b, 0, d["tm"].shape[1] - 1)]
        nets = d["net"][d["g"]]
        cells = []
        for nt in sorted(set(nets.tolist())):
            m = (nets == nt) & (S[:, 0] >= 0.5) & np.isfinite(v)
            if m.sum() >= 30:
                cells.append(f"{nt} {np.mean(v[m] >= WET10):.0%} ({int(m.sum())})")
        print("truth check, gauge wet in the 10 min after a run with >= 50 % of its circle wet (should be high):\n  "
              + ", ".join(cells) + ("" if a.all_nets or not np.isin(d["net"], SUSPECT_NETS).any()
                                else f"; left out: {', '.join(SUSPECT_NETS)}"))

    P = probabilities(d)
    title = "coefficients in use (" + ("RAIN_ALERT_COEF" if alerts.PROB_COEF != alerts.PROB_COEF_DE else "Germany default") + ")"
    if a.fit:
        cand = (P > 0) & el[:, None] & (Y >= 0)       # P > 0: near rain (m >= 0.05) and lead >= 2.5
        if len(days) > 1:
            Pcv = np.zeros_like(P)
            for dd in days:
                te = d["day"] == dd
                X, y = fit_rows(d, cand & ~te[:, None])
                if len(y) < 1000 or not te.any():
                    continue
                Pcv[te] = probabilities(d, fit_logistic(X, y), te)[te]
                del X, y
            P, title = Pcv, "REFIT leave-one-day-out (each day predicted by a fit on the other days)"
        X, y = fit_rows(d, cand)
        c = fit_logistic(X, y)
        del X, y
        print(f"\nfit on all {int(cand.sum())} rows ({', '.join(alerts.PROB_FEATURES)}):\n"
              "  RAIN_ALERT_COEF=" + ",".join(f"{x:.4f}" for x in c))

    # ---------- probability: reliability and Brier ----------
    rows = (P > 0) & el[:, None] & (Y >= 0)
    p, y, L = P[rows], Y[rows].astype(np.float64), lead[rows]
    F = np.nan_to_num(S[:, 1:], nan=0.0)
    q = np.minimum(F, np.concatenate([F[:, 1:], F[:, -1:]], axis=1))[rows]
    del F
    clim = np.zeros_like(y)
    for l0, l1 in BUCKETS:
        m = (L > l0) & (L <= l1)
        clim[m] = y[m].mean() if m.any() else 0
    rest = y[q < alerts.P_START].mean() if (q < alerts.P_START).any() else 0
    live = np.where(q >= alerts.P_START, [alerts.probability(x, 0.5) / 100 for x in L], rest)
    print(f"\nprobability, {title}: {len(y)} forecast steps near rain, rain in {y.mean():.0%}")
    print(f"  Brier {brier(p, y):.4f}; climatology by lead {brier(clim, y):.4f}; live rule as % {brier(live, y):.4f} -> "
          f"skill vs climatology {1 - brier(p, y) / brier(clim, y):+.3f}, vs live rule {1 - brier(p, y) / brier(live, y):+.3f}")
    bins = (0, .1, .2, .3, .4, .5, .6, .7, .8, .9, 1.0001)
    print("  reliability, mean P / observed      P: " + " ".join(f"{bins[i]:.1f}-{min(bins[i + 1], 1):.1f}" for i in range(len(bins) - 1)))
    for (l0, l1) in [(0, 120)] + BUCKETS:
        m = (L > l0) & (L <= l1)
        cells = []
        for i in range(len(bins) - 1):
            k = m & (p >= bins[i]) & (p < bins[i + 1])
            cells.append(f"{p[k].mean():.2f}/{y[k].mean():.2f}" if k.sum() >= 20 else "    -    ")
        print(f"  lead {l0:3d}-{l1:3d} min (n {int(m.sum()):7d}):   " + " ".join(cells))
    del p, y, L, q, clim, live

    # ---------- alerts: live rule vs thresholds ----------
    by_g = {}
    for j in np.argsort(d["issue"], kind="stable"):
        by_g.setdefault(int(d["g"][j]), []).append(j)
    ons = onsets(d)
    print(f"\nalerts by lead: FAR / POD / alerts / lead error (median, median |.|, min); {len(ons)} rain onsets")
    print("rule          " + "".join(f"{f'{l0}-{l1} min':>31s}" for l0, l1 in BUCKETS) + "    shown % / hit % (all)")

    def line(name, start, shown):
        r = score(d, el, start, shown)
        pd = pod(d, start, ons, by_g)
        cells = [f"{x['far']:.2f} / {c / n if n else np.nan:.2f} / {x['n']:5d} / {x['err'][0]:+3.0f} {x['err'][1]:2.0f}"
                 for x, (c, n) in zip(r, pd)]
        nn = sum(x["n"] for x in r)
        tail = f"{100 * sum(x['shown'] for x in r) / nn:5.0f} / {100 * sum(x['hits'] for x in r) / nn:3.0f}" if nn else ""
        print(f"{name:14s}" + "".join(f"{c:>31s}" for c in cells) + "    " + tail)

    st = rule_starts(d)
    Rf = np.nan_to_num(d["R"][:, 1:], nan=0.0)
    shown = np.full(len(st), np.nan)
    for j in np.flatnonzero(st >= 0):
        shown[j] = alerts.probability(lead[j, st[j]], float(Rf[j, st[j]:st[j] + 6].max())) / 100
    line("live rule", st, shown)
    for thr in (float(x) for x in a.thr.split(",")):
        k = prob_starts(P, lead, thr)
        line(f"P >= {thr:.2f}", k, np.where(k >= 0, P[np.arange(len(k)), np.maximum(k, 0)], np.nan))
    if src == "de":
        print("(live rule's shown %: HIT_RATE with the rate at the gauge point - the alerts use the circle's 75th "
              "percentile when higher, so this can be one class low)")


if __name__ == "__main__":
    main()
