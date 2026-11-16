"""How often would a rain alert have been right in Europe and North America? verify_alerts.py for the instances
outside Germany, against all their rain gauges.

Every forecast run (10-minute step) of a Europe / North America instance notes the alert rule's input at each of
its gauges (rainy_server.EuropeOpera.record_alert_inputs, since 16.11.2026): <LOG_DIR>/fcalert/<ref ms>.npz with
  share  uint8 (1 + 24, gauges): row 0 = the picture at ref, rows 1..24 = the forecast for ref + 5 .. ref + 120 min;
         the wet share (>= alerts.WET) of the alert circle around the gauge (same circle as AlertEngine.series,
         valid pixels only) x 250; 255 = no data (circle mostly outside the radar)
  peak   float16, same shape: the rate AlertEngine.series reports per step (mm/h), NaN = no data
The columns are the instance's gauge list: Europe (both instances) = check_national_gauges.stations()
(<ROOT>/eun/natobs/stations.json, appended only - older files have fewer columns), North America =
<LOG_DIR>/metar/stations.json (northamerica.MetarGauges). Truth: Europe = the gauge archive <ROOT>/gaugetruth
(10-minute sums per UTC day; "--live" adds what the sources have right now), North America = the hourly
<LOG_DIR>/metar/<YYYYMMDD>.jsonl (METAR hours end at the report, ~:53; Canada's SWOB on the full hour).

Rules (per run and gauge, as AlertEngine.evaluate):
  issue        the gauge's circle is dry in the picture (share < P_START * 0.6, else "raining now", no alert) and,
               for 10-minute networks, the gauge itself measured < 0.05 mm in the 10-minute sums ending ref - 20,
               ref - 10 and ref (missing values count as dry, as in verify_alerts.py)
  start        the first forecast step from which share >= P_START for MIN_STEPS steps in a row; lead = start - ref
               (the push goes out a few minutes after ref, so the real lead is a little shorter)
  intensity    the highest "peak" from the start until the share falls below P_START * 0.6 (alerts' event end),
               classed by PEAKS (light < 1, moderate 1-4, heavy >= 4 mm/h)
  hit, 10-min  the gauge measures >= 0.05 mm in a 10-minute interval overlapping start +- 15 min
               (verify_alerts.window); not counted when that is unknown (no wet value and a gap in the window)
  hit, hourly  COARSER (SYNOP "sy", Portugal "pt", METAR / SWOB): >= 0.1 mm in the hour containing the start, or in
               the next hour as well when the start is within 15 min of the hour's end; separate tables
  dedupe       per gauge and lead bucket, runs in time order: an alert whose start is within 30 min of the last
               COUNTED start of that gauge and bucket is the same event and not counted again (roughly the engine's
               "same_event"); an event is counted with the first run whose truth is known
  onsets       10-min networks: an interval with >= 0.05 mm after >= 60 min known dry (six intervals < 0.05 mm).
               Onset time o = the interval's middle; caught when any run in [o - 90, o - 5] min simulated an alert
               (raw, before dedupe and the gauge-dry filter) whose window contains the interval. Hourly networks:
               an hour with >= 0.1 mm after a known dry hour; caught by a run in [hour start - 90, hour end - 5]
               predicting a start in [hour start - 15, hour end + 15]. Only onsets whose whole run range lies in
               the window and where at least one of those runs had data at the gauge are counted.
  reliability  10-min networks, gauge dry at issue: for every REL_LEADS lead, min(share) of the step at ref + lead
               and the next (+120: the step before) against rain at the gauge in window(step time) - the measured
               probability per forecast wet share (as verify_alerts.py)
Not simulated: cooldown, MAX_PER_DAY, quiet hours, early / upgrade alerts, the user's chosen lead (all buckets count).
Groups: networks of check_national_gauges.NAMES, SYNOP per country (synop.NAMES), USA / Canada; groups with fewer
than --min-n counted alerts are listed on one line.

  docker exec -w /app deploy-api-eun-1 python verify_alerts_eu.py --hours 48      (own picture, Europe)
  docker exec -w /app deploy-api-eu-1 python verify_alerts_eu.py --hours 48       (OPERA, same gauges)
  docker exec -w /app deploy-api-na-1 python verify_alerts_eu.py --hours 48 [--na] (auto when LOG_DIR ends in /na)
  options: --day YYYYMMDD (runs issued that UTC day instead of --hours), --min-n 20, --live (Europe: also the
           sources' current values on top of the archive), --rel-groups (reliability per group, not only overall)
"""
import argparse
import glob
import json
import os
import sys
import time
from datetime import datetime, timezone

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import check_national_gauges as cng  # noqa: E402
from alerts import MIN_STEPS, P_START  # noqa: E402

# the same buckets and window as verify_alerts.py (copied: importing it pulls score.py / recorder.py)
LEADS = [(5, 15), (15, 30), (30, 45), (45, 60), (60, 90), (90, 120)]
PEAKS = [(0.3, 1.0, "leicht"), (1.0, 4.0, "mäßig"), (4.0, 999, "stark")]
TEN = 600_000
HOUR = 3600_000
REL_LEADS = (15, 30, 45, 60, 90, 120)
REL_BINS = (0.0, 0.1, 0.3, 0.5, 0.7, 0.9, 1.0001)
STEP = 300_000                 # forecast step (rainy_server.STEP_MS)
WET10, WET60 = 0.05, 0.1       # mm per 10 min / per hour counted as rain at the gauge
SAME = 30 * 60_000             # dedupe: same event
HOURLY_NETS = {"sy", "pt", "ct"}     # Europe: hourly amounts only (spread over six 10-min bins in the archive)


def window(t0):
    """End times of the gauge's 10-minute intervals (t - 10 min, t] that overlap t0 +- 15 min (verify_alerts.py)."""
    t0 = int(t0)
    return range((t0 - 900_000 + TEN - 1) // TEN * TEN, t0 + 1500_000, TEN)


def in_window(t, s):
    """t (10-min interval end) in window(s) - vectorizable."""
    return (s - 900_000 <= t) & (t < s + 1500_000)


def hour_hit(ends, mm, s):
    """Hourly gauge: True / False / None (unknown) for a predicted start s (see the docstring)."""
    j = int(np.searchsorted(ends, s, "left"))
    if j >= len(ends) or ends[j] - s >= HOUR:
        return None
    if mm[j] >= WET60:
        return True
    if ends[j] - s < 15 * 60_000:
        if j + 1 >= len(ends) or ends[j + 1] - ends[j] > 75 * 60_000:
            return None
        return bool(mm[j + 1] >= WET60)
    return False


def peak_class(p):
    p = max(p, PEAKS[0][0])
    return next(k for k, (v0, v1, _) in enumerate(PEAKS) if v0 <= p < v1)


def lead_bucket(lead):
    return next((k for k, (l0, l1) in enumerate(LEADS) if l0 <= lead < l1), None)


def fmt(ms):
    return time.strftime("%d.%m. %H:%M", time.gmtime(ms / 1000))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--hours", type=float, default=24.0)
    ap.add_argument("--day", help="only runs issued on this UTC day, e.g. 20261116")
    ap.add_argument("--min-n", type=int, default=20)
    ap.add_argument("--na", action="store_true", help="North America (auto when LOG_DIR ends in /na)")
    ap.add_argument("--live", action="store_true", help="Europe: also the sources' current values (slow)")
    ap.add_argument("--rel-groups", action="store_true", help="reliability table per group, not only overall")
    a = ap.parse_args()

    log_dir = cng.LOG_DIR
    base = os.path.basename(log_dir.rstrip("/\\"))
    na = a.na or base == "na"
    instance = "na (North America, MRMS)" if na else "eun (own picture, Europe)" if base == "eun" else "eu (OPERA, Europe)"

    # ---------- window and runs ----------
    if a.day:
        since = datetime.strptime(a.day, "%Y%m%d").replace(tzinfo=timezone.utc).timestamp() * 1000
        until = since + 86400_000 - 1
    else:
        until = time.time() * 1000
        since = until - a.hours * HOUR
    fca = os.path.join(log_dir, "fcalert")
    refs = sorted(int(os.path.basename(p)[:-4]) for p in glob.glob(os.path.join(fca, "*.npz"))
                  if os.path.basename(p)[:-4].isdigit())
    refs = [r for r in refs if since <= r <= until]
    if not refs:
        sys.exit(f"no fcalert run in {fca} in this time")

    # ---------- gauges and groups ----------
    if na:
        try:
            with open(os.path.join(log_dir, "metar", "stations.json")) as f:
                st = [tuple(s) for s in json.load(f)]
        except (OSError, ValueError):
            st = []
        gkey = [s[0] for s in st]
        hourly = np.ones(len(st), bool)
    else:
        st = cng.stations()
        if not st:  # the OPERA instance without the shared list keeps its own (rainy_server.EuropeOpera.gauges)
            try:
                with open(os.path.join(cng.OBS_DIR, "stations.json")) as f:
                    st = [tuple(s) for s in json.load(f)]
            except (OSError, ValueError):
                st = []
        # as check_forecast_gauges.group: SYNOP per country
        gkey = [f"sy-{s[1].split('-', 1)[0]}" if s[0] == "sy" else s[0] for s in st]
        hourly = np.array([s[0] in HOURLY_NETS for s in st], bool)
    if not st:
        sys.exit("no gauge list")
    ns = len(st)

    # ---------- truth ----------
    t0 = (refs[0] - 3 * HOUR) // HOUR * HOUR
    t1 = (refs[-1] + 4 * HOUR) // HOUR * HOUR
    nb = (t1 - t0) // TEN + 1                       # bin b = the 10 minutes ending t0 + b * TEN

    def bix(t):
        return (np.asarray(t, np.int64) - t0) // TEN

    tm = None                                       # 10-minute sums (gauges, bins); Europe only
    hours = {}                                      # gauge -> (hour ends, mm) for hourly gauges
    if na:
        idx = {s[1]: k for k, s in enumerate(st)}
        ser = {}
        for day in range(int(t0 // 86400_000) - 1, int(t1 // 86400_000) + 1):
            fp = os.path.join(log_dir, "metar", time.strftime("%Y%m%d", time.gmtime(day * 86400)) + ".jsonl")
            if not os.path.exists(fp):
                continue
            with open(fp) as f:
                for line in f:
                    try:
                        r = json.loads(line)
                    except ValueError:
                        continue
                    k = idx.get(r.get("id"))
                    if k is not None and t0 <= r["t"] <= t1 and r.get("mm") is not None:
                        ser.setdefault(k, {})[int(r["t"])] = float(r["mm"])
        for k, d in ser.items():
            e = np.array(sorted(d), np.int64)
            hours[k] = (e, np.array([d[t] for t in e], np.float32))
        del ser
    else:
        tm = np.full((ns, nb), np.nan, np.float32)
        idx = {f"{s[0]}:{s[1]}": k for k, s in enumerate(st)}
        for day in range(int(t0 // 86400_000), int(t1 // 86400_000) + 1):
            for key, row in cng._day_load(day).items():   # one day in memory at a time
                k = idx.get(key)
                if k is None:
                    continue
                tt = day * 86400_000 + np.arange(144, dtype=np.int64) * TEN
                ok = np.isfinite(row) & (tt >= t0) & (tt <= t1)
                tm[k, bix(tt[ok])] = row[ok]
        if a.live:
            for c, fn in cng.GAUGES.items():
                ks = {s[1]: k for k, s in enumerate(st) if s[0] == c}
                try:
                    got = fn(list(ks), t0, t1)
                except Exception as e:
                    print(f"  ({c} gauges, live: {e})")
                    continue
                for sid, series in got.items():
                    k = ks.get(sid)
                    for t, v in (series.items() if k is not None else ()):
                        if t % TEN == 0 and t0 <= t <= t1 and v is not None and np.isfinite(v):
                            tm[k, bix(t)] = v
        # hourly networks: the hour ending e = the six bins ending e - 50 .. e (t0 is a full hour)
        nh = (nb - 1) // 6
        ends = t0 + (np.arange(nh, dtype=np.int64) + 1) * HOUR
        for k in np.flatnonzero(hourly):
            h = tm[k, 1:1 + 6 * nh].reshape(nh, 6).sum(axis=1)
            ok = np.isfinite(h)
            if ok.any():
                hours[k] = (ends[ok], h[ok])

    # ---------- runs ----------
    ten = ~hourly
    acc = {}                     # group key -> {(lead bucket, peak class): [hits, n]}
    last = {}                    # (gauge, lead bucket) -> last counted start (dedupe)
    raw = {}                     # gauge -> [(ref, start)] every simulated alert (onsets)
    nrel = np.zeros((ns, len(REL_LEADS), len(REL_BINS) - 1), np.int32)   # per gauge, summed into groups later
    hrel = np.zeros_like(nrel)
    has = np.zeros((len(refs), ns), bool)   # run had data at the gauge
    n_raw = n_cnt = n_dup = n_unk = n_dry = 0
    used = []
    for ri, ref in enumerate(refs):
        try:
            z = np.load(os.path.join(fca, f"{ref}.npz"))
            share, peak = z["share"], z["peak"].astype(np.float32)
        except Exception:
            continue                         # the newest file may still be being written
        used.append(ref)
        n = min(share.shape[1], ns)
        steps = share.shape[0] - 1
        sh = share[:, :n].astype(np.float32) / 250
        sh[share[:, :n] == 255] = np.nan
        pk = peak[:, :n]
        has[ri, :n] = share[0, :n] != 255
        ts = ref + np.arange(steps + 1, dtype=np.int64) * STEP
        # gauge dry at issue (10-min networks; missing = dry)
        gdry = np.ones(n, bool)
        if tm is not None:
            b = bix([ref - 2 * TEN, ref - TEN, ref])
            gdry = ~(tm[:n, b] >= WET10).any(axis=1)
        # the rule
        okf = sh[1:] >= P_START
        cond = np.ones((steps - MIN_STEPS + 1, n), bool)
        for k in range(MIN_STEPS):
            cond &= okf[k:k + steps - MIN_STEPS + 1]
        alert = cond.any(axis=0) & (sh[0] < P_START * 0.6)
        first = cond.argmax(axis=0)
        for g in np.flatnonzero(alert):
            i = int(first[g])                          # forecast step index (row i + 1)
            s = int(ts[i + 1])
            n_raw += 1
            raw.setdefault(int(g), []).append((ref, s))
            if ten[g] and not gdry[g]:
                n_dry += 1
                continue
            end = next((j for j in range(i, steps) if sh[j + 1, g] < P_START * 0.6), steps)
            ev = pk[1 + i:1 + end, g]
            p = peak_class(float(np.nanmax(ev)) if np.isfinite(ev).any() else 0.0)
            bk = lead_bucket((s - ref) / 60_000)
            if bk is None:
                continue
            prev = last.get((g, bk))
            if prev is not None and abs(s - prev) <= SAME:
                n_dup += 1
                continue
            if ten[g]:
                v = tm[g, bix(list(window(s)))]
                hit = True if (v >= WET10).any() else (False if np.isfinite(v).all() else None)
            else:
                hit = hour_hit(*hours[g], s) if g in hours else None
            if hit is None:
                n_unk += 1
                continue
            last[(g, bk)] = s
            e = acc.setdefault(gkey[g], {}).setdefault((bk, p), [0, 0])
            e[0] += hit
            e[1] += 1
            n_cnt += 1
        # reliability (10-min networks, gauge dry at issue)
        if tm is not None:
            sel = np.flatnonzero(ten[:n] & gdry)
            for li, lead in enumerate(REL_LEADS):
                i = lead * 60_000 // STEP                 # row of ref + lead
                if i > steps:
                    continue
                j0, j1 = (i, i + 1) if i + 1 <= steps else (i - 1, i)
                v = np.minimum(sh[j0, sel], sh[j1, sel])
                w = tm[np.ix_(sel, bix(list(window(ts[i]))))]
                known = (w >= WET10).any(axis=1) | np.isfinite(w).all(axis=1)
                use = np.isfinite(v) & known
                kb = np.searchsorted(REL_BINS, v[use], "right") - 1
                kb = np.clip(kb, 0, len(REL_BINS) - 2)
                gs = sel[use]
                np.add.at(nrel, (gs, li, kb), 1)
                np.add.at(hrel, (gs, li, kb), (w[use] >= WET10).any(axis=1).astype(np.int32))
    if not used:
        sys.exit("no readable fcalert run")
    run_t = np.array(refs, np.int64)
    first_ref, last_ref = used[0], used[-1]

    # ---------- rain onsets without alert ----------
    miss = {}                    # group key -> [missed, onsets]

    def caught(g, lo, hi, pred):
        if not has[(run_t >= lo) & (run_t <= hi), g].any():
            return None                                  # no run with data at the gauge
        for r, s in raw.get(g, ()):
            if lo <= r <= hi and pred(s):
                return True
        return False

    if tm is not None:
        sub = np.flatnonzero(ten)
        x = tm[sub]
        dry = (x < WET10).astype(np.int32)               # NaN -> not dry
        cs = np.concatenate([np.zeros((len(sub), 1), np.int32), np.cumsum(dry, axis=1)], axis=1)
        dry6 = np.zeros_like(dry, bool)
        dry6[:, 6:] = (cs[:, 6:-1] - cs[:, :-7]) == 6     # bins b-6 .. b-1 all known dry
        for row, b in np.argwhere((x >= WET10) & dry6):
            g, t = int(sub[row]), t0 + int(b) * TEN
            o = t - TEN // 2
            lo, hi = o - 90 * 60_000, o - 5 * 60_000
            if lo < first_ref or hi > last_ref:
                continue
            c = caught(g, lo, hi, lambda s: in_window(t, s))
            if c is not None:
                e = miss.setdefault(gkey[g], [0, 0])
                e[0] += not c
                e[1] += 1
    for g, (ends, mm) in hours.items():
        for j in range(1, len(ends)):
            e1 = int(ends[j])
            if mm[j] < WET60 or mm[j - 1] >= WET60 or not 45 * 60_000 <= e1 - ends[j - 1] <= 75 * 60_000:
                continue
            h0 = e1 - HOUR
            lo, hi = h0 - 90 * 60_000, e1 - 5 * 60_000
            if lo < first_ref or hi > last_ref:
                continue
            c = caught(g, lo, hi, lambda s: h0 - 15 * 60_000 <= s <= e1 + 15 * 60_000)
            if c is not None:
                e = miss.setdefault(gkey[g], [0, 0])
                e[0] += not c
                e[1] += 1

    # ---------- output ----------
    print(f"instance {instance}, LOG_DIR {log_dir}")
    print(f"window {fmt(since)} - {fmt(until)} UTC; {len(used)} runs read ({fmt(first_ref)} - {fmt(last_ref)}), "
          f"{len(refs) - len(used)} unreadable")
    print(f"gauges {ns} ({int(ten.sum())} 10-minute, {int(hourly.sum())} hourly; truth for "
          f"{int(np.isfinite(tm[ten]).any(axis=1).sum()) if tm is not None else 0} 10-min / {len(hours)} hourly)")
    print(f"alerts simulated {n_raw} (all runs) -> {n_cnt} counted; not counted: {n_dry} gauge already wet, "
          f"{n_dup} same event, {n_unk} without truth (yet)")

    def table(title, ac, ms):
        print(f"\n{title}; rain onsets without alert: {ms[0]}/{ms[1]}" + (f" ({ms[0] / ms[1]:.0%})" if ms[1] else ""))
        print("lead      " + "  ".join(f"{name:>12s}" for _, _, name in PEAKS) + "      all")
        for b, (l0, l1) in enumerate(LEADS):
            cells, h, n = [], 0, 0
            for p in range(len(PEAKS)):
                e = ac.get((b, p), [0, 0])
                h, n = h + e[0], n + e[1]
                cells.append(f"{e[0] / e[1]:6.0%} ({e[1]:4d})" if e[1] else f"{'-':>12s}")
            print(f"{l0:3d}-{l1:3d}  " + "  ".join(cells) + (f"   {h / n:5.0%} ({n})" if n else ""))
        h = sum(v[0] for v in ac.values())
        n = sum(v[1] for v in ac.values())
        if n:
            print(f"all       {h / n:5.0%} ({n})")

    def merged(keys):
        ac, ms = {}, [0, 0]
        for k in keys:
            for c, v in acc.get(k, {}).items():
                e = ac.setdefault(c, [0, 0])
                e[0] += v[0]
                e[1] += v[1]
            m = miss.get(k, [0, 0])
            ms[0] += m[0]
            ms[1] += m[1]
        return ac, ms

    def section(title, groups):
        print(f"\n===== {title} =====")
        small = []
        for name, keys in groups:
            ac, ms = merged(keys)
            n = sum(v[1] for v in ac.values())
            if n < a.min_n:
                if n or ms[1]:
                    small.append(f"{name} {n}" + (f" (onsets {ms[1] - ms[0]}/{ms[1]} caught)" if ms[1] else ""))
                continue
            table(f"{name}, hit rate (n) by lead time", ac, ms)
        if small:
            print(f"\nfewer than {a.min_n} alerts: " + "; ".join(small))

    def reliability(name, mask):
        n, h = nrel[mask].sum(axis=0), hrel[mask].sum(axis=0)
        if not n.sum():
            return
        print(f"\nreliability, {name}: how often it rained at the gauge (+-15 min) by forecast wet share of the circle  % (n)")
        print("share     " + "  ".join(f"{f'+{lead}':>12s}" for lead in REL_LEADS))
        for k in range(len(REL_BINS) - 1):
            cells = [f"{h[li, k] / n[li, k]:5.0%} ({n[li, k]:5d})" if n[li, k] else f"{'-':>12s}"
                     for li in range(len(REL_LEADS))]
            print(f"{REL_BINS[k]:.1f}-{min(REL_BINS[k + 1], 1.0):.1f}   " + "  ".join(cells))

    keys = sorted(set(gkey))
    if na:
        names = {"us": "USA (METAR gauges)", "ca": "Canada (ECCC SWOB gauges)"}
        section("HOURLY gauges - COARSER: hit = >= 0.1 mm in the hour containing the start (+ next hour if the start is "
                "within 15 min of its end)",
                [(names.get(k, k), (k,)) for k in keys] + [("All North America", tuple(keys))])
    else:
        import synop
        dense = [k for k in keys if k not in HOURLY_NETS and not k.startswith("sy-")]
        hr = [k for k in keys if k in HOURLY_NETS or k.startswith("sy-")]
        groups = [(cng.NAMES.get(k, k), (k,)) for k in dense]
        if sum(k in dense for k in ("it", "iv", "ie")) > 1:
            groups.append(("Italy (all regions)", tuple(k for k in ("it", "iv", "ie", "bz", "tn") if k in dense)))
        groups.append(("All 10-minute networks", tuple(dense)))
        section("10-minute gauges: hit = >= 0.05 mm in a 10-min interval overlapping the predicted start +- 15 min", groups)
        reliability("all 10-minute networks", ten)
        if a.rel_groups:
            for k in dense:
                reliability(cng.NAMES.get(k, k), np.array([x == k for x in gkey]))
        hgroups = [(cng.NAMES[k] if k in ("pt", "ct") else f"{synop.NAMES.get(k[3:], k[3:])} (SYNOP)", (k,)) for k in hr]
        syn = tuple(k for k in hr if k.startswith("sy-"))
        if syn:
            hgroups.append(("All SYNOP stations", syn))
        hgroups.append(("All hourly gauges", tuple(hr)))
        section("HOURLY gauges - COARSER: hit = >= 0.1 mm in the hour containing the start (+ next hour if the start is "
                "within 15 min of its end)", hgroups)


if __name__ == "__main__":
    main()
