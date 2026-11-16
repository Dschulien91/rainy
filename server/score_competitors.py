"""Scores of bench_competitors.py: official open nowcasts against OUR forecast at the same rain gauges, same run time,
same lead, same truth - per provider and country (network, or SYNOP country). "Are we on par with the competitors,
especially at +120 min?"

Join per competitor run R (a full hour; CHMI / KNMI / ECCC also :30) and lead L:
  competitor  the provider's value at R + L (box mean +-5 km, like ours; --point: the cell under the gauge; ECCC has
              only the point). MET Norway's "+120" is its +115 step; MEANDER's values are 30-min means centred on L
              (+120: the last 15 min); KNMI's are the 5-min sum ending at R + L x 12 (its stored +0 is not scored).
  ours        each instance's forecast (verify_log/fcgauge = "European" (OPERA), verify_log/eun/fcgauge = "own";
              North America: verify_log/na/fcgauge) of the newest run r with R - 15 min <= r <= R (normally r = R:
              both note runs at full 10 min) at the same nominal lead L. "age" = R - r is printed (--exact: r = R only).
  persist     that run's picture at r ("nothing moves", what a forecast must beat).
  truth       10-minute networks: the gauges' 10-minute sums from the archive (+ live sources unless --archive), as in
              check_forecast_gauges.py:
                10-min pairs  mean rate over V-10 .. V+10 (two sums), thresholds 0.1 / 0.3 / 1 mm/h
                hours         the six 10-min sums of V-30 .. V+30 = mm in the hour centred on V, against the rate at V
                              (all variants alike), thresholds 0.2 / 1 / 3 mm
              SYNOP ("sy", hourly amounts spread over their six 10-min steps) and Portugal: hours only (at +30 / +90 the
              centred hour IS a SYNOP hour; at +60 / +120 it is half of two).
              North America: the METAR / SWOB hour whose centre is within 15 min of V (ECCC is read for V = :30:
              METAR hours end ~:53, SWOB hours at :00).
Only cases where every column and the truth exist are scored (the same gauges, runs and leads for all rows).
The rows also say how late each provider's run was read ("read after", the competitor's latency upper bound).

  docker exec -w /app deploy-api-1 python score_competitors.py [--hours 48 | --day 20261116] [--until YYYYMMDDHHMM]
      [--archive] [--point] [--exact] [--provider metno] [--only se,Sweden] [--inst European|own] [--min 20]
  docker exec -w /app deploy-api-na-1 python score_competitors.py --na [--hours 48]
  (--dir <verify_log root> to read another copy; Europe: the root that holds fcgauge/, eun/, bench/, gaugetruth/)
  --incabe-scale F   INCA-BE's values were stored as value x row["scale"] (6: read as mm per 10 min); F replaces
                     that factor (1 = the values were mm/h after all). Check: its "x" against the gauges in rain.
"""
import calendar
import glob
import json
import os
import sys
import time

import numpy as np

args = sys.argv[1:]
opt = lambda name, default=None: args[args.index(name) + 1] if name in args else default
if opt("--dir"):
    os.environ["VERIFY_LOG_DIR"] = opt("--dir")
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import check_national_gauges as cng  # noqa: E402  (after VERIFY_LOG_DIR is set)

NA = "--na" in args
HOURS = float(opt("--hours", 48))
UNTIL = (calendar.timegm(time.strptime(opt("--until"), "%Y%m%d%H%M")) * 1000 if opt("--until") else time.time() * 1000)
SINCE = UNTIL - HOURS * 3600_000
if opt("--day"):
    SINCE = calendar.timegm(time.strptime(opt("--day"), "%Y%m%d")) * 1000
    UNTIL = SINCE + 86400_000
MATCH_MS = 15 * 60_000
MIN_N = int(opt("--min", 20))
KIND = "pt" if "--point" in args else "v"
HOURLY_NETS = {"sy", "pt", "us", "ca"}
PROVIDERS = {"metno": "MET Norway radar nowcast (Yr)", "incabe": "RMI INCA-BE", "chmi": "CHMI COTREC",
             "knmi": "KNMI pySTEPS nowcast", "meander": "HungaroMet MEANDER", "eccc": "ECCC radar extrapolation"}

if NA:
    LOG = cng.LOG_DIR if os.path.basename(cng.LOG_DIR.rstrip("/\\")) == "na" else os.path.join(cng.LOG_DIR, "na")
    with open(os.path.join(LOG, "metar", "stations.json")) as f:
        ST = [tuple(s) for s in json.load(f)]
    INST = {"ours": os.path.join(LOG, "fcgauge")}
    BENCH = os.path.join(LOG, "bench")
else:
    ST = cng.stations()
    INST = {"European": os.path.join(cng.ROOT, "fcgauge"), "own": os.path.join(cng.ROOT, "eun", "fcgauge")}
    if opt("--inst"):
        INST = {k: v for k, v in INST.items() if k == opt("--inst")}
    BENCH = os.path.join(cng.ROOT, "bench")
INDEX = {f"{s[0]}:{s[1]}": k for k, s in enumerate(ST)}


def group(s):
    return f"sy-{s[1].split('-', 1)[0]}" if s[0] == "sy" else s[0]


def group_name(g):
    if g.startswith("sy-"):
        import synop
        return f"{synop.NAMES.get(g[3:], g[3:])} (SYNOP)"
    return {"us": "USA (METAR gauges)", "ca": "Canada (SWOB gauges)"}.get(g) or cng.NAMES.get(g, g)


# ---------- inputs ----------
def competitor_rows():
    out = []
    only = opt("--provider")
    for fp in sorted(glob.glob(os.path.join(BENCH, "*.jsonl"))):
        if only and not os.path.basename(fp).startswith(only + "_"):
            continue
        for line in open(fp):
            try:
                r = json.loads(line)
            except ValueError:
                continue
            if r.get("var", "rr") == "rr" and SINCE <= r["run"] < UNTIL:
                if r["p"] == "incabe" and opt("--incabe-scale"):
                    f = float(opt("--incabe-scale")) / float(r.get("scale", 1.0))
                    for kind in ("v", "pt"):
                        r[kind] = [[None if x is None else x * f for x in lead] for lead in r[kind]]
                out.append(r)
    return out


refs = {}
for name, d in list(INST.items()):
    have = sorted(int(os.path.basename(p)[:-4]) for p in glob.glob(os.path.join(d, "*.npz"))
                  if os.path.basename(p)[:-4].isdigit())
    if have:
        refs[name] = np.array(have, np.int64)
    else:
        del INST[name]
_cache = {}


def ours(name, run):
    """(the run r of this instance to compare with, its fc array (1 + leads, gauges)) or (None, None)."""
    a = refs[name]
    i = np.searchsorted(a, run, side="right") - 1
    if i < 0 or a[i] < run - MATCH_MS or ("--exact" in args and a[i] != run):
        return None, None
    r = int(a[i])
    if (name, r) not in _cache:
        try:
            _cache[(name, r)] = np.load(os.path.join(INST[name], f"{r}.npz"))["fc"].astype(np.float32)
        except (OSError, ValueError, KeyError):
            _cache[(name, r)] = None
    return r, _cache[(name, r)]


def truth_europe(keys):
    """The archive, and (unless --archive) what the sources have now - only for the networks that occur in the rows
    (cng.truth would also page through the networks without a gauge here)."""
    since, until = SINCE - 3600_000, min(UNTIL + 3 * 3600_000, time.time() * 1000)
    out = cng.truth([], since, until, live=False)
    if "--archive" not in args:
        st = [ST[INDEX[k]] for k in keys if k in INDEX]
        for c in sorted({s[0] for s in st}):
            try:
                for sid, series in cng.GAUGES[c]([s[1] for s in st if s[0] == c], since, until).items():
                    out.setdefault(c, {}).setdefault(sid, {}).update(series)
            except Exception as e:
                print(f"  ({c} gauges, live: {e})")
    return out


def truth_na(keys):
    want, out = {k.split(":", 1)[1] for k in keys}, {}
    for fp in glob.glob(os.path.join(LOG, "metar", "*.jsonl")):
        for line in open(fp):
            try:
                r = json.loads(line)
            except ValueError:
                continue
            if r["id"] in want and SINCE - 3600_000 <= r["t"] <= UNTIL + 4 * 3600_000:
                out.setdefault(r["id"], {})[r["t"]] = r["mm"]
    return out


def hour_na(ser, valid):
    """mm of the gauge hour whose centre is closest to `valid` (within 15 min)."""
    best = None
    for t1, mm in ser.items():
        d = abs(t1 - 1800_000 - valid)
        if d <= MATCH_MS and (best is None or d < best[0]):
            best = (d, mm)
    return None if best is None else best[1]


# ---------- join ----------
def main():
    rows = competitor_rows()
    if not rows:
        sys.exit(f"no competitor rows in {BENCH} for this time")
    if not INST:
        sys.exit("no forecast runs of our instances (fcgauge) found")
    keys = {k for r in rows for k in r["ids"]}
    truth = truth_na(keys) if NA else truth_europe(keys)
    tag = {"European": "EU", "own": "own", "ours": ""}   # short: the table prints 12 characters per label
    labels = ["competitor"] + [x for n in INST for x in (f"ours {tag[n]}".strip(), f"persist {tag[n]}".strip())]
    data = {}     # (provider, group, lead, "pairs" | "hours") -> [values per label + [truth]]
    info = {}     # provider -> {"runs", "late", "age": {inst: [...]}}
    for r in rows:
        p, run = r["p"], r["run"]
        inf = info.setdefault(p, {"runs": set(), "late": [], "age": {n: [] for n in INST}, "fp": r.get("fp")})
        inf["runs"].add(run)
        inf["late"].append((r["got"] - run) / 60000)
        mine = {n: ours(n, run) for n in INST}
        if any(fc is None for _, fc in mine.values()):
            continue
        for n, (rr, _) in mine.items():
            inf["age"][n].append((run - rr) / 60000)
        for i, lead in enumerate(r["leads"]):
            if lead not in cng.FC_LEADS:
                continue
            li = cng.FC_LEADS.index(lead)
            valid = run + lead * 60_000
            comp = r[KIND][i]
            for j, key in enumerate(r["ids"]):
                k = INDEX.get(key)
                if k is None or comp[j] is None:
                    continue
                vals = [comp[j]]
                for n, (_, fc) in mine.items():
                    if k >= fc.shape[1]:
                        break
                    vals += [fc[li + 1, k], fc[0, k]]
                if len(vals) != len(labels) or not np.isfinite(vals).all():
                    continue
                net, sid = key.split(":", 1)
                g = group(ST[k])
                if NA:
                    h = hour_na(truth.get(sid, {}), valid)
                    if h is not None:
                        data.setdefault((p, g, lead, "hours"), []).append(vals + [h])
                    continue
                q = truth.get(net, {}).get(sid, {})
                if net not in HOURLY_NETS:
                    two = [q.get(valid), q.get(valid + 600_000)]
                    if None not in two:
                        data.setdefault((p, g, lead, "pairs"), []).append(vals + [np.mean(two) * 6])
                six = [q.get(valid + m * 600_000) for m in (-2, -1, 0, 1, 2, 3)]
                if None not in six:
                    data.setdefault((p, g, lead, "hours"), []).append(vals + [float(np.sum(six))])

    summary = []
    for p in sorted(info, key=lambda x: list(PROVIDERS).index(x) if x in PROVIDERS else 99):
        inf = info[p]
        ages = ", ".join(f"{n} {np.mean(a):.1f} min" for n, a in inf["age"].items() if a)
        print(f"\n===== {PROVIDERS.get(p, p)} ({p}): {len(inf['runs'])} runs "
              f"{time.strftime('%d.%m. %H:%M', time.gmtime(min(inf['runs']) / 1000))} - "
              f"{time.strftime('%d.%m. %H:%M', time.gmtime(max(inf['runs']) / 1000))} UTC; read {np.median(inf['late']):.0f} min "
              f"after the run time (median); our run older by {ages or '-'}; values: "
              f"{'cell under the gauge' if KIND == 'pt' or inf['fp'] == 0 else 'box +-%s km' % inf['fp']}")
        groups = sorted({g for (pp, g, _, _) in data if pp == p})
        groups = sorted(groups, key=lambda g: -sum(len(v) for (pp, gg, _, _), v in data.items() if pp == p and gg == g))
        for g in groups + ["all"]:
            sel = (lambda gg: True) if g == "all" else (lambda gg, g=g: gg == g)
            name = "All gauges of this provider" if g == "all" else group_name(g)
            only = opt("--only")
            if only and g != "all" and not any(o == g or o.lower() in name.lower() for o in only.split(",")):
                continue
            for lead in cng.FC_LEADS:
                for kind, ths, unit in (("pairs", (0.1, 0.3, 1.0), "mm/h"), ("hours", (0.2, 1.0, 3.0), "mm")):
                    a = [x for (pp, gg, ll, kk), v in data.items() if pp == p and sel(gg) and ll == lead and kk == kind
                         for x in v]
                    if len(a) < MIN_N:
                        continue
                    a = np.array(a, np.float32)
                    label = "10-min pairs" if kind == "pairs" else "hour centred on the valid time"
                    cng.table(f"{name}, +{lead} min, {label}", a[:, -1], {lb: a[:, i] for i, lb in enumerate(labels)},
                              ths=ths, unit=unit)
                    if kind == "hours":
                        summary.append((p, name, lead, a))

    if summary:
        print("\n===== summary: hours centred on the valid time, CSI at 0.2 / 1 mm and amount ratio "
              f"(columns: {', '.join(labels)})")
        for p, name, lead, a in summary:
            g = a[:, -1]
            cells = []
            for i in range(len(labels)):
                q = a[:, i]
                csi = []
                for th in (0.2, 1.0):
                    h, m, f = ((g >= th) & (q >= th)).sum(), ((g >= th) & (q < th)).sum(), ((g < th) & (q >= th)).sum()
                    csi.append(f"{h / (h + m + f):.2f}" if h + m + f else "  - ")
                ratio = f"x{q.sum() / g.sum():.2f}" if g.sum() > 0.5 else "x - "
                cells.append(f"{'/'.join(csi)} {ratio}")
            print(f"  {p:7s} {name[:34]:34s} +{lead:3d} n={len(g):5d}  " + "  |  ".join(cells))


if __name__ == "__main__":
    main()
