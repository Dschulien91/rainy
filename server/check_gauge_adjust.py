"""Offline test of the gauge adjustment (gauge_adjust.py) on the recorded pictures at the gauges.

Pictures: verify_log/eun/natobs ("opera" = the OPERA instance's picture, "nat" = the own picture, both unadjusted),
truth: the gauge archive. Factors per network (or a wide kernel, "k200") from a rolling window that ends LAG hours
before each hour; 2-fold spatial split: learned on half of every network's gauges, scored on the other half. Scores
per network: hourly sums (amount, CSI at 0.2 / 1 / 3 mm) and 10-minute pairs ("t", CSI at 0.3 / 1 mm/h);
--classes adds the amount ratio per intensity class (conditioned on the gauge hour and on the picture hour).

  docker run --rm -m 1g -e VERIFY_LOG_DIR=/app/verify_log -v deploy_verify-log:/app/verify_log:ro       -v /opt/rainy/server:/work:ro -w /app deploy-api-eu python /work/check_gauge_adjust.py       --since 2026111700 --until 2026111821 --inst opera|nat --groups at,ch,bw,de,si       --configs country:12:1.5:3:0.3:0.5:3[,k200:12:1.5:3:0.3:0.33:3,...] [--classes] [--excl de]
  config = variant:window h:lag h:prior mm:wet mm/h:min factor:max factor (variant "country", "kNNN" = kernel km;
  a trailing "s" splits the factor into picture < 1 mm/h and >= 1 mm/h - tested 18.11.: drier, worse CSI).
Served settings (gauge_adjust.py): country:12:1.5:3:0.3:0.5:3.
"""
import sys, os, glob, json, time, calendar
sys.path.insert(0, "/app")
import numpy as np
import check_national_gauges as cng
A = sys.argv[1:]
opt = lambda k, d: A[A.index(k) + 1] if k in A else d
since = calendar.timegm(time.strptime(opt("--since", "2026111600"), "%Y%m%d%H")) * 1000
until = calendar.timegm(time.strptime(opt("--until", "2026111900"), "%Y%m%d%H")) * 1000
WIN_H, LAG_H, PRIOR = float(opt("--win", 12)), float(opt("--lag", 1.5)), float(opt("--prior", 3))
MIN_F, MAX_F = float(opt("--minf", 0.33)), float(opt("--maxf", 3.0))
WET = float(opt("--wet", 0))           # only 10-min steps where gauge or picture >= WET mm/h count in the sums
VARIANTS = opt("--var", "country,k100,k200").split(",")
KEY = opt("--inst", "nat")
EXCL = set(opt("--excl", "").split(",")) - {""}
ST = cng.stations(); n = len(ST)
def grp(s): return "sy-" + s[1].split("-", 1)[0] if s[0] == "sy" else s[0]
G_ = np.array([grp(s) for s in ST])
lat = np.radians([s[2] for s in ST]); lon = np.radians([s[3] for s in ST])
# picture at gauges every 10 min
t0 = since - int((WIN_H + LAG_H + 1) * 3600_000)
T = np.arange(t0 // 600_000 * 600_000, until + 7200_000, 600_000, dtype=np.int64)
ti = {int(t): i for i, t in enumerate(T)}
P = np.full((len(T), n), np.nan, np.float32)
for f in glob.glob(os.path.join(cng.ROOT, "eun", "natobs", "*.npz")):
    b = os.path.basename(f)[:-4]
    if not b.isdigit() or int(b) not in ti: continue
    z = np.load(f); v = z[KEY].astype(np.float32); m = min(n, len(v))
    P[ti[int(b)], :m] = np.where(v[:m] >= 0, v[:m], np.nan)
tr = cng.truth([], int(T[0]), int(T[-1]), live=False)
GM = np.full((len(T), n), np.nan, np.float32)
for k, s in enumerate(ST):
    q = tr.get(s[0], {}).get(s[1])
    if not q: continue
    for t, v in q.items():
        i = ti.get(int(t))
        if i is not None: GM[i, k] = v
print(f"{len(T)} steps, pictures {np.isfinite(P).any(1).sum()}, gauges with data {np.isfinite(GM).any(0).sum()}")
ok = np.isfinite(P) & np.isfinite(GM)
fold = np.zeros(n, int)
for g in set(G_):
    idx = np.flatnonzero(G_ == g); idx = idx[np.argsort(lat[idx] * 1000 + lon[idx])]; fold[idx[1::2]] = 1
train_ok = np.array([s[0] not in EXCL and grp(s) not in EXCL for s in ST])
def dist_km(a, b):
    return 6371 * np.arccos(np.clip(np.sin(lat[a])[:, None] * np.sin(lat[b])[None] +
                                    np.cos(lat[a])[:, None] * np.cos(lat[b])[None] * np.cos(lon[a][:, None] - lon[b][None]), -1, 1))
D = {f: dist_km(np.flatnonzero(fold == f), np.flatnonzero((fold != f) & train_ok)) for f in (0, 1)}
hours = np.arange(since, until, 3600_000, dtype=np.int64)
SPLIT_LO, SPLIT_HI = 0.5, 2.0   # picture rates (mm/h) between which the light-echo factor hands over to the rain one
def _field(v, gs, rs_, PRIOR, MIN_F, MAX_F, f):
    te, trn = np.flatnonzero(fold == f), np.flatnonzero((fold != f) & train_ok)
    if v.startswith("country"):
        out = np.ones(len(te))
        for g in set(G_[te]):
            a = trn[G_[trn] == g]; b = G_[te] == g
            out[b] = (gs[a].sum() + PRIOR) / (rs_[a].sum() + PRIOR)
    else:
        w = np.exp(-0.5 * (D[f] / float(v[1:].rstrip("s"))) ** 2)
        out = (w @ gs[trn] + PRIOR) / (w @ rs_[trn] + PRIOR)
    return te, np.clip(out, MIN_F, MAX_F)
def factors_for(v, WIN_H, LAG_H, PRIOR, WET, MIN_F, MAX_F):
    """(F, None) or, for a split variant (name ending in "s"), (F_light, F_rain): factors learned on the pairs whose
    picture is below / at least 1 mm/h."""
    split = v.endswith("s")
    use = ok if WET <= 0 else ok & ((GM * 6 >= WET) | (P >= WET))
    parts = [use & (P < 1.0), use & (P >= 1.0)] if split else [use]
    LAG, WIN = int(LAG_H * 6), int(WIN_H * 6)
    out = []
    for u in parts:
        cg, cr = np.cumsum(np.where(u, GM, 0), 0), np.cumsum(np.where(u, P / 6, 0), 0)
        F = np.ones((len(hours), n), np.float32)
        for h, H in enumerate(hours):
            i_end = ti[int(H)] - LAG; i0 = max(0, i_end - WIN)
            gs, rs_ = cg[i_end] - cg[i0], cr[i_end] - cr[i0]
            for f in (0, 1):
                te, val = _field(v, gs, rs_, PRIOR, MIN_F, MAX_F, f)
                F[h, te] = val
        out.append(F)
    return out if split else (out[0], None)
def apply(F, p, h, k):
    lo, hi = F
    if hi is None:
        return p * lo[h, k]
    w = np.clip(np.log(np.maximum(p, 1e-6) / SPLIT_LO) / np.log(SPLIT_HI / SPLIT_LO), 0, 1)
    return p * np.exp((1 - w) * np.log(lo[h, k]) + w * np.log(hi[h, k]))
def csi(p, g, th):
    a = ((p >= th) & (g >= th)).sum(); b = ((p < th) & (g >= th)).sum(); c = ((p >= th) & (g < th)).sum()
    return a / (a + b + c) if a + b + c else np.nan
want = opt("--groups", None).split(",")
# pairs once: per group the hour index, gauge index, gauge sum and picture sum; 10-min pairs likewise
HP, TP = {}, {}
for h, H in enumerate(hours):
    i = ti[int(H)]; sl = slice(i + 1, i + 7)
    okh = ok[sl].all(0); gh = GM[sl].sum(0); ph = (P[sl] / 6).sum(0)
    for j in range(i + 1, i + 7):
        mm = np.flatnonzero(ok[j] & ok[j + 1])
        TP.setdefault("h", []).append(np.full(len(mm), h)); TP.setdefault("k", []).append(mm); TP.setdefault("j", []).append(np.full(len(mm), j))
        TP.setdefault("g", []).append((GM[j, mm] + GM[j + 1, mm]) * 3); TP.setdefault("p", []).append(P[j, mm])
    m = np.flatnonzero(okh)
    HP.setdefault("h", []).append(np.full(len(m), h)); HP.setdefault("k", []).append(m)
    HP.setdefault("g", []).append(gh[m]); HP.setdefault("p", []).append(ph[m])
HP = {k: np.concatenate(v) for k, v in HP.items()}; TP = {k: np.concatenate(v) for k, v in TP.items()}
def cell(p, g, p10, g10):
    return f"x{p.sum() / max(g.sum(), 1e-6):4.2f} {csi(p, g, .2):.2f}/{csi(p, g, 1):.2f}/{csi(p, g, 3):.2f} t{csi(p10, g10, .3):.2f}/{csi(p10, g10, 1):.2f}"
configs = [c.split(":") for c in opt("--configs", "k200:12:1.5:3:0:0.33:3").split(",")]
Fs = [factors_for(c[0], *map(float, c[1:])) for c in configs]
print(f"inst {KEY}, {time.strftime('%d.%m.%H', time.gmtime(since/1000))}-{time.strftime('%d.%m.%H', time.gmtime(until/1000))}; configs (var:win:lag:prior:wet:minf:maxf): " + "  ".join(":".join(c) for c in configs))
ADJ = []
for F in Fs:
    PA = P.copy(); PHA = np.zeros((len(hours), n), np.float32)
    for h, H in enumerate(hours):
        i = ti[int(H)]
        for j in range(i + 1, i + 8):
            PA[j] = apply(F, P[j], np.full(n, h), np.arange(n))
        PHA[h] = np.nansum(PA[i + 1:i + 7] / 6, 0)
    ADJ.append((PA, PHA))
gH, gT = G_[HP["k"]], G_[TP["k"]]
for g in want + ["ALL"]:
    mh = (gH == g) if g != "ALL" else np.ones(len(gH), bool)
    mt = (gT == g) if g != "ALL" else np.ones(len(gT), bool)
    if mh.sum() == 0 or HP["g"][mh].sum() < 5: continue
    row = [cell(HP["p"][mh], HP["g"][mh], TP["p"][mt], TP["g"][mt])]
    for PA, PHA in ADJ:
        row.append(cell(PHA[HP["h"][mh], HP["k"][mh]], HP["g"][mh], PA[TP["j"][mt], TP["k"][mt]], TP["g"][mt]))
    print(f"{g:6s} n{mh.sum():6d} g{HP['g'][mh].mean():.3f} | " + " | ".join(row))
if opt("--save", None):
    np.savez_compressed(opt("--save", None), hours=hours, F=Fs[0][0], F2=Fs[0][1] if Fs[0][1] is not None else Fs[0][0])
if "--classes" in A:
    print("hourly sums by class: picture/gauge ratio (n) - conditioned on the GAUGE hour and on the PICTURE hour; raw -> adjusted (first config)")
    PHA = ADJ[0][1]
    for g in want:
        mh = gH == g
        if mh.sum() == 0: continue
        gg, pp, aa = HP["g"][mh], HP["p"][mh], PHA[HP["h"][mh], HP["k"][mh]]
        cells = []
        for lab, x in (("gauge", gg), ("pic", pp)):
            for a, b in ((0.2, 1), (1, 5), (5, 1e9)):
                m = (x >= a) & (x < b)
                if m.sum() >= 5:
                    cells.append(f"{lab} {a}-{b if b < 1e9 else 'inf'}: x{pp[m].sum()/max(gg[m].sum(),1e-6):.2f}->x{aa[m].sum()/max(gg[m].sum(),1e-6):.2f} (n{m.sum()})")
        print(f"{g:6s} " + " | ".join(cells))
