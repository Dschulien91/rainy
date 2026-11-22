"""Daily record of everything the servers measure, so that after a few weeks there is a plain history of who forecasts
best (DWD, ours, the variants being tested) and of everything that went wrong - without anyone having to look.

Once a day (from REPORT_HOUR_UTC on) the Germany instance sums up the previous UTC day (00-24 h, a fixed window: every
day is counted once, and writing the report again gives the same numbers) from the shared verify_log and writes
verify_log/daily/<that day YYYYMMDD>.txt:
  state           starts and versions of both instances, age of the newest images, error messages, memory, and how
                  many runs were scored per lead (monitor.py) - gaps and crash loops show here, marked "!!"
  rain Germany    score_same.py  dwd own pers            against the later radar image (~9 km tolerance, and cell by cell)
                                 + the variants being tested, on the runs they share with dwd/own
                  score.py       the same forecasts against the rain gauges (what reached the ground)
  rain Europe     score_same.py  own pers --log eugrid   (whole OPERA area and WetterOnline's view), variants
                  score_inca.py  ours against GeoSphere Austria's INCA nowcast at the Austrian gauges (bench_inca.py)
  rain alerts     verify_alerts.py  share of alerts that were right, by lead time and intensity
  sunny           score_sun.py   sat pers kpers nwp, the variants, and the analysis itself at the stations
  jumps           score_jumps.py what a user sees when moving the timeline: break at "now", jumps between images
WetterOnline itself is scored on the user's desktop (tools/wo_verify.py, wo_verify_eu.py), never from the server.

  python daily_report.py [YYYYMMDD]   -> writes that day's file now (default: yesterday UTC)
"""
import glob
import json
import os
import subprocess
import sys
import threading
import time
from collections import Counter, defaultdict
from datetime import datetime, timedelta, timezone

HERE = os.path.dirname(os.path.abspath(__file__))
LOG_DIR = os.environ.get("VERIFY_LOG_DIR", os.path.join(HERE, "verify_log"))
REPORT_HOUR_UTC = 5
JOBS = [
    ("Regen Deutschland gegen das spätere Radarbild: DWD, eigene, 'bleibt wie es ist' (pers) - CSI mit ~9 km Toleranz (Menge Vorhersage/real)",
     ["score_same.py", "dwd", "own", "pers"]),
    ("Regen Deutschland, dasselbe zellgenau (ohne Toleranz: wer trifft die Ränder)", ["score_same.py", "dwd", "own", "pers", "--strict"]),
    ("Regen Deutschland: Varianten im Test (nur gemeinsame Läufe)", ["score_same.py", "dwd", "own", "sprog0", "model", "cells"]),
    ("Regen Deutschland an den Regenmessern (rainy = DWD-Vorhersage wie in der App, own = eigene; CSI POD FAR)", ["score.py", "--rain-only"]),
    ("Regen-Alarme: Anteil richtiger Alarme (Regen am Regenmesser innerhalb +-15 min um den vorhergesagten Beginn); point = Regel am Punkt, circle = echte Alarmregel (5-km-Kreis); reliability = wie oft es je nach vorhergesagtem nassen Anteil des Kreises wirklich regnete", ["verify_alerts.py"]),
    ("Regen Europa, ganzes OPERA-Gebiet", ["score_same.py", "own", "pers", "--log", "eugrid"]),
    ("Regen Europa, zellgenau", ["score_same.py", "own", "pers", "--log", "eugrid", "--strict"]),
    ("Regen Europa: Varianten im Test (oro = Bergregen bleibt stehen, inflow = Zufluss vom Rand, cells = mit Modell-Schauern, sprog0 = ohne Mitführen des letzten Bildes)",
     ["score_same.py", "own", "oro", "inflow", "cells", "sprog0", "--log", "eugrid"]),
    # NWP_SERVE (18.11.2026): own = served with the rapid-update model blend, own0 = without it, nwp = the 4 km reference
    ("Regen Europa: Modell-Blend (own = ausgeliefert mit Blend, own0 = ohne, nwp = 4-km-Referenz des Blends)",
     ["score_same.py", "own", "own0", "nwp", "--log", "eugrid"]),
    ("Regen Europa, Ausschnitt wie WetterOnline", ["score_same.py", "own", "pers", "--log", "eugrid", "--area", "wo"]),
    ("Regen Europa an den österreichischen Regenmessern: eigene (own) gegen den Nowcast des österreichischen Wetterdienstes (inca) und 'bleibt so' (pers)",
     ["score_inca.py"]),
    ("sunny Deutschland", ["score_sun.py", "sat", "pers", "kpers", "nwp"]),
    ("sunny Deutschland: Varianten im Test (sat = ausgeliefert mit Modell-Blend seit v1.84, satadv = reine Verschiebung)", ["score_sun.py", "sat", "satadv", "lowf03", "kpers"]),
    ("sunny Europa", ["score_sun.py", "sat", "pers", "kpers", "nwp", "--prefix", "eu"]),
    ("sunny Europa: Varianten im Test", ["score_sun.py", "sat", "sharp", "shblur", "satnwp", "lowf10", "metfog", "--prefix", "eu"]),
    ("Sprünge auf der Zeitachse (was man beim Verschieben sieht)", ["score_jumps.py"]),
]
# what a full day should hold: rows per lead (a run every 5 min for rain), health rows (every 5 min)
RUNS = {"fcgrid": ("dwd", 288), "eugrid": ("own", 288), "sungrid": ("sat", None), "eusungrid": ("sat", None),
        "sunstation": ("sat", None), "eusunstation": ("sat", None)}
MAX_STARTS, MAX_RAIN_AGE, MAX_SUN_AGE = 4, 20, 50
SUB_INSTANCES = ("eun", "na")


def _rows(name, day, sub=""):
    fp = os.path.join(LOG_DIR, sub, f"{name}_{day}.jsonl")
    if not os.path.exists(fp):
        return []
    out = []
    for line in open(fp):
        try:
            out.append(json.loads(line))
        except ValueError:  # a line cut off by a kill
            pass
    return out


def state(day):
    """The 'Zustand' section: lines, the ones needing attention marked '!!'."""
    out = []
    # (label, rows): the de and eu instances share the root folder; the own Europe picture ("eun") and North America
    # ("na") write into sub-folders and call themselves "eu" - until 16.11. they were missing here (a 135-start crash
    # loop of na was invisible in this report)
    groups = []
    starts, health = _rows("starts", day), _rows("health", day)
    for ed in sorted({r["edition"] for r in starts + health}) or ["?"]:
        groups.append((ed, [r for r in starts if r["edition"] == ed], [r for r in health if r["edition"] == ed]))
    for sub in SUB_INSTANCES:
        st, hs = _rows("starts", day, sub), _rows("health", day, sub)
        if st or hs:
            groups.append((sub, st, hs))
    for ed, st, hs in groups:
        vs = sorted({r["v"] for r in st + hs})
        out.append(f"Instanz {ed}: {len(st)} Starts, Versionen {', '.join(vs) or '-'}"
                   + (f"   !! mehr als {MAX_STARTS} Starts: Absturzschleife oder viele Deploys" if len(st) > MAX_STARTS else ""))
        if not hs:
            out.append("  !! keine Zustandsdaten (Instanz lief nicht, oder älter als server-v0.93)")
            continue
        up = len(hs) / 288
        out.append(f"  Laufzeit: {len(hs)} von 288 Messpunkten ({100 * up:.0f} %)" + ("   !! Lücke" if up < 0.9 else ""))
        # the own Europe picture waits up to national.WAIT_MIN (20 min) for its national layers, so a rain age of
        # 20-25 min there is by design (the ops watch allows +10 min for eun as well); only flag it beyond that
        rain_limit = MAX_RAIN_AGE + (10 if ed == "eun" else 0)
        for key, label, limit in (("rain_age", "Radarbild", rain_limit), ("sun_age", "Satellitenbild", MAX_SUN_AGE)):
            if ed == "eun" and key == "sun_age":  # the own Europe picture has no sunny
                continue
            v = [r[key] for r in hs if r.get(key) is not None]
            if v:
                late = sum(x > limit for x in v)
                out.append(f"  neuestes {label}: im Mittel {sum(v) / len(v):.0f} min alt, höchstens {max(v):.0f} min"
                           + (f"   !! {late} Messpunkte über {limit} min (Daten standen)" if late else ""))
            else:
                out.append(f"  !! kein {label} den ganzen Tag")
        out += sources(hs)
        rss = [r["rss"] for r in hs if r.get("rss")]
        if rss:
            out.append(f"  Speicher: im Mittel {sum(rss) / len(rss):.0f} MB, höchstens {max(rss)} MB")
        errors = Counter()
        for r in hs:
            errors.update(r.get("errors") or {})
        for text, n in errors.most_common(8):
            out.append(f"  {'!! ' if n >= 10 else ''}Fehlermeldung {n}x: {text}")
        if not errors:
            out.append("  keine Fehlermeldungen")
    out.append("Ausgewertete Läufe je Vorlauf (fehlen welche, fehlen Messungen):")
    for name, (variant, full) in RUNS.items():
        per = defaultdict(set)
        for r in _rows(name, day):
            if r["variant"] == variant and r.get("area", "all") == "all":
                per[r["lead"]].add(r["ref"])
        if not per:
            out.append(f"  !! {name}: keine Zeilen")
            continue
        most = full or max(len(v) for v in per.values())
        low = any(len(v) < 0.85 * most for v in per.values())
        out.append(f"  {name}: " + " | ".join(f"+{lead}: {len(per[lead])}" for lead in sorted(per))
                   + (f"   (voll: {full})" if full else "") + ("   !! unvollständig" if low else ""))
    return out


# how old a source's newest data may be before it counts as missing (minutes): OPERA is normally 5-10 min late and
# listed every minute or two, the national layers and DWD ~5 min, the sunny satellite every 10-15 min
SOURCE_LIMIT = {"opera": 25, "mrms": 20, "dwd": 20, "sunny": MAX_SUN_AGE}
SOURCE_DEFAULT = 25


def sources(hs):
    """Per source (health rows "src", server-v1.70): minutes it was missing that day and the longest outage."""
    out, names = [], sorted({k for r in hs for k in (r.get("src") or {})})
    for name in names:
        limit = SOURCE_LIMIT.get(name, SOURCE_DEFAULT)
        late = run = longest = 0
        seen = False
        for r in hs:  # rows every 5 minutes, in time order
            if "src" not in r:  # written before server-v1.70 (no source ages yet): unknown, not an outage
                continue
            # a source added in a later server version is absent from the rows written before it: unknown, not an
            # outage (20.11.: thunder / lightning / warn:* came with v1.79 at 21:36Z and showed as "1290 min missing")
            if not seen and name not in (r.get("src") or {}):
                continue
            seen = True
            v = (r.get("src") or {}).get(name)
            if v is None or v > limit:
                late += 1
                run += 1
                longest = max(longest, run)
            else:
                run = 0
        # a single late row (5 min) is normal provider jitter; flag "!!" from 15 minutes in a row
        flag = "!! " if longest >= 3 else ""
        out.append(f"  Quelle {name}: " + (f"{flag}{5 * late} min ohne neue Daten (längster Ausfall {5 * longest} min)" if late
                                           else "lückenlos"))
    return out


def write_report(day=None):
    day = day or (datetime.now(timezone.utc) - timedelta(days=1)).strftime("%Y%m%d")
    out = [f"rainy Tagesbericht für {day[6:]}.{day[4:6]}.{day[:4]} (00-24 Uhr UTC), erstellt {datetime.now(timezone.utc):%d.%m. %H:%M} UTC", "",
           "== Zustand der Server und der Messungen"]
    try:
        out += state(day)
    except Exception as e:
        out.append(f"(failed: {e})")
    out.append("")
    env = {**os.environ, "PYTHONIOENCODING": "utf-8"}
    for title, cmd in JOBS:
        try:
            r = subprocess.run([sys.executable, os.path.join(HERE, cmd[0]), *cmd[1:], "--day", day], capture_output=True,
                               text=True, encoding="utf-8", errors="replace", timeout=900, cwd=HERE, env=env)
            body = (r.stdout or r.stderr).strip()
        except Exception as e:
            body = f"(failed: {e})"
        out += [f"== {title}", body, ""]
    os.makedirs(os.path.join(LOG_DIR, "daily"), exist_ok=True)
    path = os.path.join(LOG_DIR, "daily", f"{day}.txt")
    with open(path, "w", encoding="utf-8") as f:
        f.write("\n".join(out))
    return path


def loop(log):
    while True:
        now = datetime.now(timezone.utc)
        day = (now - timedelta(days=1)).strftime("%Y%m%d")
        if now.hour >= REPORT_HOUR_UTC and not os.path.exists(os.path.join(LOG_DIR, "daily", f"{day}.txt")):
            try:
                log(f"daily report: {write_report(day)}")
            except Exception as e:
                log(f"daily report failed: {e}")
        time.sleep(600)


def start(log):
    threading.Thread(target=loop, args=(log,), daemon=True).start()


if __name__ == "__main__":
    print(write_report(sys.argv[1] if len(sys.argv) > 1 else None))
