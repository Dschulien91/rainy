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
    ("Regen-Alarme: Anteil richtiger Alarme (Regen am Regenmesser innerhalb +-15 min um den vorhergesagten Beginn)", ["verify_alerts.py"]),
    ("Regen Europa, ganzes OPERA-Gebiet", ["score_same.py", "own", "pers", "--log", "eugrid"]),
    ("Regen Europa, zellgenau", ["score_same.py", "own", "pers", "--log", "eugrid", "--strict"]),
    ("Regen Europa: Varianten im Test (cells = mit Modell-Schauern, sprog0 = ohne Mitführen des letzten Bildes)",
     ["score_same.py", "own", "cells", "sprog0", "--log", "eugrid"]),
    ("Regen Europa, Ausschnitt wie WetterOnline", ["score_same.py", "own", "pers", "--log", "eugrid", "--area", "wo"]),
    ("Regen Europa an den österreichischen Regenmessern: eigene (own) gegen den Nowcast des österreichischen Wetterdienstes (inca) und 'bleibt so' (pers)",
     ["score_inca.py"]),
    ("sunny Deutschland", ["score_sun.py", "sat", "pers", "kpers", "nwp"]),
    ("sunny Deutschland: Varianten im Test", ["score_sun.py", "sat", "satold", "model"]),
    ("sunny Europa", ["score_sun.py", "sat", "pers", "kpers", "nwp", "--prefix", "eu"]),
    ("sunny Europa: Varianten im Test", ["score_sun.py", "sat", "satold", "model", "--prefix", "eu"]),
    ("Sprünge auf der Zeitachse (was man beim Verschieben sieht)", ["score_jumps.py"]),
]
# what a full day should hold: rows per lead (a run every 5 min for rain), health rows (every 5 min)
RUNS = {"fcgrid": ("dwd", 288), "eugrid": ("own", 288), "sungrid": ("sat", None), "eusungrid": ("sat", None),
        "sunstation": ("sat", None), "eusunstation": ("sat", None)}
MAX_STARTS, MAX_RAIN_AGE, MAX_SUN_AGE = 4, 20, 50


def _rows(name, day):
    fp = os.path.join(LOG_DIR, f"{name}_{day}.jsonl")
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
    starts = _rows("starts", day)
    health = _rows("health", day)
    for ed in sorted({r["edition"] for r in starts + health}) or ["?"]:
        st = [r for r in starts if r["edition"] == ed]
        hs = [r for r in health if r["edition"] == ed]
        vs = sorted({r["v"] for r in st + hs})
        out.append(f"Instanz {ed}: {len(st)} Starts, Versionen {', '.join(vs) or '-'}"
                   + (f"   !! mehr als {MAX_STARTS} Starts: Absturzschleife oder viele Deploys" if len(st) > MAX_STARTS else ""))
        if not hs:
            out.append("  !! keine Zustandsdaten (Instanz lief nicht, oder älter als server-v0.93)")
            continue
        up = len(hs) / 288
        out.append(f"  Laufzeit: {len(hs)} von 288 Messpunkten ({100 * up:.0f} %)" + ("   !! Lücke" if up < 0.9 else ""))
        for key, label, limit in (("rain_age", "Radarbild", MAX_RAIN_AGE), ("sun_age", "Satellitenbild", MAX_SUN_AGE)):
            v = [r[key] for r in hs if r.get(key) is not None]
            if v:
                late = sum(x > limit for x in v)
                out.append(f"  neuestes {label}: im Mittel {sum(v) / len(v):.0f} min alt, höchstens {max(v):.0f} min"
                           + (f"   !! {late} Messpunkte über {limit} min (Daten standen)" if late else ""))
            else:
                out.append(f"  !! kein {label} den ganzen Tag")
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
