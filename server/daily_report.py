"""Daily record of every forecast comparison the servers make, so that after a few weeks there is a plain history of
who forecasts best (DWD, ours, the variants being tested) - without anyone having to remember to run the scores.

Once a day (REPORT_HOUR_UTC) the Germany instance runs the score scripts over the last 24 h on the shared verify_log
and writes their output to verify_log/daily/<YYYYMMDD>.txt:
  rain Germany    score_same.py   dwd own sprog0 model cells           (fcgrid_*: DWD's and our forecast, shadows)
  rain Europe     score_same.py   own sprog sprog0 pers --log eugrid   (whole OPERA area and WetterOnline's view)
  sunny Germany   score_sun.py    sat satold pers kpers model nwp
  sunny Europe    score_sun.py    the same with --prefix eu            (stations: DWD + Austria + Switzerland)
WetterOnline itself is scored on the user's desktop (tools/wo_verify.py, wo_verify_eu.py), never from the server.

  python daily_report.py            -> writes today's file now (also used by the loop)
"""
import os
import subprocess
import sys
import threading
import time
from datetime import datetime, timezone

HERE = os.path.dirname(os.path.abspath(__file__))
LOG_DIR = os.environ.get("VERIFY_LOG_DIR", os.path.join(HERE, "verify_log"))
REPORT_HOUR_UTC = 5
JOBS = [
    ("Regen Deutschland (CSI, Menge Vorhersage/real)", ["score_same.py", "dwd", "own", "sprog0", "model", "cells"]),
    ("Regen Europa, ganzes OPERA-Gebiet", ["score_same.py", "own", "sprog", "sprog0", "pers", "--log", "eugrid"]),
    ("Regen Europa, Ausschnitt wie WetterOnline", ["score_same.py", "own", "sprog", "sprog0", "pers", "--log", "eugrid", "--area", "wo"]),
    ("sunny Deutschland", ["score_sun.py", "sat", "satold", "pers", "kpers", "model", "nwp"]),
    ("sunny Europa", ["score_sun.py", "sat", "satold", "pers", "kpers", "model", "nwp", "--prefix", "eu"]),
]


def write_report(hours=24):
    out = [f"rainy Tagesbericht {datetime.now(timezone.utc):%Y-%m-%d %H:%M} UTC, letzte {hours} h", ""]
    for title, cmd in JOBS:
        try:
            r = subprocess.run([sys.executable, os.path.join(HERE, cmd[0]), *cmd[1:], "--hours", str(hours)],
                               capture_output=True, text=True, timeout=900, cwd=HERE)
            body = (r.stdout or r.stderr).strip()
        except Exception as e:
            body = f"(failed: {e})"
        out += [f"== {title}", body, ""]
    os.makedirs(os.path.join(LOG_DIR, "daily"), exist_ok=True)
    path = os.path.join(LOG_DIR, "daily", f"{datetime.now(timezone.utc):%Y%m%d}.txt")
    with open(path, "w", encoding="utf-8") as f:
        f.write("\n".join(out))
    return path


def loop(log):
    done = None
    while True:
        now = datetime.now(timezone.utc)
        if now.hour >= REPORT_HOUR_UTC and done != now.date():
            try:
                log(f"daily report: {write_report()}")
            except Exception as e:
                log(f"daily report failed: {e}")
            done = now.date()
        time.sleep(600)


def start(log):
    threading.Thread(target=loop, args=(log,), daemon=True).start()


if __name__ == "__main__":
    print(write_report())
