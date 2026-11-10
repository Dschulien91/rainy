"""Runs tools/wo_capture.py every 15 min, for Germany and for the Europe beta (--eu), and tools/wo_sky.py once an hour (at :02/:17/:32/:47, after WetterOnline's radar update) so WetterOnline is
measured permanently, independent of any open browser window. Deletes wom_ shots older than KEEP_DAYS.
Log: tools/fc_shots/wo_loop.log.  Usage: python tools/wo_loop.py   (keeps running)
"""
import os
import subprocess
import sys
import time
from datetime import datetime

HERE = os.path.dirname(os.path.abspath(__file__))
SHOTS = os.path.join(HERE, "fc_shots")
KEEP_DAYS = 14


def log(msg):
    with open(os.path.join(SHOTS, "wo_loop.log"), "a", encoding="utf-8") as f:
        f.write(f"[{datetime.now():%d.%m. %H:%M:%S}] {msg}\n")


def prune():
    cutoff = time.time() - KEEP_DAYS * 86400
    for p in os.listdir(SHOTS):
        if p.startswith(("wom_", "woe_", "wos_", "sky_")) and os.path.getmtime(os.path.join(SHOTS, p)) < cutoff:
            os.remove(os.path.join(SHOTS, p))


def main():
    os.makedirs(SHOTS, exist_ok=True)
    log("loop started")
    while True:
        now = time.time()
        nxt = (now // 900 + 1) * 900 + 120  # 2 min after each quarter hour
        time.sleep(max(1.0, nxt - now))
        try:
            r = subprocess.run([sys.executable, os.path.join(HERE, "wo_capture.py")], capture_output=True, text=True,
                               timeout=600)
            log((r.stdout.strip() or r.stderr.strip()[-300:]) or f"exit {r.returncode}")
            # their Europe view for the Europe beta (tools/wo_verify_eu.py)
            r = subprocess.run([sys.executable, os.path.join(HERE, "wo_capture.py"), "--eu"], capture_output=True,
                               text=True, timeout=600)
            log("eu: " + ((r.stdout.strip() or r.stderr.strip()[-300:]) or f"exit {r.returncode}"))
            if time.localtime().tm_min < 15:  # once an hour: WetterOnline's cloud map next to sunny (tools/wo_sky.py)
                r = subprocess.run([sys.executable, os.path.join(HERE, "wo_sky.py")], capture_output=True, text=True,
                                   timeout=300)
                log("sky: " + ((r.stdout.strip() or r.stderr.strip()[-300:]) or f"exit {r.returncode}"))
            prune()
        except Exception as e:
            log(f"failed: {e}")


if __name__ == "__main__":
    main()
