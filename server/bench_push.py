"""How many Web Push messages per second can this machine encrypt and send? (sizes the push worker pool)

Starts a dummy push service on localhost that answers 201 after a delay like a real one (default 80 ms), creates
fake browser subscriptions with real keys and sends real encrypted pushes with pywebpush through N parallel workers.

  python bench_push.py [--n 600] [--latency 0.08]
"""
import argparse
import base64
import json
import threading
import time
from concurrent.futures import ThreadPoolExecutor
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

from cryptography.hazmat.primitives import serialization
from cryptography.hazmat.primitives.asymmetric import ec
from py_vapid import Vapid01
from pywebpush import webpush

LATENCY = 0.08


class Sink(BaseHTTPRequestHandler):
    def do_POST(self):
        self.rfile.read(int(self.headers.get("Content-Length", "0")))
        time.sleep(LATENCY)
        self.send_response(201)
        self.end_headers()

    def log_message(self, *a):
        pass


def b64(b):
    return base64.urlsafe_b64encode(b).rstrip(b"=").decode()


def fake_subscription(port):
    key = ec.generate_private_key(ec.SECP256R1())
    pub = key.public_key().public_bytes(serialization.Encoding.X962, serialization.PublicFormat.UncompressedPoint)
    return {"endpoint": f"http://127.0.0.1:{port}/push", "keys": {"p256dh": b64(pub), "auth": b64(b"0123456789abcdef")}}


def main():
    global LATENCY
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=600)
    ap.add_argument("--latency", type=float, default=0.08)
    a = ap.parse_args()
    LATENCY = a.latency
    ThreadingHTTPServer.request_queue_size = 256  # the dummy must not be the bottleneck
    srv = ThreadingHTTPServer(("127.0.0.1", 0), Sink)
    threading.Thread(target=srv.serve_forever, daemon=True).start()
    port = srv.server_address[1]
    vapid = Vapid01()  # same key object as alerts.py
    vapid.generate_keys()
    subs = [fake_subscription(port) for _ in range(a.n)]
    payload = json.dumps({"title": "🌧 Mäßiger Regen in 45 Min. – Hamburg", "body": "Ab ca. 17:40, etwa 30 Min. lang. Wahrscheinlichkeit ca. 60 %."})

    def one(sub):
        webpush(sub, payload, vapid_private_key=vapid, vapid_claims={"sub": "https://example.invalid"}, ttl=900)

    for workers in (1, 8, 16, 32, 64):
        n = min(a.n, 60) if workers == 1 else a.n
        t0 = time.time()
        with ThreadPoolExecutor(workers) as ex:
            list(ex.map(one, subs[:n]))
        dt = time.time() - t0
        print(f"{workers:3d} workers: {n} pushes in {dt:5.1f} s = {n / dt:6.1f}/s", flush=True)


if __name__ == "__main__":
    main()
