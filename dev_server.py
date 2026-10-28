"""Local static server for the app with caching disabled, so a reload always shows the newest version.

  python dev_server.py [port]
"""
import functools
import os
import sys
from http.server import SimpleHTTPRequestHandler, ThreadingHTTPServer


class NoCache(SimpleHTTPRequestHandler):
    def end_headers(self):
        self.send_header("Cache-Control", "no-cache, must-revalidate")
        super().end_headers()


if __name__ == "__main__":
    port = int(sys.argv[1]) if len(sys.argv) > 1 else 8770
    handler = functools.partial(NoCache, directory=os.path.dirname(os.path.abspath(__file__)))
    ThreadingHTTPServer(("0.0.0.0", port), handler).serve_forever()
