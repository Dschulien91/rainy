"""Optional user login for rainy: Google (OpenID Connect) and a passwordless email link.

Purpose: sync favourites and settings across devices. Everything in the app works without an account.

Storage (SQLite, DATA_DIR/users.db): users, sessions, login_tokens, user_data. Session and login tokens are stored
only as SHA-256 hashes, so a copy of the database cannot be used to log in.

Endpoints (all answers carry Cache-Control: no-store; writes need a same-site Origin/Referer):
  GET  /api/me                          -> {"email", "providers": {"google", "email"}}  (401 with email null when logged out)
                                           (the app asks only when the readable companion cookie rainy_in=1 exists;
                                           the sign-in methods for anonymous visitors come from /api/health "auth")
  GET  /api/auth/google?return=/path    -> 302 to Google (503 when GOOGLE_CLIENT_ID/SECRET are not set)
  GET  /api/auth/google/callback        -> session cookie, 302 to the return path (?login=failed on errors)
  POST /api/auth/email {email, lang, return} -> {"ok": true}  (never reveals whether an account exists)
  GET  /api/auth/email/verify?token=&return=&lang=  -> small HTML page with a "Sign in" button; does NOT use up the
                                           token (mail scanners and link previews open links too); expired/used ->
                                           page with a link back to the app
  POST /api/auth/email/verify (form: token, return) -> uses up the token, session cookie, 302 to the return path
  POST /api/auth/logout                 -> {"ok": true}, deletes the session and expires the cookie
  GET  /api/user/data                   -> {"data": {...}, "updated": ms}
  PUT  /api/user/data {data}            -> {"ok": true, "updated": ms}  (POST works too; JSON object, max 64 KB)
  DELETE /api/user (or POST /api/user/delete) -> {"ok": true}: deletes the account with its sessions, data and
                                           open login links, and expires the cookies

Configuration (env): PUBLIC_URL, GOOGLE_CLIENT_ID, GOOGLE_CLIENT_SECRET, AUTH_SECRET, SMTP_HOST, SMTP_PORT,
SMTP_USER, SMTP_PASS, MAIL_FROM. API_URL (optional) is where /api lives when it is not under PUBLIC_URL - only for
local development, where the app runs on :8770 and this server on :8771.
Privacy: tokens, email addresses and secrets are never logged. Only exception: with RAINY_DEV=1 and no SMTP_HOST
the login link is written to the log instead of being mailed (local testing without a mail server).
"""
import base64
import hashlib
import hmac
import html
import json
import os
import re
import secrets
import smtplib
import sqlite3
import ssl
import threading
import time
import urllib.parse
import urllib.request
from email.message import EmailMessage
from email.utils import formatdate, make_msgid

DEV = os.environ.get("RAINY_DEV") == "1"
PUBLIC_URL = os.environ.get("PUBLIC_URL", "http://localhost:8770" if DEV else "https://hellorainy.com").rstrip("/")
API_URL = os.environ.get("API_URL", "http://localhost:8771" if DEV else PUBLIC_URL).rstrip("/")
GOOGLE_CLIENT_ID = os.environ.get("GOOGLE_CLIENT_ID", "")
GOOGLE_CLIENT_SECRET = os.environ.get("GOOGLE_CLIENT_SECRET", "")
# Signs the short-lived OAuth state cookie. Without AUTH_SECRET a random per-process key is used: logins that are
# in progress during a restart then fail once, nothing else depends on it.
AUTH_SECRET = (os.environ.get("AUTH_SECRET") or secrets.token_hex(32)).encode()
SMTP_HOST = os.environ.get("SMTP_HOST", "")
SMTP_PORT = int(os.environ.get("SMTP_PORT") or 465)
SMTP_USER = os.environ.get("SMTP_USER", "")
SMTP_PASS = os.environ.get("SMTP_PASS", "")
MAIL_FROM = os.environ.get("MAIL_FROM") or "rainy <hey@hellorainy.com>"

GOOGLE_AUTH = "https://accounts.google.com/o/oauth2/v2/auth"
GOOGLE_TOKEN = "https://oauth2.googleapis.com/token"
GOOGLE_ISS = ("accounts.google.com", "https://accounts.google.com")

SESSION_COOKIE = "rainy_session"
# Companion cookie without any secret, readable by the app: "there may be a session, ask /api/me". Anonymous
# visitors then never call /api/me (no 401 in their browser console, no extra request).
FLAG_COOKIE = "rainy_in"
SESSION_S = 180 * 86400
OAUTH_COOKIE = "rainy_oauth"
OAUTH_S = 600
LOGIN_TOKEN_S = 15 * 60
MAX_DATA = 64 * 1024
RATE_EMAIL = (3, 15 * 60)   # login mails per address and window
RATE_IP = (20, 3600)        # login mails per IP and window
EMAIL_RE = re.compile(r"[^@\s\"<>(),;:\\]{1,64}@[A-Za-z0-9](?:[A-Za-z0-9-]*[A-Za-z0-9])?(?:\.[A-Za-z0-9](?:[A-Za-z0-9-]*[A-Za-z0-9])?)+")
DEV_HOST = re.compile(r"(localhost|127\.0\.0\.1|192\.168\.\d+\.\d+|10\.\d+\.\d+\.\d+|172\.(1[6-9]|2\d|3[01])\.\d+\.\d+)(:\d+)?")
PATHS = ("/api/me", "/api/user", "/api/auth/", "/api/user/")

MAIL_TEXT = {
    "de": ("Dein Anmeldelink für rainy",
           "Hallo,\n\nmit diesem Link meldest du dich bei rainy an:\n\n{link}\n\n"
           "Der Link gilt 15 Minuten und funktioniert nur einmal.\n"
           "Wenn du das nicht angefordert hast, kannst du diese Mail einfach ignorieren.\n\n"
           "rainy – hellorainy.com\n"),
    "en": ("Your sign-in link for rainy",
           "Hi,\n\nuse this link to sign in to rainy:\n\n{link}\n\n"
           "The link is valid for 15 minutes and works only once.\n"
           "If you did not ask for it, you can simply ignore this email.\n\n"
           "rainy – hellorainy.com\n"),
}


# Confirmation page of the mailed link: self-contained (no external resources), colours of the app's dark default.
PAGE_TEXT = {
    "de": {"title": "Bei rainy anmelden", "text": "Tippe auf den Knopf, um dich auf diesem Gerät anzumelden.",
           "button": "Anmelden", "expiredTitle": "Link abgelaufen",
           "expired": "Dieser Anmeldelink ist abgelaufen oder wurde schon benutzt. Fordere in rainy einfach einen neuen an.",
           "back": "Zu rainy"},
    "en": {"title": "Sign in to rainy", "text": "Tap the button to sign in on this device.",
           "button": "Sign in", "expiredTitle": "Link expired",
           "expired": "This sign-in link has expired or has already been used. Just request a new one in rainy.",
           "back": "Go to rainy"},
}
PAGE_CSS = ("body{margin:0;min-height:100vh;display:flex;align-items:center;justify-content:center;background:#15181e;"
            "color:#f1f3f6;font-family:system-ui,-apple-system,'Segoe UI',Roboto,sans-serif}"
            "main{width:100%;max-width:380px;padding:24px 20px;box-sizing:border-box}"
            ".brand{font-weight:800;font-size:20px;color:#8FB2FF;margin-bottom:28px}"
            "h1{font-size:24px;margin:0 0 10px}p{font-size:15px;line-height:1.45;color:#9aa2ae;margin:0 0 24px}"
            "button,.btn{display:block;width:100%;box-sizing:border-box;font:inherit;font-size:16px;font-weight:600;"
            "text-align:center;text-decoration:none;padding:13px 16px;border-radius:999px;border:0;background:#2f7bf5;"
            "color:#fff;cursor:pointer}")


def is_auth_path(path):
    return path in PATHS[:2] or path.startswith(PATHS[2:])


def _b64(data):
    return base64.urlsafe_b64encode(data).rstrip(b"=").decode()


def _unb64(text):
    return base64.urlsafe_b64decode(text + "=" * (-len(text) % 4))


def _hash(token):
    return hashlib.sha256(token.encode()).hexdigest()


def _origin(url):
    p = urllib.parse.urlsplit(url)
    return f"{p.scheme}://{p.netloc}".lower() if p.scheme and p.netloc else None


def safe_return(path):
    """Only same-site relative paths ("/..."), never "//host" or "/\\host"; anything else becomes "/"."""
    path = (path or "").strip()
    if (not path.startswith("/") or path.startswith("//") or "\\" in path or len(path) > 512
            or any(ord(c) < 32 or ord(c) == 127 for c in path)):
        return "/"
    p = urllib.parse.urlsplit(path)
    if p.scheme or p.netloc:
        return "/"
    return urllib.parse.urlunsplit(("", "", p.path, p.query, ""))


def with_param(path, key, value):
    """The return path plus ?key=value (replacing an existing key), for the app to show a message."""
    p = urllib.parse.urlsplit(path)
    q = [(k, v) for k, v in urllib.parse.parse_qsl(p.query, keep_blank_values=True) if k != key] + [(key, value)]
    return urllib.parse.urlunsplit(("", "", p.path, urllib.parse.urlencode(q), ""))


class RateLimit:
    """Sliding-window counters in memory (reset on restart, which is fine for abuse protection)."""

    def __init__(self):
        self.hits, self.lock = {}, threading.Lock()

    def allow(self, key, limit, window):
        now = time.time()
        with self.lock:
            if len(self.hits) > 50_000:  # drop stale keys now and then
                self.hits = {k: v for k, v in self.hits.items() if v and v[-1] > now - 3600}
            hits = [t for t in self.hits.get(key, ()) if t > now - window]
            ok = len(hits) < limit
            if ok:
                hits.append(now)
            self.hits[key] = hits
            return ok


class Auth:
    def __init__(self, data_dir, log):
        self.log = log
        os.makedirs(data_dir, exist_ok=True)
        self.db = sqlite3.connect(os.path.join(data_dir, "users.db"), check_same_thread=False, isolation_level=None)
        self.lock = threading.Lock()
        self.rate = RateLimit()
        with self.lock:
            self.db.execute("PRAGMA journal_mode=WAL")
            self.db.executescript("""
                CREATE TABLE IF NOT EXISTS users (id INTEGER PRIMARY KEY, email TEXT UNIQUE NOT NULL,
                    google_sub TEXT UNIQUE, created INTEGER NOT NULL);
                CREATE TABLE IF NOT EXISTS sessions (token_hash TEXT PRIMARY KEY, user_id INTEGER NOT NULL,
                    created INTEGER NOT NULL, expires INTEGER NOT NULL);
                CREATE INDEX IF NOT EXISTS sessions_user ON sessions(user_id);
                CREATE TABLE IF NOT EXISTS login_tokens (token_hash TEXT PRIMARY KEY, email TEXT NOT NULL,
                    expires INTEGER NOT NULL, used INTEGER NOT NULL DEFAULT 0);
                CREATE TABLE IF NOT EXISTS user_data (user_id INTEGER PRIMARY KEY, json TEXT NOT NULL,
                    updated INTEGER NOT NULL);
            """)
        self.public_origin = _origin(PUBLIC_URL)
        self.secure = not DEV  # local development runs on plain http
        log(f"auth: google {'on' if self.google_on() else 'off'}, email {'on' if self.email_on() else 'off'}"
            f"{' (dev: links in this log)' if DEV and not SMTP_HOST else ''}")

    # ---------- config ----------
    @staticmethod
    def google_on():
        return bool(GOOGLE_CLIENT_ID and GOOGLE_CLIENT_SECRET)

    @staticmethod
    def email_on():
        return bool(SMTP_HOST) or DEV

    def providers(self):
        return {"google": self.google_on(), "email": self.email_on()}

    # ---------- HTTP helpers ----------
    def allowed_origin(self, origin):
        if not origin:
            return False
        origin = origin.lower()
        if origin == self.public_origin:
            return True
        if DEV:
            p = urllib.parse.urlsplit(origin)
            return p.scheme in ("http", "https") and bool(DEV_HOST.fullmatch(p.netloc))
        return False

    def cors(self, h):
        """Credentialed CORS only for our own site (needed locally: app :8770, api :8771); never '*'."""
        origin = h.headers.get("Origin")
        if self.allowed_origin(origin):
            h.send_header("Access-Control-Allow-Origin", origin)
            h.send_header("Access-Control-Allow-Credentials", "true")
        h.send_header("Vary", "Origin")

    def same_site(self, h):
        """Writes must come from our own pages: Origin, else Sec-Fetch-Site, else Referer must say so. Browsers send
        Origin "null" for same-origin POSTs from pages with Referrer-Policy no-referrer (Caddy sets it site-wide), so
        Sec-Fetch-Site: same-origin (set by the browser, scripts cannot change it) counts as well."""
        src = h.headers.get("Origin")
        if src and src != "null":
            return self.allowed_origin(src)
        if h.headers.get("Sec-Fetch-Site") == "same-origin":
            return True
        ref = h.headers.get("Referer")
        return self.allowed_origin(_origin(ref) if ref else None)

    def reply(self, h, code, obj, cookies=()):
        body = json.dumps(obj).encode()
        h.send_response(code)
        h.send_header("Content-Type", "application/json")
        h.send_header("Cache-Control", "no-store")
        self.cors(h)
        for c in cookies:
            h.send_header("Set-Cookie", c)
        h.send_header("Content-Length", str(len(body)))
        h.end_headers()
        h.wfile.write(body)

    def redirect(self, h, url, cookies=()):
        h.send_response(302)
        h.send_header("Location", url)
        h.send_header("Cache-Control", "no-store")
        h.send_header("Referrer-Policy", "no-referrer")  # never leak the token in the URL to the next page
        for c in cookies:
            h.send_header("Set-Cookie", c)
        h.send_header("Content-Length", "0")
        h.end_headers()

    def cookie(self, name, value, max_age, path="/", http_only=True):
        return (f"{name}={value}; Path={path}; Max-Age={max_age}{'; HttpOnly' if http_only else ''}; SameSite=Lax"
                + ("; Secure" if self.secure else ""))

    def end_session_cookies(self):
        return [self.cookie(SESSION_COOKIE, "", 0), self.cookie(FLAG_COOKIE, "", 0, http_only=False)]

    @staticmethod
    def read_cookie(h, name):
        # parsed by hand: SimpleCookie gives up on the whole header when any other cookie is malformed
        for part in (h.headers.get("Cookie") or "").split(";"):
            k, _, v = part.strip().partition("=")
            if k == name:
                return v.strip().strip('"') or None
        return None

    @staticmethod
    def client_ip(h):
        ip = h.headers.get("CF-Connecting-IP")
        if not ip and h.headers.get("X-Forwarded-For"):
            ip = h.headers["X-Forwarded-For"].split(",")[0]
        return (ip or h.client_address[0]).strip()

    def read_json(self, h, limit):
        n = int(h.headers.get("Content-Length") or 0)
        if n > limit:
            raise ValueError("too large")
        body = json.loads(h.rfile.read(n) or b"{}")
        if not isinstance(body, dict):
            raise ValueError("not an object")
        return body

    # ---------- sessions and users ----------
    def start_session(self, user_id):
        token, now = secrets.token_urlsafe(32), int(time.time())
        with self.lock:
            self.db.execute("DELETE FROM sessions WHERE expires < ?", (now,))
            self.db.execute("INSERT INTO sessions VALUES (?, ?, ?, ?)", (_hash(token), user_id, now, now + SESSION_S))
        return [self.cookie(SESSION_COOKIE, token, SESSION_S), self.cookie(FLAG_COOKIE, "1", SESSION_S, http_only=False)]

    def session_user(self, h):
        """(user_id, email) of a valid session cookie, else None."""
        token = self.read_cookie(h, SESSION_COOKIE)
        if not token or len(token) > 100:
            return None
        with self.lock:
            return self.db.execute("SELECT u.id, u.email FROM sessions s JOIN users u ON u.id = s.user_id "
                                   "WHERE s.token_hash = ? AND s.expires > ?", (_hash(token), int(time.time()))).fetchone()

    def user_by_email(self, email):
        """Existing user with this (verified) address, or a new one."""
        with self.lock:
            row = self.db.execute("SELECT id FROM users WHERE email = ?", (email,)).fetchone()
            if row:
                return row[0]
            return self.db.execute("INSERT INTO users (email, created) VALUES (?, ?)", (email, int(time.time()))).lastrowid

    def user_by_google(self, sub, email):
        """By Google account id; else link to the account with the same verified address; else a new user."""
        with self.lock:
            row = self.db.execute("SELECT id FROM users WHERE google_sub = ?", (sub,)).fetchone()
            if row:
                return row[0]
            row = self.db.execute("SELECT id, google_sub FROM users WHERE email = ?", (email,)).fetchone()
            if row:
                if row[1] is None:
                    self.db.execute("UPDATE users SET google_sub = ? WHERE id = ?", (sub, row[0]))
                return row[0]
            return self.db.execute("INSERT INTO users (email, google_sub, created) VALUES (?, ?, ?)",
                                   (email, sub, int(time.time()))).lastrowid

    # ---------- routing ----------
    def handle(self, h, method, u):
        q = {k: v[0] for k, v in urllib.parse.parse_qs(u.query).items()}
        path = u.path
        try:
            if method in ("POST", "PUT", "DELETE") and not self.same_site(h):
                return self.reply(h, 403, {"error": "origin"})
            if path == "/api/me" and method == "GET":
                user = self.session_user(h)
                if not user:  # also drop a stale companion cookie, so the app stops asking
                    return self.reply(h, 401, {"email": None, "providers": self.providers()}, self.end_session_cookies())
                return self.reply(h, 200, {"email": user[1], "providers": self.providers()})
            if path == "/api/auth/google" and method == "GET":
                return self.google_start(h, q)
            if path == "/api/auth/google/callback" and method == "GET":
                return self.google_callback(h, q)
            if path == "/api/auth/email" and method == "POST":
                return self.email_start(h)
            if path == "/api/auth/email/verify" and method == "GET":
                return self.email_verify_page(h, q)
            if path == "/api/auth/email/verify" and method == "POST":
                return self.email_verify(h)
            if path == "/api/auth/logout" and method == "POST":
                token = self.read_cookie(h, SESSION_COOKIE)
                if token:
                    with self.lock:
                        self.db.execute("DELETE FROM sessions WHERE token_hash = ?", (_hash(token),))
                return self.reply(h, 200, {"ok": True}, self.end_session_cookies())
            if (path == "/api/user" and method == "DELETE") or (path == "/api/user/delete" and method == "POST"):
                return self.delete_user(h)
            if path == "/api/user/data":
                user = self.session_user(h)
                if not user:
                    return self.reply(h, 401, {"error": "login"})
                if method == "GET":
                    with self.lock:
                        row = self.db.execute("SELECT json, updated FROM user_data WHERE user_id = ?", (user[0],)).fetchone()
                    return self.reply(h, 200, {"data": json.loads(row[0]) if row else {}, "updated": row[1] if row else None})
                if method in ("PUT", "POST"):
                    return self.save_data(h, user[0])
            return self.reply(h, 404, {"error": "not found"})
        except Exception as e:  # message only, no request data (may contain addresses or tokens)
            self.log(f"auth: {path} failed ({type(e).__name__})")
            return self.reply(h, 400, {"error": "bad request"})

    def preflight(self, h):
        h.send_response(204)
        self.cors(h)
        h.send_header("Access-Control-Allow-Methods", "GET, POST, PUT, DELETE, OPTIONS")
        h.send_header("Access-Control-Allow-Headers", "Content-Type")
        h.send_header("Access-Control-Max-Age", "86400")
        h.send_header("Cache-Control", "no-store")
        h.end_headers()

    # ---------- account deletion ----------
    def delete_user(self, h):
        """Everything stored about the account goes: user row, all sessions, synced data, open login links."""
        user = self.session_user(h)
        if not user:
            return self.reply(h, 401, {"error": "login"})
        with self.lock:
            self.db.execute("BEGIN")
            try:
                self.db.execute("DELETE FROM sessions WHERE user_id = ?", (user[0],))
                self.db.execute("DELETE FROM user_data WHERE user_id = ?", (user[0],))
                self.db.execute("DELETE FROM login_tokens WHERE email = ?", (user[1],))
                self.db.execute("DELETE FROM users WHERE id = ?", (user[0],))
                self.db.execute("COMMIT")
            except Exception:
                self.db.execute("ROLLBACK")
                raise
        self.log("auth: account deleted")
        return self.reply(h, 200, {"ok": True}, self.end_session_cookies())

    # ---------- user data ----------
    def save_data(self, h, user_id):
        n = int(h.headers.get("Content-Length") or 0)
        if n > MAX_DATA + 1024:
            return self.reply(h, 413, {"error": "too large"})
        data = self.read_json(h, MAX_DATA + 1024).get("data")
        if not isinstance(data, dict):
            return self.reply(h, 400, {"error": "data must be an object"})
        text = json.dumps(data, ensure_ascii=False, separators=(",", ":"))
        if len(text.encode()) > MAX_DATA:
            return self.reply(h, 413, {"error": "too large"})
        now = int(time.time() * 1000)
        with self.lock:
            self.db.execute("INSERT INTO user_data VALUES (?, ?, ?) ON CONFLICT(user_id) DO UPDATE SET json = excluded.json, "
                            "updated = excluded.updated", (user_id, text, now))
        return self.reply(h, 200, {"ok": True, "updated": now})

    # ---------- Google (OpenID Connect, authorization code + PKCE) ----------
    def sign(self, obj):
        payload = _b64(json.dumps(obj, separators=(",", ":")).encode())
        return payload + "." + _b64(hmac.new(AUTH_SECRET, payload.encode(), hashlib.sha256).digest())

    def unsign(self, text):
        try:
            payload, sig = (text or "").split(".")
            good = _b64(hmac.new(AUTH_SECRET, payload.encode(), hashlib.sha256).digest())
            if not hmac.compare_digest(sig, good):
                return None
            obj = json.loads(_unb64(payload))
            return obj if obj.get("exp", 0) > time.time() else None
        except Exception:
            return None

    def google_start(self, h, q):
        if not self.google_on():
            return self.reply(h, 503, {"error": "google login not configured"})
        state, nonce, verifier = secrets.token_urlsafe(24), secrets.token_urlsafe(24), secrets.token_urlsafe(48)
        challenge = _b64(hashlib.sha256(verifier.encode()).digest())
        params = {"response_type": "code", "client_id": GOOGLE_CLIENT_ID, "scope": "openid email",
                  "redirect_uri": f"{API_URL}/api/auth/google/callback", "state": state, "nonce": nonce,
                  "code_challenge": challenge, "code_challenge_method": "S256", "prompt": "select_account"}
        flow = self.sign({"s": state, "n": nonce, "v": verifier, "r": safe_return(q.get("return")), "exp": int(time.time()) + OAUTH_S})
        return self.redirect(h, f"{GOOGLE_AUTH}?{urllib.parse.urlencode(params)}",
                             [self.cookie(OAUTH_COOKIE, flow, OAUTH_S, path="/api/auth/google")])

    def google_callback(self, h, q):
        flow = self.unsign(self.read_cookie(h, OAUTH_COOKIE))
        back = flow["r"] if flow else "/"
        clear = [self.cookie(OAUTH_COOKIE, "", 0, path="/api/auth/google")]

        def fail(why):
            self.log(f"auth: google login failed ({why})")
            return self.redirect(h, PUBLIC_URL + with_param(back, "login", "failed"), clear)
        if not self.google_on():
            return fail("not configured")
        if not flow or not q.get("state") or not hmac.compare_digest(q["state"], flow["s"]):
            return fail("state")
        if q.get("error") or not q.get("code"):  # e.g. access_denied: the user cancelled at Google, no message needed
            return self.redirect(h, PUBLIC_URL + back, clear)
        form = urllib.parse.urlencode({"code": q["code"], "client_id": GOOGLE_CLIENT_ID, "client_secret": GOOGLE_CLIENT_SECRET,
                                       "redirect_uri": f"{API_URL}/api/auth/google/callback",
                                       "grant_type": "authorization_code", "code_verifier": flow["v"]}).encode()
        req = urllib.request.Request(GOOGLE_TOKEN, data=form, headers={"Content-Type": "application/x-www-form-urlencoded"})
        try:
            with urllib.request.urlopen(req, timeout=15, context=ssl.create_default_context()) as r:
                tok = json.loads(r.read())
        except Exception as e:
            return fail(f"token endpoint {type(e).__name__}")
        # The id_token comes straight from Google's token endpoint over TLS, so per Google's OpenID Connect docs its
        # signature need not be checked; the claims still are.
        try:
            claims = json.loads(_unb64(tok["id_token"].split(".")[1]))
        except Exception:
            return fail("id_token")
        verified = claims.get("email_verified") in (True, "true")
        email = str(claims.get("email") or "").strip().lower()
        if (claims.get("iss") not in GOOGLE_ISS or claims.get("aud") != GOOGLE_CLIENT_ID
                or not isinstance(claims.get("exp"), (int, float)) or claims["exp"] < time.time() - 60
                or not hmac.compare_digest(str(claims.get("nonce") or ""), flow["n"])
                or not verified or not email or not claims.get("sub")):
            return fail("claims")
        session = self.start_session(self.user_by_google(str(claims["sub"]), email))
        return self.redirect(h, PUBLIC_URL + back, clear + session)

    # ---------- email link ----------
    def email_start(self, h):
        if not self.email_on():
            return self.reply(h, 503, {"error": "email login not configured"})
        body = self.read_json(h, 4096)
        email = str(body.get("email") or "").strip().lower()
        if len(email) > 254 or not EMAIL_RE.fullmatch(email):
            return self.reply(h, 400, {"ok": False, "error": "email"})
        lang = body.get("lang") if body.get("lang") in MAIL_TEXT else "de"
        back = safe_return(body.get("return"))
        # Over the limit: answer as usual but send nothing (no hint about the address either way).
        if not self.rate.allow("ip:" + self.client_ip(h), *RATE_IP) or not self.rate.allow("mail:" + _hash(email), *RATE_EMAIL):
            self.log("auth: login mail rate-limited")
            return self.reply(h, 200, {"ok": True})
        token, now = secrets.token_urlsafe(32), int(time.time())
        with self.lock:
            self.db.execute("DELETE FROM login_tokens WHERE expires < ?", (now - 86400,))
            self.db.execute("INSERT INTO login_tokens VALUES (?, ?, ?, 0)", (_hash(token), email, now + LOGIN_TOKEN_S))
        link = f"{API_URL}/api/auth/email/verify?" + urllib.parse.urlencode({"token": token, "return": back, "lang": lang})
        threading.Thread(target=self.send_mail, args=(email, lang, link), daemon=True).start()  # answer at once
        return self.reply(h, 200, {"ok": True})

    def send_mail(self, to, lang, link):
        subject, text = MAIL_TEXT[lang]
        if not SMTP_HOST:  # only reachable with RAINY_DEV=1 (email_on), never in production
            self.log(f"auth: DEV login link (not mailed): {link}")
            return
        msg = EmailMessage()
        msg["Subject"], msg["From"], msg["To"] = subject, MAIL_FROM, to
        msg["Date"], msg["Message-ID"] = formatdate(localtime=False), make_msgid(domain=_origin(PUBLIC_URL).split("//")[1])
        msg.set_content(text.format(link=link))
        try:
            if SMTP_PORT == 465:
                with smtplib.SMTP_SSL(SMTP_HOST, SMTP_PORT, context=ssl.create_default_context(), timeout=30) as s:
                    if SMTP_USER:
                        s.login(SMTP_USER, SMTP_PASS)
                    s.send_message(msg)
            else:  # e.g. 587: STARTTLS
                with smtplib.SMTP(SMTP_HOST, SMTP_PORT, timeout=30) as s:
                    s.starttls(context=ssl.create_default_context())
                    if SMTP_USER:
                        s.login(SMTP_USER, SMTP_PASS)
                    s.send_message(msg)
        except Exception as e:  # type and SMTP code only: the message text may contain the address
            self.log(f"auth: login mail failed ({type(e).__name__} {getattr(e, 'smtp_code', '')})")

    def token_email(self, token, use=False):
        """Email of a valid, unused login token; use=True uses it up (single use: only one request can win)."""
        if not 20 <= len(token or "") <= 100:
            return None
        th, now = _hash(token), int(time.time())
        with self.lock:
            row = self.db.execute("SELECT email FROM login_tokens WHERE token_hash = ? AND used = 0 AND expires > ?",
                                  (th, now)).fetchone()
            if row and use and self.db.execute("UPDATE login_tokens SET used = 1 WHERE token_hash = ? AND used = 0",
                                               (th,)).rowcount != 1:
                row = None
        return row[0] if row else None

    def email_verify_page(self, h, q):
        """The mailed link only shows a button: mail scanners and link previews open links as well and must not use
        up the token. The button POSTs it to this server (same origin) to sign in."""
        back = safe_return(q.get("return"))
        lang = q.get("lang") if q.get("lang") in PAGE_TEXT else "de"
        text, token = PAGE_TEXT[lang], q.get("token") or ""
        e = lambda v: html.escape(str(v), quote=True)
        valid = bool(self.token_email(token))
        if valid:
            body =(f'<h1>{e(text["title"])}</h1><p>{e(text["text"])}</p>'
                    '<form method="post" action="/api/auth/email/verify">'
                    f'<input type="hidden" name="token" value="{e(token)}">'
                    f'<input type="hidden" name="return" value="{e(back)}">'
                    f'<button type="submit">{e(text["button"])}</button></form>')
        else:
            body = (f'<h1>{e(text["expiredTitle"])}</h1><p>{e(text["expired"])}</p>'
                    f'<a class="btn" href="{e(PUBLIC_URL + back)}">{e(text["back"])}</a>')
        page = (f'<!doctype html><html lang="{lang}"><head><meta charset="utf-8">'
                '<meta name="viewport" content="width=device-width, initial-scale=1">'
                '<meta name="robots" content="noindex, nofollow"><meta name="referrer" content="same-origin">'
                f'<title>{e(text["title" if valid else "expiredTitle"])} – rainy</title><style>{PAGE_CSS}</style></head>'
                f'<body><main><div class="brand">rainy</div>{body}</main></body></html>').encode()
        h.send_response(200)
        h.send_header("Content-Type", "text/html; charset=utf-8")
        h.send_header("Cache-Control", "no-store")
        h.send_header("X-Robots-Tag", "noindex, nofollow")
        h.send_header("Referrer-Policy", "same-origin")  # (Caddy overrides it; the meta tag above still applies)
        # form-action also covers the redirect after the POST (to the app; another port in local development)
        h.send_header("Content-Security-Policy", f"default-src 'none'; style-src 'unsafe-inline'; form-action 'self' "
                                                 f"{self.public_origin}; base-uri 'none'; frame-ancestors 'none'")
        h.send_header("Content-Length", str(len(page)))
        h.end_headers()
        h.wfile.write(page)

    def email_verify(self, h):
        """POST of the confirmation page: uses up the token and signs in."""
        n = int(h.headers.get("Content-Length") or 0)
        if n > 4096:
            raise ValueError("too large")
        form = {k: v[0] for k, v in urllib.parse.parse_qs(h.rfile.read(n).decode("utf-8", "replace")).items()}
        back = safe_return(form.get("return"))
        email = self.token_email(form.get("token"), use=True)
        if not email:
            return self.redirect(h, PUBLIC_URL + with_param(back, "login", "expired"))
        return self.redirect(h, PUBLIC_URL + back, self.start_session(self.user_by_email(email)))
