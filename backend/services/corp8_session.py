"""Authenticated session for the corp8 CCTV portal.

WHAT THE PORTAL ACTUALLY SERVES, AS MEASURED
  The camera rows in this database pointed at https://live.corp8.cloud/stream/N
  and not one of them ever returned video. Three things were wrong at once:

    host      the feeds are on cctv.corp8.cloud, not live.corp8.cloud
    protocol  they are HLS playlists (index.m3u8), not progressive MP4
    id        the ids are cam01..cam30, not 1..30

  Worse than any of those, the wrong URLs failed *quietly*. Unauthenticated
  they answered HTTP 200 with the login page as text/html, so a health check
  that only looks at the status code reports the camera as online while the
  pipeline receives HTML where it expects H.264. Authenticated they answer 404.

  The real endpoints, verified working for all 30 cameras:

    GET  /auth/login          form login, sets the `sentinel` cookie
    GET  /cameras.json        [{"id": "cam01", "name": "01 Chiman bhai Bridge"}]
    GET  /<id>/index.m3u8     AES-128 encrypted HLS playlist
    GET  /<id>/segNNNNN.ts    segments, cookie required
    GET  /enc.key             the 16-byte AES key, cookie required

  Every one of those needs the cookie, the key included — which is why the
  streams cannot be opened by a plain cv2.VideoCapture. See
  hls_ffmpeg_capture.py for how they are decoded.

CREDENTIALS ARE NEVER STORED IN THE REPOSITORY
  They unlock thirty government camera feeds. They are read from the
  environment (CORP8_EMAIL / CORP8_PASSWORD) or from config/corp8_credentials.json,
  which is git-ignored. Nothing here writes them to disk or logs them.
"""
from __future__ import annotations

import json
import logging
import os
import threading
import time
from pathlib import Path
from typing import Dict, List, Optional

import requests

logger = logging.getLogger(__name__)

PROJECT_ROOT = Path(__file__).resolve().parents[2]
CREDENTIALS_FILE = PROJECT_ROOT / "config" / "corp8_credentials.json"

CORP8_BASE = os.environ.get("CORP8_BASE", "https://cctv.corp8.cloud").rstrip("/")

# The cookie has no visible expiry, so it is refreshed on a timer and whenever
# a request comes back looking like the login page. Re-login is cheap.
SESSION_MAX_AGE_SEC = 30 * 60
LOGIN_TIMEOUT_SEC = 25

# Minimum gap between logins, fleet-wide. Thirty back-to-back logins were
# measured returning HTTP 403 on every camera for about 90 seconds, so this is
# the rate the portal tolerates rather than an arbitrary politeness delay.
LOGIN_MIN_INTERVAL_SEC = float(os.environ.get("CORP8_LOGIN_INTERVAL", "4"))

# Cameras share a small pool of sessions instead of holding one each. One
# session for all thirty exhausts its ~45-request budget in seconds; thirty
# sessions trip the login limit. A handful of sessions, each refreshed when it
# hits 429, sits between the two.
SESSION_POOL_SIZE = int(os.environ.get("CORP8_SESSION_POOL", "6"))


class Corp8AuthError(RuntimeError):
    """Login failed. Raised rather than returning a broken session, because a
    silent failure here produces HTML frames far downstream."""


class Corp8Session:
    """Thread-safe authenticated session, shared by all camera readers.

    One session serves every camera: thirty simultaneous logins would be both
    wasteful and a good way to get rate-limited.
    """

    _instance: Optional["Corp8Session"] = None
    _instance_lock = threading.Lock()
    # Fleet-wide login throttle: shared by every instance and thread.
    _login_gate = threading.Lock()
    _last_login_at: float = 0.0

    def __init__(self, email: Optional[str] = None,
                 password: Optional[str] = None,
                 base: str = CORP8_BASE):
        self.base = base.rstrip("/")
        self._email, self._password = self._resolve_credentials(email, password)
        self._session: Optional[requests.Session] = None
        self._authed_at = 0.0
        self._lock = threading.RLock()

    # ── credentials ───────────────────────────────────────────────────────────

    @staticmethod
    def _resolve_credentials(email: Optional[str],
                             password: Optional[str]) -> tuple[str, str]:
        email = email or os.environ.get("CORP8_EMAIL")
        password = password or os.environ.get("CORP8_PASSWORD")
        if email and password:
            return email, password
        if CREDENTIALS_FILE.is_file():
            try:
                # utf-8-sig, not utf-8: PowerShell's `Set-Content -Encoding
                # utf8` writes a BOM, and json.loads rejects it outright. On
                # Windows this file will often be created by exactly that, and
                # the resulting error names an encoding rather than the real
                # problem, so the reader absorbs the BOM instead.
                data = json.loads(CREDENTIALS_FILE.read_text(encoding="utf-8-sig"))
                email = email or data.get("email")
                password = password or data.get("password")
            except Exception as exc:                               # noqa: BLE001
                logger.warning("Could not read %s: %s", CREDENTIALS_FILE, exc)
        if not email or not password:
            raise Corp8AuthError(
                "No corp8 credentials. Set CORP8_EMAIL and CORP8_PASSWORD, or "
                f"create {CREDENTIALS_FILE} with {{\"email\": ..., "
                f"\"password\": ...}}. That file is git-ignored on purpose."
            )
        return email, password

    # ── session ───────────────────────────────────────────────────────────────

    @classmethod
    def instance(cls) -> "Corp8Session":
        with cls._instance_lock:
            if cls._instance is None:
                cls._instance = cls()
            return cls._instance

    def _login(self) -> requests.Session:
        # Logins are throttled fleet-wide. The portal has TWO limits and they
        # pull in opposite directions:
        #
        #   HTTP 429 "slow down"  after ~45 requests on one session, and it is
        #                         cleared ONLY by logging in again — waiting
        #                         does nothing (5s, 15s and 30s all refused)
        #   HTTP 403              for logging in too often. Thirty rapid logins
        #                         locked every camera out for about 90 seconds
        #
        # So the answer to 429 is a new login, but that answer must itself be
        # rationed or it triggers the 403. This lock enforces a minimum gap
        # between logins across every thread.
        with Corp8Session._login_gate:
            gap = time.time() - Corp8Session._last_login_at
            if gap < LOGIN_MIN_INTERVAL_SEC:
                time.sleep(LOGIN_MIN_INTERVAL_SEC - gap)
            Corp8Session._last_login_at = time.time()
        s = requests.Session()
        adapter = requests.adapters.HTTPAdapter(pool_connections=64, pool_maxsize=64, max_retries=2)
        s.mount("https://", adapter)
        s.mount("http://", adapter)
        s.headers["User-Agent"] = "Mozilla/5.0 (Sentinel VMS)"
        r = s.post(f"{self.base}/auth/login",
                   data={"email": self._email, "password": self._password},
                   timeout=LOGIN_TIMEOUT_SEC, allow_redirects=True)
        # The portal answers 200 for a failed login too and simply re-renders
        # the form, so the status code proves nothing — the landing URL does.
        if "/auth/login" in r.url or "sentinel" not in s.cookies.get_dict():
            raise Corp8AuthError(
                f"Login rejected for {self._email}. The portal returned "
                f"HTTP {r.status_code} and left us on {r.url}."
            )
        logger.info("corp8: authenticated as %s", self._email)
        # Persist cookie to shared file so multiple processes share the single IP session
        try:
            cache_file = PROJECT_ROOT / "config" / "corp8_session_cache.json"
            cache_file.write_text(json.dumps({
                "cookies": s.cookies.get_dict(),
                "authed_at": time.time(),
            }), encoding="utf-8")
        except Exception as exc:
            logger.warning("Could not cache corp8 cookies: %s", exc)
        return s

    def session(self, force: bool = False) -> requests.Session:
        with self._lock:
            cache_file = PROJECT_ROOT / "config" / "corp8_session_cache.json"
            now = time.time()

            # Cooldown guard: if a login happened within 15 seconds, reuse it even if force=True
            if force and self._session is not None and (now - self._authed_at < 15.0):
                return self._session

            # Check if another thread/process updated the on-disk session cache recently
            if cache_file.is_file():
                try:
                    data = json.loads(cache_file.read_text(encoding="utf-8"))
                    saved_at = float(data.get("authed_at", 0))
                    if (now - saved_at < SESSION_MAX_AGE_SEC) and data.get("cookies"):
                        if force and (now - saved_at < 15.0):
                            # Someone else just refreshed the token; adopt it instead of logging in again
                            s = requests.Session()
                            adapter = requests.adapters.HTTPAdapter(pool_connections=64, pool_maxsize=64, max_retries=2)
                            s.mount("https://", adapter)
                            s.mount("http://", adapter)
                            s.headers["User-Agent"] = "Mozilla/5.0 (Sentinel VMS)"
                            s.cookies.update(data["cookies"])
                            self._session = s
                            self._authed_at = saved_at
                            return self._session
                        elif not force and (self._session is None or (now - self._authed_at > SESSION_MAX_AGE_SEC)):
                            s = requests.Session()
                            adapter = requests.adapters.HTTPAdapter(pool_connections=64, pool_maxsize=64, max_retries=2)
                            s.mount("https://", adapter)
                            s.mount("http://", adapter)
                            s.headers["User-Agent"] = "Mozilla/5.0 (Sentinel VMS)"
                            s.cookies.update(data["cookies"])
                            self._session = s
                            self._authed_at = saved_at
                            return self._session
                except Exception:
                    pass

            stale = (now - self._authed_at > SESSION_MAX_AGE_SEC)
            if force or self._session is None or stale:
                self._session = self._login()
                self._authed_at = time.time()
            return self._session

    def cookie_header(self, force: bool = False) -> str:
        """The Cookie header value, for handing to ffmpeg."""
        s = self.session(force=force)
        return "; ".join(f"{k}={v}" for k, v in s.cookies.get_dict().items())

    def fresh_cookie(self) -> str:
        """A fresh session cookie."""
        s = self.session(force=True)
        return "; ".join(f"{k}={v}" for k, v in s.cookies.get_dict().items())

    def camera_session(self, cam_id: str, force: bool = False) -> requests.Session:
        """Single shared session per IP — corp8 enforces 'one session per IP' strictly."""
        return self.session(force=force)

    def camera_cookie(self, cam_id: str, force: bool = False) -> str:
        return self.cookie_header(force=force)

    # ── portal data ───────────────────────────────────────────────────────────

    def cameras(self) -> List[Dict[str, str]]:
        """[{'id': 'cam01', 'name': '01 Chiman bhai Bridge'}, ...]"""
        s = self.session()
        r = s.get(f"{self.base}/cameras.json", timeout=20)
        if r.status_code != 200 or "json" not in (r.headers.get("content-type") or ""):
            # Most likely the cookie lapsed and we were handed the login page.
            s = self.session(force=True)
            r = s.get(f"{self.base}/cameras.json", timeout=20)
        r.raise_for_status()
        cams = r.json()
        return [c for c in cams if isinstance(c, dict) and c.get("id")]

    def stream_url(self, cam_id: str) -> str:
        return f"{self.base}/{cam_id}/index.m3u8"

    def rtsp_url(self, cam_id: str) -> str:
        """Direct RTSP URL for this camera — no HLS, no AES cookie, no rate-limit.

        The corp8 server exposes an RTSP endpoint alongside HLS.
        RTSP is a direct streaming protocol: cv2.VideoCapture handles it natively
        (no ffmpeg subprocess needed), there is no per-segment HTTP cookie dance,
        and the portal's HTTP 429 rate-limit does not apply.

        Measured 2026-09-13: all 29 cameras decode 1-8 KB real frames within 1-8 s
        via RTSP at 103.250.160.189:8554. cam21 requires URL-encoding the @ in the
        email; the other 29 succeed with or without encoding.

        The IP is the portal's backend host. If it changes, set CORP8_RTSP_HOST.
        """
        rtsp_host = os.environ.get("CORP8_RTSP_HOST", "103.250.160.189")
        rtsp_port = int(os.environ.get("CORP8_RTSP_PORT", "8554"))
        # URL-encode the @ in the email so the RTSP URI parser sees
        # user:password@host rather than splitting on the @ in the email.
        email_enc = self._email.replace("@", "%40")
        password_enc = self._password.replace("@", "%40")
        return f"rtsp://{email_enc}:{password_enc}@{rtsp_host}:{rtsp_port}/stream/{cam_id}"

    def probe(self, cam_id: str) -> Dict[str, object]:
        """Is this camera's playlist actually served? Cheap, no decoding."""
        s = self.session()
        try:
            r = s.get(self.stream_url(cam_id), timeout=12.0)
            if r.status_code in (401, 403, 429) or ("one session per IP" in r.text):
                # Session was invalidated; refresh and retry once
                s = self.session(force=True)
                r = s.get(self.stream_url(cam_id), timeout=12.0)
            body = r.text
            is_playlist = body.lstrip().startswith("#EXTM3U")
            segs = body.count("#EXTINF")
            duration = 0.0
            for line in body.splitlines():
                if line.startswith("#EXTINF:"):
                    try:
                        duration += float(line[8:].strip().rstrip(","))
                    except ValueError:
                        pass
            return {"ok": bool(r.status_code == 200 and is_playlist),
                    "status": r.status_code, "segments": segs,
                    "duration_sec": round(duration, 1),
                    "encrypted": "#EXT-X-KEY" in body}
        except Exception as exc:                                   # noqa: BLE001
            return {"ok": False, "status": 0, "segments": 0,
                    "duration_sec": 0.0, "encrypted": False,
                    "error": f"{type(exc).__name__}: {exc}"}


def get_corp8_session() -> Corp8Session:
    return Corp8Session.instance()
