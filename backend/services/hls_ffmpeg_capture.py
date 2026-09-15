"""Decode authenticated, AES-encrypted HLS. A drop-in for cv2.VideoCapture.

WHY OPENCV CANNOT DO THIS
  cv2.VideoCapture fails on these streams, and it fails silently — isOpened()
  simply returns False with no diagnostic. Measured, the reasons are:

    the playlist is AES-128 encrypted
        #EXT-X-KEY:METHOD=AES-128,URI="/enc.key"
    every request needs the portal's session cookie, INCLUDING the key fetch
    OpenCV's bundled FFmpeg does not forward the headers it is given to that
        key request, so decryption never starts

  Five spellings of OPENCV_FFMPEG_CAPTURE_OPTIONS were tried (headers, cookies,
  protocol_whitelist, with and without a User-Agent); all returned
  isOpened()=False. The standalone ffmpeg binary handles the same URL with the
  same cookie and produced ten real frames in 0.8 s, so the decode path here is
  a subprocess rather than a library call.

WHY THE SEEK
  The playlists are #EXT-X-PLAYLIST-TYPE:VOD with an #EXT-X-ENDLIST — recordings
  of about twelve hours, not live edges. The portal's own player fakes "live" by
  seeking to (now mod duration) and looping, and this does the same, so a frame
  pulled at 14:32 is the footage from 14:32 rather than from the start of the
  recording. Without that every camera would show the same small hours every
  time the pipeline restarted.

INTERFACE
  isOpened / read / grab / get / release, matching the parts of
  cv2.VideoCapture that CameraReaderThread actually calls, so it can be
  substituted without touching the reader's loop.
"""
from __future__ import annotations

import logging
import os
import shutil
import subprocess
import threading
import time
from pathlib import Path
from typing import Optional, Tuple

import cv2
import numpy as np

logger = logging.getLogger(__name__)

FFMPEG_BIN = os.environ.get("SENTINEL_FFMPEG", "ffmpeg")

# Decode at this size. The plate crops are cut from whatever this produces, so
# it is deliberately the native 720p of these cameras rather than something
# smaller — downscaling here would throw away plate pixels that cannot be
# recovered later.
OUT_W, OUT_H = 1280, 720

# Frames per second DECODED from each camera. This is a `-vf fps=` filter on
# frames already downloaded, not a fetch rate: segment fetching is paced by -re
# regardless (see _start), so raising this costs GPU, never portal goodwill.
#
# Two tiers. Every camera decodes at OUT_FPS. A named subset — the cameras a
# demonstration actually watches closely — decode at OUT_FPS_HIGH instead, so
# they reach clip-density frames (and read plates / measure speed as well as the
# clip run did) without paying that cost across all thirty and saturating the
# GPU. Measured on an RTX 4070: 30x5 fps hits 100% GPU and ~8% frame drop, while
# 30x1 sits at 11%; the tiers let the demo cameras have 5 and the wall have 2.
OUT_FPS = float(os.environ.get("SENTINEL_HLS_FPS", "1"))
OUT_FPS_HIGH = float(os.environ.get("SENTINEL_HLS_FPS_HIGH", "5"))
HIGH_FPS_CAMERAS = [c.strip() for c in
                    os.environ.get("SENTINEL_HLS_HIGH_CAMERAS", "").split(",")
                    if c.strip()]


def _hls_fps_for(cam_hint: Optional[str]) -> float:
    """Decode fps for one camera: OUT_FPS_HIGH if it is in HIGH_FPS_CAMERAS,
    else OUT_FPS. Matches on the id loosely — "cam09", "CAM_09" and "9" all name
    the same camera — because the id arrives here from a URL path segment while
    the allow-list is typed by an operator."""
    if not cam_hint or not HIGH_FPS_CAMERAS:
        return OUT_FPS
    hint = cam_hint.lower()
    hint_digits = "".join(c for c in hint if c.isdigit())
    for name in HIGH_FPS_CAMERAS:
        n = name.lower()
        if n == hint or (hint_digits and
                         "".join(c for c in n if c.isdigit()) == hint_digits):
            return OUT_FPS_HIGH
    return OUT_FPS

READ_TIMEOUT_SEC = 40.0

# How many times one capture will re-authenticate and relaunch before giving
# up. The portal's rate limit is cleared by a new login, so a restart is the
# correct response to it — but an unbounded loop would hide a genuinely dead
# camera behind endless reconnects.
MAX_RESTARTS = int(os.environ.get("SENTINEL_HLS_MAX_RESTARTS", "20"))


def ffmpeg_available() -> bool:
    return shutil.which(FFMPEG_BIN) is not None


class FFmpegHLSCapture:
    """One ffmpeg subprocess per camera, piping raw BGR frames."""

    def __init__(self, url: str, cookie: str,
                 width: int = OUT_W, height: int = OUT_H,
                 fps: float = OUT_FPS, seek_sec: Optional[float] = None,
                 duration_sec: Optional[float] = None):
        self.url = url
        self.cookie = cookie
        self.width = int(width)
        self.height = int(height)
        self.fps = float(fps)
        self.duration_sec = duration_sec
        self._frame_bytes = self.width * self.height * 3
        self._proc: Optional[subprocess.Popen] = None
        self._opened = False
        self._lock = threading.Lock()
        self._frames_read = 0
        self._restarts = 0
        self._stderr_tail = ""

        # Wall-clock position in the loop, so restarts resume "now" rather than
        # replaying the beginning.
        if seek_sec is None and duration_sec and duration_sec > 1:
            seek_sec = (time.time() % duration_sec)
        self.seek_sec = float(seek_sec or 0.0)
        self._start()

    # ── lifecycle ─────────────────────────────────────────────────────────────

    def _start(self) -> None:
        if not ffmpeg_available():
            logger.error("ffmpeg not found on PATH (%s); cannot decode %s",
                         FFMPEG_BIN, self.url)
            self._opened = False
            return
        # A prefetched playlist on disk is a different job from a live pull.
        #
        # Streaming this portal delivers ~1.8 fps, measured with and without -re
        # and with the raw pipe swapped for MJPEG — it is the portal, not us. But
        # the playlists are recordings, so a window can be fetched ahead by
        # backend/scripts/prefetch_live_buffer.py and decoded from disk instead:
        # measured 50 frames in 0.2 s, about 250 fps, against 1.8 fps remote.
        #
        # Two flags are required and neither is optional. ffmpeg refuses to read
        # a LOCAL encrypted playlist without them — "Filename extension of
        # 'enc.key' is not a common multimedia extension, blocked for security
        # reasons" — because the key is not a media file and AES-128 HLS needs
        # the crypto protocol. Remote playlists get a different default
        # whitelist, which is why this only bites on the local path.
        local = not str(self.url).lower().startswith(("http://", "https://"))
        if local:
            cmd = [
                FFMPEG_BIN,
                "-hide_banner", "-loglevel", "error",
                "-nostdin",
                "-allowed_extensions", "ALL",
                "-protocol_whitelist", "file,crypto,data",
                # -re is mandatory here, not a politeness setting: decoding at
                # 250 fps would burn a two-minute buffer in under a second.
                "-re",
            ]
            if self.seek_sec > 1:
                cmd += ["-ss", f"{self.seek_sec:.2f}"]
            cmd += [
                "-i", self.url,
                "-an", "-sn",
                "-vf", f"fps={self.fps},scale={self.width}:{self.height}",
                "-f", "rawvideo", "-pix_fmt", "bgr24",
                "pipe:1",
            ]
            try:
                self._proc = subprocess.Popen(
                    cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE,
                    bufsize=self._frame_bytes * 2,
                )
                self._opened = True
            except Exception as exc:                               # noqa: BLE001
                logger.error("Failed to launch ffmpeg for %s: %s", self.url, exc)
                self._opened = False
            return

        if _is_loopback_relay(self.url):
            # The local relay (backend/scripts/cam09_live_relay.py): one plain
            # MPEG-TS stream, already decrypted, fed no faster than real time.
            # Measured: the relay's earlier sliding HLS playlist decoded at
            # 0.3x real time here, because ffmpeg's live-HLS reader waits a
            # target duration between reloads; a TS stream has no playlist.
            #
            # No cookie, and deliberately no -reconnect: the relay closes the
            # connection when the recording loops, and a clean exit plus a
            # fresh open at the new live point is right, where an in-stream
            # reconnect would hand the demuxer a backwards timestamp jump.
            cmd = [
                FFMPEG_BIN, "-hide_banner", "-loglevel", "error", "-nostdin",
                "-re", "-i", self.url,
                "-an", "-sn",
                "-vf", f"fps={self.fps},scale={self.width}:{self.height}",
                "-f", "rawvideo", "-pix_fmt", "bgr24",
                "pipe:1",
            ]
            try:
                self._proc = subprocess.Popen(
                    cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE,
                    bufsize=self._frame_bytes * 2,
                )
                self._opened = True
            except Exception as exc:                               # noqa: BLE001
                logger.error("Failed to launch ffmpeg for %s: %s", self.url, exc)
                self._opened = False
            return

        cmd = [
            FFMPEG_BIN,
            "-hide_banner", "-loglevel", "error",
            "-nostdin",
            # Applies to the playlist, the segments AND the AES key request,
            # which is the part OpenCV gets wrong.
            "-headers", f"Cookie: {self.cookie}\r\nUser-Agent: Mozilla/5.0\r\n",
            "-reconnect", "1",
            "-reconnect_streamed", "1",
            "-reconnect_delay_max", "5",
            # Read at the media's own rate instead of as fast as the network
            # allows. This is what keeps the portal from rate-limiting us.
            #
            # These playlists are twelve-hour VOD recordings, and without -re
            # ffmpeg downloads them flat out — thousands of segment requests in
            # seconds. Measured, that is exactly what triggered HTTP 429 "slow
            # down", and answering the 429 with fresh logins then triggered
            # HTTP 403 for logging in too often. With -re each camera asks for
            # roughly one 6-second segment every 6 seconds, which the portal
            # serves indefinitely: thirty playlists fetched on a single session
            # at 1.5-second spacing all succeeded, where thirty fired back to
            # back did not.
        ]
        if os.environ.get("SENTINEL_HLS_NO_RE", "0") != "1":
            cmd.append("-re")
        # SENTINEL_HLS_NO_RE=1 drops the pacing above. Safe ONLY for a small
        # number of cameras, and it exists because -re is also a ceiling: asked
        # for 5 fps, a single camera measured 4.3 and swung between 2.5 and 4.25
        # once the pipeline was consuming it, because -re never lets ffmpeg get
        # ahead of a late segment. Without it ffmpeg does not run away either —
        # stdout is a two-frame pipe, so it blocks as soon as the reader stops
        # taking frames, and the reader's own throttle sets the rate. The
        # runaway the -re comment describes is what happens when nothing is
        # reading the pipe; here something always is.
        if self.seek_sec > 1:
            # Before -i: seeks by keyframe, which is fast and accurate enough
            # for a wall-clock position.
            cmd += ["-ss", f"{self.seek_sec:.2f}"]
        cmd += [
            "-i", self.url,
            "-an", "-sn",
            "-vf", f"fps={self.fps},scale={self.width}:{self.height}",
            "-f", "rawvideo", "-pix_fmt", "bgr24",
            "pipe:1",
        ]
        try:
            self._proc = subprocess.Popen(
                cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE,
                bufsize=self._frame_bytes * 2,
            )
            self._opened = True
        except Exception as exc:                                   # noqa: BLE001
            logger.error("Failed to launch ffmpeg for %s: %s", self.url, exc)
            self._opened = False

    def isOpened(self) -> bool:                                    # noqa: N802
        return bool(self._opened and self._proc and self._proc.poll() is None)

    def release(self) -> None:
        with self._lock:
            self._opened = False
            if self._proc:
                try:
                    self._proc.kill()
                except Exception:                                  # noqa: BLE001
                    pass
                try:
                    # Draining avoids leaving a zombie holding the pipe.
                    self._proc.communicate(timeout=5)
                except Exception:                                  # noqa: BLE001
                    pass
                self._proc = None

    # ── reading ───────────────────────────────────────────────────────────────

    def _read_exact(self, n: int) -> Optional[bytes]:
        """Read exactly n bytes; a short read means the stream ended."""
        assert self._proc and self._proc.stdout
        buf = bytearray()
        deadline = time.time() + READ_TIMEOUT_SEC
        while len(buf) < n:
            if time.time() > deadline:
                return None
            chunk = self._proc.stdout.read(n - len(buf))
            if not chunk:
                return None
            buf.extend(chunk)
        return bytes(buf)

    def read(self) -> Tuple[bool, Optional[np.ndarray]]:
        if not self.isOpened():
            # ffmpeg has exited. Almost always that is the portal's rate limit
            # rather than a dead camera, so try once to come back rather than
            # letting the reader fall over to local clips.
            if self._restarts < MAX_RESTARTS and self._relogin_and_restart():
                pass
            else:
                return False, None
        with self._lock:
            raw = self._read_exact(self._frame_bytes)
        if raw is None:
            self._capture_stderr()
            if self._restarts < MAX_RESTARTS and self._relogin_and_restart():
                with self._lock:
                    raw = self._read_exact(self._frame_bytes)
            if raw is None:
                return False, None
        self._frames_read += 1
        frame = np.frombuffer(raw, np.uint8).reshape(self.height, self.width, 3)
        # frombuffer gives a read-only view onto the pipe buffer; downstream
        # code draws on frames, so hand back a writable copy.
        return True, frame.copy()

    def _relogin_and_restart(self) -> bool:
        """Mint a fresh cookie and relaunch ffmpeg from the current position.

        The portal answers HTTP 429 "slow down" once a session has made about
        45 requests, and that does not expire with time — only a new login
        clears it. So the recovery is to re-authenticate, not to back off.
        """
        self._restarts += 1
        # A local source — the relay, or a prefetched playlist on disk — is not
        # behind the portal's rate limit, so it gets a plain relaunch. Forcing
        # a portal login here spent one login per hiccup, against a lockout
        # that triggers on login frequency, and could invalidate the session
        # the relay itself is streaming on.
        u = str(self.url).lower()
        local = (not u.startswith(("http://", "https://"))) or _is_loopback_relay(self.url)
        if not local:
            try:
                from backend.services.corp8_session import get_corp8_session
                cam = self.url.rstrip("/").split("/")[-2]
                cookie = get_corp8_session().camera_cookie(cam, force=True)
            except Exception as exc:                               # noqa: BLE001
                logger.warning("corp8 re-login failed for %s: %s", self.url, exc)
                return False
            self.cookie = cookie
        # Resume where the wall clock now is, so a restart does not replay.
        if self.duration_sec and self.duration_sec > 1:
            self.seek_sec = time.time() % self.duration_sec
        self.release()
        self._opened = False
        self._start()
        if self.isOpened():
            logger.info("%s: %s and resumed (restart %d)", self.url,
                        "relaunched (no portal login)" if local else "re-authenticated",
                        self._restarts)
        return self.isOpened()

    def grab(self) -> bool:
        """Discard one frame. The reader uses this to skip without decoding."""
        if not self.isOpened():
            return False
        with self._lock:
            return self._read_exact(self._frame_bytes) is not None

    def _capture_stderr(self) -> None:
        if not self._proc or not self._proc.stderr:
            return
        try:
            self._proc.stderr.flush()
            data = self._proc.stderr.read() or b""
            if data:
                self._stderr_tail = data.decode(errors="replace")[-400:]
                logger.warning("ffmpeg[%s]: %s", self.url, self._stderr_tail)
        except Exception:                                          # noqa: BLE001
            pass

    # ── cv2.VideoCapture compatibility ────────────────────────────────────────

    def get(self, prop: int) -> float:
        if prop == cv2.CAP_PROP_FRAME_WIDTH:
            return float(self.width)
        if prop == cv2.CAP_PROP_FRAME_HEIGHT:
            return float(self.height)
        if prop == cv2.CAP_PROP_FPS:
            return float(self.fps)
        if prop == cv2.CAP_PROP_FRAME_COUNT:
            return float((self.duration_sec or 0.0) * self.fps)
        return 0.0

    def set(self, prop: int, value: float) -> bool:                # noqa: A003
        return False


def _is_loopback_relay(url: Optional[str]) -> bool:
    return str(url or "").lower().startswith(("http://127.0.0.1", "http://localhost"))


def _local_buffer_for(url: str) -> Optional[str]:
    """A prefetched local playlist for this camera, if one has been built.

    SENTINEL_HLS_LOCAL_BUFFER may be either a playlist file (single camera) or a
    directory holding <CAM_ID>/index.m3u8. Built by
    backend/scripts/prefetch_live_buffer.py; see _start for why decoding from
    disk is ~250 fps where the same footage streams at ~1.8.
    """
    raw = os.environ.get("SENTINEL_HLS_LOCAL_BUFFER", "").strip()
    if not raw:
        return None
    # A local relay serving the camera as one continuous MPEG-TS stream
    # (backend/scripts/cam09_live_relay.py). Loopback only: this setting exists
    # to keep decoding off the portal, and a remote URL here would quietly put
    # it back on the network.
    if _is_loopback_relay(raw):
        return raw
    p = Path(raw)
    if p.is_file():
        return str(p)
    if p.is_dir():
        # cam09/index.m3u8 -> CAM_09
        seg = url.rstrip("/").split("/")[-2] if "/index.m3u8" in url else ""
        digits = "".join(c for c in seg if c.isdigit())
        for cand in (seg.upper(), f"CAM_{digits}", f"CAM_{digits.zfill(2)}"):
            f = p / cand / "index.m3u8"
            if f.is_file():
                return str(f)
    logger.warning("SENTINEL_HLS_LOCAL_BUFFER=%s has no playlist for %s; "
                   "falling back to the live stream", raw, url)
    return None


def open_corp8_stream(url: str) -> Optional[FFmpegHLSCapture]:
    """Open a corp8 HLS URL with a fresh authenticated cookie, or None."""
    from backend.services.corp8_session import get_corp8_session

    # Prefetched buffer wins when there is one: same footage, same camera, but
    # decoded off disk instead of pulled through the portal one segment at a
    # time. No cookie is needed because nothing it reads is remote.
    buffered = _local_buffer_for(url)
    if buffered:
        fps = _hls_fps_for(url.rstrip("/").split("/")[-2]
                           if "/index.m3u8" in url else None)
        cap = FFmpegHLSCapture(buffered, cookie="", fps=fps, seek_sec=0.0)
        if cap.isOpened():
            logger.info("Using prefetched local buffer for %s: %s", url, buffered)
            return cap
        logger.warning("Local buffer %s would not open; using the live stream",
                       buffered)

    try:
        sess = get_corp8_session()
        cam_id = url.rstrip("/").split("/")[-2] if "/index.m3u8" in url else None
        duration = None
        if cam_id:
            info = sess.probe(cam_id)
            if info.get("ok"):
                duration = float(info.get("duration_sec") or 0.0)
            elif info.get("status") == 404:
                logger.warning("corp8 %s: playlist not found on portal (%s)",
                               cam_id, info)
                return None
            else:
                logger.info("corp8 %s: probe inconclusive (%s) — attempting ffmpeg direct open",
                            cam_id, info)
        # The shared session cookie across all cameras
        cookie = sess.cookie_header()
        fps = _hls_fps_for(cam_id)
        # Where in the recording to start.
        #
        # These playlists are ~12 h VOD, and seek 0 is the recording's first
        # second — which for this fleet is the previous evening. Measured on
        # CAM_09 by reading the camera's own burnt-in clock: seek 29,700 s shows
        # 05:17, so seek 0 is 21:02 and the road is dark for the first nine
        # hours. A camera pointed at an unlit bypass at night yields no readable
        # plates no matter how good the recogniser is, so the ingest has to be
        # able to start where there is traffic: seek 37,680 s is 07:30, full
        # daylight, and sustained 4.3 fps when measured.
        seek = float(os.environ.get("SENTINEL_HLS_SEEK_SEC", "0") or 0.0)
        cap = FFmpegHLSCapture(url, cookie, fps=fps,
                               duration_sec=duration, seek_sec=seek)
        return cap if cap.isOpened() else None
    except Exception as exc:                                       # noqa: BLE001
        logger.error("Could not open corp8 stream %s: %s", url, exc)
        return None


def is_corp8_hls(url: Optional[str]) -> bool:
    return bool(url) and "corp8" in url and url.endswith(".m3u8")
