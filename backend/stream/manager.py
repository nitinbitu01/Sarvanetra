# backend/stream/manager.py

import asyncio
import logging
import os
import shutil
import subprocess
import threading
from pathlib import Path
from typing import Any

from sqlalchemy import text

try:
    from backend.core.config import settings
    from backend.db.session import SessionLocal
except ImportError:
    from sentinel.config import settings
    from sentinel.db import SessionLocal

logger = logging.getLogger(__name__)


# ── State ──────────────────────────────────────────────────────────────────────
ffmpeg_processes: dict[Any, subprocess.Popen] = {}  # camera_id → Popen
_spawn_locks:     dict[Any, threading.Lock]   = {}  # camera_id → Lock
_spawn_lock_guard = threading.Lock()
_monitor_shutdown = asyncio.Event()  # set on app shutdown to break monitor loop


# ── ffmpeg version check ──────────────────────────────────────────────────────

def check_ffmpeg_version() -> None:
    """
    Verify ffmpeg is installed and log its version at startup.
    Logs CRITICAL if not found — HLS streaming will be unavailable.
    """
    try:
        result = subprocess.run(
            ["ffmpeg", "-version"],
            capture_output=True, text=True, timeout=5
        )
        first_line = result.stdout.split('\n')[0]
        logger.info(f"[HLS] ffmpeg found: {first_line}")
    except FileNotFoundError:
        logger.critical(
            "[HLS] ffmpeg NOT FOUND. HLS streaming disabled.\n"
            "Install: apt install ffmpeg  OR  brew install ffmpeg"
        )
    except subprocess.TimeoutExpired:
        logger.error("[HLS] ffmpeg version check timed out — ffmpeg may be broken")


# ── Directory helpers ──────────────────────────────────────────────────────────

def _hls_dir(camera_id: Any) -> Path:
    hls_root = getattr(settings, "HLS_DIR", "hls")
    return Path(hls_root) / str(camera_id)

def _clear_hls_dir(camera_id: Any) -> None:
    """
    Delete and recreate HLS directory for this camera.
    """
    d = _hls_dir(camera_id)
    if d.exists():
        shutil.rmtree(d, ignore_errors=True)
    d.mkdir(parents=True, exist_ok=True)

def clear_all_hls_on_startup() -> None:
    """
    Delete the entire hls/ root on app startup.
    Prevents stale segments from a crashed previous run from being served.
    """
    hls_root = getattr(settings, "HLS_DIR", "hls")
    root = Path(hls_root)
    if root.exists():
        shutil.rmtree(root, ignore_errors=True)
        logger.info("[HLS] Cleared stale HLS directory on startup")
    root.mkdir(parents=True, exist_ok=True)


# ── Spawn lock ────────────────────────────────────────────────────────────────

def _get_spawn_lock(camera_id: Any) -> threading.Lock:
    """
    Per-camera spawn lock. Prevents concurrent double-spawn race.
    """
    with _spawn_lock_guard:
        if camera_id not in _spawn_locks:
            _spawn_locks[camera_id] = threading.Lock()
        return _spawn_locks[camera_id]


# ── ffmpeg command builder ────────────────────────────────────────────────────

# Anything with one of these schemes is a live network source that ffmpeg
# opens itself. Everything else is treated as a path on disk.
#
# This distinction used to be `is_file = not is_rtsp`, which classified every
# http/https URL — i.e. EVERY camera delivered over HLS, which is how the
# government portal serves all 30 of its feeds — as a local file. Two things
# followed from that: `-stream_loop -1` was applied to a live stream (it is
# meaningless there, and tells ffmpeg to restart a source that never ends),
# and prewarm_demo_cameras() below tested the URL with Path(url).exists(),
# which is always False for a URL, so every one of those cameras was skipped
# at startup with "source file not found". The live video wall was therefore
# empty for exactly the feeds the platform exists to show.
_NETWORK_SCHEMES = (
    "rtsp://", "rtsps://", "http://", "https://",
    "rtmp://", "rtmps://", "udp://", "srt://", "onvif://",
)


def is_network_source(stream_url: Any) -> bool:
    return str(stream_url or "").lower().startswith(_NETWORK_SCHEMES)


# Startup prewarm opens EVERY camera at once. For local clips that is free.
# For network cameras it is a stampede: 30 corp8 feeds meant 30 simultaneous
# ffmpeg processes each authenticating against the same portal, which is
# precisely the burst that got this account HTTP 403'd across every camera
# once already (see backend/services/corp8_session.py's measured notes on the
# login-frequency limit). Off by default: network cameras are served by the
# snapshot path, which is session-pooled and rate-limit aware. Turn on only
# for a deployment whose sources tolerate simultaneous connections.
PREWARM_NETWORK_SOURCES = os.environ.get("SENTINEL_HLS_PREWARM_NETWORK", "0") == "1"

# A source that cannot be opened must not be retried forever. The monitor
# restarted any dead ffmpeg unconditionally, so a permanently-failing camera
# (bad credentials, unreachable host, an expired portal session) respawned on
# every 10s tick — 380 spawns for 104 cameras in one startup, all of them
# hitting the same remote endpoint.
MAX_CONSECUTIVE_RESTARTS = int(os.environ.get("SENTINEL_HLS_MAX_RESTARTS", "3"))
_restart_failures: dict[Any, int] = {}


def _build_ffmpeg_cmd(stream_url: str, camera_id: Any) -> list[str]:
    output_m3u8 = str(_hls_dir(camera_id) / "output.m3u8")
    is_rtsp     = str(stream_url).lower().startswith(("rtsp://", "rtsps://"))
    is_file     = not is_network_source(stream_url)

    cmd = ["ffmpeg", "-y"]

    if is_rtsp:
        cmd += ["-rtsp_transport", "tcp"]

    # The portal's HLS is AES-encrypted and the key fetch is cookie-gated, so
    # ffmpeg needs the session cookie on every request or it gets the login
    # page instead of a key. Same header form that hls_ffmpeg_capture.py
    # already uses for the snapshot path, reusing that module's session so
    # this does not mint a second login per camera (the portal rate-limits
    # logins — see corp8_session.py's measured notes).
    try:
        from backend.services.hls_ffmpeg_capture import is_corp8_hls
        if is_corp8_hls(str(stream_url)):
            from backend.services.corp8_session import get_corp8_session
            cookie = get_corp8_session().camera_cookie(str(camera_id))
            if cookie:
                cmd += ["-headers",
                        f"Cookie: {cookie}\r\nUser-Agent: Mozilla/5.0\r\n"]
    except Exception as exc:                      # never block a spawn on this
        logger.debug("[HLS] no corp8 cookie for %s: %s", camera_id, exc)

    # Looping only makes sense for a finite file. A live stream has no end to
    # loop back to.
    if is_file:
        cmd += ["-stream_loop", "-1"]

    segment_duration = getattr(settings, "HLS_SEGMENT_DURATION", 2)
    list_size = getattr(settings, "HLS_LIST_SIZE", 3)

    cmd += [
        "-i", str(stream_url),
        "-c:v", "libx264",
        "-preset", "veryfast",
        "-g", "48",
        "-sc_threshold", "0",
        "-hls_time",      str(segment_duration),
        "-hls_list_size", str(list_size),
        "-hls_flags",     "delete_segments+append_list",
        "-f", "hls",
        output_m3u8,
    ]
    return cmd


# ── Spawn ──────────────────────────────────────────────────────────────────────

def spawn_ffmpeg(camera_id: Any, stream_url: str) -> bool:
    """
    Spawn an ffmpeg process for camera_id. Thread-safe via per-camera lock.
    Returns True if new process spawned, False if already running or lock busy.
    """
    lock = _get_spawn_lock(camera_id)
    if not lock.acquire(blocking=False):
        return False

    try:
        existing = ffmpeg_processes.get(camera_id)
        if existing and existing.poll() is None:
            return False  # Already alive

        _clear_hls_dir(camera_id)
        cmd = _build_ffmpeg_cmd(stream_url, camera_id)
        proc = subprocess.Popen(
            cmd,
            stdout=subprocess.DEVNULL,
            stderr=subprocess.PIPE,
        )
        ffmpeg_processes[camera_id] = proc
        logger.info(
            f"[HLS] Spawned ffmpeg for camera {camera_id} "
            f"(PID {proc.pid}, source: {stream_url})"
        )
        return True

    except FileNotFoundError:
        logger.error(
            "[HLS] ffmpeg not found. "
            "Install: apt install ffmpeg  OR  brew install ffmpeg"
        )
        return False
    except Exception as e:
        logger.error(f"[HLS] Failed to spawn ffmpeg for camera {camera_id}: {e}")
        return False
    finally:
        lock.release()


# ── Terminate ─────────────────────────────────────────────────────────────────

def terminate_ffmpeg(camera_id: Any) -> None:
    """Terminate ffmpeg for a single camera. Called on camera removal."""
    proc = ffmpeg_processes.pop(camera_id, None)
    if proc and proc.poll() is None:
        proc.terminate()
        try:
            proc.wait(timeout=5)
        except subprocess.TimeoutExpired:
            proc.kill()
        logger.info(f"[HLS] Terminated ffmpeg for camera {camera_id}")

def terminate_all_ffmpeg() -> None:
    """Terminate all ffmpeg processes. Called on FastAPI shutdown."""
    for camera_id in list(ffmpeg_processes.keys()):
        terminate_ffmpeg(camera_id)
    logger.info("[HLS] All ffmpeg processes terminated")


# ── stderr read helper ────────────────────────────────────────────────────────

def _read_stderr_safe(proc: subprocess.Popen) -> str:
    """
    Read stderr from an exited process without blocking.
    """
    if not proc.stderr:
        return ""
    try:
        _, stderr_bytes = proc.communicate(timeout=1)
        return stderr_bytes.decode(errors='replace') if stderr_bytes else ""
    except subprocess.TimeoutExpired:
        return "(stderr read timed out)"
    except ValueError:
        return ""
    except Exception:
        return ""


# ── Health monitor ─────────────────────────────────────────────────────────────

async def monitor_ffmpeg_processes() -> None:
    """
    Async task. Polls every 10s. Detects and restarts crashed ffmpeg processes.
    """
    logger.info("[HLS] Monitor started — checking every 10s")

    while not _monitor_shutdown.is_set():
        try:
            await asyncio.wait_for(_monitor_shutdown.wait(), timeout=10.0)
            break
        except asyncio.TimeoutError:
            pass

        db = SessionLocal()
        try:
            for camera_id, proc in list(ffmpeg_processes.items()):
                if proc.poll() is None:
                    # Still running: this camera is healthy, so an earlier
                    # blip should not count toward the give-up threshold.
                    _restart_failures.pop(camera_id, None)
                    continue
                if proc.poll() is not None:
                    stderr_output = _read_stderr_safe(proc)
                    logger.warning(
                        f"[HLS] ffmpeg camera {camera_id} exited "
                        f"(code {proc.returncode}). "
                        f"Stderr: {stderr_output[:200] or 'none'}. Restarting."
                    )
                    camera = db.execute(text(
                        "SELECT stream_url FROM cameras "
                        "WHERE id = :id AND stream_url IS NOT NULL"
                    ), {"id": str(camera_id)}).fetchone()
                    if not camera and str(camera_id).isdigit():
                        camera = db.execute(text(
                            "SELECT stream_url FROM cameras "
                            "WHERE id = :id AND stream_url IS NOT NULL"
                        ), {"id": int(camera_id)}).fetchone()

                    if camera and camera.stream_url:
                        fails = _restart_failures.get(camera_id, 0) + 1
                        _restart_failures[camera_id] = fails
                        if fails > MAX_CONSECUTIVE_RESTARTS:
                            ffmpeg_processes.pop(camera_id, None)
                            logger.error(
                                f"[HLS] Camera {camera_id} failed "
                                f"{fails - 1} restarts in a row — giving up "
                                f"rather than retrying forever against a source "
                                f"that is not answering. Fix the source, then "
                                f"re-add or restart to try again."
                            )
                            continue
                        spawn_ffmpeg(camera_id, camera.stream_url)
                    else:
                        ffmpeg_processes.pop(camera_id, None)
                        logger.info(
                            f"[HLS] Camera {camera_id} removed or "
                            f"stream_url cleared — stopped tracking"
                        )
        except Exception as e:
            logger.error(f"[HLS] Monitor error: {e}", exc_info=True)
        finally:
            db.close()

    logger.info("[HLS] Monitor stopped")


# ── Pre-warm ───────────────────────────────────────────────────────────────────

def prewarm_demo_cameras(db) -> None:
    """
    Spawn ffmpeg for all cameras with stream_url configured.
    Called at startup AFTER clear_all_hls_on_startup().
    """
    cameras = db.execute(text(
        "SELECT id, name, stream_url FROM cameras WHERE stream_url IS NOT NULL"
    )).fetchall()

    for cam in cameras:
        stream_url = cam.stream_url
        if not stream_url:
            continue

        # Only a LOCAL path can be checked for existence. A URL is not a path:
        # Path("https://host/x.m3u8").exists() is always False, which is what
        # silently skipped every network camera here (see is_network_source).
        if is_network_source(stream_url):
            if not PREWARM_NETWORK_SOURCES:
                logger.info(
                    f"[HLS] Not pre-warming network camera {cam.id} ({cam.name}) "
                    f"— live imagery for it is served on demand by the snapshot "
                    f"path. Set SENTINEL_HLS_PREWARM_NETWORK=1 to transcode all "
                    f"network cameras at startup."
                )
                continue
        else:
            source_path = Path(stream_url)
            if not source_path.exists():
                logger.error(
                    f"[HLS] SKIPPING camera {cam.id} ({cam.name}): "
                    f"local source file not found: {stream_url}"
                )
                continue

        logger.info(f"[HLS] Pre-warming camera {cam.id} ({cam.name})")
        spawn_ffmpeg(cam.id, stream_url)
