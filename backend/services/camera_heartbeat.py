"""
backend/services/camera_heartbeat.py — Background camera health monitor (Day 7).

Every CAMERA_HEARTBEAT_INTERVAL_SECONDS (default 30s), checks every registered
camera's feed is still readable and updates cameras.status accordingly,
broadcasting only actual transitions over the dashboard WebSocket.

check_camera_health() is the real-integration swap point: for simulated /
file-based demo feeds it checks the path is a readable non-empty file; for a
real RTSP/ONVIF camera later, replace the body with an actual protocol probe.
The signature (Camera in, bool out) must not change so that swap doesn't
ripple into heartbeat_loop() or anywhere else.

Debounce: a camera must fail CAMERA_HEARTBEAT_FAILURE_THRESHOLD (default 2)
consecutive checks before flipping ONLINE -> OFFLINE, to avoid flapping status
on one transient hiccup. Worst case this adds one extra 30s cycle, i.e. the
dashboard reflects a dead feed within 30-60s — still within the Day 7
checkpoint's tolerance. A single successful check immediately flips back to
ONLINE (no debounce on recovery — false "still down" is worse than a blip).

Concurrency: all cameras are checked concurrently via asyncio.gather(), not
in a sequential loop with blocking calls — each check runs its blocking probe
in a thread executor. This keeps a handful of test cameras fast today and
avoids an obvious rewrite later, but a single asyncio event loop with a
thread-pool executor will NOT scale to Gujarat's eventual ~80,000-camera
target — that needs sharding across multiple worker processes (or a proper
task queue), not just this loop running faster. Documented, not built.

Logging/broadcast discipline: only actual status transitions are logged to
audit_log and broadcast over WebSocket. Every camera is checked every cycle,
but a healthy camera staying ONLINE produces no audit row and no WS message —
otherwise both would fill up with "still online" noise every 30s per camera.

Usage (registered once at FastAPI startup, not per-request):
    from backend.services.camera_heartbeat import heartbeat_loop
    asyncio.create_task(heartbeat_loop())
"""
from __future__ import annotations

import asyncio
import logging
import os
import re
import time
from datetime import datetime
from pathlib import Path
from typing import Any, Optional

from backend.core.config import settings
from backend.db.models import Camera
from backend.db.session import SessionLocal
from backend.services.audit_logger import log_audit

logger = logging.getLogger(__name__)

HEARTBEAT_INTERVAL_S = settings.CAMERA_HEARTBEAT_INTERVAL_SECONDS
FAILURE_THRESHOLD = settings.CAMERA_HEARTBEAT_FAILURE_THRESHOLD

# ── corp8-backed cameras ─────────────────────────────────────────────────────
#
# The generic check below (`_check_sync`) was silently useless for these 30
# cameras: it probes `camera.ip_address`, which is None for every one of
# them — their address lives in `.url` — and even pointed at the right field,
# a plain cv2.VideoCapture cannot open an AES-encrypted, cookie-gated HLS
# stream (measured: isOpened() returns False with no diagnostic across five
# option spellings, see hls_ffmpeg_capture.py). Because settings.DEMO_MODE
# treats "no ip_address" as automatically healthy, every corp8 camera has been
# reporting ONLINE regardless of whether the feed is actually reachable — not
# lying, just never actually checking.
#
# Real checking has to respect what corp8 punishes, which was characterised
# the hard way earlier today: a session has a budget of ~45 requests before
# HTTP 429, and answering 429s with fast re-logins is what triggers HTTP 403
# across every camera on the account, sometimes for minutes. A background task
# that runs forever cannot afford to relearn that lesson on its own.
#
# So this check is deliberately light and deliberately rare:
#   - a lightweight PLAYLIST fetch (no ffmpeg, no video decode) — the same
#     probe() used to verify the fleet, not a full stream open
#   - run on its own, much longer interval (default 5 min), independent of
#     the 30s interval used for local/demo cameras
#   - sequential with real spacing between cameras, on the single shared
#     session — never one request per camera fired at once
#   - a circuit breaker: any 403 during a batch trips a 30-minute cooldown,
#     so a mistake here degrades to "stale health data" rather than another
#     lockout
#   - fails CLOSED on any exception in the sense that matters: a skipped or
#     errored check never flips a camera to OFFLINE. Only an actual completed
#     probe result changes status. Silence is not evidence of an outage.
CORP8_HEALTH_CHECK_INTERVAL_S = float(
    os.environ.get("SENTINEL_CORP8_HEALTH_INTERVAL", "300"))
# SENTINEL_CORP8_HEALTH=0 turns the corp8 probe off entirely. A long interval
# is not the same thing: the first cycle after startup always runs, and that
# cycle is thirty playlist requests on the portal session the live camera
# depends on — two-thirds of its ~45-request budget, spent before the camera
# has shown a frame. Turn it off whenever one camera is being streamed live.
CORP8_HEALTH_ENABLED = os.environ.get("SENTINEL_CORP8_HEALTH", "1") != "0"
CORP8_HEALTH_PROBE_SPACING_S = float(
    os.environ.get("SENTINEL_CORP8_HEALTH_SPACING", "1.5"))
CORP8_CIRCUIT_BREAKER_COOLDOWN_S = float(
    os.environ.get("SENTINEL_CORP8_CIRCUIT_BREAKER_COOLDOWN", "45.0"))

_CORP8_URL_RE = re.compile(r"corp8\.cloud/([A-Za-z0-9_-]+)/index\.m3u8")

_corp8_last_check_at: float = 0.0
_corp8_circuit_open_until: float = 0.0
_corp8_last_result: dict[str, bool] = {}   # camera.id -> last observed health


def _corp8_camera_id(url: Optional[str]) -> Optional[str]:
    """Extract the portal's own camera id ('cam21') from a stored stream URL.

    Read from the URL rather than a separate stored mapping — the URL is
    already the source of truth for which portal camera this row points at,
    and a second field would just be one more thing to keep in sync.
    """
    if not url:
        return None
    m = _CORP8_URL_RE.search(url)
    return m.group(1) if m else None


def _corp8_batch_probe_sync(cams: list[Camera]) -> dict[str, bool]:
    """Sequential, spaced, playlist-only probe of every corp8 camera given.

    Runs entirely inside one thread-executor call so the cameras are checked
    one after another with real spacing between the HTTP requests, never as a
    burst — bursts are exactly what the rate limit punishes.

    Returns {camera.id: is_healthy}. A camera that could not be resolved to a
    portal id, or that raised during its own probe, is simply absent from the
    result rather than recorded as unhealthy — see the module docstring on
    failing closed.
    """
    global _corp8_circuit_open_until

    now = time.monotonic()
    if now < _corp8_circuit_open_until:
        logger.debug("corp8 health check skipped — circuit breaker open for "
                     "%.0fs more", _corp8_circuit_open_until - now)
        return {}

    try:
        from backend.services.corp8_session import get_corp8_session
    except Exception as exc:                                       # noqa: BLE001
        logger.warning("corp8_session unavailable for health check: %s", exc)
        return {}

    sess = get_corp8_session()
    out: dict[str, bool] = {}
    for i, cam in enumerate(cams):
        cid = _corp8_camera_id(cam.url)
        if not cid:
            continue
        if i:
            time.sleep(CORP8_HEALTH_PROBE_SPACING_S)
        try:
            info = sess.probe(cid)
        except Exception as exc:                                   # noqa: BLE001
            logger.debug("corp8 health probe error for %s (%s): %s",
                         cam.id, cid, exc)
            continue
        if info.get("status") == 403:
            logger.warning(
                "corp8 health check hit HTTP 403 on %s — opening circuit "
                "breaker for %.0f minutes rather than risk deepening the "
                "lockout.", cam.id, CORP8_CIRCUIT_BREAKER_COOLDOWN_S / 60)
            _corp8_circuit_open_until = time.monotonic() + CORP8_CIRCUIT_BREAKER_COOLDOWN_S
            break
        out[cam.id] = bool(info.get("ok"))
    return out


def _check_sync(address: str | None) -> bool:
    """Blocking health probe. Runs in a thread executor — never called directly
    from the event loop. This is the real-RTSP/ONVIF swap point.
    """
    if settings.DEMO_MODE or getattr(settings, "SENTINEL_DEMO_MODE", False):
        if not address:
            return True

    if not address:
        return False

    path = Path(address)
    if path.exists():
        # File-based / simulated test feed.
        try:
            return path.is_file() and path.stat().st_size > 0
        except OSError:
            return False

    # Network address (RTSP/HTTP/etc.) — attempt to open it. cv2.VideoCapture
    # handles both file paths and stream URLs, so this also covers addresses
    # that don't resolve to a local path.
    cap = None
    try:
        import cv2
        cap = cv2.VideoCapture(address)
        return bool(cap.isOpened())
    except Exception:
        return False
    finally:
        if cap is not None:
            cap.release()


async def check_camera_health(camera: Camera) -> bool:
    """Check whether a camera's feed is currently readable.

    Runs the blocking probe in a thread executor so a slow/hanging camera
    can never stall the event loop or the rest of the heartbeat cycle.

    corp8-backed cameras are NOT checked here — see _corp8_batch_probe_sync
    and its module-level comment for why they need their own, much more
    conservative path.
    """
    loop = asyncio.get_event_loop()
    return await loop.run_in_executor(None, _check_sync, camera.ip_address)


def _apply_result(camera: Camera, healthy: bool, now: datetime,
                  changed: list[dict[str, Any]], db) -> None:
    """Shared transition logic: update status, log a transition if any."""
    camera.last_heartbeat_at = now

    # A camera flagged for maintenance (PATCH /cameras/{id}/maintenance) is
    # known-down for a known, deliberate reason — not "offline" in the sense
    # that word means everywhere else in this module (unreachable/broken).
    # The probe still ran and last_heartbeat_at above still records that,
    # but the status itself is left alone: an automatic health check must
    # never silently flip a camera someone just pulled for repair back to
    # ONLINE (mid-repair, briefly reachable) or OFFLINE (fighting the manual
    # flag, making the maintenance state look like it never took).
    if getattr(camera, "maintenance_mode", False):
        return

    old_status = camera.status

    if healthy:
        camera.consecutive_failures = 0
        new_status = "ONLINE"
    else:
        camera.consecutive_failures = (camera.consecutive_failures or 0) + 1
        new_status = (
            "OFFLINE" if camera.consecutive_failures >= FAILURE_THRESHOLD
            else old_status
        )

    if new_status != old_status:
        camera.status = new_status
        changed.append({
            "camera_id": camera.id,
            "camera_name": camera.name,
            "status": new_status,
        })
        log_audit(
            db, user=None,
            action="CAMERA_WENT_OFFLINE" if new_status == "OFFLINE" else "CAMERA_CAME_ONLINE",
            resource_type="camera", resource_id=camera.id,
            details={
                "old_status": old_status,
                "new_status": new_status,
                "consecutive_failures": camera.consecutive_failures,
            },
        )


async def _run_heartbeat_cycle() -> None:
    global _corp8_last_check_at

    db = SessionLocal()
    try:
        cameras = db.query(Camera).filter(Camera.is_deleted == False).all()  # noqa: E712
        if not cameras:
            return

        corp8_cams = [c for c in cameras if _corp8_camera_id(c.url)]
        other_cams = [c for c in cameras if c not in corp8_cams]

        now = datetime.utcnow()
        changed: list[dict[str, Any]] = []

        # ── Non-corp8 cameras: unchanged behaviour, every cycle ────────────────
        if other_cams:
            results = await asyncio.gather(
                *(check_camera_health(c) for c in other_cams))
            for camera, healthy in zip(other_cams, results):
                _apply_result(camera, healthy, now, changed, db)

        # ── corp8 cameras: their own gated, spaced, circuit-broken path ────────
        mono_now = time.monotonic()
        if (corp8_cams and CORP8_HEALTH_ENABLED
                and mono_now - _corp8_last_check_at >= CORP8_HEALTH_CHECK_INTERVAL_S):
            _corp8_last_check_at = mono_now
            loop = asyncio.get_event_loop()
            results_by_id = await loop.run_in_executor(
                None, _corp8_batch_probe_sync, corp8_cams)
            for camera in corp8_cams:
                if camera.id not in results_by_id:
                    # Not checked this round — circuit breaker, resolve
                    # failure, or a per-camera error. Leave it exactly as it
                    # was; see the module docstring on failing closed.
                    continue
                _apply_result(camera, results_by_id[camera.id], now, changed, db)

        db.commit()

        if changed:
            from backend.ws.dashboard_ws import broadcast
            for evt in changed:
                await broadcast({
                    "type": "camera_status_changed",
                    "camera_id": evt["camera_id"],
                    "camera_name": evt["camera_name"],
                    "status": evt["status"],
                    "timestamp": now.isoformat(),
                })
            logger.info(
                "Camera heartbeat: %d status change(s): %s",
                len(changed), [(c["camera_id"], c["status"]) for c in changed],
            )
    except Exception:
        db.rollback()
        raise
    finally:
        db.close()


async def heartbeat_loop() -> None:
    """Long-running background task — checked once, then every interval, forever.

    Any exception in a single cycle is logged and swallowed so one bad cycle
    (e.g. a transient DB hiccup) doesn't kill monitoring for every camera
    permanently; the next cycle just tries again.
    """
    logger.info(
        "Camera heartbeat loop starting (interval=%ds, failure_threshold=%d).",
        HEARTBEAT_INTERVAL_S, FAILURE_THRESHOLD,
    )
    while True:
        try:
            await _run_heartbeat_cycle()
        except asyncio.CancelledError:
            logger.info("Camera heartbeat loop cancelled — shutting down.")
            raise
        except Exception:
            logger.exception("Camera heartbeat cycle failed — will retry next interval.")
        await asyncio.sleep(HEARTBEAT_INTERVAL_S)
