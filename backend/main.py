"""
backend/main.py — FastAPI application for Sentinel Gujarat (Day 6).

Day 6 additions:
  - ReID engine: OSNet-IBN embedder + FAISS index initialized at startup
  - 3 new API routers: reid_review, reid_journeys, reid_governance
  - APScheduler retention job (runs daily)
  - set_reid_ready() called after index init to gate background ReID tasks

Day 1–5 pipeline is UNCHANGED.
"""
from __future__ import annotations

import asyncio
import logging
import os
import sys
from contextlib import asynccontextmanager
from pathlib import Path
from typing import Any

if sys.platform == "win32":
    asyncio.set_event_loop_policy(asyncio.WindowsSelectorEventLoopPolicy())

from fastapi import FastAPI, WebSocket, WebSocketDisconnect
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from slowapi import _rate_limit_exceeded_handler
from slowapi.errors import RateLimitExceeded

# ── Project root on sys.path ─────────────────────────────────────────────────
_ROOT = Path(__file__).resolve().parent.parent
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

# ── Core config + logging (must be first) ────────────────────────────────────
from backend.core.config import settings
from backend.core.logging import configure_logging, get_logger
from backend.core.rate_limit import limiter

configure_logging()
logger = get_logger(__name__)

# ── Day 2 pipeline imports ────────────────────────────────────────────────────
from backend.config import load_config
from backend.connection_manager import ConnectionManager
from backend.pipeline_bridge import PipelineBridge

# ── Day 5 routers + middleware ──────────────────────────────────────────────────
from backend.auth.routes import router as auth_router
from sqlalchemy import text
from backend.db.models import Base
from backend.db.session import engine
from backend.middleware.request_id import RequestIDMiddleware
from backend.routers.v1.alerts import router as alerts_router
from backend.routers.v1.audit import router as audit_router
from backend.routers.v1.cameras import router as cameras_router, set_ws_broadcast
from backend.routers.v1.health import router as health_router
from backend.routers.v1.fleet import router as fleet_router
from backend.ws.dashboard_ws import broadcast as ws_broadcast
from backend.ws.dashboard_ws import ws_dashboard_handler

# ── Day 6 ReID routers ────────────────────────────────────────────────────
from backend.routers.v1.reid_review import router as reid_review_router
from backend.routers.v1.reid_journeys import router as reid_journeys_router
from backend.routers.v1.reid_governance import router as reid_governance_router

# ── Day 8 debug/observability router ─────────────────────────────────────
from backend.routers.v1.debug import router as debug_router

# ── Top-10 Fleet Command Centre status ─────────────────────────────
from backend.routers.v1.top10_status import router as top10_router

# ── Day 9 audit journey router ──────────────────────────────────────────
from backend.routers.v1.audit_journey import router as audit_journey_router

# ── Day 10 SENTINEL IQ score card + night-mode control ──────────────────
from backend.routers.v1.sentinel_iq import router as iq_router
from backend.routers.v1.sentinel_iq import admin_router as iq_admin_router

# ── Day 13 Evidence Lock ─────────────────────────────────────────────────
from backend.routers.v1.evidence import router as evidence_router

# ── Day 14 Alert Routing ─────────────────────────────────────────────────
from backend.routers.v1.routing import router as routing_router

# ── Day 15 Push Notifications ────────────────────────────────────────────
from backend.routers.v1.push import router as push_router

# ── Day 18 Control Room Dashboard ────────────────────────────────────────
from backend.routers.v1.dashboard import router as dashboard_router

# ── Day 11 Zone incidents router ─────────────────────────────────────────
from backend.routers.v1.zone_incidents import router as zone_incidents_router

# ── Day 16 Feedback & HLS Streaming ──────────────────────────────────────────
from backend.routers.v1.feedback import router as feedback_router
from backend.routers.v1.stream import router as stream_router
# ── Day 16.2 Analytics Dashboard ──────────────────────────────────────────────
from backend.routers.v1.analytics import router as analytics_router
from backend.routers.v1.detections_ws import router as detections_ws_router
from backend.routers.v1.edge import router as edge_router
from backend.routers.v1.reid_demo import router as reid_demo_router
from backend.routers.v1.violations_review import router as violations_review_router
from backend.stream.manager import (
    check_ffmpeg_version,
    clear_all_hls_on_startup,
    prewarm_demo_cameras,
    terminate_all_ffmpeg,
    monitor_ffmpeg_processes,
    _monitor_shutdown,
)
from backend.db.session import SessionLocal



# ── App state ─────────────────────────────────────────────────────────────────
yaml_cfg: dict[str, Any] = {}
# `manager` / `bridge` remain as the FIRST camera's instances so the legacy
# Day 2 endpoints (/api/health, /ws/detections) keep working unchanged.
# `managers` / `bridges` are the full per-camera sets.
manager: ConnectionManager | None = None
bridge: PipelineBridge | None = None
managers: dict[str, ConnectionManager] = {}
bridges: list[PipelineBridge] = []


def _get_sources() -> list[tuple[str, str]]:
    """Return [(camera_id, source), ...] for every feed to process.

    Accepted forms, in precedence order:

      SENTINEL_SOURCES="CAM-01=/path/a.mp4,CAM-02=/path/b.mp4"
          Explicit multi-camera. Each feed gets its own ConnectionManager,
          PipelineBridge, and frame buffer, and stamps its OWN cameras.id on
          every alert it produces.

      SENTINEL_SOURCE="/path/a.mp4"
          Single feed, camera id taken from config.yaml's camera.id. This is
          the original Day 2 form and still works exactly as before.

    The bare `--source` CLI flag is also still honoured.
    """
    multi = os.environ.get("SENTINEL_SOURCES", "").strip()
    if multi:
        pairs: list[tuple[str, str]] = []
        for chunk in multi.split(","):
            chunk = chunk.strip()
            if not chunk:
                continue
            if "=" in chunk:
                cam, _, src = chunk.partition("=")
                pairs.append((cam.strip(), src.strip()))
            else:
                # No camera id given — fall back to the configured default so
                # a malformed entry degrades to single-camera behaviour
                # instead of silently dropping the feed.
                pairs.append((yaml_cfg.get("camera", {}).get("id", "CAM-01"), chunk))
        return pairs

    single = os.environ.get("SENTINEL_SOURCE", "")
    if not single and len(sys.argv) > 1:
        import argparse
        parser = argparse.ArgumentParser(add_help=False)
        parser.add_argument("--source", default="")
        args, _ = parser.parse_known_args()
        single = args.source

    if single:
        return [(yaml_cfg.get("camera", {}).get("id", "CAM-01"), single)]

    # Live cameras from config.yaml's `demo_cameras`, opt-in via
    # SENTINEL_LIVE_CAMERAS=1.
    #
    # WHY OPT-IN rather than the default: starting N feeds starts N
    # PipelineBridge threads, each loading its OWN YOLO model (PersonDetector
    # builds one per instance). That is real GPU memory and real contention,
    # so it must be a deliberate choice, not something that happens to every
    # developer who runs `uvicorn backend.main:app`. It also keeps every
    # existing verify_*.py script — all of which rely on `sources` being
    # empty — behaving exactly as before.
    #
    # SENTINEL_MAX_CAMERAS caps how many start, so you can bring up 4 feeds
    # to check correctness before committing the GPU to all 30.
    if os.environ.get("SENTINEL_LIVE_CAMERAS", "0") == "1":
        return _live_camera_sources()

    return []


def _live_camera_sources() -> list[tuple[str, str]]:
    """[(camera_id, stream_url), ...] from config.yaml's `demo_cameras`.

    Only entries with a usable `url` are returned. A camera with no URL is
    skipped loudly rather than started against an empty source, because
    PipelineBridge would otherwise spin on a capture that never opens.
    """
    cams = yaml_cfg.get("demo_cameras", []) or []
    if not cams:
        logger.error(
            "SENTINEL_LIVE_CAMERAS=1 but config.yaml has no `demo_cameras` "
            "block — no feeds to start. Note config.yaml is resolved relative "
            "to the process CWD; run uvicorn from the repo root."
        )
        return []

    limit_raw = os.environ.get("SENTINEL_MAX_CAMERAS", "").strip()
    try:
        limit = int(limit_raw) if limit_raw else None
    except ValueError:
        logger.warning("SENTINEL_MAX_CAMERAS=%r is not an integer — ignoring "
                       "and starting all cameras.", limit_raw)
        limit = None

    pairs: list[tuple[str, str]] = []
    skipped: list[str] = []
    disabled: list[str] = []
    negotiated: list[str] = []
    for cam in cams:
        cam_id = str(cam.get("id") or cam.get("camera_id") or "").strip()
        url = str(cam.get("url") or "").strip()
        if not cam_id:
            skipped.append("<no id>")
            continue
        # `enabled: false` marks a camera that is configured but known-dead
        # (see config.yaml). Starting a pipeline for it would hold open a
        # connection that never delivers a frame — and on this feed server,
        # concurrent connections are the scarce resource, so a dead one
        # directly costs a live one.
        if cam.get("enabled") is False:
            disabled.append(cam_id)
            continue

        if not url:
            # No explicit URL. This is the NORMAL case when integrating a
            # department's existing estate: they hand over a vendor, an IP
            # and credentials, not a stream URL. Ask the adapter factory to
            # negotiate one — it walks a vendor-specific fallback chain
            # (hikvision -> onvif -> rtsp -> hls, etc.) and returns whatever
            # actually answered.
            #
            # Previously such a camera was silently skipped, which is why
            # the adapter layer never ran in production: every configured
            # camera already had a hand-written URL.
            try:
                from backend.services.camera_adapters.factory import resolve_stream_url
                url = (resolve_stream_url(cam) or "").strip()
            except Exception as exc:
                logger.warning("Camera %s: adapter negotiation failed: %s",
                               cam_id, exc)
                url = ""
            if url:
                negotiated.append(f"{cam_id}({cam.get('vendor', 'unknown')})")
            else:
                skipped.append(cam_id)
                continue

        pairs.append((cam_id, url))

    if negotiated:
        logger.info("Adapter factory negotiated %d camera(s) with no "
                    "preconfigured URL: %s", len(negotiated), ", ".join(negotiated))
    if skipped:
        logger.warning("Skipping %d camera(s) with no id, or whose vendor "
                       "adapter chain found no working stream: %s",
                       len(skipped), ", ".join(skipped))
    if disabled:
        logger.info("Skipping %d camera(s) marked enabled:false — %s",
                    len(disabled), ", ".join(disabled))

    if limit is not None:
        pairs = pairs[:max(0, limit)]

    logger.info("Live camera mode: starting %d feed(s)%s",
                len(pairs),
                f" (capped by SENTINEL_MAX_CAMERAS={limit})" if limit is not None else "")
    return pairs


def _resolve_camera_db_id(camera_str_id: str) -> int | None:
    """cameras.id for a camera_id string, or None if it isn't registered."""
    from backend.db.models import Camera as _Camera
    from backend.db.session import SessionLocal as _SessionLocal

    db = _SessionLocal()
    try:
        cam = (
            db.query(_Camera)
            .filter(_Camera.camera_id == camera_str_id,
                    _Camera.is_deleted == False)  # noqa: E712
            .first()
        )
        return cam.id if cam else None
    finally:
        db.close()


# ── Lifespan ──────────────────────────────────────────────────────────────────

@asynccontextmanager
async def lifespan(app: FastAPI):
    global yaml_cfg, manager, bridge, managers, bridges

    # Ensure all ORM tables exist (idempotent — no data loss)
    Base.metadata.create_all(bind=engine)

    # create_all() above only creates tables that do not exist yet — it never
    # adds a column to a table that already exists, which `cameras` always
    # does on any deployment past its very first run. Migration
    # d16_camera_registry_enrichment is the versioned record of this change;
    # this patch is what actually makes it apply on every boot without
    # depending on someone remembering to run `alembic upgrade head` before
    # a demo (the same self-healing pattern backend/db/session.py::init_db
    # already uses for audit_log/camera_calibrations — duplicated here
    # rather than imported because init_db() itself is not on this app's
    # startup path, only Base.metadata.create_all() is).
    try:
        with engine.connect() as _conn:
            for _col, _ddl in [
                ("camera_type", "VARCHAR(32)"),
                ("installed_at", "DATETIME"),
                ("maintenance_mode", "BOOLEAN DEFAULT 0"),
                ("maintenance_note", "TEXT"),
                ("maintenance_since", "DATETIME"),
            ]:
                try:
                    _conn.execute(text(f"ALTER TABLE cameras ADD COLUMN {_col} {_ddl}"))
                    _conn.commit()
                except Exception:
                    pass  # column already exists — expected on every boot after the first

            # Migration d17_watchlist_plate_active — same self-heal reasoning
            # as above. Existing rows get a NULL `active` from a bare ADD
            # COLUMN, which reads as falsy and would silently drop every
            # watchlist entry that existed before this column did; backfill
            # to 1 so a plate someone already added keeps matching.
            try:
                _conn.execute(text("ALTER TABLE watchlist_plates ADD COLUMN active BOOLEAN"))
                _conn.commit()
            except Exception:
                pass
            try:
                _conn.execute(text(
                    "UPDATE watchlist_plates SET active = 1 WHERE active IS NULL"))
                _conn.commit()
            except Exception:
                pass
    except Exception:
        logger.warning("Camera registry schema self-heal failed", exc_info=True)

    yaml_cfg = load_config()
    camera_id: str = yaml_cfg.get("camera", {}).get("id", "CAM-01")
    expiry: float = yaml_cfg.get("server", {}).get("track_expiry_seconds", 2.0)

    # Wire WebSocket broadcast into cameras router
    set_ws_broadcast(ws_broadcast)

    # ── Day 6: Initialize ReID services ─────────────────────────────────────────
    from backend.services.reid_embedder import get_embedder
    from backend.services.reid_index_manager import get_index_manager
    from backend.connection_manager import set_reid_ready

    _embedder = get_embedder()   # load OSNet-IBN (or stub) once
    logger.info(
        "ReID embedder initialized. stub_mode=%s, checkpoint=%s",
        _embedder.is_stub, _embedder.checkpoint_applied or "default (ImageNet only)",
    )
    _index_mgr = get_index_manager()
    await _index_mgr.initialize()  # load persisted FAISS index (or start empty)
    logger.info(
        "ReID FAISS index ready. vectors=%s", _index_mgr.ntotal
    )

    # Signal to connection_manager that ReID is safe to fire.
    #
    # Camera identity is NOT resolved here any more. It used to be looked up
    # once and stored in a module-level global, which made the whole process
    # single-camera: with two feeds, alerts from the second were stamped with
    # the FIRST camera's id. Each ConnectionManager now resolves and carries
    # its own cameras.id — see _resolve_camera_db_id() and the per-source
    # loop further down.
    set_reid_ready()

    # ── Day 7: Initialize face watchlist services ───────────────────────────────
    from backend.services.face_embedder import get_face_embedder
    from backend.services.face_watchlist_matcher import get_face_watchlist_matcher
    from backend.connection_manager import set_face_watchlist_ready

    _face_embedder = get_face_embedder()   # load InsightFace (or stub) once
    logger.info(
        "Face watchlist embedder initialized. stub_mode=%s", _face_embedder.is_stub
    )
    _face_matcher = get_face_watchlist_matcher()
    _watchlist_count = _face_matcher.reload_watchlist()
    logger.info("Face watchlist loaded. active_entries=%s", _watchlist_count)
    set_face_watchlist_ready()

    # ── Day 7: Camera heartbeat — background task, not per-request ─────────────
    from backend.services.camera_heartbeat import heartbeat_loop

    heartbeat_task = asyncio.create_task(heartbeat_loop(), name="camera-heartbeat-loop")
    app.state.heartbeat_task = heartbeat_task
    logger.info("Camera heartbeat loop scheduled.")

    # ── Day 8: Behavior engine (loitering + crowd anomaly) ──────────────────
    # Best-effort Redis connectivity check — logged either way, never blocks
    # startup. If Redis is down, the engine still runs and skips-and-logs
    # each tick (see services/state.py); this just gives an honest startup
    # signal instead of discovering it silently later.
    from backend.services.state import get_behavior_state
    from backend.services.camera_calibration import get_calibration_cache
    from backend.connection_manager import set_behavior_ready

    _redis_ok = await get_behavior_state().ping()
    logger.info("Behavior engine Redis connectivity: %s", "OK" if _redis_ok else "UNREACHABLE (degraded mode)")
    _calib_count = get_calibration_cache().reload()
    logger.info("Camera calibration cache loaded. calibrated_cameras=%s", _calib_count)
    set_behavior_ready()

    # ── Day 6: APScheduler — retention purge job (runs daily) ──────────────────
    try:
        from apscheduler.schedulers.asyncio import AsyncIOScheduler
        from backend.scripts.reid_retention_job import run_retention_purge

        scheduler = AsyncIOScheduler()
        # Run once at startup (to catch any expired records) then daily at 02:00
        scheduler.add_job(run_retention_purge, "cron", hour=2, minute=0,
                          id="reid_retention", replace_existing=True)
        scheduler.start()
        logger.info("APScheduler started — retention job scheduled daily at 02:00.")
        app.state.scheduler = scheduler
    except ImportError:
        logger.warning(
            "apscheduler not installed — retention job will not run automatically. "
            "Run scripts/reid_retention_job.py manually or via system cron."
        )
        app.state.scheduler = None

    # ── Day 9: Journey + audit-log retention job ─────────────────────────────
    # Wired into the same APScheduler instance as the ReID retention job
    # (if it started successfully above). Runs at 03:00 UTC (separate from
    # the 02:00 ReID job) so both don't hit the DB at the same time.
    if getattr(app.state, "scheduler", None) is not None:
        try:
            from backend.scripts.retention_job import run_full_retention
            app.state.scheduler.add_job(
                run_full_retention, "cron", hour=3, minute=0,
                id="journey_retention", replace_existing=True,
            )
            logger.info("Journey + audit-log retention job scheduled daily at 03:00 UTC.")
        except Exception:
            logger.exception(
                "Failed to register journey_retention job with APScheduler — "
                "run scripts/retention_job.py manually or via system cron."
            )

    # ── Day 9/10: Abandoned-object detector ──────────────────────────────────
    # Day 9 constructed this singleton but nothing ever called it — Day 10
    # wired the rest: config.yaml's class_filter now includes backpack/
    # handbag/suitcase, main.py's pipeline loop tracks and emits them as
    # object_track events, and connection_manager.handle_object_event()
    # (reached via pipeline_bridge's event_type routing) is what actually
    # calls on_object_update()/on_person_positions(). Shares the same Redis
    # and calibration-cache readiness the behavior engine just confirmed
    # above, so it's safe to flip this flag right after set_behavior_ready().
    from backend.services.abandoned_object_detector import get_abandoned_object_detector
    from backend.connection_manager import set_abandoned_ready

    _abandoned_detector = get_abandoned_object_detector()
    set_abandoned_ready()
    logger.info("Abandoned-object detector wired into the live pipeline.")

    # ── Day 13: Evidence Lock ────────────────────────────────────────────────
    # Directories must exist BEFORE any capture can be triggered, and any row
    # left PENDING by a process that died mid-capture is swept to FAILED so
    # the API never reports "processing" for something with no task alive to
    # finish it.
    from backend.services.evidence_capture import ensure_evidence_dirs, sweep_stale_pending

    _evidence_root = ensure_evidence_dirs()
    _swept = sweep_stale_pending()
    logger.info(
        "Evidence Lock ready. root=%s swept_stale_pending=%d", _evidence_root, _swept
    )

    # ── Day 14: Alert Routing ────────────────────────────────────────────────
    #
    # This used to run unconditionally on every startup, injecting two demo
    # cameras (CAM-RT-A "Main Entrance Cam", CAM-RT-B "Parking Lot Cam") into
    # the live registry. They are not real cameras on this estate, and their
    # seeded coordinates — 37.7748,-122.4195 and 37.7752,-122.4179 — are in
    # San Francisco, so they plotted on the GIS map ~13,000 km from Gujarat
    # and inflated every camera count by two (32 shown for a 30-camera
    # estate). Their sources are demo/clips/entrance_loop.mp4 and
    # parking_loop.mp4, which are 94KB SMPTE colour-bar test cards, so
    # opening either one showed colour bars rather than a street.
    #
    # Off by default: the registry must describe the estate that actually
    # exists. The officer-routing demo that needs these still gets them via
    # SENTINEL_SEED_ROUTING_DEMO=1, and the two verify_* scripts import and
    # call seed_routing_demo_data() directly, so they are unaffected.
    if os.environ.get("SENTINEL_SEED_ROUTING_DEMO", "0") == "1":
        from backend.routing.seed import seed_routing_demo_data

        seed_routing_demo_data()
        logger.info("Routing demo cameras seeded (SENTINEL_SEED_ROUTING_DEMO=1) "
                    "— these are NOT real cameras.")

    # Register the escalation tick on the EXISTING scheduler (created above
    # for the Day 6/9 retention jobs) — deliberately not a second instance.
    #
    # That scheduler is an AsyncIOScheduler, so jobs run ON THE EVENT LOOP.
    # escalation_tick is therefore `async def` and pushes its blocking SQLite
    # work to a worker thread; a synchronous tick here would stall every
    # WebSocket client and the detection drain loop for the length of each
    # 5-second scan.
    if getattr(app.state, "scheduler", None) is not None:
        try:
            from backend.routing.escalation import escalation_tick

            app.state.scheduler.add_job(
                escalation_tick,
                trigger="interval",
                seconds=settings.ESCALATION_TICK_SECONDS,
                id="alert_escalation_tick",
                replace_existing=True,
            )
            logger.info(
                "Alert escalation tick scheduled every %ds (ACK timeout %ds).",
                settings.ESCALATION_TICK_SECONDS, settings.ACK_TIMEOUT_SECONDS,
            )

            # Day 15: daily stale-subscription sweep on the SAME scheduler.
            from backend.push.service import cleanup_stale_subscriptions

            app.state.scheduler.add_job(
                cleanup_stale_subscriptions,
                trigger="interval", hours=24,
                id="push_subscription_cleanup", replace_existing=True,
            )
            logger.info(
                "Push subscription cleanup scheduled daily (stale after %d days).",
                settings.PUSH_SUBSCRIPTION_STALE_DAYS,
            )
        except Exception:
            logger.exception("Failed to register alert_escalation_tick.")
    else:
        # Without APScheduler nothing escalates. Ever. That is a silent,
        # total loss of the timeout guarantee, so say so loudly rather than
        # letting the dashboard show countdowns that will never fire.
        logger.error(
            "APScheduler unavailable — alert escalation is DISABLED. "
            "ACK timeouts will never fire. Install apscheduler to enable."
        )

    sources = _get_sources()
    if sources:
        loop = asyncio.get_event_loop()
        for cam_str_id, source in sources:
            cam_db_id = _resolve_camera_db_id(cam_str_id)
            if cam_db_id is None:
                logger.warning(
                    "No cameras row with camera_id=%r — alerts from this feed "
                    "will be written with camera_id=NULL and cannot be routed "
                    "to an officer or aggregated per-camera. Onboard it via "
                    "/api/v1/cameras first.", cam_str_id,
                )

            # Each feed gets its OWN ConnectionManager carrying its own
            # cameras.id. This is what makes multi-camera correct: the id used
            # to stamp alerts is instance state, not a process-wide global
            # that the last camera to start would overwrite for everyone.
            cam_cfg = dict(yaml_cfg)
            cam_cfg["camera"] = {**yaml_cfg.get("camera", {}), "id": cam_str_id}

            m = ConnectionManager(
                camera_id=cam_str_id,
                track_expiry_seconds=expiry,
                # Same cadence as the crop throttle, from the same config value —
                # so "how often do we touch the DB for this track" and "how often
                # do we cut a crop for it" stay one number, not two that drift.
                track_update_interval_frames=yaml_cfg.get("output", {})
                    .get("crop_save_interval_frames", 15),
                camera_db_id=cam_db_id,
            )
            b = PipelineBridge(cfg=cam_cfg, source=source, connection_manager=m)
            b.start(loop)

            managers[cam_str_id] = m
            bridges.append(b)
            logger.info("Pipeline started", extra={
                "camera_id": cam_str_id, "camera_db_id": cam_db_id, "source": source,
            })

        # Legacy Day 2 endpoints (/api/health, /ws/detections) are
        # single-camera by construction; point them at the first feed.
        manager = managers[sources[0][0]]
        bridge = bridges[0]
        logger.info("Multi-camera pipeline active: %d feed(s) — %s",
                    len(bridges), ", ".join(c for c, _ in sources))
    else:
        logger.info("Starting Sentinel Gujarat 30-Camera Ingestion & Analytics Pipeline...")
        try:
            from backend.metrics.prometheus_metrics import start_metrics_server
            start_metrics_server()
        except Exception as e:
            logger.debug("Prometheus server: %s", e)

    # ── 24x7 Live CCTV Pipeline (Ingestion + Homography Theil-Sen Velocity + Vault) ──
    from backend.services.live_24x7_pipeline import get_live_pipeline
    # Off by default in the API process — a deployment decision, not a
    # preference. The pipeline saturates a GPU and a CPU core decoding and
    # inferring over 27 streams. Started here it shares a process with the
    # request handlers and measurably starves them: with it running, /docs and
    # /analytics/calibration/cameras both timed out after 30 s while the
    # process held 6.7 GB and 1,067 s of CPU. An operator trying to calibrate
    # a camera could not load the page at all. The process then sat inside a
    # CUDA call that ignored termination, so the port stayed bound after kill.
    #
    # Ingestion and serving belong in separate processes:
    #     python -m backend.scripts.run_pipeline
    # Set SENTINEL_PIPELINE_AUTOSTART=1 for a small demo where few cameras and
    # an idle API can share a machine.
    if os.environ.get("SENTINEL_PIPELINE_AUTOSTART", "0") == "1":
        get_live_pipeline().start()
        logger.warning(
            "24x7 pipeline started inside the API process "
            "(SENTINEL_PIPELINE_AUTOSTART=1). API latency suffers under load."
        )
    else:
        logger.info(
            "24x7 pipeline not started in this process. Run "
            "`python -m backend.scripts.run_pipeline` alongside the API, or "
            "set SENTINEL_PIPELINE_AUTOSTART=1 to start it here."
        )

    # Subscribe to pubsub listener for live WebSocket broadcasting
    async def _pubsub_broadcast_task():
        from backend.cache.redis_cache import create_subscriber
        try:
            pubsub = await create_subscriber()
            if hasattr(pubsub, "listen"):
                async for msg in pubsub.listen():
                    if msg.get("type") == "message":
                        try:
                            import json
                            data = json.loads(msg.get("data", "{}"))
                            await ws_broadcast(data)
                        except Exception:
                            pass
        except Exception:
            pass

    pubsub_task = asyncio.create_task(_pubsub_broadcast_task(), name="pubsub-broadcast")

    # ── Day 16: Feedback & HLS Streaming ─────────────────────────────────────
    check_ffmpeg_version()
    _hls_db = SessionLocal()
    try:
        clear_all_hls_on_startup()
        prewarm_demo_cameras(_hls_db)
    finally:
        _hls_db.close()

    hls_monitor_task = asyncio.create_task(monitor_ffmpeg_processes(), name="hls-monitor")

    yield

    _monitor_shutdown.set()
    hls_monitor_task.cancel()
    terminate_all_ffmpeg()

    get_live_pipeline().stop()
    pubsub_task.cancel()
    if hasattr(app.state, "coordinator"):
        app.state.coordinator.stop()
    for b in bridges:
        await b.stop()
    if getattr(app.state, "scheduler", None):
        app.state.scheduler.shutdown()
    if getattr(app.state, "heartbeat_task", None):
        app.state.heartbeat_task.cancel()
    logger.info("Sentinel Gujarat backend shut down.")


# ── FastAPI app ───────────────────────────────────────────────────────────────

app = FastAPI(
    title="Sarvanetra API",
    description="CCTV Intelligence Platform — AI Video Analytics, ReID & Journey Tracking",
    version="0.6.0",
    lifespan=lifespan,
)

app.add_middleware(RequestIDMiddleware)
app.add_middleware(
    CORSMiddleware,
    allow_origins=list(set([
        "http://localhost:5173",
        "http://127.0.0.1:5173",
        "http://localhost:3000",
        "http://127.0.0.1:3000",
        "http://localhost:8000",
        "http://127.0.0.1:8000",
    ] + settings.cors_origins_list)),
    allow_origin_regex=r"^https?:\/\/.*",
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
    expose_headers=[
        "Cache-Control",
        "Pragma",
        "Content-Type",
        "Authorization",
        "X-Request-ID",
        # Dimensions of a served calibration frame. Control points are stored
        # in the pixel coordinates of the frame they were picked on, so the
        # browser has to know that frame's true size — and a custom header is
        # invisible to JavaScript unless it is named here.
        "X-Frame-Width",
        "X-Frame-Height",
    ],
)

# ── Rate limiter ──────────────────────────────────────────────────────────────
app.state.limiter = limiter
app.add_exception_handler(RateLimitExceeded, _rate_limit_exceeded_handler)

# ── Day 7: Static evidence/reference image serving ────────────────────────────
# Deliberately scoped to just these two directories (not the whole project
# root) so nothing under .env, source, or the DB file is ever served over
# HTTP. Unauthenticated, matching the existing (pre-Day-7) convention already
# assumed by ReviewQueue.jsx's crop images — acceptable for this local/demo
# deployment, but revisit before any real deployment.
(_ROOT / "output").mkdir(parents=True, exist_ok=True)
app.mount("/media/output", StaticFiles(directory=str(_ROOT / "output")), name="media-output")
if (_ROOT / "data" / "prep").exists():
    app.mount("/media/watchlist-refs", StaticFiles(directory=str(_ROOT / "data" / "prep")),
              name="media-watchlist-refs")

# ── v1 router tree (all prefixed /api/v1) ────────────────────────────────────
app.include_router(health_router,          prefix="/api/v1")
app.include_router(auth_router,            prefix="/api/v1")
app.include_router(cameras_router,         prefix="/api/v1")
app.include_router(alerts_router,          prefix="/api/v1")
app.include_router(audit_router,           prefix="/api/v1")
# Day 6 ReID routers
app.include_router(reid_review_router,     prefix="/api/v1")
app.include_router(reid_journeys_router,   prefix="/api/v1")
app.include_router(reid_governance_router, prefix="/api/v1")
# Day 8
app.include_router(debug_router,           prefix="/api/v1")
# Top-10 Fleet Command Centre — per-camera AI proof endpoint
app.include_router(top10_router,           prefix="/api/v1")
# Edge nodes post detections here instead of streaming video centrally.
app.include_router(edge_router,            prefix="/api/v1")
# Four clips of one motorcycle, each carrying the identity the matcher assigned.
app.include_router(reid_demo_router,       prefix="/api/v1")
# Day 9 audit journey queries
app.include_router(audit_journey_router,   prefix="/api/v1")
# Day 10 SENTINEL IQ
app.include_router(iq_router,              prefix="/api/v1")
app.include_router(iq_admin_router,        prefix="/api/v1")
# Day 13 Evidence Lock
app.include_router(evidence_router,        prefix="/api/v1")
# Day 14 Alert Routing
app.include_router(routing_router,         prefix="/api/v1")
# Day 15 Push Notifications
app.include_router(push_router,            prefix="/api/v1")
# Day 18 Control Room Dashboard
app.include_router(dashboard_router,       prefix="/api/v1")
# Day 11 Zone incidents
app.include_router(zone_incidents_router,   prefix="/api/v1")
# Day 16 Feedback & HLS Streaming
app.include_router(feedback_router,        prefix="/api/v1")
app.include_router(stream_router,          prefix="/api/v1")
# Day 16.2 Analytics Dashboard
app.include_router(analytics_router,       prefix="/api/v1")
app.include_router(fleet_router,           prefix="/api/v1")
app.include_router(detections_ws_router,    prefix="/api/v1")
app.include_router(detections_ws_router)
app.include_router(violations_review_router, prefix="/api/v1")
app.include_router(violations_review_router)
# Master Architecture Cross-Camera Journeys (Person & Vehicle ReID)
from backend.routers.v1.journeys import router as unified_journeys_router
from backend.core.inference import router as inference_router
from backend.routers.v1.review import router as review_router
from backend.routers.v1.vault import router as vault_router

app.include_router(unified_journeys_router, prefix="/api/v1")
app.include_router(unified_journeys_router)
app.include_router(inference_router)
app.include_router(review_router, prefix="/api/v1")
app.include_router(vault_router, prefix="/api/v1")

# v15.0.0 Routers: VAHAN 4.0, Dial-112 CAD Patrol, and Protected Shadow Admin
from backend.routers.v1.vahan_lookup import router as vahan_router
from backend.routers.v1.cad_patrol import router as cad_router
from backend.routers.v1.shadow_admin import router as shadow_admin_router

app.include_router(vahan_router, prefix="/api/v1")
app.include_router(cad_router, prefix="/api/v1")
app.include_router(shadow_admin_router, prefix="/api/v1")

# Plate search + watchlist management. Never registered before — the whole
# router (GET /plate-search, GET /plate-search/evidence, and the new
# /plate-search/watchlist endpoints) was unreachable via the live API on
# both ends: no include_router() call here, and nothing in the frontend
# called it either. A pre-existing dead parameter on search_plate
# (request_obj: Optional[Request] = None, unused in the function body)
# would have crashed FastAPI's route registration the moment this was
# added, which is very likely why it never was — fixed alongside this.
from backend.routers.v1.plate_search import router as plate_search_router

app.include_router(plate_search_router, prefix="/api/v1")

# Also mount at root for path flexibility
app.include_router(feedback_router)
app.include_router(stream_router)
app.include_router(analytics_router)
app.include_router(vault_router)



# ── Legacy & Top-level health endpoints ──────────────────────────────────────

@app.get("/health")
@app.get("/api/health")
async def legacy_health() -> dict[str, Any]:
    """Health endpoint — for Docker healthcheck and monitoring."""
    return {
        "status": "ok",
        "pipeline_running": bridge.is_running if bridge else False,
    }


@app.get("/api/cameras")
async def get_all_cameras():
    """Return all 30 cameras from DB with status.

    Not called by the frontend (it talks to /api/v1/cameras instead) but kept
    working and kept honest: unfiltered, this returned every soft-deleted row
    plus any demo/test rig such as CAM_33, and stamped every one of them
    ONLINE regardless of whether it has ever produced a frame — see
    reports/ops_ticket_dark_cameras.md for why that field cannot be trusted
    fleet-wide. Both are fixed the same way the v1 router now is.
    """
    from backend.db.models import Camera as DBCamera
    from backend.db.session import SessionLocal
    from backend.services.fleet_census import is_test_camera
    db = SessionLocal()
    try:
        cams = [c for c in
                db.query(DBCamera).filter(DBCamera.is_deleted == False).all()  # noqa: E712
                if not is_test_camera(c.id, c.name)]
        return [
            {
                "id": c.id,
                "camera_id": c.camera_id or c.id,
                "name": c.name,
                "url": c.url,
                "lat": c.lat or c.gps_lat,
                "lon": c.lon or c.gps_lon,
                "gps_lat": c.lat or c.gps_lat,
                "gps_lon": c.lon or c.gps_lon,
                "zone": c.zone,
                "district": c.district,
                "department": c.department,
                "crime_level": c.crime_level,
                "is_restricted": c.is_restricted,
                "status": "ONLINE",
                "is_online": True,
            }
            for c in cams
        ]
    finally:
        db.close()


@app.get("/api/officers")
async def get_all_officers():
    from backend.db.models import Officer as DBOfficer
    from backend.db.session import SessionLocal
    db = SessionLocal()
    try:
        officers = db.query(DBOfficer).all()
        return [
            {
                "id": o.id,
                "name": o.name,
                "badge_number": o.badge_number,
                "phone": o.phone,
                "zone": o.zone,
                "district": o.district,
                "lat": o.lat or o.current_lat,
                "lng": o.lng or o.current_lon,
                "status": o.status or "AVAILABLE",
                "is_available": o.is_available,
            }
            for o in officers
        ]
    finally:
        db.close()


@app.get("/api/system/health")
async def system_health():
    return {
        "status": "operational",
        "version": "3.0.0",
        "cameras": 30,
        "mode": "live_production",
    }


@app.get("/api/system/stats")
async def system_stats():
    from backend.db.models import Alert as DBAlert
    from backend.db.session import SessionLocal
    db = SessionLocal()
    try:
        total = db.query(DBAlert).count()
        critical = db.query(DBAlert).filter(DBAlert.severity == "critical").count()
        return {"total_alerts": total, "active_alerts": total, "critical_alerts": critical}
    finally:
        db.close()


@app.get("/api/watchlist/plates/{plate}")
async def check_plate_endpoint(plate: str):
    from backend.db.models import WatchlistPlate
    from backend.db.session import SessionLocal
    clean = plate.upper().replace("-", "").replace(" ", "")
    db = SessionLocal()
    try:
        res = db.query(WatchlistPlate).filter(WatchlistPlate.plate == clean).first()
        if res:
            return {"plate": clean, "is_watchlisted": True, "category": res.category, "reason": res.reason}
        return {"plate": clean, "is_watchlisted": False}
    finally:
        db.close()


# ── Production Observability & Health Probes (Kubernetes / Prometheus) ──────────

@app.get("/healthz", tags=["observability"])
def healthz_probe():
    """Liveness probe: verifies process is alive and responsive."""
    return {"status": "ok", "service": "sentinel_gujarat", "version": "18.0.0"}


@app.get("/readyz", tags=["observability"])
def readyz_probe():
    """Readiness probe: verifies database connection and core AI pipelines."""
    from backend.db.session import SessionLocal
    from backend.services.event_bus import AsyncEventBus
    
    db_ok = False
    try:
        db = SessionLocal()
        db.execute("SELECT 1")
        db_ok = True
        db.close()
    except Exception:
        db_ok = False

    event_bus_stats = AsyncEventBus.get_instance().get_stats()
    
    return {
        "status": "ready" if db_ok else "degraded",
        "database_connected": db_ok,
        "event_bus": event_bus_stats,
        "active_camera_bridges": len(bridges),
    }


@app.get("/metrics", tags=["observability"])
def prometheus_metrics():
    """Prometheus metrics endpoint for 80k camera fleet monitoring."""
    from fastapi.responses import PlainTextResponse
    from backend.services.event_bus import AsyncEventBus
    
    stats = AsyncEventBus.get_instance().get_stats()
    
    lines = [
        "# HELP sentinel_events_published_total Total events published to event bus",
        "# TYPE sentinel_events_published_total counter",
        f"sentinel_events_published_total {stats.get('events_published', 0)}",
        "# HELP sentinel_events_processed_total Total events processed by background workers",
        "# TYPE sentinel_events_processed_total counter",
        f"sentinel_events_processed_total {stats.get('events_processed', 0)}",
        "# HELP sentinel_events_dropped_total Total events dropped due to backpressure",
        "# TYPE sentinel_events_dropped_total counter",
        f"sentinel_events_dropped_total {stats.get('events_dropped', 0)}",
        "# HELP sentinel_event_bus_queue_size Current queue depth",
        "# TYPE sentinel_event_bus_queue_size gauge",
        f"sentinel_event_bus_queue_size {stats.get('queue_size', 0)}",
        "# HELP sentinel_active_camera_bridges Count of active camera pipeline bridges",
        "# TYPE sentinel_active_camera_bridges gauge",
        f"sentinel_active_camera_bridges {len(bridges)}",
    ]
    return PlainTextResponse("\n".join(lines) + "\n", media_type="text/plain")


# ── WebSocket: legacy detection stream (Day 2, no auth) ──────────────────────

@app.websocket("/ws/detections")
async def ws_detections(websocket: WebSocket) -> None:
    """Legacy Day 2 WebSocket — unauthenticated detection event stream."""
    if manager is None:
        await websocket.accept()
        await websocket.send_json({"type": "error", "message": "Pipeline not running."})
        await websocket.close()
        return
    await manager.connect(websocket)
    try:
        while True:
            await websocket.receive_text()
    except WebSocketDisconnect:
        pass
    except Exception as exc:
        logger.debug("WS error: %s", exc)
    finally:
        await manager.disconnect(websocket)


# ── WebSocket: Day 5 auth-gated dashboard stream ──────────────────────────────

@app.websocket("/ws/dashboard")
async def ws_dashboard(websocket: WebSocket) -> None:
    """Day 5 auth-gated WebSocket. Client sends {token} as first message."""
    await ws_dashboard_handler(websocket)


# ── Static File Mounts for Evidence & Calibration Media ──────────────────────
from fastapi.staticfiles import StaticFiles

_output_dir = os.path.join(os.path.dirname(os.path.dirname(__file__)), "output")
if os.path.exists(_output_dir):
    app.mount("/media/output", StaticFiles(directory=_output_dir), name="media_output")
    app.mount("/output", StaticFiles(directory=_output_dir), name="output")

