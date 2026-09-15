# sentinel/main.py
"""
sentinel/main.py — Production-ready enterprise FastAPI platform for Sentinel Gujarat (Day 16.2).
"""

import asyncio
import uuid
from contextlib import asynccontextmanager
from pathlib import Path

import structlog
from fastapi import FastAPI, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.middleware.gzip import GZipMiddleware
from fastapi.responses import JSONResponse
from fastapi.staticfiles import StaticFiles
from slowapi.errors import RateLimitExceeded
from sqlalchemy import text

from sentinel.config import settings
from sentinel.db import engine
from sentinel.health.router import router as health_router
from sentinel.auth.router import router as auth_router
from sentinel.analytics.router import router as analytics_router
from sentinel.audit.router import router as audit_router
from sentinel.retention.router import router as retention_router
from sentinel.stream.router import router as stream_router
from sentinel.feedback.router import router as feedback_router
from sentinel.logging_config import configure_logging
from sentinel.middleware.rate_limit import limiter, rate_limit_exceeded_handler
from sentinel.scheduler import scheduler
from sentinel.stream.manager import (
    check_ffmpeg_version,
    clear_all_hls_on_startup,
    spawn_ffmpeg,
    terminate_all_ffmpeg,
    monitor_ffmpeg_processes,
    _monitor_shutdown,
)

configure_logging()
log = structlog.get_logger(__name__)


@asynccontextmanager
async def lifespan(app: FastAPI):
    # ── Startup ────────────────────────────────────────────────────────────────
    log.info("sentinel.starting", env=settings.ENV, debug=settings.DEBUG)

    check_ffmpeg_version()
    clear_all_hls_on_startup()

    # Pre-warm demo cameras
    try:
        async with engine.connect() as conn:
            cameras = (await conn.execute(text(
                "SELECT id, name, stream_url FROM cameras WHERE stream_url IS NOT NULL"
            ))).fetchall()

            for cam in cameras:
                cam_url = getattr(cam, "stream_url", None)
                if cam_url:
                    if not str(cam_url).startswith("rtsp://"):
                        if not Path(cam_url).exists():
                            log.warning("hls.source_missing", camera_id=cam.id, stream_url=cam_url)
                            continue
                    spawn_ffmpeg(cam.id, str(cam_url))
    except Exception as e:
        log.warning("hls.prewarm_skipped", reason=str(e))

    asyncio.create_task(monitor_ffmpeg_processes(), name="hls-monitor")

    # Start APScheduler
    scheduler.start()
    log.info("scheduler.started", jobs=[j.id for j in scheduler.get_jobs()])

    log.info("sentinel.ready", env=settings.ENV)
    yield

    # ── Shutdown ───────────────────────────────────────────────────────────────
    log.info("sentinel.shutting_down")
    _monitor_shutdown.set()
    terminate_all_ffmpeg()
    scheduler.shutdown(wait=False)
    await engine.dispose()
    log.info("sentinel.stopped")


app = FastAPI(
    title="Sentinel IQ",
    version="1.0.0",
    lifespan=lifespan,
    docs_url="/api/docs" if settings.ENV != "production" else None,
    redoc_url=None,
)

# ── Middleware (applied in reverse registration order) ────────────────────────
app.add_middleware(GZipMiddleware, minimum_size=1000)
app.add_middleware(
    CORSMiddleware,
    allow_origins=settings.ALLOWED_ORIGINS + ["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
    expose_headers=["Cache-Control", "Pragma", "X-Request-ID", "Content-Type"],
    max_age=600,
)

# ── Rate limiter ──────────────────────────────────────────────────────────────
app.state.limiter = limiter
app.add_exception_handler(RateLimitExceeded, rate_limit_exceeded_handler)


# ── Request ID middleware ─────────────────────────────────────────────────────
@app.middleware("http")
async def request_id_middleware(request: Request, call_next):
    request_id = str(uuid.uuid4())
    structlog.contextvars.clear_contextvars()
    structlog.contextvars.bind_contextvars(
        request_id=request_id,
        path=request.url.path,
        method=request.method,
        ip=request.client.host if request.client else "unknown",
    )
    response = await call_next(request)
    response.headers["X-Request-ID"] = request_id
    return response


# ── Global exception handler ──────────────────────────────────────────────────
@app.exception_handler(Exception)
async def global_exception_handler(request: Request, exc: Exception):
    req_id = request.headers.get("X-Request-ID", "unknown")
    log.error("unhandled_exception",
              path=request.url.path,
              error=str(exc),
              request_id=req_id,
              exc_info=True)
    return JSONResponse(
        status_code=500,
        content={
            "error": "Internal server error",
            "request_id": req_id,
        }
    )


# ── Routers (prefixed and root mounted) ────────────────────────────────────────
app.include_router(health_router, prefix="/api/v1")
app.include_router(auth_router, prefix="/api/v1")
app.include_router(analytics_router, prefix="/api/v1")
app.include_router(audit_router, prefix="/api/v1")
app.include_router(retention_router, prefix="/api/v1")
app.include_router(stream_router, prefix="/api/v1")
app.include_router(feedback_router, prefix="/api/v1")

# Also mount at root
app.include_router(health_router)
app.include_router(auth_router)
app.include_router(analytics_router)
app.include_router(audit_router)
app.include_router(retention_router)
app.include_router(stream_router)
app.include_router(feedback_router)
