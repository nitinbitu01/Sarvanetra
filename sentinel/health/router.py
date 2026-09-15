# sentinel/health/router.py
"""
sentinel/health/router.py — Production health probes (/health, /ready, /metrics).
"""

from datetime import datetime, timezone

import structlog
from fastapi import APIRouter
from fastapi.responses import JSONResponse
from sqlalchemy import text

from sentinel.db import engine
from sentinel.stream.manager import ffmpeg_processes

log = structlog.get_logger(__name__)
router = APIRouter(tags=["health"])

_startup_time = datetime.now(timezone.utc)


@router.get("/health")
async def health_check():
    """
    Liveness probe. Returns 200 if the process is alive.
    """
    now = datetime.now(timezone.utc)
    return {
        "status": "ok",
        "timestamp": now.isoformat(),
        "uptime_seconds": (now - _startup_time).total_seconds(),
    }


@router.get("/ready")
async def readiness_check():
    """
    Readiness probe. Returns 200 only if critical dependencies are healthy.
    """
    checks = {}
    healthy = True

    # 1. Database check
    try:
        async with engine.connect() as conn:
            await conn.execute(text("SELECT 1"))
        checks["database"] = {"status": "ok"}
    except Exception as e:
        checks["database"] = {"status": "error", "detail": str(e)}
        healthy = False
        log.error("health.database_check_failed", error=str(e))

    # 2. HLS streams check
    alive_count = sum(1 for p in ffmpeg_processes.values() if p.poll() is None)
    total_count = len(ffmpeg_processes)
    checks["hls"] = {
        "status": "ok" if (total_count == 0 or alive_count == total_count) else "degraded",
        "alive_streams": alive_count,
        "total_streams": total_count,
    }

    status_code = 200 if healthy else 503
    return JSONResponse(
        status_code=status_code,
        content={
            "status": "ok" if healthy else "degraded",
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "checks": checks,
        }
    )


@router.get("/metrics")
async def metrics():
    """
    Lightweight metrics endpoint for monitoring tools.
    """
    alive_streams = sum(1 for p in ffmpeg_processes.values() if p.poll() is None)
    now = datetime.now(timezone.utc)

    return {
        "timestamp": now.isoformat(),
        "uptime_seconds": (now - _startup_time).total_seconds(),
        "hls": {
            "alive_streams": alive_streams,
            "total_tracked": len(ffmpeg_processes),
        },
    }
