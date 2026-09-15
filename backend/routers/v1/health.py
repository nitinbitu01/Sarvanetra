"""backend/routers/v1/health.py — GET /api/v1/health"""
from __future__ import annotations

from datetime import datetime, timezone

from fastapi import APIRouter
from fastapi.responses import JSONResponse

from backend.core.config import settings
from backend.db.session import check_db_connection

router = APIRouter(tags=["health"])


@router.get("/health")
async def health_check():
    """System health check — no auth required.

    Returns 200 with status:'ok' if all components are healthy,
    or 503 with status:'degraded' listing the failing component(s).
    """
    checks: dict[str, str] = {}

    # 1. Database
    checks["db"] = "ok" if check_db_connection() else "error"

    # 2. WebSocket manager — just checks the module can be imported
    try:
        from backend import main as _m  # noqa: F401
        checks["websocket"] = "ok"
    except Exception:
        checks["websocket"] = "error"

    overall = "ok" if all(v == "ok" for v in checks.values()) else "degraded"
    payload = {
        "status": overall,
        **checks,
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "env": settings.ENV,
    }
    status_code = 200 if overall == "ok" else 503
    return JSONResponse(content=payload, status_code=status_code)
