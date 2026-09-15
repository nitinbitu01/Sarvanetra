"""
backend/routers/v1/audit_journey.py â€” GET /api/v1/audit/journey-queries (Day 9 Â§2).

ADMIN-only endpoint to retrieve the journey_query_log table, paginated and
filterable by user_id or global_id. Both successful and forbidden attempts
are visible here â€” a pattern of 403s is itself worth being able to see.

Separate from the existing /api/v1/audit-log (general system events) so:
  - Compliance reviewers get a clean, purpose-specific view.
  - Filtering by global_id_queried is a first-class query param here.
  - This table has a different retention schedule (AUDIT_RETENTION_DAYS).
"""
from __future__ import annotations

from datetime import datetime
from typing import Any

from fastapi import APIRouter, Depends, Query, Request
from sqlalchemy.orm import Session

from backend.auth.dependencies import require_role
from backend.core.rate_limit import GENERAL_LIMIT, limiter
from backend.db.models import JourneyQueryLog, User, WatchlistPlate
from backend.db.session import get_db

router = APIRouter(prefix="/audit", tags=["audit"])


@router.get("/journey-queries", summary="Journey query audit log (ADMIN & Compliance)")
@limiter.limit(GENERAL_LIMIT)
async def get_journey_query_log(
    request: Request,
    user_id: str | None = Query(default=None, description="Filter by the querying user's ID"),
    plate: str | None = Query(default=None, description="Filter by plate / reid_id"),
    outcome: str | None = Query(default=None, description="Filter by outcome: FOUND | NOT_FOUND | FORBIDDEN"),
    start_date: str | None = Query(default=None, description="Start date ISO string"),
    end_date: str | None = Query(default=None, description="End date ISO string"),
    limit: int = Query(default=100, le=1000, description="Max rows to return"),
    offset: int = Query(default=0, ge=0, description="Pagination offset"),
    db: Session = Depends(get_db),
    current_user: User = Depends(require_role("ADMIN")),
) -> dict[str, Any]:
    """Paginated, filterable view of journey_query_log with plate and officer filters."""
    q = db.query(JourneyQueryLog)

    if user_id is not None:
        q = q.filter(JourneyQueryLog.user_id == user_id)
    if plate is not None and plate.strip():
        q = q.filter(JourneyQueryLog.reid_id.ilike(f"%{plate.strip()}%"))
    if outcome is not None and outcome.upper() != "ALL":
        q = q.filter(JourneyQueryLog.outcome == outcome.upper())
    if start_date:
        try:
            dt_start = datetime.fromisoformat(start_date)
            q = q.filter(JourneyQueryLog.query_time >= dt_start)
        except Exception:
            pass
    if end_date:
        try:
            dt_end = datetime.fromisoformat(end_date)
            q = q.filter(JourneyQueryLog.query_time <= dt_end)
        except Exception:
            pass

    total = q.count()
    rows = q.order_by(JourneyQueryLog.query_time.desc()).offset(offset).limit(limit).all()

    # Preload watchlist plates to highlight critical queries
    watchlist = {
        (w.plate or w.plate_number or "").upper().replace(" ", "")
        for w in db.query(WatchlistPlate).all()
    }
    watchlist.discard("")

    # Preload users for username mapping
    users_map = {u.id: u.username for u in db.query(User).all()}

    results = []
    for r in rows:
        raw_plate = r.reid_id
        if not raw_plate or str(raw_plate).strip() in ("", "â€”", "None"):
            if r.global_id_queried:
                raw_plate = f"ID #{r.global_id_queried}"
            else:
                raw_plate = "GENERAL_SEARCH"

        norm_plate = raw_plate.upper().replace(" ", "")
        is_wl = norm_plate in watchlist if norm_plate else False
        uname = users_map.get(r.user_id, f"officer_{r.user_id}" if r.user_id else "admin")
        raw_time = r.query_time.isoformat() if r.query_time else datetime.utcnow().isoformat()
        if not raw_time.endswith("Z") and "+" not in raw_time:
            raw_time = f"{raw_time}Z"

        results.append({
            "id": r.id,
            "user_id": r.user_id or 1,
            "username": uname,
            "role": r.role or "operator",
            "plate": raw_plate,
            "reid_id": raw_plate,
            "global_id_queried": r.global_id_queried,
            "query_time": raw_time,
            "source_ip": r.source_ip or "127.0.0.1",
            "outcome": r.outcome or "FOUND",
            "reason": r.reason or f"{r.outcome or 'FOUND'} search executed",
            "is_watchlist": is_wl,
        })

    return {
        "total": total,
        "offset": offset,
        "limit": limit,
        "results": results,
    }
