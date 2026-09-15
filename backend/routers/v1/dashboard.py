"""backend/routers/v1/dashboard.py — Day 18 control-room support endpoints.

  GET  /alerts/summary            severity counts for the summary panel
  GET  /reid/review-queue/count   pending review count for the queue badge
  POST /client-error              client-side panel crash reports

SEVERITY CASING
  This system writes severity in LOWERCASE ('critical'), set by
  evidence_capture.classify_severity(). The summary groups case-insensitively
  and returns UPPERCASE keys, because the dashboard's colour map is keyed on
  CRITICAL/HIGH/MEDIUM/LOW. Grouping on the raw column would split 'critical'
  and 'CRITICAL' into two buckets the moment any other writer disagreed —
  and the legacy raw-SQL ANPR path is exactly such a writer.

  Alerts with severity NULL are counted under "UNSCORED" rather than dropped.
  Silently omitting them would make the panel's total disagree with the alert
  feed, and the first person to notice would be an operator, mid-demo.

NOTE: no `from __future__ import annotations` — matches the other routers.
"""
from datetime import datetime, timedelta, timezone
from typing import Any, Optional

from fastapi import APIRouter, Depends, Query, Request
from pydantic import BaseModel
from sqlalchemy import text
from sqlalchemy.orm import Session

from backend.auth.dependencies import get_current_user
from backend.core.logging import get_logger
from backend.core.rate_limit import GENERAL_LIMIT, limiter
from backend.db.models import User
from backend.db.session import get_db

router = APIRouter(tags=["dashboard"])
logger = get_logger(__name__)

_KNOWN = ("CRITICAL", "HIGH", "MEDIUM", "LOW")


@router.get("/alerts/summary")
@limiter.limit(GENERAL_LIMIT)
async def alerts_summary(
    request: Request,
    since: str = Query("today", pattern="^(today|session|all)$"),
    db: Session = Depends(get_db),
    current_user: User = Depends(get_current_user),
) -> Any:
    if since == "today":
        cutoff = datetime.now(timezone.utc).replace(
            hour=0, minute=0, second=0, microsecond=0, tzinfo=None)
    elif since == "session":
        # "This shift" — the last 8 hours. There is no session table to key
        # off, so this is an explicit approximation rather than an invented
        # notion of session boundaries.
        cutoff = datetime.utcnow() - timedelta(hours=8)
    else:
        cutoff = datetime(1970, 1, 1)

    rows = db.execute(text("""
        SELECT UPPER(COALESCE(severity, 'UNSCORED')) AS sev, COUNT(*) AS n
        FROM   alerts
        WHERE  is_deleted = 0
        AND    created_at >= :cutoff
        GROUP  BY UPPER(COALESCE(severity, 'UNSCORED'))
    """), {"cutoff": cutoff}).mappings().fetchall()

    counts = {k: 0 for k in _KNOWN}
    counts["UNSCORED"] = 0
    for r in rows:
        counts[r["sev"]] = counts.get(r["sev"], 0) + r["n"]

    return {
        **counts,
        "since": cutoff.replace(tzinfo=timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
        "window": since,
        "total": sum(counts.values()),
    }


@router.get("/reid/review-queue/count")
@limiter.limit(GENERAL_LIMIT)
async def review_queue_count(
    request: Request,
    db: Session = Depends(get_db),
    current_user: User = Depends(get_current_user),
) -> Any:
    """Pending count for the badge. Cheap COUNT — the badge polls on events,
    and fetching the whole queue just to length it would be an N-row transfer
    for one integer."""
    n = db.execute(text(
        "SELECT COUNT(*) FROM reid_review_items WHERE status = 'PENDING'"
    )).scalar()
    return {"pending": int(n or 0)}


class ClientErrorReport(BaseModel):
    panel: Optional[str] = None
    error: Optional[str] = None
    stack: Optional[str] = None
    component: Optional[str] = None
    timestamp: Optional[str] = None


@router.post("/client-error")
@limiter.limit("10/minute")
async def client_error(body: ClientErrorReport, request: Request) -> Any:
    """Record a panel crash from the browser.

    Unauthenticated on purpose: a panel can crash before auth state settles,
    and an error reporter that requires a valid session cannot report the one
    class of failure most worth knowing about.

    Rate-limited to 10/min per IP so a render loop cannot turn one broken
    panel into a log flood. Always returns 200 — the client must never treat
    error reporting as something that can itself fail.
    """
    logger.error(
        "[client-error] panel=%s error=%s component=%s",
        body.panel, (body.error or "")[:300], (body.component or "")[:200],
        extra={"client_ip": request.client.host if request.client else None},
    )
    return {"status": "recorded"}
