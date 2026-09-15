# backend/routers/v1/feedback.py

import logging
from collections import defaultdict
from datetime import datetime, timezone
from time import time
from typing import Any

from fastapi import APIRouter, Depends, HTTPException, status
from pydantic import BaseModel
from sqlalchemy import text
from sqlalchemy.orm import Session

try:
    from backend.auth.dependencies import require_officer_auth
    from backend.db.session import get_db
    from backend.core.config import settings
    from backend.feedback.service import submit_feedback, ALLOWED_VERDICTS
    from backend.ws.dashboard_ws import broadcast
except ImportError:
    from sentinel.auth import require_officer_auth
    from sentinel.db import get_db
    from sentinel.config import settings
    from sentinel.feedback.service import submit_feedback, ALLOWED_VERDICTS
    from sentinel.websocket import broadcast

logger = logging.getLogger(__name__)
router = APIRouter(tags=["feedback"])

# â”€â”€ Module-level startup timestamp â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€
APP_START_TIME: datetime = datetime.now(timezone.utc)


# â”€â”€ Pydantic models â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€

class FeedbackBody(BaseModel):
    verdict: str

class ResetBody(BaseModel):
    zone_name: str


# â”€â”€ Rate limiter â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€
_feedback_rate: dict[Any, list[float]] = defaultdict(list)

def _check_rate_limit(officer_id: Any, max_per_minute: int = 30) -> bool:
    """Return True if under limit. False if rate exceeded."""
    if not officer_id:
        return True
    now = time()
    timestamps = _feedback_rate[officer_id]
    _feedback_rate[officer_id] = [t for t in timestamps if now - t < 60]
    if len(_feedback_rate[officer_id]) >= max_per_minute:
        return False
    _feedback_rate[officer_id].append(now)
    return True


# â”€â”€ POST /alerts/{alert_id}/feedback â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€

@router.post("/alerts/{alert_id}/feedback")
async def submit_alert_feedback(
    alert_id: Any,
    body: FeedbackBody,
    db: Session = Depends(get_db),
    officer=Depends(require_officer_auth),
):
    """
    Upsert feedback verdict for an alert.
    Broadcasts feedback.logged WS event after successful commit.
    """
    # â”€â”€ Rate limit â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€
    officer_id = getattr(officer, "id", None) if officer else None
    if officer_id and not _check_rate_limit(officer_id):
        raise HTTPException(
            status_code=status.HTTP_429_TOO_MANY_REQUESTS,
            detail={"error": "Too many feedback submissions. Wait 60s."}
        )

    # â”€â”€ Verdict validation â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€
    if body.verdict not in ALLOWED_VERDICTS:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail={
                "error": "Invalid verdict",
                "allowed": sorted(ALLOWED_VERDICTS),
            }
        )

    # â”€â”€ Alert existence check â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€
    alert_exists = db.execute(
        text("SELECT id FROM alerts WHERE id = :id"),
        {"id": str(alert_id)}
    ).fetchone()
    if not alert_exists and str(alert_id).isdigit():
        alert_exists = db.execute(
            text("SELECT id FROM alerts WHERE id = :id"),
            {"id": int(alert_id)}
        ).fetchone()

    if not alert_exists:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail={"error": f"Alert {alert_id} not found"}
        )

    # â”€â”€ Service call + commit â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€
    try:
        payload = submit_feedback(alert_id, body.verdict, officer_id, db)
        db.commit()
    except ValueError as e:
        db.rollback()
        raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail={"error": str(e)})
    except Exception as e:
        db.rollback()
        logger.error(f"[Feedback] Upsert failed for alert {alert_id}: {e}", exc_info=True)
        raise HTTPException(status_code=status.HTTP_500_INTERNAL_SERVER_ERROR, detail={"error": "Feedback submission failed"})

    # â”€â”€ WebSocket broadcast (AFTER commit) â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€
    zs = payload["zone_summary"]
    try:
        await broadcast({
            "type":              "feedback.logged",
            "alert_id":          payload["alert_id"],
            "verdict":           payload["verdict"],
            "officer_id":        payload["officer_id"],
            "officer_name":      payload["officer_name"],
            "zone_name":         zs["zone_name"],
            "false_alarm_count": zs["false_alarm_count"],
            "threshold_flag":    zs["threshold_flag"],
            "flag_message":      zs["flag_message"],
        })
    except Exception as exc:
        logger.debug("Failed to broadcast feedback.logged event: %s", exc)

    return payload


# â”€â”€ GET /feedback/summary â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€

@router.get("/feedback/summary")
async def feedback_summary(
    since: str = "today",
    db: Session = Depends(get_db),
    officer=Depends(require_officer_auth),
):
    """
    Returns verdict counts grouped by zone.
    ?since=today    â€” counts since midnight UTC (default)
    ?since=session  â€” counts since app startup
    ?since=all      â€” all-time counts
    """
    if since == "today":
        midnight = datetime.now(timezone.utc).replace(hour=0, minute=0, second=0, microsecond=0)
        time_filter = "af.created_at >= :midnight"
        params = {"midnight": midnight}
        since_label = midnight.isoformat()

    elif since == "session":
        time_filter = "af.created_at >= :start_time"
        params = {
            "start_time": APP_START_TIME.strftime("%Y-%m-%d %H:%M:%S")
        }
        since_label = APP_START_TIME.isoformat()

    elif since == "all":
        time_filter = "1=1"
        params = {}
        since_label = "all-time"

    else:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail={
                "error": "Invalid 'since' value",
                "allowed": ["today", "session", "all"],
            }
        )

    rows = db.execute(text(f"""
        SELECT
            COALESCE(c.location_label, c.zone, 'Unzoned')                AS zone_name,
            SUM(CASE WHEN af.verdict='FALSE_ALARM'   THEN 1 ELSE 0 END)  AS false_alarm_count,
            SUM(CASE WHEN af.verdict='GENUINE'        THEN 1 ELSE 0 END)  AS genuine_count,
            SUM(CASE WHEN af.verdict='INVESTIGATING'  THEN 1 ELSE 0 END)  AS investigating_count
        FROM   alert_feedback af
        JOIN   alerts  a ON a.id = af.alert_id
        JOIN   cameras c ON c.id = a.camera_id
        WHERE  {time_filter}
        GROUP  BY COALESCE(c.location_label, c.zone, 'Unzoned')
        ORDER  BY false_alarm_count DESC
    """), params).fetchall()

    threshold = getattr(settings, "FALSE_ALARM_FLAG_THRESHOLD", 3)
    summary = []

    for row in rows:
        fa_count = int(row.false_alarm_count or 0)
        threshold_flag = fa_count >= threshold
        summary.append({
            "zone_name":           row.zone_name,
            "false_alarm_count":   fa_count,
            "genuine_count":       int(row.genuine_count or 0),
            "investigating_count": int(row.investigating_count or 0),
            "threshold_flag":      threshold_flag,
            "flag_message": (
                f"{fa_count} false alarms logged â€” "
                f"loitering threshold for {row.zone_name} flagged for review"
            ) if threshold_flag else None,
        })

    return {
        "since":     since_label,
        "threshold": threshold,
        "summary":   summary,
    }


# â”€â”€ POST /feedback/reset â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€

@router.post("/feedback/reset")
async def reset_feedback_flag(
    body: ResetBody,
    db: Session = Depends(get_db),
    officer=Depends(require_officer_auth),
):
    """
    Clears the System Learning banner for a zone by setting
    feedback_flag_log.resolved = 1 on open flags.
    """
    # â”€â”€ Admin guard â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€
    is_admin = False
    if officer:
        role = getattr(officer, "role", "")
        officer_id = getattr(officer, "id", None)
        if str(role).lower() == "admin" or str(officer_id) in ("1", "USER_ADMIN_01"):
            is_admin = True
    else:
        is_admin = True  # Demo unauthenticated fallback

    if not is_admin:
        raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN,
            detail={
                "error": "Admin access required",
                "hint":  "Only admin or officer ID 1 can reset flags",
            }
        )

    # â”€â”€ Verify zone exists in feedback log â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€
    zone_exists = db.execute(text("""
        SELECT 1 FROM feedback_flag_log
        WHERE zone_name = :zone_name
        LIMIT 1
    """), {"zone_name": body.zone_name}).fetchone()

    if not zone_exists:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail={
                "error":     f"No flag log found for zone: {body.zone_name}",
                "zone_name": body.zone_name,
            }
        )

    # â”€â”€ Resolve flags â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€
    result = db.execute(text("""
        UPDATE feedback_flag_log
        SET    resolved = 1
        WHERE  zone_name = :zone_name
        AND    resolved  = 0
    """), {"zone_name": body.zone_name})
    db.commit()

    return {
        "cleared_flags": result.rowcount,
        "zone_name":     body.zone_name,
        "note":          "Feedback history preserved. Flag cleared for display only.",
    }
