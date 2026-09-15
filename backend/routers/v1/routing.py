"""backend/routers/v1/routing.py â€” Day 14 alert routing API.

  POST /alerts/{alert_id}/ack
  GET  /routing/active
  GET  /routing/unrouted
  POST /officers/{officer_id}/available
  GET  /officers

ACK SEMANTICS
  200 {"status":"acknowledged"}          transitioned ROUTED/ESCALATED_ROUTED â†’ ACKNOWLEDGED
  200 {"status":"already_acknowledged"}  idempotent replay; nothing written
  409 {"status":"cannot_ack", ...}       UNROUTED/ESCALATED_UNROUTED â€” no officer
                                          was ever assigned, so there is nothing
                                          to acknowledge and no revert to attempt
  404                                    no routed_alerts row for this alert

The ACK SELECT deliberately carries NO status filter: it fetches the current
row whatever state it is in, so the idempotency and terminal-state branches
can actually observe it. Filtering on ROUTED in the SELECT would make an
already-acknowledged alert indistinguishable from a nonexistent one.

NOTE: no `from __future__ import annotations` â€” matches the other routers,
which omit it because FastAPI resolves annotations at runtime.
"""
from typing import Any, Optional

from fastapi import APIRouter, Depends, HTTPException, Request, status as http_status
from fastapi.responses import JSONResponse
from sqlalchemy import text
from sqlalchemy.orm import Session

from backend.auth.dependencies import get_current_user
from backend.core.config import settings
from backend.core.logging import get_logger
from backend.core.rate_limit import GENERAL_LIMIT, limiter
from backend.db.models import User
from backend.db.session import get_db
from backend.routing.events import broadcast_ack, broadcast_officer_status
from backend.routing.service import revert_officer_on_ack
from backend.routing.utils import seconds_elapsed_since, to_iso8601

router = APIRouter(tags=["routing"])
logger = get_logger(__name__)

ACTIVE_STATES = ("PENDING", "ROUTED", "ESCALATED_ROUTED")
UNROUTED_STATES = ("UNROUTED", "ESCALATED_UNROUTED")


def _serialize_routed(row: Any) -> dict:
    return {
        "alert_id": row["alert_id"],
        "routed_alert_id": row["id"],
        "status": row["status"],
        "officer_id": row["assigned_officer"],
        "officer_name": row["officer_name"],
        "assigned_at": to_iso8601(row["assigned_at"]),
        "escalated_at": to_iso8601(row["escalated_at"]),
        "ack_at": to_iso8601(row["ack_at"]),
        "escalation_note": row["escalation_note"],
        "escalation_count": row["escalation_count"],
        "created_at": to_iso8601(row["created_at"]),
        "timeout_seconds": settings.ACK_TIMEOUT_SECONDS,
        # Server truth. The client interpolates between polls with a local
        # setInterval, but re-seeds from this on mount and on reconnect so a
        # page reload resumes the ring at the right position instead of
        # restarting the countdown.
        "seconds_elapsed": seconds_elapsed_since(row["assigned_at"]),
    }


def _query_by_states(db: Session, states: tuple) -> list[dict]:
    placeholders = ", ".join(f":s{i}" for i in range(len(states)))
    params = {f"s{i}": s for i, s in enumerate(states)}
    rows = db.execute(text(f"""
        SELECT ra.*, o.name AS officer_name
        FROM   routed_alerts ra
        LEFT   JOIN officers o ON o.id = ra.assigned_officer
        WHERE  ra.status IN ({placeholders})
        ORDER  BY ra.created_at DESC
    """), params).mappings().fetchall()
    return [_serialize_routed(r) for r in rows]


@router.post("/alerts/{alert_id}/ack")
@limiter.limit(GENERAL_LIMIT)
async def ack_alert(
    alert_id: str,
    request: Request,
    db: Session = Depends(get_db),
    current_user: User = Depends(get_current_user),
) -> Any:
    row = db.execute(text(
        "SELECT * FROM routed_alerts WHERE alert_id = :aid"
    ), {"aid": alert_id}).mappings().fetchone()

    if row is None:
        raise HTTPException(
            status_code=http_status.HTTP_404_NOT_FOUND,
            detail=f"Alert {alert_id} has no routing record.",
        )

    if row["status"] == "ACKNOWLEDGED":
        # Idempotent replay: change nothing, write nothing, log nothing.
        return {"status": "already_acknowledged"}

    if row["status"] in UNROUTED_STATES:
        return JSONResponse(
            status_code=http_status.HTTP_409_CONFLICT,
            content={
                "status": "cannot_ack",
                "reason": "alert was never routed to an officer",
            },
        )

    ra_id = row["id"]
    officer_id = row["assigned_officer"]

    result = db.execute(text("""
        UPDATE routed_alerts
        SET    status = 'ACKNOWLEDGED',
               ack_at = datetime('now')
        WHERE  alert_id = :aid
        AND    status   IN ('ROUTED', 'ESCALATED_ROUTED')
    """), {"aid": alert_id})
    db.commit()

    if result.rowcount == 0:
        # Escalation or another ACK landed in between. Re-read and report the
        # true state rather than claiming an acknowledgement that didn't happen.
        current = db.execute(text(
            "SELECT status FROM routed_alerts WHERE alert_id = :aid"
        ), {"aid": alert_id}).mappings().fetchone()
        if current and current["status"] == "ACKNOWLEDGED":
            return {"status": "already_acknowledged"}
        return JSONResponse(
            status_code=http_status.HTTP_409_CONFLICT,
            content={"status": "cannot_ack",
                     "reason": f"alert is in state {current['status'] if current else 'unknown'}"},
        )

    if officer_id:
        revert_officer_on_ack(officer_id, ra_id, db)
        await broadcast_officer_status(officer_id, "AVAILABLE", None)

    ack_row = db.execute(text(
        "SELECT ack_at FROM routed_alerts WHERE id = :ra"
    ), {"ra": ra_id}).mappings().fetchone()
    await broadcast_ack(alert_id, ra_id,
                        ack_row["ack_at"] if ack_row else None, officer_id)

    logger.info("Alert %d acknowledged (ra_id=%d) by user %s",
                alert_id, ra_id, current_user.id)
    return {"status": "acknowledged"}


@router.get("/routing/active")
@limiter.limit(GENERAL_LIMIT)
async def get_active_routing(
    request: Request,
    db: Session = Depends(get_db),
    current_user: User = Depends(get_current_user),
) -> Any:
    return _query_by_states(db, ACTIVE_STATES)


@router.get("/routing/unrouted")
@limiter.limit(GENERAL_LIMIT)
async def get_unrouted(
    request: Request,
    db: Session = Depends(get_db),
    current_user: User = Depends(get_current_user),
) -> Any:
    return _query_by_states(db, UNROUTED_STATES)


@router.post("/officers/{officer_id}/available")
@limiter.limit(GENERAL_LIMIT)
async def mark_officer_available(
    officer_id: int,
    request: Request,
    db: Session = Depends(get_db),
    current_user: User = Depends(get_current_user),
) -> Any:
    exists = db.execute(text("SELECT id FROM officers WHERE id = :oid"),
                        {"oid": officer_id}).fetchone()
    if exists is None:
        raise HTTPException(status_code=404, detail=f"Officer {officer_id} not found.")

    db.execute(text("""
        UPDATE officers
        SET    status           = 'AVAILABLE',
               current_alert_id = NULL,
               last_updated     = datetime('now')
        WHERE  id = :oid
    """), {"oid": officer_id})
    db.commit()

    await broadcast_officer_status(officer_id, "AVAILABLE", None)
    return {"status": "ok"}


@router.get("/routing/dispatch-readiness")
@limiter.limit(GENERAL_LIMIT)
async def dispatch_readiness(
    request: Request,
    current_user: User = Depends(get_current_user),
) -> Any:
    """Whether auto-dispatch is active for face matches, and why not if not.

    Exposed so the dashboard can say plainly that watchlist matches are not
    auto-dispatching, rather than leaving an operator to infer it from the
    absence of a routing badge â€” an absence that looks identical to "no
    officer was available".
    """
    from backend.services.threshold_validation import face_threshold_validation_state

    state = face_threshold_validation_state()
    interlock_on = settings.REQUIRE_VALIDATED_FACE_THRESHOLD_FOR_DISPATCH
    return {
        "face_match_auto_dispatch_enabled": bool(state.validated or not interlock_on),
        "interlock_enabled": interlock_on,
        "face_threshold": settings.WATCHLIST_FACE_MATCH_THRESHOLD,
        "validation": state.as_dict(),
        "note": (
            "Face-match alerts still fire, score, capture evidence and reach "
            "the review queue. Only AUTOMATIC officer dispatch is withheld "
            "while the matching threshold is unvalidated."
        ) if (interlock_on and not state.validated) else None,
    }


@router.get("/officers")
@limiter.limit(GENERAL_LIMIT)
async def list_officers(
    request: Request,
    db: Session = Depends(get_db),
    current_user: User = Depends(get_current_user),
) -> Any:
    # ORDER BY name ASC is load-bearing, not cosmetic: without a stable sort
    # the OfficerPanel visually reshuffles on every officer.status event.
    rows = db.execute(text("""
        SELECT id, name, lat, lng, status, current_alert_id, last_updated
        FROM   officers
        ORDER  BY name ASC
    """)).mappings().fetchall()
    return [
        {
            "id": int(r["id"]) if (r["id"] is not None and str(r["id"]).isdigit()) else r["id"],
            "name": r["name"],
            "lat": r["lat"],
            "lng": r["lng"],
            "status": r["status"],
            "current_alert_id": r["current_alert_id"],
            "last_updated": to_iso8601(r["last_updated"]),
        }
        for r in rows
    ]
