"""
backend/routing/service.py — nearest-officer dispatch (Day 14).

THREE-PHASE ATOMIC ASSIGNMENT
  Never SELECT a nearest officer and then UPDATE to claim them: the gap
  between those two statements is exactly where two concurrent CRITICAL
  alerts both decide the same officer is free. The claim is instead a
  conditional UPDATE carrying `AND status = 'AVAILABLE'`, and the winner is
  identified by rowcount == 1. Losers fall through to the next candidate.

  Phase 0  read camera coords + AVAILABLE candidates   (own transaction)
  Phase 1  insert routed_alerts row as PENDING          (own transaction)
  Phase 2  attempt atomic claim, nearest-first          (own transaction each)
  Phase 3  finalise to ROUTED or UNROUTED               (own transaction)

TRANSACTION HANDLING — why this uses execute()+commit(), not `with db.begin():`
  route_alert operates on the CALLER'S Session, and it does not control that
  session's transaction state. The real caller is
  evidence_capture.on_alert_fired(), which does `alert.severity = ...;
  db.flush()` immediately beforehand — and flush() opens a transaction. A
  `with db.begin():` on top of that raises
  `InvalidRequestError: A transaction is already begun on this Session`
  (verified against SQLAlchemy 2.0.23 here), which would have broken routing
  on every CRITICAL alert in production, not just in tests.

  So each phase is `db.execute(...)` followed by an explicit `db.commit()`,
  which works whether or not the caller left a transaction open. Committing
  the caller's in-flight Alert is intended: we are about to dispatch an
  officer to that alert, so it must be durable first. evidence_capture's
  create_pending_row() already commits this same session moments later.

  The escalation tick is different — it owns a fresh Session created inside
  its worker thread, so it can and does use `with db.begin():` blocks.

SCHEMA BINDING NOTE
  Camera coordinates live in `cameras.gps_lat` / `cameras.gps_lon` in this
  codebase (not lat/lng), and `alerts.camera_id` is the integer FK to
  `cameras.id`. Coordinates are resolved by join at routing time and are
  deliberately NOT denormalised onto the alert row.

  Severity values written by this system are LOWERCASE ('critical'), set by
  evidence_capture.classify_severity(). The predicate below is
  case-insensitive so it matches both those and the legacy raw-SQL ANPR path,
  which writes its own 'critical'.
"""
from __future__ import annotations

import logging
from typing import Any

from sqlalchemy import text
from sqlalchemy.orm import Session

from backend.routing.utils import haversine_km

logger = logging.getLogger(__name__)

STATUS_PENDING = "PENDING"
STATUS_ROUTED = "ROUTED"
STATUS_UNROUTED = "UNROUTED"
STATUS_ACKNOWLEDGED = "ACKNOWLEDGED"
STATUS_ESCALATED_ROUTED = "ESCALATED_ROUTED"
STATUS_ESCALATED_UNROUTED = "ESCALATED_UNROUTED"

TERMINAL_STATES = (STATUS_ACKNOWLEDGED, STATUS_ESCALATED_UNROUTED, STATUS_UNROUTED)
ESCALATABLE_STATES = (STATUS_ROUTED, STATUS_ESCALATED_ROUTED)


def is_critical_severity(severity: str | None) -> bool:
    """Case-insensitive CRITICAL test.

    classify_severity() writes 'critical'; the legacy ANPR path also writes
    'critical'. A literal `severity = 'CRITICAL'` comparison would match
    neither, and would silently route nothing at all.
    """
    return (severity or "").strip().upper() == "CRITICAL"


def route_alert(alert_id: int, db: Session) -> dict[str, Any] | None:
    """Assign the nearest AVAILABLE officer to a CRITICAL alert.

    Returns the final routed_alerts row as a dict, or None if the alert or
    its camera could not be resolved.

    Idempotent: an alert already routed returns its existing row rather than
    creating a second dispatch for the same event.
    """
    # ── Guard: already routed? ──────────────────────────────────────────────
    existing = db.execute(text(
        "SELECT * FROM routed_alerts WHERE alert_id = :aid"
    ), {"aid": alert_id}).mappings().fetchone()
    if existing:
        logger.info("Alert %d already routed (ra_id=%s, status=%s) — no re-route.",
                    alert_id, existing["id"], existing["status"])
        return dict(existing)

    # ── Phase 0: camera coords + AVAILABLE candidates ───────────────────────
    row = db.execute(text("""
        SELECT c.gps_lat AS lat, c.gps_lon AS lng
        FROM   alerts a
        JOIN   cameras c ON c.id = a.camera_id
        WHERE  a.id = :alert_id
    """), {"alert_id": alert_id}).fetchone()

    if row is None:
        logger.warning(
            "Alert %d has no resolvable camera — cannot route. "
            "(alerts.camera_id is NULL, or points at a missing camera.)",
            alert_id,
        )
        return None

    cam_lat, cam_lng = row.lat, row.lng

    candidates = db.execute(text("""
        SELECT id, lat, lng
        FROM   officers
        WHERE  status = 'AVAILABLE'
    """)).fetchall()

    if cam_lat is None or cam_lng is None:
        # A registered camera with no GPS fix. Routing by distance is
        # meaningless here, so this resolves to UNROUTED rather than
        # silently dispatching to an arbitrary officer.
        logger.warning("Camera for alert %d has no GPS coordinates — UNROUTED.", alert_id)
        candidates = []
        sorted_candidates: list[Any] = []
    else:
        sorted_candidates = sorted(
            candidates,
            key=lambda o: haversine_km(cam_lat, cam_lng, o.lat, o.lng),
        )

    # ── Phase 1: insert PENDING row ─────────────────────────────────────────
    result = db.execute(text("""
        INSERT INTO routed_alerts (alert_id, status, escalation_count, created_at)
        VALUES (:alert_id, 'PENDING', 0, datetime('now'))
    """), {"alert_id": alert_id})
    ra_id = result.lastrowid
    db.commit()

    # ── Phase 2: atomic claim, nearest first ────────────────────────────────
    assigned_officer_id: int | None = None
    for candidate in sorted_candidates:
        claim = db.execute(text("""
            UPDATE officers
            SET    status           = 'BUSY',
                   current_alert_id = :ra_id,
                   last_updated     = datetime('now')
            WHERE  id     = :oid
            AND    status = 'AVAILABLE'
        """), {"ra_id": ra_id, "oid": candidate.id})
        db.commit()
        # rowcount == 1 → we won the race. 0 → someone else claimed them
        # between Phase 0's read and now; try the next-nearest.
        if claim.rowcount == 1:
            assigned_officer_id = candidate.id
            break

    # ── Phase 3: finalise — PENDING never survives this function ────────────
    if assigned_officer_id:
        db.execute(text("""
            UPDATE routed_alerts
            SET    status           = 'ROUTED',
                   assigned_officer = :officer_id,
                   assigned_at      = datetime('now')
            WHERE  id     = :ra_id
            AND    status = 'PENDING'
        """), {"officer_id": assigned_officer_id, "ra_id": ra_id})
    else:
        db.execute(text("""
            UPDATE routed_alerts
            SET    status          = 'UNROUTED',
                   escalation_note = 'No officers available at routing time'
            WHERE  id     = :ra_id
            AND    status = 'PENDING'
        """), {"ra_id": ra_id})
    db.commit()

    final = db.execute(text(
        "SELECT * FROM routed_alerts WHERE id = :id"
    ), {"id": ra_id}).mappings().fetchone()
    db.commit()

    return dict(final) if final else None


def revert_officer_on_ack(officer_id: int, ra_id: int, db: Session) -> None:
    """Free an officer on ACK — but only if they are still on THIS alert.

    The `AND current_alert_id = :ra_id` predicate is what stops a late ACK on
    alert A from stealing an officer who has since been dispatched to alert B.
    rowcount == 0 means they were already reassigned or freed; that is a
    correct no-op, not an error.
    """
    db.execute(text("""
        UPDATE officers
        SET    status           = 'AVAILABLE',
               current_alert_id = NULL,
               last_updated     = datetime('now')
        WHERE  id               = :officer_id
        AND    status           = 'BUSY'
        AND    current_alert_id = :ra_id
    """), {"officer_id": officer_id, "ra_id": ra_id})
    db.commit()
