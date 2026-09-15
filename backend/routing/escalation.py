"""
backend/routing/escalation.py — ACK-timeout escalation tick (Day 14).

WHY THE TICK IS ASYNC HERE
  Contract 12 assumed the tick runs in "APScheduler's thread pool". This
  codebase uses AsyncIOScheduler (backend/main.py lifespan), which runs jobs
  ON THE EVENT LOOP. A synchronous tick doing blocking SQLite I/O every 5
  seconds would stall every WebSocket client, the detection drain loop, and
  every in-flight HTTP request for the duration of the scan.

  So the entry point is `async def`, and all blocking DB work is pushed to a
  worker thread via asyncio.to_thread. The Session is created INSIDE that
  thread and closed there — Sessions are not thread-safe and must never be
  shared across the boundary.

RESTART SURVIVAL
  All timing derives from routed_alerts.assigned_at in the database. There is
  no in-memory timer, so a server restart mid-countdown loses nothing: the
  next tick recomputes overdue-ness from the stored timestamp.

THE TWO RACES THIS HAS TO WIN
  1. ACK vs escalation — both are conditional UPDATEs on mutually exclusive
     preconditions. Escalation carries `AND escalation_count = 0 AND status
     IN ('ROUTED','ESCALATED_ROUTED')`; ACK sets status='ACKNOWLEDGED'. If ACK
     lands between this tick's SELECT and its UPDATE, the UPDATE matches zero
     rows and escalation backs out — including releasing any officer it had
     already speculatively claimed.
  2. Double escalation — `escalation_count = 0` in the WHERE clause, flipped
     to 1 in the same UPDATE. The cap is enforced by the database, not by
     application logic, so it holds regardless of worker count.

THE STRANDED-OFFICER BUG THIS AVOIDS
  Whenever an alert leaves an officer behind, that officer is explicitly
  reverted to AVAILABLE — in BOTH the found-a-new-officer branch and the
  no-officer-found branch. Without the second one, a single missed ACK
  strands the original officer as BUSY on a dead alert forever, permanently
  shrinking the dispatch pool.
"""
from __future__ import annotations

import asyncio
import logging
from typing import Any

from sqlalchemy import text
from sqlalchemy.orm import Session

from backend.routing.events import broadcast_escalated, broadcast_officer_status
from backend.routing.utils import haversine_km

logger = logging.getLogger(__name__)


async def escalation_tick() -> None:
    """APScheduler entry point. Never raises — the job must keep rescheduling."""
    try:
        results = await asyncio.to_thread(_run_escalation_tick_sync)
    except Exception as exc:
        logger.error("[escalation_tick] failed: %s", exc, exc_info=True)
        return

    # Broadcasts happen back on the event loop, after all DB work is done and
    # committed. Emitting from inside the worker thread would touch the
    # WebSocket client set from the wrong thread.
    for ev in results:
        try:
            if ev["kind"] == "escalated":
                await broadcast_escalated(
                    ev["alert_id"], ev["ra_id"], ev["new_officer_id"],
                    ev["new_officer_name"], ev["new_assigned_at"], ev["final_status"],
                )
            elif ev["kind"] == "officer":
                await broadcast_officer_status(
                    ev["officer_id"], ev["status"], ev["current_alert_id"]
                )
        except Exception as exc:
            logger.debug("Escalation broadcast failed: %s", exc)


def _run_escalation_tick_sync() -> list[dict[str, Any]]:
    """Blocking body. Runs in a worker thread with its own Session."""
    from backend.db.session import SessionLocal

    db = SessionLocal()
    try:
        return _scan_and_escalate(db)
    except Exception:
        db.rollback()
        raise
    finally:
        db.close()


def _scan_and_escalate(db: Session) -> list[dict[str, Any]]:
    from datetime import datetime, timedelta, timezone
    from backend.core.config import settings

    # Read at tick time, never cached at startup — changing the value and
    # restarting takes effect without a code change.
    timeout = settings.ACK_TIMEOUT_SECONDS
    cutoff = datetime.now(timezone.utc) - timedelta(seconds=timeout)
    cutoff_naive = datetime.utcnow() - timedelta(seconds=timeout)

    # Own transaction block: a bare execute would leave an implicit
    # transaction open and make the next `with db.begin():` raise.
    with db.begin():
        overdue = db.execute(text("""
            SELECT ra.id              AS ra_id,
                   ra.alert_id        AS alert_id,
                   ra.assigned_officer AS assigned_officer,
                   ra.status          AS status,
                   c.gps_lat          AS cam_lat,
                   c.gps_lon          AS cam_lng
            FROM   routed_alerts ra
            JOIN   alerts  a ON a.id = ra.alert_id
            JOIN   cameras c ON c.id = a.camera_id
            WHERE  ra.status IN ('ROUTED', 'ESCALATED_ROUTED')
            AND    ra.escalation_count = 0
            AND    ra.assigned_at IS NOT NULL
            AND    (ra.assigned_at < :cutoff OR ra.assigned_at < :cutoff_naive)
        """), {"cutoff": cutoff, "cutoff_naive": cutoff_naive}).mappings().fetchall()

    events: list[dict[str, Any]] = []
    for row in overdue:
        try:
            events.extend(_escalate_one(dict(row), db))
        except Exception as exc:
            # One bad row must not abort the whole tick.
            logger.error("Escalation failed for ra_id=%s: %s", row["ra_id"], exc,
                         exc_info=True)
            db.rollback()
    return events


def _escalate_one(row: dict[str, Any], db: Session) -> list[dict[str, Any]]:
    ra_id = row["ra_id"]
    alert_id = row["alert_id"]
    previous_officer = row["assigned_officer"]
    events: list[dict[str, Any]] = []

    # ── Candidates, excluding the officer who just timed out ────────────────
    with db.begin():
        candidates = db.execute(text("""
            SELECT id, lat, lng
            FROM   officers
            WHERE  status = 'AVAILABLE'
            AND    (:current_officer IS NULL OR id != :current_officer)
        """), {"current_officer": previous_officer}).fetchall()

    cam_lat, cam_lng = row["cam_lat"], row["cam_lng"]
    if cam_lat is None or cam_lng is None:
        sorted_candidates: list[Any] = []
    else:
        sorted_candidates = sorted(
            candidates, key=lambda o: haversine_km(cam_lat, cam_lng, o.lat, o.lng)
        )

    # ── Speculatively claim the nearest available officer ───────────────────
    new_officer_id: int | None = None
    for candidate in sorted_candidates:
        with db.begin():
            claim = db.execute(text("""
                UPDATE officers
                SET    status           = 'BUSY',
                       current_alert_id = :ra_id,
                       last_updated     = datetime('now')
                WHERE  id     = :oid
                AND    status = 'AVAILABLE'
            """), {"ra_id": ra_id, "oid": candidate.id})
        if claim.rowcount == 1:
            new_officer_id = candidate.id
            break

    if new_officer_id is not None:
        with db.begin():
            result = db.execute(text("""
                UPDATE routed_alerts
                SET    status           = 'ESCALATED_ROUTED',
                       assigned_officer = :new_officer_id,
                       assigned_at      = datetime('now'),
                       escalated_at     = datetime('now'),
                       escalation_count = 1,
                       escalation_note  = 'Timeout: reassigned to next officer'
                WHERE  id               = :ra_id
                AND    status           IN ('ROUTED', 'ESCALATED_ROUTED')
                AND    escalation_count = 0
            """), {"new_officer_id": new_officer_id, "ra_id": ra_id})

        if result.rowcount == 0:
            # ACK landed between the SELECT and this UPDATE. Give back the
            # officer we speculatively claimed — otherwise ACK wins the alert
            # but the escalation target stays BUSY on nothing.
            with db.begin():
                db.execute(text("""
                    UPDATE officers
                    SET    status           = 'AVAILABLE',
                           current_alert_id = NULL,
                           last_updated     = datetime('now')
                    WHERE  id               = :oid
                    AND    current_alert_id = :ra_id
                """), {"oid": new_officer_id, "ra_id": ra_id})
            events.append({"kind": "officer", "officer_id": new_officer_id,
                           "status": "AVAILABLE", "current_alert_id": None})
            logger.info("Escalation for ra_id=%s lost to ACK — officer %s released.",
                        ra_id, new_officer_id)
            return events

        # Release the officer who timed out — they are no longer responsible.
        if previous_officer:
            with db.begin():
                db.execute(text("""
                    UPDATE officers
                    SET    status           = 'AVAILABLE',
                           current_alert_id = NULL,
                           last_updated     = datetime('now')
                    WHERE  id               = :oid
                    AND    current_alert_id = :ra_id
                """), {"oid": previous_officer, "ra_id": ra_id})
            events.append({"kind": "officer", "officer_id": previous_officer,
                           "status": "AVAILABLE", "current_alert_id": None})

        with db.begin():
            info = db.execute(text("""
                SELECT o.name AS name, ra.assigned_at AS assigned_at
                FROM   routed_alerts ra
                JOIN   officers o ON o.id = ra.assigned_officer
                WHERE  ra.id = :ra_id
            """), {"ra_id": ra_id}).mappings().fetchone()

        events.append({"kind": "officer", "officer_id": new_officer_id,
                       "status": "BUSY", "current_alert_id": ra_id})
        events.append({
            "kind": "escalated", "alert_id": alert_id, "ra_id": ra_id,
            "new_officer_id": new_officer_id,
            "new_officer_name": info["name"] if info else None,
            "new_assigned_at": info["assigned_at"] if info else None,
            "final_status": "ESCALATED_ROUTED",
        })
        logger.info("ra_id=%s escalated: officer %s -> %s", ra_id,
                    previous_officer, new_officer_id)
        return events

    # ── No officer available for escalation ─────────────────────────────────
    with db.begin():
        result = db.execute(text("""
            UPDATE routed_alerts
            SET    status           = 'ESCALATED_UNROUTED',
                   escalated_at     = datetime('now'),
                   escalation_count = 1,
                   escalation_note  = 'Timeout: no officers available for escalation'
            WHERE  id               = :ra_id
            AND    status           IN ('ROUTED', 'ESCALATED_ROUTED')
            AND    escalation_count = 0
        """), {"ra_id": ra_id})

    if result.rowcount == 0:
        return events  # ACK won

    # Release the timed-out officer even though escalation found nobody.
    # Skipping this is what strands an officer as BUSY on a dead alert.
    if previous_officer:
        with db.begin():
            db.execute(text("""
                UPDATE officers
                SET    status           = 'AVAILABLE',
                       current_alert_id = NULL,
                       last_updated     = datetime('now')
                WHERE  id               = :oid
                AND    current_alert_id = :ra_id
            """), {"oid": previous_officer, "ra_id": ra_id})
        events.append({"kind": "officer", "officer_id": previous_officer,
                       "status": "AVAILABLE", "current_alert_id": None})

    events.append({
        "kind": "escalated", "alert_id": alert_id, "ra_id": ra_id,
        "new_officer_id": None, "new_officer_name": None,
        "new_assigned_at": None, "final_status": "ESCALATED_UNROUTED",
    })
    logger.info("ra_id=%s escalated to UNROUTED (no officers available).", ra_id)
    return events
