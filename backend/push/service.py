"""
backend/push/service.py — officer-targeted Web Push fan-out (Day 15).

TARGETING
  Push goes to the officer routed_alerts assigned, not to every subscribed
  device. Broadcasting would be alert fatigue AND a data leak: the payload
  carries a camera name and location, and an officer with no role in an
  incident should not receive the location of a surveillance event.

  When nothing was assigned (UNROUTED, or dispatch withheld by the
  face-match interlock), push falls back to all AVAILABLE officers — someone
  needs to know an alert reached nobody. The payload says so explicitly so
  the phone does not imply an assignment that does not exist.

PAYLOAD
  Identifiers and a camera name only. No crop, no clip, no subject label.
  Push payloads transit a third-party relay (FCM/Mozilla) and land in an OS
  notification tray visible on a lock screen — neither is a place to put the
  face or name of someone the system has merely flagged. The device fetches
  the detail over authenticated HTTPS after the tap.

FAILURE POSTURE
  Every send attempt writes a push_delivery_log row. Nothing in this module
  raises into the alert pipeline: an alert that fired must not be undone by
  a notification that could not be delivered.
"""
from __future__ import annotations

import json
import logging
import threading
import time
from typing import Any

logger = logging.getLogger(__name__)

STATUS_SENT = "sent"
STATUS_FAILED = "failed"
STATUS_EXPIRED = "expired"
STATUS_RATE_LIMITED = "rate_limited"

# ── Rate limiting ────────────────────────────────────────────────────────────
# In-memory, per subscription, 60-second sliding window. Deliberately simple
# and deliberately NOT durable: this is a demo-scale guard against a stuck
# detection loop vibrating a phone continuously, not a distributed quota.
#
# Two consequences, both recorded in BACKLOG.md rather than hidden:
#   - the window resets on restart, so a burst can resume after a redeploy
#   - it is per-process, so N workers each allow the limit independently
# Replacing this with Redis sorted sets is the P0 item before real use.
_rate_lock = threading.Lock()
_rate_store: dict[int, list[float]] = {}
_RATE_WINDOW_S = 60.0


def _is_rate_limited(subscription_id: int, limit: int) -> bool:
    """True if this subscription has already had `limit` sends this minute."""
    now = time.time()
    with _rate_lock:
        stamps = [t for t in _rate_store.get(subscription_id, [])
                  if now - t < _RATE_WINDOW_S]
        if len(stamps) >= limit:
            _rate_store[subscription_id] = stamps
            return True
        stamps.append(now)
        _rate_store[subscription_id] = stamps
        return False


def reset_rate_limits() -> None:
    """Clear the window. For tests only."""
    with _rate_lock:
        _rate_store.clear()


# ── Fan-out ──────────────────────────────────────────────────────────────────

def send_critical_push(alert: Any, route_result: Any, db: Any) -> dict[str, int]:
    """Push a CRITICAL alert to the responsible officer's devices.

    `route_result` is the dict returned by routing.service.route_alert(), or
    None when routing was skipped entirely (e.g. the face-match interlock
    withheld dispatch). Both are handled — a withheld dispatch still needs
    somebody notified, it just must not claim an assignment.

    Returns a small tally for observability/tests. Never raises.
    """
    from sqlalchemy import text

    from backend.core.config import settings

    tally = {"sent": 0, "failed": 0, "expired": 0, "rate_limited": 0, "targets": 0}
    try:
        assigned_officer = None
        if isinstance(route_result, dict):
            assigned_officer = route_result.get("assigned_officer")

        if assigned_officer:
            officer_ids = [assigned_officer]
            assigned = True
        else:
            rows = db.execute(text(
                "SELECT id FROM officers WHERE status = 'AVAILABLE'"
            )).fetchall()
            officer_ids = [r.id for r in rows]
            assigned = False

        if not officer_ids:
            logger.info("Alert %s: no officers to notify.", alert.id)
            return tally

        # Parameter expansion by hand: SQLite has no array binding, and
        # string-formatting the ids in would be an injection surface even
        # though they come from our own table.
        placeholders = ", ".join(f":o{i}" for i in range(len(officer_ids)))
        params = {f"o{i}": oid for i, oid in enumerate(officer_ids)}
        subs = db.execute(text(f"""
            SELECT id, officer_id, endpoint, p256dh, auth
            FROM   push_subscriptions
            WHERE  officer_id IN ({placeholders})
        """), params).mappings().fetchall()

        tally["targets"] = len(subs)
        if not subs:
            logger.info(
                "Alert %s: %d target officer(s) but no subscribed devices.",
                alert.id, len(officer_ids),
            )
            return tally

        payload = _build_payload(alert, db, assigned)
        ttl = settings.ACK_TIMEOUT_SECONDS
        limit = settings.PUSH_RATE_LIMIT_PER_SUB_PER_MINUTE

        for sub in subs:
            outcome = _send_one(dict(sub), payload, alert.id, db, ttl, limit)
            tally[outcome] = tally.get(outcome, 0) + 1

        logger.info("Alert %s push fan-out: %s", alert.id, tally)
        return tally
    except Exception as exc:
        # A notification failure must never unwind an alert that already fired.
        logger.error("send_critical_push failed for alert=%s: %s",
                     getattr(alert, "id", "?"), exc, exc_info=True)
        return tally


def _build_payload(alert: Any, db: Any, assigned: bool) -> dict[str, Any]:
    """Identifiers only — see the module docstring on why."""
    from sqlalchemy import text

    from backend.routing.utils import to_iso8601

    cam = None
    if alert.camera_id:
        cam = db.execute(text(
            "SELECT name, zone FROM cameras WHERE id = :cid"
        ), {"cid": alert.camera_id}).mappings().fetchone()

    return {
        "alert_id": alert.id,
        "camera_id": alert.camera_id,
        "camera_name": cam["name"] if cam else "Unknown camera",
        # This codebase has no cameras.location_label; `zone` is the
        # equivalent human-readable placement label.
        "camera_location": (cam["zone"] if cam else None) or "",
        "severity": alert.severity,
        # Day 10's engine writes iq_contribution; the legacy `score` column is
        # only populated by the old raw-SQL ANPR path. Prefer the live one and
        # fall back, so the phone never shows "Score: null".
        "score": alert.iq_contribution if alert.iq_contribution is not None else alert.score,
        "created_at": to_iso8601(alert.created_at),
        # False means "nobody is assigned to this" — the phone must not imply
        # an assignment that the routing layer did not make.
        "assigned": assigned,
    }


def _send_one(sub: dict, payload: dict, alert_id: int, db: Any,
              ttl: int, limit: int) -> str:
    """Send to one subscription. Returns the outcome string."""
    from sqlalchemy import text

    from backend.core.config import settings

    if _is_rate_limited(sub["id"], limit):
        _log_delivery(db, alert_id, sub["id"], sub["officer_id"],
                      STATUS_RATE_LIMITED, None,
                      f"Exceeded {limit} sends/60s for this subscription")
        return STATUS_RATE_LIMITED

    try:
        from pywebpush import WebPushException, webpush

        webpush(
            subscription_info={
                "endpoint": sub["endpoint"],
                "keys": {"p256dh": sub["p256dh"], "auth": sub["auth"]},
            },
            data=json.dumps(payload),
            vapid_private_key=settings.VAPID_PRIVATE_KEY,
            vapid_claims={"sub": f"mailto:{settings.VAPID_CONTACT_EMAIL}"},
            ttl=ttl,
        )
        db.execute(text("""
            UPDATE push_subscriptions
            SET    last_seen_at = datetime('now')
            WHERE  id = :id
        """), {"id": sub["id"]})
        db.commit()
        _log_delivery(db, alert_id, sub["id"], sub["officer_id"],
                      STATUS_SENT, 201, None)
        return STATUS_SENT

    except Exception as exc:
        status_code = None
        response = getattr(exc, "response", None)
        if response is not None:
            status_code = getattr(response, "status_code", None)

        # 404/410 mean the push service has permanently retired this
        # endpoint — the device uninstalled the PWA, cleared site data, or
        # was replaced. Retrying is pointless forever, so the row is deleted
        # rather than left to fail on every future alert.
        if status_code in (404, 410):
            try:
                db.execute(text("DELETE FROM push_subscriptions WHERE id = :id"),
                           {"id": sub["id"]})
                db.commit()
            except Exception:
                db.rollback()
            _log_delivery(db, alert_id, None, sub["officer_id"],
                          STATUS_EXPIRED, status_code,
                          "Subscription deleted — push service returned "
                          f"{status_code} (endpoint permanently gone)")
            return STATUS_EXPIRED

        _log_delivery(db, alert_id, sub["id"], sub["officer_id"],
                      STATUS_FAILED, status_code, str(exc)[:500])
        return STATUS_FAILED


def _log_delivery(db: Any, alert_id: int, subscription_id: int | None,
                  officer_id: int | None, status: str,
                  status_code: int | None, error_detail: str | None) -> None:
    """Write one audit row. Never raises — logging cannot break delivery."""
    from sqlalchemy import text

    try:
        db.execute(text("""
            INSERT INTO push_delivery_log
                (alert_id, subscription_id, officer_id, status,
                 status_code, error_detail, sent_at)
            VALUES
                (:alert_id, :sub_id, :officer_id, :status,
                 :status_code, :error_detail, datetime('now'))
        """), {
            "alert_id": alert_id,
            "sub_id": subscription_id,
            "officer_id": officer_id,
            "status": status,
            "status_code": status_code,
            "error_detail": error_detail,
        })
        db.commit()
    except Exception as exc:
        db.rollback()
        logger.error("[push] delivery log write failed: %s", exc)


# ── Cleanup job ──────────────────────────────────────────────────────────────

def cleanup_stale_subscriptions() -> int:
    """Delete subscriptions with no successful send in N days.

    Registered on the existing APScheduler instance (daily). Returns the
    number removed so the caller can log it.
    """
    from sqlalchemy import text

    from backend.core.config import settings
    from backend.db.session import SessionLocal

    days = settings.PUSH_SUBSCRIPTION_STALE_DAYS
    db = SessionLocal()
    try:
        result = db.execute(text(f"""
            DELETE FROM push_subscriptions
            WHERE last_seen_at < datetime('now', '-{int(days)} days')
        """))
        db.commit()
        removed = result.rowcount or 0
        if removed:
            logger.info("[push cleanup] Removed %d stale subscription(s).", removed)
        return removed
    except Exception as exc:
        db.rollback()
        logger.error("[push cleanup] failed: %s", exc, exc_info=True)
        return 0
    finally:
        db.close()
