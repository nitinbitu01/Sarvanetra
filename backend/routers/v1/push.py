"""backend/routers/v1/push.py — Day 15 push subscription + alert detail API.

  GET    /push/vapid-public-key    (no auth — the key is designed to be public)
  POST   /push/subscribe           (auth — upsert by endpoint)
  DELETE /push/subscribe           (auth — explicit opt-out)
  GET    /push/subscriptions       (admin — ops visibility)
  GET    /alerts/{id}/detail       (auth — single aggregated source for the card)

OFFICER BINDING
  A subscription must belong to an officer, but this system's `users` and
  `officers` are separate tables — a logged-in User is not automatically an
  Officer row. The binding is resolved by matching User.username to
  Officer.name, and a request that cannot be bound is REFUSED rather than
  silently attached to an arbitrary officer. Guessing here would mean
  routing one officer's alerts to another officer's phone.

NOTE: no `from __future__ import annotations` — matches the other routers.
"""
from typing import Any, Optional

from fastapi import APIRouter, Depends, HTTPException, Request, status as http_status
from pydantic import BaseModel
from sqlalchemy import text
from sqlalchemy.orm import Session

from backend.auth.dependencies import get_current_user, normalize_role
from backend.core.config import settings
from backend.core.logging import get_logger
from backend.core.rate_limit import GENERAL_LIMIT, limiter
from backend.db.models import User
from backend.db.session import get_db
from backend.routing.utils import to_iso8601

router = APIRouter(tags=["push"])
logger = get_logger(__name__)


def _resolve_officer_id(user: User, db: Session) -> Any:
    """Map the authenticated user to an officers row, or None.

    Matched on username == officers.name. Deliberately strict: returning a
    default officer when the match fails would silently subscribe a device to
    somebody else's alert stream.
    """
    row = db.execute(text(
        "SELECT id FROM officers WHERE name = :n"
    ), {"n": user.username}).fetchone()
    if not row:
        return None
    val = row[0]
    return int(val) if str(val).isdigit() else val


class SubscribeKeys(BaseModel):
    p256dh: str
    auth: str


class SubscribeRequest(BaseModel):
    endpoint: str
    # Browsers hand back PushSubscription.toJSON(), where p256dh and auth are
    # NESTED under "keys". Accepting them flat here would look like it worked
    # and then fail at encryption time with an opaque error.
    keys: SubscribeKeys
    device_label: Optional[str] = None


class UnsubscribeRequest(BaseModel):
    endpoint: str


@router.get("/push/vapid-public-key")
async def get_vapid_public_key() -> Any:
    """The VAPID public key. Intentionally unauthenticated.

    This key is applicationServerKey — it is designed to be handed to any
    browser that wants to subscribe, and it authenticates the SERVER to the
    push service, not the client to us. Fetching it (rather than hardcoding
    it in the bundle) is what makes key rotation possible without a frontend
    redeploy.
    """
    return {"public_key": settings.VAPID_PUBLIC_KEY}


@router.post("/push/subscribe")
@limiter.limit(GENERAL_LIMIT)
async def subscribe(
    body: SubscribeRequest,
    request: Request,
    db: Session = Depends(get_db),
    current_user: User = Depends(get_current_user),
) -> Any:
    officer_id = _resolve_officer_id(current_user, db)
    if officer_id is None:
        raise HTTPException(
            status_code=http_status.HTTP_409_CONFLICT,
            detail=(
                f"No officers row matches user '{current_user.username}'. "
                "Push subscriptions are officer-scoped; create the officer "
                "record first so alerts route to the right person."
            ),
        )

    # Upsert on endpoint: the push service returns the SAME endpoint for a
    # given device+origin, so a user tapping Enable Alerts twice must update
    # in place rather than accumulate duplicate rows (and duplicate buzzes).
    # ON CONFLICT also re-binds officer_id, which is what makes a shared
    # device correctly follow whoever logged in last.
    db.execute(text("""
        INSERT INTO push_subscriptions
            (officer_id, endpoint, p256dh, auth, device_label,
             created_at, last_seen_at)
        VALUES
            (:officer_id, :endpoint, :p256dh, :auth, :device_label,
             datetime('now'), datetime('now'))
        ON CONFLICT(endpoint) DO UPDATE SET
            officer_id   = excluded.officer_id,
            p256dh       = excluded.p256dh,
            auth         = excluded.auth,
            device_label = excluded.device_label,
            last_seen_at = datetime('now')
    """), {
        "officer_id": officer_id,
        "endpoint": body.endpoint,
        "p256dh": body.keys.p256dh,
        "auth": body.keys.auth,
        "device_label": body.device_label,
    })
    db.commit()

    row = db.execute(text(
        "SELECT id FROM push_subscriptions WHERE endpoint = :e"
    ), {"e": body.endpoint}).fetchone()

    logger.info("Push subscription upserted", extra={
        "officer_id": officer_id, "subscription_id": row.id if row else None,
    })
    return {"status": "subscribed",
            "subscription_id": row.id if row else None,
            "officer_id": officer_id}


@router.delete("/push/subscribe")
@limiter.limit(GENERAL_LIMIT)
async def unsubscribe(
    body: UnsubscribeRequest,
    request: Request,
    db: Session = Depends(get_db),
    current_user: User = Depends(get_current_user),
) -> Any:
    officer_id = _resolve_officer_id(current_user, db)
    # Scoped to the caller's own officer_id: one officer must not be able to
    # unsubscribe another officer's device by guessing an endpoint.
    result = db.execute(text("""
        DELETE FROM push_subscriptions
        WHERE endpoint = :e AND officer_id = :oid
    """), {"e": body.endpoint, "oid": officer_id})
    db.commit()
    return {"status": "unsubscribed", "removed": result.rowcount or 0}


@router.get("/push/subscriptions")
@limiter.limit(GENERAL_LIMIT)
async def list_subscriptions(
    request: Request,
    db: Session = Depends(get_db),
    current_user: User = Depends(get_current_user),
) -> Any:
    if normalize_role(current_user) != "ADMIN":
        raise HTTPException(status_code=403, detail="Admin only.")
    rows = db.execute(text("""
        SELECT ps.id, ps.officer_id, o.name AS officer_name, ps.device_label,
               ps.endpoint, ps.created_at, ps.last_seen_at
        FROM   push_subscriptions ps
        LEFT   JOIN officers o ON o.id = ps.officer_id
        ORDER  BY o.name ASC, ps.created_at DESC
    """)).mappings().fetchall()
    return [
        {
            "id": r["id"],
            "officer_id": r["officer_id"],
            "officer_name": r["officer_name"],
            "device_label": r["device_label"],
            # Endpoints are long and contain a device-unique token. Truncated
            # so an ops screen doesn't casually display something that
            # identifies a specific handset.
            "endpoint_preview": (r["endpoint"] or "")[:48] + "…",
            "created_at": to_iso8601(r["created_at"]),
            "last_seen_at": to_iso8601(r["last_seen_at"]),
        }
        for r in rows
    ]


@router.get("/alerts/{alert_id}/detail")
@limiter.limit(GENERAL_LIMIT)
async def alert_detail(
    alert_id: str,
    request: Request,
    db: Session = Depends(get_db),
    current_user: User = Depends(get_current_user),
) -> Any:
    """Everything the full-screen card needs, in ONE query.

    The card is opened from a notification tap on a phone, frequently on a
    poor connection. Three parallel fetches would mean three chances to hang
    and a card that renders in pieces.
    """
    row = db.execute(text("""
        SELECT a.id            AS alert_id,
               a.alert_type    AS alert_type,
               a.severity      AS severity,
               a.score         AS legacy_score,
               a.iq_contribution AS iq_contribution,
               a.subject_label AS subject_label,
               a.snapshot_path AS snapshot_path,
               a.created_at    AS created_at,
               a.camera_id     AS camera_id,
               c.name          AS camera_name,
               c.zone          AS camera_zone,
               COALESCE(c.location_label, c.zone) AS location_label,
               ra.status       AS routing_status,
               ra.assigned_at  AS assigned_at,
               ra.ack_at       AS ack_at,
               o.name          AS officer_name,
               e.status        AS evidence_status,
               e.sha256        AS evidence_sha256,
               af.verdict      AS feedback_verdict,
               af.officer_id   AS feedback_officer_id,
               af.created_at   AS feedback_updated_at,
               fo.name         AS feedback_officer_name
        FROM   alerts a
        LEFT   JOIN cameras c        ON c.id = a.camera_id
        LEFT   JOIN routed_alerts ra ON ra.alert_id = a.id
        LEFT   JOIN officers o       ON o.id = ra.assigned_officer
        LEFT   JOIN evidence e       ON e.alert_id = a.id
        LEFT   JOIN alert_feedback af ON af.alert_id = a.id
        LEFT   JOIN officers fo      ON fo.id = af.officer_id
        WHERE  a.id = :aid
    """), {"aid": alert_id}).mappings().fetchone()

    if row is None:
        raise HTTPException(status_code=404, detail=f"Alert {alert_id} not found.")

    # Evidence clips are served through the authenticated Day 13 endpoint,
    # never as a raw filesystem path — evidence must not become a public URL.
    clip_url = (f"/api/v1/evidence/{alert_id}/clip"
                if row["evidence_status"] == "COMPLETE" else None)
    crop_url = None
    snap = row["snapshot_path"]
    if snap:
        clean = str(snap).replace("\\", "/").lstrip("./")
        if clean.startswith("output/"):
            crop_url = "/media/output/" + clean[len("output/"):]

    return {
        "alert_id": row["alert_id"],
        "alert_type": row["alert_type"],
        "severity": row["severity"],
        # Prefer the live Day 10 score; the legacy `score` column is only
        # written by the old ANPR path and is NULL for every modern alert.
        "score": row["iq_contribution"] if row["iq_contribution"] is not None
                 else row["legacy_score"],
        "subject_label": row["subject_label"],
        "created_at": to_iso8601(row["created_at"]),
        "camera_id": row["camera_id"],
        "camera_name": row["camera_name"],
        "camera_location": row["camera_zone"],
        "location_label": row["location_label"],
        "clip_url": clip_url,
        "crop_url": crop_url,
        "evidence_status": row["evidence_status"],
        "evidence_sha256": row["evidence_sha256"],
        "routing": {
            "status": row["routing_status"],
            "officer_name": row["officer_name"],
            "assigned_at": to_iso8601(row["assigned_at"]),
            "ack_at": to_iso8601(row["ack_at"]),
        },
        "feedback": {
            "verdict": row["feedback_verdict"],
            "officer_id": row["feedback_officer_id"],
            "officer_name": row["feedback_officer_name"],
            "updated_at": to_iso8601(row["feedback_updated_at"]),
        },
    }
