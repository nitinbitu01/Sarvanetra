"""backend/routers/v1/sentinel_iq.py — SENTINEL IQ score card + night-mode control.

Two routers in one file, because they are two halves of one feature:

  /api/v1/iq/cameras        read the live per-camera score card
  /api/v1/admin/night-mode  force / release the night multiplier (demo control)

Nothing here recomputes a score. Every number returned is summed from
Alert.iq_contribution values that were computed once at fire time and stored.
See backend/services/sentinel_iq.py for why that asymmetry exists.

NOTE: No 'from __future__ import annotations' — matches alerts.py, which
avoids it because FastAPI resolves the response/param annotations at runtime.
"""
from typing import Optional

from fastapi import APIRouter, Depends, HTTPException, Query, Request
from pydantic import BaseModel
from sqlalchemy.orm import Session

from backend.auth.dependencies import get_current_user, normalize_role
from backend.core.logging import get_logger
from backend.core.rate_limit import GENERAL_LIMIT, limiter
from backend.db.models import User
from backend.db.session import get_db
from backend.services.audit_logger import log_audit
from backend.services.sentinel_iq import (
    IQ_BASE_SCORES,
    IQ_ROADMAP_TRIGGERS,
    ROADMAP_LABEL,
    get_all_camera_iq_scores,
    get_camera_iq_score,
    get_night_override,
    set_night_override,
)

router = APIRouter(prefix="/iq", tags=["sentinel-iq"])
admin_router = APIRouter(prefix="/admin", tags=["sentinel-iq"])
logger = get_logger(__name__)

DEFAULT_WINDOW_MINUTES = 60


def _roadmap_payload():
    """Roadmap-only triggers, as LABELS with no numeric field of any kind.

    Deliberately not `{"score": 0.0}` and not `{"score": null}` — a score key
    at all invites a UI to render "0" next to PERIMETER_BREACH, which reads
    as "perimeter checked, nothing found" rather than "no detector exists".
    """
    return [
        {"alert_type": t, "status": ROADMAP_LABEL}
        for t in IQ_ROADMAP_TRIGGERS
    ]


def _summary_payload(summary):
    return {
        "camera_id": summary.camera_id,
        "camera_name": summary.camera_name,
        "camera_zone": summary.camera_zone,
        "total_score": summary.total_score,
        "alert_count": summary.alert_count,
        "unscored_count": summary.unscored_count,
        "window_minutes": summary.window_minutes,
        "contributions": summary.contributions,
    }


@router.get("/cameras")
@limiter.limit(GENERAL_LIMIT)
async def list_camera_iq_scores(
    request: Request,
    window_minutes: int = Query(default=DEFAULT_WINDOW_MINUTES, ge=1, le=1440),
    db: Session = Depends(get_db),
    current_user: User = Depends(get_current_user),
):
    """Live score card: one entry per camera with an active alert in the window.

    Cameras with no active alerts are absent, not listed at 0.0 — see
    get_all_camera_iq_scores().
    """
    summaries = get_all_camera_iq_scores(window_minutes, db=db)
    return {
        "window_minutes": window_minutes,
        "night_mode": _night_mode_payload(),
        "base_scores": IQ_BASE_SCORES,
        "roadmap_triggers": _roadmap_payload(),
        "cameras": [_summary_payload(s) for s in summaries],
    }


@router.get("/cameras/{camera_id}")
@limiter.limit(GENERAL_LIMIT)
async def get_single_camera_iq_score(
    camera_id: int,
    request: Request,
    window_minutes: int = Query(default=DEFAULT_WINDOW_MINUTES, ge=1, le=1440),
    db: Session = Depends(get_db),
    current_user: User = Depends(get_current_user),
):
    # An IQ score is a per-camera operational report — detection counts,
    # uptime, quality. Same department boundary as the footage it summarises.
    from backend.auth.dependencies import assert_camera_in_scope
    assert_camera_in_scope(camera_id, current_user, db)

    summary = get_camera_iq_score(camera_id, window_minutes, db=db)
    return {
        "window_minutes": window_minutes,
        "night_mode": _night_mode_payload(),
        "roadmap_triggers": _roadmap_payload(),
        "camera": _summary_payload(summary),
    }


# ── Night mode control ────────────────────────────────────────────────────────

class NightModeRequest(BaseModel):
    # Tri-state on purpose: true = force night, false = force day,
    # null = release the override and go back to the clock.
    enabled: Optional[bool] = None


def _night_mode_payload():
    override = get_night_override()
    return {
        "override": override,
        "mode": "clock" if override is None else "manual_override",
    }


@admin_router.get("/night-mode")
@limiter.limit(GENERAL_LIMIT)
async def read_night_mode(
    request: Request,
    current_user: User = Depends(get_current_user),
):
    return _night_mode_payload()


@admin_router.post("/night-mode")
@limiter.limit(GENERAL_LIMIT)
async def update_night_mode(
    body: NightModeRequest,
    request: Request,
    db: Session = Depends(get_db),
    current_user: User = Depends(get_current_user),
):
    """Force the night multiplier on/off, or pass null to return to the clock.

    Admin-only, and audit-logged: this changes how every subsequently-fired
    alert is scored, so who flipped it and when is not something to leave
    untracked. Already-fired alerts are NOT rescored — their contribution was
    frozen at fire time by design.
    """
    if normalize_role(current_user) != "ADMIN":
        raise HTTPException(
            status_code=403,
            detail="Only ADMIN may change the SENTINEL IQ night-mode override.",
        )

    set_night_override(body.enabled)

    log_audit(
        db, current_user, "IQ_NIGHT_MODE_SET", "settings", None,
        {"enabled": body.enabled},
        request.client.host if request.client else None,
    )
    logger.info("SENTINEL IQ night mode override changed", extra={
        "enabled": body.enabled, "user_id": current_user.id,
    })

    return {
        **_night_mode_payload(),
        "note": (
            "Applies to alerts fired from now on. Alerts already scored keep "
            "the contribution they were given at fire time."
        ),
    }
