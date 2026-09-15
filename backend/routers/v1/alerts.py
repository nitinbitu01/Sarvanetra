"""backend/routers/v1/alerts.py — /api/v1/alerts endpoints.
NOTE: No 'from __future__ import annotations'.
"""
import json
from datetime import datetime, timezone
from typing import Optional

from fastapi import APIRouter, Depends, HTTPException, Request
from pydantic import BaseModel, field_validator
from sqlalchemy import func
from sqlalchemy.orm import Session

from backend.auth.dependencies import get_current_user
from backend.core.logging import get_logger
from backend.core.rate_limit import GENERAL_LIMIT, limiter
from backend.db.models import Alert, Camera, User
from backend.db.session import get_db
from backend.services.audit_logger import log_audit
from backend.services.evidence import verify_file

router = APIRouter(prefix="/alerts", tags=["alerts"])
logger = get_logger(__name__)


def _format_alert_dict(r: Alert, cam_map: dict, merged_counts: dict) -> dict:
    return {
        "id": r.id,
        "alert_type": r.alert_type,
        "subject_label": r.subject_label,
        "confidence": r.confidence,
        "danger_score": r.danger_score,
        "evidence_hash": r.evidence_hash,
        "snapshot_path": r.snapshot_path,
        "status": r.status,
        "false_positive_reason": r.false_positive_reason,
        # Day 8 — general lifecycle
        "lifecycle_status": r.lifecycle_status,
        "reviewed_at": r.reviewed_at.isoformat() if r.reviewed_at else None,
        "feedback": r.feedback,
        # Day 10 — SENTINEL IQ
        "iq_contribution": r.iq_contribution,
        "iq_breakdown": json.loads(r.iq_breakdown_json) if r.iq_breakdown_json else None,
        # Day 11 — Alert fatigue dedup
        "merged_into_alert_id": r.merged_into_alert_id,
        "merged_count": merged_counts.get(r.id, 0),
        "metadata": json.loads(r.meta_json) if r.meta_json else None,
        "camera_name": cam_map[r.camera_id].name if r.camera_id and r.camera_id in cam_map else None,
        "camera_zone": cam_map[r.camera_id].zone if r.camera_id and r.camera_id in cam_map else None,
        "created_at": r.created_at.isoformat() if r.created_at else "",

        # Fields the feed used to drop. `is_simulated` is the one that matters
        # most: without it a staged or rehearsal alert is indistinguishable
        # from a real detection through the API, and anything reading this
        # endpoint — the UI, an export, an officer — would have no way to tell.
        # The rest are what an operator needs before acting: which camera, how
        # serious, what happened, and to which vehicle.
        "camera_id": r.camera_id,
        "severity": r.severity,
        "description": r.description,
        "plate_text": r.plate_text,
        "is_simulated": bool(r.is_simulated),
        "lat": r.lat,
        "lon": r.lon,
        "district": r.district,
        "timestamp": r.timestamp.isoformat() if r.timestamp else None,
        "score_breakdown": r.score_breakdown,
    }


@router.get("")
@limiter.limit(GENERAL_LIMIT)
async def list_alerts(
    request: Request,
    limit: int = 50,
    include_merged: bool = False,
    db: Session = Depends(get_db),
    current_user: User = Depends(get_current_user),
):
    query = db.query(Alert).filter(Alert.is_deleted == False)
    if not include_merged:
        # Default feed hides merged (losing) alerts from the primary feed
        query = query.filter(Alert.merged_into_alert_id.is_(None))

    rows = (
        query
        .order_by(Alert.created_at.desc())
        .limit(limit)
        .all()
    )

    try:
        log_audit(db, current_user, "ALERT_VIEW", "alert", None,
                  {"count": len(rows), "include_merged": include_merged},
                  request.client.host if request.client else None)
        db.commit()
    except Exception:
        db.rollback()

    cam_ids = {r.camera_id for r in rows if r.camera_id}
    cam_map = {}
    if cam_ids:
        # Look a camera up by registry id, by camera_id, AND by name.
        #
        # Alert.camera_id is written by whichever producer raised the alert,
        # and they do not agree on what a "camera id" is: the 24x7 pipeline
        # writes its own cam_id, which for a camera added through
        # POST /cameras is the human NAME (see _negotiate_and_create_camera,
        # which sets camera_id=body.name). Matching only on Camera.id meant
        # every STOLEN_VEHICLE_WATCHLIST_HIT rendered with a blank camera in
        # the alert feed — the one alert type where the operator most needs
        # to know which camera saw the vehicle.
        cams = (db.query(Camera)
                .filter((Camera.id.in_(cam_ids))
                        | (Camera.camera_id.in_(cam_ids))
                        | (Camera.name.in_(cam_ids)))
                .all())
        cam_map = {}
        for c in cams:
            for key in (c.id, c.camera_id, c.name):
                if key and key not in cam_map:
                    cam_map[key] = c

    row_ids = [r.id for r in rows]
    merged_counts = {}
    if row_ids:
        counts = (
            db.query(Alert.merged_into_alert_id, func.count(Alert.id))
            .filter(Alert.merged_into_alert_id.in_(row_ids), Alert.is_deleted == False)
            .group_by(Alert.merged_into_alert_id)
            .all()
        )
        merged_counts = {parent_id: cnt for parent_id, cnt in counts if parent_id is not None}

    return [_format_alert_dict(r, cam_map, merged_counts) for r in rows]


@router.get("/{alert_id}/merged")
@limiter.limit(GENERAL_LIMIT)
async def get_merged_alerts(
    alert_id: str,
    request: Request,
    db: Session = Depends(get_db),
    current_user: User = Depends(get_current_user),
):
    """Retrieve all constituent alerts that were merged into this primary alert."""
    parent = db.query(Alert).filter(Alert.id == alert_id, Alert.is_deleted == False).first()
    if not parent:
        raise HTTPException(status_code=404, detail="Primary alert not found")

    merged_rows = (
        db.query(Alert)
        .filter(Alert.merged_into_alert_id == alert_id, Alert.is_deleted == False)
        .order_by(Alert.created_at.asc())
        .all()
    )

    cam_ids = {r.camera_id for r in merged_rows if r.camera_id}
    cam_map = {}
    if cam_ids:
        # Look a camera up by registry id, by camera_id, AND by name.
        #
        # Alert.camera_id is written by whichever producer raised the alert,
        # and they do not agree on what a "camera id" is: the 24x7 pipeline
        # writes its own cam_id, which for a camera added through
        # POST /cameras is the human NAME (see _negotiate_and_create_camera,
        # which sets camera_id=body.name). Matching only on Camera.id meant
        # every STOLEN_VEHICLE_WATCHLIST_HIT rendered with a blank camera in
        # the alert feed — the one alert type where the operator most needs
        # to know which camera saw the vehicle.
        cams = (db.query(Camera)
                .filter((Camera.id.in_(cam_ids))
                        | (Camera.camera_id.in_(cam_ids))
                        | (Camera.name.in_(cam_ids)))
                .all())
        cam_map = {}
        for c in cams:
            for key in (c.id, c.camera_id, c.name):
                if key and key not in cam_map:
                    cam_map[key] = c

    return [_format_alert_dict(r, cam_map, {}) for r in merged_rows]


class MarkFalsePositiveRequest(BaseModel):
    reason: str

    @field_validator("reason")
    @classmethod
    def _reason_not_blank(cls, v: str) -> str:
        if not v or not v.strip():
            raise ValueError("reason is required")
        return v.strip()


@router.post("/{alert_id}/mark-false-positive")
@limiter.limit(GENERAL_LIMIT)
async def mark_false_positive(
    alert_id: str,
    body: MarkFalsePositiveRequest,
    request: Request,
    db: Session = Depends(get_db),
    current_user: User = Depends(get_current_user),
):
    """Officer marks a fired alert as a false positive.

    Never deletes the alert (evidence integrity) — records the reason and
    reviewer, and logs ALERT_MARKED_FALSE_POSITIVE to the audit log.
    """
    alert = db.query(Alert).filter(Alert.id == alert_id, Alert.is_deleted == False).first()
    if not alert:
        raise HTTPException(status_code=404, detail="Alert not found")

    alert.status = "false_positive"
    alert.false_positive_reason = body.reason
    alert.reviewed_by_user_id = current_user.id
    alert.resolved_at = datetime.now(timezone.utc)

    log_audit(db, current_user, "ALERT_MARKED_FALSE_POSITIVE", "alert", alert.id,
              {"reason": alert.false_positive_reason},
              request.client.host if request.client else None)

    logger.info("Alert marked false positive",
                extra={"alert_id": alert.id, "user_id": current_user.id})

    return {
        "id": alert.id,
        "status": alert.status,
        "false_positive_reason": alert.false_positive_reason,
        "reviewed_by_user_id": alert.reviewed_by_user_id,
    }


class DismissRequest(BaseModel):
    feedback: Optional[str] = None

    @field_validator("feedback")
    @classmethod
    def _feedback_valid(cls, v: Optional[str]) -> Optional[str]:
        if v is not None and v not in ("TRUE_POSITIVE", "FALSE_POSITIVE"):
            raise ValueError("feedback must be 'TRUE_POSITIVE', 'FALSE_POSITIVE', or omitted")
        return v


async def _apply_lifecycle_transition(
    alert_id: str, new_status: str, action: str, db: Session, current_user: User,
    request: Request, feedback: Optional[str] = None,
) -> dict:
    """Shared body for acknowledge/dismiss/escalate — same shape, different
    target status + audit action. Broadcasts alert_status_update over the
    existing dashboard WebSocket so a second connected operator's feed
    updates live (§3) — two people acting on the same alert independently
    is exactly what this prevents.

    Reviewer identity is pulled from the authenticated JWT (current_user),
    NOT a client-supplied name string — this matches the precedent already
    set by Day 7's /mark-false-positive (which does the same for the same
    reason: a free-text reviewer field is not something the audit log
    should trust). The Day 8 prompt's literal `{reviewed_by}` request body
    is folded into this — reviewed_by_user_id is set from auth, not the body.
    """
    from datetime import datetime, timezone as _tz
    from backend.db.models import Alert

    alert = db.query(Alert).filter(Alert.id == alert_id, Alert.is_deleted == False).first()
    if not alert:
        raise HTTPException(status_code=404, detail="Alert not found")

    alert.lifecycle_status = new_status
    alert.reviewed_by_user_id = current_user.id
    alert.reviewed_at = datetime.now(_tz.utc)
    if feedback is not None:
        alert.feedback = feedback

    log_audit(
        db, current_user, action, "alert", alert.id,
        {"new_status": new_status, "feedback": feedback},
        request.client.host if request.client else None,
    )

    logger.info("Alert lifecycle transition", extra={
        "alert_id": alert.id, "new_status": new_status, "user_id": current_user.id,
    })

    payload = {
        "type": "alert_status_update",
        "alert_id": alert.id,
        "lifecycle_status": alert.lifecycle_status,
        "reviewed_by_user_id": alert.reviewed_by_user_id,
        "reviewed_at": alert.reviewed_at.isoformat(),
        "feedback": alert.feedback,
    }
    try:
        from backend.ws.dashboard_ws import broadcast
        await broadcast(payload)
    except Exception:
        pass  # broadcast is best-effort; the DB write above is the source of truth

    return {
        "id": alert.id,
        "lifecycle_status": alert.lifecycle_status,
        "reviewed_by_user_id": alert.reviewed_by_user_id,
        "reviewed_at": alert.reviewed_at.isoformat(),
        "feedback": alert.feedback,
    }


@router.post("/{alert_id}/acknowledge")
@limiter.limit(GENERAL_LIMIT)
async def acknowledge_alert(
    alert_id: str, request: Request,
    db: Session = Depends(get_db), current_user: User = Depends(get_current_user),
):
    """Officer acknowledges an alert — signals "seen, being handled" to
    every other connected operator via alert_status_update."""
    return await _apply_lifecycle_transition(
        alert_id, "ACKNOWLEDGED", "ALERT_ACKNOWLEDGED", db, current_user, request,
    )


@router.post("/{alert_id}/dismiss")
@limiter.limit(GENERAL_LIMIT)
async def dismiss_alert(
    alert_id: str, body: DismissRequest, request: Request,
    db: Session = Depends(get_db), current_user: User = Depends(get_current_user),
):
    """Officer dismisses an alert, optionally recording TRUE_POSITIVE /
    FALSE_POSITIVE feedback. This is captured for later analysis — this
    pass does NOT feed it back into auto-tuning thresholds (§3, §7:
    explicitly deferred)."""
    return await _apply_lifecycle_transition(
        alert_id, "DISMISSED", "ALERT_DISMISSED", db, current_user, request,
        feedback=body.feedback,
    )


@router.post("/{alert_id}/escalate")
@limiter.limit(GENERAL_LIMIT)
async def escalate_alert(
    alert_id: str, request: Request,
    db: Session = Depends(get_db), current_user: User = Depends(get_current_user),
):
    """Officer escalates an alert for further/senior attention."""
    return await _apply_lifecycle_transition(
        alert_id, "ESCALATED", "ALERT_ESCALATED", db, current_user, request,
    )


@router.get("/{alert_id}/verify")
@limiter.limit(GENERAL_LIMIT)
async def verify_alert_evidence(
    alert_id: str,
    request: Request,
    db: Session = Depends(get_db),
    current_user: User = Depends(get_current_user),
):
    alert = db.query(Alert).filter(Alert.id == alert_id, Alert.is_deleted == False).first()
    if not alert:
        raise HTTPException(status_code=404, detail="Alert not found")

    if not alert.evidence_hash:
        return {"intact": None, "reason": "no_hash_stored"}

    path = alert.snapshot_path or alert.evidence_path
    intact = verify_file(path, alert.evidence_hash) if path else False

    log_audit(db, current_user, "EVIDENCE_VERIFY", "alert", alert.id,
              {"intact": intact, "path": path}, request.client.host if request.client else None)

    logger.info("Evidence verified", extra={"alert_id": alert_id, "intact": intact})
    return {"intact": intact, "alert_id": alert_id}


@router.get("/{alert_id}/audio")
@limiter.limit(GENERAL_LIMIT)
async def get_alert_voice_audio(
    alert_id: str,
    request: Request,
    lang: str = "hi",
    db: Session = Depends(get_db),
    current_user: User = Depends(get_current_user),
):
    """Synthesize and stream a spoken multilingual voice alert for control room PA broadcast."""
    from fastapi.responses import FileResponse
    from backend.services.voice_alert_service import get_voice_service

    alert = db.query(Alert).filter(Alert.id == alert_id, Alert.is_deleted == False).first()
    if not alert:
        raise HTTPException(status_code=404, detail="Alert not found")

    cam = db.query(Camera).filter(Camera.id == alert.camera_id).first() if alert.camera_id else None
    cam_name = cam.name if cam else (alert.camera_id or "Station Camera")
    subject = alert.subject_label or "Suspect"

    voice_svc = get_voice_service()
    prompt = voice_svc.build_prompt(
        alert_type=alert.alert_type or "SECURITY_ALERT",
        camera_name=cam_name,
        subject=subject,
        lang=lang,
    )
    import asyncio
    audio_path = await asyncio.to_thread(
        voice_svc.synthesize_speech, prompt, lang, alert_id
    )
    if not audio_path.exists():
        raise HTTPException(status_code=500, detail="Voice audio synthesis failed")

    media_type = "audio/mpeg" if audio_path.suffix == ".mp3" else "audio/wav"
    return FileResponse(
        str(audio_path),
        media_type=media_type,
        filename=audio_path.name,
    )

