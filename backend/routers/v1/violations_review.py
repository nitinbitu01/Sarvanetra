"""
backend/routers/v1/violations_review.py — Human Review & e-Challan Workflow API.

The ONLY code path that issues or dismisses traffic violation e-Challans.
All reviewer identities are cryptographically bound to their JWT Bearer token.
"""
from __future__ import annotations

from datetime import datetime
from typing import Any, Optional

from fastapi import APIRouter, Depends, HTTPException, Request
from pydantic import BaseModel
from sqlalchemy.orm import Session

from backend.auth import ReviewerToken, get_current_reviewer
from backend.db.models import ReviewAuditLog, TrafficViolation
from backend.db.session import get_db

router = APIRouter(prefix="/violations", tags=["Traffic Violation Review"])


class DismissBody(BaseModel):
    reason: str  # mandatory for dismissal


# ── GET: reviewer queue ───────────────────────────────────────────────

@router.get("")
def list_violations(
    status: str = "pending_review",
    page: int = 1,
    page_size: int = 20,
    db: Session = Depends(get_db),
    reviewer: ReviewerToken = Depends(get_current_reviewer),
):
    offset = (page - 1) * page_size
    query = db.query(TrafficViolation)
    if status and status != "all":
        query = query.filter(TrafficViolation.challan_status == status)
    
    total = query.count()
    rows = (
        query.order_by(TrafficViolation.created_at.desc())
        .offset(offset)
        .limit(page_size)
        .all()
    )
    return {
        "items": [
            {
                "id": r.id,
                "violation_uuid": r.violation_uuid,
                "violation_type": r.violation_type,
                "camera_id": r.camera_id,
                "track_id": r.track_id,
                "vehicle_class": r.vehicle_class,
                "license_plate": r.license_plate,
                "plate_confidence": r.plate_confidence,
                "plate_format_valid": r.plate_format_valid,
                "flow_angle_deg": r.flow_angle_deg,
                "speed_kmh": r.speed_kmh,
                "confidence": r.confidence,
                "crop_path": r.crop_path,
                "plate_crop_path": r.plate_crop_path,
                "evidence_clip_path": r.evidence_clip_path,
                "evidence_clip_hash": r.evidence_clip_hash,
                "challan_status": r.challan_status,
                "reviewed_by": r.reviewed_by,
                "reviewed_at": r.reviewed_at.isoformat() if r.reviewed_at else None,
                "dismiss_reason": r.dismiss_reason,
                "challan_amount_inr": r.challan_amount_inr,
                "created_at": r.created_at.isoformat() if r.created_at else None,
            }
            for r in rows
        ],
        "total": total,
        "page": page,
        "page_size": page_size,
    }


# ── GET: single violation with full evidence ──────────────────────────

@router.get("/{vid}")
def get_violation(
    vid: int,
    db: Session = Depends(get_db),
    reviewer: ReviewerToken = Depends(get_current_reviewer),
):
    v = db.get(TrafficViolation, vid)
    if not v:
        raise HTTPException(404, "Violation not found.")
    return v


# ── POST: confirm -> challan issued ───────────────────────────────────

@router.post("/{vid}/confirm")
def confirm_violation(
    vid: int,
    request: Request,
    db: Session = Depends(get_db),
    reviewer: ReviewerToken = Depends(get_current_reviewer),
):
    v = db.get(TrafficViolation, vid)
    if not v:
        raise HTTPException(404, "Violation not found.")
    if v.challan_status != "pending_review":
        raise HTTPException(
            400, f"Cannot confirm: status is '{v.challan_status}', not 'pending_review'."
        )

    # Update violation row
    v.challan_status = "issued"
    v.reviewed_by = reviewer.reviewer_id
    v.reviewed_at = datetime.utcnow()

    # Append-only audit record
    client_ip = request.client.host if request.client else None
    audit = ReviewAuditLog(
        violation_id=vid,
        action="confirmed",
        reviewer_id=reviewer.reviewer_id,
        reviewer_ip=client_ip,
        reason=None,
    )
    db.add(audit)
    db.commit()

    # Route to retraining as true positive
    try:
        from training.retraining_queue import route_to_retraining

        route_to_retraining(
            violation_id=vid,
            crop_path=v.crop_path or "",
            label="true_positive",
            rider_count=getattr(v, "rider_count", 1),
            majority_ratio=getattr(v, "majority_ratio", 1.0),
            camera_id=v.camera_id,
            weather_regime=getattr(v, "weather_regime", "NORMAL") or "NORMAL",
        )
    except Exception:
        pass

    return {
        "status": "issued",
        "violation_id": vid,
        "reviewed_by": reviewer.username,
        "challan_amount_inr": v.challan_amount_inr,
    }


# ── POST: dismiss -> no challan ────────────────────────────────────────

@router.post("/{vid}/dismiss")
def dismiss_violation(
    vid: int,
    body: DismissBody,
    request: Request,
    db: Session = Depends(get_db),
    reviewer: ReviewerToken = Depends(get_current_reviewer),
):
    v = db.get(TrafficViolation, vid)
    if not v:
        raise HTTPException(404, "Violation not found.")
    if v.challan_status != "pending_review":
        raise HTTPException(
            400, f"Cannot dismiss: status is '{v.challan_status}'."
        )

    v.challan_status = "dismissed"
    v.reviewed_by = reviewer.reviewer_id
    v.reviewed_at = datetime.utcnow()
    v.dismiss_reason = body.reason

    # Append-only audit record
    client_ip = request.client.host if request.client else None
    audit = ReviewAuditLog(
        violation_id=vid,
        action="dismissed",
        reviewer_id=reviewer.reviewer_id,
        reviewer_ip=client_ip,
        reason=body.reason,
    )
    db.add(audit)
    db.commit()

    # Route to retraining as false positive — valuable feedback signal
    try:
        from training.retraining_queue import route_to_retraining

        route_to_retraining(
            violation_id=vid,
            crop_path=v.crop_path or "",
            label="false_positive",
            rider_count=getattr(v, "rider_count", 1),
            majority_ratio=getattr(v, "majority_ratio", 1.0),
            camera_id=v.camera_id,
            weather_regime=getattr(v, "weather_regime", "NORMAL") or "NORMAL",
        )
    except Exception:
        pass

    return {
        "status": "dismissed",
        "violation_id": vid,
        "reviewed_by": reviewer.username,
        "reason": body.reason,
    }

