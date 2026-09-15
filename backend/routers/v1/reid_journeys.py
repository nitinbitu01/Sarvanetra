"""
backend/routers/v1/reid_journeys.py — Journey timeline endpoints (Day 9 hardened).

Endpoints:
  GET /api/v1/reid/journeys/{global_person_id}  — ordered journey for a person

Day 9 changes (§1 + §2):
  - RBAC enforcement: VIEWER → 403 always; OPERATOR → 403 unless global_id has
    an active (non-DISMISSED) alert; ADMIN → unrestricted.
  - Every call (success or 403) writes a row to journey_query_log.
  - ADMIN callers may supply ?reason=<text> — prompted by the frontend for
    unrestricted lookups; logged in journey_query_log. OPERATOR queries are
    self-justifying (they require an active alert) so reason is NULL there.

Product decision baked in: nobody browses movement histories of people who
haven't been flagged for something. If that requirement changes (e.g. ADMIN
needs unrestricted investigative access), that's a role-config change,
not a code rewrite.
"""
from __future__ import annotations

import logging
from datetime import datetime
# Optional is required at RUNTIME, not just for type checking. `from __future__
# import annotations` turns annotations into strings, which is why the missing
# import stayed invisible everywhere else in this file — but Pydantic resolves
# a model's ForwardRefs when it builds the schema, and VehicleSearchRequest
# below annotates two fields Optional[str]. Without this import that resolution
# raises NameError, which surfaces as PydanticUndefinedAnnotation and aborts
# the import of backend.main — i.e. the whole API failed to start.
from typing import Any, Optional

from fastapi import APIRouter, Depends, HTTPException, Query, Request, status
from sqlalchemy.orm import Session

from backend.auth.dependencies import get_current_user, normalize_role
from backend.db.models import Alert, GlobalPerson, Journey, JourneyQueryLog, User
from backend.db.session import get_db

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/reid", tags=["ReID"])


def _log_query(
    db: Session,
    user: User,
    role: str,
    global_person_id: int,
    source_ip: str | None,
    outcome: str,
    reason: str | None,
) -> None:
    """Write one journey_query_log row. Flush but do NOT commit — callers commit."""
    entry = JourneyQueryLog(
        user_id=user.id,
        role=role,
        global_id_queried=global_person_id,
        query_time=datetime.utcnow(),
        source_ip=source_ip,
        outcome=outcome,
        reason=reason,
    )
    db.add(entry)
    try:
        db.flush()
    except Exception:
        # Logging failure must never block the response — just note it.
        logger.exception(
            "Failed to flush journey_query_log row for user_id=%s global_id=%s",
            user.id, global_person_id,
        )


def _get_client_ip(request: Request) -> str | None:
    """Extract client IP, respecting X-Forwarded-For if present."""
    forwarded = request.headers.get("X-Forwarded-For")
    if forwarded:
        return forwarded.split(",")[0].strip()
    if request.client:
        return request.client.host
    return None


@router.get(
    "/journeys/{global_person_id}",
    summary="Get journey timeline for a GlobalPerson",
)
async def get_journey(
    global_person_id: int,
    request: Request,
    reason: str | None = Query(default=None, description=(
        "Optional free-text reason for this lookup. "
        "Required by policy for ADMIN unrestricted queries — the frontend "
        "prompts for this before sending the request."
    )),
    db: Session = Depends(get_db),
    current_user: User = Depends(get_current_user),
) -> dict[str, Any]:
    """Return all Journey entries for a GlobalPerson, ordered by seen_at.

    Access control (Day 9 §1):
      VIEWER  → 403 always
      OPERATOR → allowed ONLY if global_person_id has at least one alert with
                 lifecycle_status != 'DISMISSED' (i.e. an active watchlist
                 match or behaviour flag). Otherwise 403.
      ADMIN   → unrestricted; reason param is logged for accountability.

    Every call (success or 403) is written to journey_query_log (§2).
    """
    role = normalize_role(current_user)
    source_ip = _get_client_ip(request)

    # ── VIEWER: hard deny ────────────────────────────────────────────────────
    if role == "VIEWER":
        _log_query(db, current_user, role, global_person_id, source_ip,
                   outcome="FORBIDDEN", reason=None)
        db.commit()
        raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN,
            detail="VIEWER role cannot query journey histories.",
        )

    # ── OPERATOR: allowed only for actively-flagged identities ───────────────
    if role == "OPERATOR":
        has_active_alert = (
            db.query(Alert.id)
            .filter(
                Alert.global_id == str(global_person_id),
                Alert.lifecycle_status != "DISMISSED",
                Alert.is_deleted == False,  # noqa: E712
            )
            .first()
        )
        if not has_active_alert:
            _log_query(db, current_user, role, global_person_id, source_ip,
                       outcome="FORBIDDEN", reason=None)
            db.commit()
            raise HTTPException(
                status_code=status.HTTP_403_FORBIDDEN,
                detail=(
                    f"GlobalPerson {global_person_id} has no active alert. "
                    "OPERATOR access requires at least one non-DISMISSED alert "
                    "tied to this identity."
                ),
            )

    # ── Fetch the GlobalPerson (ADMIN + permitted OPERATOR reach here) ────────
    gp = db.query(GlobalPerson).filter(
        GlobalPerson.id == global_person_id,
        GlobalPerson.is_deleted == False,  # noqa: E712
    ).first()
    if not gp:
        # Log the attempt even on 404 — the attempt itself is auditable.
        _log_query(db, current_user, role, global_person_id, source_ip,
                   outcome="FORBIDDEN", reason=reason)
        db.commit()
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"GlobalPerson {global_person_id} not found.",
        )

    entries = (
        db.query(Journey)
        .filter(Journey.global_person_id == global_person_id)
        .order_by(Journey.seen_at.asc())
        .all()
    )

    # ── Log successful query ─────────────────────────────────────────────────
    _log_query(db, current_user, role, global_person_id, source_ip,
               outcome="SUCCESS", reason=reason if role == "ADMIN" else None)
    db.commit()

    return {
        "global_person_id": global_person_id,
        "total_sightings": gp.total_sightings,
        "first_seen_at": gp.first_seen_at.isoformat() if gp.first_seen_at else None,
        "last_seen_at": gp.last_seen_at.isoformat() if gp.last_seen_at else None,
        "legal_basis": gp.legal_basis,
        "retention_hold": gp.retention_hold,
        "journey": [
            {
                "journey_id": e.id,
                "camera_id": e.camera_id,           # legacy string (display-friendly)
                "camera_db_id": e.camera_db_id,     # int FK to cameras.id
                "local_track_id": e.local_track_id,
                "seen_at": e.seen_at.isoformat() if e.seen_at else None,
                "confidence": round(e.confidence, 4) if e.confidence else None,
                "confidence_caveat": e.confidence_caveat,
            }
            for e in entries
        ],
    }


# ─────────────────────────────────────────────────────────────────────────────
# VEHICLE RE-ID & MULTI-CAMERA JOURNEY ENDPOINTS
# ─────────────────────────────────────────────────────────────────────────────
from pydantic import BaseModel
import base64
import cv2
import numpy as np


class VehicleSearchRequest(BaseModel):
    camera_id: int
    track_id: str
    timestamp_utc: float
    plate_text: Optional[str] = None
    crop_base64: Optional[str] = None
    top_k: int = 10
    visual_threshold: float = 0.75


@router.get(
    "/vehicles/records",
    summary="List indexed vehicle sightings across all cameras",
)
async def list_vehicle_records(
    limit: int = 50,
    current_user: User = Depends(get_current_user),
) -> dict[str, Any]:
    from backend.services.vehicle_reid_engine import get_vehicle_reid_engine
    v_engine = get_vehicle_reid_engine()
    records = list(v_engine.vehicle_map.values())[-limit:]
    return {
        "total_indexed": len(v_engine.vehicle_map),
        "records": [
            {
                "reid_id": r.reid_id,
                "camera_id": r.camera_id,
                "timestamp_utc": r.timestamp_utc,
                "track_id": r.track_id,
                "color": r.color,
                "vehicle_type": r.vehicle_type,
                "plate_text": r.plate_text,
            }
            for r in reversed(records)
        ],
    }


@router.post(
    "/vehicles/search",
    summary="Search for matching vehicles across all cameras using multimodal ReID (without plate text)",
)
async def search_vehicle_by_appearance(
    req: VehicleSearchRequest,
    current_user: User = Depends(get_current_user),
) -> dict[str, Any]:
    from backend.services.vehicle_reid_engine import get_vehicle_reid_engine
    v_engine = get_vehicle_reid_engine()

    crop = None
    if req.crop_base64:
        try:
            img_data = base64.b64decode(req.crop_base64.split(",")[-1])
            nparr = np.frombuffer(img_data, np.uint8)
            crop = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
        except Exception:
            crop = None

    emb = v_engine.extract_embedding(
        vehicle_crop=crop,
        track_id=req.track_id,
        camera_id=req.camera_id,
        timestamp_utc=req.timestamp_utc,
        plate_text=req.plate_text,
    )

    matches = v_engine.search_vehicle(
        emb=emb,
        top_k=req.top_k,
        visual_threshold=req.visual_threshold,
    )

    return {
        "query": {
            "camera_id": req.camera_id,
            "track_id": req.track_id,
            "color": emb.color,
            "vehicle_type": emb.vehicle_type,
            "plate_text": emb.plate_text,
        },
        "total_matches": len(matches),
        "matches": [
            {
                "reid_id": m.record.reid_id,
                "camera_id": m.record.camera_id,
                "timestamp_utc": m.record.timestamp_utc,
                "track_id": m.record.track_id,
                "color": m.record.color,
                "vehicle_type": m.record.vehicle_type,
                "plate_text": m.record.plate_text,
                "raw_visual_cosine": round(m.cosine_score, 4),
                "color_similarity": round(m.color_similarity, 4),
                "final_fused_score": round(m.final_score, 4),
                "match_type": m.match_type,
                "clm_feasibility": {
                    "is_feasible": m.feasibility.is_feasible,
                    "confidence_boost": round(m.feasibility.confidence_boost, 4),
                    "delta_t_seconds": round(m.feasibility.delta_t_seconds, 1),
                },
            }
            for m in matches
        ],
    }

