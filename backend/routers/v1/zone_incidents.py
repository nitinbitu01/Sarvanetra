"""backend/routers/v1/zone_incidents.py — /api/v1/zone-incidents endpoints (Day 11).
NOTE: No 'from __future__ import annotations'.
"""
import json
from datetime import datetime, timezone
from typing import Optional

from fastapi import APIRouter, Depends, HTTPException, Request
from sqlalchemy.orm import Session

from backend.auth.dependencies import get_current_user
from backend.core.logging import get_logger
from backend.core.rate_limit import GENERAL_LIMIT, limiter
from backend.db.models import Alert, Camera, User, ZoneIncident
from backend.db.session import get_db
from backend.services.audit_logger import log_audit

router = APIRouter(prefix="/zone-incidents", tags=["zone-incidents"])
logger = get_logger(__name__)


def _format_incident(inc: ZoneIncident, db: Session, expand_alerts: bool = False) -> dict:
    try:
        alert_id_list = json.loads(inc.alert_ids) if inc.alert_ids else []
    except Exception:
        alert_id_list = []

    res = {
        "id": inc.id,
        "zone": inc.zone,
        "center_lat": inc.center_lat,
        "center_lon": inc.center_lon,
        "opened_at": inc.opened_at.isoformat() if inc.opened_at else "",
        "closed_at": inc.closed_at.isoformat() if inc.closed_at else None,
        "alert_count": len(alert_id_list),
        "alert_ids": alert_id_list,
    }

    if expand_alerts and alert_id_list:
        alerts = (
            db.query(Alert)
            .filter(Alert.id.in_(alert_id_list), Alert.is_deleted == False)
            .order_by(Alert.created_at.asc())
            .all()
        )
        cam_ids = {a.camera_id for a in alerts if a.camera_id}
        cam_map = {}
        if cam_ids:
            cams = db.query(Camera).filter(Camera.id.in_(cam_ids)).all()
            cam_map = {c.id: c for c in cams}

        res["alerts"] = [
            {
                "id": a.id,
                "alert_type": a.alert_type,
                "subject_label": a.subject_label,
                "confidence": a.confidence,
                "danger_score": a.danger_score,
                "iq_contribution": a.iq_contribution,
                "lifecycle_status": a.lifecycle_status,
                "camera_name": cam_map[a.camera_id].name if a.camera_id and a.camera_id in cam_map else None,
                "camera_zone": cam_map[a.camera_id].zone if a.camera_id and a.camera_id in cam_map else None,
                "created_at": a.created_at.isoformat() if a.created_at else "",
            }
            for a in alerts
        ]
    return res


@router.get("")
@limiter.limit(GENERAL_LIMIT)
async def list_zone_incidents(
    request: Request,
    include_closed: bool = False,
    expand_alerts: bool = False,
    db: Session = Depends(get_db),
    current_user: User = Depends(get_current_user),
):
    """List zone incidents. By default returns only open (unclosed) incidents."""
    query = db.query(ZoneIncident)
    if not include_closed:
        query = query.filter(ZoneIncident.closed_at.is_(None))

    rows = query.order_by(ZoneIncident.opened_at.desc()).all()

    log_audit(
        db, current_user, "ZONE_INCIDENT_LIST", "zone_incident", None,
        {"count": len(rows)}, request.client.host if request.client else None,
    )

    return [_format_incident(r, db, expand_alerts=expand_alerts) for r in rows]


@router.get("/{incident_id}")
@limiter.limit(GENERAL_LIMIT)
async def get_zone_incident(
    incident_id: int,
    request: Request,
    db: Session = Depends(get_db),
    current_user: User = Depends(get_current_user),
):
    """Get single zone incident with full expanded constituent alert details."""
    incident = db.query(ZoneIncident).filter(ZoneIncident.id == incident_id).first()
    if not incident:
        raise HTTPException(status_code=404, detail="Zone incident not found")

    return _format_incident(incident, db, expand_alerts=True)
