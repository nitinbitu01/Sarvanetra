"""backend/routers/v1/audit.py — GET /api/v1/audit-log (admin only).

The Model 1 registry deliverable calls for "role-based search/audit trails".
Writing is already covered (services/audit_logger.log_audit is called from
every mutating camera/auth route); this is the search half — filterable, not
just a raw newest-N dump, which is what "search" actually implies.
"""
from __future__ import annotations

import json
from datetime import datetime
from typing import Optional

from fastapi import APIRouter, Depends, HTTPException, Query, Request, status
from sqlalchemy.orm import Session

from backend.auth.dependencies import require_role
from backend.core.rate_limit import GENERAL_LIMIT, limiter
from backend.db.models import AuditLog, User
from backend.db.session import get_db

router = APIRouter(prefix="/audit-log", tags=["audit"])


def _parse_iso(value: Optional[str], field: str) -> Optional[datetime]:
    if not value:
        return None
    try:
        return datetime.fromisoformat(value)
    except ValueError:
        raise HTTPException(
            status.HTTP_400_BAD_REQUEST,
            f"{field} must be ISO-8601 (e.g. 2026-09-01T00:00:00), got {value!r}",
        )


@router.get("")
@limiter.limit(GENERAL_LIMIT)
async def get_audit_log(
    request: Request,
    limit: int = Query(100, ge=1, le=2000),
    action: Optional[str] = Query(
        None, description="Exact match, e.g. CAMERA_ADD, LOGIN, CAMERA_WENT_OFFLINE"),
    resource_type: Optional[str] = Query(None, description="e.g. camera, alert, user"),
    resource_id: Optional[str] = Query(None),
    user_id: Optional[str] = Query(None, description="Who performed the action"),
    date_from: Optional[str] = Query(None, description="ISO-8601, inclusive"),
    date_to: Optional[str] = Query(None, description="ISO-8601, inclusive"),
    db: Session = Depends(get_db),
    current_user: User = Depends(require_role("admin")),
):
    """Searchable audit log — admin only, newest first.

    Admin-only rather than department-scoped: an audit trail is only
    trustworthy if it cannot be edited or hidden by the department it
    concerns, so this deliberately does NOT filter by the caller's own
    department the way the camera registry does — the whole point of an
    audit trail is that acting on your own resources doesn't make the record
    of it yours to curate.
    """
    q = db.query(AuditLog)
    if action:
        q = q.filter(AuditLog.action == action)
    if resource_type:
        q = q.filter(AuditLog.resource_type == resource_type)
    if resource_id:
        q = q.filter(AuditLog.resource_id == resource_id)
    if user_id:
        q = q.filter(AuditLog.user_id == user_id)
    dt_from = _parse_iso(date_from, "date_from")
    dt_to = _parse_iso(date_to, "date_to")
    if dt_from:
        q = q.filter(AuditLog.created_at >= dt_from)
    if dt_to:
        q = q.filter(AuditLog.created_at <= dt_to)

    rows = q.order_by(AuditLog.created_at.desc()).limit(limit).all()
    return [
        {
            "id": r.id,
            "user_id": r.user_id,
            "action": r.action,
            "resource_type": r.resource_type,
            "resource_id": r.resource_id,
            "details": json.loads(r.details) if r.details else None,
            "ip_address": r.ip_address,
            "created_at": r.created_at.isoformat() if r.created_at else None,
        }
        for r in rows
    ]


@router.get("/actions")
@limiter.limit(GENERAL_LIMIT)
async def list_audit_actions(
    request: Request,
    db: Session = Depends(get_db),
    current_user: User = Depends(require_role("admin")),
):
    """Distinct action/resource_type values actually present in the log.

    A search filter is only usable if the operator knows what values exist
    to filter by — this backs an autocomplete/dropdown rather than making
    someone guess "CAMERA_ADD" vs "camera_add" vs "AddCamera".
    """
    actions = [r[0] for r in db.query(AuditLog.action).distinct().all() if r[0]]
    resource_types = [r[0] for r in db.query(AuditLog.resource_type).distinct().all()
                      if r[0]]
    return {"actions": sorted(actions), "resource_types": sorted(resource_types)}
