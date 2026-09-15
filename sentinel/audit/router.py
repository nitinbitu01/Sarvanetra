# sentinel/audit/router.py
"""
sentinel/audit/router.py — Audit log queries and chain verification endpoints.
"""

from datetime import datetime, timezone
from typing import Optional

import structlog
from fastapi import APIRouter, Depends, Query
from sqlalchemy import select, desc
from sqlalchemy.ext.asyncio import AsyncSession

from sentinel.auth.dependencies import SupervisorOrAdmin, AnyOfficer
from sentinel.audit.service import verify_audit_chain
from sentinel.db import get_db
from sentinel.models import AuditLog

log = structlog.get_logger(__name__)
router = APIRouter(prefix="/audit", tags=["audit"])


@router.get("/logs")
async def get_audit_logs(
    officer: AnyOfficer,
    event_type: Optional[str] = Query(None),
    actor_id: Optional[int] = Query(None),
    limit: int = Query(50, ge=1, le=500),
    offset: int = Query(0, ge=0),
    db: AsyncSession = Depends(get_db),
):
    """
    Retrieve paginated audit logs.
    """
    query = select(AuditLog)
    if event_type:
        query = query.where(AuditLog.event_type == event_type)
    if actor_id is not None:
        query = query.where(AuditLog.actor_id == actor_id)

    query = query.order_by(desc(AuditLog.id)).offset(offset).limit(limit)
    result = await db.execute(query)
    rows = result.scalars().all()

    return {
        "count": len(rows),
        "offset": offset,
        "limit": limit,
        "logs": [
            {
                "id": r.id,
                "event_type": r.event_type,
                "actor_id": r.actor_id,
                "actor_type": r.actor_type,
                "target_type": r.target_type,
                "target_id": r.target_id,
                "payload": r.payload,
                "ip_address": r.ip_address,
                "prev_hash": r.prev_hash,
                "row_hash": r.row_hash,
                "created_at": r.created_at.isoformat() if hasattr(r.created_at, "isoformat") else str(r.created_at),
            }
            for r in rows
        ],
    }


@router.post("/verify")
async def verify_chain(
    officer: SupervisorOrAdmin,
    db: AsyncSession = Depends(get_db),
):
    """
    Verify tamper-evident hash chain integrity across all audit log rows.
    """
    result = await verify_audit_chain(db)
    log.info("audit.verification_requested",
             officer_id=officer.id,
             valid=result["valid"],
             rows=result.get("rows_checked", 0))
    return result
