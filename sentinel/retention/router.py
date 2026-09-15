# sentinel/retention/router.py
"""
sentinel/retention/router.py — Retention enforcement and GDPR erasure API endpoints.
"""

import structlog
from fastapi import APIRouter, Depends, HTTPException
from pydantic import BaseModel, EmailStr
from sqlalchemy.ext.asyncio import AsyncSession

from sentinel.auth.dependencies import AdminOnly
from sentinel.db import get_db
from sentinel.retention.service import enforce_retention_policies, handle_deletion_request

log = structlog.get_logger(__name__)
router = APIRouter(prefix="/retention", tags=["retention"])


class ErasureRequest(BaseModel):
    subject_email: str


@router.post("/enforce")
async def trigger_retention(
    admin: AdminOnly,
    db: AsyncSession = Depends(get_db),
):
    """
    Manually trigger data retention policy enforcement (Admin only).
    """
    return await enforce_retention_policies(db)


@router.post("/gdpr-erasure")
async def trigger_erasure(
    payload: ErasureRequest,
    admin: AdminOnly,
    db: AsyncSession = Depends(get_db),
):
    """
    Execute GDPR Article 17 Right-to-Erasure anonymization (Admin only).
    """
    result = await handle_deletion_request(
        subject_email=payload.subject_email,
        requested_by=admin.id,
        db=db,
    )
    if result.get("status") == "not_found":
        raise HTTPException(status_code=404, detail="Subject not found")
    return result
