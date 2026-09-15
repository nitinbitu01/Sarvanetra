"""
backend/routers/v1/reid_governance.py â€” Data retention & privacy governance endpoints.

Endpoints:
  POST   /api/v1/reid/persons/{id}/hold    â€” set retention_hold=True (admin)
  DELETE /api/v1/reid/persons/{id}/hold    â€” remove retention_hold (admin)
  GET    /api/v1/reid/validation-report    â€” serve latest validation report markdown

The retention-hold mechanism is critical for lawful operation: it prevents the
automatic retention purge from deleting a GlobalPerson linked to an active
investigation or a watchlist match. Admins set/remove holds via this API.
The action is always audit-logged.
"""
from __future__ import annotations

import os
from pathlib import Path
from typing import Any

from fastapi import APIRouter, Depends, HTTPException, Request, status
from fastapi.responses import PlainTextResponse
from sqlalchemy.orm import Session

from backend.auth.dependencies import require_role
from backend.db.models import GlobalPerson, User
from backend.db.session import get_db
from backend.services.audit_logger import log_audit

router = APIRouter(prefix="/reid", tags=["ReID Governance"])

_PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent.parent
_REPORTS_DIR = _PROJECT_ROOT / "reports"

_admin = require_role("admin")


# â”€â”€ Retention hold â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€

@router.post(
    "/persons/{person_id}/hold",
    summary="Set retention hold â€” exempts record from automatic purge",
)
async def set_retention_hold(
    person_id: int,
    request: Request,
    db: Session = Depends(get_db),
    current_user: User = Depends(_admin),
) -> dict[str, Any]:
    """Set retention_hold=True on a GlobalPerson.

    While hold is active, the record will NOT be deleted by the retention
    purge job even if retention_expires_at has passed. Use for active
    investigations or watchlist-linked identities.
    """
    gp = db.query(GlobalPerson).filter(GlobalPerson.id == person_id).first()
    if not gp:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND,
                            detail=f"GlobalPerson {person_id} not found.")

    gp.retention_hold = True
    log_audit(
        db, user=current_user, action="REID_RETENTION_HOLD_SET",
        resource_type="global_person", resource_id=person_id,
        details={"legal_basis": gp.legal_basis},
        ip_address=request.client.host if request.client else None,
    )
    db.commit()
    return {"global_person_id": person_id, "retention_hold": True}


@router.delete(
    "/persons/{person_id}/hold",
    summary="Remove retention hold",
)
async def remove_retention_hold(
    person_id: int,
    request: Request,
    db: Session = Depends(get_db),
    current_user: User = Depends(_admin),
) -> dict[str, Any]:
    """Remove retention_hold from a GlobalPerson.

    Record will be eligible for automatic purge on its next scheduled run
    once retention_expires_at has passed.
    """
    gp = db.query(GlobalPerson).filter(GlobalPerson.id == person_id).first()
    if not gp:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND,
                            detail=f"GlobalPerson {person_id} not found.")

    gp.retention_hold = False
    log_audit(
        db, user=current_user, action="REID_RETENTION_HOLD_REMOVED",
        resource_type="global_person", resource_id=person_id,
        details={"retention_expires_at": (
            gp.retention_expires_at.isoformat() if gp.retention_expires_at else None
        )},
        ip_address=request.client.host if request.client else None,
    )
    db.commit()
    return {"global_person_id": person_id, "retention_hold": False}


# â”€â”€ Validation report serving â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€

@router.get(
    "/validation-report",
    summary="Get latest validation report markdown (admin only)",
    response_class=PlainTextResponse,
)
async def get_validation_report(
    current_user: User = Depends(_admin),
) -> str:
    """Return the contents of the most recent reid_validation_*.md report.

    If no report exists yet, returns an instructional message.
    Run scripts/reid_validate.py to generate one.
    """
    if not _REPORTS_DIR.exists():
        return (
            "# No Validation Report Found\n\n"
            "Run `python -m backend.scripts.reid_validate` to generate one.\n"
            "See Â§4 of the Day 6 implementation spec for details.\n"
        )

    reports = sorted(
        _REPORTS_DIR.glob("reid_validation_*.md"),
        key=lambda p: p.stat().st_mtime,
        reverse=True,
    )
    if not reports:
        return (
            "# No Validation Report Found\n\n"
            "No reid_validation_*.md file found in the reports/ directory.\n"
            "Run `python -m backend.scripts.reid_validate` to generate one.\n"
        )

    latest = reports[0]
    return latest.read_text(encoding="utf-8")
