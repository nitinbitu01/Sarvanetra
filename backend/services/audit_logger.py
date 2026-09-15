"""
backend/services/audit_logger.py — Audit log helper.

All security-relevant actions write a row to audit_log.
Called from route handlers — never from pipeline threads.
"""
from __future__ import annotations

import json
from typing import Any

from sqlalchemy.orm import Session

from backend.db.models import AuditLog, User


def log_audit(
    db: Session,
    user: User | None,
    action: str,
    resource_type: str | None = None,
    resource_id: int | None = None,
    details: dict[str, Any] | None = None,
    ip_address: str | None = None,
) -> AuditLog:
    """Insert an audit log row and flush (but don't commit — caller owns the tx).

    Args:
        db: Active SQLAlchemy session.
        user: The User performing the action (None for system actions).
        action: e.g. 'CAMERA_ADD', 'CAMERA_SOFT_DELETE', 'LOGIN', 'EVIDENCE_VERIFY'.
        resource_type: e.g. 'camera', 'alert', 'user'.
        resource_id: PK of the affected resource.
        details: Free-form dict serialized to JSON.
        ip_address: Client IP from request.

    Returns:
        The created AuditLog instance (id populated after flush).
    """
    entry = AuditLog(
        user_id=user.id if user else None,
        action=action,
        resource_type=resource_type,
        resource_id=resource_id,
        details=json.dumps(details) if details else None,
        ip_address=ip_address,
    )
    db.add(entry)
    db.flush()   # populate entry.id without committing — caller commits
    return entry
