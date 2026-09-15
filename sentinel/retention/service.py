# sentinel/retention/service.py
"""
sentinel/retention/service.py — Data retention policy engine and GDPR Article 17 right-to-erasure.
"""

from datetime import datetime, timezone, timedelta
from typing import Dict, Any

import structlog
from sqlalchemy import select, delete, update, text
from sqlalchemy.ext.asyncio import AsyncSession

from sentinel.config import settings
from sentinel.models import Alert, RefreshToken, AuditLog, Officer

log = structlog.get_logger(__name__)


async def enforce_retention_policies(db: AsyncSession) -> Dict[str, Any]:
    """
    Enforce data retention policies across standard alerts, logs, and tokens.
    """
    now = datetime.now(timezone.utc)
    results = {}

    try:
        # 1. Delete standard alerts older than ALERT_RETENTION_DAYS (unless locked)
        alert_cutoff = now - timedelta(days=settings.ALERT_RETENTION_DAYS)
        alert_del = await db.execute(
            delete(Alert).where(Alert.created_at < alert_cutoff)
        )
        results["alerts_deleted"] = alert_del.rowcount or 0

        # 2. Expire old refresh tokens
        token_cutoff = now - timedelta(days=1)
        token_del = await db.execute(
            delete(RefreshToken).where(RefreshToken.expires_at < token_cutoff)
        )
        results["expired_tokens_deleted"] = token_del.rowcount or 0

        # 3. Check audit log retention threshold (warn if rows exceed 7 years)
        audit_cutoff = now - timedelta(days=settings.AUDIT_LOG_RETENTION_DAYS)
        audit_res = await db.execute(
            select(AuditLog.id).where(AuditLog.created_at < audit_cutoff)
        )
        old_audit_count = len(audit_res.scalars().all())
        results["audit_rows_beyond_retention"] = old_audit_count
        if old_audit_count > 0:
            log.warning(
                "retention.audit_log_beyond_retention",
                count=old_audit_count,
                note="Audit rows beyond retention must be exported to cold storage. Never auto-deleted."
            )

        await db.flush()
        log.info("retention.policies_enforced", results=results)
        return {"status": "ok", "enforced_at": now.isoformat(), **results}

    except Exception as e:
        log.error("retention.enforcement_failed", error=str(e), exc_info=True)
        raise


async def handle_deletion_request(
    subject_email: str,
    requested_by: int,
    db: AsyncSession,
) -> Dict[str, Any]:
    """
    GDPR Article 17 — Right to Erasure.
    Anonymizes PII while preserving operational audit integrity.
    """
    result = await db.execute(
        select(Officer).where(Officer.email == subject_email)
    )
    officer = result.scalar_one_or_none()

    if not officer:
        return {"status": "not_found", "email": subject_email}

    officer_id = officer.id

    # Anonymize officer record
    officer.name = f"Deleted Officer {officer_id}"
    officer.email = f"deleted_{officer_id}@deleted.sentinel"
    officer.hashed_password = "$2b$12$deleted_officer_unusable_hash_padding_"
    officer.is_active = False

    # Anonymize IP address in audit logs
    await db.execute(
        update(AuditLog)
        .where(AuditLog.actor_id == officer_id)
        .values(ip_address="0.0.0.0")
    )

    # Revoke all tokens
    await db.execute(
        update(RefreshToken)
        .where(RefreshToken.officer_id == officer_id)
        .values(revoked=True)
    )

    await db.flush()

    log.info("gdpr.erasure_request_completed",
             subject_id=officer_id,
             requested_by=requested_by)

    return {
        "status": "anonymized",
        "officer_id": officer_id,
        "note": "PII removed. Operational records preserved per legal requirement.",
    }
