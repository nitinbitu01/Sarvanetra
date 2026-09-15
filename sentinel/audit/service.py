# sentinel/audit/service.py
"""
sentinel/audit/service.py — Append-only hash-chained audit logging and verification.
"""

import asyncio
import hashlib
import json
from datetime import datetime, timezone
from typing import Optional, Dict, Any

import httpx
import structlog
from sqlalchemy import select, text
from sqlalchemy.ext.asyncio import AsyncSession

from sentinel.config import settings
from sentinel.models import AuditLog

log = structlog.get_logger(__name__)

MAX_PAYLOAD_BYTES = 10_240


def _compute_row_hash(
    prev_hash: str,
    event_type: str,
    actor_id: Optional[int],
    target_id: Optional[int],
    payload: Optional[str],
    created_at: Any,
) -> str:
    """
    SHA-256 hash of the concatenated fields of this row.
    Deterministic and chained to prev_hash.
    """
    if isinstance(created_at, str):
        ts_str = created_at.replace(" ", "T").split("+")[0].split("Z")[0]
    elif hasattr(created_at, "strftime"):
        ts_str = created_at.strftime("%Y-%m-%dT%H:%M:%S.%f")
    else:
        ts_str = str(created_at)

    content = (
        f"{prev_hash}|"
        f"{event_type}|"
        f"{actor_id or ''}|"
        f"{target_id or ''}|"
        f"{payload or ''}|"
        f"{ts_str}"
    )
    return hashlib.sha256(content.encode("utf-8")).hexdigest()


async def _get_last_hash(db: AsyncSession) -> str:
    """
    Get the row_hash of the most recent audit log entry.
    Returns '0' * 64 for the genesis row.
    """
    result = await db.execute(
        select(AuditLog.row_hash).order_by(AuditLog.id.desc()).limit(1)
    )
    row = result.scalar_one_or_none()
    return row if row else "0" * 64


async def log_event(
    db: AsyncSession,
    event_type: str,
    actor_id: Optional[int] = None,
    actor_type: str = "officer",
    target_type: Optional[str] = None,
    target_id: Optional[int] = None,
    payload: Optional[dict] = None,
    ip_address: Optional[str] = None,
) -> bool:
    """
    Append a hash-chained audit event into the current transaction.
    """
    payload_str = None
    if payload:
        try:
            payload_str = json.dumps(payload, default=str)
            if len(payload_str.encode("utf-8")) > MAX_PAYLOAD_BYTES:
                log.warning("audit.payload_too_large",
                            event_type=event_type,
                            size=len(payload_str.encode("utf-8")))
                payload_str = json.dumps({
                    "_truncated": True,
                    "_reason": "payload exceeded 10KB limit",
                    "event_type": event_type,
                })
        except Exception as e:
            log.error("audit.payload_serialization_failed", error=str(e))
            payload_str = json.dumps({"_error": "payload not serializable"})

    now = datetime.now(timezone.utc)

    try:
        prev_hash = await _get_last_hash(db)
        row_hash = _compute_row_hash(
            prev_hash, event_type, actor_id, target_id, payload_str, now
        )

        audit_row = AuditLog(
            event_type=event_type,
            actor_id=actor_id,
            actor_type=actor_type,
            target_type=target_type,
            target_id=target_id,
            payload=payload_str,
            ip_address=ip_address,
            prev_hash=prev_hash,
            row_hash=row_hash,
            created_at=now,
        )
        db.add(audit_row)
        await db.flush()

        if settings.EXTERNAL_LOG_SINK_URL:
            try:
                loop = asyncio.get_running_loop()
                loop.create_task(_send_to_external_sink(audit_row))
            except Exception:
                pass

        return True

    except Exception as e:
        log.error("audit.write_failed",
                  event_type=event_type,
                  error=str(e),
                  exc_info=True)
        return False


async def _send_to_external_sink(audit_row: AuditLog) -> None:
    """
    Send audit event to external immutable log sink (fire-and-forget).
    """
    if not settings.EXTERNAL_LOG_SINK_URL:
        return
    try:
        async with httpx.AsyncClient(timeout=5.0) as client:
            await client.post(
                settings.EXTERNAL_LOG_SINK_URL,
                json={
                    "row_hash": audit_row.row_hash,
                    "prev_hash": audit_row.prev_hash,
                    "event_type": audit_row.event_type,
                    "actor_id": audit_row.actor_id,
                    "created_at": audit_row.created_at.isoformat() if hasattr(audit_row.created_at, "isoformat") else str(audit_row.created_at),
                },
                headers={"Content-Type": "application/json"},
            )
    except Exception as e:
        log.warning("audit.external_sink_failed", error=str(e))


async def verify_audit_chain(db: AsyncSession) -> Dict[str, Any]:
    """
    Verify the hash chain integrity of the entire audit log.
    """
    result = await db.execute(
        select(AuditLog).order_by(AuditLog.id.asc())
    )
    rows = result.scalars().all()

    if not rows:
        return {"valid": True, "rows_checked": 0}

    prev_hash = "0" * 64
    for idx, row in enumerate(rows):
        expected_hash = _compute_row_hash(
            prev_hash=prev_hash,
            event_type=row.event_type,
            actor_id=row.actor_id,
            target_id=row.target_id,
            payload=row.payload,
            created_at=row.created_at,
        )
        if expected_hash != row.row_hash:
            log.critical("audit.chain_broken",
                         row_id=row.id,
                         expected=expected_hash,
                         actual=row.row_hash)
            return {
                "valid": False,
                "first_broken_at": row.id,
                "rows_checked": idx + 1,
            }
        prev_hash = row.row_hash

    return {"valid": True, "rows_checked": len(rows)}
