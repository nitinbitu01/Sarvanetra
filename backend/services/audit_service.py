"""
backend/services/audit_service.py — Cryptographic hash-chained audit logging
and incremental checkpoint verification service.
"""
import json
import hashlib
import logging
from datetime import datetime
from typing import Dict, Any, Optional, List
from sqlalchemy.orm import Session
from sqlalchemy import select, desc

from backend.db.models import AuditLog, AuditCheckpoint

logger = logging.getLogger("sentinel.audit")


class AuditService:
    def __init__(self, db: Session):
        self.db = db

    def log_event(
        self,
        event_type: str,
        entity_id: Optional[Any] = None,
        entity_type: Optional[str] = None,
        actor_id: Optional[str] = "SYSTEM",
        model_version: Optional[str] = None,
        payload: Optional[Dict[str, Any]] = None,
    ) -> AuditLog:
        payload = payload or {}
        payload_json = json.dumps(payload, sort_keys=True)
        payload_hash = hashlib.sha256(payload_json.encode("utf-8")).hexdigest()

        # Fetch previous log hash for chain integrity
        last_log = (
            self.db.query(AuditLog)
            .order_by(desc(AuditLog.id))
            .first()
        )
        prev_hash = last_log.payload_hash if last_log and last_log.payload_hash else "GENESIS_SENTINEL_GUJARAT"

        audit_entry = AuditLog(
            event_type=event_type,
            action=event_type,
            entity_id=str(entity_id) if entity_id is not None else None,
            entity_type=entity_type,
            actor_id=actor_id,
            user_id=actor_id,
            model_version=model_version,
            payload=payload,
            details=payload_json,
            payload_hash=payload_hash,
            prev_log_hash=prev_hash,
            created_at=datetime.utcnow(),
            logged_at=datetime.utcnow(),
        )
        self.db.add(audit_entry)
        self.db.commit()
        return audit_entry

    def verify_integrity(self, batch_size: int = 500) -> Dict[str, Any]:
        """
        Incrementally verifies cryptographic chain from last checkpoint.
        """
        checkpoint = self.db.query(AuditCheckpoint).filter(AuditCheckpoint.id == 1).first()
        if not checkpoint:
            checkpoint = AuditCheckpoint(id=1, last_verified_id=0, last_status="ok")
            self.db.add(checkpoint)
            self.db.commit()

        start_id = checkpoint.last_verified_id
        logs = (
            self.db.query(AuditLog)
            .filter(AuditLog.id > start_id)
            .order_by(AuditLog.id.asc())
            .limit(batch_size)
            .all()
        )

        if not logs:
            return {"verified_count": 0, "broken_links": [], "integrity_ok": True, "skipped": True}

        broken_links = []
        prev_expected_hash = None
        if start_id > 0:
            last_valid = self.db.query(AuditLog).filter(AuditLog.id == start_id).first()
            if last_valid:
                prev_expected_hash = last_valid.payload_hash

        for entry in logs:
            payload_json = json.dumps(entry.payload or {}, sort_keys=True)
            expected_hash = hashlib.sha256(payload_json.encode("utf-8")).hexdigest()

            if entry.payload_hash and entry.payload_hash != expected_hash:
                broken_links.append({
                    "id": entry.id,
                    "error": "payload_hash_mismatch",
                    "recorded": entry.payload_hash,
                    "expected": expected_hash,
                })

            if prev_expected_hash and entry.prev_log_hash and entry.prev_log_hash != prev_expected_hash:
                broken_links.append({
                    "id": entry.id,
                    "error": "prev_hash_broken_link",
                    "recorded": entry.prev_log_hash,
                    "expected": prev_expected_hash,
                })

            prev_expected_hash = entry.payload_hash

        # Update checkpoint
        checkpoint.last_verified_id = logs[-1].id
        checkpoint.last_verified_at = datetime.utcnow()
        checkpoint.last_status = "ok" if not broken_links else "broken"
        checkpoint.last_broken_links = broken_links if broken_links else None
        self.db.commit()

        return {
            "verified_count": len(logs),
            "broken_links": broken_links,
            "integrity_ok": len(broken_links) == 0,
            "last_verified_id": checkpoint.last_verified_id,
        }
