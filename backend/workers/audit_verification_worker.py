"""
backend/workers/audit_verification_worker.py — Checkpointed Hash-Chain Verification Worker.
Periodically verifies cryptographic SHA-256 links in audit_log to detect tampering.
"""
import logging
from sqlalchemy.orm import Session
from backend.db.session import SessionLocal
from backend.services.audit_service import AuditService

logger = logging.getLogger("sentinel.audit_worker")


def run_audit_verification(batch_size: int = 500):
    db = SessionLocal()
    try:
        service = AuditService(db)
        res = service.verify_integrity(batch_size=batch_size)
        if res.get("skipped"):
            logger.info("[AUDIT WORKER] No new audit entries to verify.")
        elif res.get("integrity_ok"):
            logger.info(f"[AUDIT WORKER] Verified {res.get('verified_count')} entries. Integrity OK ✅")
        else:
            logger.critical(f"[AUDIT WORKER] Tamper detected! Broken links: {res.get('broken_links')}")
        return res
    finally:
        db.close()


if __name__ == "__main__":
    run_audit_verification()
