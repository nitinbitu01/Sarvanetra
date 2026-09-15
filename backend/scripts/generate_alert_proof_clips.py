"""backend/scripts/generate_alert_proof_clips.py

Iterates over alerts in the database, extracts an authentic CCTV clip from
the corresponding camera's video footage, computes SHA-256 hash, and seals
the record into the Evidence table and Alert table.
"""
import os
import sys
import logging
from pathlib import Path

# Add project root to sys.path
root_dir = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(root_dir))

from backend.db.session import SessionLocal
from backend.db.models import Alert, Evidence
from backend.services.evidence_capture import _capture_sync

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger("generate_proof_clips")

def main():
    db = SessionLocal()
    try:
        alerts = db.query(Alert).filter(Alert.is_deleted == False).order_by(Alert.created_at.desc()).all()
        logger.info(f"Found {len(alerts)} alerts in database to evaluate for proof clips.")
        
        success_count = 0
        skipped_count = 0
        failed_count = 0

        for idx, alert in enumerate(alerts):
            aid = str(alert.id)
            ev = db.query(Evidence).filter(Evidence.alert_id == aid, Evidence.status == "COMPLETE").first()
            if ev and ev.clip_path and (root_dir / ev.clip_path).exists():
                skipped_count += 1
                continue
            
            logger.info(f"[{idx+1}/{len(alerts)}] Generating authentic CCTV proof clip for alert={aid} (cam={alert.camera_id})...")
            status = _capture_sync(aid)
            if status == "COMPLETE":
                success_count += 1
            else:
                failed_count += 1

        logger.info(f"Summary: {success_count} clips generated, {skipped_count} already complete, {failed_count} failed.")
    finally:
        db.close()

if __name__ == "__main__":
    main()
