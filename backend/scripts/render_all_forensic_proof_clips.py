"""
backend/scripts/render_all_forensic_proof_clips.py — Re-renders authentic forensic evidence proof clips
for all active alerts in the database with dynamic YOLO suspect targeting reticles.
"""

from __future__ import annotations

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
logger = logging.getLogger("render_proof_clips")


def main():
    db = SessionLocal()
    try:
        # Delete stale evidence rows to force re-render with upgraded YOLO targeting HUD
        logger.info("Purging stale evidence rows to force real YOLO forensic re-render...")
        db.query(Evidence).delete()
        db.commit()

        alerts = db.query(Alert).filter(Alert.is_deleted == False).order_by(Alert.created_at.desc()).all()
        logger.info(f"Rendering authentic forensic proof clips for {len(alerts)} alerts...")

        success_count = 0
        failed_count = 0

        for idx, alert in enumerate(alerts):
            aid = str(alert.id)
            logger.info(f"[{idx+1}/{len(alerts)}] Rendering YOLO suspect-targeting clip for alert={aid} ({alert.alert_type} @ {alert.camera_id})...")
            status = _capture_sync(aid)
            if status == "COMPLETE":
                success_count += 1
            else:
                failed_count += 1

        logger.info(f"Forensic Proof Clips Render Complete: {success_count} SUCCESS, {failed_count} FAILED.")
    finally:
        db.close()


if __name__ == "__main__":
    main()
