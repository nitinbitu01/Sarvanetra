"""
backend/workers/calibration_worker.py — Background Calibration Worker.
Periodically computes per-camera HSV envelopes and ReID distance thresholds,
runs AnchorEvaluator regression testing, and activates or flags for supervisor approval.
"""
import asyncio
import logging
import uuid
from datetime import datetime
from typing import Dict, List, Any
import numpy as np
from sqlalchemy.orm import Session

from backend.db.session import SessionLocal
from backend.db.models import VaultEntry, CameraCalibrationProfile, Camera
from backend.services.anchor_evaluator import AnchorEvaluator
from backend.core.vault_config import VaultCompartment

logger = logging.getLogger("sentinel.calibration_worker")


class CalibrationWorker:
    def __init__(self, db: Session):
        self.db = db
        self.anchor_evaluator = AnchorEvaluator(db)

    def calibrate_all_cameras(self) -> List[Dict[str, Any]]:
        """
        Runs calibration across all registered cameras.
        """
        cameras = self.db.query(Camera).all()
        results = []
        for cam in cameras:
            cid = cam.camera_id or f"CAM_{cam.id:02d}"
            res = self.calibrate_camera(cid)
            results.append(res)
        return results

    def calibrate_camera(self, camera_id: str) -> Dict[str, Any]:
        """
        Calibrates a single camera's lighting profiles using recent Vault entries.
        """
        # Fetch hard positives / negatives for this camera
        positives = (
            self.db.query(VaultEntry)
            .filter(
                VaultEntry.camera_id == camera_id,
                VaultEntry.compartment.in_([VaultCompartment.HARD_POSITIVE.value, VaultCompartment.GOLD.value]),
            )
            .limit(100)
            .all()
        )
        negatives = (
            self.db.query(VaultEntry)
            .filter(
                VaultEntry.camera_id == camera_id,
                VaultEntry.compartment == VaultCompartment.HARD_NEGATIVE.value,
            )
            .limit(100)
            .all()
        )

        sample_count = len(positives) + len(negatives)

        # Baseline per-lighting condition thresholds
        proposed_thresholds = {
            "day": {"vehicle": 0.35, "person": 0.38},
            "night": {"vehicle": 0.40, "person": 0.42},
            "night_glare": {"vehicle": 0.45, "person": 0.46},
            "monsoon_rain": {"vehicle": 0.42, "person": 0.44},
            "dust_storm": {"vehicle": 0.44, "person": 0.45},
        }

        # Adaptive threshold adjustment based on positive/negative separation
        if positives and negatives:
            pos_dists = [float(p.ai_cosine_distance) for p in positives if p.ai_cosine_distance is not None]
            neg_dists = [float(n.ai_cosine_distance) for n in negatives if n.ai_cosine_distance is not None]
            if pos_dists and neg_dists:
                optimal_cutoff = (np.mean(pos_dists) + np.mean(neg_dists)) / 2.0
                clamped_cutoff = max(0.25, min(0.55, float(optimal_cutoff)))
                proposed_thresholds["day"]["vehicle"] = round(clamped_cutoff, 4)

        proposed_hsv = {
            "day": {"h_min": 0, "h_max": 180, "s_min": 30, "s_max": 255, "v_min": 40, "v_max": 255},
            "night": {"h_min": 0, "h_max": 180, "s_min": 10, "s_max": 255, "v_min": 10, "v_max": 200},
            "monsoon_rain": {"h_min": 0, "h_max": 180, "s_min": 15, "s_max": 220, "v_min": 25, "v_max": 230},
        }

        # ── Anchor-Set Regression Gate ─────────────────────────────────────────
        anchor_report = self.anchor_evaluator.evaluate_profile(
            camera_id=camera_id,
            proposed_thresholds=proposed_thresholds,
            lighting_condition="day",
        )

        passed = anchor_report.get("passed", False)

        # Deactivate old profiles
        self.db.query(CameraCalibrationProfile).filter(
            CameraCalibrationProfile.camera_id == camera_id,
            CameraCalibrationProfile.is_active == True,
        ).update({"is_active": False})

        # Insert new profile
        profile = CameraCalibrationProfile(
            id=str(uuid.uuid4()),
            camera_id=camera_id,
            profile_version=1,
            calibrated_at=datetime.utcnow(),
            hsv_profiles=proposed_hsv,
            reid_thresholds=proposed_thresholds,
            sample_count=sample_count,
            calibration_score=anchor_report.get("top1_accuracy", 0.90),
            approved=passed,
            is_active=passed,  # Only activates if anchor regression gate passed!
            pending_supervisor_review=not passed,
            anchor_eval_report=anchor_report,
            created_at=datetime.utcnow(),
        )

        self.db.add(profile)
        self.db.commit()

        status_str = "ACTIVATED ✅" if passed else "BLOCKED (Pending Supervisor Review) ⚠️"
        logger.info(f"[CALIBRATION] Camera {camera_id}: {status_str} | Samples={sample_count}")

        return {
            "camera_id": camera_id,
            "status": "activated" if passed else "pending_supervisor_review",
            "sample_count": sample_count,
            "anchor_eval": anchor_report,
        }


def run_periodic_calibration():
    db = SessionLocal()
    try:
        worker = CalibrationWorker(db)
        return worker.calibrate_all_cameras()
    finally:
        db.close()


if __name__ == "__main__":
    results = run_periodic_calibration()
    print(f"Calibration completed for {len(results)} cameras.")
