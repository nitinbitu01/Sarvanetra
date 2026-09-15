"""
backend/workers/central_reid_worker.py
Central ReID & Threat Intelligence Worker.
Coordinates:
- Vehicle ReID (ResNet50 + Color/Type + CLM Gate)
- Person ReID (OSNet + Attributes + CLM Gate)
- Danger Score Engine (Composite 0.0-1.0 threat level)
- Evidence Vault (SHA-256 + HMAC auto-sealing for high threat events)
- Dispatch Router (GPS nearest officer push)
"""

import asyncio
import logging
from typing import Optional, Dict, Any

from backend.services.camera_link_model import CameraLinkModel
from backend.services.vehicle_reid_engine import VehicleReIDEngine
from backend.services.person_reid_engine import PersonReIDEngine
from backend.services.danger_score_engine import DangerScoreEngine, DangerContext, AlertPriority
from backend.services.evidence_vault import EvidenceVault
from backend.services.dispatch_router import DispatchRouter

logger = logging.getLogger("sentinel.central_worker")


class CentralReIDWorker:
    def __init__(self, db=None, ws_manager=None):
        self.db = db
        self.ws = ws_manager
        self.clm = CameraLinkModel(mode="mixed", db=db)
        self.vehicle_reid = VehicleReIDEngine(self.clm)
        self.person_reid = PersonReIDEngine(self.clm)
        self.danger_engine = DangerScoreEngine()
        self.evidence_vault = EvidenceVault(db=db)
        self.dispatch_router = DispatchRouter(db=db, ws_manager=ws_manager)
        self.is_running = False

    async def process_detection(self, detection: dict) -> Optional[dict]:
        """
        Main pipeline execution on every detection.
        """
        obj_type = detection.get("object_type", "person").lower()
        crop = detection.get("crop")
        cam_id = int(detection.get("camera_id", 1))
        track_id = str(detection.get("track_id", "0"))
        ts = detection.get("timestamp_utc", 0.0)
        plate_text = detection.get("plate_text")

        matches = []
        if obj_type in ["car", "truck", "bus", "vehicle", "motorcycle", "rickshaw"]:
            emb = self.vehicle_reid.extract_embedding(
                vehicle_crop=crop,
                track_id=track_id,
                camera_id=cam_id,
                timestamp_utc=ts,
                plate_text=plate_text
            )
            matches = self.vehicle_reid.search_vehicle(emb)
            reid_id = self.vehicle_reid.add_to_index(emb)
        else:
            emb = self.person_reid.extract_embedding(
                person_crop=crop,
                track_id=track_id,
                camera_id=cam_id,
                timestamp_utc=ts
            )
            matches = self.person_reid.search_person(emb)
            reid_id = self.person_reid.add_to_index(emb)

        # Danger Scoring
        top_match = matches[0] if matches else None
        visual_score = top_match.final_score if top_match else 0.0
        is_watchlist = detection.get("is_watchlist", False)
        is_stolen = detection.get("is_stolen", False)

        ctx = DangerContext(
            is_on_watchlist=is_watchlist,
            watchlist_severity="WANTED" if is_watchlist else "NONE",
            is_stolen_vehicle=is_stolen,
            visual_match_score=visual_score,
            match_type="plate_confirmed" if plate_text else "visual_reid",
            camera_id=cam_id,
            zone_risk_level=detection.get("zone_risk_level", 1),
            num_cameras_in_30min=len(matches) + 1
        )

        danger = self.danger_engine.compute(ctx)

        alert_record = None
        if danger.should_alert:
            logger.info(
                f"CENTRAL ALERT: {obj_type.upper()} ({reid_id}) | "
                f"Score: {danger.total} ({danger.priority.name}) | {danger.explanation}"
            )

            # Auto seal evidence if HIGH or CRITICAL
            evidence_rec = None
            if danger.evidence_required and crop is not None:
                try:
                    evidence_rec = await self.evidence_vault.seal_evidence(
                        alert_id=reid_id,
                        camera_id=cam_id,
                        frames=[crop],
                        fps=25.0
                    )
                except Exception as e:
                    logger.warning(f"Evidence seal error: {e}")

            # Auto dispatch nearest officer if HIGH / CRITICAL
            if danger.priority >= AlertPriority.HIGH:
                try:
                    cam_lat = float(detection.get("lat", 23.0225))
                    cam_lon = float(detection.get("lon", 72.5714))
                    await self.dispatch_router.dispatch_alert(
                        alert_id=reid_id,
                        danger_score=danger,
                        camera_lat=cam_lat,
                        camera_lon=cam_lon,
                        evidence_id=evidence_rec.evidence_id if evidence_rec else None
                    )
                except Exception as e:
                    logger.warning(f"Dispatch error: {e}")

            alert_record = {
                "alert_id": reid_id,
                "object_type": obj_type,
                "danger_score": danger.total,
                "priority": danger.priority.name,
                "explanation": danger.explanation,
                "matches_count": len(matches),
            }

        return alert_record
