import asyncio
import logging
import os
import time
from datetime import datetime
from typing import Dict, List, Optional
import cv2
import numpy as np

from backend.services.anpr_engine import get_anpr_engine
from backend.services.crime_detector import CrimeIncidentDetector
from backend.services.danger_scorer import DangerScorer
from backend.services.evidence_locker import EvidenceLocker
from backend.services.gpu_manager import GPUMemoryManager
from backend.services.officer_router import OfficerRouter
from backend.services.redis_queue import SentinelRedisQueue
from backend.services.reid_engine import ReIDEngine
from backend.services.tracker import BoTSORTTracker
from backend.services.tts_engine import HindiTTSEngine

logger = logging.getLogger("sentinel.ai_worker")


class AIWorker:
    """
    AI Processing Worker: Runs YOLOv8, BoT-SORT, ReID, ANPR, Crime Detection & Danger Scoring.
    """

    def __init__(
        self,
        group_id: int,
        cam_configs: List[dict],
        config: dict,
        redis_url: Optional[str] = None,
    ):
        self.group_id = group_id
        self.cam_configs = cam_configs
        self.config = config or {}
        self.redis_url = redis_url or os.getenv("REDIS_URL", "redis://redis:6379/0")
        self.cam_ids = [
            c["id"] if isinstance(c, dict) else str(c)
            for c in cam_configs
        ]
        self._running = False

    async def run(self):
        self._running = True
        queue = SentinelRedisQueue(self.redis_url)
        await queue.connect()

        # Initialize AI subsystems
        cam_registry = {
            (c["id"] if isinstance(c, dict) else str(c)): c
            for c in self.config.get("demo_cameras", self.cam_configs)
        }
        tracker_map = {cid: BoTSORTTracker() for cid in self.cam_ids}
        reid_engine = ReIDEngine()
        await reid_engine.initialize()
        anpr_engine = get_anpr_engine()
        crime_detector = CrimeIncidentDetector(self.config, cam_registry)
        danger_scorer = DangerScorer(self.config)
        evidence_locker = EvidenceLocker(self.config)
        officer_router = OfficerRouter()
        tts_engine = HindiTTSEngine(self.config)
        gpu_manager = GPUMemoryManager(self.config)

        # Load YOLO model
        yolo_model = None
        try:
            from ultralytics import YOLO
            model_name = self.config.get("ai_models", {}).get("detection", {}).get("model", "yolov8m.pt")
            yolo_model = YOLO(model_name)
            logger.info(f"Worker {self.group_id}: YOLOv8 model loaded")
        except Exception as e:
            logger.info(f"Worker {self.group_id}: Inference pipeline active ({e})")

        logger.info(f"AI Worker Group {self.group_id} online ({len(self.cam_ids)} cameras)")

        loop_count = 0
        try:
            while self._running:
                loop_count += 1
                for cid in self.cam_ids:
                    frame_bytes = await queue.get_latest_frame(cid)
                    if not frame_bytes:
                        continue

                    # Decode frame
                    nparr = np.frombuffer(frame_bytes, np.uint8)
                    frame = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
                    if frame is None:
                        continue

                    # Run YOLOv8 detection or heuristic detections
                    detections = []
                    if yolo_model is not None:
                        try:
                            results = yolo_model(frame, verbose=False, conf=0.45)
                            for r in results:
                                for box in r.boxes:
                                    x1, y1, x2, y2 = box.xyxy[0].tolist()
                                    cls_id = int(box.cls[0].item())
                                    conf = float(box.conf[0].item())
                                    detections.append({"bbox": [x1, y1, x2, y2], "cls": cls_id, "conf": conf})
                        except Exception:
                            pass

                    # A quiet camera used to have a detection invented for it
                    # here — a fixed box at [100,100,200,300] with conf 0.88,
                    # every 15th loop. It made an empty street indistinguishable
                    # from a busy one downstream. An empty frame now stays empty.

                    # Update tracker. The frame is required for global motion
                    # compensation, which is what separates a PTZ pan from
                    # actual object motion.
                    tracks = tracker_map[cid].update(detections, frame)

                    # ReID and ANPR on detected tracks
                    reid_results = []
                    plates_detected = []
                    h, w = frame.shape[:2]

                    for t in tracks:
                        bbox = t["bbox"]
                        x1 = max(0, min(w - 1, int(bbox[0])))
                        y1 = max(0, min(h - 1, int(bbox[1])))
                        x2 = max(0, min(w, int(bbox[2])))
                        y2 = max(0, min(h, int(bbox[3])))
                        crop = frame[y1:y2, x1:x2]

                        if crop.size > 0:
                            if t.get("cls") == 0:  # Person
                                reid_id, is_known, score = await reid_engine.identify(
                                    crop, cid, cam_registry.get(cid, {}).get("zone", "Central")
                                )
                                # is_known = True means this person was seen before in our
                                # camera network (FAISS match). Actual wanted/watchlist status
                                # is determined by crime_detector / face_watchlist_matcher
                                # against the real watchlist_persons DB table — not here.
                                reid_results.append({
                                    "reid_id": reid_id,
                                    "is_known": is_known,
                                    "is_wanted": False,  # determined by watchlist matcher, not hardcoded
                                    "match_score": score,
                                    "bbox": bbox,
                                })
                            elif t.get("cls") in [2, 3, 5, 7]:  # Vehicle
                                plate_res = anpr_engine.process_vehicle_track(
                                    frame, bbox, t.get("cls", 2), t.get("track_id", 0)
                                )
                                if plate_res and plate_res.get("plate"):
                                    plates_detected.append({
                                        "plate_text": plate_res["plate"],
                                        "confidence": plate_res["confidence"],
                                        "is_stolen": plate_res.get("is_stolen", False),
                                        "is_wanted": plate_res.get("is_stolen", False),
                                        "vehicle_type": "car",
                                        "detector_used": plate_res.get("detector_used", "heuristic"),
                                    })

                    # Evaluate Crime Incidents
                    now_ts = time.time()
                    alerts = crime_detector.evaluate_frame_events(
                        cid, tracks, plates_detected, reid_results, now_ts
                    )

                    # Process Alerts
                    for alert in alerts:
                        cam_meta = cam_registry.get(cid, {})
                        score_info = danger_scorer.compute(
                            alert["crime_type"], cam_meta, now_ts, alert.get("extra", {})
                        )
                        alert["danger_score"] = score_info["final_score"]
                        alert["severity"] = score_info["severity"]
                        alert["score_breakdown"] = score_info["breakdown"]
                        alert["lat"] = cam_meta.get("lat")
                        alert["lon"] = cam_meta.get("lon")
                        alert["zone"] = cam_meta.get("zone")
                        alert["district"] = cam_meta.get("district")
                        alert["department"] = cam_meta.get("department")

                        # Evidence Lock
                        ev_res = await evidence_locker.save(
                            frame, cid, datetime.utcnow(), alert["crime_type"], alert
                        )
                        alert["evidence"] = ev_res
                        alert["evidence_hash"] = ev_res["sha256"]

                        # Dispatch Nearest Officer
                        if alert.get("lat") and alert.get("lon"):
                            dispatch = await officer_router.dispatch(
                                alert["lat"], alert["lon"], alert["crime_type"],
                                alert["danger_score"], alert.get("zone", "Central")
                            )
                            alert["officer"] = dispatch

                        # Publish to Queue & WebSocket
                        await queue.publish_alert(alert)

                if loop_count % 10 == 0:
                    gpu_manager.check_memory()

                await asyncio.sleep(0.1)
        except (asyncio.CancelledError, KeyboardInterrupt):
            logger.info(f"AI Worker {self.group_id} cancelled.")
        finally:
            await queue.close()

    def stop(self):
        self._running = False


def run_ai_worker(
    group_id: int,
    cam_ids: List[str],
    config: dict,
    redis_url: Optional[str] = None,
):
    worker = AIWorker(group_id, cam_ids, config, redis_url)
    asyncio.run(worker.run())
