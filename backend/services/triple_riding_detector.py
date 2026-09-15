"""
backend/services/triple_riding_detector.py — Production Triple Riding Detection Engine.

Integrates:
  - Weather regime enhancement
  - Vehicle class guard & homography speed gating (15 <= V <= 150 km/h)
  - Tri-modal rider counting (Hungarian + CLAHE Head-Peak + Haar tiebreaker)
  - TrackletMemory Re-ID buffer preservation (HSV histogram match >= 0.85)
  - Sliding-window temporal majority voting (R_k >= 0.65 over 40 frames)
  - Helmet detection & MV Act S128 + S129 fine computation
  - Async 4-panel court-admissible evidence builder with SHA-256 digests
  - Shadow Mode logging vs live event dispatch
"""
from __future__ import annotations

import asyncio
import hashlib
import os
import sqlite3
import time
import uuid
from dataclasses import dataclass
from typing import Any, Optional

import cv2
import numpy as np

from backend.monitoring.latency_tracker import LatencyTracker
from backend.preprocessing.frame_enhancer import WeatherRegime, enhance_frame

from .evidence_builder import build_collage_async
from .fine_calculator import ViolationBreakdown
from .helmet_detector import detect_helmets
from .rider_counter import BikeBox, PersonBox, count_riders
from .tracklet_memory import TrackletMemory

MOTORCYCLE_CLASSES = {"motorcycle", "scooter", "moped", "two_wheeler"}


@dataclass
class Detection:
    track_id: int
    bbox: tuple[float, float, float, float]  # x1,y1,x2,y2 pixels
    vehicle_class: str
    confidence: float
    speed_kmh: float = 0.0


@dataclass
class PersonDetection:
    bbox: tuple[float, float, float, float]
    confidence: float


@dataclass
class ViolationEvent:
    violation_uuid: str
    camera_id: str
    track_id: int
    rider_count: int
    majority_ratio: float
    counting_method: str
    speed_kmh: float
    license_plate: Optional[str]
    plate_confidence: float
    helmet_results: list[dict[str, Any]]
    fine_breakdown: dict[str, Any]
    collage_path: str
    collage_hash: str
    panel_hashes: dict[str, str]
    weather_regime: str
    timestamp: float
    challan_status: str = "pending_review"


def _iou_raw(
    a: tuple[float, float, float, float],
    b: tuple[float, float, float, float],
) -> float:
    ix1 = max(a[0], b[0])
    iy1 = max(a[1], b[1])
    ix2 = min(a[2], b[2])
    iy2 = min(a[3], b[3])
    inter = max(0.0, ix2 - ix1) * max(0.0, iy2 - iy1)
    if inter == 0.0:
        return 0.0
    aa = (a[2] - a[0]) * (a[3] - a[1])
    ab = (b[2] - b[0]) * (b[3] - b[1])
    overlap_a = inter / max(aa, 1e-6)
    union = aa + ab - inter
    standard_iou = inter / max(union, 1e-6)
    return max(standard_iou, overlap_a)



def _log_shadow(event: ViolationEvent, db_path: str = "sentinel.db") -> None:
    try:
        conn = sqlite3.connect(db_path)
        conn.execute(
            """
            CREATE TABLE IF NOT EXISTS shadow_violations (
                id              INTEGER PRIMARY KEY AUTOINCREMENT,
                violation_uuid  TEXT    UNIQUE NOT NULL,
                camera_id       TEXT    NOT NULL,
                track_id        INTEGER NOT NULL,
                rider_count     INTEGER NOT NULL,
                majority_ratio  REAL    NOT NULL,
                speed_kmh       REAL    NOT NULL,
                weather_regime  TEXT    NOT NULL,
                ground_truth    TEXT    NOT NULL DEFAULT 'pending',
                created_at      TIMESTAMP DEFAULT CURRENT_TIMESTAMP
            )
        """
        )
        conn.execute(
            """
            INSERT INTO shadow_violations
            (violation_uuid, camera_id, track_id, rider_count,
             majority_ratio, speed_kmh, weather_regime)
            VALUES (?, ?, ?, ?, ?, ?, ?)
        """,
            (
                event.violation_uuid,
                event.camera_id,
                event.track_id,
                event.rider_count,
                event.majority_ratio,
                event.speed_kmh,
                event.weather_regime,
            ),
        )
        conn.commit()
        conn.close()
    except Exception:
        pass


class TripleRidingDetector:
    """
    Per-camera stateful triple riding detection engine.
    One instance per camera stream.
    """

    def __init__(
        self,
        camera_id: str,
        H: np.ndarray,
        camera_angle_deg: float = 0.0,
        cfg: Optional[dict] = None,
        shadow_mode: bool = False,
    ):
        self.camera_id = camera_id
        self.H = H  # homography matrix
        self.camera_angle_deg = camera_angle_deg
        self.cfg = cfg or {}
        self.shadow_mode = shadow_mode
        self.memory = TrackletMemory()
        self.latency = LatencyTracker(camera_id)

    async def process_frame(
        self,
        raw_frame: np.ndarray,
        bikes: list[Detection],
        persons: list[PersonDetection],
        now: float,
    ) -> list[ViolationEvent]:
        events: list[ViolationEvent] = []

        with self.latency.measure("frame_enhance"):
            eframe = enhance_frame(raw_frame)
        frame = eframe.frame

        for bike_det in bikes:
            # Class guard
            if bike_det.vehicle_class not in MOTORCYCLE_CLASSES:
                continue

            bid = bike_det.track_id
            x1, y1, x2, y2 = bike_det.bbox
            crop = raw_frame[int(y1) : int(y2), int(x1) : int(x2)]

            # TrackState via Re-ID memory
            with self.latency.measure("reid"):
                state = self.memory.get_or_create(bid, crop, now)

            speed = bike_det.speed_kmh

            # Speed gate
            min_s = self.cfg.get("speed_gate_low_kmh", 15.0)
            max_s = self.cfg.get("speed_gate_high_kmh", 150.0)
            if not (min_s <= speed <= max_s):
                state.push(0)
                continue

            # Plate cache update (every 10th frame)
            if state.plate_cache.should_sample() and crop.size > 0:
                tmp_dir = "evidence/plate_crops"
                os.makedirs(tmp_dir, exist_ok=True)
                tmp_path = os.path.join(tmp_dir, f"tmp_crop_{bid}_{int(now*1000)}.jpg")
                cv2.imwrite(tmp_path, crop)
                await state.plate_cache.update_async(tmp_path, 0.75)


            # Rider count (three-path)
            bike_box = BikeBox(x1, y1, x2, y2)
            person_boxes = [
                PersonBox(p.bbox[0], p.bbox[1], p.bbox[2], p.bbox[3], p.confidence)
                for p in persons
            ]
            with self.latency.measure("rider_count"):
                n_riders, method = count_riders(
                    frame, bike_box, person_boxes, self.camera_angle_deg
                )

            state.push(n_riders)

            # Warmup guard + violation check
            if not state.is_warmed_up():
                continue
            if not state.is_violation():
                continue
            if not state.can_alert(now):
                continue

            # ── Violation confirmed ───────────────────────────────────
            state.record_alert(now)
            r_ratio = state.majority_ratio
            vid_uuid = str(uuid.uuid4()).replace("-", "")[:20].upper()

            # Helmet detection on rider crops (sync, fast)
            rider_bboxes = [
                (p.bbox[0], p.bbox[1], p.bbox[2], p.bbox[3])
                for p in persons
                if _iou_raw(p.bbox, bike_det.bbox) >= 0.20
            ][:n_riders]

            with self.latency.measure("helmet_detect"):
                helmet_res = detect_helmets(frame, rider_bboxes)

            # Fine calculation
            breakdown = ViolationBreakdown(
                rider_count=n_riders,
                helmet_results=helmet_res,
            )
            breakdown.compute()

            # Best plate
            plate_res = state.plate_cache.get_best()
            plate_text = plate_res.text if plate_res else None
            plate_conf = plate_res.confidence if plate_res else 0.0

            # Async evidence collage
            out_path = f"evidence/collages/{self.camera_id}_{vid_uuid}.jpg"
            meta = {
                "camera_id": self.camera_id,
                "ts": time.strftime("%Y-%m-%d %H:%M:%S", time.localtime(now)),
                "speed": speed,
                "rider_count": n_riders,
                "majority_ratio": r_ratio,
                "plate": plate_text or "",
                "plate_conf": plate_conf,
            }
            # Wide context panel
            p1 = frame.copy()
            # Zoomed bike panel
            p2 = frame[max(0, int(y1)) : int(y2), max(0, int(x1)) : int(x2)].copy()
            p2 = p2 if p2.size > 0 else np.zeros((100, 100, 3), np.uint8)
            # Plate panel
            p3 = (
                cv2.imread(plate_res.crop_path)
                if (plate_res and plate_res.crop_path and os.path.exists(plate_res.crop_path))
                else np.zeros((80, 200, 3), np.uint8)
            )
            # Helmet panel — stack rider head annotations
            p4 = np.zeros((120, 200, 3), np.uint8)
            for i, hr in enumerate(helmet_res):
                txt = f"R{i+1}:{hr.label}({hr.confidence:.0%})"
                cv2.putText(
                    p4,
                    txt,
                    (5, 20 + i * 25),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.55,
                    (0, 255, 0) if hr.has_helmet else (0, 0, 255),
                    1,
                )

            evidence_pkg = await build_collage_async(p1, p2, p3, p4, meta, out_path)

            event = ViolationEvent(
                violation_uuid=vid_uuid,
                camera_id=self.camera_id,
                track_id=bid,
                rider_count=n_riders,
                majority_ratio=r_ratio,
                counting_method=method,
                speed_kmh=speed,
                license_plate=plate_text,
                plate_confidence=plate_conf,
                helmet_results=[
                    {
                        "rider_index": h.rider_index,
                        "has_helmet": h.has_helmet,
                        "confidence": h.confidence,
                        "label": h.label,
                    }
                    for h in helmet_res
                ],
                fine_breakdown=breakdown.to_dict(),
                collage_path=evidence_pkg.collage_path,
                collage_hash=evidence_pkg.collage_hash,
                panel_hashes=evidence_pkg.panel_source_hashes,
                weather_regime=eframe.regime.name,
                timestamp=now,
            )

            if self.shadow_mode:
                _log_shadow(event)
            else:
                events.append(event)

        return events
