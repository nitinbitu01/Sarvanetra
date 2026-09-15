"""
backend/services/wrong_way_detector.py — Production Wrong-Way Driving Detection Engine.

Key Capabilities:
  - Homography projection to world-plane meters.
  - ConstantVelocityKalman trajectory smoothing with dynamic dt.
  - Sticky polygonal lane zones with straight/curved polyline support.
  - Dual speed gating: 10.0 km/h <= speed <= 150.0 km/h (eliminates stationary jitter and ID-swap spikes).
  - Cosine direction threshold: cos alpha < -0.50 (angle discrepancy alpha > 120 deg).
  - Wall-clock time debounce: >= 800ms sustained violation streak.
  - 30-second per-track cooldown against duplicate records.
  - Automatic SHA-256 evidence lock and 3-stage Indian ANPR license plate lookup.
"""
from __future__ import annotations

import logging
import time
from dataclasses import dataclass
from typing import Any, Optional

import numpy as np

from .anpr import PlateResult, read_plate
from .calibration import pixel_to_world
from .evidence import lock_evidence
from .track_state import TrackState, Zone

logger = logging.getLogger(__name__)

SPEED_GATE_LOW_KMH   = 10.0     # below: stationary / jitter
SPEED_GATE_HIGH_KMH  = 150.0    # above: ID-swap artifact
COSINE_THRESHOLD     = -0.50    # cos alpha < -0.50 -> wrong-way (alpha > 120 deg)
DEBOUNCE_MS          = 800.0
ANPR_MIN_CONFIDENCE  = 0.75


@dataclass
class ViolationEvent:
    track_id: int
    zone_id: str
    speed_kmh: float
    discrepancy_angle_deg: float
    plate: PlateResult
    evidence_clip_hash: str
    evidence_clip_path: str
    crop_path: str
    vehicle_class: str
    camera_id: str
    timestamp: float


class WrongWayDetector:
    """Stateful per-camera wrong-way detection engine."""

    def __init__(
        self,
        camera_id: str,
        H: np.ndarray,
        zones: list[Zone],
        config: dict[str, Any] | None = None,
    ) -> None:
        self.camera_id = camera_id
        self.H = H
        self.zones = zones
        self.cfg = config or {
            "speed_gate_kmh": SPEED_GATE_LOW_KMH,
            "speed_max_kmh": SPEED_GATE_HIGH_KMH,
            "cosine_violation_threshold": COSINE_THRESHOLD,
            "debounce_ms": DEBOUNCE_MS,
            "anpr_min_confidence": ANPR_MIN_CONFIDENCE,
        }
        self._tracks: dict[int, TrackState] = {}

    def process_detections(
        self,
        detections: list[dict[str, Any]],   # [{track_id, bbox, vehicle_class}]
        frame: np.ndarray,                  # raw video frame
        now: float | None = None,           # wall-clock time
    ) -> list[ViolationEvent]:
        """Process one frame's worth of tracker output."""
        ts = now if now is not None else time.monotonic()
        active_ids = set()
        events = []

        for det in detections:
            tid = det.get("track_id")
            if tid is None:
                continue
            active_ids.add(tid)

            # Centroid in pixel space
            bbox = det.get("bbox", [0, 0, 0, 0])
            x1, y1, x2, y2 = bbox
            cx, cy = (x1 + x2) / 2.0, (y1 + y2) / 2.0

            # Project to world meters
            wx, wy = pixel_to_world(self.H, cx, cy)

            if tid not in self._tracks:
                self._tracks[tid] = TrackState(
                    tid, wx, wy, ts,
                    warmup_frames=self.cfg.get("track_warmup_frames"),
                    violation_cooldown_sec=self.cfg.get("violation_cooldown_sec"),
                )

            track = self._tracks[tid]
            track.update(wx, wy, ts, self.zones)

            event = self._evaluate(track, det, frame, ts)
            if event:
                events.append(event)

        # Evict stale tracks
        stale = [tid for tid in self._tracks if tid not in active_ids]
        for tid in stale:
            del self._tracks[tid]

        return events

    def _evaluate(
        self,
        track: TrackState,
        det: dict[str, Any],
        frame: np.ndarray,
        now: float,
    ) -> Optional[ViolationEvent]:
        # 1. Warm-up guard (10 frames)
        if not track.is_ready_for_detection():
            return None

        # 2. Must be inside an assigned zone
        if not track.in_zone():
            return None

        speed = track.kalman.speed_kmh
        min_spd = self.cfg.get("speed_gate_kmh", SPEED_GATE_LOW_KMH)
        max_spd = self.cfg.get("speed_max_kmh", SPEED_GATE_HIGH_KMH)

        # 3. Upper & lower speed gates
        if not (min_spd <= speed <= max_spd):
            track.update_debounce(False, now, self.cfg.get("debounce_ms", DEBOUNCE_MS))
            return None

        # 4. Heading from Kalman filter
        heading = track.kalman.heading_unit_vector
        if heading is None:
            track.update_debounce(False, now, self.cfg.get("debounce_ms", DEBOUNCE_MS))
            return None

        # 5. Local flow vector at vehicle's world position
        zone = track.assigned_zone
        wx, wy = track.world_pos
        flow = zone.local_flow_vector(wx, wy)

        # 6. Direction cosine dot product
        cos_alpha = float(heading[0] * flow[0] + heading[1] * flow[1])
        thresh = self.cfg.get("cosine_violation_threshold", COSINE_THRESHOLD)
        is_violation = cos_alpha < thresh

        # 7. Time-based debounce (>= 800ms)
        confirmed = track.update_debounce(is_violation, now, self.cfg.get("debounce_ms", DEBOUNCE_MS))
        if not confirmed:
            return None

        # 8. 30-second cooldown check
        if not track.can_raise_violation(now):
            return None

        # ── Violation confirmed ────────────────────────────────────────
        track.record_violation_raised(now)
        angle_deg = float(np.degrees(np.arccos(np.clip(cos_alpha, -1.0, 1.0))))

        # Evidence lock. The bbox is passed so the crop is the OFFENDING
        # VEHICLE, not the whole frame - ANPR below reads this crop, and
        # plate OCR over a full 1080p scene is far less reliable.
        clip_path, clip_hash, crop_path = lock_evidence(
            frame, track.track_id, self.camera_id, now,
            bbox=det.get("bbox"),
        )

        # ANPR recognition on vehicle crop
        plate = read_plate(crop_path, self.cfg.get("anpr_min_confidence", ANPR_MIN_CONFIDENCE))

        logger.warning(
            "TRAFFIC WRONG_WAY CONFIRMED: camera=%s track=%d speed=%.1f km/h angle=%.1f deg plate=%s",
            self.camera_id, track.track_id, speed, angle_deg, plate.text or "UNREAD"
        )

        return ViolationEvent(
            track_id=track.track_id,
            zone_id=zone.lane_id,
            speed_kmh=round(speed, 1),
            discrepancy_angle_deg=round(angle_deg, 1),
            plate=plate,
            evidence_clip_hash=clip_hash,
            evidence_clip_path=clip_path,
            crop_path=crop_path,
            vehicle_class=det.get("vehicle_class", "car"),
            camera_id=self.camera_id,
            timestamp=now,
        )
