"""
backend/services/traffic_rollup.py — 1-Minute Pre-Aggregated Traffic Rollups & Health State (Phase 2).

Aggregates `vehicle_track` into `camera_metrics_1m` summary rows:
  - 15th percentile, Median, 85th percentile robust speeds.
  - Vehicle classification distributions.
  - Confidence weighting by calibration error and detector score.
  - Camera health status integration (OFFLINE / DEGRADED / ONLINE / COVERAGE_GAP).
"""
from __future__ import annotations

import json
import logging
from collections import Counter, defaultdict
from datetime import datetime, timedelta
from typing import Any, Optional, Sequence

import numpy as np
from sqlalchemy import func
from sqlalchemy.orm import Session

from backend.db.models import Camera, CameraHomographyCalibration, CameraMetrics1M, VehicleTrack
from backend.db.session import SessionLocal

logger = logging.getLogger("sentinel.traffic_rollup")


def compute_camera_rollup_1m(
    camera_id: str,
    bucket_start: datetime,
    tracks: Sequence[VehicleTrack],
    camera_record: Optional[Camera] = None,
    calib_record: Optional[CameraHomographyCalibration] = None,
) -> CameraMetrics1M:
    """Computes a 1-minute rollup row for a single camera from a sequence of tracks."""
    vehicle_count = len(tracks)
    class_counts = Counter(t.vehicle_class for t in tracks if t.vehicle_class)

    valid_speeds = [t.speed_kmh for t in tracks if t.speed_kmh is not None]
    speed_samples = len(valid_speeds)

    median_speed = None
    p15_speed = None
    p85_speed = None

    if speed_samples > 0:
        speeds_arr = np.array(valid_speeds, dtype=np.float64)
        median_speed = float(np.median(speeds_arr))
        p15_speed = float(np.percentile(speeds_arr, 15))
        p85_speed = float(np.percentile(speeds_arr, 85))

    # Compute confidence weighting
    conf_scores = []
    for t in tracks:
        det_conf = t.detector_conf if t.detector_conf is not None else 0.85
        calib_penalty = 0.0
        if t.calibration_err_m is not None:
            calib_penalty = min(1.0, max(0.0, t.calibration_err_m / 1.0))
        weight = det_conf * (1.0 - 0.5 * calib_penalty)
        conf_scores.append(weight)

    avg_confidence = float(np.mean(conf_scores)) if conf_scores else 0.0

    # Determine health status
    health_status = "ONLINE"
    if camera_record is not None:
        status_str = str(camera_record.status or "ONLINE").upper()
        if status_str in ("OFFLINE", "DISCONNECTED"):
            health_status = "OFFLINE"
        elif camera_record.consecutive_failures and camera_record.consecutive_failures >= 2:
            health_status = "DEGRADED"
        elif camera_record.last_heartbeat_at:
            age_sec = (datetime.utcnow() - camera_record.last_heartbeat_at).total_seconds()
            if age_sec > 120.0:
                health_status = "DEGRADED"

    if health_status == "ONLINE" and vehicle_count == 0:
        health_status = "COVERAGE_GAP"

    rollup = CameraMetrics1M(
        bucket_start=bucket_start,
        camera_id=camera_id,
        vehicle_count=vehicle_count,
        count_by_class=dict(class_counts),
        median_speed_kmh=round(median_speed, 1) if median_speed is not None else None,
        p15_speed_kmh=round(p15_speed, 1) if p15_speed is not None else None,
        p85_speed_kmh=round(p85_speed, 1) if p85_speed is not None else None,
        speed_samples=speed_samples,
        confidence=round(avg_confidence, 3),
        congestion_index=None, # populated by congestion engine
        health_status=health_status,
    )

    return rollup


def run_rollups_for_window(
    start_time: datetime,
    end_time: datetime,
    db: Optional[Session] = None,
) -> int:
    """Aggregates all tracks between start_time and end_time into 1-minute buckets."""
    own_session = False
    if db is None:
        db = SessionLocal()
        own_session = True
    try:
        # Load cameras and calibrations
        cameras = {c.camera_id: c for c in db.query(Camera).filter(Camera.is_deleted == False).all()}
        calibs = {c.camera_id: c for c in db.query(CameraHomographyCalibration).filter(CameraHomographyCalibration.is_active == True).all()}

        # Query all tracks in time window
        tracks = db.query(VehicleTrack).filter(
            VehicleTrack.first_seen >= start_time,
            VehicleTrack.first_seen < end_time,
        ).order_by(VehicleTrack.first_seen).all()

        # Group by (bucket_start, camera_id)
        bucketed: dict[tuple[datetime, str], list[VehicleTrack]] = defaultdict(list)
        for t in tracks:
            dt = t.first_seen
            b_start = dt.replace(second=0, microsecond=0)
            bucketed[(b_start, t.camera_id)].append(t)

        # Also account for all cameras with 0 sightings in each minute bucket
        current_min = start_time.replace(second=0, microsecond=0)
        end_min = end_time.replace(second=0, microsecond=0)
        step = timedelta(minutes=1)

        rollups_to_save: list[CameraMetrics1M] = []
        while current_min < end_min:
            for cam_id, cam in cameras.items():
                cam_tracks = bucketed.get((current_min, cam_id), [])
                rollup = compute_camera_rollup_1m(
                    camera_id=cam_id,
                    bucket_start=current_min,
                    tracks=cam_tracks,
                    camera_record=cam,
                    calib_record=calibs.get(cam_id),
                )
                rollups_to_save.append(rollup)
            current_min += step

        # Upsert or merge into camera_metrics_1m
        for r in rollups_to_save:
            db.merge(r)
        db.commit()

        logger.info("Generated %d camera metric rollups between %s and %s.", len(rollups_to_save), start_time, end_time)
        return len(rollups_to_save)
    finally:
        if own_session:
            db.close()
