"""
backend/services/track_recorder.py — Track-Level Measurement Recorder & Batch Ingestion (Phase 1).

Persists one summary row per vehicle per camera in `vehicle_track` (~460K rows/day across 32 cameras),
avoiding per-frame row explosions (69M rows/day).
"""
from __future__ import annotations

import asyncio
import json
import logging
import time
from datetime import datetime
from typing import Any, Optional, Sequence

import numpy as np
from sqlalchemy.orm import Session

from backend.db.models import Camera, CameraHomographyCalibration, VehicleTrack
from backend.db.session import SessionLocal
from backend.services.calibration import (
    pixel_to_world_m,
    world_to_wgs84,
)
from backend.services.speed_estimator import (
    TrackSpeedResult,
    estimate_track_speed,
)

logger = logging.getLogger("sentinel.track_recorder")


class TrackRecorder:
    """Manages calibration caching, coordinate transforms, and batched persistence for vehicle tracks."""

    def __init__(self):
        self._calib_cache: dict[str, dict[str, Any]] = {}
        self._batch_queue: list[dict[str, Any]] = []
        self._lock = asyncio.Lock()
        self._last_flush = time.time()
        self._flush_interval_sec = 2.0
        self._max_batch_size = 50

    def refresh_calibrations(self, db: Optional[Session] = None) -> None:
        """Loads all active homography calibrations into memory."""
        own_session = False
        if db is None:
            db = SessionLocal()
            own_session = True
        try:
            calibs = db.query(CameraHomographyCalibration).filter(
                CameraHomographyCalibration.is_active == True
            ).all()

            cache = {}
            for c in calibs:
                try:
                    h_list = json.loads(c.homography_matrix)
                    H = np.array(h_list, dtype=np.float64).reshape(3, 3)
                    cache[c.camera_id] = {
                        "H": H,
                        "reprojection_error_m": c.reprojection_error_m,
                        "quality_gate": getattr(c, "quality_gate", "good"),
                        "gps_anchor_lat": c.gps_anchor_lat,
                        "gps_anchor_lon": c.gps_anchor_lon,
                        "bearing_deg": getattr(c, "bearing_deg", 0.0) or 0.0,
                    }
                except Exception as e:
                    logger.warning("Failed to parse homography for %s: %s", c.camera_id, e)
            self._calib_cache = cache
            logger.info("TrackRecorder loaded %d camera calibrations into memory.", len(cache))
        finally:
            if own_session:
                db.close()

    def process_completed_track(
        self,
        camera_id: str,
        track_id: int,
        vehicle_class: str,
        pixel_positions: Sequence[tuple[float, float]], # (px, py) centers
        timestamps: Sequence[float],                    # epoch timestamps
        detector_conf: float = 0.85,
        db: Optional[Session] = None,
    ) -> Optional[VehicleTrack]:
        """Calculates world-space trajectory, Theil-Sen speed, WGS-84 entry/exit and returns a VehicleTrack row."""
        if not self._calib_cache:
            self.refresh_calibrations(db)

        calib = self._calib_cache.get(camera_id)
        if calib is None:
            calib_quality = "uncalibrated"
            H = None
            calib_err = None
            anchor_lat = None
            anchor_lon = None
            bearing_deg = 0.0
        else:
            H = calib["H"]
            calib_err = calib["reprojection_error_m"]
            calib_quality = calib["quality_gate"]
            anchor_lat = calib["gps_anchor_lat"]
            anchor_lon = calib["gps_anchor_lon"]
            bearing_deg = calib["bearing_deg"]

        # Convert pixel trajectory to world coordinates
        world_pts: list[tuple[float, float]] = []
        if H is not None:
            for px, py in pixel_positions:
                wx, wy = pixel_to_world_m(H, px, py)
                world_pts.append((wx, wy))

        # Run robust speed fit
        speed_res: TrackSpeedResult = estimate_track_speed(
            world_points=world_pts,
            timestamps=timestamps,
            calibration_quality=calib_quality,
            camera_bearing_deg=bearing_deg,
        )

        first_seen = datetime.utcfromtimestamp(timestamps[0]) if timestamps else datetime.utcnow()
        last_seen = datetime.utcfromtimestamp(timestamps[-1]) if timestamps else datetime.utcnow()

        # Compute entry and exit WGS-84 coordinates
        entry_lat, entry_lon, exit_lat, exit_lon = None, None, None, None
        if world_pts and anchor_lat is not None and anchor_lon is not None:
            entry_lat, entry_lon = world_to_wgs84(world_pts[0][0], world_pts[0][1], anchor_lat, anchor_lon, bearing_deg)
            exit_lat, exit_lon = world_to_wgs84(world_pts[-1][0], world_pts[-1][1], anchor_lat, anchor_lon, bearing_deg)

        track_record = VehicleTrack(
            camera_id=camera_id,
            track_id=track_id,
            vehicle_class=vehicle_class,
            first_seen=first_seen,
            last_seen=last_seen,
            entry_lat=round(entry_lat, 6) if entry_lat is not None else None,
            entry_lon=round(entry_lon, 6) if entry_lon is not None else None,
            exit_lat=round(exit_lat, 6) if exit_lat is not None else None,
            exit_lon=round(exit_lon, 6) if exit_lon is not None else None,
            heading_deg=speed_res.heading_deg,
            speed_kmh=speed_res.speed_kmh,
            speed_ci_kmh=speed_res.speed_ci_kmh,
            path_length_m=speed_res.path_length_m,
            n_frames=len(pixel_positions),
            detector_conf=round(float(detector_conf), 3),
            calibration_err_m=round(float(calib_err), 3) if calib_err is not None else None,
            quality=speed_res.quality,
        )

        return track_record

    def save_track(self, track: VehicleTrack, db: Optional[Session] = None) -> VehicleTrack:
        """Saves a single vehicle track synchronously."""
        own_session = False
        if db is None:
            db = SessionLocal()
            own_session = True
        try:
            db.add(track)
            db.commit()
            db.refresh(track)
            return track
        finally:
            if own_session:
                db.close()

    def save_tracks_batch(self, tracks: Sequence[VehicleTrack], db: Optional[Session] = None) -> int:
        """Saves a list of vehicle tracks in a single atomic database transaction."""
        if not tracks:
            return 0
        own_session = False
        if db is None:
            db = SessionLocal()
            own_session = True
        try:
            db.bulk_save_objects(tracks)
            db.commit()
            return len(tracks)
        except Exception as e:
            db.rollback()
            logger.error("Failed to batch save %d tracks: %s", len(tracks), e)
            raise
        finally:
            if own_session:
                db.close()


_global_recorder: Optional[TrackRecorder] = None


def get_track_recorder() -> TrackRecorder:
    global _global_recorder
    if _global_recorder is None:
        _global_recorder = TrackRecorder()
    return _global_recorder
