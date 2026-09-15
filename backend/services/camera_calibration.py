"""
backend/services/camera_calibration.py — Per-camera px-per-meter lookup (Day 8, §2).

Converts LOITER_RADIUS_METERS into pixels for a specific camera, using that
camera's calibration_camera row if one exists. Cached in memory (same
reload-on-demand pattern as Day 7's FaceWatchlistMatcher) since this is read
on every single loitering tick and calibration rows change rarely.

Flat px-per-meter is an acceptable approximation for a roughly top-down or
moderate-angle camera view for THIS pass — see the Day 8 prompt §2 and §7.
Wide-angle/oblique cameras will need proper homography (a point-to-point
perspective transform) later; that is a bigger change (radius becomes
position-dependent, not a single scalar) and is explicitly deferred, not
built here.
"""
from __future__ import annotations

import logging

logger = logging.getLogger(__name__)


class CalibrationCache:
    def __init__(self) -> None:
        # camera_id (string) -> (px_per_meter, calibration_method)
        self._cache: dict[str, tuple[float, str]] = {}

    def reload(self) -> int:
        """Re-read camera_calibration from the DB into memory.

        Call once at startup, and again any time a calibration row is
        written by scripts/calibrate_camera.py, so the running detector
        picks up a fresh calibration without a process restart.

        Returns:
            Number of calibration rows loaded.
        """
        from backend.db.models import CameraCalibration
        from backend.db.session import SessionLocal

        db = SessionLocal()
        try:
            rows = db.query(CameraCalibration).all()
            self._cache = {
                row.camera_id: (row.px_per_meter, row.calibration_method) for row in rows
            }
            logger.info("Camera calibration cache loaded: %d camera(s).", len(self._cache))
            return len(self._cache)
        finally:
            db.close()

    def get(self, camera_id: str) -> tuple[float, str]:
        """Return (px_per_meter, calibration_method) for a camera.

        Falls back to settings.DEFAULT_PX_PER_METER with
        calibration_method='uncalibrated' if the camera has no row — this is
        the ONLY place that fallback happens, so callers never need to
        special-case a missing calibration themselves.
        """
        if camera_id in self._cache:
            return self._cache[camera_id]

        from backend.core.config import settings
        return (settings.DEFAULT_PX_PER_METER, "uncalibrated")


_cache_instance: CalibrationCache | None = None


def get_calibration_cache() -> CalibrationCache:
    global _cache_instance
    if _cache_instance is None:
        _cache_instance = CalibrationCache()
    return _cache_instance


def get_px_per_meter(camera_id: str) -> tuple[float, str]:
    """Convenience wrapper — (px_per_meter, calibration_method) for a camera."""
    return get_calibration_cache().get(camera_id)
