"""
backend/services/track_state.py — Track State Lifecycle & Sticky Zone Assignment Manager.

Key Capabilities:
  - WARMUP_FRAMES (10 frames) guard: Suppresses heading calculation during initial Kalman convergence.
  - Sticky Zone Assignment: Locks onto entering lane zone, preventing boundary flicker.
  - Wall-Clock Time Debounce (>= 800ms sustained) + 30s violation cooldown per track.
"""
from __future__ import annotations

import cv2
import numpy as np
from dataclasses import dataclass
from typing import Optional

from .calibration import nearest_tangent
from .kalman_track import ConstantVelocityKalman


@dataclass
class Zone:
    lane_id: str
    polygon_world_m: list[tuple[float, float]]   # world-meter coordinates
    geometry_type: str                            # "straight" | "polyline"
    flow_vector: Optional[tuple[float, float]] = None   # straight zones (unit vector)
    flow_polyline_world_m: Optional[list[tuple[float, float]]] = None  # curved zones
    speed_limit_kmh: float = 60.0

    def contains(self, wx: float, wy: float) -> bool:
        """Point-in-polygon test using OpenCV in world-plane coordinates."""
        if not self.polygon_world_m or len(self.polygon_world_m) < 3:
            return False
        pts = np.array(self.polygon_world_m, dtype=np.float32)
        result = cv2.pointPolygonTest(pts, (float(wx), float(wy)), measureDist=False)
        return result >= 0

    def local_flow_vector(self, wx: float, wy: float) -> tuple[float, float]:
        """Returns the local flow unit vector at world position (wx, wy)."""
        if self.geometry_type == "straight" and self.flow_vector:
            return self.flow_vector
        if self.flow_polyline_world_m and len(self.flow_polyline_world_m) >= 2:
            return nearest_tangent(self.flow_polyline_world_m, (wx, wy))
        return (0.0, 1.0)


class TrackState:
    """Owns all per-track mutable state and filters for one vehicle track lifetime."""

    # Defaults only. Both are overridable per instance so config.yaml's
    # traffic_intelligence.track_warmup_frames / .violation_cooldown_sec
    # actually take effect - previously these were class constants that no
    # caller could reach, so editing those config keys silently did nothing.
    WARMUP_FRAMES: int = 10              # frames before direction checks
    VIOLATION_COOLDOWN_SEC: float = 30.0 # minimum gap between alerts on same track

    def __init__(
        self,
        track_id: int,
        x0: float,
        y0: float,
        now: float,
        warmup_frames: int | None = None,
        violation_cooldown_sec: float | None = None,
    ) -> None:
        self.track_id = track_id
        self.kalman = ConstantVelocityKalman(x0, y0)
        self.frame_count: int = 0
        self.assigned_zone: Optional[Zone] = None
        self._violation_streak_start: Optional[float] = None
        self._last_violation_time: Optional[float] = None
        self.world_pos: tuple[float, float] = (x0, y0)
        if warmup_frames is not None:
            self.WARMUP_FRAMES = int(warmup_frames)
        if violation_cooldown_sec is not None:
            self.VIOLATION_COOLDOWN_SEC = float(violation_cooldown_sec)

    def update(
        self,
        wx: float,
        wy: float,
        now: float,
        all_zones: list[Zone],
    ) -> None:
        """Call once per detected frame for this track."""
        self.world_pos = (wx, wy)
        self.kalman.update((wx, wy), now)
        self.frame_count += 1
        self._update_zone_assignment(all_zones, wx, wy)

    def _update_zone_assignment(
        self, all_zones: list[Zone], wx: float, wy: float
    ) -> None:
        """Sticky zone assignment: locks onto zone until vehicle exits ALL zones."""
        if self.assigned_zone is None:
            for zone in all_zones:
                if zone.contains(wx, wy):
                    self.assigned_zone = zone
                    return
        else:
            in_any_zone = any(z.contains(wx, wy) for z in all_zones)
            if not in_any_zone:
                self.assigned_zone = None

    def is_ready_for_detection(self) -> bool:
        """Suppress direction checks during Kalman warm-up."""
        return self.frame_count >= self.WARMUP_FRAMES

    def in_zone(self) -> bool:
        return self.assigned_zone is not None

    def update_debounce(
        self, is_violation: bool, now: float, debounce_ms: float
    ) -> bool:
        """Updates violation streak timer. Returns True when streak first reaches debounce_ms."""
        if not is_violation:
            self._violation_streak_start = None
            return False

        if self._violation_streak_start is None:
            self._violation_streak_start = now
            return False

        elapsed_ms = (now - self._violation_streak_start) * 1000.0
        if elapsed_ms >= debounce_ms:
            self._violation_streak_start = None
            return True

        return False

    def can_raise_violation(self, now: float) -> bool:
        """Enforces 30-second cooldown between violations on same track."""
        if self._last_violation_time is None:
            return True
        return (now - self._last_violation_time) >= self.VIOLATION_COOLDOWN_SEC

    def record_violation_raised(self, now: float) -> None:
        """Records timestamp of confirmed violation."""
        self._last_violation_time = now
