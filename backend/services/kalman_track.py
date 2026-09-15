"""
backend/services/kalman_track.py — Constant-Velocity Kalman Filter with Dynamic dt.

State vector: [x, y, vx, vy] in world-plane meters and m/s.
- Dynamic dt is computed directly from wall-clock timestamps (handles dropped/jittered RTSP frames).
- Anisotropic base process noise Q scales with elapsed time dt.
"""
from __future__ import annotations

import time
import numpy as np


class ConstantVelocityKalman:
    """Constant-velocity Kalman filter for a single vehicle track in world meters."""

    # Anisotropic base process noise
    # Units: m² for position, m²/s² for velocity
    _Q_BASE = np.diag([
        0.05,   # x position variance
        0.05,   # y position variance
        0.50,   # vx variance
        0.50,   # vy variance
    ])

    def __init__(self, x0: float, y0: float) -> None:
        self.x = np.array([x0, y0, 0.0, 0.0], dtype=np.float64)
        self.P = np.eye(4, dtype=np.float64) * 5.0

        # Observation matrix: measures [x, y]
        self._H_obs = np.array(
            [[1, 0, 0, 0],
             [0, 1, 0, 0]],
            dtype=np.float64,
        )

        # Measurement noise: ~0.3m std-dev in world coords (0.3² = 0.09)
        self._R = np.eye(2, dtype=np.float64) * 0.09
        self._last_time: float | None = None

    def _build_F(self, dt: float) -> np.ndarray:
        return np.array(
            [[1, 0, dt, 0],
             [0, 1, 0, dt],
             [0, 0,  1, 0],
             [0, 0,  0, 1]],
            dtype=np.float64,
        )

    def _predict_step(self, dt: float) -> None:
        F = self._build_F(dt)
        Q = self._Q_BASE * dt
        self.x = F @ self.x
        self.P = F @ self.P @ F.T + Q

    def update(self, z_xy: tuple[float, float], now: float) -> None:
        """Call predict internally with dynamic dt, then update with measurement."""
        if self._last_time is not None:
            dt = now - self._last_time
            if 0.0 < dt <= 2.0:
                self._predict_step(dt)
        self._last_time = now

        z = np.array(z_xy, dtype=np.float64)
        y = z - self._H_obs @ self.x
        S = self._H_obs @ self.P @ self._H_obs.T + self._R
        K = self.P @ self._H_obs.T @ np.linalg.inv(S)
        self.x = self.x + K @ y
        self.P = (np.eye(4) - K @ self._H_obs) @ self.P

    @property
    def velocity_mps(self) -> tuple[float, float]:
        """Filtered velocity estimate in m/s (world plane)."""
        return (float(self.x[2]), float(self.x[3]))

    @property
    def speed_kmh(self) -> float:
        """Filtered scalar speed in km/h."""
        vx, vy = self.velocity_mps
        return float(np.hypot(vx, vy) * 3.6)

    @property
    def heading_unit_vector(self) -> tuple[float, float] | None:
        """Unit vector in direction of filtered velocity, or None if stationary."""
        vx, vy = self.velocity_mps
        mag = float(np.hypot(vx, vy))
        if mag < 1e-6:
            return None
        return (vx / mag, vy / mag)
