"""
tests/test_kalman_track.py — Dynamic dt & Anisotropic Q Kalman Filter Tests.

Covers:
  - Test 4: Dynamic dt Kalman handling irregular frame intervals (50ms, 200ms, 33ms).
  - Test 5: Anisotropic Q convergence within 10-20 frames with Gaussian measurement noise.
"""
import numpy as np
import pytest

from backend.services.kalman_track import ConstantVelocityKalman


def test_dynamic_dt_kalman():
    # True vehicle moving at vx = 10 m/s (36 km/h), vy = 0
    kalman = ConstantVelocityKalman(x0=0.0, y0=0.0)

    t = 0.0
    x_true = 0.0
    intervals = [0.033, 0.050, 0.200, 0.033, 0.100, 0.033, 0.050, 0.200, 0.033, 0.050, 0.100, 0.033]

    for dt in intervals:
        t += dt
        x_true += 10.0 * dt
        kalman.update((x_true, 0.0), now=t)

    # Velocity estimate should be within 5% of 10.0 m/s (36 km/h)
    vx, vy = kalman.velocity_mps
    assert np.isclose(vx, 10.0, atol=0.5)
    assert np.isclose(vy, 0.0, atol=0.5)
    assert np.isclose(kalman.speed_kmh, 36.0, atol=2.0)


def test_anisotropic_q_convergence():
    kalman = ConstantVelocityKalman(x0=0.0, y0=0.0)
    vx_true, vy_true = 15.0, 5.0  # ~56.9 km/h

    np.random.seed(42)
    t = 0.0
    for i in range(25):
        dt = 0.05
        t += dt
        x = vx_true * t + np.random.normal(0, 0.1)
        y = vy_true * t + np.random.normal(0, 0.1)
        kalman.update((x, y), now=t)

    speed_true = np.hypot(vx_true, vy_true) * 3.6
    assert abs(kalman.speed_kmh - speed_true) < (0.05 * speed_true)
    heading = kalman.heading_unit_vector
    assert heading is not None
    assert np.isclose(heading[0], vx_true / np.hypot(vx_true, vy_true), atol=0.05)
