"""
tests/test_calibration.py — Camera Homography & World Calibration Tests.

Covers:
  - Test 1: Calibration round-trip (held-out point within 0.3m).
  - Test 2: Calibration rejection (validate_homography raises ValueError if error > 0.3m).
  - Test 3: Polyline coordinate conversion (convert_polyline_to_world).
"""
import numpy as np
import pytest

from backend.services.calibration import (
    calibrate_camera,
    compute_homography,
    convert_polyline_to_world,
    nearest_tangent,
    pixel_to_world,
    validate_homography,
)


def test_calibration_round_trip():
    # Ground truth mapping: 100px = 10m (1px = 0.1m)
    img_pts = [(0.0, 0.0), (100.0, 0.0), (100.0, 100.0), (0.0, 100.0)]
    world_pts = [(0.0, 0.0), (10.0, 0.0), (10.0, 10.0), (0.0, 10.0)]

    held_out_img = (50.0, 50.0)
    held_out_world = (5.0, 5.0)

    res = calibrate_camera(
        img_pts,
        world_pts,
        held_out_img,
        held_out_world,
        calibrated_by="field-ops",
        calibrated_on="2026-08-24",
        max_error_m=0.30,
    )

    assert res.reprojection_error_m < 0.05
    assert res.H.shape == (3, 3)


def test_calibration_rejection():
    img_pts = [(0.0, 0.0), (100.0, 0.0), (100.0, 100.0), (0.0, 100.0)]
    world_pts = [(0.0, 0.0), (10.0, 0.0), (10.0, 10.0), (0.0, 10.0)]

    # Intentionally corrupt held-out world point
    bad_held_out_world = (50.0, 50.0)  # Off by 45 meters

    with pytest.raises(ValueError, match="exceeds maximum allowed"):
        calibrate_camera(
            img_pts,
            world_pts,
            (50.0, 50.0),
            bad_held_out_world,
            calibrated_by="field-ops",
            calibrated_on="2026-08-24",
            max_error_m=0.30,
        )


def test_polyline_coord_conversion_and_tangent():
    H = np.eye(3, dtype=np.float64)  # 1px = 1m
    norm_polyline = [(0.1, 0.1), (0.2, 0.3), (0.4, 0.7)]
    world_pts = convert_polyline_to_world(H, norm_polyline, frame_w=1000, frame_h=1000)

    assert len(world_pts) == 3
    assert world_pts[0] == (100.0, 100.0)
    assert world_pts[1] == (200.0, 300.0)

    # Test nearest tangent
    tangent = nearest_tangent(world_pts, (105.0, 105.0))
    dx, dy = 100.0, 200.0
    expected_norm = np.hypot(dx, dy)
    assert np.isclose(tangent[0], dx / expected_norm, atol=1e-3)
    assert np.isclose(tangent[1], dy / expected_norm, atol=1e-3)
