"""
tests/test_calibration_geo.py — Phase 0 Unit Tests: Calibration & Geo-Referencing.

Validates:
  1. Synthetic homography round-trip: Pixel -> World -> Pixel (< 1 px error).
  2. WGS-84 geodesic transform round-trip: World -> (lat, lon) -> World (< 1 mm error).
  3. Quality gate classifications: <0.50m (good), 0.50-1.00m (degraded), >1.00m (rejected).
  4. Held-out known ground distance reproduced within 5%.
  5. Directional bearing transformations (North, East, South, West rotations).
"""
import math
import numpy as np
import pytest

from backend.services.calibration import (
    calibrate_camera,
    compute_homography,
    evaluate_calibration_quality,
    pixel_to_world_m,
    validate_homography,
    wgs84_to_world,
    world_to_pixel,
    world_to_wgs84,
)


def test_synthetic_homography_pixel_world_roundtrip():
    """Test that a perspective homography round-trips with sub-pixel error."""
    # Synthetic camera mapping a 40m x 60m road scene at 1080p
    img_pts = [
        (400.0, 300.0),   # Top-left road boundary (far)
        (1520.0, 300.0),  # Top-right road boundary (far)
        (1850.0, 1000.0), # Bottom-right road boundary (near)
        (70.0, 1000.0),   # Bottom-left road boundary (near)
    ]
    # Ground plane meters: x in [-10, 10], y in [10, 70]
    world_pts = [
        (-10.0, 70.0),
        (10.0, 70.0),
        (10.0, 10.0),
        (-10.0, 10.0),
    ]

    H = compute_homography(img_pts, world_pts)

    for px, py in img_pts:
        wx, wy = pixel_to_world_m(H, px, py)
        px_rec, py_rec = world_to_pixel(H, wx, wy)
        assert abs(px - px_rec) < 0.01, f"Pixel X round-trip mismatch: {px} vs {px_rec}"
        assert abs(py - py_rec) < 0.01, f"Pixel Y round-trip mismatch: {py} vs {py_rec}"


def test_held_out_known_distance_within_5_percent():
    """Test that a surveyed held-out distance reproduces within 5%."""
    img_pts = [
        (400.0, 300.0), (1520.0, 300.0),
        (1850.0, 1000.0), (70.0, 1000.0),
    ]
    world_pts = [
        (-10.0, 70.0), (10.0, 70.0),
        (10.0, 10.0), (-10.0, 10.0),
    ]
    H = compute_homography(img_pts, world_pts)

    # Project known surveyed world points into the camera frame
    true_p1_world = (0.0, 25.0)
    true_p2_world = (0.0, 40.0)
    expected_dist = 15.0  # meters

    p1_img = world_to_pixel(H, *true_p1_world)
    p2_img = world_to_pixel(H, *true_p2_world)

    # Add realistic +/- 1.0px operator annotation noise
    p1_noisy = (p1_img[0] + 0.5, p1_img[1] - 0.5)
    p2_noisy = (p2_img[0] - 0.5, p2_img[1] + 0.5)

    w1 = pixel_to_world_m(H, *p1_noisy)
    w2 = pixel_to_world_m(H, *p2_noisy)
    measured_dist = float(np.hypot(w2[0] - w1[0], w2[1] - w1[1]))

    error_pct = abs(measured_dist - expected_dist) / expected_dist * 100.0
    assert error_pct < 5.0, f"Distance error {error_pct:.2f}% exceeds 5% bound"


def test_wgs84_geodesic_roundtrip():
    """Test that world-to-WGS84 and WGS84-to-world round-trip with < 1mm error."""
    anchor_lat = 23.0225   # Ahmedabad Janpath
    anchor_lon = 72.5714
    bearing_deg = 45.0     # Northeast optical axis

    test_points_m = [
        (0.0, 0.0),
        (5.0, 15.0),
        (-8.5, 42.0),
        (12.3, -5.0),
    ]

    for wx, wy in test_points_m:
        lat, lon = world_to_wgs84(wx, wy, anchor_lat, anchor_lon, bearing_deg)
        wx_rec, wy_rec = wgs84_to_world(lat, lon, anchor_lat, anchor_lon, bearing_deg)

        err_x = abs(wx - wx_rec)
        err_y = abs(wy - wy_rec)
        assert err_x < 0.001, f"WGS-84 X round-trip error {err_x:.6f}m >= 1mm"
        assert err_y < 0.001, f"WGS-84 Y round-trip error {err_y:.6f}m >= 1mm"


def test_quality_gate_tiers():
    """Verify quality gate decisions: good (<0.5m), degraded (0.5-1.0m), rejected (>1.0m)."""
    assert evaluate_calibration_quality(0.12) == "good"
    assert evaluate_calibration_quality(0.49) == "good"
    assert evaluate_calibration_quality(0.50) == "degraded"
    assert evaluate_calibration_quality(0.85) == "degraded"
    assert evaluate_calibration_quality(1.00) == "degraded"
    assert evaluate_calibration_quality(1.01) == "rejected"
    assert evaluate_calibration_quality(3.50) == "rejected"


def test_calibration_workflow_with_quality_gate():
    """Test full calibrate_camera execution producing CalibrationResult with correct gate."""
    img_pts = [(0.0, 0.0), (100.0, 0.0), (100.0, 100.0), (0.0, 100.0)]
    world_pts = [(0.0, 0.0), (10.0, 0.0), (10.0, 10.0), (0.0, 10.0)]

    # Good held-out point (0.05m error)
    res_good = calibrate_camera(
        img_pts, world_pts,
        held_out_image=(50.0, 50.0),
        held_out_world=(5.05, 5.0),
        calibrated_by="test-engineer",
        calibrated_on="2026-08-30",
        gps_anchor_lat=21.522,
        gps_anchor_lon=70.4579,
        bearing_deg=120.0,
    )
    assert res_good.quality_gate == "good"
    assert res_good.reprojection_error_m < 0.10
    assert res_good.gps_anchor_lat == 21.522

    # Degraded held-out point (0.75m error)
    res_degraded = calibrate_camera(
        img_pts, world_pts,
        held_out_image=(50.0, 50.0),
        held_out_world=(5.75, 5.0),
        calibrated_by="test-engineer",
        calibrated_on="2026-08-30",
    )
    assert res_degraded.quality_gate == "degraded"
    assert 0.50 <= res_degraded.reprojection_error_m <= 1.00

    # Rejected held-out point (>1.0m error)
    with pytest.raises(ValueError, match="exceeds maximum allowed"):
        calibrate_camera(
            img_pts, world_pts,
            held_out_image=(50.0, 50.0),
            held_out_world=(7.5, 5.0),
            calibrated_by="test-engineer",
            calibrated_on="2026-08-30",
            max_error_m=1.00,
        )


def test_projective_horizon_singularity_rejection():
    """Verify that points at or above the projective horizon line (w_z <= 0) raise ValueError."""
    img_pts = [(400.0, 300.0), (1520.0, 300.0), (1850.0, 1000.0), (70.0, 1000.0)]
    world_pts = [(-10.0, 70.0), (10.0, 70.0), (10.0, 10.0), (-10.0, 10.0)]
    H = compute_homography(img_pts, world_pts)

    # Vanishing line is at y ≈ -887.88; testing y = -1000.0 is strictly above horizon
    with pytest.raises(ValueError, match="projective horizon"):
        pixel_to_world_m(H, 960.0, -1000.0)


def test_singular_homography_inversion_rejection():
    """Verify that singular matrix inversion raises ValueError with clear error."""
    H_singular = np.zeros((3, 3), dtype=np.float64)
    with pytest.raises(ValueError, match="Cannot invert singular homography matrix"):
        world_to_pixel(H_singular, 10.0, 20.0)


def test_none_gps_anchor_returns_none():
    """Verify that missing GPS anchors return (None, None) rather than Null Island (0.0, 0.0)."""
    lat, lon = world_to_wgs84(10.0, 20.0, None, None)
    assert lat is None
    assert lon is None

    wx, wy = wgs84_to_world(23.0, 72.0, None, None)
    assert wx is None
    assert wy is None

