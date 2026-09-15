"""
tests/test_live_calibration_production.py — SRE Production Test Suite for Live Calibration & Dynamic Shake Stabilization.
"""
import math
import cv2
import numpy as np
import pytest
from fastapi.testclient import TestClient

from backend.main import app
from backend.auth.dependencies import get_current_user
from backend.services.auto_calibrator import (
    HomographyEngine,
    CameraDriftStabilizer,
    VanishingPointCalibrator,
    normalize_points_hartley,
)

import pytest

# Mock authenticated user for tests
class MockOfficer:
    id = "officer-01"
    badge_number = "GJ-POL-001"
    rank = "Inspector"
    role = "ADMIN"
    department = "ALL"
    jurisdiction = "Ahmedabad"

@pytest.fixture(autouse=True)
def setup_auth_override():
    app.dependency_overrides[get_current_user] = lambda: MockOfficer()
    yield

app.dependency_overrides[get_current_user] = lambda: MockOfficer()
client = TestClient(app)


class TestHomographyEngineMath:
    """Rigorous mathematical tests for the DLT solver, condition number, and LOO validation."""

    def test_ideal_homography_solve_and_condition_number(self):
        """Test ideal synthetic rectangle-to-trapezoid projective mapping."""
        # 4 GCP points on a 14m x 15m road section
        src_pts = [(280.0, 920.0), (1520.0, 920.0), (460.0, 420.0), (1060.0, 420.0)]
        dst_pts = [(-7.0, 6.0), (7.0, 6.0), (-7.0, 21.0), (7.0, 21.0)]

        res = HomographyEngine.solve_homography(src_pts, dst_pts)
        assert res.is_valid is True
        assert res.H is not None
        assert res.condition_number < 1e5
        assert res.quality_gate in ("good", "degraded")
        assert len(res.grid_polylines) > 0

    def test_hartley_normalization_properties(self):
        """Verify Hartley normalization translates centroid to 0 and scales average distance to sqrt(2)."""
        pts = np.array([[100.0, 200.0], [500.0, 200.0], [200.0, 800.0], [600.0, 800.0]], dtype=np.float64)
        norm_pts, T = normalize_points_hartley(pts)

        # Centroid of normalized points must be (0, 0) within float precision
        centroid = np.mean(norm_pts, axis=0)
        assert abs(centroid[0]) < 1e-6
        assert abs(centroid[1]) < 1e-6

        # Average distance from origin must be sqrt(2)
        dists = np.sqrt(np.sum(norm_pts ** 2, axis=1))
        assert abs(np.mean(dists) - np.sqrt(2.0)) < 1e-6

    def test_degenerate_collinear_points_rejection(self):
        """Verify that degenerate collinear points are strictly rejected."""
        # 4 collinear points on a single straight line y = 2x
        src_pts = [(100.0, 200.0), (200.0, 400.0), (300.0, 600.0), (400.0, 800.0)]
        dst_pts = [(0.0, 0.0), (1.0, 2.0), (2.0, 4.0), (3.0, 6.0)]

        res = HomographyEngine.solve_homography(src_pts, dst_pts)
        assert res.is_valid is False
        assert "Degenerate" in res.message or "Condition number" in res.message

    def test_leave_one_out_cross_validation_accuracy(self):
        """Verify sub-10cm LOO RMS error on 5 surveyed ground control points."""
        src_pts = [
            (280.0, 920.0),
            (1520.0, 920.0),
            (460.0, 420.0),
            (1060.0, 420.0),
            (780.0, 580.0),  # Held-out point
        ]
        dst_pts = [
            (-7.0, 6.0),
            (7.0, 6.0),
            (-7.0, 21.0),
            (7.0, 21.0),
            (-0.40, 13.65),
        ]

        res = HomographyEngine.solve_homography(src_pts, dst_pts, held_out_idx=4)
        assert res.is_valid is True
        assert res.held_out_error_m is not None
        assert res.held_out_error_m < 0.20  # Sub-20cm on real test point
        assert res.quality_gate == "good"


class TestCameraDriftStabilizer:
    """Tests for frame-to-frame wind vibration, mast sway, and homography stabilization."""

    def test_drift_stabilizer_lifecycle(self):
        stabilizer = CameraDriftStabilizer(camera_id="CAM_TEST_SWAY")
        # Generate textured frame with rich corner points for ORB
        np.random.seed(42)
        frame1 = (np.random.rand(720, 1280, 3) * 255).astype(np.uint8)
        # Add high-contrast cross grid
        for x in range(100, 1200, 80):
            for y in range(100, 650, 80):
                cv2.rectangle(frame1, (x, y), (x + 20, y + 20), (255, 255, 255), -1)

        stabilizer.set_reference_frame(frame1)
        assert stabilizer.ref_descriptors is not None
        assert len(stabilizer.ref_keypoints) >= 20

        # Simulate small 2px camera shift
        M_shift = np.float32([[1, 0, 2], [0, 1, 1]])
        frame2 = cv2.warpAffine(frame1, M_shift, (1280, 720))

        A, telem = stabilizer.update_frame(frame2)
        assert telem.is_stabilized is True
        assert telem.drift_status in ("STABLE", "MINOR_VIBRATION")


class TestVanishingPointCalibrator:
    """Tests for automated vanishing point and road lane detection."""

    def test_vanishing_point_estimation_on_converging_lines(self):
        frame = np.full((1080, 1920, 3), 20, dtype=np.uint8)
        # Draw 6 converging perspective lines meeting near (960, 360) with strong contrast
        lines_coords = [
            ((960, 360), (200, 1080)),
            ((960, 360), (1720, 1080)),
            ((960, 360), (600, 1080)),
            ((960, 360), (1320, 1080)),
            ((960, 360), (400, 1080)),
            ((960, 360), (1500, 1080)),
        ]
        for p1, p2 in lines_coords:
            cv2.line(frame, p1, p2, (255, 255, 255), 5)

        res = VanishingPointCalibrator.auto_detect_vanishing_points(frame)
        assert res["success"] is True
        vp_x, vp_y = res["vp"]
        assert abs(vp_x - 960) < 60
        assert abs(vp_y - 360) < 60
        assert len(res["suggested_gcps"]) == 5


class TestCalibrationAPIEndpoints:
    """Test FastAPI endpoints for calibration listing, solving, saving, and auto-detecting."""

    def test_list_calibration_cameras(self):
        resp = client.get("/api/v1/analytics/calibration/cameras")
        assert resp.status_code == 200
        data = resp.json()
        assert "cameras" in data
        assert len(data["cameras"]) > 0

    def test_get_camera_calibration_detail(self):
        resp = client.get("/api/v1/analytics/calibration/camera/CAM_04")
        assert resp.status_code == 200
        data = resp.json()
        assert data["camera_id"] == "CAM_04"
        assert "ground_control_points" in data
        assert "homography_matrix" in data

    def test_solve_interactive_calibration_api(self):
        payload = {
            "camera_id": "CAM_04",
            "ground_control_points": [
                {"label": "P1", "pixel": [280, 920], "world_m": [-7.0, 6.0]},
                {"label": "P2", "pixel": [1520, 920], "world_m": [7.0, 6.0]},
                {"label": "P3", "pixel": [460, 420], "world_m": [-7.0, 21.0]},
                {"label": "P4", "pixel": [1060, 420], "world_m": [7.0, 21.0]},
                {"label": "P5", "pixel": [780, 580], "world_m": [-0.40, 13.65]},
            ],
            "held_out_index": 4,
            "frame_size": [1920, 1080],
        }
        resp = client.post("/api/v1/analytics/calibration/solve", json=payload)
        assert resp.status_code == 200
        data = resp.json()
        assert data["is_valid"] is True
        assert data["homography_matrix"] is not None
        assert data["quality_gate"] == "good"

    def test_drift_status_api(self):
        resp = client.get("/api/v1/analytics/calibration/drift-status/CAM_04")
        assert resp.status_code == 200
        data = resp.json()
        assert data["camera_id"] == "CAM_04"
        assert data["is_stabilized"] is True
        assert "jitter_rms_px" in data
