"""
tests/test_traffic_rollup.py — Phase 2 Unit Tests: 1-Minute Traffic Rollups & Camera Health States.

Validates:
  1. 1-minute bucket aggregation with median, p15, and p85 robust speed percentiles.
  2. Vehicle classification breakdown (car, motorcycle, bus, truck).
  3. Camera health state mapping: OFFLINE, DEGRADED, COVERAGE_GAP, ONLINE.
  4. Null speed preservation when no valid calibrated tracks exist.
"""
from datetime import datetime, timedelta
import pytest

from backend.db.models import Camera, CameraMetrics1M, VehicleTrack
from backend.db.session import SessionLocal, init_db
from backend.services.traffic_rollup import compute_camera_rollup_1m, run_rollups_for_window


def test_1m_rollup_speed_percentiles_and_counts():
    """Verify median, p15, p85, and class breakdown calculations."""
    bucket_start = datetime(2026, 8, 30, 10, 0, 0)

    # 10 synthetic vehicle tracks with speeds [30, 35, 40, 45, 50, 55, 60, 65, 70, 75] km/h
    speeds = [30.0, 35.0, 40.0, 45.0, 50.0, 55.0, 60.0, 65.0, 70.0, 75.0]
    classes = ["car", "car", "motorcycle", "car", "bus", "car", "motorcycle", "truck", "car", "car"]

    tracks = [
        VehicleTrack(
            camera_id="CAM_01",
            track_id=i,
            vehicle_class=classes[i],
            first_seen=bucket_start + timedelta(seconds=i * 5),
            last_seen=bucket_start + timedelta(seconds=i * 5 + 3),
            speed_kmh=speeds[i],
            detector_conf=0.90,
            calibration_err_m=0.02,
            quality="good",
        )
        for i in range(10)
    ]

    cam_mock = Camera(camera_id="CAM_01", name="Ahmedabad Bridge", status="ONLINE")

    rollup = compute_camera_rollup_1m(
        camera_id="CAM_01",
        bucket_start=bucket_start,
        tracks=tracks,
        camera_record=cam_mock,
    )

    assert rollup.vehicle_count == 10
    assert rollup.speed_samples == 10
    assert rollup.median_speed_kmh == 52.5  # median of 50 and 55
    assert rollup.p15_speed_kmh is not None
    assert rollup.p85_speed_kmh is not None
    assert rollup.p15_speed_kmh < rollup.median_speed_kmh < rollup.p85_speed_kmh
    assert rollup.count_by_class == {"car": 6, "motorcycle": 2, "bus": 1, "truck": 1}
    assert rollup.health_status == "ONLINE"
    assert rollup.confidence > 0.85


def test_camera_offline_and_coverage_gap_health():
    """Verify that dead cameras or unobserved segments are flagged as OFFLINE / COVERAGE_GAP."""
    bucket_start = datetime(2026, 8, 30, 10, 0, 0)

    # 1. Offline camera
    cam_offline = Camera(camera_id="CAM_02", name="Janpath", status="OFFLINE")
    rollup_offline = compute_camera_rollup_1m(
        camera_id="CAM_02",
        bucket_start=bucket_start,
        tracks=[],
        camera_record=cam_offline,
    )
    assert rollup_offline.health_status == "OFFLINE"
    assert rollup_offline.median_speed_kmh is None
    assert rollup_offline.vehicle_count == 0

    # 2. Online camera with 0 tracks -> COVERAGE_GAP (not empty road)
    cam_online = Camera(camera_id="CAM_03", name="ONGC", status="ONLINE")
    rollup_gap = compute_camera_rollup_1m(
        camera_id="CAM_03",
        bucket_start=bucket_start,
        tracks=[],
        camera_record=cam_online,
    )
    assert rollup_gap.health_status == "COVERAGE_GAP"
    assert rollup_gap.median_speed_kmh is None
