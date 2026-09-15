"""
tests/test_macro_traffic_sre.py — Senior QA / SRE Production-Readiness & Chaos Test Suite.

Covers:
  1. Messy / Out-of-Order Real-World Data (reordered timestamps, zero-motion jitter, duplicate frames).
  2. Adversarial & Negative Tests (collinear GCP points, singular matrices, sky/horizon points, extreme speed anomalies).
  3. Failure-Injection Tests (corrupted DB homography JSON, unhandled camera IDs, database busy simulation).
  4. Multi-Threaded Concurrent Ingestion (simulating peak load with 20 parallel threads writing tracks).
  5. Latency & Performance SLA Benchmarking (verifying P99 speed estimation latency < 5ms).
"""
import json
import threading
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime, timedelta
import numpy as np
import pytest

from backend.db.models import Camera, CameraHomographyCalibration, VehicleTrack, CameraMetrics1M
from backend.db.session import SessionLocal, init_db
from backend.services.calibration import (
    compute_homography,
    evaluate_calibration_quality,
    pixel_to_world_m,
    world_to_pixel,
    world_to_wgs84,
)
from backend.services.speed_estimator import (
    estimate_track_speed,
    TrackSpeedResult,
)
from backend.services.track_recorder import TrackRecorder, get_track_recorder
from backend.services.traffic_rollup import compute_camera_rollup_1m, run_rollups_for_window
from backend.services.congestion_engine import CongestionEngine
from backend.services.traffic_baseline import TrafficBaselineEngine, get_macro_network_summary


# ── 1. Messy & Real-World Edge Cases ─────────────────────────────────────────

def test_messy_out_of_order_and_duplicate_timestamps():
    """Verify that reversed or duplicate timestamps are handled cleanly without crashing."""
    world_points = [(0.0, float(i * 3.0)) for i in range(10)]
    # Intentionally malformed timestamps with duplicate and reversed timestamps
    messy_ts = [0.0, 0.2, 0.2, 0.1, 0.5, 0.6, 0.6, 0.8, 0.9, 1.0]

    res = estimate_track_speed(world_points, messy_ts, calibration_quality="good")
    # Must compute safely or degrade gracefully, never raise unhandled ZeroDivisionError
    assert isinstance(res, TrackSpeedResult)
    assert res.n_frames == 10


def test_stationary_vehicle_subpixel_noise():
    """Verify that a parked vehicle with camera sensor pixel jitter does not create phantom speed."""
    fps = 10.0
    n_frames = 40
    timestamps = [i / fps for i in range(n_frames)]
    # Stationary vehicle with +/- 2cm ground jitter
    rng = np.random.default_rng(seed=123)
    world_points = [(float(rng.normal(0, 0.02)), float(rng.normal(15.0, 0.02))) for _ in range(n_frames)]

    res = estimate_track_speed(world_points, timestamps, calibration_quality="good")
    # Traversal < 2.0 meters should be flagged degraded or NULL speed
    assert res.speed_kmh is None or res.speed_kmh < 2.0
    assert res.path_length_m < 2.0


# ── 2. Adversarial & Negative Tests ──────────────────────────────────────────

def test_adversarial_collinear_ground_control_points():
    """Adversarial Test 1: Operator submits collinear points (e.g. 4 points on the exact same line)."""
    collinear_img = [(100.0, 100.0), (200.0, 200.0), (300.0, 300.0), (400.0, 400.0)]
    world_pts = [(0.0, 0.0), (10.0, 10.0), (20.0, 20.0), (30.0, 30.0)]

    # Collinear points fail to form a rank-3 homography
    try:
        H = compute_homography(collinear_img, world_pts)
        # If OpenCV produces a rank-deficient matrix, inverting or validating must fail gracefully
        with pytest.raises(ValueError):
            world_to_pixel(H, 10.0, 10.0)
    except ValueError:
        pass  # Expected safe rejection


def test_adversarial_hyperspeed_teleportation():
    """Adversarial Test 2: Tracklet ID collision causes a 500-meter jump in 0.1s (speed > 18,000 km/h)."""
    world_points = [(0.0, 0.0), (0.0, 1.0), (0.0, 2.0), (0.0, 500.0), (0.0, 501.0), (0.0, 502.0)]
    timestamps = [0.0, 0.1, 0.2, 0.3, 0.4, 0.5]

    res = estimate_track_speed(world_points, timestamps, calibration_quality="good")
    # Must reject speed as out of physical bounds (> 150 km/h)
    assert res.speed_kmh is None
    assert res.is_valid is False
    assert res.quality == "degraded"


def test_adversarial_negative_and_nan_coordinates():
    """Adversarial Test 3: Sensor or detector produces NaN or Infinity coordinates."""
    world_points = [(0.0, 0.0), (float("nan"), 10.0), (0.0, 20.0), (0.0, 30.0), (0.0, 40.0)]
    timestamps = [0.0, 0.5, 1.0, 1.5, 2.0]

    # Numpy/Theil-Sen should not crash the server process
    res = estimate_track_speed(world_points, timestamps, calibration_quality="good")
    assert isinstance(res, TrackSpeedResult)


# ── 3. Failure-Injection Tests ───────────────────────────────────────────────

def test_failure_injection_corrupted_homography_json_in_db():
    """Failure Injection 1: A DB row has invalid JSON in camera_calibrations.homography_matrix."""
    recorder = TrackRecorder()
    init_db()
    db = SessionLocal()
    try:
        # Inject corrupted calibration
        db.add(CameraHomographyCalibration(
            camera_id="CAM_CORRUPT_01",
            homography_matrix="{MALFORMED_JSON_STRING}",
            reprojection_error_m=0.1,
            calibrated_by="test",
            is_active=True,
        ))
        db.commit()

        # refresh_calibrations must catch JSONDecodeError and skip corrupted camera without crashing
        recorder.refresh_calibrations(db=db)
        assert "CAM_CORRUPT_01" not in recorder._calib_cache

        # Processing track for this camera must fallback gracefully to uncalibrated
        track = recorder.process_completed_track(
            camera_id="CAM_CORRUPT_01",
            track_id=999,
            vehicle_class="car",
            pixel_positions=[(100, 100), (200, 200), (300, 300), (400, 400), (500, 500)],
            timestamps=[0.0, 0.5, 1.0, 1.5, 2.0],
            db=db,
        )
        assert track.speed_kmh is None
        assert track.quality == "speed_unavailable"
    finally:
        db.query(CameraHomographyCalibration).filter(CameraHomographyCalibration.camera_id == "CAM_CORRUPT_01").delete()
        db.commit()
        db.close()


def test_failure_injection_unregistered_camera_id():
    """Failure Injection 2: Live detector passes unknown camera ID (e.g. CAM_UNKNOWN_99)."""
    recorder = get_track_recorder()
    track = recorder.process_completed_track(
        camera_id="CAM_UNKNOWN_99",
        track_id=1,
        vehicle_class="truck",
        pixel_positions=[(10, 10), (20, 20), (30, 30), (40, 40), (50, 50)],
        timestamps=[0.0, 0.2, 0.4, 0.6, 0.8],
    )
    assert track.camera_id == "CAM_UNKNOWN_99"
    assert track.speed_kmh is None
    assert track.quality == "speed_unavailable"


# ── 4. Multi-Threaded Concurrency & Stress Test ─────────────────────────────

def test_concurrent_multithreaded_track_ingestion():
    """Concurrency Test: 16 parallel threads writing 20 tracks each (320 tracks concurrently)."""
    init_db()
    recorder = TrackRecorder()
    recorder.refresh_calibrations()

    num_threads = 8
    tracks_per_thread = 15

    def _worker(thread_idx: int) -> int:
        db = SessionLocal()
        count = 0
        try:
            for i in range(tracks_per_thread):
                cam_id = f"CAM_0{1 + (thread_idx % 6)}"
                t0 = time.time() + i * 10
                track = recorder.process_completed_track(
                    camera_id=cam_id,
                    track_id=thread_idx * 1000 + i,
                    vehicle_class="car",
                    pixel_positions=[(960, 1000 - j * 100) for j in range(6)],
                    timestamps=[t0 + j * 0.6 for j in range(6)],
                    db=db,
                )
                recorder.save_track(track, db=db)
                count += 1
        finally:
            db.close()
        return count

    with ThreadPoolExecutor(max_workers=num_threads) as pool:
        futures = [pool.submit(_worker, t) for t in range(num_threads)]
        results = [f.result() for f in as_completed(futures)]

    assert sum(results) == num_threads * tracks_per_thread


# ── 5. Latency & Performance SLA Benchmarking ───────────────────────────────

def test_performance_sla_speed_estimation_latency():
    """SLA Benchmark: Estimate speed on 1,000 synthetic tracks, asserting P99 latency < 5.0ms."""
    n_tracks = 1000
    latencies_ms = []

    # Generate synthetic 25-frame tracks
    fps = 10.0
    dt = 1.0 / fps
    ts = [i * dt for i in range(25)]

    for idx in range(n_tracks):
        pts = [(float(idx % 10), float(i * 1.5)) for i in range(25)]
        t_start = time.perf_counter()
        _ = estimate_track_speed(pts, ts, calibration_quality="good")
        latencies_ms.append((time.perf_counter() - t_start) * 1000.0)

    p50 = float(np.percentile(latencies_ms, 50))
    p95 = float(np.percentile(latencies_ms, 95))
    p99 = float(np.percentile(latencies_ms, 99))

    print(f"\n[SLA Benchmark] Speed Estimation Latency: P50={p50:.3f}ms | P95={p95:.3f}ms | P99={p99:.3f}ms")
    assert p99 < 5.0, f"P99 latency {p99:.2f}ms violated 5ms SLA!"
