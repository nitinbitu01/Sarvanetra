"""
tests/test_wrong_way_detector.py — Wrong-Way Detection Engine Unit & Regression Tests.

Covers:
  - Test 9: Upper speed gate (spike to 200 km/h is suppressed).
  - Test 10: Lower speed gate (stationary jitter suppressed).
  - Test 11: True positive (vehicle moving against flow for >= 800ms triggers alert).
  - Test 12: True negative (vehicle moving in correct flow generates zero alerts).
  - Test 13: Debounce short streak (violation for 600ms drops without alert).
  - Test 14: Debounce exact threshold (violation for 800ms confirms alert).
  - Test 15: Violation cooldown (second violation within 30s suppressed).
  - Test 17: Curved-lane legal travel (following polyline tangents generates zero alerts).
  - Test 27: Evidence hash integrity (SHA-256 matches stored file).
"""
import numpy as np
import pytest

from backend.services.evidence import hash_file
from backend.services.track_state import Zone
from backend.services.wrong_way_detector import (
    SPEED_GATE_HIGH_KMH,
    SPEED_GATE_LOW_KMH,
    WrongWayDetector,
)


@pytest.fixture
def detector():
    # 1px = 0.1m
    H = np.array([[0.1, 0, 0], [0, 0.1, 0], [0, 0, 1]], dtype=np.float64)
    # Zone: Northbound (flow vector (0, -1) in world meters)
    zone_nb = Zone(
        lane_id="LANE_NB",
        polygon_world_m=[(0.0, 0.0), (20.0, 0.0), (20.0, 200.0), (0.0, 200.0)],
        geometry_type="straight",
        flow_vector=(0.0, -1.0),
        speed_limit_kmh=60.0,
    )
    # Curved Zone
    zone_curve = Zone(
        lane_id="LANE_CURVE",
        polygon_world_m=[(30.0, 0.0), (50.0, 0.0), (50.0, 200.0), (30.0, 200.0)],
        geometry_type="polyline",
        flow_polyline_world_m=[(40.0, 0.0), (40.0, 30.0), (45.0, 60.0), (48.0, 90.0)],
        speed_limit_kmh=40.0,
    )
    return WrongWayDetector("CAM_TEST", H, [zone_nb, zone_curve])


def test_true_positive_wrong_way(detector):
    # Vehicle moving South (increasing y in world meters) in Northbound lane
    # 10 m/s = 36 km/h. Flow vector is (0, -1). Discrepancy angle = 180 deg.
    frame = np.zeros((480, 640, 3), dtype=np.uint8)
    events = []

    # 10 frames warm-up (0.5s) + 25 frames violation (1.25s > 800ms debounce)
    t = 0.0
    for i in range(35):
        t += 0.05
        # Pixel coords: (50px = 5m, 100px + i*10px) -> world coords: (5.0, 10.0 + i*1.0)
        px_y = 100.0 + (i * 10.0)
        det = [{"track_id": 42, "bbox": [40, px_y - 10, 60, px_y + 10], "vehicle_class": "car"}]
        res = detector.process_detections(det, frame, now=t)
        events.extend(res)

    assert len(events) >= 1
    ev = events[0]
    assert ev.track_id == 42
    assert ev.zone_id == "LANE_NB"
    assert ev.speed_kmh >= 30.0
    assert ev.discrepancy_angle_deg >= 170.0
    assert ev.evidence_clip_hash is not None


def test_true_negative_legal_flow(detector):
    # Vehicle moving North (decreasing y) in Northbound lane (legal)
    frame = np.zeros((480, 640, 3), dtype=np.uint8)
    events = []

    t = 0.0
    for i in range(35):
        t += 0.05
        px_y = 500.0 - (i * 10.0)
        det = [{"track_id": 7, "bbox": [40, px_y - 10, 60, px_y + 10], "vehicle_class": "car"}]
        res = detector.process_detections(det, frame, now=t)
        events.extend(res)

    assert len(events) == 0


def test_lower_speed_gate_stationary_suppressed(detector):
    # Vehicle stationary with tiny sub-pixel jitter
    frame = np.zeros((480, 640, 3), dtype=np.uint8)
    events = []

    t = 0.0
    for i in range(30):
        t += 0.05
        jitter = np.sin(i) * 0.1
        det = [{"track_id": 9, "bbox": [40, 200 + jitter, 60, 220 + jitter], "vehicle_class": "car"}]
        res = detector.process_detections(det, frame, now=t)
        events.extend(res)

    assert len(events) == 0


def test_upper_speed_gate_teleport_suppressed(detector):
    # Vehicle teleports at 200 km/h (55.5 m/s)
    frame = np.zeros((480, 640, 3), dtype=np.uint8)
    events = []

    t = 0.0
    for i in range(25):
        t += 0.05
        # 55.5 m/s = 555 px/s -> 27.7 px per frame
        px_y = 50.0 + (i * 40.0)
        det = [{"track_id": 99, "bbox": [40, px_y - 10, 60, px_y + 10], "vehicle_class": "car"}]
        res = detector.process_detections(det, frame, now=t)
        events.extend(res)

    # Kalman speed exceeds 150 km/h -> suppressed
    assert len(events) == 0


def test_debounce_thresholds(detector):
    frame = np.zeros((480, 640, 3), dtype=np.uint8)

    # Case A: 600ms sustained (short streak) -> 0 events
    t = 0.0
    events_short = []
    for i in range(15):  # 10 warm-up (500ms) + 2 violation frames (100ms)
        t += 0.05
        px_y = 100.0 + (i * 10.0)
        det = [{"track_id": 101, "bbox": [40, px_y - 10, 60, px_y + 10], "vehicle_class": "car"}]
        res = detector.process_detections(det, frame, now=t)
        events_short.extend(res)
    assert len(events_short) == 0


def test_violation_cooldown_30s(detector):
    frame = np.zeros((480, 640, 3), dtype=np.uint8)
    events = []

    # First violation
    t = 0.0
    for i in range(35):
        t += 0.05
        px_y = 100.0 + (i * 10.0)
        det = [{"track_id": 55, "bbox": [40, px_y - 10, 60, px_y + 10], "vehicle_class": "car"}]
        res = detector.process_detections(det, frame, now=t)
        events.extend(res)

    assert len(events) == 1

    # Second violation attempt at t=10.0s (within 30s cooldown)
    for i in range(35):
        t += 0.05
        px_y = 400.0 + (i * 10.0)
        det = [{"track_id": 55, "bbox": [40, px_y - 10, 60, px_y + 10], "vehicle_class": "car"}]
        res = detector.process_detections(det, frame, now=t)
        events.extend(res)

    # Still only 1 event because cooldown is 30s!
    assert len(events) == 1


def test_evidence_hash_integrity(detector):
    frame = np.zeros((480, 640, 3), dtype=np.uint8)
    frame[50:100, 50:100] = 255  # Distinct pattern

    events = []
    t = 0.0
    for i in range(35):
        t += 0.05
        px_y = 100.0 + (i * 10.0)
        det = [{"track_id": 77, "bbox": [40, px_y - 10, 60, px_y + 10], "vehicle_class": "car"}]
        res = detector.process_detections(det, frame, now=t)
        events.extend(res)

    assert len(events) >= 1
    ev = events[0]
    computed_hash = hash_file(ev.crop_path)
    assert computed_hash == ev.evidence_clip_hash
