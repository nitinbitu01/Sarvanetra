"""
tests/test_triple_riding_detector.py — Triple Riding Main Integration Tests (Tests 32-35).
"""
import numpy as np
import pytest

from backend.services.triple_riding_detector import (
    Detection,
    PersonDetection,
    TripleRidingDetector,
)


@pytest.mark.asyncio
async def test_true_positive_3_riders():
    H = np.eye(3)
    detector = TripleRidingDetector(
        camera_id="CAM_TEST",
        H=H,
        camera_angle_deg=0.0,
        cfg={"speed_gate_low_kmh": 15.0, "speed_gate_high_kmh": 150.0},
        shadow_mode=False,
    )

    frame = np.zeros((480, 640, 3), dtype=np.uint8)
    events = []

    # 35 frames: 3 riders on motorcycle at 35 km/h
    for t in range(35):
        now = t * 0.05
        bike = Detection(
            track_id=1,
            bbox=(100.0, 100.0, 300.0, 300.0),
            vehicle_class="motorcycle",
            confidence=0.95,
            speed_kmh=35.0,
        )
        p1 = PersonDetection(bbox=(110.0, 80.0, 160.0, 250.0), confidence=0.9)
        p2 = PersonDetection(bbox=(160.0, 80.0, 210.0, 250.0), confidence=0.9)
        p3 = PersonDetection(bbox=(210.0, 80.0, 260.0, 250.0), confidence=0.9)

        # Place 3 heads in frame
        frame[90:120, 120:150] = 200
        frame[90:120, 170:200] = 200
        frame[90:120, 220:250] = 200

        res = await detector.process_frame(frame, [bike], [p1, p2, p3], now=now)
        events.extend(res)

    assert len(events) >= 1
    ev = events[0]
    assert ev.track_id == 1
    assert ev.rider_count == 3
    assert ev.speed_kmh == 35.0
    assert ev.majority_ratio >= 0.65
    assert ev.collage_hash is not None


@pytest.mark.asyncio
async def test_true_negative_legal_2_riders():
    H = np.eye(3)
    detector = TripleRidingDetector(
        camera_id="CAM_TEST",
        H=H,
        camera_angle_deg=0.0,
        cfg={"speed_gate_low_kmh": 15.0, "speed_gate_high_kmh": 150.0},
        shadow_mode=False,
    )

    frame = np.zeros((480, 640, 3), dtype=np.uint8)
    events = []

    # 40 frames: 2 riders on motorcycle (legal)
    for t in range(40):
        now = t * 0.05
        bike = Detection(
            track_id=2,
            bbox=(100.0, 100.0, 300.0, 300.0),
            vehicle_class="motorcycle",
            confidence=0.95,
            speed_kmh=35.0,
        )
        p1 = PersonDetection(bbox=(120.0, 80.0, 180.0, 250.0), confidence=0.9)
        p2 = PersonDetection(bbox=(190.0, 80.0, 250.0, 250.0), confidence=0.9)

        res = await detector.process_frame(frame, [bike], [p1, p2], now=now)
        events.extend(res)

    # 0 alerts for legal 2 riders
    assert len(events) == 0


@pytest.mark.asyncio
async def test_bicycle_speed_gate_rejection():
    H = np.eye(3)
    detector = TripleRidingDetector(
        camera_id="CAM_TEST",
        H=H,
        camera_angle_deg=0.0,
        cfg={"speed_gate_low_kmh": 15.0, "speed_gate_high_kmh": 150.0},
        shadow_mode=False,
    )

    frame = np.zeros((480, 640, 3), dtype=np.uint8)
    events = []

    # 3 riders moving slowly at 8 km/h (< 15 km/h) -> suppressed
    for t in range(35):
        now = t * 0.05
        bike = Detection(
            track_id=3,
            bbox=(100.0, 100.0, 300.0, 300.0),
            vehicle_class="motorcycle",
            confidence=0.95,
            speed_kmh=8.0,  # 8 km/h < 15 km/h
        )
        p1 = PersonDetection(bbox=(110.0, 80.0, 160.0, 250.0), confidence=0.9)
        p2 = PersonDetection(bbox=(160.0, 80.0, 210.0, 250.0), confidence=0.9)
        p3 = PersonDetection(bbox=(210.0, 80.0, 260.0, 250.0), confidence=0.9)

        res = await detector.process_frame(frame, [bike], [p1, p2, p3], now=now)
        events.extend(res)

    assert len(events) == 0
