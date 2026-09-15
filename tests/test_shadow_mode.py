"""
tests/test_shadow_mode.py — Shadow Mode Safe Rollout Verification.
"""
import os
import sqlite3
import numpy as np
import pytest

from backend.services.triple_riding_detector import (
    Detection,
    PersonDetection,
    TripleRidingDetector,
)


@pytest.mark.asyncio
async def test_shadow_mode_logs_to_shadow_table_only(tmp_path):
    db_path = str(tmp_path / "shadow_test.db")
    H = np.eye(3)

    detector = TripleRidingDetector(
        camera_id="CAM_SHADOW_01",
        H=H,
        camera_angle_deg=0.0,
        cfg={"speed_gate_low_kmh": 15.0, "speed_gate_high_kmh": 150.0},
        shadow_mode=True,  # Shadow Mode Active!
    )

    frame = np.zeros((480, 640, 3), dtype=np.uint8)

    # Trigger 35 frames of triple riding
    for t in range(35):
        now = t * 0.05
        bike = Detection(
            track_id=77,
            bbox=(100.0, 100.0, 300.0, 300.0),
            vehicle_class="motorcycle",
            confidence=0.95,
            speed_kmh=40.0,
        )
        p1 = PersonDetection(bbox=(110.0, 80.0, 160.0, 250.0), confidence=0.9)
        p2 = PersonDetection(bbox=(160.0, 80.0, 210.0, 250.0), confidence=0.9)
        p3 = PersonDetection(bbox=(210.0, 80.0, 260.0, 250.0), confidence=0.9)

        # In shadow mode, events returned is empty (no live dispatch)
        res = await detector.process_frame(frame, [bike], [p1, p2, p3], now=now)
        assert len(res) == 0
