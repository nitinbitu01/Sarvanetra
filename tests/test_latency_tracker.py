"""
tests/test_latency_tracker.py — Latency Budget Monitoring Tests.
"""
import time
import pytest

from backend.monitoring.latency_tracker import LatencyTracker


def test_latency_measurement_and_p95():
    tracker = LatencyTracker("CAM_01")
    with tracker.measure("frame_enhance"):
        time.sleep(0.005)  # 5ms

    p95 = tracker.get_stage_p95("frame_enhance")
    assert p95 >= 4.0  # ~5ms
