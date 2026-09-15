"""
tests/test_tracklet_memory.py — Tracklet Re-ID Memory & Buffer Preservation Tests (Tests 16-19).
"""
import numpy as np
import pytest

from backend.services.tracklet_memory import (
    REID_SIM_THRESHOLD,
    TrackletMemory,
    TrackState,
)


def test_reid_buffer_restored():
    mem = TrackletMemory()
    crop_red = np.zeros((100, 100, 3), dtype=np.uint8)
    crop_red[:, :, 2] = 255  # Solid red motorcycle

    # Track 1 appears and accumulates 15 frames of 3-rider violation
    s1 = mem.get_or_create(track_id=1, crop=crop_red, now=0.0)
    for _ in range(15):
        s1.push(3)
    assert s1.majority_ratio == 1.0

    # Track 1 occluded / evicted at t=1.0s
    mem.evict(track_id=1, crop=crop_red, now=1.0)
    assert 1 not in mem._live

    # Tracker creates new ID 42 for same motorcycle at t=2.0s (1.0s gap < 3.0s max age)
    s2 = mem.get_or_create(track_id=42, crop=crop_red, now=2.0)
    # State buffer should be restored
    assert s2.majority_ratio == 1.0
    assert len(s2.frame_buffer) == 15


def test_reid_rejected_different_vehicle():
    mem = TrackletMemory()
    crop_red = np.zeros((100, 100, 3), dtype=np.uint8)
    crop_red[:, :, 2] = 255

    crop_blue = np.zeros((100, 100, 3), dtype=np.uint8)
    crop_blue[:, :, 0] = 255  # Blue motorcycle

    s1 = mem.get_or_create(track_id=1, crop=crop_red, now=0.0)
    for _ in range(10):
        s1.push(3)
    mem.evict(track_id=1, crop=crop_red, now=1.0)

    # Blue motorcycle appears
    s2 = mem.get_or_create(track_id=2, crop=crop_blue, now=2.0)
    # Different vehicle -> fresh buffer
    assert len(s2.frame_buffer) == 0


def test_reid_pool_expiry():
    mem = TrackletMemory()
    crop = np.zeros((100, 100, 3), dtype=np.uint8)
    crop[:, :, 1] = 200

    s1 = mem.get_or_create(track_id=5, crop=crop, now=0.0)
    s1.push(3)
    mem.evict(track_id=5, crop=crop, now=1.0)

    # Re-appears at t=5.0s (gap of 4.0s > 3.0s TTL)
    s2 = mem.get_or_create(track_id=99, crop=crop, now=5.0)
    # Expired -> fresh buffer
    assert len(s2.frame_buffer) == 0


def test_cooldown_transferred_across_reid():
    mem = TrackletMemory()
    crop = np.zeros((100, 100, 3), dtype=np.uint8)

    s1 = mem.get_or_create(track_id=10, crop=crop, now=0.0)
    s1.record_alert(now=5.0)
    assert not s1.can_alert(now=10.0)  # in 30s cooldown until 35.0

    mem.evict(track_id=10, crop=crop, now=11.0)
    s2 = mem.get_or_create(track_id=20, crop=crop, now=12.0)
    assert not s2.can_alert(now=15.0)
    assert s2.can_alert(now=36.0)
