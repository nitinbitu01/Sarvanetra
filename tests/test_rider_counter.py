"""
tests/test_rider_counter.py — Rider Counter Tri-Modal Fusion Unit Tests (Tests 6-15).
"""
import numpy as np
import pytest

from backend.services.rider_counter import (
    BikeBox,
    PersonBox,
    _path_a,
    _path_b,
    count_riders,
    iou,
)


def test_path_a_pedestrian_excluded():
    bike = BikeBox(100, 100, 200, 200)
    # Distant pedestrian (IoU ~ 0)
    ped = PersonBox(300, 300, 350, 400, confidence=0.9)
    assert iou(ped, bike) < 0.20
    assert _path_a([ped], bike) == 0


def test_path_a_pillion_included():
    bike = BikeBox(100, 100, 200, 200)
    # Rider overlapping bike (IoU >= 0.20)
    rider = PersonBox(110, 80, 170, 190, confidence=0.9)
    assert iou(rider, bike) >= 0.20
    assert _path_a([rider], bike) == 1


def test_path_a_hungarian_cost_matrix():
    bike = BikeBox(100, 100, 200, 200)
    rider1 = PersonBox(110, 80, 150, 180, confidence=0.95)
    rider2 = PersonBox(140, 80, 180, 180, confidence=0.95)
    rider3 = PersonBox(160, 80, 195, 180, confidence=0.90)
    ped = PersonBox(220, 220, 250, 300, confidence=0.85)

    count = _path_a([rider1, rider2, rider3, ped], bike)
    assert count == 3


def test_path_b_frontal_3_riders():
    # Synthetic frame with 3 bright head peaks along the horizontal slice
    frame = np.full((300, 400, 3), 50, dtype=np.uint8)
    bike = BikeBox(50, 50, 350, 250)

    # 3 distinct heads in upper slice
    frame[60:90, 100:130] = 240
    frame[60:90, 180:210] = 240
    frame[60:90, 260:290] = 240

    count = _path_b(frame, bike, camera_angle_deg=0.0)
    assert count == 3


def test_path_b_dark_clothing():
    # Frame with subtle peaks
    frame = np.full((300, 400, 3), 40, dtype=np.uint8)
    bike = BikeBox(50, 50, 350, 250)

    frame[60:90, 100:130] = 75
    frame[60:90, 180:210] = 75
    frame[60:90, 260:290] = 75

    count = _path_b(frame, bike, camera_angle_deg=0.0)
    assert count == 3


def test_fusion_agreement_and_haar_tiebreak():
    frame = np.zeros((300, 400, 3), dtype=np.uint8)
    bike = BikeBox(100, 100, 250, 250)

    # Case 1: Paths A and B agree (both 2)
    p1 = PersonBox(110, 80, 160, 200, 0.9)
    p2 = PersonBox(170, 80, 220, 200, 0.9)

    # Put 2 heads in frame for Path B
    frame[110:130, 130:150] = 220
    frame[110:130, 190:210] = 220

    res, desc = count_riders(frame, bike, [p1, p2])
    assert res == 2
    assert "ab_agree" in desc
