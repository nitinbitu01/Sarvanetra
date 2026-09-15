"""
tests/test_reviewer_queue.py — Reviewer Queue Prioritization Tests (v15.0.0).
"""
import time
import pytest

from backend.services.reviewer_queue import compute_priority


def test_stolen_vehicle_scores_highest():
    # Stolen vehicle gets +50 points bonus
    v_stolen = {
        "id": 1,
        "majority_ratio": 0.85,
        "vahan_stolen": True,
        "vahan_insurance_valid": True,
        "vahan_pucc_valid": True,
        "violation_type": "TRIPLE_RIDING",
        "created_at_epoch": time.time(),
    }
    v_normal = {
        "id": 2,
        "majority_ratio": 0.85,
        "vahan_stolen": False,
        "vahan_insurance_valid": True,
        "vahan_pucc_valid": True,
        "violation_type": "TRIPLE_RIDING",
        "created_at_epoch": time.time(),
    }
    score_stolen = compute_priority(v_stolen)
    score_normal = compute_priority(v_normal)

    assert score_stolen >= score_normal + 50.0


def test_borderline_ratio_scores_lower():
    v_high_conf = {
        "id": 3,
        "majority_ratio": 0.95,
        "vahan_stolen": False,
        "vahan_insurance_valid": True,
        "vahan_pucc_valid": True,
        "violation_type": "WRONG_WAY",
        "created_at_epoch": time.time(),
    }
    v_borderline = {
        "id": 4,
        "majority_ratio": 0.66,
        "vahan_stolen": False,
        "vahan_insurance_valid": True,
        "vahan_pucc_valid": True,
        "violation_type": "WRONG_WAY",
        "created_at_epoch": time.time(),
    }
    assert compute_priority(v_high_conf) > compute_priority(v_borderline)
