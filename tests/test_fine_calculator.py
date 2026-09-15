"""
tests/test_fine_calculator.py — Motor Vehicles Act 2019 Fine Computation Tests (Tests 26-28).
"""
import pytest

from backend.services.fine_calculator import ViolationBreakdown
from backend.services.helmet_detector import HelmetResult


def test_fine_3_riders_0_helmets():
    helmets = [
        HelmetResult(0, False, 0.85, "No Helmet"),
        HelmetResult(1, False, 0.85, "No Helmet"),
        HelmetResult(2, False, 0.85, "No Helmet"),
    ]
    vb = ViolationBreakdown(rider_count=3, helmet_results=helmets, is_repeat=False)
    vb.compute()
    # S128 (1000) + S129 (3 * 1000) = 4000
    assert vb.total_inr == 4000
    assert len(vb.line_items) == 4


def test_fine_3_riders_1_helmet():
    helmets = [
        HelmetResult(0, True, 0.90, "Helmet"),
        HelmetResult(1, False, 0.85, "No Helmet"),
        HelmetResult(2, False, 0.85, "No Helmet"),
    ]
    vb = ViolationBreakdown(rider_count=3, helmet_results=helmets, is_repeat=False)
    vb.compute()
    # S128 (1000) + S129 (2 * 1000) = 3000
    assert vb.total_inr == 3000


def test_fine_repeat_offender():
    helmets = [
        HelmetResult(0, False, 0.85, "No Helmet"),
        HelmetResult(1, False, 0.85, "No Helmet"),
        HelmetResult(2, False, 0.85, "No Helmet"),
    ]
    vb = ViolationBreakdown(rider_count=3, helmet_results=helmets, is_repeat=True)
    vb.compute()
    # S128 repeat (2000) + S129 (3 * 1000) = 5000
    assert vb.total_inr == 5000
