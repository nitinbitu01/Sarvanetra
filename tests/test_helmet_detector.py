"""
tests/test_helmet_detector.py — Helmet Detection & No-Helmet Counting Tests (Tests 23-25).
"""
import numpy as np
import pytest

from backend.services.helmet_detector import (
    HelmetResult,
    count_no_helmet,
    detect_helmets,
)


def test_helmet_results_counting():
    res = [
        HelmetResult(rider_index=0, has_helmet=True, confidence=0.88, label="Helmet"),
        HelmetResult(rider_index=1, has_helmet=False, confidence=0.82, label="No Helmet"),
        HelmetResult(rider_index=2, has_helmet=False, confidence=0.90, label="No Helmet"),
    ]
    assert count_no_helmet(res) == 2


def test_unknown_confidence_excluded_from_fine():
    res = [
        HelmetResult(rider_index=0, has_helmet=True, confidence=0.88, label="Helmet"),
        HelmetResult(rider_index=1, has_helmet=False, confidence=0.40, label="Unknown"),
    ]
    # Unknown label is not counted in no_helmet fines to avoid unfair penalties
    assert count_no_helmet(res) == 0
