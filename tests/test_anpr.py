"""
tests/test_anpr.py — 3-Stage Indian ANPR Format & Penalty Tests.

Covers:
  - Test 18: ANPR valid standard plate format (e.g. GJ01BV9921, 24BH1234A).
  - Test 19: ANPR low-confidence handling (preserves text, flags unconfirmed).
  - Test 20: ANPR invalid format (penalizes confidence by 0.5x, format_valid = False).
"""
import pytest

from backend.services.anpr import (
    PlateResult,
    _validate_format,
    read_plate,
    should_auto_populate,
)


def test_validate_format_standard_and_bh():
    # Valid standard Gujarat plate
    valid, mult = _validate_format("GJ01BV9921")
    assert valid is True
    assert mult == 1.0

    # Valid standard plate with 3 series letters (e.g. DL01CAA1234)
    valid, mult = _validate_format("DL01CAA1234")
    assert valid is True
    assert mult == 1.0

    # Valid Bharat (BH) series plate
    valid, mult = _validate_format("24BH1234A")
    assert valid is True
    assert mult == 1.0

    valid, mult = _validate_format("22BH9999AB")
    assert valid is True
    assert mult == 1.0


def test_validate_format_invalid_penalty():
    # Non-conforming plate text
    valid, mult = _validate_format("XYZ12345Z")
    assert valid is False
    assert mult == 0.5  # 0.5x penalty

    valid, mult = _validate_format("INVALID_PLATE")
    assert valid is False
    assert mult == 0.5


def test_should_auto_populate():
    res_high = PlateResult("GJ01BV9921", confidence=0.85, crop_path=None, format_valid=True)
    assert should_auto_populate(res_high, min_conf=0.75) is True

    res_low = PlateResult("GJ01BV9921", confidence=0.60, crop_path=None, format_valid=True)
    assert should_auto_populate(res_low, min_conf=0.75) is False

    res_none = PlateResult(None, confidence=0.0, crop_path=None, format_valid=False)
    assert should_auto_populate(res_none, min_conf=0.75) is False
