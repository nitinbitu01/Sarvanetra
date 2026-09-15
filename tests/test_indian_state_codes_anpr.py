"""
tests/test_indian_state_codes_anpr.py — Verification of All-India State/UT Codes & OCR Optical Confusion Resolution.

Validates that:
1. All 32 State & UT codes specified by MoRTH are recognized and validated.
2. Common CCTV OCR confusions (CJ, LJ, 6J, OJ, QJ, GI, GL, CI, LI, 6I, 6L) on Gujarat plates ('GJ')
   are accurately projected and resolved to 'GJ' under optical confusion distance.
3. Out-of-state plates (MH, DL, RJ, UP, MP, KA, HR, TN, KL, WB, AP, BR, etc.) are 100% preserved.
4. ANPREngine and decode_plate produce valid Indian plates with correct state codes.
"""
import pytest
from backend.scripts.indian_plate_grammar import (
    STATE_CODES,
    SPECIAL_PREFIX,
    decode_plate,
    resolve_state_code,
    apply_state_prior,
    is_plausible,
)
from backend.services.anpr import _validate_format


# 32 States and Union Territories requested by user
USER_SPECIFIED_STATES = [
    ("AN", "Andaman and Nicobar"),
    ("AP", "Andhra Pradesh"),
    ("AR", "Arunachal Pradesh"),
    ("AS", "Assam"),
    ("BR", "Bihar"),
    ("CH", "Chandigarh"),
    ("DN", "Dadra and Nagar Haveli"),
    ("DD", "Daman and Diu"),
    ("DL", "Delhi"),
    ("GA", "Goa"),
    ("GJ", "Gujarat"),
    ("HR", "Haryana"),
    ("HP", "Himachal Pradesh"),
    ("JK", "Jammu and Kashmir"),
    ("KA", "Karnataka"),
    ("KL", "Kerala"),
    ("LD", "Lakshadweep"),
    ("MP", "Madhya Pradesh"),
    ("MH", "Maharashtra"),
    ("MN", "Manipur"),
    ("ML", "Meghalaya"),
    ("MZ", "Mizoram"),
    ("NL", "Nagaland"),
    ("OR", "Orissa"),
    ("PY", "Pondicherry"),
    ("PN", "Punjab"),
    ("RJ", "Rajasthan"),
    ("SK", "Sikkim"),
    ("TN", "TamilNadu"),
    ("TR", "Tripura"),
    ("UP", "Uttar Pradesh"),
    ("WB", "West Bengal"),
]


def test_all_32_user_specified_state_codes_present():
    """Verify all 32 state/UT codes are recognized in the system."""
    for code, name in USER_SPECIFIED_STATES:
        assert code in STATE_CODES, f"State code {code} ({name}) must be in STATE_CODES"


@pytest.mark.parametrize("ocr_prefix,expected_state", [
    ("CJ", "GJ"),  # Missing G horizontal crossbar
    ("LJ", "GJ"),  # Misclassified G/J stroke
    ("6J", "GJ"),  # Round contour misclassified as 6
    ("OJ", "GJ"),  # Oval loop misclassified as O
    ("QJ", "GJ"),  # Tail misclassified as Q
    ("GI", "GJ"),  # J bottom curve clipped as I
    ("GL", "GJ"),  # J bottom curve clipped as L
    ("CI", "GJ"),  # C + I optical confusion for G + J
    ("LI", "GJ"),  # L + I optical confusion for G + J
    ("6I", "GJ"),  # 6 + I optical confusion for G + J
    ("6L", "GJ"),  # 6 + L optical confusion for G + J
])
def test_gujarat_ocr_confusions_resolve_to_gj(ocr_prefix, expected_state):
    """Verify that common CCTV OCR misreadings of GJ (CJ, LJ, 6J, GI, etc.) resolve to GJ."""
    resolved, cost = resolve_state_code(ocr_prefix, local_state="GJ")
    assert resolved == expected_state, f"Raw prefix {ocr_prefix} must resolve to {expected_state}, got {resolved} (cost {cost})"
    assert cost <= 0.60, f"Optical cost for {ocr_prefix} -> {expected_state} should be <= 0.60, got {cost}"


@pytest.mark.parametrize("state_code", [
    "AN", "AP", "AR", "AS", "BR", "CH", "DN", "DD", "DL", "GA", "GJ", "HR",
    "HP", "JK", "KA", "KL", "LD", "MP", "MH", "MN", "ML", "MZ", "NL", "OD",
    "OR", "PB", "PN", "PY", "RJ", "SK", "TN", "TR", "UP", "WB", "CG", "JH",
    "TS", "UK", "LA"
])
def test_valid_indian_state_codes_preserved_intact(state_code):
    """Verify that all genuine Indian state codes are preserved with 0.0 cost."""
    resolved, cost = resolve_state_code(state_code)
    assert resolved == state_code
    assert cost == 0.0


def test_decode_plate_fixes_cj_and_lj_in_full_strings():
    """Verify full plate strings starting with CJ or LJ or 6J decode cleanly to GJ."""
    # CJ01BV9921 -> GJ01BV9921
    d1 = decode_plate("CJ01BV9921")
    assert d1["plate"] == "GJ01BV9921"
    assert d1["state"] == "GJ"
    assert is_plausible(d1)

    # LJ01BV9921 -> GJ01BV9921
    d2 = decode_plate("LJ01BV9921")
    assert d2["plate"] == "GJ01BV9921"
    assert d2["state"] == "GJ"
    assert is_plausible(d2)

    # 6J32K4588 -> GJ32K4588
    d3 = decode_plate("6J32K4588")
    assert d3["plate"] == "GJ32K4588"
    assert d3["state"] == "GJ"
    assert is_plausible(d3)

    # GI01BV9921 -> GJ01BV9921
    d4 = decode_plate("GI01BV9921")
    assert d4["plate"] == "GJ01BV9921"
    assert d4["state"] == "GJ"
    assert is_plausible(d4)


def test_decode_plate_all_india_coverage():
    """Verify decoding across diverse Indian states and series."""
    samples = [
        ("MH12AB1234", "MH12AB1234", "MH"),
        ("DL8CAF5030", "DL8CAF5030", "DL"),
        ("RJ14CV0002", "RJ14CV0002", "RJ"),
        ("UP32BC1234", "UP32BC1234", "UP"),
        ("KA05MK4321", "KA05MK4321", "KA"),
        ("22BH1234AB", "22BH1234AB", "BH"),
    ]
    for raw, expected_plate, expected_state in samples:
        d = decode_plate(raw)
        assert d["plate"] == expected_plate, f"Expected {expected_plate}, got {d['plate']}"
        assert d["state"] == expected_state, f"Expected state {expected_state}, got {d['state']}"
        assert is_plausible(d), f"Plate {expected_plate} should be plausible"


def test_format_validation_with_state_codes():
    """Verify format validator accepts all valid Indian state plates and rejects invalid."""
    assert _validate_format("GJ01BV9921")[0] is True
    assert _validate_format("MH12AB1234")[0] is True
    assert _validate_format("DL01CAA1234")[0] is True
    assert _validate_format("22BH1234AB")[0] is True
    assert _validate_format("KA05MK4321")[0] is True

    # Invalid state prefixes (e.g. XX, ZZ)
    assert _validate_format("XX01BV9921")[0] is False
    assert _validate_format("ZZ12AB1234")[0] is False
