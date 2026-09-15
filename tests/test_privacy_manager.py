"""
tests/test_privacy_manager.py — DPDPA 2023 Data Privacy & Retention Tests (v15.0.0).
"""
import pytest

from backend.services.privacy_manager import (
    pseudonymize_plate,
    purge_expired_evidence,
)


def test_pseudonymization_one_way_and_salted():
    plate = "GJ01AB1234"
    salt1 = "salt_alpha"
    salt2 = "salt_beta"

    hash1_a = pseudonymize_plate(plate, salt1)
    hash1_b = pseudonymize_plate(plate, salt1)
    hash2 = pseudonymize_plate(plate, salt2)

    # Same salt produces exact identical hash
    assert hash1_a == hash1_b
    # Different salt produces distinct hash
    assert hash1_a != hash2
    # Output does not contain raw plate
    assert plate not in hash1_a


def test_purge_expired_evidence_runs_safely(tmp_path):
    db_file = str(tmp_path / "privacy_test.db")
    res = purge_expired_evidence(db_path=db_file)
    assert "deleted_clips" in res
    assert "anonymized_records" in res
