"""
tests/test_vahan_service.py — VAHAN 4.0 Three-Tier Fallback Tests (v15.0.0).
"""
import os
import sqlite3
import pytest

from backend.services.vahan_service import (
    LookupTier,
    init_vahan_mirror,
    lookup_plate,
)


def test_vahan_tier3_format_validation():
    # Valid Gujarat plate with no DB mirror -> Tier 3
    rec = lookup_plate("GJ01AB1234", db_path="non_existent.db")
    assert rec.lookup_tier == LookupTier.FORMAT_ONLY
    assert rec.license_plate == "GJ01AB1234"
    assert rec.is_stolen is False


def test_vahan_tier2_local_db_mirror(tmp_path):
    db_file = str(tmp_path / "vahan_test.db")
    init_vahan_mirror(db_file)

    conn = sqlite3.connect(db_file)
    conn.execute("""
        INSERT INTO vehicles (license_plate, owner_name, make_model, vehicle_class, insurance_expiry, pucc_validity, is_stolen)
        VALUES ('GJ01CD5678', 'Rahul Sharma', 'Honda Activa 6G', '2W', '2027-12-31', '2027-06-30', 0)
    """)
    conn.commit()
    conn.close()

    rec = lookup_plate("GJ01CD5678", db_path=db_file)
    assert rec.lookup_tier == LookupTier.LOCAL_DB
    assert rec.owner_name == "Rahul Sharma"
    assert rec.make_model == "Honda Activa 6G"
    assert rec.insurance_valid is True
    assert rec.pucc_valid is True
    assert rec.is_stolen is False


def test_vahan_stolen_vehicle_alert(tmp_path):
    db_file = str(tmp_path / "vahan_test.db")
    init_vahan_mirror(db_file)

    conn = sqlite3.connect(db_file)
    conn.execute("""
        INSERT INTO vehicles (license_plate, owner_name, make_model, is_stolen, cctns_fir_number)
        VALUES ('GJ01ST9999', 'Unknown', 'Bajaj Pulsar', 1, 'FIR/2026/089')
    """)
    conn.commit()
    conn.close()

    rec = lookup_plate("GJ01ST9999", db_path=db_file)
    assert rec.is_stolen is True
    assert rec.cctns_fir_number == "FIR/2026/089"
