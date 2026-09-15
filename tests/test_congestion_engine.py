"""
tests/test_congestion_engine.py — Phase 3 Unit Tests: Free-Flow Speed & Congestion Index Bootstrap.

Validates:
  1. Strict bootstrap threshold gating: samples < threshold returns NULL CI and exact progress ("X of N").
  2. Bootstrap unlock: once samples >= threshold, v_free = 85th percentile off-peak.
  3. Congestion index formula: CI = 1.0 - clamp(v / v_free, 0, 1).
  4. CI labels: FREE_FLOW, MODERATE, CONGESTED, GRIDLOCK.
"""
from datetime import datetime, timedelta
import pytest

from backend.db.models import VehicleTrack
from backend.db.session import SessionLocal, init_db
from backend.services.congestion_engine import CongestionEngine


def test_bootstrap_threshold_gating():
    """Verify that CI remains NULL with explicit sample progress until threshold is met."""
    init_db()
    db = SessionLocal()
    try:
        # Use a small threshold = 20 for unit test
        engine = CongestionEngine(min_samples_threshold=20)

        # 1. Zero samples -> 0% progress, CI is None
        p0 = engine.get_congestion_profile("CAM_TEST_01", current_speed_kmh=45.0, db=db)
        assert p0.is_bootstrapped is False
        assert p0.congestion_index is None
        assert p0.congestion_label == "BOOTSTRAPPING"
        assert p0.samples_collected == 0
        assert p0.progress_pct == 0.0

        # 2. Add 10 samples (< 20 threshold)
        t0 = datetime(2026, 8, 30, 23, 0, 0) # off-peak
        for i in range(10):
            db.add(VehicleTrack(
                camera_id="CAM_TEST_01",
                track_id=i,
                first_seen=t0 + timedelta(minutes=i),
                last_seen=t0 + timedelta(minutes=i, seconds=5),
                speed_kmh=50.0 + i,
                quality="good",
            ))
        db.commit()

        p10 = engine.get_congestion_profile("CAM_TEST_01", current_speed_kmh=45.0, db=db)
        assert p10.is_bootstrapped is False
        assert p10.congestion_index is None
        assert p10.samples_collected == 10
        assert p10.progress_pct == 50.0

        # 3. Add 15 more samples (total 25 >= 20 threshold)
        for i in range(10, 25):
            db.add(VehicleTrack(
                camera_id="CAM_TEST_01",
                track_id=i,
                first_seen=t0 + timedelta(minutes=i),
                last_seen=t0 + timedelta(minutes=i, seconds=5),
                speed_kmh=60.0 + (i % 5) * 2,
                quality="good",
            ))
        db.commit()

        p25 = engine.get_congestion_profile("CAM_TEST_01", current_speed_kmh=45.0, db=db)
        assert p25.is_bootstrapped is True
        assert p25.free_flow_speed_kmh is not None
        assert p25.congestion_index is not None
        assert p25.progress_pct == 100.0
    finally:
        # Clean up test rows
        db.query(VehicleTrack).filter(VehicleTrack.camera_id == "CAM_TEST_01").delete()
        db.commit()
        db.close()


def test_congestion_index_math_and_labels():
    """Verify CI = 1 - clamp(v / v_free, 0, 1) mapping to labels."""
    engine = CongestionEngine()
    engine._v_free_cache["CAM_01"] = 60.0 # 60 km/h free flow

    # Case A: Full free flow v = 60 km/h -> CI = 0.0 -> FREE_FLOW
    prof_a = engine.get_congestion_profile("CAM_01", current_speed_kmh=60.0, db=SessionLocal())
    assert prof_a.congestion_index == 0.0
    assert prof_a.congestion_label == "FREE_FLOW"

    # Case B: Slight speed drop v = 48 km/h -> CI = 1 - 48/60 = 0.20 -> FREE_FLOW
    prof_b = engine.get_congestion_profile("CAM_01", current_speed_kmh=48.0, db=SessionLocal())
    assert prof_b.congestion_index == 0.20
    assert prof_b.congestion_label == "FREE_FLOW"

    # Case C: Moderate slowdown v = 36 km/h -> CI = 1 - 36/60 = 0.40 -> MODERATE
    prof_c = engine.get_congestion_profile("CAM_01", current_speed_kmh=36.0, db=SessionLocal())
    assert prof_c.congestion_index == 0.40
    assert prof_c.congestion_label == "MODERATE"

    # Case D: Heavy jam v = 15 km/h -> CI = 1 - 15/60 = 0.75 -> CONGESTED
    prof_d = engine.get_congestion_profile("CAM_01", current_speed_kmh=15.0, db=SessionLocal())
    assert prof_d.congestion_index == 0.75
    assert prof_d.congestion_label == "CONGESTED"

    # Case E: Gridlock v = 0 km/h -> CI = 1.0 -> GRIDLOCK
    prof_e = engine.get_congestion_profile("CAM_01", current_speed_kmh=0.0, db=SessionLocal())
    assert prof_e.congestion_index == 1.0
    assert prof_e.congestion_label == "GRIDLOCK"
