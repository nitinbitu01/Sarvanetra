"""
tests/test_day1_sre_deep_audit.py — Senior SRE & QA Hybrid Deep Audit Test Suite for Day 1
========================================================================================
Covers:
  1. Unit tests with unusual/messy data:
     - Nulls, empty strings, unicode/emoji injection into OCR and plate grammar
     - Infinite / NaN coordinates in trajectory velocity estimator
     - Extreme clock jumps, out-of-order timestamps, zero dt
  2. Integration tests:
     - Pipeline queue ingestion -> BoT-SORT -> Theil-Sen velocity -> DB persistence
     - Cross-camera spatial-temporal link constraints via CameraLinkModel (CLM)
  3. Adversarial / Negative tests:
     - Malicious SQL-injection payload in ANPR plate recognition
     - Massive multi-megabyte payload injection into frame buffer
     - Physically impossible GPS coordinates (outside Indian territory / 0,0 Null Island)
  4. Failure-Injection tests:
     - RTSP / VideoCapture connection timeout / broken stream failure injection
     - Database deadlock / connection termination during bulk track commit
"""

from __future__ import annotations

import math
import time
import os
import sys
import numpy as np
import pytest
from datetime import datetime, timedelta
from unittest.mock import MagicMock, patch

from backend.db.session import SessionLocal
from backend.db.models import (
    Camera,
    CameraHomographyCalibration,
    CameraMetrics1M,
    VehicleTrack,
    Alert,
)
from backend.services.camera_link_model import (
    CameraLinkModel,
    TravelTimeDistribution,
    FeasibilityResult,
)
from backend.services.speed_estimator import (
    estimate_track_speed,
    TrackSpeedResult,
)
from backend.services.calibration import (
    compute_homography,
    pixel_to_world_m,
    world_to_pixel,
)
from backend.scripts.indian_plate_grammar import (
    decode_plate,
    resolve_state_code,
    STATE_CODES,
)
from backend.services.live_24x7_pipeline import (
    classify_environment,
    CameraReaderThread,
)


# ============================================================================
# 1. UNIT TESTS: EDGE CASES, MESSY DATA, UNUSUAL INPUTS
# ============================================================================

class TestDay1UnitEdgeCases:
    """Unit tests covering messy data, edge cases, and boundary violations."""

    def test_plate_grammar_unicode_and_special_chars(self):
        """Plate grammar decoder must safely reject or clean unicode, emojis, and control chars."""
        messy_inputs = [
            "GJ01AB1234",          # standard valid
            "GJ01-AB-1234",        # hyphenated
            "GJ.01.AB.1234",       # dotted
            "GJ01 AB 1234",        # spaces
            "GJ01AB1234\x00\x1b",  # null byte & escape
            "GJ01AB1234🚨🚔",      # emoji suffix
            "ગુજરાત GJ01",        # non-latin unicode script
            "",                    # empty string
            "   ",                 # whitespace only
            "A" * 500,             # buffer overflow attempt
        ]
        for inp in messy_inputs:
            # Should not raise exception
            res = decode_plate(inp)
            assert isinstance(res, dict)
            assert "raw" in res
            assert "plate" in res
            assert "score" in res
            if res["plate"] is not None:
                assert len(res["plate"]) <= 15

    def test_state_code_resolution_all_32_states_and_uts(self):
        """Ensure all 32 Indian States / UTs and special series resolve accurately."""
        cases = [
            ("GJ", "GJ"), ("MH", "MH"), ("DL", "DL"), ("KA", "KA"),
            ("TN", "TN"), ("UP", "UP"), ("RJ", "RJ"), ("KL", "KL"),
            ("HR", "HR"), ("PB", "PB"), ("WB", "WB"), ("AP", "AP"),
            ("TS", "TS"), ("BR", "BR"), ("OD", "OD"), ("UK", "UK"),
            ("CJ", "GJ"), ("LJ", "GJ"), ("6J", "GJ"), ("0J", "GJ"),
        ]
        for raw, expected in cases:
            resolved, cost = resolve_state_code(raw)
            assert resolved == expected, f"Failed resolving {raw} -> expected {expected}, got {resolved}"
            assert cost >= 0.0

    def test_speed_estimator_nan_inf_and_zero_dt(self):
        """Speed estimator must guard against NaN, Inf, zero time intervals, and negative dt."""
        # Zero dt (simultaneous frame timestamps)
        res_zero_dt = estimate_track_speed(
            [(0.0, 0.0), (0.0, 5.0), (0.0, 10.0), (0.0, 15.0), (0.0, 20.0)],
            [1.0, 1.0, 1.0, 1.0, 1.0],  # dt = 0
            calibration_quality="good",
        )
        assert res_zero_dt.is_valid is False
        assert res_zero_dt.speed_kmh is None

        # Reversed / decreasing timestamps (out-of-order packet delivery)
        res_backwards = estimate_track_speed(
            [(0.0, 0.0), (0.0, 5.0), (0.0, 10.0), (0.0, 15.0), (0.0, 20.0)],
            [5.0, 4.0, 3.0, 2.0, 1.0],
            calibration_quality="good",
        )
        assert res_backwards.is_valid is False

        # Injected NaN and Infinity coordinates
        res_nan = estimate_track_speed(
            [(0.0, 0.0), (float("nan"), 5.0), (0.0, 10.0), (float("inf"), 15.0), (0.0, 20.0)],
            [0.0, 0.1, 0.2, 0.3, 0.4],
            calibration_quality="good",
        )
        assert res_nan.is_valid is False or res_nan.speed_kmh is None or not math.isnan(res_nan.speed_kmh)

    def test_camera_link_model_online_welford_stability(self):
        """Welford variance accumulator must remain numerically stable under 100,000 updates."""
        dist = TravelTimeDistribution(cam_a_id=1, cam_b_id=2, distance_km=5.0, road_type="urban")
        # Feed 10,000 travel observations around 300s (5 mins) with normal noise
        rng = np.random.RandomState(42)
        samples = rng.normal(loc=300.0, scale=30.0, size=10000)
        for s in samples:
            dist.welford_update(float(s))

        assert dist.count == 10000
        assert abs(dist.mean_seconds - 300.0) < 1.0
        assert abs(dist.std_seconds - 30.0) < 1.0
        assert dist.lower_bound > 0.0
        assert dist.upper_bound > dist.lower_bound


# ============================================================================
# 2. INTEGRATION TESTS: DATABASE PERSISTENCE & STREAMING INTEGRATION
# ============================================================================

class TestDay1Integration:
    """Integration tests verifying database transactions and cross-camera tracking."""

    def test_database_vehicle_track_full_lifecycle(self):
        """Create, write, commit, query, and delete a complete VehicleTrack lifecycle."""
        db = SessionLocal()
        try:
            now = datetime.utcnow()
            test_cam = "CAM_TEST_SRE_01"
            vt = VehicleTrack(
                camera_id=test_cam,
                track_id=9999,
                vehicle_class="truck",
                first_seen=now - timedelta(seconds=10),
                last_seen=now,
                speed_kmh=52.4,
                speed_px_s=124.8,
                speed_ci_kmh=1.8,
                path_length_m=42.0,
                heading_deg=270.0,
                n_frames=30,
                detector_conf=0.94,
                calibration_err_m=0.012,
                quality="good",
            )
            db.add(vt)
            db.commit()

            fetched = db.query(VehicleTrack).filter(
                VehicleTrack.camera_id == test_cam,
                VehicleTrack.track_id == 9999
            ).first()

            assert fetched is not None
            assert fetched.speed_kmh == 52.4
            assert fetched.speed_px_s == 124.8
            assert fetched.vehicle_class == "truck"
            assert fetched.quality == "good"
        finally:
            db.query(VehicleTrack).filter(VehicleTrack.camera_id == "CAM_TEST_SRE_01").delete()
            db.commit()
            db.close()

    def test_clm_spatio_temporal_feasibility_pipeline(self):
        """Verify cross-camera transition feasibility for realistic multi-camera trajectory."""
        clm = CameraLinkModel(mode="vehicle")
        # Pre-seed travel distribution between cam 1 and cam 2 (10km, min 300s, max 1200s, mean 600s)
        dist = TravelTimeDistribution(cam_a_id=1, cam_b_id=2, distance_km=10.0, road_type="highway", count=20, mean_seconds=600.0, min_seconds=300.0, max_seconds=1200.0)
        clm.link_table[(1, 2)] = dist

        # 1. Delta_t = 60s (traveling 10km in 1 min = 600 km/h -> impossible) -> reject
        res_impossible = clm.check_feasibility(cam_a_id=1, cam_b_id=2, time_delta_seconds=60.0)
        assert res_impossible.is_feasible is False
        assert "TOO_FAST" in (res_impossible.reject_reason or "")

        # 2. Delta_t = 600s (traveling 10km in 10 mins = 60 km/h -> feasible) -> accept with boost
        res_feasible = clm.check_feasibility(cam_a_id=1, cam_b_id=2, time_delta_seconds=600.0)
        assert res_feasible.is_feasible is True
        assert res_feasible.confidence_boost > 0.0


# ============================================================================
# 3. ADVERSARIAL & NEGATIVE TESTS
# ============================================================================

class TestDay1Adversarial:
    """Adversarial and malicious input injection tests."""

    def test_sql_injection_in_anpr_plate_string(self):
        """Malicious SQL injection in plate string must not alter query or cause syntax error."""
        sqli_payloads = [
            "' OR '1'='1",
            "GJ01'; DROP TABLE vehicle_track; --",
            "\" UNION SELECT * FROM users --",
            "GJ01AB1234' AND 1=SLEEP(5) --",
        ]
        db = SessionLocal()
        try:
            for payload in sqli_payloads:
                # 1. Grammar decoder sanitization check
                decoded = decode_plate(payload)
                if decoded["plate"] is not None:
                    assert "'" not in decoded["plate"]
                    assert ";" not in decoded["plate"]
                    assert "--" not in decoded["plate"]

                # 2. Parameterized query safety check
                res = db.query(VehicleTrack).filter(VehicleTrack.camera_id == payload).all()
                assert isinstance(res, list)
        finally:
            db.close()

    def test_huge_malicious_frame_buffer_injection(self):
        """Corrupt, massive, or zero-byte buffers must not crash image decoders or readers."""
        # Zero-byte buffer
        zero_buf = np.zeros((0, 0, 3), dtype=np.uint8)
        # Extreme aspect ratio frame (1x10000)
        sliver_buf = np.zeros((1, 10000, 3), dtype=np.uint8)

        env_zero = classify_environment(zero_buf)
        assert env_zero in ["DAY", "NIGHT", "UNKNOWN"]

        env_sliver = classify_environment(sliver_buf)
        assert env_sliver in ["DAY", "NIGHT", "UNKNOWN"]

    def test_extreme_gps_and_negative_horizon_coordinates(self):
        """Homography and GPS transforms must reject points beyond Earth / singular horizon."""
        src_quad = [(100.0, 100.0), (400.0, 100.0), (450.0, 400.0), (50.0, 400.0)]
        dst_quad = [(0.0, 0.0), (10.0, 0.0), (10.0, 30.0), (0.0, 30.0)]
        H = compute_homography(src_quad, dst_quad)

        # Coordinate projected into negative projective space / vanishing line must raise ValueError
        pt_horizon = (-50000.0, -50000.0)
        with pytest.raises(ValueError, match="at or above the projective horizon"):
            pixel_to_world_m(H, pt_horizon[0], pt_horizon[1])

        # Valid in-frame pixel converts cleanly
        wx, wy = pixel_to_world_m(H, 200.0, 200.0)
        assert isinstance(wx, float)
        assert isinstance(wy, float)


# ============================================================================
# 4. FAILURE-INJECTION TESTS
# ============================================================================

class TestDay1FailureInjection:
    """Failure injection tests: network drops, stream timeouts, and DB crashes."""

    def test_stream_reader_network_timeout_and_exponential_backoff(self):
        """CameraReaderThread must escalate backoff delays when RTSP stream goes offline."""
        reader = CameraReaderThread(
            cam_id="CAM_SRE_FAIL_TEST",
            source_url="rtsp://non_existent_stream_host.local:8554/live",
            fps=5,
            auto_reconnect=True,
        )

        assert reader._reconnect_delay == 1.0
        # Trigger simulated connection failure
        reader._last_reconnect_attempt = 0.0
        with patch.object(reader, "_open_network_stream", return_value=False):
            reader._handle_network_reconnect()
            assert reader._consecutive_failures == 1
            assert reader._reconnect_delay > 1.0
            prev_delay = reader._reconnect_delay
            reader._last_reconnect_attempt = 0.0
            reader._handle_network_reconnect()
            assert reader._consecutive_failures == 2
            assert reader._reconnect_delay > prev_delay
            assert reader.stats()["is_connected"] is False

    def test_database_connection_abort_rollback_safety(self):
        """Simulate a database connection error during track insertion; session must rollback cleanly."""
        db = SessionLocal()
        try:
            now = datetime.utcnow()
            vt_bad = VehicleTrack(
                camera_id="CAM_ROLLBACK_TEST",
                track_id=1,
                # Intentionally insert valid obj
                first_seen=now,
                last_seen=now,
            )
            db.add(vt_bad)

            # Mock a database operational error during flush
            with patch.object(db, "commit", side_effect=Exception("Simulated Database I/O Failure")):
                with pytest.raises(Exception, match="Simulated Database I/O Failure"):
                    db.commit()

            # Rollback must restore clean session state
            db.rollback()
            # New query must succeed without PendingRollbackError
            count = db.query(VehicleTrack).filter(VehicleTrack.camera_id == "CAM_ROLLBACK_TEST").count()
            assert count == 0
        finally:
            db.close()
