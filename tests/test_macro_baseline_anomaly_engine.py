"""
tests/test_macro_baseline_anomaly_engine.py — SRE QA Test Suite for Gap 3 (Empirical Bayes Baseline & Contextual Anomaly Engine).
"""
import math
from datetime import datetime, timedelta, timezone
import numpy as np
import pytest
from fastapi.testclient import TestClient

from backend.main import app
from backend.auth.dependencies import get_current_user
from backend.db.session import SessionLocal
from backend.db.models import Camera, CameraMetrics1M
from backend.services.traffic_baseline import (
    TrafficBaselineEngine,
    get_macro_network_summary,
    PRIOR_PSEUDO_COUNT_N0,
    DEFAULT_URBAN_MEAN_SPEED,
)

import pytest

# Mock authenticated admin user
class MockAdminOfficer:
    id = "officer-01"
    badge_number = "GJ-POL-001"
    rank = "Superintendent"
    role = "ADMIN"
    department = "ALL"
    jurisdiction = "Ahmedabad"

@pytest.fixture(autouse=True)
def setup_auth_override():
    app.dependency_overrides[get_current_user] = lambda: MockAdminOfficer()
    yield

app.dependency_overrides[get_current_user] = lambda: MockAdminOfficer()
client = TestClient(app)


class TestEmpiricalBayesBaselineMath:
    """Rigorous tests for Empirical Bayesian shrinkage of seasonal speed and volume baselines."""

    def test_bayesian_shrinkage_on_small_samples(self):
        """Small sample (n=2) must shrink significantly toward the road prior mu_0 = 42 km/h."""
        engine = TrafficBaselineEngine()
        db = SessionLocal()
        cam_id = "CAM_01"
        try:
            now = datetime.now(timezone.utc)
            db.query(CameraMetrics1M).filter(CameraMetrics1M.camera_id == cam_id).delete()
            for i in range(2):
                db.add(CameraMetrics1M(
                    camera_id=cam_id,
                    bucket_start=now - timedelta(days=7 * (i + 1)),
                    median_speed_kmh=60.0,
                    vehicle_count=15,
                ))
            db.commit()

            engine.build_baseline_matrix(db=db, force_refresh=True)
            base = engine.get_baseline_for_camera(cam_id, now)

            # With n=2 and n0=15, shrunk mean should be around (2*60 + 15*42)/17 = 44.1 km/h
            assert base["is_default"] is False
            assert base["mean_speed"] < 50.0  # Shrunk significantly from 60 towards 42
            assert base["sample_buckets"] == 2
        finally:
            db.query(CameraMetrics1M).filter(CameraMetrics1M.camera_id == cam_id).delete()
            db.commit()
            db.close()

    def test_bayesian_convergence_on_large_samples(self):
        """Large sample (n=100) on same (dow, hour) must converge close to empirical mean."""
        engine = TrafficBaselineEngine()
        db = SessionLocal()
        cam_id = "CAM_04"
        try:
            now = datetime.now(timezone.utc)
            db.query(CameraMetrics1M).filter(CameraMetrics1M.camera_id == cam_id).delete()

            # Add 100 samples across consecutive weeks at the exact same hour
            for i in range(100):
                db.add(CameraMetrics1M(
                    camera_id=cam_id,
                    bucket_start=now - timedelta(days=7 * (i + 1)),
                    median_speed_kmh=58.0 + (i % 5),
                    vehicle_count=20,
                ))
            db.commit()

            engine.build_baseline_matrix(db=db, force_refresh=True)
            base = engine.get_baseline_for_camera(cam_id, now)

            assert base["is_default"] is False
            assert abs(base["mean_speed"] - 60.0) < 3.0  # Dominates prior
            assert base["sample_buckets"] == 100
        finally:
            db.query(CameraMetrics1M).filter(CameraMetrics1M.camera_id == cam_id).delete()
            db.commit()
            db.close()


class TestContextualAnomalyDetection:
    """Tests for Shockwave crash detection and sustained speed drop gating."""

    def test_shockwave_crash_plunge_detection(self):
        """Sudden collapse from 52 km/h to 12 km/h in consecutive 1m buckets triggers CRITICAL shockwave crash alert."""
        engine = TrafficBaselineEngine()
        db = SessionLocal()
        cam_id = "CAM_08"
        try:
            now = datetime.now(timezone.utc)
            db.query(CameraMetrics1M).filter(CameraMetrics1M.camera_id == cam_id).delete()

            # Add normal speed 2 mins ago, then severe crash plunge 1 min ago
            db.add(CameraMetrics1M(
                camera_id=cam_id,
                bucket_start=now - timedelta(minutes=2),
                median_speed_kmh=52.0,
                vehicle_count=15,
            ))
            db.add(CameraMetrics1M(
                camera_id=cam_id,
                bucket_start=now - timedelta(minutes=1),
                median_speed_kmh=12.0,  # >70% drop!
                vehicle_count=12,
            ))
            db.commit()

            anomalies = engine.detect_anomalies(lookback_minutes=10, db=db)
            shockwave_anom = next((a for a in anomalies if a.camera_id == cam_id and a.anomaly_type == "SHOCKWAVE_CRASH"), None)

            assert shockwave_anom is not None
            assert shockwave_anom.severity == "CRITICAL"
            assert shockwave_anom.deviation_pct >= 50.0
            assert "shockwave" in shockwave_anom.explanation.lower()
        finally:
            db.query(CameraMetrics1M).filter(CameraMetrics1M.camera_id == cam_id).delete()
            db.commit()
            db.close()

    def test_sustained_drop_gating_vs_isolated_noise(self):
        """Single 1-minute slow vehicle must NOT trigger sustained congestion; 3 consecutive mins must trigger."""
        engine = TrafficBaselineEngine()
        db = SessionLocal()
        cam_id = "CAM_10"
        try:
            now = datetime.now(timezone.utc)
            db.query(CameraMetrics1M).filter(CameraMetrics1M.camera_id == cam_id).delete()

            # Scenario 1: Only 1 minute slow, followed by normal -> NO sustained alert
            db.add(CameraMetrics1M(camera_id=cam_id, bucket_start=now - timedelta(minutes=4), median_speed_kmh=45.0, vehicle_count=10))
            db.add(CameraMetrics1M(camera_id=cam_id, bucket_start=now - timedelta(minutes=3), median_speed_kmh=18.0, vehicle_count=10))
            db.add(CameraMetrics1M(camera_id=cam_id, bucket_start=now - timedelta(minutes=2), median_speed_kmh=44.0, vehicle_count=10))
            db.commit()

            anomalies = engine.detect_anomalies(lookback_minutes=10, db=db)
            assert not any(a.camera_id == cam_id and a.anomaly_type == "SUSTAINED_CONGESTION" for a in anomalies)

            # Scenario 2: 3 consecutive minutes slow -> Triggers sustained alert!
            db.query(CameraMetrics1M).filter(CameraMetrics1M.camera_id == cam_id).delete()
            db.commit()

            db.add(CameraMetrics1M(camera_id=cam_id, bucket_start=now - timedelta(minutes=3), median_speed_kmh=16.0, vehicle_count=10))
            db.add(CameraMetrics1M(camera_id=cam_id, bucket_start=now - timedelta(minutes=2), median_speed_kmh=15.0, vehicle_count=10))
            db.add(CameraMetrics1M(camera_id=cam_id, bucket_start=now - timedelta(minutes=1), median_speed_kmh=14.0, vehicle_count=10))
            db.commit()

            anomalies = engine.detect_anomalies(lookback_minutes=10, db=db)
            sustained_anom = next((a for a in anomalies if a.camera_id == cam_id and a.anomaly_type == "SUSTAINED_CONGESTION"), None)
            assert sustained_anom is not None
            assert sustained_anom.sustained_minutes >= 3
        finally:
            db.query(CameraMetrics1M).filter(CameraMetrics1M.camera_id == cam_id).delete()
            db.commit()
            db.close()


class TestCorridorODFlowMatrix:
    """Tests for Origin-Destination corridor travel time and transit metrics."""

    def test_corridor_matrix_structure(self):
        engine = TrafficBaselineEngine()
        db = SessionLocal()
        try:
            corridors = engine.build_corridor_flow_matrix(db=db)
            assert len(corridors) > 0
            c1 = corridors[0]
            assert c1.distance_km > 0
            assert c1.median_transit_time_sec > 0
            assert c1.transit_speed_kmh > 0
            assert c1.delay_ratio > 0
            assert c1.confidence in ("good", "moderate", "low")
        finally:
            db.close()


class TestMacroAnalyticsAPIEndpoints:
    """End-to-end FastAPI test client tests for all macro intelligence routes."""

    def test_get_macro_summary_endpoint(self):
        resp = client.get("/api/v1/analytics/macro/summary")
        assert resp.status_code == 200
        data = resp.json()
        assert "summary" in data
        assert "cameras" in data
        assert "corridors" in data
        assert "anomalies" in data
        assert data["summary"]["calibrated_cameras"] >= 30

    def test_get_macro_baseline_curve_endpoint(self):
        resp = client.get("/api/v1/analytics/macro/baseline-curve/CAM_04")
        assert resp.status_code == 200
        data = resp.json()
        assert data["camera_id"] == "CAM_04"
        assert "current_baseline_speed" in data
        assert len(data["curve"]) == 24
        assert "baseline_speed" in data["curve"][0]
        assert "upper_bound_95" in data["curve"][0]
        assert "lower_bound_95" in data["curve"][0]

    def test_get_macro_corridor_matrix_endpoint(self):
        resp = client.get("/api/v1/analytics/macro/corridor-matrix")
        assert resp.status_code == 200
        data = resp.json()
        assert "corridors" in data
        assert data["count"] > 0

    def test_recompute_macro_baseline_endpoint(self):
        resp = client.post("/api/v1/analytics/macro/recompute-baseline")
        assert resp.status_code == 200
        data = resp.json()
        assert data["success"] is True
        assert "recomputed_entries" in data
