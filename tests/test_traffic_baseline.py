"""
tests/test_traffic_baseline.py — Phase 5 Unit Tests: Seasonal Baselines & Anomaly Detection.

Validates:
  1. Time-of-Day x Day-of-Week matrix aggregation.
  2. Sustained anomaly detection: transient 1-minute drop ignored, >= 3 minutes sustained drop flagged.
  3. Macro network summary generation with honest absence & calibration metadata.
"""
from datetime import datetime, timedelta
import pytest

from backend.db.models import Camera, CameraMetrics1M
from backend.db.session import SessionLocal, init_db
from backend.services.traffic_baseline import TrafficBaselineEngine, get_macro_network_summary


def test_baseline_matrix_and_sustained_anomaly_detection():
    """Verify that sustained speed drop generates an alert, while transient 1-minute dip does not."""
    init_db()
    db = SessionLocal()
    try:
        now = datetime.utcnow()
        cam_id = "CAM_TEST_ANOM"

        # 1. Populate 10 baseline metrics at 50 km/h
        for i in range(10):
            db.add(CameraMetrics1M(
                camera_id=cam_id,
                bucket_start=now - timedelta(days=7) + timedelta(minutes=i),
                median_speed_kmh=50.0,
                vehicle_count=15,
                speed_samples=15,
                health_status="ONLINE",
            ))
        db.commit()

        engine = TrafficBaselineEngine()
        engine.build_baseline_matrix(db=db)
        base = engine.get_baseline_for_camera(cam_id, now)
        assert base["empirical_mean"] == 50.0
        assert 44.0 <= base["mean_speed"] <= 50.0

        # 2. Add a transient 1-minute drop to 20 km/h (< 3 mins)
        db.add(CameraMetrics1M(
            camera_id=cam_id,
            bucket_start=now - timedelta(minutes=2),
            median_speed_kmh=20.0,
            vehicle_count=5,
            speed_samples=5,
            health_status="ONLINE",
        ))
        db.commit()

        anomalies_transient = engine.detect_anomalies(lookback_minutes=10, camera_id=cam_id, db=db)
        assert len(anomalies_transient) == 0, "Transient 1-minute drop must NOT trigger a macro alert"

        # 3. Add 2 more consecutive dropped minutes (total 3 minutes at 20 km/h)
        db.add(CameraMetrics1M(
            camera_id=cam_id,
            bucket_start=now - timedelta(minutes=1),
            median_speed_kmh=18.0,
            vehicle_count=4,
            speed_samples=4,
            health_status="ONLINE",
        ))
        db.add(CameraMetrics1M(
            camera_id=cam_id,
            bucket_start=now,
            median_speed_kmh=15.0,
            vehicle_count=3,
            speed_samples=3,
            health_status="ONLINE",
        ))
        db.commit()

        anomalies_sustained = engine.detect_anomalies(lookback_minutes=10, camera_id=cam_id, db=db)
        assert len(anomalies_sustained) == 1, "3 consecutive minutes of speed drop MUST trigger a sustained anomaly alert"
        assert anomalies_sustained[0].camera_id == cam_id
        assert anomalies_sustained[0].sustained_minutes >= 3
        assert anomalies_sustained[0].deviation_pct >= 50.0
        assert anomalies_sustained[0].severity == "CRITICAL"
    finally:
        db.query(CameraMetrics1M).filter(CameraMetrics1M.camera_id == "CAM_TEST_ANOM").delete()
        db.commit()
        db.close()


def test_macro_network_summary():
    """Verify get_macro_network_summary combines summary, cameras, corridors, and caveats."""
    init_db()
    db = SessionLocal()
    try:
        summary = get_macro_network_summary(db=db)
        assert "summary" in summary
        assert "cameras" in summary
        assert "corridors" in summary
        assert "anomalies" in summary
        assert "measurement_basis" in summary
        assert summary["summary"]["total_cameras"] >= 30
        assert summary["summary"]["calibrated_cameras"] >= 6
    finally:
        db.close()
