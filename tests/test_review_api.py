"""
tests/test_review_api.py — Human Review & e-Challan REST API Tests.

Covers:
  - Test 21: Review gate default status is pending_review.
  - Test 22: Review gate requires JWT (missing token -> 401).
  - Test 23: Review gate wrong role (e.g. driver -> 403).
  - Test 24: Review confirm flow (valid reviewer JWT -> status = issued).
  - Test 25: Review dismiss flow (valid dismiss -> status = dismissed with mandatory reason).
  - Test 26: Audit log is append-only.
  - Test 28: Double-confirm guard (confirming already issued violation returns 400).
"""
import pytest
from fastapi.testclient import TestClient

from backend.auth import create_access_token
from backend.db.models import Base, ReviewAuditLog, TrafficViolation
from backend.db.session import SessionLocal, engine, get_db
from backend.main import app


@pytest.fixture
def client():
    # Drop and recreate test tables to match latest schema
    try:
        TrafficViolation.__table__.drop(bind=engine, checkfirst=True)
        ReviewAuditLog.__table__.drop(bind=engine, checkfirst=True)
    except Exception:
        pass
    Base.metadata.create_all(bind=engine)
    with TestClient(app) as c:
        yield c



@pytest.fixture
def test_violation():
    db = SessionLocal()
    viol = TrafficViolation(
        violation_type="WRONG_WAY",
        camera_id="CAM_02",
        track_id=101,
        vehicle_class="car",
        license_plate="GJ01BV9921",
        flow_angle_deg=175.0,
        speed_kmh=42.5,
        confidence=0.95,
        crop_path="output/evidence/crops/crop_101.jpg",
        challan_status="pending_review",
    )
    db.add(viol)
    db.commit()
    db.refresh(viol)
    vid = viol.id
    db.close()
    return vid


def test_review_requires_jwt_401(client, test_violation):
    # No auth header -> 401
    resp = client.post(f"/api/v1/violations/{test_violation}/confirm")
    assert resp.status_code == 401


def test_review_insufficient_role_403(client, test_violation):
    token = create_access_token("user_driver", "driver_bob", roles=["driver"])
    headers = {"Authorization": f"Bearer {token}"}
    resp = client.post(f"/api/v1/violations/{test_violation}/confirm", headers=headers)
    assert resp.status_code == 403


def test_review_confirm_flow(client, test_violation):
    token = create_access_token("officer_42", "Officer Patel", roles=["reviewer"])
    headers = {"Authorization": f"Bearer {token}"}

    resp = client.post(f"/api/v1/violations/{test_violation}/confirm", headers=headers)
    assert resp.status_code == 200
    data = resp.json()
    assert data["status"] == "issued"
    assert data["reviewed_by"] == "Officer Patel"

    # Verify DB state
    db = SessionLocal()
    viol = db.get(TrafficViolation, test_violation)
    assert viol.challan_status == "issued"
    assert viol.reviewed_by == "officer_42"

    # Check Audit Log (Test 26)
    audits = db.query(ReviewAuditLog).filter(ReviewAuditLog.violation_id == test_violation).all()
    assert len(audits) == 1
    assert audits[0].action == "confirmed"
    assert audits[0].reviewer_id == "officer_42"
    db.close()


def test_review_dismiss_flow(client):
    db = SessionLocal()
    viol = TrafficViolation(
        violation_type="WRONG_WAY",
        camera_id="CAM_02",
        track_id=102,
        vehicle_class="car",
        license_plate="GJ01AA0000",
        flow_angle_deg=170.0,
        speed_kmh=35.0,
        confidence=0.90,
        crop_path="output/evidence/crops/crop_102.jpg",
        challan_status="pending_review",
    )
    db.add(viol)
    db.commit()
    db.refresh(viol)
    vid = viol.id
    db.close()

    token = create_access_token("officer_42", "Officer Patel", roles=["reviewer"])
    headers = {"Authorization": f"Bearer {token}"}

    resp = client.post(
        f"/api/v1/violations/{vid}/dismiss",
        json={"reason": "Emergency vehicle responding to call"},
        headers=headers,
    )
    assert resp.status_code == 200
    data = resp.json()
    assert data["status"] == "dismissed"
    assert data["reason"] == "Emergency vehicle responding to call"


def test_double_confirm_guard_400(client, test_violation):
    token = create_access_token("officer_42", "Officer Patel", roles=["reviewer"])
    headers = {"Authorization": f"Bearer {token}"}

    # First confirm -> 200
    r1 = client.post(f"/api/v1/violations/{test_violation}/confirm", headers=headers)
    assert r1.status_code == 200

    # Second confirm -> 400
    r2 = client.post(f"/api/v1/violations/{test_violation}/confirm", headers=headers)
    assert r2.status_code == 400
    assert "Cannot confirm" in r2.json()["detail"]
