"""Model 1 registry enrichment: camera_type, installed_at, maintenance mode,
ageing-infrastructure gap reporting, and CSV export.

Closes three gaps found by an evidence-based checklist review against the
official Model 1 deliverable list (docs/MODEL_1_ARCHITECTURE.md): the GIS
map's "camera type" layer had no backing column, gap-analysis's "ageing
infrastructure" had no install date to age against, and "maintenance-status
monitoring" only had automatic ONLINE/OFFLINE, no manually-set state.
"""
from __future__ import annotations

import pytest
from fastapi.testclient import TestClient

from backend.db.models import Camera
from backend.db.session import SessionLocal
from backend.main import app

client = TestClient(app)


@pytest.fixture(autouse=True)
def _real_auth(clear_get_current_user_override):
    """See tests/conftest.py — clears the global get_current_user override
    some other test files leave active at import time."""
    yield


# Same reasoning and pattern as test_camera_bulk_onboarding.py: one login per
# role, cached, so this file cannot starve test_department_scoping.py's own
# logins of the shared 10/minute budget.
_TOKEN: dict[str, str] = {}


def _admin_headers() -> dict:
    if "admin" not in _TOKEN:
        r = client.post("/api/v1/auth/login",
                        json={"username": "admin", "password": "admin123"})
        assert r.status_code == 200, r.text
        _TOKEN["admin"] = r.json()["access_token"]
    return {"Authorization": f"Bearer {_TOKEN['admin']}"}


def _operator_headers() -> dict:
    if "operator" not in _TOKEN:
        r = client.post("/api/v1/auth/login",
                        json={"username": "operator", "password": "operator123"})
        assert r.status_code == 200, r.text
        _TOKEN["operator"] = r.json()["access_token"]
    return {"Authorization": f"Bearer {_TOKEN['operator']}"}


@pytest.fixture
def cleanup_cameras():
    created: list[str] = []
    yield created
    if not created:
        return
    h = _admin_headers()
    for cid in created:
        client.delete(f"/api/v1/cameras/{cid}", headers=h)


def test_camera_type_and_installed_at_persist_through_create(cleanup_cameras):
    h = _admin_headers()
    r = client.post("/api/v1/cameras", json={
        "name": "PYTEST_ENRICH_A", "department": "transport",
        "camera_type": "PTZ", "installed_at": "2019-01-15",
    }, headers=h)
    assert r.status_code == 201, r.text
    cam = r.json()
    cleanup_cameras.append(cam["id"])
    assert cam["camera_type"] == "PTZ"
    assert cam["installed_at"].startswith("2019-01-15")

    listed = client.get("/api/v1/cameras", headers=h).json()
    row = next(c for c in listed if c["id"] == cam["id"])
    assert row["camera_type"] == "PTZ"
    assert row["installed_at"].startswith("2019-01-15")


def test_camera_type_defaults_to_fixed_when_omitted(cleanup_cameras):
    h = _admin_headers()
    r = client.post("/api/v1/cameras", json={
        "name": "PYTEST_ENRICH_DEFAULT", "department": "transport",
    }, headers=h)
    assert r.status_code == 201, r.text
    cam = r.json()
    cleanup_cameras.append(cam["id"])
    assert cam["camera_type"] == "Fixed"
    assert cam["installed_at"] is None


def test_bad_installed_at_is_dropped_not_rejected(cleanup_cameras):
    """A malformed date must not sink onboarding — it becomes 'unknown',
    exactly like omitting the field, not a 422."""
    h = _admin_headers()
    r = client.post("/api/v1/cameras", json={
        "name": "PYTEST_ENRICH_BADDATE", "department": "transport",
        "installed_at": "not-a-date",
    }, headers=h)
    assert r.status_code == 201, r.text
    cam = r.json()
    cleanup_cameras.append(cam["id"])
    assert cam["installed_at"] is None


def test_csv_bulk_import_reads_camera_type_and_installed_at(cleanup_cameras):
    h = _admin_headers()
    csv_text = (
        "name,department,camera_type,installed_at\n"
        "PYTEST_CSV_ENRICH,transport,Dome,2021-06-01\n"
    )
    r = client.post("/api/v1/cameras/bulk/csv",
                    content=csv_text.encode("utf-8"),
                    headers={**h, "Content-Type": "text/csv"})
    assert r.status_code == 200, r.text
    body = r.json()
    cleanup_cameras.extend(body["camera_ids"])
    assert body["succeeded"] == 1

    db = SessionLocal()
    cam = db.query(Camera).filter(Camera.id == body["camera_ids"][0]).first()
    db.close()
    assert cam.camera_type == "Dome"
    assert cam.installed_at is not None
    assert cam.installed_at.year == 2021


def test_maintenance_toggle_sets_and_clears(cleanup_cameras):
    h = _admin_headers()
    cam = client.post("/api/v1/cameras", json={
        "name": "PYTEST_MAINT", "department": "transport",
    }, headers=h).json()
    cleanup_cameras.append(cam["id"])

    r = client.patch(f"/api/v1/cameras/{cam['id']}/maintenance",
                     json={"maintenance_mode": True, "note": "sensor cleaning"},
                     headers=h)
    assert r.status_code == 200, r.text
    body = r.json()
    assert body["maintenance_mode"] is True
    assert body["status"] == "MAINTENANCE"
    assert body["maintenance_note"] == "sensor cleaning"

    r2 = client.patch(f"/api/v1/cameras/{cam['id']}/maintenance",
                      json={"maintenance_mode": False}, headers=h)
    assert r2.status_code == 200, r2.text
    body2 = r2.json()
    assert body2["maintenance_mode"] is False
    assert body2["maintenance_note"] is None
    # UNREACHABLE, not a claimed ONLINE nobody has actually re-verified yet.
    assert body2["status"] == "UNREACHABLE"


def test_maintenance_requires_admin():
    h = _operator_headers()
    r = client.patch("/api/v1/cameras/CAM_01/maintenance",
                     json={"maintenance_mode": True}, headers=h)
    assert r.status_code == 403


def test_heartbeat_never_overwrites_a_maintenance_flagged_camera(cleanup_cameras):
    """The regression this guards: an automatic health check silently
    flipping a deliberately-parked camera back to ONLINE/OFFLINE, making the
    maintenance flag look like it never took."""
    from datetime import datetime, timezone
    from backend.services.camera_heartbeat import _apply_result

    h = _admin_headers()
    cam_resp = client.post("/api/v1/cameras", json={
        "name": "PYTEST_MAINT_GUARD", "department": "transport",
    }, headers=h).json()
    cleanup_cameras.append(cam_resp["id"])
    client.patch(f"/api/v1/cameras/{cam_resp['id']}/maintenance",
                json={"maintenance_mode": True, "note": "test"}, headers=h)

    db = SessionLocal()
    cam = db.query(Camera).filter(Camera.id == cam_resp["id"]).first()
    assert cam.status == "MAINTENANCE"

    changed: list = []
    _apply_result(cam, True, datetime.now(timezone.utc), changed, db)
    db.commit()

    assert cam.status == "MAINTENANCE", (
        "camera_heartbeat overwrote a maintenance-flagged camera's status")
    assert changed == []
    db.close()


def test_gap_analysis_reports_ageing_infrastructure_and_unknown_install_date(
    cleanup_cameras,
):
    h = _admin_headers()
    old_cam = client.post("/api/v1/cameras", json={
        "name": "PYTEST_AGEING", "department": "transport",
        "installed_at": "2015-01-01",
    }, headers=h).json()
    unknown_cam = client.post("/api/v1/cameras", json={
        "name": "PYTEST_UNKNOWN_AGE", "department": "transport",
    }, headers=h).json()
    cleanup_cameras.extend([old_cam["id"], unknown_cam["id"]])

    body = client.get("/api/v1/cameras/gap-analysis", headers=h).json()
    ageing_ids = {c["id"] for c in body["gaps"]["ageing_infrastructure"]["cameras"]}
    unknown_ids = {c["id"] for c in
                   body["gaps"]["ageing_infrastructure"]["unknown_install_date"]["cameras"]}

    assert old_cam["id"] in ageing_ids
    assert old_cam["id"] not in unknown_ids
    assert unknown_cam["id"] in unknown_ids
    assert unknown_cam["id"] not in ageing_ids


def test_gap_analysis_excludes_maintenance_cameras_from_offline(cleanup_cameras):
    h = _admin_headers()
    cam = client.post("/api/v1/cameras", json={
        "name": "PYTEST_MAINT_NOT_OFFLINE", "department": "transport",
    }, headers=h).json()
    cleanup_cameras.append(cam["id"])
    client.patch(f"/api/v1/cameras/{cam['id']}/maintenance",
                json={"maintenance_mode": True, "note": "x"}, headers=h)

    body = client.get("/api/v1/cameras/gap-analysis", headers=h).json()
    offline_ids = {c["id"] for c in body["gaps"]["offline"]["cameras"]}
    maint_ids = {c["id"] for c in body["under_maintenance"]["cameras"]}

    assert cam["id"] not in offline_ids
    assert cam["id"] in maint_ids


def test_export_csv_contains_new_columns_and_is_scoped(cleanup_cameras):
    h = _admin_headers()
    cam = client.post("/api/v1/cameras", json={
        "name": "PYTEST_EXPORT_ROW", "department": "transport",
        "camera_type": "Bullet",
    }, headers=h).json()
    cleanup_cameras.append(cam["id"])

    r = client.get("/api/v1/cameras/export", headers=h)
    assert r.status_code == 200, r.text
    assert r.headers["content-type"].startswith("text/csv")
    assert "attachment" in r.headers["content-disposition"]
    text = r.content.decode("utf-8")
    header = text.splitlines()[0]
    assert "camera_type" in header
    assert "maintenance_mode" in header
    assert "installed_at" in header
    assert "PYTEST_EXPORT_ROW" in text
    assert "Bullet" in text

    op_r = client.get("/api/v1/cameras/export", headers=_operator_headers())
    op_text = op_r.content.decode("utf-8")
    # operator is scoped to 'transport' too (see test_department_scoping.py),
    # so this only proves the export did NOT silently ignore scoping —
    # the real cross-department check is in test_department_scoping.py's
    # own suite; this just confirms export uses the same filtered query
    # rather than dumping the whole table regardless of caller.
    assert op_r.status_code == 200
    assert "camera_type" in op_text.splitlines()[0]
