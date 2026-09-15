"""Bulk camera onboarding: partial failure must mean partial, not total.

The regression this guards against: BulkCameraRequest.cameras was typed as
list[CameraCreateRequest], which pydantic validates for the WHOLE request
body before any route code runs — so one malformed row (a blank name) 422'd
the entire import rather than being reported as one failed row among many
successes. A department handing over a 200-camera inventory should not lose
all 200 to one typo.
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
    """These tests check REAL per-user RBAC (admin-only bulk onboarding) —
    see tests/conftest.py for why the global override needs clearing first."""
    yield


# Cached at module level, not re-fetched per call. /auth/login is rate-limited
# to 10/minute deliberately (see backend/auth/routes.py), and this module
# alone used to call it 4+ times — once per test via the fixture, again per
# test from cleanup's own uncached login — which was enough on its own to
# exhaust the budget before test_department_scoping.py's tests ever got a
# token, failing THAT file with 429s that had nothing to do with its own
# logic. One login per role, reused everywhere, is also just what a real
# client does.
_TOKEN: dict[str, str] = {}


def _admin_headers() -> dict:
    if "admin" not in _TOKEN:
        r = client.post("/api/v1/auth/login",
                        json={"username": "admin", "password": "admin123"})
        assert r.status_code == 200, r.text
        _TOKEN["admin"] = r.json()["access_token"]
    return {"Authorization": f"Bearer {_TOKEN['admin']}"}


@pytest.fixture(scope="module")
def admin_headers():
    return _admin_headers()


@pytest.fixture
def cleanup_cameras():
    """Soft-delete any cameras a test created, keyed by returned id."""
    created: list[str] = []
    yield created
    if not created:
        return
    h = _admin_headers()
    for cid in created:
        client.delete(f"/api/v1/cameras/{cid}", headers=h)


def test_json_bulk_partial_failure_does_not_sink_the_batch(
    admin_headers, cleanup_cameras,
):
    payload = {
        "cameras": [
            {"name": "PYTEST_BULK_A", "url": "https://example.test/a.m3u8",
             "department": "transport"},
            {"name": "", "department": "transport"},          # invalid
            {"name": "PYTEST_BULK_B", "url": "https://example.test/b.m3u8",
             "department": "police"},
        ],
        "negotiate_urls": False,
    }
    r = client.post("/api/v1/cameras/bulk", json=payload, headers=admin_headers)
    assert r.status_code == 200, r.text
    body = r.json()
    cleanup_cameras.extend(body["camera_ids"])

    assert body["total"] == 3
    assert body["succeeded"] == 2
    assert body["failed"] == 1
    assert [row["success"] for row in body["results"]] == [True, False, True]
    assert "name" in body["results"][1]["error"].lower() or \
           "short" in body["results"][1]["error"].lower()

    db = SessionLocal()
    names = {c.name for c in db.query(Camera)
             .filter(Camera.id.in_(body["camera_ids"])).all()}
    db.close()
    assert names == {"PYTEST_BULK_A", "PYTEST_BULK_B"}


def test_blank_name_is_rejected_not_silently_stored(
    admin_headers, cleanup_cameras,
):
    """The original bug: an empty name created a real, nameless camera."""
    payload = {"cameras": [{"name": "   ", "department": "transport"}]}
    r = client.post("/api/v1/cameras/bulk", json=payload, headers=admin_headers)
    body = r.json()
    assert body["succeeded"] == 0
    assert body["camera_ids"] == []

    db = SessionLocal()
    # is_deleted excluded deliberately: a soft-deleted row from an unrelated
    # earlier test (or a real deletion) is invisible to every normal read
    # path already and is not evidence of this endpoint accepting bad data.
    blank = db.query(Camera).filter(
        Camera.name == "", Camera.is_deleted == False).count()  # noqa: E712
    db.close()
    assert blank == 0, "a blank-named camera is live in the registry"


def test_csv_bulk_reports_parse_errors_separately_from_insert_errors(
    admin_headers, cleanup_cameras,
):
    csv_text = (
        "name,url,department\n"
        "PYTEST_CSV_A,https://example.test/c.m3u8,municipality\n"
        ",https://example.test/bad.m3u8,municipality\n"   # missing name
    )
    r = client.post("/api/v1/cameras/bulk/csv",
                    content=csv_text.encode("utf-8"),
                    headers={**admin_headers, "Content-Type": "text/csv"})
    assert r.status_code == 200, r.text
    body = r.json()
    cleanup_cameras.extend(body["camera_ids"])

    assert body["succeeded"] == 1
    assert len(body["csv_parse_errors"]) == 1
    assert body["csv_parse_errors"][0]["row"] == 1
