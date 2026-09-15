"""Gap-analysis report: scoped correctly, and honest about what it can't know.

Does not assert exact counts against the live fleet — those change as the
heartbeat loop runs and cameras are added/removed elsewhere in the suite.
What's tested is the CONTRACT: the report is scoped the same way the camera
list is, every declared gap category is present and internally consistent,
and it never claims a canonical department count it has no data for.
"""
from __future__ import annotations

import pytest
from fastapi.testclient import TestClient

from backend.main import app

client = TestClient(app)


@pytest.fixture(autouse=True)
def _real_auth(clear_get_current_user_override):
    """These tests check REAL per-user department scoping — see
    tests/conftest.py for why the global override needs clearing first."""
    yield


# Cached — see test_camera_bulk_onboarding.py for why: uncached logins across
# the new test files added up to more than the 10/minute limit and starved
# test_department_scoping.py's own logins with unrelated 429s.
_TOKENS: dict[str, str] = {}


def _login(username: str, password: str) -> dict:
    if username not in _TOKENS:
        r = client.post("/api/v1/auth/login",
                        json={"username": username, "password": password})
        assert r.status_code == 200, r.text
        _TOKENS[username] = r.json()["access_token"]
    return {"Authorization": f"Bearer {_TOKENS[username]}"}


def test_gap_analysis_shape_and_internal_consistency():
    h = _login("admin", "admin123")
    r = client.get("/api/v1/cameras/gap-analysis", headers=h)
    assert r.status_code == 200, r.text
    body = r.json()

    for key in ("missing_gps", "missing_department", "offline",
               "never_health_checked", "stale_health"):
        assert key in body["gaps"], f"missing gap category: {key}"
        assert body["gaps"][key]["count"] == len(body["gaps"][key]["cameras"])

    # Every gap count must fit inside the total — a scoped report double
    # counting or leaking another department's cameras would break this.
    for key, section in body["gaps"].items():
        assert section["count"] <= body["total_cameras"], (
            f"{key} count {section['count']} exceeds total_cameras "
            f"{body['total_cameras']}")

    # The one thing this report must NOT claim: a canonical "X of 26
    # departments" figure. There is no stored list of all 26 department
    # names, so any such claim would be invented rather than measured.
    import json
    text = json.dumps(body).lower()
    assert "26" not in text or "department" not in text.split("26")[0][-40:], (
        "gap-analysis appears to assert a canonical department count that "
        "is not backed by any stored data")


def test_gap_analysis_is_department_scoped_like_the_camera_list():
    admin_h = _login("admin", "admin123")
    op_h = _login("operator", "operator123")

    admin_body = client.get("/api/v1/cameras/gap-analysis",
                            headers=admin_h).json()
    op_body = client.get("/api/v1/cameras/gap-analysis", headers=op_h).json()

    assert op_body["scope"] == "transport"
    assert op_body["total_cameras"] <= admin_body["total_cameras"]
    for cam in op_body["gaps"]["offline"]["cameras"]:
        assert (cam["department"] or "").lower() in ("transport", ""), (
            "operator's gap report leaked a camera outside their department")
