"""Department boundaries hold at the point of ACCESS, not just in the list.

The failure this guards against is subtle and common: GET /cameras was already
department-filtered, so the system looked correct — an operator saw only their
own cameras. But every endpoint that took a camera id served whatever it was
asked for. Hiding a camera from a menu while serving it on request is not
access control, and video was the thing being served.

These tests therefore check the boundary from both sides:
  - the list shows only what the caller owns
  - naming another department's camera directly is REFUSED

Data used is the real seeded fleet: 32 cameras across 8 departments, an
operator scoped to 'transport' (2 cameras) and a viewer scoped to 'police'
(13). Admin is state-wide.
"""
from __future__ import annotations

import pytest
from fastapi.testclient import TestClient

from backend.db.models import Camera, User
from backend.db.session import SessionLocal
from backend.main import app

client = TestClient(app)

CREDENTIALS = {
    "admin": ("admin", "admin123"),
    "operator": ("operator", "operator123"),
    "viewer": ("viewer", "viewer123"),
}


# Tokens are cached for the module. /auth/login is rate-limited to 10 per
# minute per client, deliberately, and logging in once per assertion exhausted
# that budget when the whole suite ran — these tests then failed with 429 while
# passing in isolation, which looks like a scoping bug and is not one. One
# login per account is also simply what a real client does.
_TOKENS: dict[str, str] = {}


def _token(who: str) -> str:
    if who not in _TOKENS:
        username, password = CREDENTIALS[who]
        r = client.post("/api/v1/auth/login",
                        json={"username": username, "password": password})
        assert r.status_code == 200, f"login failed for {who}: {r.text}"
        _TOKENS[who] = r.json()["access_token"]
    return _TOKENS[who]


def _headers(who: str) -> dict:
    return {"Authorization": f"Bearer {_token(who)}"}


@pytest.fixture(autouse=True)
def known_departments():
    """Pin the departments these tests reason about, and restore afterwards.

    Run alone this file passed; run inside the full suite it failed, reporting
    the operator as state-wide. Another test in the suite writes to the users
    table, so depending on the seeded departments made these tests read as a
    scoping bug when the scoping was fine — the demo script shows the real
    behaviour is correct.

    A test that asserts on department boundaries should establish those
    boundaries rather than inherit them, so it does that here and puts back
    whatever it found.
    """
    # Clear any global auth override left behind by another test module.
    #
    # tests/test_live_calibration_production.py and
    # tests/test_macro_baseline_anomaly_engine.py both do
    #     app.dependency_overrides[get_current_user] = lambda: MockOfficer()
    # at MODULE level — not inside a fixture — and never remove it. Importing
    # either one therefore authenticates every later request in the whole
    # suite as their mock user, whatever token is sent. These tests exist to
    # prove real tokens map to real departments, so they cannot run under a
    # stub that returns the same user for everyone.
    #
    # Saved and restored rather than deleted outright, so those modules still
    # work if they run after this one.
    from backend.auth.dependencies import get_current_user as _gcu
    saved_override = app.dependency_overrides.pop(_gcu, None)

    db = SessionLocal()
    # ROLE matters as much as department. caller_department() grants
    # state-wide scope to role == 'admin' regardless of department, so a test
    # elsewhere that promotes a user makes the operator look unscoped here —
    # which is exactly what happened, and why role is pinned too rather than
    # merely saved and restored.
    wanted = {
        "operator": ("transport", "operator"),
        "viewer": ("police", "viewer"),
        "admin": (None, "admin"),
    }
    previous: dict[str, tuple] = {}
    for username, (dept, role) in wanted.items():
        u = db.query(User).filter(User.username == username).first()
        if u is None:
            continue
        previous[username] = (u.department, u.role)
        u.department = dept
        u.role = role
    db.commit()
    db.close()

    yield

    db = SessionLocal()
    for username, (dept, role) in previous.items():
        u = db.query(User).filter(User.username == username).first()
        if u is not None:
            u.department, u.role = dept, role
    db.commit()
    db.close()
    if saved_override is not None:
        app.dependency_overrides[_gcu] = saved_override


@pytest.fixture(scope="module")
def fleet():
    """(cameras by department, a camera each department owns)."""
    db = SessionLocal()
    cams = db.query(Camera).filter(Camera.is_deleted == False).all()  # noqa: E712
    by_dept: dict[str, list] = {}
    for c in cams:
        by_dept.setdefault((c.department or "").strip().lower(), []).append(c)
    db.close()
    return by_dept


@pytest.fixture(scope="module")
def user_departments():
    db = SessionLocal()
    out = {u.username: (u.department or "").strip().lower()
           for u in db.query(User).all()}
    db.close()
    return out


def test_seed_supports_a_cross_department_test(fleet, user_departments):
    """The tests below are meaningless unless two departments really differ."""
    assert user_departments.get("operator") == "transport"
    assert user_departments.get("viewer") == "police"
    assert fleet.get("transport"), "no transport cameras seeded"
    assert fleet.get("police"), "no police cameras seeded"
    assert set(fleet) >= {"transport", "police"}, (
        "need at least two departments owning cameras")


def test_list_is_scoped_per_department(fleet):
    """Each caller sees their own department; admin sees everything."""
    admin = client.get("/api/v1/cameras", headers=_headers("admin"))
    assert admin.status_code == 200
    admin_ids = {c["id"] for c in admin.json()}

    op = client.get("/api/v1/cameras", headers=_headers("operator"))
    assert op.status_code == 200
    op_depts = {(c["department"] or "").lower() for c in op.json()}

    # An operator sees their own department (and unassigned), never another's.
    assert op_depts <= {"transport", ""}, f"operator saw {op_depts}"
    assert len(op.json()) < len(admin_ids), (
        "operator sees as much as admin — scoping is not applied")


def test_direct_access_to_another_departments_camera_is_refused(fleet):
    """The regression that matters: the camera is hidden AND refused.

    Before this was enforced, a transport operator could request a stream
    token for a police camera by id and receive a valid one.
    """
    police_cam = fleet["police"][0]
    cam_key = police_cam.camera_id or str(police_cam.id)
    h = _headers("operator")   # scoped to transport

    # It must not appear in their list...
    listed = {c["id"] for c in client.get("/api/v1/cameras", headers=h).json()}
    assert police_cam.id not in listed

    # ...and asking for it by name must be refused, not served.
    r = client.get(f"/api/v1/analytics/live/stream-token/{cam_key}", headers=h)
    assert r.status_code == 403, (
        f"transport operator got HTTP {r.status_code} for police camera "
        f"{cam_key} — expected 403")
    assert "police" in r.text.lower(), (
        "the refusal should name the owning department so the operator knows "
        "whom to ask")


def test_own_department_camera_is_still_allowed(fleet):
    """Scoping must not break the legitimate case."""
    transport_cam = fleet["transport"][0]
    cam_key = transport_cam.camera_id or str(transport_cam.id)
    r = client.get(f"/api/v1/analytics/live/stream-token/{cam_key}",
                   headers=_headers("operator"))
    assert r.status_code == 200, (
        f"operator refused their OWN transport camera: {r.text}")
    assert r.json()["camera_id"] == cam_key


def test_admin_crosses_every_boundary(fleet):
    """State-wide scope is what admin is for."""
    for dept in ("police", "transport"):
        cam = fleet[dept][0]
        key = cam.camera_id or str(cam.id)
        r = client.get(f"/api/v1/analytics/live/stream-token/{key}",
                       headers=_headers("admin"))
        assert r.status_code == 200, f"admin refused {dept} camera {key}"


def test_scope_endpoint_explains_the_boundary():
    """An operator seeing 2 of 32 cameras must know it is policy, not outage."""
    r = client.get("/api/v1/cameras/scope", headers=_headers("operator"))
    assert r.status_code == 200
    body = r.json()
    assert body["department"] == "transport"
    assert body["state_wide_access"] is False
    assert body["cameras_visible"] < body["cameras_total"]
    assert "transport" in body["explanation"].lower()

    r = client.get("/api/v1/cameras/scope", headers=_headers("admin"))
    assert r.json()["state_wide_access"] is True


def test_video_requires_authentication_at_all():
    """These endpoints served video to anyone on the network until fixed."""
    for path in ("/api/v1/analytics/live/frame/CAM_01",
                 "/api/v1/analytics/live/stream/CAM_01"):
        assert client.get(path).status_code == 401, f"{path} is unauthenticated"
