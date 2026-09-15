"""Camera snapshot: auth boundaries, and the cache actually caches.

The regression this guards specifically: the cache timestamp used to be taken
BEFORE the (7-8.5s, measured) frame grab rather than after, so every entry
was already "expired" relative to how long it took to produce — the cache
compiled, ran, and did nothing. Mocked here rather than hitting the real
corp8 account again: the thing under test is the caching/auth logic, not the
grab itself, which was already verified against one real camera.
"""
from __future__ import annotations

import pytest
from fastapi.testclient import TestClient

import backend.routers.v1.cameras as cams_mod
from backend.main import app

client = TestClient(app)


@pytest.fixture(autouse=True)
def _real_auth(clear_get_current_user_override):
    """test_snapshot_endpoint_still_scopes_by_department checks REAL per-user
    scoping — see tests/conftest.py for why the global override needs
    clearing first."""
    yield


@pytest.fixture(autouse=True)
def clean_snapshot_cache():
    cams_mod._snapshot_cache.clear()
    yield
    cams_mod._snapshot_cache.clear()


@pytest.fixture
def fake_grab(monkeypatch):
    """Replace the real (network-hitting) frame grab with a counted stub."""
    calls = {"n": 0}

    def _stub(cam):
        calls["n"] += 1
        return b"\xff\xd8\xff\xe0fake-jpeg-bytes"

    monkeypatch.setattr(cams_mod, "_grab_one_frame", _stub)
    return calls


def _admin_headers() -> dict:
    r = client.post("/api/v1/auth/login",
                    json={"username": "admin", "password": "admin123"})
    assert r.status_code == 200, r.text
    return {"Authorization": f"Bearer {r.json()['access_token']}"}


def test_second_call_within_ttl_is_a_cache_hit_and_skips_the_grab(fake_grab):
    h = _admin_headers()
    r1 = client.get("/api/v1/cameras/CAM_01/snapshot", headers=h)
    assert r1.status_code == 200
    assert r1.headers["x-snapshot-cache"] == "miss"
    assert fake_grab["n"] == 1

    r2 = client.get("/api/v1/cameras/CAM_01/snapshot", headers=h)
    assert r2.status_code == 200
    assert r2.headers["x-snapshot-cache"] == "hit"
    assert fake_grab["n"] == 1, "a cache hit must not call the grab again"
    assert r2.content == r1.content


def test_expired_entry_is_served_stale_and_refreshed_behind_the_request(
    fake_grab, monkeypatch,
):
    """Stale-while-revalidate: an expired entry is returned IMMEDIATELY and a
    refresh runs behind the response.

    This replaced a blocking re-grab. A fresh corp8 grab measures 33-50s (a
    twelve-hour VOD playlist, fetched with `-re` because reading flat out is
    what previously earned this account HTTP 429/403), so making the operator
    wait for a picture already held on disk turned every camera tile into
    "Connecting…" and read as an outage. The frame is still refreshed — just
    not in front of the viewer — and the response carries the age so no
    caller is misled about how fresh it is.
    """
    monkeypatch.setattr(cams_mod, "SNAPSHOT_CACHE_TTL_S", 0.05)
    h = _admin_headers()
    client.get("/api/v1/cameras/CAM_01/snapshot", headers=h)
    assert fake_grab["n"] == 1

    import time
    time.sleep(0.1)

    r2 = client.get("/api/v1/cameras/CAM_01/snapshot", headers=h)
    assert r2.status_code == 200
    assert r2.headers["x-snapshot-cache"] == "stale"
    # Age must be reported, so a stale frame is never passed off as current.
    assert float(r2.headers["x-snapshot-age-seconds"]) >= 0.05
    # TestClient runs background tasks before returning, so the refresh has
    # already happened by now — the point is that it was not what the caller
    # waited on.
    assert fake_grab["n"] == 2, "an expired entry must still trigger a refresh"


def test_a_camera_never_seen_before_still_blocks_for_a_real_frame(fake_grab):
    """The stale path only applies when something is cached. With no entry at
    all there is nothing honest to serve, so this must do the real grab."""
    cams_mod._snapshot_cache.pop("CAM_02", None)
    r = client.get("/api/v1/cameras/CAM_02/snapshot", headers=_admin_headers())
    assert r.status_code == 200
    assert r.headers["x-snapshot-cache"] == "miss"
    assert fake_grab["n"] == 1


def test_no_auth_at_all_is_refused(fake_grab):
    r = client.get("/api/v1/cameras/CAM_01/snapshot")
    assert r.status_code == 401
    assert fake_grab["n"] == 0, "an unauthenticated request must never reach the grab"


def test_grab_failure_is_503_not_a_500(fake_grab, monkeypatch):
    monkeypatch.setattr(cams_mod, "_grab_one_frame", lambda cam: None)
    h = _admin_headers()
    r = client.get("/api/v1/cameras/CAM_01/snapshot", headers=h)
    assert r.status_code == 503


def test_snapshot_endpoint_still_scopes_by_department(fake_grab):
    """The endpoint calls assert_camera_in_scope on the bearer-token path —
    confirm a department-scoped user is refused another department's camera
    exactly as the video/stream-token endpoints already are."""
    r = client.post("/api/v1/auth/login",
                    json={"username": "operator", "password": "operator123"})
    h = {"Authorization": f"Bearer {r.json()['access_token']}"}
    # operator is scoped to 'transport'; CAM_02 belongs to 'police' in the
    # seeded fleet (see tests/test_department_scoping.py).
    r2 = client.get("/api/v1/cameras/CAM_02/snapshot", headers=h)
    assert r2.status_code in (403, 404), (
        "a department-scoped user reached another department's camera "
        f"snapshot: {r2.status_code} {r2.text}")
