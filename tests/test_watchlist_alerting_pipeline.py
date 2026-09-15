"""The core hackathon requirement: live feeds matched against a searchable
watchlist database, with automated alerts on a hit.

What this guards, found by an evidence-based audit against the official
challenge brief (see docs/WATCHLIST_ALERTING_ARCHITECTURE.md):

1. The live ANPR pipeline (backend/services/watchlist_service.py, called
   from backend/services/live_24x7_pipeline.py's process_vehicle_track)
   used to match plates against a standalone watchlist.json file nothing in
   the app could edit — while GET /plate-search and journey lookups already
   read a completely different table, watchlist_plates, for the same
   question. Two watchlists, only one of them reachable by an operator.
   WatchlistService now reads watchlist_plates as its source of truth.

2. No endpoint anywhere could add, edit, or retire a watchlist entry.
   New: POST/GET/PATCH/DELETE /plate-search/watchlist.

3. backend/routers/v1/plate_search.py — GET /plate-search, GET
   /plate-search/evidence, and now the watchlist endpoints above — was
   never registered in backend/main.py at all (no include_router call),
   and a pre-existing dead parameter (search_plate's unused
   `request_obj: Optional[Request] = None`) would have crashed FastAPI's
   route registration the instant someone tried to fix that. Both fixed.
"""
from __future__ import annotations

import pytest
from fastapi.testclient import TestClient

from backend.db.models import WatchlistPlate
from backend.db.session import SessionLocal
from backend.main import app

client = TestClient(app)


@pytest.fixture(autouse=True)
def _real_auth(clear_get_current_user_override):
    yield


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
def cleanup_plates():
    created: list[str] = []
    yield created
    if not created:
        return
    h = _admin_headers()
    for p in created:
        client.delete(f"/api/v1/plate-search/watchlist/{p}", headers=h)


def test_plate_search_router_is_reachable():
    """Regression for the router never being registered at all."""
    r = client.get("/api/v1/plate-search", params={"plate": "GJ05AB1234"})
    assert r.status_code != 404, "plate-search router is not mounted"
    assert r.status_code == 200, r.text
    body = r.json()
    assert "accuracy" in body and "sightings" in body


def test_add_list_deactivate_delete_watchlist_plate(cleanup_plates):
    h = _admin_headers()

    r = client.post("/api/v1/plate-search/watchlist", headers=h, json={
        "plate": "PYTEST WATCH 001", "reason": "test entry", "category": "suspect",
    })
    assert r.status_code == 201, r.text
    row = r.json()
    cleanup_plates.append(row["plate"])
    assert row["plate"] == "PYTESTWATCH001"  # non-alnum stripped
    assert row["active"] is True

    listed = client.get("/api/v1/plate-search/watchlist", headers=h).json()
    assert any(e["plate"] == "PYTESTWATCH001" for e in listed["entries"])

    r2 = client.patch(f"/api/v1/plate-search/watchlist/PYTESTWATCH001",
                      headers=h, json={"active": False})
    assert r2.status_code == 200, r2.text
    assert r2.json()["active"] is False

    active_only = client.get("/api/v1/plate-search/watchlist", headers=h).json()
    assert not any(e["plate"] == "PYTESTWATCH001" for e in active_only["entries"])
    all_entries = client.get("/api/v1/plate-search/watchlist?include_inactive=true",
                             headers=h).json()
    assert any(e["plate"] == "PYTESTWATCH001" for e in all_entries["entries"])

    r3 = client.delete("/api/v1/plate-search/watchlist/PYTESTWATCH001", headers=h)
    assert r3.status_code == 204
    cleanup_plates.clear()  # already deleted, nothing left for the fixture to do

    final = client.get("/api/v1/plate-search/watchlist?include_inactive=true",
                       headers=h).json()
    assert not any(e["plate"] == "PYTESTWATCH001" for e in final["entries"])


def test_watchlist_mutation_requires_admin():
    h = _operator_headers()
    r = client.post("/api/v1/plate-search/watchlist", headers=h,
                    json={"plate": "PYTEST_RBAC", "reason": "x"})
    assert r.status_code == 403


def test_re_adding_a_retired_plate_reactivates_rather_than_conflicts(cleanup_plates):
    h = _admin_headers()
    client.post("/api/v1/plate-search/watchlist", headers=h,
               json={"plate": "PYTESTREACT01", "reason": "first"})
    cleanup_plates.append("PYTESTREACT01")
    client.patch("/api/v1/plate-search/watchlist/PYTESTREACT01",
                headers=h, json={"active": False})

    r = client.post("/api/v1/plate-search/watchlist", headers=h,
                    json={"plate": "PYTESTREACT01", "reason": "flagged again"})
    assert r.status_code == 201, r.text
    body = r.json()
    assert body["active"] is True
    assert body["reason"] == "flagged again"

    db = SessionLocal()
    count = db.query(WatchlistPlate).filter(
        WatchlistPlate.plate == "PYTESTREACT01").count()
    db.close()
    assert count == 1, "reactivating created a duplicate row instead of updating"


def test_watchlist_service_reads_watchlist_plates_not_the_json_file(cleanup_plates):
    """The actual pipeline-facing fix: a plate added the normal way (through
    the API, i.e. watchlist_plates) must be visible to the live matcher,
    without editing watchlist.json by hand."""
    from backend.services.watchlist_service import WatchlistService

    h = _admin_headers()
    client.post("/api/v1/plate-search/watchlist", headers=h, json={
        "plate": "PYTESTLIVEMATCH", "reason": "designated vehicle", "category": "suspect",
    })
    cleanup_plates.append("PYTESTLIVEMATCH")

    svc = WatchlistService()  # fresh instance — must not depend on watchlist.json
    match = svc._fuzzy_match("PYTESTLIVEMATCH")
    assert match is not None, "newly-added DB plate was not picked up by the matcher"
    assert match[0] == "PYTESTLIVEMATCH"
    assert match[1] == "exact"


def test_load_from_db_empty_result_does_not_silently_fall_back_to_json(monkeypatch):
    """Regression: `self._load_from_db() or _load_watchlist(...)` treated a
    successful-but-empty dict the same as a failed DB lookup (both falsy in
    Python), silently substituting watchlist.json/DEFAULT_WATCHLIST — so a
    genuinely empty (or freshly-cleared) DB watchlist would resurrect
    whatever demo plates happened to be in the JSON fallback instead of
    correctly reporting no matches. Exercises the real constructor with
    _load_from_db patched to return {} (a successful, empty query — not a
    failure, which returns None), so this fails again if __init__ ever goes
    back to the `or` form."""
    import backend.services.watchlist_service as wl_mod

    monkeypatch.setattr(wl_mod.WatchlistService, "_load_from_db", lambda self: {})
    svc = wl_mod.WatchlistService()
    assert svc.watchlist == {}, (
        "an empty-but-successful DB read fell back to watchlist.json/"
        "DEFAULT_WATCHLIST instead of being trusted as genuinely empty")
