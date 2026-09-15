"""The hackathon technical evaluation's central task, guarded as a test.

The brief: "participants will be provided with a designated vehicle
registration number. The solution must demonstrate its capability to identify,
trace, and present the movement of the corresponding vehicle across the
integrated CCTV network as it appears at different camera locations and
times", producing "complete route traversed by the designated vehicle,
including timestamped and location-wise movement history".

The regression this guards: GET /journeys sorted its results by recency alone
(and ASCENDING at that — oldest first, contradicting its own "most recently
seen" comment). Searching a registration number therefore ranked a
single-sighting journey belonging to a DIFFERENT, fuzzily-similar plate above
the multi-camera route of the vehicle actually asked for. The answer was in
the response, just not first — which on a live evaluation is the same as
wrong.

Nothing here hardcodes a plate: the vehicle under test is discovered from
whatever journey data is present, so this keeps working when the data changes.
"""
from __future__ import annotations

import pytest
from fastapi.testclient import TestClient
from sqlalchemy import func

from backend.db.models import JourneyEvent
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


def _best_traced_vehicle() -> tuple[str, int]:
    """The plate with the most distinct cameras in the journey data, and that
    camera count. Discovered, never hardcoded."""
    db = SessionLocal()
    try:
        row = (db.query(JourneyEvent.reid_id,
                        func.count(func.distinct(JourneyEvent.camera_id)).label("cams"))
               .group_by(JourneyEvent.reid_id)
               .order_by(func.count(func.distinct(JourneyEvent.camera_id)).desc())
               .first())
        return (row.reid_id, int(row.cams)) if row else ("", 0)
    finally:
        db.close()


def test_designated_vehicle_search_returns_its_complete_route_first():
    plate, cam_count = _best_traced_vehicle()
    if not plate or cam_count < 2:
        pytest.skip("no multi-camera journey in the current dataset to trace")

    r = client.get("/api/v1/journeys", headers=_admin_headers(),
                   params={"mode": "vehicle", "search": plate})
    assert r.status_code == 200, r.text
    journeys = r.json()["journeys"]
    assert journeys, f"searching {plate} returned nothing"

    top = journeys[0]
    # The canonical identity of the top hit must BE the vehicle asked for —
    # not a fuzzily-similar plate that happens to be more recent.
    assert top["reid_id"] == plate, (
        f"top result for {plate} was {top['reid_id']} "
        f"with {top.get('stops_count')} stop(s); the complete route was "
        f"ranked below a partial match")
    # And it must be the complete route, not a fragment of it.
    assert top["stops_count"] >= cam_count, (
        f"top result carried {top['stops_count']} stops but {plate} is "
        f"present on {cam_count} distinct cameras")


def test_route_carries_timestamp_and_location_for_every_stop():
    """"timestamped and location-wise movement history" — every stop must
    actually carry both, or the route cannot be presented as the brief
    requires."""
    plate, cam_count = _best_traced_vehicle()
    if not plate or cam_count < 2:
        pytest.skip("no multi-camera journey in the current dataset to trace")

    r = client.get("/api/v1/journeys", headers=_admin_headers(),
                   params={"mode": "vehicle", "search": plate})
    top = r.json()["journeys"][0]

    assert top["stops"], "route has no stops"
    for s in top["stops"]:
        assert s.get("timestamp_utc"), f"stop on {s.get('camera_id')} has no timestamp"
        assert s.get("lat") is not None and s.get("lon") is not None, (
            f"stop on {s.get('camera_id')} has no location")
        assert s.get("camera_id"), "stop has no camera"

    # Stops must be in chronological order — a route presented out of order
    # is not a movement history.
    times = [s["timestamp_utc"] for s in top["stops"]]
    assert times == sorted(times), "route stops are not in chronological order"

    # And the map layer needs coordinates for the polyline.
    assert len(top.get("route_points") or []) >= 2, (
        "route_points has fewer than 2 coordinates — nothing to draw on the GIS map")


def test_unsearched_feed_is_newest_first():
    """Regression: the default (no search) ordering sorted ascending by
    last_time, i.e. oldest first, while its own comment said 'most recently
    seen'."""
    r = client.get("/api/v1/journeys", headers=_admin_headers(),
                   params={"mode": "vehicle", "limit": 25})
    assert r.status_code == 200, r.text
    journeys = r.json()["journeys"]
    if len(journeys) < 2:
        pytest.skip("not enough journeys to assert ordering")

    # Within the same watchlist tier, later last_time must come first.
    non_watchlist = [j for j in journeys if not j.get("watchlist_match")]
    if len(non_watchlist) < 2:
        pytest.skip("not enough non-watchlist journeys to assert ordering")
    times = [j["last_time"] for j in non_watchlist]
    assert times == sorted(times, reverse=True), (
        "journey feed is not newest-first")
