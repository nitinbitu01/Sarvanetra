"""
tests/test_track_state.py — TrackState Warm-Up, Sticky Zones, & Debounce Tests.

Covers:
  - Test 6: Warm-up guard (track with < 10 frames is not ready for detection).
  - Test 7: Zone stickiness (touching adjacent boundary does not switch zone).
  - Test 8: Zone release and re-entry (exiting all zones releases lock, allows re-assignment).
"""
import pytest

from backend.services.track_state import TrackState, Zone


@pytest.fixture
def zones():
    # Zone A: x in [0, 10], y in [0, 50]
    z_a = Zone(
        lane_id="ZONE_A",
        polygon_world_m=[(0.0, 0.0), (10.0, 0.0), (10.0, 50.0), (0.0, 50.0)],
        geometry_type="straight",
        flow_vector=(0.0, -1.0),
    )
    # Zone B: x in [10, 20], y in [0, 50] (Adjacent lane)
    z_b = Zone(
        lane_id="ZONE_B",
        polygon_world_m=[(10.0, 0.0), (20.0, 0.0), (20.0, 50.0), (10.0, 50.0)],
        geometry_type="straight",
        flow_vector=(0.0, 1.0),
    )
    return [z_a, z_b]


def test_warmup_guard(zones):
    track = TrackState(track_id=1, x0=5.0, y0=25.0, now=0.0)

    for i in range(1, 10):
        track.update(wx=5.0, wy=25.0 - i, now=i * 0.1, all_zones=zones)
        assert not track.is_ready_for_detection()

    # Frame 10: now ready
    track.update(wx=5.0, wy=15.0, now=1.0, all_zones=zones)
    assert track.is_ready_for_detection()


def test_zone_stickiness(zones):
    track = TrackState(track_id=1, x0=5.0, y0=25.0, now=0.0)
    track.update(wx=5.0, wy=25.0, now=0.0, all_zones=zones)
    assert track.assigned_zone is not None
    assert track.assigned_zone.lane_id == "ZONE_A"

    # Car moves across into adjacent Zone B boundary (e.g. x=10.5)
    track.update(wx=10.5, wy=25.0, now=0.5, all_zones=zones)
    # Sticky logic preserves Zone A (does not flip to Zone B)
    assert track.assigned_zone is not None
    assert track.assigned_zone.lane_id == "ZONE_A"


def test_zone_release_and_reentry(zones):
    track = TrackState(track_id=1, x0=5.0, y0=25.0, now=0.0)
    track.update(wx=5.0, wy=25.0, now=0.0, all_zones=zones)
    assert track.assigned_zone.lane_id == "ZONE_A"

    # Car moves far outside all zones (x = 100.0)
    track.update(wx=100.0, wy=25.0, now=1.0, all_zones=zones)
    assert track.assigned_zone is None  # Released!

    # Car enters Zone B (x = 15.0)
    track.update(wx=15.0, wy=25.0, now=2.0, all_zones=zones)
    assert track.assigned_zone is not None
    assert track.assigned_zone.lane_id == "ZONE_B"
