"""Replace the placeholder positions of CAM_M1..M4 with surveyed ones.

WHY THIS MATTERS MORE THAN IT LOOKS
  The four handheld cameras were registered with invented coordinates — four
  points a notional 100 m apart near Junagadh — because the filming location
  had not been recorded. Everything downstream was then correctly refusing to
  use them: leg speed was suppressed, the network map excluded them, and the
  route drew in the wrong city.

  The operator has now supplied the real positions, and as SEGMENTS rather than
  points: for each camera, where the vehicle entered its view and where it
  left. That is better than a single fix, because the route can be drawn along
  the ground the vehicle actually covered instead of as straight lines between
  four dots.

  Note the segments join: CAM_M1's exit is CAM_M2's entry, to the digit. These
  are consecutive stretches of one road, not four unrelated points.

WHAT THIS DOES NOT UNLOCK
  Position is now surveyed; TIME still is not. The clips carry no wall-clock,
  so the timestamps on these sightings come from clip order, not from when the
  vehicle actually passed. A speed computed from a real distance and an
  invented duration is still meaningless, so leg speed stays suppressed — the
  reason changes from "position unknown" to "time unknown", and the UI says so.

  python -m backend.scripts.set_reid_camera_survey
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from backend.db.models import Camera                                # noqa: E402
from backend.db.session import SessionLocal                         # noqa: E402

OUT = ROOT / "config" / "reid_camera_survey.json"

# The stretch each camera watched, as a list of points in travel order.
#
# TWO POINTS DRAW A STRAIGHT LINE, AND A ROAD IS RARELY STRAIGHT.
#   Give as many points as the bend needs. The first is where the vehicle
#   entered the camera's view, the last is where it left, and anything between
#   is a point along the path it actually took. Every point drawn on the map
#   comes from this list, so the route is only ever as detailed as what was
#   surveyed — nothing is interpolated into a curve that nobody measured.
#
# WHY THE ROAD NETWORK CANNOT SUPPLY THE BEND HERE — measured, 2026-09-10.
#   OSRM was tried first, since it already snaps the fleet's long corridors to
#   the road. It fails at this location: /nearest snaps these points up to
#   67.5 m away onto unnamed ways, /match returns confidence 0.02, and the
#   routed "road" distance comes back SHORTER than the straight line (27 m
#   against 90 m) because both ends collapse onto one node. These are internal
#   campus roads and OpenStreetMap does not carry them in usable detail. So the
#   path has to come from the operator, not from map data.
#
#   Add points by reading them off satellite imagery along the route.
SEGMENTS = [
    ("CAM_M1", [(23.153339551285015, 72.88646240189381),
                (23.153371766731233, 72.88720312809580),
                (23.153181180523060, 72.88732362140287)]),
    ("CAM_M2", [(23.153181180523060, 72.88732362140287),
                (23.153382229813065, 72.88715002306932),
                (23.153591771000904, 72.88716254145764)]),
    ("CAM_M3", [(23.153852788579044, 72.88717051571237),
                (23.153368279037120, 72.88708933161050),
                (23.153358614419660, 72.88696159023887)]),
    # Two points only: this stretch was surveyed as a straight run. If it bends
    # on the ground, add the mid points here and the line will follow them.
    ("CAM_M4", [(23.153349816050152, 72.88674150080878),
                (23.153357148024785, 72.88713542899163)]),
]


def main() -> int:
    db = SessionLocal()
    survey = {}
    for cam_id, path in SEGMENTS:
        if len(path) < 2:
            print(f"  {cam_id}: needs at least an entry and an exit — skipped")
            continue
        a, b = path[0], path[-1]
        # The camera watches the stretch, so its map position is the middle of
        # the path it saw rather than either end of it.
        mid = path[len(path) // 2] if len(path) % 2 else (
            ((path[len(path) // 2 - 1][0] + path[len(path) // 2][0]) / 2.0,
             (path[len(path) // 2 - 1][1] + path[len(path) // 2][1]) / 2.0))
        mid_lat, mid_lon = mid[0], mid[1]
        row = db.query(Camera).filter(Camera.camera_id == cam_id).first()
        if row is None:
            print(f"  {cam_id}: not registered — run register_reid_test_cameras first")
            continue
        old = (row.lat, row.lon)
        row.lat, row.gps_lat = mid_lat, mid_lat
        row.lon, row.gps_lon = mid_lon, mid_lon
        row.location_label = "handheld camera, surveyed by operator"
        survey[cam_id] = {
            "entry": [a[0], a[1]],
            "exit": [b[0], b[1]],
            "path": [[p[0], p[1]] for p in path],
            "position": [mid_lat, mid_lon],
        }
        print(f"  {cam_id}: {old[0]:.5f},{old[1]:.5f}  ->  "
              f"{mid_lat:.6f},{mid_lon:.6f}   ({len(path)} surveyed point(s))")
    # The sightings carry their own copy of the position, written when they
    # were created. Leaving those stale would put the pins in Junagadh while
    # the camera registry says Gandhinagar — and the map reads the sighting.
    from backend.db.models import JourneyEvent
    moved = 0
    for cam_id, _ in SEGMENTS:
        s = survey.get(cam_id)
        if not s:
            continue
        for ev in (db.query(JourneyEvent)
                   .filter(JourneyEvent.camera_id == cam_id).all()):
            ev.lat, ev.lon = s["position"][0], s["position"][1]
            moved += 1
    print(f"\n  {moved} journey_event row(s) repositioned")

    db.commit()
    db.close()

    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text(json.dumps({
        "note": ("Observed entry and exit for each handheld camera, supplied by "
                 "the operator. Positions are surveyed; sighting TIMES are not "
                 "— they follow clip order, so leg speed remains suppressed."),
        "cameras": survey,
    }, indent=2), encoding="utf-8")
    print(f"\n  segments -> {OUT}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
