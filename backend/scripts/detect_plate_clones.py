"""Scan the plate reads for cloned registrations and raise alerts.

    python -m backend.scripts.detect_plate_clones                 # report
    python -m backend.scripts.detect_plate_clones --write         # raise alerts
    python -m backend.scripts.detect_plate_clones --stage-demo    # add a staged case

On this deployment's real footage the honest result is **no alerts**: only 2 of
327 plates were ever read at two different cameras, and both transitions are
physically possible. A rare-event detector that stays silent when the event did
not happen is working, not broken.

`--stage-demo` writes a scenario so the capability can be shown when no real
clone is passing. Everything it writes is marked `is_simulated=1`, carries
"STAGED DEMONSTRATION" in its own description, and is removable with
`--clear-demo`. It exists to demonstrate a detector, never to inflate a count.
"""
from __future__ import annotations

import argparse
import json
import sys
import uuid
from datetime import datetime, timedelta, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from backend.services.plate_clone_detector import (                 # noqa: E402
    MAX_ROAD_SPEED_KMH, MIN_SEPARATION_KM, CloneAlert, Sighting,
    find_clones, haversine_km, load_sightings_from_db,
)

ALERT_TYPE = "PLATE_CLONE_SUSPECTED"
DEMO_PLATE = "GJ18AH5409"          # 99 real reads on CAM_08, so the plate itself
                                   # is one this system genuinely recognises
DEMO_MARKER = "STAGED DEMONSTRATION"


def _alert_row(a: CloneAlert, cams: dict, simulated: bool):
    from backend.db.models import Alert

    cam = cams.get(a.to_camera, {})
    return Alert(
        id=str(uuid.uuid4()),
        camera_id=a.to_camera,
        alert_type=ALERT_TYPE,
        severity="critical",
        danger_score=9.5,
        plate_text=a.plate,
        subject_label="%s — one registration, two vehicles" % a.plate,
        description=(("[%s] " % DEMO_MARKER) if simulated else "") + a.describe(),
        lat=cam.get("lat"), lon=cam.get("lon"),
        district=cam.get("district") or "Gujarat",
        zone=cam.get("zone") or "Central",
        department="Traffic Police",
        status="new",
        lifecycle_status="OPEN",
        is_simulated=bool(simulated),
        timestamp=a.to_time,
        created_at=datetime.now(timezone.utc),
        score_breakdown={
            "plate": a.plate,
            "from_camera": a.from_camera, "to_camera": a.to_camera,
            "distance_km": a.distance_km,
            "gap_minutes": round(a.gap_seconds / 60.0, 2),
            "implied_kmh": a.implied_kmh,
            "ceiling_kmh": a.max_allowed_kmh,
            "excess_factor": round(a.excess_factor, 2),
            "basis": "great-circle distance; the road is longer, so the "
                     "implied speed is a lower bound",
            "staged": bool(simulated),
        },
    )


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--write", action="store_true", help="raise alerts for real findings")
    ap.add_argument("--stage-demo", action="store_true", help="add a staged, labelled case")
    ap.add_argument("--clear-demo", action="store_true", help="remove staged cases")
    ap.add_argument("--max-kmh", type=float, default=MAX_ROAD_SPEED_KMH)
    args = ap.parse_args()

    from backend.db.models import Alert, Camera
    from backend.db.session import SessionLocal

    db = SessionLocal()
    try:
        cams = {}
        for c in db.query(Camera).all():
            cams[c.camera_id] = {
                "lat": getattr(c, "lat", None) or getattr(c, "gps_lat", None),
                "lon": getattr(c, "lon", None) or getattr(c, "gps_lon", None),
                "district": getattr(c, "district", None),
                "zone": getattr(c, "zone", None),
                "name": c.name,
            }

        if args.clear_demo:
            n = db.query(Alert).filter(
                Alert.alert_type == ALERT_TYPE,
                Alert.is_simulated == True,                          # noqa: E712
            ).delete(synchronize_session=False)
            db.commit()
            print("removed %d staged alert(s)" % n)
            return 0

        sightings = load_sightings_from_db(db)
        plates = len({s.plate for s in sightings})
        print("scanning %d plate sightings across %d distinct registrations"
              % (len(sightings), plates))
        print("ceiling %.0f km/h, minimum separation %.0f km\n"
              % (args.max_kmh, MIN_SEPARATION_KM))

        found = find_clones(sightings, max_speed_kmh=args.max_kmh)
        if found:
            print("SUSPECTED CLONES: %d" % len(found))
            for a in found:
                print("   %s" % a.describe())
        else:
            print("No cloned plate found.")
            print("   Correct for this footage: a clone needs one registration")
            print("   at two distant cameras, and only 2 of %d plates were read"
                  % plates)
            print("   at more than one camera at all — both legitimately.")

        if args.write and found:
            for a in found:
                db.add(_alert_row(a, cams, simulated=False))
            db.commit()
            print("\nraised %d alert(s)" % len(found))

        if args.stage_demo:
            # A staged pair using a plate this system really does read, placed
            # at two real cameras 272 km apart, six minutes apart.
            a_cam, b_cam = "CAM_08", "CAM_14"
            pa, pb = cams.get(a_cam, {}), cams.get(b_cam, {})
            if not (pa.get("lat") and pb.get("lat")):
                print("\ncannot stage: those cameras have no GPS")
                return 1
            t1 = datetime.now(timezone.utc).replace(microsecond=0)
            t0 = t1 - timedelta(minutes=6)
            dist = haversine_km(pa["lat"], pa["lon"], pb["lat"], pb["lon"])
            staged = find_clones([
                Sighting(DEMO_PLATE, a_cam, t0, pa["lat"], pa["lon"], 0.98),
                Sighting(DEMO_PLATE, b_cam, t1, pb["lat"], pb["lon"], 0.97),
            ], max_speed_kmh=args.max_kmh)
            if not staged:
                print("\nthe staged pair is not impossible (%.1f km in 6 min) —"
                      " nothing written" % dist)
                return 1
            db.query(Alert).filter(
                Alert.alert_type == ALERT_TYPE,
                Alert.is_simulated == True,                          # noqa: E712
            ).delete(synchronize_session=False)
            for a in staged:
                db.add(_alert_row(a, cams, simulated=True))
            db.commit()
            print("\nstaged 1 alert, marked is_simulated=1 and labelled %r"
                  % DEMO_MARKER)
            print("   %s" % staged[0].describe())
            print("   remove it with --clear-demo")
    finally:
        db.close()
    return 0


if __name__ == "__main__":
    sys.exit(main())
