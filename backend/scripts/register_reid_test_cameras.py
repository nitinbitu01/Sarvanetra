"""Register the four phone clips as four real cameras.

The point is that they should be treated as cameras, not as a demonstration.
Once they are rows in `cameras` with clips on disk, the ordinary pipeline
discovers them, detects every vehicle in them, tracks it, reads what plates it
can, embeds appearance for re-identification and writes the same VehicleTrack
and JourneyEvent rows it writes for CAM_09. Nothing about them is special-cased
downstream — which is the only way the result means anything.

The coordinates are placeholders spaced roughly a hundred metres apart along one
road, because the true filming location was not recorded. They are honest as
topology (M1 then M2 then M3 then M4) and NOT as survey positions, so any
distance or corridor speed derived from them would be meaningless. The names say
so.

  python -m backend.scripts.register_reid_test_cameras
"""
from __future__ import annotations

import shutil
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from backend.db.models import Camera                               # noqa: E402
from backend.db.session import SessionLocal                        # noqa: E402

SRC = ROOT / "data" / "reid_test"
CLIPS = ROOT / "data" / "clips"

# Four points ~100 m apart. Placeholders — see the module docstring.
CAMS = [
    ("CAM_M1", "Handheld Test 1 (approx. GPS)", 21.5300, 21.5300, 70.4600),
    ("CAM_M2", "Handheld Test 2 (approx. GPS)", 21.5309, 21.5309, 70.4600),
    ("CAM_M3", "Handheld Test 3 (approx. GPS)", 21.5318, 21.5318, 70.4600),
    ("CAM_M4", "Handheld Test 4 (approx. GPS)", 21.5327, 21.5327, 70.4600),
]


def main() -> int:
    db = SessionLocal()
    made = 0
    for cam_id, name, lat, glat, lon in CAMS:
        src = sorted((SRC / cam_id).glob("*.mp4"))
        if not src:
            print(f"  {cam_id}: no video in data/reid_test/{cam_id} — skipped")
            continue

        dest_dir = CLIPS / cam_id
        dest_dir.mkdir(parents=True, exist_ok=True)
        dest = dest_dir / f"{cam_id}_test.mp4"
        if not dest.is_file():
            shutil.copy2(src[0], dest)
        print(f"  {cam_id}: clip -> {dest.relative_to(ROOT)}")

        row = db.query(Camera).filter(Camera.camera_id == cam_id).first()
        if row is None:
            # The ORM declares `id` as an autoincrement Integer, but the table
            # it maps to has it as VARCHAR(64) NOT NULL holding the same string
            # as camera_id ('CAM_01'). Letting SQLAlchemy allocate it fails with
            # "NOT NULL constraint failed: cameras.id", so it is set here to
            # match every row already in the table.
            row = Camera(id=cam_id, camera_id=cam_id)
            db.add(row)
            made += 1
        row.name = name
        row.lat, row.gps_lat = lat, glat
        row.lon, row.gps_lon = lon, lon
        row.district = "Test"
        row.zone = "Re-ID Test"
        row.department = "police"
        row.status = "ONLINE"
        row.is_online = True
        row.is_deleted = False
        row.protocol = "file"
        row.location_label = "handheld phone clip"
    db.commit()
    total = db.query(Camera).filter(Camera.is_deleted == False).count()  # noqa: E712
    db.close()
    print(f"\n  {made} camera(s) created; {total} cameras now registered.")
    print("  Run the pipeline over just these four:")
    print("    $env:SENTINEL_PIPELINE_CAMERAS = 'CAM_M1,CAM_M2,CAM_M3,CAM_M4'")
    print("    $env:SENTINEL_FORCE_CLIPS = '1'; $env:SENTINEL_STRICT_LIVE = '0'")
    print("    python -m backend.scripts.run_pipeline")
    return 0


if __name__ == "__main__":
    sys.exit(main())
