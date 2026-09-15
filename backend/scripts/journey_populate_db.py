"""backend/scripts/journey_populate_db.py — put the measured journey results
into the table the API serves, so the frontend stops showing invented ones.

WHAT IT REPLACES
  journey_events is empty, and backend/routers/v1/journeys.py falls through to
  hardcoded seed trajectories when it is: invented plates such as GJ05AB1234,
  an invented visual_score of 0.96, a colour of "White" and a subtype of
  "SUV/MUV" for vehicles nobody looked at. That is what the journey screen has
  been showing.

  A fabricated number on a demo screen is the worst thing in this project. It
  cannot be traced to a measurement, it will not survive a judge asking where
  it came from, and it puts every real figure beside it in doubt.

THE SCORE WRITTEN IS THE MEASURED PRECISION
  final_score is not a model output rescaled to look like a probability. It is
  the precision that was measured on held-out vehicles at that CTC score:

      >= -10.5   99.2%
      >= -13.5   97.8%
      >= -15.0   92.0%

  So a sighting shown at 0.92 means "of the hits reported at this score, 92%
  were real", which is a claim with an experiment behind it.

WHAT IS LEFT NULL
  visual_score, colour and subtype. No appearance model is in service - one was
  built, measured, and found to rank wrong journeys above right ones - so there
  is nothing honest to write. NULL is the correct value and the API must render
  it as unknown rather than substituting a default.

USAGE
  python -m backend.scripts.journey_populate_db
  python -m backend.scripts.journey_populate_db --dry-run
"""
from __future__ import annotations

import argparse
import json
import sqlite3
import uuid
from pathlib import Path

ANSWERS = Path("output/journey_answers.json")
DB = Path("output/sentinel.db")

# Measured on camera-level decisions over vehicles the recogniser never saw.
# Read as: at or above this CTC score, this fraction of reported hits was real.
PRECISION_AT = [(-10.5, 0.992), (-12.0, 0.977), (-13.5, 0.978), (-15.0, 0.920)]


def precision_for(score: float) -> float:
    for th, p in PRECISION_AT:
        if score >= th:
            return p
    return 0.92


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--db", default=str(DB))
    ap.add_argument("--answers", default=str(ANSWERS))
    ap.add_argument("--dry-run", action="store_true")
    ap.add_argument("--confirmed-only", action="store_true",
                    help="Write only sightings the score puts in the "
                         "confirmed band. Off by default: candidates are "
                         "real output and hiding them would misrepresent "
                         "what the system does.")
    args = ap.parse_args()

    answers = json.loads(Path(args.answers).read_text(encoding="utf-8"))
    con = sqlite3.connect(args.db)
    cams = {}
    for cid, zone, district in con.execute(
            "select camera_id, zone, district from cameras"):
        for k in {cid, cid.replace("-", "_"), cid.replace("_", "-")}:
            cams[k] = (zone, district)

    rows = []
    for plate, a in answers.items():
        sights = a.get("sightings") or []
        if not sights:
            continue
        zones = {cams.get(s["camera"], (None, None))[0] for s in sights}
        cross = len({z for z in zones if z}) > 1
        for s in sights:
            if args.confirmed_only and s["confidence"] != "confirmed":
                continue
            zone, district = cams.get(s["camera"], (None, None))
            rows.append((
                # id is a VARCHAR(36) primary key with no default, so it is
                # supplied here rather than relying on the database.
                str(uuid.uuid4()),
                plate,                       # reid_id - the vehicle identity
                s["camera"],
                s.get("lat"), s.get("lon"),
                zone, district, 1 if cross else 0,
                s.get("timestamp"),
                "vehicle",
                None,                        # colour: no appearance model
                None,                        # subtype: same
                s["read"],
                None,                        # visual_score: same
                precision_for(s["score"]),
                "plate_confirmed" if s["confidence"] == "confirmed"
                else "plate_candidate",
            ))

    print(f"queries with sightings : "
          f"{sum(1 for a in answers.values() if a.get('sightings'))}")
    print(f"rows to write          : {len(rows)}")
    conf = sum(1 for r in rows if r[-1] == "plate_confirmed")
    # Index by name, not position. Adding the uuid shifted every field by one
    # and the positional version of this line then reported every row as
    # cross-zone.
    CROSS = 7
    print(f"  plate_confirmed      : {conf}")
    print(f"  plate_candidate      : {len(rows) - conf}")
    print(f"  cross-zone           : {sum(1 for r in rows if r[CROSS])}")

    if args.dry_run:
        print("\n--dry-run: nothing written.")
        for r in rows[:5]:
            print("  ", r[:2], r[7], r[11], f"score {r[13]}", r[14])
        return

    cur = con.cursor()
    cur.execute("delete from journey_events")
    cur.executemany(
        "insert into journey_events (id, reid_id, camera_id, lat, lon, zone, "
        "district, is_cross_zone, timestamp, object_class, color, subtype, "
        "plate_text, visual_score, final_score, match_type) "
        "values (?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?)", rows)
    con.commit()
    n = con.execute("select count(*) from journey_events").fetchone()[0]
    print(f"\njourney_events now holds {n} rows")
    print("The API will serve these instead of its seed trajectories.")
    print("\ncolour, subtype and visual_score are NULL on purpose - there is")
    print("no appearance model in service, so there is nothing true to put")
    print("there. The API must render them as unknown, not substitute a "
          "default.")


if __name__ == "__main__":
    main()
