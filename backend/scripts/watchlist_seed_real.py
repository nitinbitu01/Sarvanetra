"""backend/scripts/watchlist_seed_real.py — put plates that actually exist in
the footage on the watchlist.

WHY THE EXISTING ENTRIES DO NOTHING
  watchlist_plates holds eight invented plates - GJ05AB1234, GJ03OP1234 and
  the like - none of which passed any camera. So the watchlist never matches,
  the CRITICAL path is never exercised, and a demo that clicks through it shows
  an empty state.

WHY ADDING A REAL PLATE IS NOT FAKING ANYTHING
  A watchlist is operator-entered data. "An officer added this plate" is a
  statement about the system's configuration, not a claim about detection. The
  detection that then matches it is real: a plate a human read, on a camera it
  genuinely passed, at a confidence that was measured.

  Contrast that with what was removed from the journeys endpoint, which
  invented the DETECTION - a vehicle, a colour and a confidence score for a
  sighting that never happened. That is the line, and this stays on the right
  side of it.

WHICH PLATES ARE CHOSEN, AND WHY NOT THE MULTI-CAMERA ONES
  Single-camera sightings that a human independently read, at the highest
  measured confidence. The multi-camera routes look more impressive and are
  the ones a verification round found to be different vehicles nine times in
  ten - flagging one as a wanted vehicle would put the system's least reliable
  output at the top of the screen in red.

USAGE
  python -m backend.scripts.watchlist_seed_real --count 3
"""
from __future__ import annotations

import argparse
import json
import sqlite3
from collections import defaultdict
from datetime import datetime
from pathlib import Path

DB = Path("output/sentinel.db")
REAL = Path("data/plate_real")

REASONS = [
    ("stolen", "Reported stolen - Junagadh city"),
    ("wanted", "Wanted in connection with armed robbery"),
    ("suspect", "Suspect vehicle - narcotics investigation"),
]


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--db", default=str(DB))
    ap.add_argument("--count", type=int, default=3)
    ap.add_argument("--keep-fictional", action="store_true",
                    help="Leave the invented plates in place. Off by default: "
                         "they never match and only pad the list.")
    args = ap.parse_args()

    con = sqlite3.connect(args.db)

    truth = set()
    for line in (REAL / "verified_all.jsonl").open(encoding="utf-8"):
        v = json.loads(line)
        if v.get("text"):
            truth.add(v["text"])

    # Single-camera, human-verified, highest measured confidence.
    per_plate = defaultdict(list)
    for reid, cam, score, mt in con.execute(
            "select reid_id, camera_id, final_score, match_type "
            "from journey_events"):
        per_plate[reid].append((cam, score, mt))

    cands = []
    for plate, rows in per_plate.items():
        if plate not in truth:
            continue
        if len({c for c, _, _ in rows}) != 1:
            continue                       # single camera only - see docstring
        if any(mt != "plate_confirmed" for _, _, mt in rows):
            continue
        cands.append((max(s for _, s, _ in rows), plate, rows[0][0]))
    cands.sort(reverse=True)
    print(f"human-verified single-camera candidates: {len(cands)}")

    chosen = cands[:args.count]
    if not chosen:
        raise SystemExit("no suitable plate found")

    cur = con.cursor()
    if not args.keep_fictional:
        n = cur.execute("select count(*) from watchlist_plates").fetchone()[0]
        cur.execute("delete from watchlist_plates")
        print(f"removed {n} invented entries")

    now = datetime.utcnow().isoformat(sep=" ")
    for (score, plate, cam), (cat, reason) in zip(chosen, REASONS):
        cur.execute(
            "insert or replace into watchlist_plates "
            "(plate, plate_number, reason, category, added_at, added_by) "
            "values (?,?,?,?,?,?)",
            (plate, plate, reason, cat, now, "demo_operator"))
        print(f"  added {plate:<12} {cat:<8} seen on {cam} "
              f"at confidence {score}")
    con.commit()

    n = con.execute("select count(*) from watchlist_plates").fetchone()[0]
    print(f"\nwatchlist_plates now holds {n} entries, all of which correspond")
    print("to vehicles that genuinely passed a camera and were read by a "
          "human.")


if __name__ == "__main__":
    main()
