"""backend/scripts/field_campaign_tracker.py — where each site measurement has
got to, for a campaign run entirely at a distance.

WHY A TRACKER AND NOT A SPREADSHEET
  Nineteen cameras across nine districts, measured by people who are not in
  this conversation. Without a record of what was asked and what came back,
  "how is it going" has no answer, and the first sign of a problem is a trip
  that produced nothing.

  It also enforces the rule that matters most here: **process each trip's data
  as it lands, not the whole campaign at the end.** The plate-scale attempt is
  the argument for that — it looked right on the first camera and was 19% wrong
  on the second, and had it been applied to nineteen unknown cameras in one
  batch there would have been nothing to catch it against. Validating one
  district at a time keeps a systematic error to one district.

STATES
  requested   in the field request, nobody has taken it yet
  scheduled   someone has committed to a date
  measured    the visit happened
  received    numbers are in hand
  ingested    entered into the calibration studio and solved
  validated   passed the gate and promoted

  A camera can also be `failed` with a reason — a pole that could not be
  reached, points that could not be seen in frame — which is information, not
  an absence. `--reason` takes one of FAIL_REASONS so that failures can be
  counted; `--note` carries the specifics.

  `non_planar_road` is the reason worth naming separately. A homography maps
  plane to plane, so a crest, dip or strong camber breaks the one assumption
  the ground-point route makes. It is the only failure here that is a property
  of the SITE rather than of the visit — a second trip will not fix it, and it
  should not be retried as though it were a scheduling problem.

Run:
  python -m backend.scripts.field_campaign_tracker                 # show
  python -m backend.scripts.field_campaign_tracker --init          # seed
  python -m backend.scripts.field_campaign_tracker --set CAM_25 received
  python -m backend.scripts.field_campaign_tracker --set CAM_19 failed --reason no_access --note "pole on private land"
"""
from __future__ import annotations

import argparse
import json
import sqlite3
import sys
from datetime import datetime
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

DB = ROOT / "output" / "sentinel.db"
STATE = ROOT / "reports" / "calibration_status.json"
TRACK = ROOT / "reports" / "field_campaign.json"

STATES = ["requested", "scheduled", "measured", "received", "ingested",
          "validated", "failed"]

# Why a camera could not be measured. Split into two groups because they lead
# to opposite decisions.
FAIL_REASONS = {
    # Retryable — the site is fine, the visit was not.
    "no_access":        "pole or road could not be reached",
    "points_not_in_view": "no permanent ground feature visible in frame",
    "camera_moved":     "camera no longer points where the footage shows",
    # Terminal for THIS method. A homography maps plane to plane; a crest, dip
    # or strong camber violates that, and no number of return visits changes
    # the shape of the road. Fitting one anyway yields confident wrong
    # distances, which is worse than leaving the camera uncalibrated.
    "non_planar_road":  "crest, dip or strong camber — breaks coplanarity",
}
TERMINAL_REASONS = {"non_planar_road"}

# Buckets a site visit can clear — every blocker except `no_footage`. This
# includes `needs_legs`: those cameras are stuck only because the LEG route
# needs eight cross-camera matches to solve for a focal length, and the
# ground-point route solves no focal length at all. Four of the six have zero
# legs, so "it clears on its own" was a hope rather than a plan.
WANTED = {"needs_height", "not_one_road", "no_vp1", "needs_legs"}


def load() -> dict:
    if TRACK.is_file():
        return json.loads(TRACK.read_text(encoding="utf-8"))
    return {"cameras": {}, "history": []}


def save(d: dict) -> None:
    TRACK.parent.mkdir(parents=True, exist_ok=True)
    TRACK.write_text(json.dumps(d, indent=1), encoding="utf-8")


def init(d: dict) -> dict:
    if not STATE.is_file():
        raise SystemExit("run calibration_status first")
    rows = [r for r in json.loads(STATE.read_text(encoding="utf-8"))["cameras"]
            if r["code"] in WANTED]
    con = sqlite3.connect(str(DB))
    meta = {r[0]: r[1] for r in con.execute(
        "SELECT id, district FROM cameras").fetchall()}
    con.close()
    added = 0
    for r in rows:
        if r["camera"] in d["cameras"]:
            # Fields added after a camera was first seeded, filled in without
            # disturbing its state or history.
            cur = d["cameras"][r["camera"]]
            cur.setdefault("reason", "")
            if not cur.get("needs"):
                cur["needs"] = "points" if r["has_height"] else "height+points"
            continue
        d["cameras"][r["camera"]] = {
            "state": "requested", "blocker": r["code"],
            "district": meta.get(r["camera"]) or "Unknown",
            # What this particular camera is short of, so a visit is not spent
            # re-measuring a height already in the database.
            "needs": "points" if r["has_height"] else "height+points",
            "reason": "", "note": "",
            "updated": datetime.utcnow().isoformat(timespec="seconds"),
        }
        added += 1
    print("seeded %d camera(s); %d already tracked (fields backfilled)"
          % (added, len(rows) - added))
    return d


def main() -> int:
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--init", action="store_true")
    ap.add_argument("--set", nargs=2, metavar=("CAMERA", "STATE"))
    ap.add_argument("--reason", default=None, choices=sorted(FAIL_REASONS),
                    help="why a `failed` camera failed")
    ap.add_argument("--note", default=None)
    args = ap.parse_args()

    d = load()
    if args.init:
        d = init(d)
        save(d)

    if args.set:
        cam, st = args.set[0], args.set[1]
        if st not in STATES:
            raise SystemExit("state must be one of: %s" % ", ".join(STATES))
        if cam not in d["cameras"]:
            raise SystemExit("%s is not in the campaign (run --init?)" % cam)
        # A failure without a reason is the thing this tracker exists to
        # prevent: it looks the same as a trip nobody has done yet.
        if st == "failed" and not args.reason:
            raise SystemExit("`failed` needs --reason, one of: %s"
                             % ", ".join(sorted(FAIL_REASONS)))
        was = d["cameras"][cam]["state"]
        d["cameras"][cam]["state"] = st
        d["cameras"][cam]["updated"] = datetime.utcnow().isoformat(timespec="seconds")
        if args.reason is not None:
            d["cameras"][cam]["reason"] = args.reason
        if args.note is not None:
            d["cameras"][cam]["note"] = args.note
        d["history"].append({"camera": cam, "from": was, "to": st,
                             "at": d["cameras"][cam]["updated"],
                             "reason": args.reason or "",
                             "note": args.note or ""})
        save(d)
        print("%s: %s -> %s" % (cam, was, st))
        if args.reason in TERMINAL_REASONS:
            print("   %s is terminal for the ground-point route (%s).\n"
                  "   Do not re-send anyone; this camera needs a different "
                  "method, not a second visit." % (args.reason,
                                                   FAIL_REASONS[args.reason]))

    cams = d["cameras"]
    if not cams:
        print("campaign not started — run with --init")
        return 0

    by_district: dict = {}
    for c, v in cams.items():
        by_district.setdefault(v["district"], []).append((c, v))

    print("\n%-14s %-9s %-12s %-13s %-11s  %s"
          % ("district", "camera", "blocker", "measure", "state", "reason / note"))
    print("-" * 92)
    for dist, items in sorted(by_district.items(), key=lambda kv: -len(kv[1])):
        for c, v in sorted(items):
            tail = v.get("reason", "") or v.get("note", "")[:28]
            print("%-14s %-9s %-12s %-13s %-11s  %s"
                  % (dist, c, v["blocker"], v.get("needs", "?"), v["state"],
                     tail))

    print("\n%-12s %s" % ("STATE", "count"))
    for s in STATES:
        n = sum(1 for v in cams.values() if v["state"] == s)
        if n:
            print("   %-12s %d" % (s, n))

    # A district is worth chasing when it is entirely untouched: one journey
    # clears the most cameras.
    untouched = [(dist, [c for c, v in items if v["state"] == "requested"])
                 for dist, items in by_district.items()]
    untouched = [(dst, cs) for dst, cs in untouched if cs]
    if untouched:
        dst, cs = max(untouched, key=lambda kv: len(kv[1]))
        print("\nBest next trip: %s — %d camera(s) still at `requested` (%s)."
              % (dst, len(cs), ", ".join(sorted(cs))))

    dead = sorted(c for c, v in cams.items()
                  if v["state"] == "failed"
                  and v.get("reason") in TERMINAL_REASONS)
    if dead:
        print("\nTerminal for this method (do NOT re-schedule): %s"
              % ", ".join(dead))
        print("The road is not planar at these sites, so a homography cannot "
              "represent them\nno matter how carefully the points are "
              "measured. They need a different technique.")

    ready = [c for c, v in cams.items() if v["state"] == "received"]
    if ready:
        print("\nData in hand and not yet ingested: %s" % ", ".join(sorted(ready)))
        print("Ingest these now rather than waiting for the rest of the "
              "campaign —\na systematic error caught on one district is not "
              "a systematic error across nineteen.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
