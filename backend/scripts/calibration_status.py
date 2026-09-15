"""backend/scripts/calibration_status.py — where every camera stands, why, and
what would move it.

WHY THIS EXISTS RATHER THAN A LIST OF FAILURES
  Six different things stop a camera calibrating, and they are not
  interchangeable. A flat "not calibrated" list invites someone to spend a week
  on the ones that cannot move. So every camera carries a reason code and, more
  usefully, a REMEDY: what would actually change its state.

  The remedies have collapsed to two, which is the useful discovery here.
  Twenty-five cameras — every blocker except `no_footage` — are waiting on the
  same site visit, because the ground-point route depends on none of the things
  the automatic routes get stuck on. The remaining three have never delivered a
  frame and are an ops ticket, not a calibration problem.

  `needs_legs` used to be its own remedy, `wait`. That was wrong: those cameras
  only need legs because the LEG method solves for a focal length, and the
  ground-point method solves for none. Four of the six have zero legs.

RE-RUN THIS RATHER THAN REMEMBERING
  `--check-upgrades` compares against the previous run and reports only what
  changed. A camera on the leg route can still cross the threshold by itself,
  so a scheduled re-run beats a diary entry — it just should not be the plan.

Run:
  python -m backend.scripts.calibration_status
  python -m backend.scripts.calibration_status --check-upgrades
"""
from __future__ import annotations

import argparse
import json
import sqlite3
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

DB = ROOT / "output" / "sentinel.db"
CALIB = ROOT / "output" / "camera_calibration" / "calibration.json"
VP1R = ROOT / "output" / "camera_calibration" / "vp1_robust.json"
VP2 = ROOT / "output" / "camera_calibration" / "vp2_from_edges.json"
LEGFIT = ROOT / "reports" / "focal_from_legs_20260906.json"
STATE = ROOT / "reports" / "calibration_status.json"

MIN_LEGS = 8

# What a blocker actually implies for whoever reads this.
REMEDY = {
    "calibrated":        ("done", "in camera_calibrations, speed is live"),
    # `survey`, not `wait`. These cameras have VP1 and a height and lack only a
    # focal length, which the leg route solves from eight cross-camera matches
    # — but the ground-point route solves no focal length at all, so the leg
    # count is irrelevant to it. Four of the six have ZERO legs against 36 in
    # the whole fleet, and Phase 2B measured that a better detector adds none,
    # so "clears on its own" was a hope with no date attached. All six sit in
    # districts already being visited. They only need the points, not a height.
    "needs_legs":        ("survey", "has VP1 and height, lacks a focal length; "
                                    "ground points bypass it — legs would also "
                                    "work but may never arrive"),
    "needs_height":      ("person", "mounting height cannot be recovered from "
                                    "this camera's footage — needs an "
                                    "installation record or a site measurement"),
    # `survey`, not `method`. These cameras defeat the AUTOMATIC routes, which
    # both rest on a vanishing point — but a ground-control-point homography
    # (cv2.findHomography over 4+ point correspondences) uses no vanishing
    # point, no orthogonality and no straight-road assumption, so neither
    # blocker below touches it. Calling this bucket `method` invited the
    # reading that it was permanent. It is not: it is waiting on a site visit,
    # the same as `person`.
    "no_vp1":            ("survey", "no usable direction of travel in the "
                                    "footage; needs ground control points"),
    "not_one_road":      ("survey", "tracks span too many directions for one "
                                    "vanishing point; needs ground control points"),
    "no_footage":        ("deployment", "zero frames ever; a site visit cannot "
                                        "help — see reports/ops_ticket_dark_"
                                        "cameras.md"),
}


def load(p: Path) -> dict:
    if not p.is_file():
        return {}
    return {r["camera"]: r for r in json.loads(p.read_text(encoding="utf-8"))}


def main() -> int:
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--check-upgrades", action="store_true")
    args = ap.parse_args()

    from backend.db.session import SessionLocal
    from backend.services.corridor_legs import by_camera, extract_legs
    from backend.services.fleet_census import census

    db = SessionLocal()
    try:
        cen = census(db)
    finally:
        db.close()

    calib, vp1r, vp2, legfit = load(CALIB), load(VP1R), load(VP2), load(LEGFIT)
    legs = {c: sum(len(v) for v in cor.values())
            for c, cor in by_camera(extract_legs(DB)).items()}

    from backend.services.fleet_census import is_test_camera

    con = sqlite3.connect(str(DB))
    promoted = {r[0] for r in con.execute(
        "SELECT camera_id FROM camera_calibrations WHERE is_active = 1").fetchall()}
    # Same exclusions as census(), applied through the same predicate. A
    # separate `is_deleted = 0` here is how the demo rig CAM_33 appeared in an
    # earlier draft of this report as a camera awaiting calibration.
    fleet = [r[0] for r in con.execute(
        "SELECT id, name FROM cameras WHERE is_deleted = 0 ORDER BY id").fetchall()
        if not is_test_camera(r[0], r[1])]
    con.close()

    rows = []
    for cam in fleet:
        n_legs = legs.get(cam, 0)
        has_h = bool((calib.get(cam, {}) or {}).get("height_m"))
        has_vp1 = bool((calib.get(cam, {}) or {}).get("vp1")
                       or (vp1r.get(cam, {}) or {}).get("vp1"))
        vp1_status = (vp1r.get(cam, {}) or {}).get("status", "")

        if cam in promoted:
            code = "calibrated"
        elif cam in cen.dark:
            code = "no_footage"
        elif not has_vp1:
            code = "no_vp1"
        elif vp1_status.startswith("not_one_road"):
            code = "not_one_road"
        elif not has_h:
            code = "needs_height"
        elif n_legs < MIN_LEGS:
            code = "needs_legs"
        else:
            code = "needs_legs"

        kind, note = REMEDY[code]
        # A leg shortfall is only meaningful for a camera whose ONLY remaining
        # gap is the focal length. Printing "needs 8 legs" beside a camera with
        # no vanishing point or no footage reads as "get legs and this fixes
        # itself", which is false in both cases.
        needed = max(0, MIN_LEGS - n_legs) if code == "needs_legs" else None
        rows.append({"camera": cam, "code": code, "remedy": kind,
                     "legs": n_legs, "legs_needed": needed,
                     "has_vp1": has_vp1, "has_height": has_h, "note": note})

    if args.check_upgrades and STATE.is_file():
        prev = {r["camera"]: r for r in json.loads(STATE.read_text(encoding="utf-8"))["cameras"]}
        moved = [r for r in rows
                 if prev.get(r["camera"], {}).get("code") != r["code"]
                 or prev.get(r["camera"], {}).get("legs", 0) != r["legs"]]
        print("changes since the last run: %d" % len(moved))
        for r in moved:
            was = prev.get(r["camera"], {})
            print("   %-9s %s (%d legs) -> %s (%d legs)"
                  % (r["camera"], was.get("code", "?"), was.get("legs", 0),
                     r["code"], r["legs"]))
        if not moved:
            print("   none — no camera crossed a threshold")

    order = ["calibrated", "needs_legs", "needs_height", "not_one_road",
             "no_vp1", "no_footage"]
    print("\n" + cen.headline())
    print("\n%-9s %-14s %-11s %6s %7s  %s"
          % ("camera", "status", "remedy", "legs", "need", "note"))
    print("-" * 104)
    for code in order:
        for r in sorted([x for x in rows if x["code"] == code],
                        key=lambda x: -x["legs"]):
            print("%-9s %-14s %-11s %6d %7s  %s"
                  % (r["camera"], r["code"], r["remedy"], r["legs"],
                     r["legs_needed"] if r["legs_needed"] else "-",
                     r["note"][:44]))

    print("\n%-14s %s" % ("SUMMARY", ""))
    for code in order:
        n = sum(1 for r in rows if r["code"] == code)
        if n:
            print("   %-14s %2d   (%s)" % (code, n, REMEDY[code][0]))

    waiting = [r for r in rows if r["code"] == "needs_legs"]
    if waiting:
        closest = min(waiting, key=lambda r: r["legs_needed"])
        zero = sum(1 for r in waiting if r["legs"] == 0)
        print("\nOn the leg route, %s is closest: %d more leg(s). But %d of "
              "these %d have\nzero legs, so treat that as a bonus if it "
              "arrives, not as the plan."
              % (closest["camera"], closest["legs_needed"], zero, len(waiting)))

    n_survey = sum(1 for r in rows if r["remedy"] in ("survey", "person"))
    n_ops = sum(1 for r in rows if r["remedy"] == "deployment")
    print("\n%d camera(s) are waiting on the same site visit — see "
          "reports/field_measurement_request.md.\nNone of them is permanently "
          "blocked. The other %d have never delivered a frame and\nneed an ops "
          "fix, not a visit — see reports/ops_ticket_dark_cameras.md."
          % (n_survey, n_ops))

    STATE.parent.mkdir(parents=True, exist_ok=True)
    STATE.write_text(json.dumps({"fleet": cen.as_dict(), "cameras": rows},
                                indent=1), encoding="utf-8")
    print(f"\nwrote {STATE}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
