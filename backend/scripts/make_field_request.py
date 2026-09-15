"""backend/scripts/make_field_request.py — the one document to hand to whoever
can stand in front of these cameras.

WHY THIS ASKS FOR MORE THAN A HEIGHT
  The seven cameras in the `person` bucket are blocked on a mounting height,
  and it is tempting to ask only for that. It would be asking for too little.

  A height alone unblocks one step. The camera would still need a focal length,
  and the two automatic routes to one have both been exhausted on this fleet:
  the geometric route (VP2 from vehicle edges) succeeded on 1 of 16 cameras,
  and the leg route needs eight cross-camera plate matches that most of these
  cameras do not have and will not soon get.

  Four to six GROUND POINTS with the distances between them unlock something
  strictly better. They feed `LiveCalibrationStudio`, which is already built and
  already validated, needs no vanishing point, no second traffic stream and no
  legs — and produces sub-metre POSITIONAL accuracy rather than the speed-only,
  metre-scale calibration the leg route gives.

  Measuring a few distances on the road is barely more work than measuring a
  pole height, and it is the difference between unblocking one stage and
  finishing the camera. Someone standing there anyway should be asked for both.

WHY THE `needs_legs` CAMERAS ARE HERE TOO
  They were left out of the first version of this request, on the reasoning
  that they clear by themselves once eight cross-camera plate matches arrive.
  That reasoning does not survive contact with the numbers: four of the six
  have ZERO legs, and the whole fleet has produced 36 in total. Waiting is a
  hope, not a schedule — and Phase 2B measured that improving the detector adds
  no legs at all.

  The decisive point is that the ground-point route does not need a focal
  length, so the leg count is irrelevant to it. These cameras were blocked by a
  limitation of the LEG method, never by the site. All six sit in districts
  already being visited, every one within 7.4 km of a camera already on the
  list — so they cost travel time, not a trip.

  Their ask is smaller: height and VP1 are already on record, so ground points
  alone finish them. The per-camera column below says which is which, because
  sending someone up a pole for a number already in the database is how a
  campaign loses goodwill.

WHAT MAKES A GOOD GROUND POINT
  Permanent, flat on the road surface, and identifiable in the camera image: a
  lane-marking corner, a kerb corner, the end of a zebra stripe, a manhole rim.
  Spread them across the area the camera watches rather than clustering them —
  four points in one corner constrain the solve far less than four spread out.

Run:  python -m backend.scripts.make_field_request
"""
from __future__ import annotations

import json
import sqlite3
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

DB = ROOT / "output" / "sentinel.db"
STATE = ROOT / "reports" / "calibration_status.json"
OUT = ROOT / "reports" / "field_measurement_request.md"

# Buckets whose blocker a site visit can actually clear — which is all of them
# except `no_footage`, since the ground-point route depends on none of the
# things the automatic routes get stuck on.
WANTED = {"needs_height", "not_one_road", "no_vp1", "needs_legs"}


def main() -> int:
    if not STATE.is_file():
        print("run calibration_status first")
        return 1
    state = json.loads(STATE.read_text(encoding="utf-8"))
    rows = [r for r in state["cameras"] if r["code"] in WANTED]

    con = sqlite3.connect(str(DB))
    meta = {r[0]: r for r in con.execute(
        "SELECT id, name, district, zone, lat, lon FROM cameras").fetchall()}
    con.close()

    # What to actually ask for, per camera. Not every camera needs both: a
    # height already in the database is not worth a pole climb, and saying so
    # explicitly is what stops the visit being spent on it.
    for r in rows:
        r["height_alone_suffices"] = bool(r["has_vp1"] and r["legs"] >= 8)
        if r["height_alone_suffices"]:
            r["ask"] = "**height only** — nothing else is missing"
        elif not r["has_height"]:
            r["ask"] = "height **and** ground points"
        else:
            r["ask"] = "**ground points only** — height already on record"

    lines = [
        "# Site measurement request — camera calibration",
        "",
        "**%d cameras.** Two measurements, both doable in one visit of about "
        "fifteen minutes. **The per-camera table below says which of the two "
        "each camera actually needs** — several already have a height on "
        "record and need only the ground points." % len(rows),
        "",
        "## What to measure, per camera",
        "",
        "### 1. Mounting height  *(where the table asks for it)*",
        "",
        "Vertical distance from the **road surface directly below the camera** "
        "to the **centre of the lens**, in metres to the nearest 0.05 m.",
        "",
        "A laser rangefinder pointed down from the mount is ideal. A tape from "
        "the pole base works if the camera is on a straight pole — note it if "
        "the mount is on an arm, because then the lens is not above the pole "
        "base and the horizontal offset matters too.",
        "",
        "### 2. Four to six ground reference points  *(this is the valuable part)*",
        "",
        "Pick features on the **road surface** that are permanent and clearly "
        "visible in the camera's view — a lane-marking corner, a kerb corner, "
        "the end of a zebra stripe, a manhole rim.",
        "",
        "Then measure the **distance between each pair** (or from one chosen "
        "origin point to each of the others), in metres to the nearest 0.1 m.",
        "",
        "Please **spread the points across the whole area the camera watches**, "
        "near and far, left and right. Four points bunched in one corner are "
        "worth much less than four spread out.",
        "",
        "A photo of the scene with the chosen points marked, alongside the "
        "numbers, removes all ambiguity about which feature is which.",
        "",
        "### 3. Is the road flat?  *(one line, and it matters)*",
        "",
        "The maths behind this maps one **flat plane** to another. A bend in "
        "the road is fine - the surface is still flat. What breaks it is the "
        "road going up or down within the camera's view: a **crest**, a "
        "**dip**, or a strongly **cambered** (banked) surface.",
        "",
        "So please add one line per camera: `road: flat` or `road: crest` / "
        "`dip` / `camber`.",
        "",
        "**If it is not flat, take 7-8 points instead of 4-6**, and keep them "
        "on the flattest stretch you can that still spans the view. The extra "
        "points let us detect the curvature and fit around it rather than "
        "silently absorbing it as error - a sloped road fitted as a flat one "
        "gives confident, wrong distances, which is worse than no calibration "
        "at all.",
        "",
        "This cannot be judged reliably from satellite imagery - a crest and a "
        "camber both look flat from directly overhead. Standing there is the "
        "only good way to answer it, which is why it is on this list.",
        "",
        "### Why both",
        "",
        "The height alone unblocks one stage of processing. The ground points "
        "complete the calibration outright, and give position accuracy an "
        "order of magnitude better. The extra effort is a few tape measurements.",
        "",
        "---",
        "",
        "## Cameras",
        "",
        "| Camera | Location | District / Zone | GPS | Blocker | What to measure |",
        "|---|---|---|---|---|---|",
    ]

    for r in sorted(rows, key=lambda x: (not x["height_alone_suffices"],
                                         x["has_height"], x["camera"])):
        m = meta.get(r["camera"], (r["camera"], "?", "?", "?", None, None))
        gps = ("%.4f, %.4f" % (m[4], m[5])) if m[4] and m[5] else "—"
        lines.append("| **%s** | %s | %s / %s | %s | `%s` | %s |"
                     % (r["camera"], m[1] or "—", m[2] or "—", m[3] or "—",
                        gps, r["code"], r["ask"]))

    n_easy = sum(1 for r in rows if r["height_alone_suffices"])
    n_points_only = sum(1 for r in rows
                        if r["has_height"] and not r["height_alone_suffices"])
    lines += [""]
    if n_easy:
        lines.append(
            "For the %d camera(s) needing a **height only**, that number is "
            "genuinely all that is missing." % n_easy)
    if n_points_only:
        lines.append(
            "For the %d camera(s) needing **ground points only**, the height "
            "is already in our records - please do not spend the visit "
            "re-measuring it." % n_points_only)
    lines += [
        "Where both are asked for, the ground points are the part that "
        "finishes the camera. If time runs short, skip a whole camera rather "
        "than doing half of one - half a camera cannot be calibrated, and we "
        "would not know the visit had happened.",
        "",
    ]

    # Grouped by district, because these cluster and a visit is a journey.
    by_district: dict = {}
    for r in rows:
        m = meta.get(r["camera"])
        by_district.setdefault((m[2] if m else None) or "Unknown", []).append(r["camera"])
    lines += [
        "## Suggested trips",
        "",
        "These cluster geographically, so the list is a handful of journeys "
        "rather than %d separate ones:" % len(rows),
        "",
        "| District | Cameras | Count |",
        "|---|---|---:|",
    ]
    for dist, cams in sorted(by_district.items(), key=lambda kv: -len(kv[1])):
        lines.append("| %s | %s | %d |" % (dist, ", ".join(sorted(cams)),
                                           len(cams)))
    lines += [
        "",
        "The largest cluster is worth doing first: it is the most cameras "
        "finished per journey.",
        "",
        "## Sending the results back",
        "",
        "Plain text or a spreadsheet is fine. Per camera:",
        "",
        "```",
        "CAM_xx",
        "  mounting_height_m: 6.20    # omit if the table says ground points only",
        "  arm_offset_m: 0.0          # 0 if the lens is directly above the pole base",
        "  road: flat                 # or crest / dip / camber -> then 7-8 points",
        "  points:",
        "    A -> B: 3.50            # e.g. lane width",
        "    A -> C: 12.00",
        "    B -> D: 9.20",
        "  photo: CAM_xx_points.jpg   # scene with A, B, C, D marked",
        "```",
        "",
        "Points are entered into the calibration studio against the camera's "
        "own frame, so their identity in the photo is what matters — not any "
        "particular naming.",
    ]

    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text("\n".join(lines) + "\n", encoding="utf-8")

    print("cameras in the request : %d" % len(rows))
    print("  height only          : %d" % n_easy)
    print("  ground points only   : %d" % n_points_only)
    print("  height + points      : %d" % (len(rows) - n_easy - n_points_only))
    by_code = {}
    for r in rows:
        by_code[r["code"]] = by_code.get(r["code"], 0) + 1
    for k, v in sorted(by_code.items()):
        print("    %-14s %d" % (k, v))
    print(f"\nwrote {OUT}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
