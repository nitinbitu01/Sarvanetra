"""backend/scripts/promote_calibrations.py — write a validated calibration into
the table `speed_estimator` trusts. Requires explicit sign-off.

WHAT THIS FIELD MEANS DOWNSTREAM
  `camera_calibrations.homography_matrix` is read by `speed_estimator` and
  becomes km/h on the dashboard, congestion index, density and wrong-way
  detection. A matrix written here without evidence is not a missing feature,
  it is a confident wrong number — which this project has already had to
  withdraw twice for exactly that reason.

  So promotion is a separate script from every fitter, defaults to a dry run,
  and takes a backup first.

THE HOMOGRAPHY
  Built from the planar-VP mapping the fitters produce:

      X = h (u - vpx) / (v - vpy)        lateral metres
      Y = f h / (v - vpy)                metres along the road

  In homogeneous form, world = H . [u, v, 1] with

      H = [[  h,   0, -h*vpx ],
           [  0,   0,  f*h   ],
           [  0,   1, -vpy   ]]

  which maps a pixel to (X, Y) after the perspective divide.

ON `reprojection_error_m`
  The column is NOT NULL and means positional error in metres. There are no
  ground control points on this fleet, so a reprojection error against surveyed
  points does not exist and inventing one would be the exact failure this gate
  is here to prevent.

  What IS measured is speed: the focal length carries a bootstrap confidence
  interval, and a relative error in f becomes the same relative error in
  distance along the road. So the value written is that relative width applied
  at the camera's own median observed range — an INFERRED positional
  uncertainty, recorded as such in `notes`, and typically metres rather than
  the sub-metre a survey would give. `quality_gate` is set accordingly and
  never to "good".

Run:
  python -m backend.scripts.promote_calibrations                 # dry run
  python -m backend.scripts.promote_calibrations --commit CAM_08 CAM_11
"""
from __future__ import annotations

import argparse
import json
import math
import shutil
import sqlite3
import sys
from datetime import datetime
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

DB = ROOT / "output" / "sentinel.db"
LEGFIT = ROOT / "reports" / "focal_from_legs_20260906.json"
VP2FIT = ROOT / "output" / "camera_calibration" / "vp2_from_edges.json"
CALIB = ROOT / "output" / "camera_calibration" / "calibration.json"
GEOM = ROOT / "output" / "road_geometry"


def homography(vpx: float, vpy: float, h: float, f: float) -> list:
    return [h, 0.0, -h * vpx,
            0.0, 0.0, f * h,
            0.0, 1.0, -vpy]


def median_range_m(cam: str, vpy: float, h: float, f: float) -> float | None:
    """Median along-road distance of this camera's own tracks, in metres."""
    p = GEOM / f"{cam}.json"
    if not p.is_file():
        return None
    ys = []
    for pts in json.loads(p.read_text(encoding="utf-8"))["tracks"].values():
        for q in pts:
            dv = q["v"] - vpy
            if dv > 1.0:
                ys.append(f * h / dv)
    if not ys:
        return None
    ys.sort()
    return ys[len(ys) // 2]


def build(cam: str, legfit: dict, vp2: dict, prior: dict) -> dict | None:
    if legfit.get("status") != "solved":
        return None
    f = float(legfit["focal_px"])
    vp1 = prior["vp1"]
    h = float(prior["height_m"])
    lo, hi = legfit["focal_ci95"]
    rel = (hi - lo) / f                       # relative width of the interval

    rng = median_range_m(cam, vp1[1], h, f)
    err_m = round(rel * rng, 2) if rng else None

    notes = {
        "method": "vp1_motion + focal_from_cross_camera_legs",
        "focal_px": f,
        "focal_ci95": [lo, hi],
        "focal_ci_rel_width": round(rel, 3),
        "height_m": h,
        "height_iqr_m": prior.get("height_iqr_m"),
        "legs_used": legfit.get("legs_total"),
        "legs_fit": legfit.get("legs_fit"),
        "legs_holdout": legfit.get("legs_holdout"),
        "holdout_speed_ratio": legfit.get("holdout_ratio"),
        "corridors": legfit.get("corridors"),
        "median_range_m": round(rng, 1) if rng else None,
        "error_basis": ("positional error INFERRED from the focal confidence "
                        "interval at this camera's median range; there are no "
                        "ground control points on this fleet, so it is NOT a "
                        "reprojection error against surveyed points"),
        "speed_validated": ("held-out cross-camera legs, ratio %s"
                            % legfit.get("holdout_ratio")),
    }
    if vp2 and vp2.get("focal_px"):
        notes["independent_geometric_focal_px"] = vp2["focal_px"]
        notes["leg_vs_geometric_ratio"] = round(f / vp2["focal_px"], 3)

    return {
        "camera_id": cam,
        "homography_matrix": json.dumps(homography(vp1[0], vp1[1], h, f)),
        "reprojection_error_m": err_m if err_m is not None else 999.0,
        "held_out_error_m": None,
        # Never "good": that gate means sub-metre against measured points.
        "quality_gate": "degraded",
        "calibrated_by": "fit_focal_from_legs",
        "notes": json.dumps(notes),
    }


def main() -> int:
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("cameras", nargs="*")
    ap.add_argument("--commit", action="store_true",
                    help="actually write. Without it this is a dry run.")
    args = ap.parse_args()

    legfits = {r["camera"]: r for r in json.loads(LEGFIT.read_text(encoding="utf-8"))}
    vp2 = {r["camera"]: r for r in json.loads(VP2FIT.read_text(encoding="utf-8"))} \
        if VP2FIT.is_file() else {}
    priors = {r["camera"]: r for r in json.loads(CALIB.read_text(encoding="utf-8"))}

    cams = args.cameras or [c for c, r in legfits.items()
                            if r.get("status") == "solved"]
    rows = []
    for cam in cams:
        row = build(cam, legfits.get(cam, {}), vp2.get(cam), priors.get(cam, {}))
        if row:
            rows.append(row)
        else:
            print(f"  {cam}: not solved — skipped")

    if not rows:
        print("nothing to promote")
        return 0

    print("\n%-9s %14s %12s %10s  %s"
          % ("camera", "reproj err m", "quality", "holdout", "notes"))
    print("-" * 92)
    for r in rows:
        n = json.loads(r["notes"])
        print("%-9s %14s %12s %10s  f=%s CI±%.0f%% legs=%s range=%sm"
              % (r["camera_id"], r["reprojection_error_m"], r["quality_gate"],
                 n["holdout_speed_ratio"], n["focal_px"],
                 n["focal_ci_rel_width"] * 100, n["legs_used"],
                 n["median_range_m"]))

    if not args.commit:
        print("\nDRY RUN — nothing written. Re-run with --commit to promote.")
        return 0

    bak = DB.with_suffix(".db.bak-precalib-%s"
                         % datetime.now().strftime("%Y%m%d_%H%M%S"))
    shutil.copy2(DB, bak)
    print(f"\nbacked up -> {bak.name}")

    con = sqlite3.connect(str(DB))
    try:
        cur = con.cursor()
        before = cur.execute("SELECT COUNT(1) FROM camera_calibrations").fetchone()[0]
        for r in rows:
            cur.execute("""
                INSERT INTO camera_calibrations
                  (camera_id, homography_matrix, reprojection_error_m,
                   held_out_error_m, quality_gate, calibrated_by,
                   calibrated_on, is_active, created_at, camera_angle_deg,
                   notes)
                VALUES (?,?,?,?,?,?,?,?,?,?,?)""",
                        (r["camera_id"], r["homography_matrix"],
                         r["reprojection_error_m"], r["held_out_error_m"],
                         r["quality_gate"], r["calibrated_by"],
                         datetime.utcnow(), 1, datetime.utcnow(), 0.0,
                         r["notes"]))
        con.commit()
        after = cur.execute("SELECT COUNT(1) FROM camera_calibrations").fetchone()[0]
        print(f"camera_calibrations: {before} -> {after} rows")
    finally:
        con.close()

    print("\nspeed_estimator will now emit km/h for these cameras. Every other")
    print("camera continues to return NULL, which remains correct.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
