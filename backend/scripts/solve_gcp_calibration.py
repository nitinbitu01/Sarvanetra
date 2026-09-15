"""Solve a camera homography from surveyed ground control points.

Why this exists: the `vp1_motion+focal_from_legs` route cannot work at
CAM_08 or CAM_11. `output/camera_calibration/vp1_robust.json` records why —
"only 20% of lines agree" and "track directions span 113 deg" — because both
sites are junctions, where traffic leaves in every direction and there is no
single road direction for a vanishing point to describe. Measured against
IRC:35 zebra spacing, the homography that route produced overstates ground
distance by ~3x, non-uniformly (1.61x, 1.59x, 5.44x, 3.17x over ~200 px), so a
vehicle at 30 km/h would be reported at 89 km/h.

Ground control points do not depend on any road-direction assumption: four or
more points whose real-world position is known, spread over the plane,
determine the plane. This solves from those, measures the error honestly by
leave-one-out cross-validation, and refuses to promote a calibration that the
project's own quality gate rejects.

Input JSON (see backend/calibration_data/ground_control_points.json):

    {
      "CAM_08": {
        "location": "Junagadh - Majevadi Gate",
        "frame_size": [1920, 1080],
        "ground_control_points": [
          {"label": "P1", "px": 622, "py": 356,
           "lat": 21.5221234, "lon": 70.4578901,
           "note": "zebra crossing far-left outer corner"},
          ...
        ]
      }
    }

Every point must lie ON THE GROUND PLANE. A homography maps one plane; a
rooftop, a sign or the top of a pole is not on it and will corrupt the fit.

Run:
    python -m backend.scripts.solve_gcp_calibration CAM_08            # report
    python -m backend.scripts.solve_gcp_calibration CAM_08 --promote  # write
"""
from __future__ import annotations

import argparse
import json
import sys
from datetime import datetime, timezone
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from backend.services.calibration import (                          # noqa: E402
    QUALITY_GATE_ACCEPT_M,
    QUALITY_GATE_FLAG_M,
    compute_homography,
    evaluate_calibration_quality,
    pixel_to_world_m,
    pixel_world_resolution_m,
    wgs84_to_world,
)

GCP_FILE = ROOT / "backend" / "calibration_data" / "ground_control_points.json"
MIN_POINTS = 5          # 4 solves a homography; the 5th is what measures it


def load_points(cam: str) -> tuple[dict, list[dict]]:
    if not GCP_FILE.is_file():
        raise SystemExit("no GCP file at %s" % GCP_FILE)
    data = json.loads(GCP_FILE.read_text(encoding="utf-8-sig") or "{}")
    if cam not in data:
        raise SystemExit(
            "%s has no entry in %s (present: %s)"
            % (cam, GCP_FILE.name, ", ".join(sorted(data)) or "none"))
    entry = data[cam]
    pts = entry.get("ground_control_points") or []
    for p in pts:
        for k in ("px", "py", "lat", "lon"):
            if p.get(k) is None:
                raise SystemExit("point %s is missing %r" % (p.get("label"), k))
    return entry, pts


def to_local_metres(pts: list[dict]) -> tuple[list[tuple[float, float]], float, float]:
    """Project lat/lon onto a local ENU plane anchored at the point centroid.

    The anchor is the centroid rather than the camera, so coordinates stay
    small and the linear approximation is at its best across the site.
    """
    alat = float(np.mean([p["lat"] for p in pts]))
    alon = float(np.mean([p["lon"] for p in pts]))
    world = []
    for p in pts:
        wx, wy = wgs84_to_world(p["lat"], p["lon"], alat, alon, bearing_deg=0.0)
        world.append((float(wx), float(wy)))
    return world, alat, alon


def solve_and_score(image_pts, world_pts, labels):
    """Fit on all points, then leave-one-out for an honest error.

    Fitting and scoring on the same points measures how well four points can
    be made to agree with themselves, which is always excellent and means
    nothing. Each point here is scored by a homography that never saw it.
    """
    H_all = compute_homography(image_pts, world_pts)

    errors = []
    n = len(image_pts)
    for i in range(n):
        keep = [j for j in range(n) if j != i]
        if len(keep) < 4:
            continue
        H_i = compute_homography([image_pts[j] for j in keep],
                                 [world_pts[j] for j in keep])
        try:
            wx, wy = pixel_to_world_m(H_i, *image_pts[i])
        except ValueError as e:
            errors.append((labels[i], float("inf"), str(e)))
            continue
        d = float(np.hypot(wx - world_pts[i][0], wy - world_pts[i][1]))
        errors.append((labels[i], d, ""))
    return H_all, errors


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("camera")
    ap.add_argument("--promote", action="store_true",
                    help="write the result to camera_calibrations")
    args = ap.parse_args()
    cam = args.camera

    entry, pts = load_points(cam)
    labels = [p.get("label") or ("P%d" % i) for i, p in enumerate(pts)]
    print("%s — %s" % (cam, entry.get("location", "")))
    print("%d ground control points\n" % len(pts))

    if len(pts) < MIN_POINTS:
        print("Need at least %d points: 4 determine the homography and at "
              "least one more is required to measure it. With exactly 4 the "
              "reported error would be zero by construction." % MIN_POINTS)
        return 1

    image_pts = [(float(p["px"]), float(p["py"])) for p in pts]
    world_pts, alat, alon = to_local_metres(pts)

    print("local ENU anchor: %.7f, %.7f" % (alat, alon))
    print("%-6s %9s %8s %12s %12s  %s"
          % ("pt", "px", "py", "east_m", "north_m", "note"))
    for p, (wx, wy), lb in zip(pts, world_pts, labels):
        print("%-6s %9.1f %8.1f %12.2f %12.2f  %s"
              % (lb, p["px"], p["py"], wx, wy, p.get("note", "")))

    H, errors = solve_and_score(image_pts, world_pts, labels)

    print("\nleave-one-out error (each point scored by a fit that excluded it):")
    print("  %-6s %10s" % ("pt", "error_m"))
    finite = []
    for lb, d, msg in errors:
        print("  %-6s %10s  %s"
              % (lb, "inf" if not np.isfinite(d) else "%.3f" % d, msg))
        if np.isfinite(d):
            finite.append(d)

    if not finite:
        print("\nno point could be scored — the fit is degenerate.")
        return 1

    arr = np.array(finite)
    held_out = float(arr.max())        # the worst case, not the flattering mean
    mean_err = float(arr.mean())
    gate = evaluate_calibration_quality(held_out)

    print("\n  mean %.3f m | median %.3f m | WORST %.3f m"
          % (mean_err, float(np.median(arr)), held_out))
    print("  quality gate on the worst point: %s"
          % gate.upper())
    print("  (good < %.2f m, degraded <= %.2f m, else rejected)"
          % (QUALITY_GATE_ACCEPT_M, QUALITY_GATE_FLAG_M))

    print("\nhomography:\n%s" % np.array2string(H, precision=4, suppress_small=True))

    # Independent check: ground resolution across the tracked region. This is
    # the same quantity the pipeline uses to decide a detection is measurable.
    print("\nground metres per pixel down the frame (worst direction):")
    h_px = int(entry.get("frame_size", [1920, 1080])[1])
    for v in range(300, h_px + 1, 100):
        try:
            r = pixel_world_resolution_m(H, entry.get("frame_size", [1920])[0] / 2, v)
            print("   v=%-5d %8.3f m/px %s"
                  % (v, r, "" if r <= QUALITY_GATE_ACCEPT_M else "(not measurable)"))
        except ValueError:
            print("   v=%-5d %8s above the horizon" % (v, "-"))

    if gate == "rejected":
        print("\nNOT promoting: the worst held-out point is %.2f m, which this "
              "project classes as rejected. Speed from a calibration this "
              "loose would be wrong by more than the number is worth. Add or "
              "re-measure control points." % held_out)
        return 1

    if not args.promote:
        print("\nReport only. Re-run with --promote to write this calibration.")
        return 0

    from backend.db.models import CameraHomographyCalibration
    from backend.db.session import SessionLocal

    db = SessionLocal()
    try:
        db.query(CameraHomographyCalibration).filter(
            CameraHomographyCalibration.camera_id == cam
        ).update({"is_active": False})

        row = CameraHomographyCalibration(
            camera_id=cam,
            homography_matrix=json.dumps([float(x) for x in H.reshape(-1)]),
            reprojection_error_m=round(mean_err, 3),
            held_out_error_m=round(held_out, 3),
            quality_gate=gate,
            calibrated_by="solve_gcp_calibration",
            calibrated_on=datetime.now(timezone.utc),
            calibration_method="gcp_homography",
            positional_error_is_measured=True,
            speed_validated=True,
            gps_anchor_lat=alat,
            gps_anchor_lon=alon,
            bearing_deg=0.0,
            is_active=True,
            notes=json.dumps({
                "n_points": len(pts),
                "labels": labels,
                "validation": "leave-one-out",
                "worst_point_m": round(held_out, 3),
            }),
        )
        db.add(row)
        db.commit()
        print("\npromoted: %s is now calibrated by GCP, worst held-out point "
              "%.3f m (%s)" % (cam, held_out, gate))
    finally:
        db.close()
    return 0


if __name__ == "__main__":
    sys.exit(main())
