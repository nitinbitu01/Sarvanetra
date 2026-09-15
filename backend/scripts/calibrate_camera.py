"""
backend/scripts/calibrate_camera.py — One-time per-camera calibration (Day 8, §2).

PURPOSE:
  Operator identifies two points in a reference frame that are a KNOWN
  real-world distance apart (e.g. two marks 5 meters apart on the ground, a
  parking-space width, a lane-marking gap). The script computes
  px_per_meter = pixel_distance(point1, point2) / real_distance_meters and
  writes/upserts a row in camera_calibration.

  Flat px-per-meter is an approximation valid for a roughly top-down or
  moderate-angle camera view — see loitering_detector.py / camera_calibration.py
  module docstrings for why this is acceptable for this pass and what a
  wide-angle/oblique camera needs instead (proper homography, not built here).

MODES:
  1. Interactive (default, if a display is available and --frame-image is
     given): opens the reference frame in an OpenCV window, operator clicks
     two points with the mouse, distance typed at the prompt.
  2. Non-interactive (--point1, --point2 given directly): for scripting,
     headless environments, or when pixel coordinates were already read off
     the frame some other way. This is also what the eval harness / CI uses.

HOW TO RUN:
  Interactive:
    python -m backend.scripts.calibrate_camera --camera-id CAM-01 \\
        --frame-image path/to/reference_frame.jpg --real-distance-meters 5.0

  Non-interactive:
    python -m backend.scripts.calibrate_camera --camera-id CAM-01 \\
        --point1 120,400 --point2 540,410 --real-distance-meters 5.0 \\
        --notes "5m mark on north crosswalk"

  List existing calibrations:
    python -m backend.scripts.calibrate_camera --list
"""
from __future__ import annotations

import argparse
import math
import sys
from datetime import datetime
from pathlib import Path

_PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))


def _parse_point(raw: str) -> tuple[float, float]:
    try:
        x_str, y_str = raw.split(",")
        return float(x_str), float(y_str)
    except (ValueError, AttributeError):
        raise argparse.ArgumentTypeError(f"Expected 'x,y' (e.g. '120,400'), got: {raw!r}")


def _interactive_pick_points(frame_image_path: str) -> tuple[tuple[float, float], tuple[float, float]]:
    """Open the frame in an OpenCV window; operator clicks two points."""
    import cv2

    img = cv2.imread(frame_image_path)
    if img is None:
        print(f"[ERROR] Could not read image: {frame_image_path}", file=sys.stderr)
        sys.exit(1)

    points: list[tuple[int, int]] = []
    window = "Click two points a known distance apart, then press any key"

    def _on_click(event, x, y, flags, param):  # noqa: ARG001
        if event == cv2.EVENT_LBUTTONDOWN and len(points) < 2:
            points.append((x, y))
            cv2.circle(img, (x, y), 5, (0, 0, 255), -1)
            if len(points) == 2:
                cv2.line(img, points[0], points[1], (0, 255, 0), 2)
            cv2.imshow(window, img)

    cv2.imshow(window, img)
    cv2.setMouseCallback(window, _on_click)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

    if len(points) != 2:
        print(f"[ERROR] Expected 2 clicks, got {len(points)}. Re-run and click exactly two points.",
              file=sys.stderr)
        sys.exit(1)
    return (float(points[0][0]), float(points[0][1])), (float(points[1][0]), float(points[1][1]))


def _upsert_calibration(
    camera_id: str, px_per_meter: float, calibration_method: str, notes: str | None,
) -> None:
    from backend.db.models import CameraCalibration
    from backend.db.session import SessionLocal

    db = SessionLocal()
    try:
        row = db.query(CameraCalibration).filter(CameraCalibration.camera_id == camera_id).first()
        if row:
            row.px_per_meter = px_per_meter
            row.calibration_method = calibration_method
            row.calibrated_at = datetime.utcnow()
            row.notes = notes
            print(f"Updated existing calibration for {camera_id}.")
        else:
            row = CameraCalibration(
                camera_id=camera_id, px_per_meter=px_per_meter,
                calibration_method=calibration_method, notes=notes,
            )
            db.add(row)
            print(f"Created new calibration for {camera_id}.")
        db.commit()
    finally:
        db.close()

    print(f"  px_per_meter = {px_per_meter:.3f}")
    print(f"  method       = {calibration_method}")
    print("Restart the backend (or trigger a reload) to pick this up — "
          "camera_calibration.py caches in memory. A future pass could expose "
          "a reload endpoint the way Day 7's watchlist reload works.")


def _list_calibrations() -> None:
    from backend.db.models import CameraCalibration
    from backend.db.session import SessionLocal

    db = SessionLocal()
    try:
        rows = db.query(CameraCalibration).order_by(CameraCalibration.camera_id).all()
        if not rows:
            print("No cameras calibrated yet.")
            return
        print(f"{'camera_id':<16} {'px_per_meter':>14} {'method':<20} {'calibrated_at'}")
        for r in rows:
            print(f"{r.camera_id:<16} {r.px_per_meter:>14.3f} {r.calibration_method:<20} "
                  f"{r.calibrated_at.isoformat() if r.calibrated_at else ''}")
    finally:
        db.close()


def main() -> None:
    parser = argparse.ArgumentParser(description="Per-camera px-per-meter calibration (Day 8)")
    parser.add_argument("--camera-id", help="Camera string id, e.g. CAM-01")
    parser.add_argument("--real-distance-meters", type=float,
                         help="Known real-world distance between the two points, in meters")
    parser.add_argument("--point1", type=_parse_point, help="First pixel point as 'x,y'")
    parser.add_argument("--point2", type=_parse_point, help="Second pixel point as 'x,y'")
    parser.add_argument("--frame-image", help="Reference frame image for interactive point-picking")
    parser.add_argument("--notes", default=None, help="Free-text note, e.g. what the two points mark")
    parser.add_argument("--list", action="store_true", help="List existing calibrations and exit")
    args = parser.parse_args()

    if args.list:
        _list_calibrations()
        return

    if not args.camera_id or not args.real_distance_meters:
        parser.error("--camera-id and --real-distance-meters are required unless --list is given")

    if args.point1 and args.point2:
        p1, p2 = args.point1, args.point2
        method = "manual_two_point"
    elif args.frame_image:
        p1, p2 = _interactive_pick_points(args.frame_image)
        method = "manual_two_point"
    else:
        parser.error("Provide either --point1/--point2, or --frame-image for interactive picking.")
        return  # unreachable, satisfies type checkers

    pixel_distance = math.hypot(p2[0] - p1[0], p2[1] - p1[1])
    if pixel_distance < 1e-6:
        print("[ERROR] The two points are identical — cannot calibrate from zero pixel distance.",
              file=sys.stderr)
        sys.exit(1)

    px_per_meter = pixel_distance / args.real_distance_meters
    _upsert_calibration(args.camera_id, px_per_meter, method, args.notes)


if __name__ == "__main__":
    main()
