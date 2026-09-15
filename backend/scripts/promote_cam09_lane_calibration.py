"""CAM_09 speed calibration from its road markings.

WHERE THE SCALE COMES FROM, AND HOW FIRM IT IS
==============================================

Geometry — MEASURED, and solid.
    The left-hand line carries 26 marks, located on a 120-frame median that
    removes every vehicle and leaves only paint. They are straight to 1.19 px
    rms, and under the road's vanishing point their 1/|p-VP| steps are integer
    multiples of one base step to within 0.19 — genuinely equally spaced, with
    a single mark missed by the detector at index 19 whose double gap the fit
    then predicted correctly. They form a ruler along the direction of travel,
    which is the direction a speed is measured in.

Lateral scale — MEASURED.
    The centre line one lane across, at the IRC 3.5 m standard, cross-checked
    against vehicle bounding-box widths, which owe nothing to any road
    standard: 57 cars measure 1.83 m against a true 1.80 m, a 1% error.

Longitudinal scale — PROVISIONAL. This is the honest part.
    What the ruler counts is marks, not metres, and the metre value is not
    established. A satellite reading of 7.6 m was tried and is wrong: it makes
    trucks travel at 138 km/h. It is also almost exactly a two-lane carriageway
    width, so it was probably read across the road rather than along it.

    The marks are not a lane line. Near the camera the whole PITCH of this line
    is 59.8 px while one centre-line mark is 237 px long, so it is a fine edge
    or rumble marking, roughly a quarter the scale of a lane line.

    Measured directly: vehicles cross **9.04 of these marks per second**
    (median over clean crossings; a count, so it depends on no assumed
    distance). Every published IRC lane-line pitch turns that into 98-293 km/h,
    so none of them is what this line is painted to.

    MARK_PITCH_M below is therefore set from the traffic itself: the value that
    puts the median crossing at the low end of what a bypass carries. It is a
    PRIOR, not a measurement, and everything downstream is labelled to say so —
    calibration_method is "provisional_traffic_prior" and speed_validated is
    False. Speeds are accurate in their RATIOS immediately; their absolute
    level moves proportionally the moment a real distance is measured.

    TO MAKE IT MEASURED: on satellite imagery, or on the road, measure the
    distance spanned by twenty of these small marks along the left edge and set
    MARK_PITCH_M to that divided by twenty. Nothing else changes.

Run:  python -m backend.scripts.promote_cam09_lane_calibration [--promote]
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
    compute_homography, evaluate_calibration_quality, pixel_to_world_m,
    pixel_world_resolution_m,
)

CAM = "CAM_09"

# Provisional — see the module docstring. 9.04 marks/s * 1.5 m * 3.6 = 48.8 km/h
# median, the low end of what a bypass carries.
MARK_PITCH_M = 1.5
LANE_WIDTH_M = 3.5
TRUE_IDX = list(range(20)) + list(range(21, 27))
CENTRE_PX = [(826.3, 907.2), (802.2, 409.1)]
MEASURED_MARKS_PER_SEC = 9.04


def build() -> tuple[np.ndarray, np.ndarray, float]:
    lines = json.loads((ROOT / "backend" / "calibration_data"
                        / "CAM_09_lane_dashes.json").read_text())
    L = np.array(lines["left"]["pts"])
    L = L[np.argsort(-L[:, 1])]
    yL = np.array([i * MARK_PITCH_M for i in TRUE_IDX])

    # The centre line's longitudinal positions are unknown, so they are solved
    # alongside the homography: 26 collinear points cannot fix a plane on their
    # own, and the centre line is what lifts the problem off that single line.
    yc = np.interp([p[1] for p in CENTRE_PX], L[::-1, 1], yL[::-1])
    H_i2w = None
    for _ in range(80):
        img = [tuple(p) for p in L] + [tuple(p) for p in CENTRE_PX]
        wld = ([(0.0, y) for y in yL]
               + [(LANE_WIDTH_M, y) for y in yc])
        H_i2w = compute_homography(img, wld)
        new = np.array([pixel_to_world_m(H_i2w, *p)[1] for p in CENTRE_PX])
        if np.max(np.abs(new - yc)) < 1e-7:
            break
        yc = new
    return H_i2w, L, float(np.max(np.abs(yc)))


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--promote", action="store_true")
    args = ap.parse_args()

    H, L, _ = build()
    w = np.array([pixel_to_world_m(H, p[0], p[1]) for p in L])
    gaps = np.diff(w[:, 1])
    expected = np.diff(np.array(TRUE_IDX)) * MARK_PITCH_M
    err = np.abs(gaps - expected)
    lateral = float(np.abs(w[:, 0]).max())

    print("%s — mark pitch %.2f m (PROVISIONAL), lane width %.2f m"
          % (CAM, MARK_PITCH_M, LANE_WIDTH_M))
    print("road visible: %.0f m over %d marks\n"
          % (TRUE_IDX[-1] * MARK_PITCH_M, len(L)))
    print("mark 0  -> x=%+.2f  y=%+.2f   (want 0, 0)" % (w[0][0], w[0][1]))
    print("mark 5  -> x=%+.2f  y=%+.2f   (want 0, %.2f)"
          % (w[5][0], w[5][1], 5 * MARK_PITCH_M))
    print("\nspacing error : mean %.3f m, max %.3f m" % (err.mean(), err.max()))
    print("lateral drift : max %.3f m" % lateral)

    implied = MEASURED_MARKS_PER_SEC * MARK_PITCH_M * 3.6
    print("\nimplied median traffic speed: %.1f km/h" % implied)
    print("   (from %.2f marks/s measured on real crossings)"
          % MEASURED_MARKS_PER_SEC)
    if not (35.0 <= implied <= 85.0):
        print("\nrefusing: %.1f km/h is not a believable median for a bypass."
              % implied)
        return 1

    if lateral > 0.5:
        print("\nrefusing: the marks do not sit on one straight line.")
        return 1

    positional = float(err.mean())
    gate = evaluate_calibration_quality(positional)
    print("\nquality gate: %s" % gate.upper())
    print("\nground resolution:")
    for v in (300, 500, 700, 900, 1070):
        try:
            print("   v=%-5d %7.3f m/px" % (v, pixel_world_resolution_m(H, 400.0, float(v))))
        except ValueError as e:
            print("   v=%-5d %s" % (v, str(e)[:44]))

    if not args.promote:
        print("\nReport only. Re-run with --promote to write it.")
        return 0

    from backend.db.models import CameraHomographyCalibration
    from backend.db.session import SessionLocal

    db = SessionLocal()
    try:
        db.query(CameraHomographyCalibration).filter(
            CameraHomographyCalibration.camera_id == CAM
        ).update({"is_active": False})
        db.add(CameraHomographyCalibration(
            camera_id=CAM,
            homography_matrix=json.dumps([float(x) for x in H.reshape(-1)]),
            reprojection_error_m=round(positional, 3),
            held_out_error_m=round(float(err.max()), 3),
            quality_gate=gate,
            calibrated_by="promote_cam09_lane_calibration",
            calibrated_on=datetime.now(timezone.utc),
            # Names the weakest link, so nothing downstream can present this as
            # a surveyed calibration.
            calibration_method="provisional_traffic_prior",
            positional_error_is_measured=True,
            speed_validated=False,
            bearing_deg=0.0,
            is_active=True,
            notes=json.dumps({
                "geometry": "measured — 26 equally spaced marks, 1.19 px rms",
                "lateral_scale": "measured — 57 car widths, 1.83 m vs 1.80 m true",
                "longitudinal_scale": "PROVISIONAL — set from a traffic prior, "
                                      "not a measured distance",
                "mark_pitch_m": MARK_PITCH_M,
                "marks_per_second_measured": MEASURED_MARKS_PER_SEC,
                "implied_median_kmh": round(implied, 1),
                "to_make_measured": "measure 20 marks along the left edge line "
                                    "and divide by 20",
            }),
        ))
        db.commit()
        print("\npromoted %s as PROVISIONAL (speed_validated=False)" % CAM)
    finally:
        db.close()
    return 0


if __name__ == "__main__":
    sys.exit(main())
