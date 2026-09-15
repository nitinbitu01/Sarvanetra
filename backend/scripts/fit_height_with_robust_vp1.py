"""backend/scripts/fit_height_with_robust_vp1.py — recover a mounting height
for cameras whose VP1 only exists via the robust estimator.

WHY THIS GAP EXISTS
  `fit_camera_calibration` bails at `insufficient_tracks` before it ever
  reaches the height step. CAM_25 has 14 tracks and 294 track points with
  vehicle classes present — enough width observations to work with — but only
  2 tracks pass the 3% straightness filter, so the script stops at the
  vanishing-point stage and reports no height.

  `fit_vp1_robust` then found a VP1 for it anyway, and a good one: 12 of 13
  lines agreeing at 10.9 degrees of direction spread. That VP1 was never fed
  back into the height estimator, so the camera stayed blocked on a step that
  had already been unblocked.

  This script closes that loop. It changes no estimator — it reuses
  `estimate_height` exactly as written, with a vanishing point from a different
  source.

WHAT IT DOES NOT DO
  A height here is not a calibration. It feeds the focal-length steps, which
  have their own gates, and those in turn feed the validation gate. Nothing
  reaches the database from this script.

Run:  python -m backend.scripts.fit_height_with_robust_vp1 [camera ...]
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from backend.scripts.fit_camera_calibration import (      # noqa: E402
    MAX_HEIGHT_M, MIN_HEIGHT_M, estimate_height)

GEOM = ROOT / "output" / "road_geometry"
VP1R = ROOT / "output" / "camera_calibration" / "vp1_robust.json"
CALIB = ROOT / "output" / "camera_calibration" / "calibration.json"
OUT = ROOT / "output" / "camera_calibration" / "height_from_robust_vp1.json"

MAX_IQR_RATIO = 0.45


def main() -> int:
    robust = {r["camera"]: r for r in json.loads(VP1R.read_text(encoding="utf-8"))}
    prior = {r["camera"]: r for r in json.loads(CALIB.read_text(encoding="utf-8"))}
    cams = sys.argv[1:] or sorted(robust)

    rows = []
    for cam in cams:
        rb = robust.get(cam, {})
        if rb.get("status") != "solved":
            continue
        had = prior.get(cam, {}).get("height_m")
        p = GEOM / f"{cam}.json"
        if not p.is_file():
            continue
        tracks = json.loads(p.read_text(encoding="utf-8"))["tracks"]
        vp = np.array(rb["vp1"], dtype=float)

        h, iqr, n = estimate_height(tracks, vp)
        r = {"camera": cam, "vp1_robust": rb["vp1"],
             "vp1_inlier_frac": rb.get("inlier_frac"),
             "existing_height_m": had, "height_m": None if h is None else round(h, 2),
             "height_iqr_m": round(iqr, 2), "observations": n}

        if h is None:
            r["status"] = "insufficient_width_observations"
        elif not (MIN_HEIGHT_M <= h <= MAX_HEIGHT_M):
            r["status"] = f"implausible_height ({h:.2f} m)"
        elif h > 0 and iqr / h > MAX_IQR_RATIO:
            r["status"] = f"unstable (IQR is {iqr / h:.0%} of the estimate)"
        else:
            r["status"] = "solved"
            r["iqr_ratio"] = round(iqr / h, 2)
        rows.append(r)

    print("%-9s %14s %10s %8s %7s %9s  %s"
          % ("camera", "existing h", "new h", "IQR", "n obs", "inlier", "status"))
    print("-" * 92)
    for r in rows:
        print("%-9s %14s %10s %8s %7d %9s  %s"
              % (r["camera"], str(r["existing_height_m"] or "-"),
                 str(r["height_m"] or "-"), str(r["height_iqr_m"]),
                 r["observations"], str(r["vp1_inlier_frac"]), r["status"]))

    new = [r for r in rows if r["status"] == "solved" and not r["existing_height_m"]]
    print(f"\n{len(new)} camera(s) gained a height that had none")
    for r in new:
        print(f"   {r['camera']}: {r['height_m']} m "
              f"(IQR {r['iqr_ratio']:.0%}, {r['observations']} observations)")

    OUT.write_text(json.dumps(rows, indent=1), encoding="utf-8")
    print(f"\nwrote {OUT}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
