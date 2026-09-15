"""backend/scripts/principal_point_probe.py — is the orthogonality failure a
cropped sensor, or is the geometry simply wrong?

THE QUESTION
  Tier 1 derives focal length from f^2 = -(vp1 - c).(vp2 - c), where c is the
  principal point. Every implementation of that formula, including this
  project's, assumes c sits at the image centre. CCTV delivered through an NVR
  is often cropped or digitally zoomed, which moves the optical centre away
  from the frame centre and breaks the assumption silently.

  Measured separations between the two vanishing directions, as seen from the
  image centre:

      CAM_21   85.9 deg      4 degrees short of the 90 needed
      CAM_08   35.9 deg      not close
      CAM_09    0.2 deg      VP2 collapsed onto VP1
      CAM_06       -         no consensus at all

  CAM_21 is the informative one. A four-degree miss is what a modest crop
  would produce; a fifty-degree miss is not.

THE TEST, WHICH NEEDS NO METADATA
  Rather than hunting for a crop offset in vendor metadata that may not exist,
  ask the question backwards: over a grid of candidate principal points, where
  does f^2 come out POSITIVE with a field of view a CCTV lens could actually
  have? Then measure how far the nearest such point is from the frame centre.

      a few tens of pixels   consistent with a real crop; worth pursuing
      several hundred        not a crop, and the geometry is wrong for another
                             reason; stop here

  This cannot prove a particular principal point is correct — that needs a
  third vanishing point. It can show whether ANY plausible one exists, which
  is enough to decide whether to keep going.

Run:  python -m backend.scripts.principal_point_probe [camera ...]
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

VP2 = ROOT / "output" / "camera_calibration" / "vp2_from_edges.json"
MIN_FOV_DEG, MAX_FOV_DEG = 15.0, 120.0
# Search out to a full frame width either side; a crop bigger than that is not
# a crop, it is a different camera.
GRID_HALF = 1.0
GRID_STEP = 8.0


def probe(rec: dict) -> dict:
    W, H = rec["frame_size"]
    centre = np.array([W / 2.0, H / 2.0])
    v1 = np.array(rec["vp1"], dtype=float)
    v2 = np.array(rec["vp2"], dtype=float)

    xs = np.arange(centre[0] - GRID_HALF * W, centre[0] + GRID_HALF * W, GRID_STEP)
    ys = np.arange(centre[1] - GRID_HALF * H, centre[1] + GRID_HALF * H, GRID_STEP)
    gx, gy = np.meshgrid(xs, ys)

    d1x, d1y = v1[0] - gx, v1[1] - gy
    d2x, d2y = v2[0] - gx, v2[1] - gy
    dot = d1x * d2x + d1y * d2y
    f2 = -dot

    ok = f2 > 0
    f = np.where(ok, np.sqrt(np.maximum(f2, 1e-9)), np.nan)
    fov = 2.0 * np.degrees(np.arctan(W / (2.0 * np.maximum(f, 1e-9))))
    good = ok & (fov >= MIN_FOV_DEG) & (fov <= MAX_FOV_DEG)

    out = {"camera": rec["camera"], "frame_size": [W, H],
           "vp1": rec["vp1"], "vp2": rec["vp2"],
           "orthogonal_at_centre": bool(-(v1 - centre) @ (v2 - centre) > 0),
           "grid_points": int(gx.size),
           "plausible_points": int(good.sum())}

    if not good.any():
        out["verdict"] = "no plausible principal point anywhere on the grid"
        return out

    dist = np.hypot(gx - centre[0], gy - centre[1])
    dist_ok = np.where(good, dist, np.inf)
    k = np.unravel_index(np.argmin(dist_ok), dist_ok.shape)
    best = np.array([gx[k], gy[k]])
    out["nearest_plausible_pp"] = [round(float(best[0]), 1), round(float(best[1]), 1)]
    out["offset_from_centre_px"] = round(float(dist[k]), 1)
    out["offset_as_frac_of_width"] = round(float(dist[k] / W), 3)
    out["focal_px_there"] = round(float(f[k]), 1)
    out["fov_deg_there"] = round(float(fov[k]), 1)

    off = float(dist[k])
    if off <= 0.05 * W:
        out["verdict"] = "consistent with a modest crop — worth pursuing"
    elif off <= 0.15 * W:
        out["verdict"] = "large offset; possible but would be an unusual crop"
    else:
        out["verdict"] = "offset too large to be a crop — geometry wrong for another reason"
    return out


def main() -> int:
    if not VP2.is_file():
        print("run fit_vp2_from_edges first")
        return 1
    rows = json.loads(VP2.read_text(encoding="utf-8"))
    cams = sys.argv[1:] or None

    print("%-9s %10s %9s %11s %10s %9s  %s"
          % ("camera", "sep@centre", "offset px", "as frac W", "focal px",
             "FOV deg", "verdict"))
    print("-" * 108)
    for r in sorted(rows, key=lambda x: x["camera"]):
        if cams and r["camera"] not in cams:
            continue
        if not r.get("vp2"):
            print("%-9s %10s %9s %11s %10s %9s  %s"
                  % (r["camera"], "-", "-", "-", "-", "-",
                     "no VP2 to test (%s)" % r.get("status")))
            continue
        p = probe(r)
        print("%-9s %10s %9s %11s %10s %9s  %s"
              % (p["camera"], str(r.get("vp_separation_deg", "-")),
                 str(p.get("offset_from_centre_px", "-")),
                 str(p.get("offset_as_frac_of_width", "-")),
                 str(p.get("focal_px_there", "-")),
                 str(p.get("fov_deg_there", "-")), p["verdict"]))
    print("\nA principal point cannot be confirmed without a third vanishing")
    print("point. This only says whether a plausible one could exist at all.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
