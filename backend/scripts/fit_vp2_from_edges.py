"""backend/scripts/fit_vp2_from_edges.py — Tier 1: the second vanishing point
from the vehicles themselves, so a camera does not need a second traffic
stream to yield a focal length.

THE PROBLEM THIS SOLVES
  fit_camera_calibration needs two orthogonal vanishing points. It takes the
  second from a second stream of traffic, and most of this fleet watches
  one-directional roads. After pooling every clip, six cameras reached a
  plausible pole height and NONE reached a metric ground plane, so absolute
  km/h stayed unavailable network-wide.

THE METHOD
  Dubská, Herout & Sochor (IEEE T-ITS 2014). A vehicle is a box. Its bumper
  line, roofline and window frames run ACROSS the direction of travel, so
  those edges converge on the transverse vanishing point — the orthogonal
  partner VP1 has been missing.

  Edges along the direction of travel converge on VP1 instead, and they are
  the majority. They are removed first, by testing whether a segment's line
  passes close to VP1. What remains votes for VP2.

WHY RANSAC AND NOT LEAST SQUARES
  The surviving set still contains shadows, lane markings, railings and
  background structure caught inside a box. A least-squares intersection is
  dragged anywhere by those; a consensus fit ignores them by construction.

WHAT IT REFUSES TO DO
  A focal length is accepted only if
    * VP1 and VP2 are separated enough for the orthogonality constraint to be
      conditioned — near-parallel directions make f swing wildly with noise,
    * f^2 = -(vp1-c).(vp2-c) comes out positive, meaning the two directions
      really are orthogonal on the ground rather than an artefact,
    * the implied field of view is physically plausible for a CCTV lens, and
    * the consensus rests on enough inliers to be more than a coincidence.

  A camera failing any of these is reported with the reason. Nothing here is
  written to the database; `camera_calibrations.homography_matrix` is read by
  speed_estimator, and an unvalidated matrix there is exactly the failure this
  project has already corrected once.

Run:  python -m backend.scripts.fit_vp2_from_edges [camera ...]
"""
from __future__ import annotations

import json
import sys
from pathlib import Path
from typing import Optional

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

EDGES = ROOT / "output" / "vehicle_edges"
CALIB = ROOT / "output" / "camera_calibration" / "calibration.json"
OUT = ROOT / "output" / "camera_calibration" / "vp2_from_edges.json"

# A segment pointing within this angle of VP1 is a motion-aligned edge and
# belongs to VP1, not VP2.
#
# WHY ANGULAR AND NOT A PIXEL DISTANCE
#   The first version asked whether a segment's line passed within 60 px of
#   VP1. That is the correct geometric question and the wrong threshold, and
#   the failure it produced is instructive. CAM_21's VP1 sits at (1027, -508),
#   far outside the frame; a 40 px edge aligned to within two degrees of the
#   road still misses a point that distant by hundreds of pixels. So the test
#   rejected only 4% of segments (11,808 -> 11,364), the road-aligned majority
#   survived, RANSAC locked onto them, and the "second" vanishing point came
#   back 1.2 degrees from the first — a separation the fitter then correctly
#   refused.
#
#   Measuring the angle between the segment and the direction to VP1 removes
#   the dependence on how far away VP1 happens to be, which is the property
#   that varies most between these cameras.
VP1_REJECT_DEG = 12.0
# How far off VP1's row a candidate VP2 may sit and still be taken as a
# ground-plane direction, as a fraction of frame height. Allows for a little
# camera roll without admitting the vertical vanishing point.
HORIZON_TOL_FRAC = 0.15
# Consensus threshold for a segment to support a candidate VP2.
RANSAC_INLIER_PX = 45.0
RANSAC_ITERS = 4000
MIN_INLIERS = 80
# Two vanishing directions closer than this leave f poorly conditioned.
MIN_VP_SEPARATION_DEG = 25.0
# A CCTV lens on a 1920-wide sensor: roughly 15 to 120 degrees horizontal.
MIN_FOV_DEG, MAX_FOV_DEG = 15.0, 120.0


def seg_line(p) -> np.ndarray:
    """Homogeneous line through a segment's two endpoints."""
    a = np.array([p[0], p[1], 1.0])
    b = np.array([p[2], p[3], 1.0])
    return np.cross(a, b)


def angle_to_vp(p, vp: np.ndarray) -> float:
    """Angle in degrees between a segment and the direction to `vp`.

    Both are treated modulo 180: a segment has no head or tail, so pointing at
    the vanishing point and pointing directly away from it are the same
    alignment.
    """
    d = np.array([p[2] - p[0], p[3] - p[1]], dtype=float)
    mid = np.array([(p[0] + p[2]) / 2.0, (p[1] + p[3]) / 2.0])
    to_vp = vp - mid
    n1, n2 = np.hypot(*d), np.hypot(*to_vp)
    if n1 < 1e-9 or n2 < 1e-9:
        return 90.0
    cos = float(np.clip(abs(d @ to_vp) / (n1 * n2), 0.0, 1.0))
    return float(np.degrees(np.arccos(cos)))


def ransac_vp(lines: list, rng: np.random.Generator,
              horizon_y: Optional[float] = None,
              horizon_tol: float = 0.0) -> tuple:
    """Best-consensus intersection of many lines.

    `horizon_y` constrains candidates to the horizon.

    WHY THAT CONSTRAINT IS NEEDED
      Two directions on the ground plane both vanish ON the horizon line, and
      for a camera with negligible roll — the assumption the height estimator
      already makes — that line is horizontal, so both share an image row.
      Without the constraint the search is free to pick any strong consensus,
      and on CAM_21 it did: 3,192 inliers on a point 494 px above the centre
      while VP1 sat 974 px above it. Two different rows, so not two ground
      directions — the winner was vertical structure (vehicle sides, poles),
      whose vanishing point is the third one, not the transverse one. The
      orthogonality test then correctly refused a focal length.
    """
    if len(lines) < MIN_INLIERS:
        return None, 0
    L = np.array(lines, dtype=float)
    best, best_n = None, 0
    n = len(L)
    for _ in range(RANSAC_ITERS):
        i, j = rng.integers(0, n, 2)
        if i == j:
            continue
        p = np.cross(L[i], L[j])
        if abs(p[2]) < 1e-9:                 # parallel: intersection at infinity
            continue
        pt = np.array([p[0] / p[2], p[1] / p[2]])
        if not np.all(np.isfinite(pt)) or np.abs(pt).max() > 1e6:
            continue
        if horizon_y is not None and abs(pt[1] - horizon_y) > horizon_tol:
            continue                          # not on the horizon: not a
            #                                   ground-plane direction
        d = np.abs(L[:, 0] * pt[0] + L[:, 1] * pt[1] + L[:, 2]) / \
            np.maximum(np.hypot(L[:, 0], L[:, 1]), 1e-9)
        cnt = int((d < RANSAC_INLIER_PX).sum())
        if cnt > best_n:
            best, best_n = pt, cnt
    if best is None or best_n < MIN_INLIERS:
        return None, best_n
    # Refine on the inliers only.
    d = np.abs(L[:, 0] * best[0] + L[:, 1] * best[1] + L[:, 2]) / \
        np.maximum(np.hypot(L[:, 0], L[:, 1]), 1e-9)
    inl = L[d < RANSAC_INLIER_PX]
    sol, *_ = np.linalg.lstsq(inl[:, :2], -inl[:, 2], rcond=None)
    return sol, best_n


def fit_one(cam: str, edges: dict, prior: dict) -> dict:
    W, H = edges["frame_size"]
    centre = np.array([W / 2.0, H / 2.0])
    res = {"camera": cam, "frame_size": [W, H],
           "segments": len(edges.get("segments", []))}

    vp1 = prior.get("vp1")
    if not vp1:
        res["status"] = "no_vp1"        # nothing to be orthogonal to
        return res
    vp1 = np.array(vp1, dtype=float)
    res["vp1"] = [round(float(v), 1) for v in vp1]

    kept = []
    for s in edges.get("segments", []):
        if angle_to_vp(s["p"], vp1) < VP1_REJECT_DEG:
            continue                     # this edge belongs to VP1
        kept.append(seg_line(s["p"]))
    res["after_vp1_filter"] = len(kept)
    if len(kept) < MIN_INLIERS:
        res["status"] = "too_few_transverse_edges"
        return res

    rng = np.random.default_rng(7)
    # Both ground directions share the horizon row under the zero-roll
    # assumption this pipeline already makes elsewhere.
    tol = HORIZON_TOL_FRAC * H
    res["horizon_y"] = round(float(vp1[1]), 1)
    res["horizon_tol_px"] = round(float(tol), 1)
    vp2, inliers = ransac_vp(kept, rng, horizon_y=float(vp1[1]), horizon_tol=tol)
    res["inliers"] = int(inliers)
    if vp2 is None:
        res["status"] = "no_consensus_vp2"
        return res
    res["vp2"] = [round(float(v), 1) for v in vp2]

    a1 = np.degrees(np.arctan2(*(vp1 - centre)[::-1]))
    a2 = np.degrees(np.arctan2(*(vp2 - centre)[::-1]))
    sep = abs((a1 - a2 + 90) % 180 - 90)
    res["vp_separation_deg"] = round(float(sep), 1)
    if sep < MIN_VP_SEPARATION_DEG:
        res["status"] = "vp_directions_too_close"
        return res

    dot = float((vp1 - centre) @ (vp2 - centre))
    if dot >= 0:
        res["status"] = "vps_not_orthogonal"
        return res
    f = float(np.sqrt(-dot))
    fov = 2.0 * np.degrees(np.arctan(W / (2.0 * f)))
    res["focal_px"] = round(f, 1)
    res["fov_deg"] = round(float(fov), 1)
    if not (MIN_FOV_DEG <= fov <= MAX_FOV_DEG):
        res["status"] = "implausible_field_of_view"
        return res

    height = prior.get("height_m")
    if height is None:
        # A focal length without a height gives shape but no scale.
        res["status"] = "focal_only_no_height"
        return res

    res["height_m"] = height
    res["status"] = "full"
    res["mapping"] = {"type": "planar_vp_edges", "vp_x": float(vp1[0]),
                      "vp_y": float(vp1[1]), "height_m": float(height),
                      "focal_px": f}
    return res


def main() -> int:
    if not EDGES.is_dir():
        print("run collect_vehicle_edges first")
        return 1
    prior = {r["camera"]: r for r in json.loads(CALIB.read_text(encoding="utf-8"))}
    cams = sys.argv[1:] or None

    rows = []
    for p in sorted(EDGES.glob("CAM_*.json")):
        e = json.loads(p.read_text(encoding="utf-8"))
        if cams and e["camera"] not in cams:
            continue
        rows.append(fit_one(e["camera"], e, prior.get(e["camera"], {})))

    print(f"{'camera':<9}{'segs':>8}{'transverse':>12}{'inliers':>9}"
          f"{'sep deg':>9}{'focal px':>10}{'FOV':>8}  status")
    print("-" * 84)
    for r in sorted(rows, key=lambda x: x["camera"]):
        print(f"{r['camera']:<9}{r.get('segments', 0):>8}"
              f"{r.get('after_vp1_filter', 0):>12}{r.get('inliers', 0):>9}"
              f"{str(r.get('vp_separation_deg', '-')):>9}"
              f"{str(r.get('focal_px', '-')):>10}"
              f"{str(r.get('fov_deg', '-')):>8}  {r['status']}")

    full = [r for r in rows if r["status"] == "full"]
    print(f"\n{len(full)} camera(s) reach a metric ground plane via Tier 1")
    for r in full:
        print(f"   {r['camera']}: f={r['focal_px']}px  FOV={r['fov_deg']}deg  "
              f"h={r['height_m']}m  ({r['inliers']} inliers)")

    OUT.write_text(json.dumps(rows, indent=1), encoding="utf-8")
    print(f"\nwrote {OUT}")
    print("\nNothing written to camera_calibrations. A focal length that agrees"
          "\nwith no independent measurement is a candidate, not a calibration.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
