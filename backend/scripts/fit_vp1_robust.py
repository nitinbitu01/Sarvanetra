"""backend/scripts/fit_vp1_robust.py — Phase 2C: a motion vanishing point for
the cameras that currently have none.

WHAT IS BROKEN
  `fit_camera_calibration` builds VP1 from tracks it first filters for
  straightness: a track is discarded unless its lateral deviation is under 3%
  of its length. That is a sensible instinct — a vehicle on a road travels
  straight, and a wandering track is a turning vehicle or an association error.

  It is also brutal. Measured after pooling every clip:

      CAM_10    238 tracks ->   6 straight      no_vp1
      CAM_12    130 tracks ->   1 straight      no_vp1
      CAM_07     66 tracks ->   2 straight      no_vp1
      CAM_03     10 tracks ->   1 straight      no_vp1

  A 200 px track is allowed 6 px of wander, and detector jitter alone is
  several pixels. So four cameras with hundreds of tracks between them produce
  no vanishing point at all, and everything downstream — height, focal length,
  speed — is unreachable for them.

THE FIX IS NOT A LOOSER GATE
  Raising the threshold lets genuinely curved and mis-associated tracks vote,
  and a vanishing point fitted by least squares is dragged anywhere by them.
  That trades a starved estimate for a wrong one.

  Instead the gate is removed and the ESTIMATOR is made robust: every track
  with enough span contributes a line, and RANSAC finds the point that the
  largest consistent set of them passes through. A turning vehicle is then an
  outlier the consensus ignores, rather than an input that has to be predicted
  and excluded in advance.

  This is the same change made to VP2 earlier in this pipeline, for the same
  reason, and it is the general lesson: when a pre-filter is discarding most of
  the data, make the estimator tolerate outliers rather than making the filter
  admit them.

WHAT IT STILL REFUSES
  A vanishing point is meaningless if the lines are near-parallel: their
  intersection is then determined by noise and swings wildly. Cameras whose
  consensus rests on too few lines, or whose track directions span too small an
  angle, are reported as such rather than given a point.

Run:  python -m backend.scripts.fit_vp1_robust [camera ...]
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

GEOM = ROOT / "output" / "road_geometry"
CALIB = ROOT / "output" / "camera_calibration" / "calibration.json"
OUT = ROOT / "output" / "camera_calibration" / "vp1_robust.json"

# A track shorter than this in pixels has no reliable direction.
MIN_SPAN_PX = 40.0
MIN_POINTS = 4
# Consensus tolerance: how close a line must pass to the candidate point.
INLIER_PX = 25.0
RANSAC_ITERS = 3000
MIN_INLIERS = 10
# Track directions must span at least this angle, or the lines are parallel and
# their intersection is noise.
MIN_DIRECTION_SPREAD_DEG = 6.0

# ── Acceptance gate ──────────────────────────────────────────────────────────
# Removing the straightness filter admits junk as well as usable tracks, and
# RANSAC tolerating outliers is not the same as RANSAC being right about a set
# that is mostly outliers. Checked against the cameras whose VP1 was already
# known good, the pattern is unambiguous:
#
#     CAM_06   inliers 0.55, spread  21.7 deg  ->  agrees to  20 px
#     CAM_09   inliers 0.66, spread  28.8 deg  ->  agrees to  57 px
#     CAM_08   inliers 0.20, spread 109.7 deg  ->  differs by 329 px
#     CAM_02   inliers 0.47, spread 164.5 deg  ->  differs by 1261 px
#     CAM_13   inliers 0.32, spread  63.6 deg  ->  differs by 3878 px
#
# A direction spread above about a right angle means the tracks are not all on
# one road — they are several flows plus association errors — and a single
# vanishing point does not describe them however good the consensus looks.
MIN_INLIER_FRAC = 0.50
MAX_DIRECTION_SPREAD_DEG = 60.0


def track_line(points: list):
    """Line through a track's ground-contact points, with no straightness gate.

    Returns (homogeneous line, span, unit direction) or None.
    """
    p = np.array([[q["u"], q["v"]] for q in points], dtype=float)
    if len(p) < MIN_POINTS:
        return None
    span = float(np.hypot(*(p[-1] - p[0])))
    if span < MIN_SPAN_PX:
        return None
    centred = p - p.mean(0)
    _, s, vt = np.linalg.svd(centred, full_matrices=False)
    direction = vt[0]
    # Total-least-squares line through the centroid along the principal axis.
    normal = np.array([-direction[1], direction[0]])
    c = -normal @ p.mean(0)
    # Residual is reported, not gated on — RANSAC decides what to believe.
    resid = float(s[1] / max(span, 1e-6))
    return np.array([normal[0], normal[1], c]), span, direction, resid


def ransac_point(L: np.ndarray, w: np.ndarray, rng) -> tuple:
    """Point that the largest weighted consensus of lines passes through."""
    n = len(L)
    if n < MIN_INLIERS:
        return None, 0
    norms = np.maximum(np.hypot(L[:, 0], L[:, 1]), 1e-9)
    best, best_score = None, -1.0
    for _ in range(RANSAC_ITERS):
        i, j = rng.integers(0, n, 2)
        if i == j:
            continue
        p = np.cross(L[i], L[j])
        if abs(p[2]) < 1e-9:
            continue
        pt = np.array([p[0] / p[2], p[1] / p[2]])
        if not np.all(np.isfinite(pt)) or np.abs(pt).max() > 1e7:
            continue
        d = np.abs(L[:, 0] * pt[0] + L[:, 1] * pt[1] + L[:, 2]) / norms
        inl = d < INLIER_PX
        # Weighted by track length: a long track constrains its direction far
        # better than a short one and should count for more.
        score = float(w[inl].sum())
        if score > best_score:
            best, best_score, best_mask = pt, score, inl
    if best is None:
        return None, 0
    # Refine on the inliers, still weighted.
    sub, sw = L[best_mask], np.sqrt(w[best_mask])
    sol, *_ = np.linalg.lstsq(sub[:, :2] * sw[:, None], -sub[:, 2] * sw,
                              rcond=None)
    return sol, int(best_mask.sum())


def fit(cam: str, data: dict) -> dict:
    tracks = data["tracks"]
    W, H = data["frame_size"]
    lines, spans, dirs, resids = [], [], [], []
    for pts in tracks.values():
        r = track_line(pts)
        if r is None:
            continue
        lines.append(r[0])
        spans.append(r[1])
        dirs.append(r[2])
        resids.append(r[3])

    out = {"camera": cam, "frame_size": [W, H],
           "tracks_total": len(tracks), "lines_used": len(lines)}
    if len(lines) < MIN_INLIERS:
        out["status"] = "too_few_tracks_with_span"
        return out

    ang = np.array([np.degrees(np.arctan2(d[1], d[0])) % 180.0 for d in dirs])
    spread = float(np.percentile(ang, 90) - np.percentile(ang, 10))
    out["direction_spread_deg"] = round(spread, 1)
    if spread < MIN_DIRECTION_SPREAD_DEG:
        out["status"] = "directions_too_parallel"
        return out

    rng = np.random.default_rng(3)
    vp, inl = ransac_point(np.array(lines), np.array(spans), rng)
    out["inliers"] = inl
    out["inlier_frac"] = round(inl / len(lines), 2)
    if vp is None or inl < MIN_INLIERS:
        out["status"] = "no_consensus"
        return out

    out["vp1"] = [round(float(vp[0]), 1), round(float(vp[1]), 1)]
    out["vp1_in_frame"] = bool(0 <= vp[0] <= W and 0 <= vp[1] <= H)
    out["median_straightness"] = round(float(np.median(resids)), 4)
    # How many of these lines would the old 3% straightness gate have kept?
    out["would_pass_old_gate"] = int(sum(1 for r in resids if r < 0.03))

    if out["inlier_frac"] < MIN_INLIER_FRAC:
        out["status"] = (f"weak_consensus: only {out['inlier_frac']:.0%} of "
                         f"lines agree")
        return out
    if spread > MAX_DIRECTION_SPREAD_DEG:
        out["status"] = (f"not_one_road: track directions span {spread:.0f} "
                         f"deg, so a single vanishing point does not describe "
                         f"them")
        return out

    out["status"] = "solved"
    return out


def main() -> int:
    prior = {}
    if CALIB.is_file():
        prior = {r["camera"]: r for r in json.loads(CALIB.read_text(encoding="utf-8"))}
    cams = sys.argv[1:] or None

    rows = []
    for p in sorted(GEOM.glob("CAM_*.json")):
        d = json.loads(p.read_text(encoding="utf-8"))
        if not d.get("frame_size"):
            continue
        if cams and d["camera"] not in cams:
            continue
        r = fit(d["camera"], d)
        old = prior.get(d["camera"], {})
        r["old_status"] = old.get("status")
        r["old_vp1"] = old.get("vp1")
        rows.append(r)

    print("%-9s %7s %7s %8s %8s %9s %18s  %s"
          % ("camera", "tracks", "lines", "old-gate", "inliers", "spread",
             "vp1", "status"))
    print("-" * 104)
    for r in rows:
        print("%-9s %7d %7d %8s %8s %9s %18s  %s"
              % (r["camera"], r["tracks_total"], r["lines_used"],
                 str(r.get("would_pass_old_gate", "-")),
                 str(r.get("inliers", "-")),
                 str(r.get("direction_spread_deg", "-")),
                 str(r.get("vp1", "-")), r["status"]))

    rescued = [r for r in rows if r["status"] == "solved"
               and r.get("old_status") in ("insufficient_tracks", None)]
    print(f"\n{sum(1 for r in rows if r['status'] == 'solved')} camera(s) with a "
          f"robust VP1")
    if rescued:
        print("newly reachable (had no VP1 before):")
        for r in rescued:
            print(f"   {r['camera']}: {r['lines_used']} lines, "
                  f"{r['inliers']} inliers, only {r['would_pass_old_gate']} "
                  f"would have passed the old straightness gate")

    OUT.write_text(json.dumps(rows, indent=1), encoding="utf-8")
    print(f"\nwrote {OUT}")
    print("A VP1 is not a calibration. These feed the height and focal steps,"
          "\nwhich have their own gates, and nothing reaches the database here.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
