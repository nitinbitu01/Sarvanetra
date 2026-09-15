"""Estimate a real ground-plane homography from the footage itself.

The calibrations this replaces were fitted to four image points and four world
points typed into a source file by hand — a symmetrical trapezoid mapped to a
symmetrical rectangle, the same shape reused for a bridge deck and a market
junction. Four points determine an 8-DOF homography exactly, so their reported
0.017 m reprojection error was zero by construction, and held_out_error_m was
a copy of it.

What is actually available without a survey team:

  vanishing point, from trajectories
      Vehicles travel along the road, so the lines their tracks sweep out meet
      at the road's vanishing point. Robustly estimated over many tracks, this
      fixes the direction of travel in the image.

  camera height, from known vehicle widths
      For a camera with negligible roll looking at a plane, a vehicle of real
      width W seen w pixels wide with its wheels at image row v satisfies
          h = W (v - vp_y) / w
      where h is the camera's height above the road. Focal length cancels,
      which is why width alone cannot give it.

  focal length, from two orthogonal vanishing points
      This is the part that is genuinely not free. At a junction, traffic runs
      in two perpendicular directions, and each stream yields its own
      vanishing point. For a camera whose principal point is at the image
      centre, two orthogonal directions constrain the focal length:
          f^2 = -(vp1 - c) . (vp2 - c)
      A camera with only one traffic direction has no second vanishing point,
      and no amount of processing invents one. Those cameras get a lateral
      scale and no metric depth, and are recorded as such rather than being
      given a matrix that looks like the others.

Every estimate here is reported with the spread of the observations behind it,
and a camera whose numbers are physically implausible — a 30 m mounting pole,
a negative height — is rejected rather than stored.

Run:  python -m backend.scripts.fit_camera_calibration [--apply]
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
OUT = ROOT / "output" / "camera_calibration"

CLASS_WIDTH_M = {2: 1.75, 5: 2.60, 7: 2.45}

# A pole-mounted traffic camera sits between about 3 m (a shopfront bracket)
# and 12 m (a gantry). Anything outside that is an estimation failure, not an
# unusual installation.
MIN_HEIGHT_M, MAX_HEIGHT_M = 3.0, 12.0
# Two vanishing points must be far apart in direction for the orthogonality
# constraint to be conditioned; near-parallel streams give a focal length that
# swings wildly with noise.
MIN_VP_SEPARATION_DEG = 25.0


def track_line(points: list) -> tuple | None:
    """Homogeneous line through a track's ground-contact points."""
    p = np.array([[q["u"], q["v"]] for q in points], dtype=float)
    if len(p) < 3:
        return None
    span = float(np.hypot(*(p[-1] - p[0])))
    if span < 40.0:                      # barely moved; direction is noise
        return None
    centred = p - p.mean(0)
    _, s, vt = np.linalg.svd(centred, full_matrices=False)
    if s[1] / max(span, 1e-6) > 0.03:    # not straight enough to be a road
        return None
    direction = vt[0]
    normal = np.array([-direction[1], direction[0]])
    c = -normal @ p.mean(0)
    return np.array([normal[0], normal[1], c]), span, direction


def vanishing_point(lines: list, weights: list) -> np.ndarray:
    """Least-squares intersection of many lines, weighted by track length.

    A long track constrains its direction far better than a short one, so it
    should count for more.
    """
    A = np.array([[l[0], l[1]] for l in lines], dtype=float)
    b = -np.array([l[2] for l in lines], dtype=float)
    w = np.sqrt(np.array(weights, dtype=float))
    sol, *_ = np.linalg.lstsq(A * w[:, None], b * w, rcond=None)
    return sol


def cluster_directions(dirs: list, spans: list) -> list:
    """Split tracks into traffic streams by direction.

    Directions are taken modulo 180 degrees: a road carries traffic both ways
    and both directions share one vanishing point.
    """
    ang = np.array([np.degrees(np.arctan2(d[1], d[0])) % 180.0 for d in dirs])
    order = np.argsort(ang)
    clusters, current = [], [order[0]]
    for prev, idx in zip(order, order[1:]):
        if ang[idx] - ang[prev] <= 20.0:
            current.append(idx)
        else:
            clusters.append(current)
            current = [idx]
    clusters.append(current)
    # The set wraps at 180 degrees, so the first and last clusters may be one.
    if len(clusters) > 1 and (ang[clusters[0][0]] + 180.0
                              - ang[clusters[-1][-1]]) <= 20.0:
        clusters[0] = clusters[0] + clusters.pop()
    return sorted(clusters, key=lambda c: -sum(spans[i] for i in c))


def estimate_height(tracks: dict, vp: np.ndarray) -> tuple:
    """Camera height above the road, from vehicles of regulated width.

    The detector's box is axis-aligned, so its width equals the vehicle's true
    width only when the vehicle is seen end-on. A car crossing obliquely
    presents a box spanning part of its length as well, and using that width
    makes the camera look lower than it is — the first run of this estimator
    returned 0.41 m for CAM_18 and 1.25 m for CAM_02, both of which are
    mounted well above head height.

    A vehicle is seen end-on when it sits near the vanishing point's vertical
    axis, since that is the direction it is travelling. Restricting to
    observations where the lateral offset is small compared with the distance
    below the horizon keeps the ones whose box width is the vehicle width.
    """
    est = []
    rejected = 0
    for pts in tracks.values():
        for q in pts:
            W = CLASS_WIDTH_M.get(q["cls"])
            if W is None or q["w"] < 24:
                continue
            dv = q["v"] - vp[1]
            if dv <= 1.0:                # at or above the horizon
                continue
            # tan of the angle off the viewing axis, on the ground plane.
            if abs(q["u"] - vp[0]) / dv > 0.45:      # beyond ~24 degrees
                rejected += 1
                continue
            est.append(W * dv / q["w"])
    if len(est) < 20:
        return None, 0.0, len(est)
    a = np.array(est)
    # The median resists mis-classified vehicles; the interquartile spread
    # says whether the observations agree at all.
    return (float(np.median(a)),
            float(np.subtract(*np.percentile(a, [75, 25]))), len(a))


def fit_camera(cam: str, data: dict) -> dict:
    tracks = data["tracks"]
    W, H = data["frame_size"]
    centre = np.array([W / 2.0, H / 2.0])

    lines, spans, dirs, keys = [], [], [], []
    for k, pts in tracks.items():
        r = track_line(pts)
        if r is None:
            continue
        line, span, d = r
        lines.append(line)
        spans.append(span)
        dirs.append(d)
        keys.append(k)

    result = {"camera": cam, "frame_size": [W, H], "straight_tracks": len(lines)}
    if len(lines) < 8:
        result["status"] = "insufficient_tracks"
        return result

    clusters = cluster_directions(dirs, spans)
    result["streams"] = [len(c) for c in clusters]

    vp1 = vanishing_point([lines[i] for i in clusters[0]],
                          [spans[i] for i in clusters[0]])
    result["vp1"] = [round(float(v), 1) for v in vp1]

    height, iqr, n_obs = estimate_height(
        {keys[i]: tracks[keys[i]] for i in clusters[0]}, vp1)
    result["height_m"] = None if height is None else round(height, 2)
    result["height_iqr_m"] = round(iqr, 2)
    result["height_observations"] = n_obs

    if height is None:
        result["status"] = "insufficient_width_observations"
        return result
    if not (MIN_HEIGHT_M <= height <= MAX_HEIGHT_M):
        result["status"] = "implausible_height"
        return result
    # A spread wider than the estimate itself means the observations disagree.
    if iqr > height:
        result["status"] = "height_estimate_unstable"
        return result

    # Second traffic stream, for focal length.
    vp2 = None
    if len(clusters) > 1 and len(clusters[1]) >= 4:
        vp2 = vanishing_point([lines[i] for i in clusters[1]],
                              [spans[i] for i in clusters[1]])
        a1 = np.degrees(np.arctan2(*(vp1 - centre)[::-1]))
        a2 = np.degrees(np.arctan2(*(vp2 - centre)[::-1]))
        sep = abs((a1 - a2 + 90) % 180 - 90)
        result["vp2"] = [round(float(v), 1) for v in vp2]
        result["vp_separation_deg"] = round(float(sep), 1)
        if sep < MIN_VP_SEPARATION_DEG:
            vp2 = None
            result["vp2_rejected"] = "streams too close in direction"

    if vp2 is not None:
        # Orthogonal vanishing points: f^2 = -(vp1-c).(vp2-c)
        dot = float((vp1 - centre) @ (vp2 - centre))
        if dot < 0:
            f = float(np.sqrt(-dot))
            result["focal_px"] = round(f, 1)
            result["status"] = "full"
            # Ground-plane mapping, metres, origin under the camera:
            #   X = h (u - vpx) / (v - vpy)      lateral
            #   Y = f h / (v - vpy)              along the road
            result["mapping"] = {
                "type": "planar_vp",
                "vp_x": float(vp1[0]), "vp_y": float(vp1[1]),
                "height_m": float(height), "focal_px": f,
            }
            return result
        result["vp2_rejected"] = "vanishing points not orthogonal"

    # One traffic direction only: lateral scale is metric, depth is not.
    result["status"] = "lateral_only"
    result["mapping"] = {
        "type": "lateral_only",
        "vp_x": float(vp1[0]), "vp_y": float(vp1[1]),
        "height_m": float(height),
    }
    return result


def main() -> int:
    if not GEOM.is_dir():
        print("run collect_road_geometry first")
        return 1
    OUT.mkdir(parents=True, exist_ok=True)

    results = []
    for path in sorted(GEOM.glob("CAM_*.json")):
        data = json.loads(path.read_text(encoding="utf-8"))
        if not data.get("frame_size"):
            continue
        results.append(fit_camera(data["camera"], data))

    print(f"{'camera':<9}{'tracks':>7}{'streams':>10}{'height m':>10}"
          f"{'IQR':>7}{'n':>6}{'focal px':>10}  status")
    print("-" * 76)
    for r in sorted(results, key=lambda x: x["camera"]):
        print(f"{r['camera']:<9}{r.get('straight_tracks', 0):>7}"
              f"{str(r.get('streams', '-')):>10}"
              f"{str(r.get('height_m', '-')):>10}"
              f"{str(r.get('height_iqr_m', '-')):>7}"
              f"{r.get('height_observations', 0):>6}"
              f"{str(r.get('focal_px', '-')):>10}  {r['status']}")

    full = [r for r in results if r["status"] == "full"]
    lateral = [r for r in results if r["status"] == "lateral_only"]
    print(f"\n{len(full)} cameras with a metric ground plane "
          f"(speed in km/h is meaningful)")
    print(f"{len(lateral)} with lateral scale only "
          f"(one traffic direction, so no metric depth)")
    print(f"{len(results)-len(full)-len(lateral)} not calibratable from their "
          f"own footage")

    (OUT / "calibration.json").write_text(json.dumps(results, indent=1),
                                          encoding="utf-8")
    print(f"\nwrote {OUT/'calibration.json'}")
    print("""
Nothing is written to the database yet. A homography that has not been
validated against an independent measurement is exactly what was removed, so
the next step is to check these against vehicle speeds derived from
camera-to-camera travel time, which depends only on GPS and timestamps.""")
    return 0


if __name__ == "__main__":
    sys.exit(main())
