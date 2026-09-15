"""backend/scripts/fit_focal_from_legs.py — Phase 2A: recover focal length from
the network's own re-identifications, bypassing the second vanishing point.

THE SUBSTITUTION
  With VP1 and a camera height, the ground mapping is

      X = h (u - vpx) / dv        lateral    — no f in it
      Y = f h / dv                along road — one unknown

  Tier 1 gets f from a second, orthogonal vanishing point. Fifteen of sixteen
  cameras fail that: most watch a single traffic direction, and the edge-derived
  VP2 lands non-orthogonal. But f is a single scalar, so ONE independent speed
  measurement determines it — and the network already produces those. A plate
  read at two cameras gives distance-over-time from GPS and clocks alone,
  touching no camera model.

  So a geometric constraint is traded for a data-driven one, and unlike a site
  survey it improves on its own as traffic accumulates.

SPEED IS NOT LINEAR IN f
  Worth stating because it invites the wrong estimator. Displacement between
  two track points is

      d(f) = sqrt( dX^2 + f^2 dY1^2 ),   dY1 = delta( h / dv )

  which is sqrt(a + b f^2), not a straight line. It is monotonic in f, so it is
  solved by bisection rather than fitted by regression. On a camera looking
  along the road dX is small and the relation is nearly linear — but "nearly"
  is not a reason to use a model that is wrong at the edges.

THE ASSUMPTION, STATED
  A leg speed is an average over the whole corridor; the camera measures speed
  where it stands. Solving f by matching them assumes the median vehicle is
  travelling at the camera about as fast as it travels over the corridor. That
  holds on a free-flowing link and fails beside a signal. Where a camera also
  has a VP2-derived f, the two are compared — that comparison is the only
  direct test of this assumption available.

THREE GUARDS AGAINST A SMALL SAMPLE
  robust solve     f is solved on many bootstrap resamples of the legs, and the
                   spread of those solutions is reported as a confidence
                   interval. A single mis-matched plate — two vehicles read as
                   one — then shows up as a wide interval instead of a
                   confident wrong number.
  conditioning     f is constrained mostly by tracks seen at different
                   distances. If every track sits at a similar dv the fit is
                   ill-conditioned however many legs there are, so the dv
                   spread is measured and reported.
  held-out legs    legs are split; f is solved on one part and checked against
                   the other. Fitting and validating on the same legs would
                   confirm the method against itself.

Run:  python -m backend.scripts.fit_focal_from_legs [camera ...]
"""
from __future__ import annotations

import json
import math
import sqlite3
import sys
from collections import defaultdict
from datetime import datetime
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

CALIB = ROOT / "output" / "camera_calibration" / "calibration.json"
VP2 = ROOT / "output" / "camera_calibration" / "vp2_from_edges.json"
GEOM = ROOT / "output" / "road_geometry"
DB = ROOT / "output" / "sentinel.db"
OUT = ROOT / "reports" / "focal_from_legs_20260906.json"

NATIVE_FPS = 25.0
MIN_TRACK_SECONDS = 0.6
# A leg outside this is a mis-match, not a vehicle.
#
# The ceiling is 110 because Indian expressway limits run 100-120 and these are
# urban and inter-urban links, not because it produces a nicer answer. It does
# produce one, and that is worth recording as evidence rather than as the
# reason: two legs at 141 and 147 km/h — two different vehicles read as the
# same plate, creating an impossibly fast journey — were on their own widening
# CAM_08's focal-length interval from 13% of the estimate to 153%. Dropping
# them changed f from 1190 px to 1087 px and the interval from
# [1056, 2874] to [1037, 1181].
#
# The bootstrap is what made that visible. A point estimate would have reported
# 1190 px with no sign that two rows were carrying it.
LEG_MIN_KMH, LEG_MAX_KMH = 5.0, 110.0
MIN_LEGS = 8
HOLDOUT_FRAC = 0.35
MIN_HOLDOUT = 3
BOOTSTRAP = 400
# f is searched over a range that covers every plausible CCTV lens on a
# 1920-wide frame (FOV roughly 15 to 150 degrees).
F_LO, F_HI = 300.0, 8000.0
SPEED_SANE_MIN, SPEED_SANE_MAX = 2.0, 200.0


def legs_by_camera() -> dict:
    """Per-camera leg speeds, sourced from the shared corridor extractor.

    The filtering rules — minimum and maximum leg distance, plausible speed
    band — live in `corridor_legs` so that this script and anything else that
    consumes legs cannot drift apart on what counts as one. This function
    flattens corridors into a single list per camera, which is only safe when
    `camera_profile(...)["single_median_is_safe"]` says so; `fit_camera` checks
    that before using the result.
    """
    from backend.services.corridor_legs import by_camera, extract_legs
    per_cam = by_camera(extract_legs(DB))
    return {cam: [l.kmh for ls in cor.values() for l in ls]
            for cam, cor in per_cam.items()}


def track_terms(cam: str, vpx: float, vpy: float, h: float):
    """Per track: (dX, dY1, dt) so speed(f) = hypot(dX, f*dY1)/dt.

    Also returns the dv values, which say how well conditioned f is.
    """
    p = GEOM / f"{cam}.json"
    if not p.is_file():
        return [], []
    tracks = json.loads(p.read_text(encoding="utf-8"))["tracks"]
    terms, dvs = [], []
    for pts in tracks.values():
        if len(pts) < 3:
            continue
        a, b = pts[0], pts[-1]
        dt = (b["frame"] - a["frame"]) / NATIVE_FPS
        if dt < MIN_TRACK_SECONDS:
            continue
        dva, dvb = a["v"] - vpy, b["v"] - vpy
        if dva <= 1.0 or dvb <= 1.0:
            continue
        dX = h * (b["u"] - vpx) / dvb - h * (a["u"] - vpx) / dva
        dY1 = h / dvb - h / dva
        terms.append((dX, dY1, dt))
        dvs.extend([dva, dvb])
    return terms, dvs


def median_speed(terms, f: float) -> float:
    v = [math.hypot(dX, f * dY1) / dt * 3.6 for dX, dY1, dt in terms]
    v = [x for x in v if SPEED_SANE_MIN <= x <= SPEED_SANE_MAX]
    if not v:
        return 0.0
    v.sort()
    return v[len(v) // 2]


def solve_f(terms, target_kmh: float) -> float | None:
    """Smallest f whose median track speed matches the target. Monotonic in f."""
    lo, hi = F_LO, F_HI
    if median_speed(terms, lo) > target_kmh or median_speed(terms, hi) < target_kmh:
        return None
    for _ in range(60):
        mid = 0.5 * (lo + hi)
        if median_speed(terms, mid) < target_kmh:
            lo = mid
        else:
            hi = mid
    return 0.5 * (lo + hi)


def fit_camera(cam: str, prior: dict, legs: list, vp2fit: dict | None) -> dict:
    r = {"camera": cam, "legs_total": len(legs)}
    vp1, h = prior.get("vp1"), prior.get("height_m")
    if not vp1 or not h:
        r["status"] = "no_vp1_or_height"
        return r
    if len(legs) < MIN_LEGS:
        r["status"] = f"too_few_legs (have {len(legs)}, need {MIN_LEGS})"
        return r

    # A camera fed by corridors of different character has no single "median
    # leg speed" to solve against — the number would depend on which corridor
    # happened to contribute more legs. Refuse rather than average them.
    from backend.services.corridor_legs import (by_camera, camera_profile,
                                                extract_legs)
    prof = camera_profile(cam, by_camera(extract_legs(DB)).get(cam, {}))
    r["n_corridors"] = prof["n_corridors"]
    r["dominant_corridor_share"] = prof["dominant_share"]
    r["corridor_speed_spread"] = prof["corridor_speed_spread"]
    r["corridors"] = prof["corridors"]
    if not prof["single_median_is_safe"]:
        r["status"] = ("corridors_disagree: speeds differ %.1fx across this "
                       "camera's corridors — solve per corridor"
                       % prof["corridor_speed_spread"])
        return r

    terms, dvs = track_terms(cam, vp1[0], vp1[1], float(h))
    r["tracks_usable"] = len(terms)
    if len(terms) < 20:
        r["status"] = "too_few_usable_tracks"
        return r

    # Conditioning: f is pinned down by tracks seen at DIFFERENT distances.
    dvs = np.array(dvs)
    spread = float(np.percentile(dvs, 90) / max(np.percentile(dvs, 10), 1e-6))
    r["dv_p10_p50_p90"] = [round(float(np.percentile(dvs, q)), 1)
                           for q in (10, 50, 90)]
    r["dv_spread_ratio"] = round(spread, 2)
    if spread < 1.5:
        r["status"] = "ill_conditioned: all tracks at a similar distance"
        return r

    rng = np.random.default_rng(11)
    legs_arr = np.array(sorted(legs))
    n_hold = max(MIN_HOLDOUT, int(len(legs_arr) * HOLDOUT_FRAC))
    idx = rng.permutation(len(legs_arr))
    hold = legs_arr[idx[:n_hold]]
    fit = legs_arr[idx[n_hold:]]
    r["legs_fit"], r["legs_holdout"] = len(fit), len(hold)
    if len(fit) < 4:
        r["status"] = "too_few_legs_after_holdout"
        return r

    # Bootstrap the solve so a single bad leg widens an interval instead of
    # moving a point estimate.
    fs = []
    for _ in range(BOOTSTRAP):
        samp = rng.choice(fit, size=len(fit), replace=True)
        f = solve_f(terms, float(np.median(samp)))
        if f is not None:
            fs.append(f)
    if len(fs) < BOOTSTRAP * 0.5:
        r["status"] = "solve_failed_on_most_resamples"
        return r
    fs = np.array(fs)
    f_hat = float(np.median(fs))
    r["focal_px"] = round(f_hat, 1)
    r["focal_ci95"] = [round(float(np.percentile(fs, 2.5)), 1),
                       round(float(np.percentile(fs, 97.5)), 1)]
    r["focal_ci_width_frac"] = round(
        float((np.percentile(fs, 97.5) - np.percentile(fs, 2.5)) / f_hat), 3)
    W = prior["frame_size"][0]
    r["fov_deg"] = round(2 * math.degrees(math.atan(W / (2 * f_hat))), 1)

    # Held-out check: legs never seen by the solve.
    own = median_speed(terms, f_hat)
    r["camera_median_kmh_at_f"] = round(own, 1)
    r["holdout_median_kmh"] = round(float(np.median(hold)), 1)
    r["holdout_ratio"] = round(own / max(float(np.median(hold)), 1e-6), 2)

    if vp2fit and vp2fit.get("focal_px"):
        r["vp2_focal_px"] = vp2fit["focal_px"]
        r["leg_vs_vp2_ratio"] = round(f_hat / vp2fit["focal_px"], 3)

    r["status"] = "solved"
    return r


def main() -> int:
    priors = {x["camera"]: x for x in json.loads(CALIB.read_text(encoding="utf-8"))}
    vp2 = {}
    if VP2.is_file():
        vp2 = {x["camera"]: x for x in json.loads(VP2.read_text(encoding="utf-8"))}
    legs = legs_by_camera()
    cams = sys.argv[1:] or sorted(priors)

    out = []
    for cam in cams:
        out.append(fit_camera(cam, priors.get(cam, {}), legs.get(cam, []),
                              vp2.get(cam)))

    print("%-9s %6s %8s %10s %18s %8s %8s  %s"
          % ("camera", "legs", "tracks", "focal px", "95% CI", "FOV",
             "hold r", "status"))
    print("-" * 104)
    for r in out:
        ci = (f"[{r['focal_ci95'][0]:.0f}, {r['focal_ci95'][1]:.0f}]"
              if r.get("focal_ci95") else "-")
        print("%-9s %6d %8s %10s %18s %8s %8s  %s"
              % (r["camera"], r.get("legs_total", 0),
                 str(r.get("tracks_usable", "-")), str(r.get("focal_px", "-")),
                 ci, str(r.get("fov_deg", "-")),
                 str(r.get("holdout_ratio", "-")), r["status"]))

    solved = [r for r in out if r["status"] == "solved"]
    print(f"\n{len(solved)} camera(s) solved from legs")
    for r in solved:
        if "leg_vs_vp2_ratio" in r:
            print(f"   {r['camera']}: leg-derived {r['focal_px']} px vs "
                  f"VP2-derived {r['vp2_focal_px']} px "
                  f"(ratio {r['leg_vs_vp2_ratio']}) "
                  f"<- independent check of the whole method")

    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text(json.dumps(out, indent=1), encoding="utf-8")
    print(f"\nwritten -> {OUT}")
    print("Nothing promoted to camera_calibrations.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
