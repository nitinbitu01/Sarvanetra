"""backend/services/trajectory_analyzer.py — HOW someone stayed, not just how long.

WHY DURATION ALONE IS NOT ENOUGH
  The zone engine answers "is this a place where waiting is normal". It cannot
  separate two people standing in the SAME place for the SAME duration:

      waiting for a lift     stands still, faces one way, then leaves
      casing a location      paces, circles, drifts off and returns

  Both hold position for ten minutes. Only the second is worth an officer's
  attention. The difference is in the SHAPE of the path, which is exactly what
  the loitering window already stores and then throws away - it keeps the
  points only to compute a max-distance test.

WHAT IS MEASURED, AND WHAT EACH ONE MEANS
  displacement_ratio  path walked / straight-line start-to-end. Someone
                      standing still scores near 1. Someone pacing back and
                      forth covers a lot of ground and ends where they began,
                      scoring high.
  direction_changes   count of turns sharper than 45 degrees. Pacing and
                      scanning produce many; walking through produces few.
  circular_score      how constant the distance from the path's centre is.
                      Walking a loop scores high; standing still is excluded
                      by a minimum-radius check so it does not read as a
                      degenerate circle.
  return_count        distinct grid cells revisited three or more times.
                      Approach-retreat-approach is a documented precursor to
                      vehicle break-ins, and is invisible to a dwell timer.
  area_coverage       bounding-box area of the whole path, in px^2. Small area
                      over long duration means genuinely rooted to one spot.

HONEST LIMITS
  These are geometric descriptors, not a trained classifier, and no thresholds
  here have been validated against labelled loitering footage from this fleet.
  They are returned as measurements for the scorer to weigh, with the raw
  values kept in the alert so a reviewer can judge whether the signal was real
  rather than trusting a single fused number.
"""
from __future__ import annotations

import math
from dataclasses import asdict, dataclass


@dataclass
class TrajectoryFeatures:
    n_points: int
    displacement_ratio: float
    direction_changes: int
    circular_score: float
    return_count: int
    area_coverage_px2: float
    path_length_px: float

    def to_dict(self) -> dict:
        return {k: (round(v, 3) if isinstance(v, float) else v)
                for k, v in asdict(self).items()}


# Below this many samples the descriptors are noise, not signal.
MIN_POINTS = 8


def analyse(points: list[tuple[float, float]]) -> TrajectoryFeatures | None:
    """Describe the shape of a track's path. None if too few samples."""
    if not points or len(points) < MIN_POINTS:
        return None

    path = 0.0
    for i in range(len(points) - 1):
        path += math.dist(points[i], points[i + 1])
    direct = math.dist(points[0], points[-1])

    # Guard the ratio: a person who ends exactly where they started would
    # otherwise divide by ~0 and produce a meaningless spike. One pixel is
    # below tracking precision, so it is a safe floor rather than a fudge.
    ratio = path / max(direct, 1.0)

    turns = 0
    for i in range(1, len(points) - 1):
        ax, ay = points[i][0] - points[i - 1][0], points[i][1] - points[i - 1][1]
        bx, by = points[i + 1][0] - points[i][0], points[i + 1][1] - points[i][1]
        na = math.hypot(ax, ay)
        nb = math.hypot(bx, by)
        # Sub-pixel jitter has no meaningful direction; counting it would make
        # a perfectly still person look like they were pacing.
        if na < 1.5 or nb < 1.5:
            continue
        cosang = max(-1.0, min(1.0, (ax * bx + ay * by) / (na * nb)))
        if math.degrees(math.acos(cosang)) > 45.0:
            turns += 1

    cx = sum(p[0] for p in points) / len(points)
    cy = sum(p[1] for p in points) / len(points)
    radii = [math.dist(p, (cx, cy)) for p in points]
    mean_r = sum(radii) / len(radii)
    if mean_r > 12.0:                      # else it is standing still, not a loop
        var = sum((r - mean_r) ** 2 for r in radii) / len(radii)
        circular = max(0.0, 1.0 - (math.sqrt(var) / mean_r))
    else:
        circular = 0.0

    # Revisits: coarse grid so ordinary jitter inside one spot does not read as
    # "returning". A cell is only a return once it has been entered 3 times.
    cells: dict[tuple[int, int], int] = {}
    returns = 0
    for x, y in points:
        key = (int(x // 40), int(y // 40))
        cells[key] = cells.get(key, 0) + 1
        if cells[key] == 3:
            returns += 1

    xs = [p[0] for p in points]
    ys = [p[1] for p in points]
    area = (max(xs) - min(xs)) * (max(ys) - min(ys))

    return TrajectoryFeatures(
        n_points=len(points),
        displacement_ratio=ratio,
        direction_changes=turns,
        circular_score=circular,
        return_count=returns,
        area_coverage_px2=area,
        path_length_px=path,
    )


def suspicion_bonus(f: TrajectoryFeatures | None) -> tuple[float, list[str]]:
    """Extra danger score from path shape, with the reason for each point.

    Returns (bonus, reasons). Reasons are carried into the alert so a reviewer
    sees WHY the score rose rather than an unexplained number - an alert an
    officer cannot interrogate is one they will learn to ignore.

    Thresholds are starting values chosen from the geometry, NOT tuned against
    labelled data. They should be revised once false positives are measured on
    real footage.
    """
    if f is None:
        return 0.0, []

    bonus = 0.0
    why: list[str] = []

    # Walking far but ending where you started: pacing, not waiting.
    if f.displacement_ratio > 6.0 and f.path_length_px > 150:
        bonus += 2.0
        why.append(f"pacing — walked {f.path_length_px:.0f}px but returned to "
                   f"start (ratio {f.displacement_ratio:.1f})")

    if f.direction_changes >= 8:
        bonus += 1.5
        why.append(f"{f.direction_changes} sharp direction changes — scanning "
                   f"or restless movement")

    if f.circular_score > 0.72:
        bonus += 2.0
        why.append(f"circling — path stays a constant distance from its centre "
                   f"({f.circular_score:.2f})")

    # Approach, leave, come back. A dwell timer cannot see this at all.
    if f.return_count >= 3:
        bonus += 2.5
        why.append(f"returned to the same spot {f.return_count} times")
    elif f.return_count == 2:
        bonus += 1.0
        why.append("returned to the same spot twice")

    return round(bonus, 2), why
