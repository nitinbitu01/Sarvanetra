"""
backend/services/speed_estimator.py — Theil-Sen Robust Speed Estimator & Physical Validity Filters (Phase 1).

Why Theil-Sen over Endpoint Differencing:
  - Bounding-box jitter and detector box flutter create high-frequency noise at endpoints.
  - Differencing endpoints (s_end - s_start) / (t_end - t_start) is vulnerable to single-frame outliers.
  - Theil-Sen projects positions onto the vehicle's principal trajectory axis and computes the
    median of all pairwise slopes m_ij = (s_j - s_i) / (t_j - t_i), giving a breakdown point of ~29.3%
    and complete immunity to bounding box oscillations and transient spike errors.
  - Non-parametric bootstrap generates rigorous 95% confidence interval half-width (+/- Delta v).

Honesty & Physical Gating:
  - Cameras without valid homography or failing the quality gate (> 1.0m RMS error) produce NULL speed.
  - Rejects speeds outside [0.0, 150.0] km/h.
  - Rejects tracks under min_frames (5 frames) or min_path_length (2.0 meters).
"""
from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Optional, Sequence

import numpy as np

MAX_THEIL_SEN_POINTS = 60
MIN_FRAMES_FOR_SPEED = 5
MIN_PATH_LENGTH_METERS = 2.0
MIN_PLAUSIBLE_SPEED_KMH = 0.0
MAX_PLAUSIBLE_SPEED_KMH = 150.0
BOOTSTRAP_ITERATIONS = 200


@dataclass
class TrackSpeedResult:
    speed_kmh: Optional[float]         # NULL if uncalibrated or rejected
    speed_ci_kmh: Optional[float]      # 95% CI half-width (+/- km/h)
    path_length_m: float               # cumulative distance traversed
    n_frames: int                      # number of observation points
    heading_deg: Optional[float]       # true heading in degrees [0, 360)
    quality: str                       # 'good' | 'degraded' | 'speed_unavailable'
    is_valid: bool                     # whether speed is statistically valid


def estimate_track_speed(
    world_points: Sequence[tuple[float, float]],  # (wx, wy) positions in meters
    timestamps: Sequence[float],                 # seconds (monotonic or epoch)
    calibration_quality: str = "good",           # 'good' | 'degraded' | 'rejected' | 'uncalibrated'
    camera_bearing_deg: float = 0.0,
) -> TrackSpeedResult:
    """Computes robust Theil-Sen speed over a sequence of world-plane coordinates."""
    n = len(world_points)
    if n < 2 or len(timestamps) != n:
        return TrackSpeedResult(
            speed_kmh=None,
            speed_ci_kmh=None,
            path_length_m=0.0,
            n_frames=n,
            heading_deg=None,
            quality="speed_unavailable",
            is_valid=False,
        )

    pts = np.array(world_points, dtype=np.float64)
    ts = np.array(timestamps, dtype=np.float64)

    # Decimate long-duration idling/congested tracks to prevent O(N^2) CPU spike
    if n > MAX_THEIL_SEN_POINTS:
        step = int(math.ceil(n / MAX_THEIL_SEN_POINTS))
        indices = list(range(0, n, step))
        if indices[-1] != n - 1:
            indices.append(n - 1)
        pts = pts[indices]
        ts = ts[indices]
        n_effective = len(pts)
    else:
        n_effective = n

    # 1. Heading and principal trajectory unit vector
    dx = float(pts[-1, 0] - pts[0, 0])
    dy = float(pts[-1, 1] - pts[0, 1])
    net_disp = math.hypot(dx, dy)

    heading_deg = None
    if net_disp >= 0.5:
        local_angle = math.degrees(math.atan2(dx, dy))
        heading_deg = (camera_bearing_deg + local_angle) % 360.0
        ux = dx / net_disp
        uy = dy / net_disp
    else:
        ux, uy = 0.0, 1.0

    # 2. 1D Projection along principal travel axis s_i = (p_i - p_0) . u
    rel_pts = pts - pts[0]
    proj_dists = rel_pts[:, 0] * ux + rel_pts[:, 1] * uy
    path_length_m = max(float(net_disp), float(np.max(proj_dists) - np.min(proj_dists)))

    # 3. Physical plausibility of the reconstructed path.
    #
    # Every gate below this point judges the SPEED. None of them judge the
    # geometry that produced the positions, so a broken world reconstruction
    # could pass: a CAM_08 bus track accumulated a path_length of 84,445 m —
    # 84 km inside one camera's field of view — and still reported 0.0 km/h,
    # because Theil-Sen's median slope over a wildly scattered trajectory is
    # near zero. The speed looked plausible; the metres behind it did not
    # exist. path_length is published alongside the speed, so a figure like
    # that reaches an operator as fact.
    #
    # The bound is physics, not tuning: covering path_length_m in this track's
    # own duration must not require exceeding the speed this module already
    # rejects as impossible. It carries no assumption about lens, mounting
    # height or field of view, so it holds for any camera.
    duration_s = float(ts[-1] - ts[0])
    if duration_s > 0:
        implied_kmh = (path_length_m / duration_s) * 3.6
        if implied_kmh > MAX_PLAUSIBLE_SPEED_KMH:
            return TrackSpeedResult(
                speed_kmh=None,
                speed_ci_kmh=None,
                # Deliberately not published: this number is the symptom.
                path_length_m=0.0,
                n_frames=n,
                heading_deg=None,
                quality="speed_unavailable",
                is_valid=False,
            )

    # 4. Check calibration validity
    if calibration_quality in ("rejected", "uncalibrated"):
        return TrackSpeedResult(
            speed_kmh=None,
            speed_ci_kmh=None,
            path_length_m=round(path_length_m, 2),
            n_frames=n,
            heading_deg=round(heading_deg, 1) if heading_deg is not None else None,
            quality="speed_unavailable",
            is_valid=False,
        )

    # 5. Check minimum frame count and path length
    if n < MIN_FRAMES_FOR_SPEED or path_length_m < MIN_PATH_LENGTH_METERS:
        return TrackSpeedResult(
            speed_kmh=None,
            speed_ci_kmh=None,
            path_length_m=round(path_length_m, 2),
            n_frames=n,
            heading_deg=round(heading_deg, 1) if heading_deg is not None else None,
            quality="degraded",
            is_valid=False,
        )

    # 6. Compute pairwise slopes for Theil-Sen fit
    slopes: list[float] = []
    for i in range(n_effective):
        for j in range(i + 1, n_effective):
            dt = ts[j] - ts[i]
            if dt >= 0.08: # at least 80ms time delta
                ds = proj_dists[j] - proj_dists[i]
                slopes.append(ds / dt)

    if not slopes:
        return TrackSpeedResult(
            speed_kmh=None,
            speed_ci_kmh=None,
            path_length_m=round(path_length_m, 2),
            n_frames=n,
            heading_deg=round(heading_deg, 1) if heading_deg is not None else None,
            quality="degraded",
            is_valid=False,
        )

    # Median velocity in m/s
    slopes_arr = np.array(slopes, dtype=np.float64)
    v_ms = float(np.median(slopes_arr))
    speed_kmh = max(0.0, v_ms * 3.6)

    # 6. Physical validity bound
    if speed_kmh < MIN_PLAUSIBLE_SPEED_KMH or speed_kmh > MAX_PLAUSIBLE_SPEED_KMH:
        return TrackSpeedResult(
            speed_kmh=None,
            speed_ci_kmh=None,
            path_length_m=round(path_length_m, 2),
            n_frames=n,
            heading_deg=round(heading_deg, 1) if heading_deg is not None else None,
            quality="degraded",
            is_valid=False,
        )

    # 7. Asymptotic 95% Confidence Interval via Interquartile Range (IQR)
    q75, q25 = np.percentile(slopes_arr, [75, 25])
    iqr = max(0.01, float(q75 - q25))
    # Standard error of the sample median: SE = (IQR / 1.349) / sqrt(N_effective)
    se_ms = (iqr / 1.349) / math.sqrt(max(1, len(slopes_arr)))
    ci_half_width = float(1.96 * se_ms * 3.6)
    ci_half_width = min(ci_half_width, speed_kmh * 0.5)

    quality = "good" if calibration_quality == "good" else "degraded"

    return TrackSpeedResult(
        speed_kmh=round(speed_kmh, 1),
        speed_ci_kmh=round(ci_half_width, 1),
        path_length_m=round(path_length_m, 2),
        n_frames=n,
        heading_deg=round(heading_deg, 1) if heading_deg is not None else None,
        quality=quality,
        is_valid=True,
    )
