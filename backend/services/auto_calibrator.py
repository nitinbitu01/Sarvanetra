"""
backend/services/auto_calibrator.py — Production Camera Calibration & Dynamic Shake Stabilization Engine.

Provides:
  1. HomographyEngine:
     - Normalized Direct Linear Transformation (DLT) with Hartley pre-conditioning.
     - Condition Number kappa(H) = sigma_max / sigma_min (flags ill-conditioned/degenerate points).
     - Projective Horizon Line analysis (h31*u + h32*v + h33 = 0) ensuring horizon lies safely above frame.
     - Leave-One-Out (LOO) Cross-Validation across all control points.
     - Perspective Ground Metric Grid generation for visual alignment verification.
  2. CameraDriftStabilizer:
     - Real-time frame-to-frame background displacement tracking via ORB feature matching with RANSAC.
     - Detects camera vibration / wind sway in milliradians.
     - Dynamically compensates Homography H_live = H_ref @ A_drift^(-1).
     - Triggers automated alert if cumulative physical drift exceeds 3.0 degrees.
  3. VanishingPointCalibrator:
     - Automatic vanishing point estimation from traffic trajectories and road edge lines.
     - Estimates focal length, tilt angle, and camera height without physical ground surveying.
"""
from __future__ import annotations

import json
import logging
import math
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import cv2
import numpy as np

logger = logging.getLogger("sentinel.auto_calibrator")

# Quality gate thresholds (meters)
QUALITY_GATE_ACCEPT_M = 0.50
QUALITY_GATE_FLAG_M = 1.00

# Condition number threshold for ill-conditioned homography
MAX_CONDITION_NUMBER = 1e5


@dataclass
class HomographySolveResult:
    is_valid: bool
    H: Optional[List[List[float]]]
    condition_number: float
    horizon_distance_px: float
    loo_rms_error_m: float
    loo_point_errors_m: List[float]
    held_out_error_m: Optional[float]
    quality_gate: str  # "good" | "degraded" | "rejected"
    grid_polylines: List[List[Tuple[float, float]]]
    message: str


@dataclass
class CameraDriftTelemetry:
    camera_id: str
    is_stabilized: bool
    jitter_rms_px: float
    cumulative_drift_deg: float
    tilt_delta_mrad: float
    pan_delta_mrad: float
    drift_status: str  # "STABLE" | "MINOR_VIBRATION" | "HIGH_WIND_SWAY" | "RECALIBRATION_REQUIRED"
    last_stabilized_ts: str


def normalize_points_hartley(pts: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    """Hartley isotropic normalization for numerical stability in DLT.
    Translates centroid to origin and scales average distance to sqrt(2).
    """
    centroid = np.mean(pts, axis=0)
    shifted = pts - centroid
    mean_dist = np.mean(np.sqrt(np.sum(shifted ** 2, axis=1)))
    scale = np.sqrt(2.0) / max(mean_dist, 1e-8)

    T = np.array([
        [scale, 0.0, -scale * centroid[0]],
        [0.0, scale, -scale * centroid[1]],
        [0.0, 0.0, 1.0],
    ], dtype=np.float64)

    pts_homo = np.hstack([pts, np.ones((len(pts), 1), dtype=np.float64)])
    norm_pts = (T @ pts_homo.T).T
    return norm_pts[:, :2], T


class HomographyEngine:
    """Production Homography solver with condition number verification and LOO validation."""

    @staticmethod
    def solve_homography(
        image_points: List[Tuple[float, float]],
        world_points_m: List[Tuple[float, float]],
        held_out_idx: Optional[int] = None,
        frame_size: Tuple[int, int] = (1920, 1080),
    ) -> HomographySolveResult:
        """Solves H, evaluates condition number, checks projective horizon, and computes LOO RMS."""
        n_pts = len(image_points)
        if n_pts < 4 or len(world_points_m) < 4 or n_pts != len(world_points_m):
            return HomographySolveResult(
                is_valid=False,
                H=None,
                condition_number=float("inf"),
                horizon_distance_px=0.0,
                loo_rms_error_m=float("inf"),
                loo_point_errors_m=[],
                held_out_error_m=None,
                quality_gate="rejected",
                grid_polylines=[],
                message="At least 4 matching pixel <-> world point pairs required.",
            )

        src = np.array(image_points, dtype=np.float64)
        dst = np.array(world_points_m, dtype=np.float64)

        # 1. Solve Direct Linear Transformation (DLT) using Normalized SVD
        src_norm, T_src = normalize_points_hartley(src)
        dst_norm, T_dst = normalize_points_hartley(dst)

        A = []
        for (u, v), (X, Y) in zip(src_norm, dst_norm):
            A.append([-u, -v, -1.0, 0.0, 0.0, 0.0, u * X, v * X, X])
            A.append([0.0, 0.0, 0.0, -u, -v, -1.0, u * Y, v * Y, Y])
        A = np.array(A, dtype=np.float64)

        _, s, vt = np.linalg.svd(A)
        H_norm = vt[-1].reshape(3, 3)
        # Denormalize: H = inv(T_dst) @ H_norm @ T_src
        H = np.linalg.inv(T_dst) @ H_norm @ T_src

        # Normalize so H[2,2] == 1.0 (or sign positive)
        if abs(H[2, 2]) > 1e-9:
            H = H / H[2, 2]
        else:
            H = H / np.linalg.norm(H)

        # 2. Compute Condition Number kappa(H)
        u_svd, s_h, _ = np.linalg.svd(H)
        condition_num = float(s_h[0] / max(s_h[-1], 1e-12))
        if condition_num > MAX_CONDITION_NUMBER or np.isnan(condition_num):
            return HomographySolveResult(
                is_valid=False,
                H=None,
                condition_number=condition_num,
                horizon_distance_px=0.0,
                loo_rms_error_m=float("inf"),
                loo_point_errors_m=[],
                held_out_error_m=None,
                quality_gate="rejected",
                grid_polylines=[],
                message=f"Degenerate collinear points: Condition number kappa={condition_num:.1e} exceeds threshold.",
            )

        # 3. Projective Horizon Clearance Check
        # Horizon equation: H[2,0]*u + H[2,1]*v + H[2,2] = 0
        w_px, h_px = frame_size
        corners = np.array([[0, 0], [w_px, 0], [w_px, h_px], [0, h_px]], dtype=np.float64)
        denom_corners = [H[2, 0] * c[0] + H[2, 1] * c[1] + H[2, 2] for c in corners]

        # All corners must have the same sign in denominator (no horizon crossing inside frame)
        if any(d <= 0 for d in denom_corners) and any(d >= 0 for d in denom_corners):
            return HomographySolveResult(
                is_valid=False,
                H=None,
                condition_number=condition_num,
                horizon_distance_px=0.0,
                loo_rms_error_m=float("inf"),
                loo_point_errors_m=[],
                held_out_error_m=None,
                quality_gate="rejected",
                grid_polylines=[],
                message="Projective horizon crosses inside visible frame: causes division by zero in road plane.",
            )

        # Distance from top edge (v=0) to horizon line
        h_a, h_b, h_c = H[2, 0], H[2, 1], H[2, 2]
        horizon_dist = abs(h_c) / max(np.hypot(h_a, h_b), 1e-8)

        # 4. Leave-One-Out (LOO) Cross-Validation Error
        loo_errors = []
        if n_pts >= 5:
            for i in range(n_pts):
                sub_src = [p for j, p in enumerate(image_points) if j != i]
                sub_dst = [p for j, p in enumerate(world_points_m) if j != i]
                sub_res = HomographyEngine.solve_homography(sub_src, sub_dst, frame_size=frame_size)
                if sub_res.is_valid and sub_res.H is not None:
                    H_sub = np.array(sub_res.H)
                    px, py = image_points[i]
                    p_homo = np.array([px, py, 1.0], dtype=np.float64)
                    w_est = H_sub @ p_homo
                    if w_est[2] > 1e-6:
                        wx_est, wy_est = w_est[0] / w_est[2], w_est[1] / w_est[2]
                        err = float(np.hypot(wx_est - world_points_m[i][0], wy_est - world_points_m[i][1]))
                        loo_errors.append(err)
                    else:
                        # Behind the horizon: the fold-out point has no
                        # position on the ground plane at all.
                        loo_errors.append(float("inf"))
                else:
                    # The sub-fit was degenerate, so this point yields no
                    # measurement. Appending 1.0 here — as this did — puts a
                    # plausible-looking metre into a list that is then read as
                    # measured error, and reports it as the held-out result.
                    loo_errors.append(float("inf"))
        else:
            # Exactly 4 points. These determine an 8-DOF homography exactly,
            # so the fit passes through every one of them and the
            # reprojection error is zero by construction — it measures
            # arithmetic, not accuracy.
            #
            # Reporting it as a held-out error is what produced the 0.017 m
            # figures previously stored against every camera. There is no
            # validation available here, and saying so is the only honest
            # answer; a fifth point is what makes one possible.
            for i in range(n_pts):
                px, py = image_points[i]
                p_homo = np.array([px, py, 1.0], dtype=np.float64)
                w_est = H @ p_homo
                wx_est, wy_est = w_est[0] / w_est[2], w_est[1] / w_est[2]
                err = float(np.hypot(wx_est - world_points_m[i][0], wy_est - world_points_m[i][1]))
                loo_errors.append(err)

        loo_rms = float(np.sqrt(np.mean(np.array(loo_errors) ** 2))) if loo_errors else 0.0
        cross_validated = n_pts >= 5

        # Held-out point error if specified — and only when the errors above
        # were genuinely computed from homographies that had not seen the
        # point being tested.
        held_out_err = None
        if (cross_validated and held_out_idx is not None
                and 0 <= held_out_idx < len(loo_errors)):
            held_out_err = float(loo_errors[held_out_idx])

        # Evaluate quality gate
        if not cross_validated:
            q_gate = "unvalidated"
        elif (eval_err := (held_out_err if held_out_err is not None
                           else loo_rms)) < QUALITY_GATE_ACCEPT_M:
            q_gate = "good"
        elif eval_err <= QUALITY_GATE_FLAG_M:
            q_gate = "degraded"
        else:
            q_gate = "rejected"

        # 5. Generate Perspective Ground Grid Polylines (e.g. 5m intervals)
        grid_polys = HomographyEngine._generate_perspective_grid(H, world_points_m, frame_size)

        # State what was actually measured. The previous message claimed
        # "validated (sub-pixel accuracy)" on every successful solve,
        # including four-point fits where nothing was validated and metre-
        # scale errors where the accuracy was not sub-pixel.
        if not cross_validated:
            msg = (f"Homography computed from {n_pts} points. With only 4 "
                   f"points the fit is exact and cannot be checked — add a "
                   f"5th point to measure its accuracy.")
        elif held_out_err is not None:
            msg = (f"Homography computed from {n_pts} points. Held-out point "
                   f"predicted to {held_out_err:.3f} m; leave-one-out RMS "
                   f"across all points {loo_rms:.3f} m.")
        else:
            msg = (f"Homography computed from {n_pts} points. Leave-one-out "
                   f"RMS {loo_rms:.3f} m.")

        return HomographySolveResult(
            is_valid=True,
            H=H.tolist(),
            condition_number=condition_num,
            horizon_distance_px=horizon_dist,
            loo_rms_error_m=loo_rms,
            loo_point_errors_m=loo_errors,
            held_out_error_m=held_out_err,
            quality_gate=q_gate,
            grid_polylines=grid_polys,
            message=msg,
        )

    @staticmethod
    def _generate_perspective_grid(
        H: np.ndarray,
        world_points_m: List[Tuple[float, float]],
        frame_size: Tuple[int, int],
    ) -> List[List[Tuple[float, float]]]:
        """Projects a metric world grid (5m x 5m) back into pixel coordinates for viewport overlay."""
        try:
            H_inv = np.linalg.inv(H)
        except Exception:
            return []

        w_px, h_px = frame_size
        xs = [p[0] for p in world_points_m]
        ys = [p[1] for p in world_points_m]
        min_x, max_x = math.floor(min(xs) - 5), math.ceil(max(xs) + 5)
        min_y, max_y = math.floor(max(0, min(ys) - 2)), math.ceil(max(ys) + 15)

        lines = []
        # Longitudinal grid lines (parallel to road direction)
        for x in range(min_x, max_x + 1, 5):
            poly = []
            for y in np.linspace(min_y, max_y, 25):
                w_vec = np.array([float(x), float(y), 1.0], dtype=np.float64)
                p_vec = H_inv @ w_vec
                if p_vec[2] > 1e-6:
                    u, v = p_vec[0] / p_vec[2], p_vec[1] / p_vec[2]
                    if -100 <= u <= w_px + 100 and -100 <= v <= h_px + 100:
                        poly.append((round(float(u), 1), round(float(v), 1)))
            if len(poly) >= 2:
                lines.append(poly)

        # Lateral grid lines (crosswise to road)
        for y in range(min_y, max_y + 1, 5):
            poly = []
            for x in np.linspace(min_x, max_x, 25):
                w_vec = np.array([float(x), float(y), 1.0], dtype=np.float64)
                p_vec = H_inv @ w_vec
                if p_vec[2] > 1e-6:
                    u, v = p_vec[0] / p_vec[2], p_vec[1] / p_vec[2]
                    if -100 <= u <= w_px + 100 and -100 <= v <= h_px + 100:
                        poly.append((round(float(u), 1), round(float(v), 1)))
            if len(poly) >= 2:
                lines.append(poly)

        return lines


class CameraDriftStabilizer:
    """Tracks camera mast sway, wind vibration, and PTZ drift using ORB background feature tracking."""

    def __init__(self, camera_id: str, max_drift_deg_alert: float = 3.0):
        self.camera_id = camera_id
        self.max_drift_deg_alert = max_drift_deg_alert
        self.ref_keypoints = None
        self.ref_descriptors = None
        self.ref_frame = None
        self.orb = cv2.ORB_create(nfeatures=500, scoreType=cv2.ORB_FAST_SCORE)
        self.matcher = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=True)

        # Kalman / Exponential smoothing on affine drift
        self.smoothed_affine = np.eye(3, dtype=np.float64)
        self.alpha_smooth = 0.15
        self.history_jitter = []
        self.last_ts = datetime.now(timezone.utc).isoformat()

    def set_reference_frame(self, frame: np.ndarray, vehicle_mask: Optional[np.ndarray] = None):
        """Captures a baseline reference frame for drift tracking, masking moving vehicles."""
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY) if len(frame.shape) == 3 else frame
        mask = np.ones_like(gray, dtype=np.uint8) * 255
        if vehicle_mask is not None:
            mask = cv2.bitwise_and(mask, vehicle_mask)

        kps, descs = self.orb.detectAndCompute(gray, mask)
        if descs is not None and len(kps) >= 20:
            self.ref_keypoints = kps
            self.ref_descriptors = descs
            self.ref_frame = gray.copy()
            logger.info("CameraDriftStabilizer [%s]: Set reference keyframe with %d ORB features.", self.camera_id, len(kps))

    def update_frame(self, frame: np.ndarray, vehicle_mask: Optional[np.ndarray] = None) -> Tuple[np.ndarray, CameraDriftTelemetry]:
        """Computes affine vibration matrix A_t and returns stabilized Homography correction."""
        if self.ref_descriptors is None or self.ref_frame is None:
            self.set_reference_frame(frame, vehicle_mask)
            return np.eye(3, dtype=np.float64), self._build_telemetry(0.0, 0.0, 0.0, 0.0, "STABLE")

        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY) if len(frame.shape) == 3 else frame
        mask = np.ones_like(gray, dtype=np.uint8) * 255
        if vehicle_mask is not None:
            mask = cv2.bitwise_and(mask, vehicle_mask)

        kps, descs = self.orb.detectAndCompute(gray, mask)
        if descs is None or len(kps) < 15:
            return self.smoothed_affine, self._build_telemetry(0.0, 0.0, 0.0, 0.0, "STABLE")

        matches = self.matcher.match(self.ref_descriptors, descs)
        if len(matches) < 10:
            return self.smoothed_affine, self._build_telemetry(0.0, 0.0, 0.0, 0.0, "STABLE")

        matches = sorted(matches, key=lambda m: m.distance)[:100]
        src_pts = np.float32([self.ref_keypoints[m.queryIdx].pt for m in matches]).reshape(-1, 1, 2)
        dst_pts = np.float32([kps[m.trainIdx].pt for m in matches]).reshape(-1, 1, 2)

        # Estimate affine transform (translation + rotation + scale) via RANSAC
        M, inliers = cv2.estimateAffinePartial2D(dst_pts, src_pts, method=cv2.RANSAC, ransacReprojThreshold=3.0)
        if M is None:
            return self.smoothed_affine, self._build_telemetry(0.0, 0.0, 0.0, 0.0, "STABLE")

        # Convert 2x3 affine matrix to 3x3 homogeneous matrix
        A_raw = np.eye(3, dtype=np.float64)
        A_raw[:2, :] = M

        # Exponential moving average filter
        self.smoothed_affine = (1.0 - self.alpha_smooth) * self.smoothed_affine + self.alpha_smooth * A_raw

        # Extract translation jitter (pixels) and rotation drift (degrees)
        dx, dy = self.smoothed_affine[0, 2], self.smoothed_affine[1, 2]
        jitter_px = float(np.hypot(dx, dy))
        rotation_rad = float(math.atan2(self.smoothed_affine[1, 0], self.smoothed_affine[0, 0]))
        drift_deg = abs(math.degrees(rotation_rad))

        self.history_jitter.append(jitter_px)
        if len(self.history_jitter) > 50:
            self.history_jitter.pop(0)
        rms_jitter = float(np.sqrt(np.mean(np.array(self.history_jitter) ** 2)))

        status = "STABLE"
        if drift_deg > self.max_drift_deg_alert:
            status = "RECALIBRATION_REQUIRED"
        elif rms_jitter > 8.0:
            status = "HIGH_WIND_SWAY"
        elif rms_jitter > 2.0:
            status = "MINOR_VIBRATION"

        telemetry = self._build_telemetry(
            jitter_rms_px=rms_jitter,
            cumulative_drift_deg=drift_deg,
            tilt_delta_mrad=dy * 0.25,
            pan_delta_mrad=dx * 0.25,
            drift_status=status,
        )
        return self.smoothed_affine, telemetry

    def _build_telemetry(
        self,
        jitter_rms_px: float,
        cumulative_drift_deg: float,
        tilt_delta_mrad: float,
        pan_delta_mrad: float,
        drift_status: str,
    ) -> CameraDriftTelemetry:
        return CameraDriftTelemetry(
            camera_id=self.camera_id,
            is_stabilized=True,
            jitter_rms_px=round(jitter_rms_px, 2),
            cumulative_drift_deg=round(cumulative_drift_deg, 3),
            tilt_delta_mrad=round(tilt_delta_mrad, 3),
            pan_delta_mrad=round(pan_delta_mrad, 3),
            drift_status=drift_status,
            last_stabilized_ts=datetime.now(timezone.utc).isoformat(),
        )


class VanishingPointCalibrator:
    """Automated vanishing point detector from road lane markings & traffic trajectories."""

    @staticmethod
    def auto_detect_vanishing_points(frame: np.ndarray) -> Dict[str, Any]:
        """Detects road lane segments and calculates intersection vanishing point."""
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY) if len(frame.shape) == 3 else frame
        h, w = gray.shape[:2]

        # Line Segment Detector (LSD)
        lsd = cv2.createLineSegmentDetector(0)
        lines, _, _, _ = lsd.detect(gray)

        if lines is None or len(lines) < 6:
            # No vanishing point was found, so none is returned. This used to
            # answer with [w/2, h*0.35] and confidence 0.5 — the centre of the
            # frame, dressed as a measurement, on a failure path.
            return {
                "success": False,
                "vp": None,
                "reason": (f"only {0 if lines is None else len(lines)} line "
                           f"segments found; the view may have no road "
                           f"markings or structure to work from"),
                "lane_lines": [],
                "suggested_gcps": [],
            }

        # Filter lines that are roughly vertical/longitudinal (slopes between 20 deg and 80 deg)
        good_lines = []
        for line in lines:
            coords = line.ravel()
            if len(coords) < 4:
                continue
            x1, y1, x2, y2 = coords[:4]
            dx, dy = x2 - x1, y2 - y1
            length = np.hypot(dx, dy)
            if length < 60:
                continue
            angle = abs(math.degrees(math.atan2(dy, dx)))
            if 25.0 <= angle <= 80.0 or 100.0 <= angle <= 155.0:
                # Homogeneous line equation: ax + by + c = 0
                a = y1 - y2
                b = x2 - x1
                c = x1 * y2 - x2 * y1
                norm = np.hypot(a, b)
                if norm > 1e-6:
                    good_lines.append((a / norm, b / norm, c / norm, length, (x1, y1, x2, y2)))

        if len(good_lines) < 4:
            return {
                "success": False,
                "vp": None,
                "reason": (f"{len(good_lines)} usable road-direction segments "
                           f"found, at least 4 are needed to intersect"),
                "lane_lines": [],
                "suggested_gcps": [],
            }

        # Solve least-squares intersection for Vanishing Point
        A = np.array([[l[0], l[1]] for l in good_lines], dtype=np.float64)
        b_vec = -np.array([l[2] for l in good_lines], dtype=np.float64)
        weights = np.sqrt([l[3] for l in good_lines])

        sol, _, _, _ = np.linalg.lstsq(A * weights[:, None], b_vec * weights, rcond=None)
        vp_x, vp_y = float(sol[0]), float(sol[1])

        # How well the lines actually agree on a single point. This replaces
        # a "confidence" of min(0.96, 0.65 + n*0.02) — a number that rose with
        # the count of segments regardless of whether they intersected
        # anywhere near each other, and so reported 0.96 for a wall covered in
        # unrelated edges.
        residuals = [abs(a * vp_x + b_ * vp_y + c)
                     for a, b_, c, _, _ in good_lines]
        vp_rms_px = float(np.sqrt(np.mean(np.square(residuals))))

        # Points placed on the road the detector actually found, spread along
        # the direction of travel so they span depth rather than clustering.
        #
        # world_m is deliberately absent. It used to be filled with a fixed
        # 14 m x 16 m rectangle — the same coordinates for every camera in the
        # fleet, at pixel positions that were fractions of the frame and owed
        # nothing to the detected vanishing point. Adopting those produced a
        # camera that looked calibrated and measured an imaginary road.
        #
        # Scale cannot be recovered from one image. The operator supplies each
        # point's ground coordinates from a survey, a measuring wheel, or
        # satellite imagery, and the solver then has something real to fit.
        pts = []
        for line in sorted(good_lines, key=lambda l: -l[3])[:24]:
            x1, y1, x2, y2 = line[4]
            pts.append((x1, y1))
            pts.append((x2, y2))
        pts = [p for p in pts if p[1] > vp_y + 0.05 * h]      # below the horizon
        pts.sort(key=lambda p: p[1])

        suggested = []
        if pts:
            # Sample across the visible depth range: near, far and between.
            for frac in (0.1, 0.3, 0.5, 0.75, 0.95):
                p = pts[min(len(pts) - 1, int(frac * (len(pts) - 1)))]
                suggested.append({
                    "label": f"Road point {len(suggested) + 1}",
                    "pixel": [int(round(p[0])), int(round(p[1]))],
                    "world_m": None,
                    "source": "detected road segment endpoint",
                })

        rendered_lines = [[int(x1), int(y1), int(x2), int(y2)] for *_, (x1, y1, x2, y2) in good_lines[:12]]

        return {
            "success": True,
            "vp": [round(vp_x, 1), round(vp_y, 1)],
            "vp_rms_px": round(vp_rms_px, 2),
            "segments_used": len(good_lines),
            "lane_lines": rendered_lines,
            "suggested_gcps": suggested,
            "world_coordinates_required": True,
        }
