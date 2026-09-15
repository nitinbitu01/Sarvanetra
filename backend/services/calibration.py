"""
backend/services/calibration.py — Camera Homography & World Coordinates Calibration (Phase 0).

Provides:
  - compute_homography: Computes 3x3 homography matrix H.
  - pixel_to_world_m: Converts image pixel (x, y) to real-world ground meters (wx, wy).
  - world_to_pixel: Projects real-world ground meters (wx, wy) back to pixel space.
  - world_to_wgs84: Geodesic transformation from local ground meters to WGS-84 (lat, lon).
  - wgs84_to_world: Inverse transformation from WGS-84 (lat, lon) to local ground meters.
  - evaluate_calibration_quality: Strict multi-tier quality gate (<0.5m good, 0.5-1.0m degraded, >1.0m rejected).
  - validate_homography: Validates held-out control points against ground truth.
  - convert_polyline_to_world: Converts normalized lane polylines (0.0–1.0) to world meters.
  - nearest_tangent: Finds the polyline tangent vector nearest to vehicle world position.
"""
from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Optional, Sequence

import cv2
import numpy as np

# WGS-84 ellipsoid constants
WGS84_A = 6378137.0          # semi-major axis (meters)
WGS84_F = 1.0 / 298.257223563 # flattening
WGS84_E2 = 2 * WGS84_F - WGS84_F ** 2  # first eccentricity squared

# Quality gate thresholds (meters)
QUALITY_GATE_ACCEPT_M = 0.50
QUALITY_GATE_FLAG_M = 1.00


@dataclass
class CalibrationResult:
    H: np.ndarray                              # 3x3 homography matrix
    reprojection_error_m: float                # held-out point error in meters
    quality_gate: str                          # "good" | "degraded" | "rejected"
    calibrated_by: str
    calibrated_on: str                         # ISO date string
    gps_anchor_lat: Optional[float] = None
    gps_anchor_lon: Optional[float] = None
    bearing_deg: float = 0.0


def compute_homography(
    image_points: Sequence[tuple[float, float]],   # >= 4 pixel coords
    world_points_m: Sequence[tuple[float, float]], # matching real-world ground meters
) -> np.ndarray:
    """Computes homography H such that world_homogeneous = H @ pixel_homogeneous."""
    if len(image_points) < 4 or len(world_points_m) < 4:
        raise ValueError("At least 4 point correspondences required.")
    if len(image_points) != len(world_points_m):
        raise ValueError("image_points and world_points_m must have equal length.")

    src = np.array(image_points, dtype=np.float32)
    dst = np.array(world_points_m, dtype=np.float32)
    H, _ = cv2.findHomography(src, dst, method=0)
    if H is None:
        raise ValueError("cv2.findHomography failed to estimate a valid matrix.")
    return H.astype(np.float64)


def pixel_to_world_m(H: np.ndarray, x: float, y: float) -> tuple[float, float]:
    """Projects a single pixel coordinate (x, y) to world-plane meters (wx, wy) via homography."""
    p = np.array([float(x), float(y), 1.0], dtype=np.float64)
    w = H @ p
    if w[2] <= 1e-6:
        raise ValueError(f"Pixel ({x:.1f}, {y:.1f}) is at or above the projective horizon (w_z={w[2]:.6e})")
    return (float(w[0] / w[2]), float(w[1] / w[2]))


# Alias for backwards compatibility
pixel_to_world = pixel_to_world_m


def pixel_world_resolution_m(H: np.ndarray, x: float, y: float) -> float:
    """Ground metres spanned by one pixel at image point (x, y).

    Passing the horizon test is not the same as being measurable. A projective
    map compresses the whole far half of the world into the last few rows of
    pixels above the vanishing line, so a point one pixel below the horizon is
    formally valid and physically useless: on CAM_08 the third row of H is
    [0, 1, -248.9], so at v=250 one pixel of detector jitter moves the world
    point by kilometres. That is how a single bus track accumulated a
    path_length of 84,445 m.

    This returns the local scale factor of the pixel -> world map, computed as
    the largest singular value of its 2x2 Jacobian — the worst-case ground
    displacement produced by one pixel of image displacement, in any direction.
    Callers compare it against the positional tolerance they need; nothing here
    is camera-specific, so the same test works on any calibration.

    Raises the same ValueError as pixel_to_world_m at or above the horizon.
    """
    H = np.asarray(H, dtype=np.float64).reshape(3, 3)
    p = np.array([float(x), float(y), 1.0], dtype=np.float64)
    w = H @ p
    d = w[2]
    if d <= 1e-6:
        raise ValueError(
            f"Pixel ({x:.1f}, {y:.1f}) is at or above the projective horizon "
            f"(w_z={d:.6e})"
        )

    # d(w0/d, w1/d) / d(x, y), by the quotient rule. H[i, 0] and H[i, 1] are
    # the partials of w_i with respect to x and y.
    jac = np.empty((2, 2), dtype=np.float64)
    for i in (0, 1):
        for j in (0, 1):
            jac[i, j] = (H[i, j] * d - w[i] * H[2, j]) / (d * d)

    return float(np.linalg.svd(jac, compute_uv=False)[0])


def world_to_pixel(H: np.ndarray, wx: float, wy: float) -> tuple[float, float]:
    """Projects world-plane meters (wx, wy) back to pixel coordinate (px, py)."""
    try:
        H_inv = np.linalg.inv(H)
    except np.linalg.LinAlgError as e:
        raise ValueError(f"Cannot invert singular homography matrix: {e}") from e

    w = np.array([float(wx), float(wy), 1.0], dtype=np.float64)
    p = H_inv @ w
    if abs(p[2]) < 1e-9:
        return (float(p[0]), float(p[1]))
    return (float(p[0] / p[2]), float(p[1] / p[2]))


def evaluate_calibration_quality(reprojection_error_m: float) -> str:
    """Strict quality gate:
      - < 0.50 m: 'good' (accepted for full speed & congestion products)
      - 0.50–1.00 m: 'degraded' (flagged in dashboard, down-weighted)
      - > 1.00 m: 'rejected' (speed marked NULL, excluded from network figures)
    """
    if reprojection_error_m < QUALITY_GATE_ACCEPT_M:
        return "good"
    elif reprojection_error_m <= QUALITY_GATE_FLAG_M:
        return "degraded"
    else:
        return "rejected"


def world_to_wgs84(
    wx: float,
    wy: float,
    anchor_lat: Optional[float],
    anchor_lon: Optional[float],
    bearing_deg: float = 0.0,
) -> tuple[Optional[float], Optional[float]]:
    """Converts camera-local ground meters (wx, wy) to WGS-84 latitude and longitude.

    Convention:
      - wx: lateral offset in meters (positive = right of camera optical axis)
      - wy: longitudinal forward offset in meters (positive = along camera viewing direction)
      - bearing_deg: camera azimuth clockwise from true North (0° = North, 90° = East, 180° = South, 270° = West)
    """
    if anchor_lat is None or anchor_lon is None:
        return (None, None)

    theta = math.radians(bearing_deg)
    # Rotate (wx, wy) into North and East displacements
    delta_north = wy * math.cos(theta) - wx * math.sin(theta)
    delta_east = wy * math.sin(theta) + wx * math.cos(theta)

    # Meridian and Prime Vertical radii of curvature at anchor latitude
    lat_rad = math.radians(anchor_lat)
    sin_lat = math.sin(lat_rad)
    denom = 1.0 - WGS84_E2 * (sin_lat ** 2)
    m = WGS84_A * (1.0 - WGS84_E2) / (denom ** 1.5)  # meters per radian latitude
    n = WGS84_A / math.sqrt(denom)                  # meters per radian prime vertical

    d_lat_deg = (delta_north / m) * (180.0 / math.pi)
    d_lon_deg = (delta_east / (n * math.cos(lat_rad))) * (180.0 / math.pi)

    return (anchor_lat + d_lat_deg, anchor_lon + d_lon_deg)


def wgs84_to_world(
    lat: Optional[float],
    lon: Optional[float],
    anchor_lat: Optional[float],
    anchor_lon: Optional[float],
    bearing_deg: float = 0.0,
) -> tuple[Optional[float], Optional[float]]:
    """Inverse transformation: converts WGS-84 (lat, lon) to local ground meters (wx, wy)."""
    if lat is None or lon is None or anchor_lat is None or anchor_lon is None:
        return (None, None)

    lat_rad = math.radians(anchor_lat)
    sin_lat = math.sin(lat_rad)
    denom = 1.0 - WGS84_E2 * (sin_lat ** 2)
    m = WGS84_A * (1.0 - WGS84_E2) / (denom ** 1.5)
    n = WGS84_A / math.sqrt(denom)

    d_lat_rad = math.radians(lat - anchor_lat)
    d_lon_rad = math.radians(lon - anchor_lon)

    delta_north = d_lat_rad * m
    delta_east = d_lon_rad * n * math.cos(lat_rad)

    theta = math.radians(bearing_deg)
    wy = delta_north * math.cos(theta) + delta_east * math.sin(theta)
    wx = delta_east * math.cos(theta) - delta_north * math.sin(theta)

    return (wx, wy)


def validate_homography(
    H: np.ndarray,
    held_out_image: tuple[float, float],
    held_out_world: tuple[float, float],
    max_error_m: float = 1.00,
) -> float:
    """Projects the held-out image point and computes Euclidean distance from expected world position.

    Raises ValueError if error exceeds max_error_m.
    """
    wx, wy = pixel_to_world_m(H, *held_out_image)
    error = float(np.hypot(wx - held_out_world[0], wy - held_out_world[1]))
    if error > max_error_m:
        raise ValueError(
            f"Homography reprojection error {error:.3f}m exceeds "
            f"maximum allowed {max_error_m}m. "
            f"Recalibrate: add more ground-control points or verify "
            f"measured world coordinates."
        )
    return error


def calibrate_camera(
    image_points: Sequence[tuple[float, float]],
    world_points_m: Sequence[tuple[float, float]],
    held_out_image: tuple[float, float],
    held_out_world: tuple[float, float],
    calibrated_by: str,
    calibrated_on: str,
    max_error_m: float = 1.00,
    gps_anchor_lat: Optional[float] = None,
    gps_anchor_lon: Optional[float] = None,
    bearing_deg: float = 0.0,
) -> CalibrationResult:
    """Full calibration workflow: compute H, validate on held-out point, evaluate quality gate."""
    H = compute_homography(image_points, world_points_m)
    error = validate_homography(H, held_out_image, held_out_world, max_error_m)
    gate = evaluate_calibration_quality(error)
    return CalibrationResult(
        H=H,
        reprojection_error_m=error,
        quality_gate=gate,
        calibrated_by=calibrated_by,
        calibrated_on=calibrated_on,
        gps_anchor_lat=gps_anchor_lat,
        gps_anchor_lon=gps_anchor_lon,
        bearing_deg=bearing_deg,
    )


def convert_polyline_to_world(
    H: np.ndarray,
    normalized_polyline: Sequence[tuple[float, float]],
    frame_w: int = 1920,
    frame_h: int = 1080,
) -> list[tuple[float, float]]:
    """Converts a normalized-image-coordinate polyline (0.0–1.0) to world-meter coordinates."""
    world_pts = []
    for (nx, ny) in normalized_polyline:
        px = nx * frame_w
        py = ny * frame_h
        wx, wy = pixel_to_world_m(H, px, py)
        world_pts.append((wx, wy))
    return world_pts


def nearest_tangent(
    world_polyline: Sequence[tuple[float, float]],
    vehicle_pos: tuple[float, float],
) -> tuple[float, float]:
    """Finds polyline segment nearest to vehicle_pos and returns unit tangent vector."""
    if not world_polyline or len(world_polyline) < 2:
        return (0.0, 1.0)

    vx, wy_ = vehicle_pos
    best_idx = 0
    best_dist = float("inf")

    for i, (px, py) in enumerate(world_polyline[:-1]):
        dist = float(np.hypot(vx - px, wy_ - py))
        if dist < best_dist:
            best_dist = dist
            best_idx = i

    x0, y0 = world_polyline[best_idx]
    x1, y1 = world_polyline[best_idx + 1]
    dx, dy = x1 - x0, y1 - y0
    norm = float(np.hypot(dx, dy))
    if norm < 1e-9:
        return (1.0, 0.0)
    return (dx / norm, dy / norm)
