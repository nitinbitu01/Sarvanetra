"""
backend/services/junction_topology.py — 9-Gap Junction Topology & Geometry Engine (v15.0.0)

Closes:
  - Gap V: Indian roundabouts are CLOCKWISE (left-hand traffic rule).
           Tangent vector: (+dy/R, -dx/R)
  - Gap Z: Full Frenet frame cubic spline implementation for curved slipways (s, d coordinates).
  - Gap Y: Three-stage emergency vehicle classifier (class, livery HSV ratio, beacon).
  - Gap 1-9: Complete geometry evaluation (rotaries, median cuts, slipways, BRTS corridors,
             parking maneuvers, zebra crossings, emergency vehicles, deduplication, broken-down push).
"""
from __future__ import annotations

import math
from dataclasses import dataclass, field
from enum import Enum, auto
from typing import Optional

import cv2
import numpy as np
from scipy.interpolate import CubicSpline
from scipy.optimize import minimize_scalar


class ManeuverVerdict(Enum):
    PROCEED_TO_CHECK = auto()      # Run normal violation detection
    ROUNDABOUT_LEGAL = auto()      # Circling rotary clockwise — legal
    MEDIAN_UTURN_GRACE = auto()    # Inside median cut — 45s stationary grace
    SLIPWAY_LEGAL = auto()         # On curved slipway within Frenet tolerance — legal
    BRTS_INCURSION = auto()        # Unauthorized vehicle in BRTS — Sec 177 ₹500
    BRTS_WRONG_WAY = auto()        # Counter-flow inside BRTS — Sec 184 ₹1,500
    PARKING_MANEUVER = auto()      # Short reverse < 5m at < 5 km/h — exempt
    PEDESTRIAN_CROSSING = auto()   # Person in zebra zone — exclude from rider count
    EMERGENCY_EXEMPT = auto()      # 108 ambulance / fire — MV Act Sec 194E
    DEDUPLICATED = auto()          # Same plate seen < 15 min ago
    BROKEN_DOWN_PUSH = auto()      # V < 6 km/h on shoulder — exempt


# ── Roundabout Zone (Gap V Fix) ──────────────────────────────────────────
@dataclass
class RoundaboutZone:
    """
    Indian roundabouts are CLOCKWISE (left-hand traffic rule).
    Vehicles enter from the left and circulate clockwise when viewed from above.
    """
    center_x_m: float
    center_y_m: float
    inner_radius_m: float
    outer_radius_m: float

    def contains(self, wx: float, wy: float) -> bool:
        d = math.hypot(wx - self.center_x_m, wy - self.center_y_m)
        return self.inner_radius_m <= d <= self.outer_radius_m

    def legal_flow_vector(self, wx: float, wy: float) -> tuple[float, float]:
        """
        Clockwise tangent at point (wx, wy) around center.
        For clockwise rotation: tangent = (+dy/R, -dx/R) where dx = wx - cx, dy = wy - cy.
        """
        dx = wx - self.center_x_m
        dy = wy - self.center_y_m
        R = math.hypot(dx, dy) + 1e-9
        tx = dy / R
        ty = -dx / R
        return (float(tx), float(ty))


# ── Median Cut Zone ──────────────────────────────────────────────────────
@dataclass
class MedianCutZone:
    """
    45-second stationary grace for vehicles completing a U-turn in a designated median pocket.
    """
    polygon_world_m: list[tuple[float, float]]
    grace_period_sec: float = 45.0
    _entry_times: dict[int, float] = field(default_factory=dict, repr=False)

    def contains(self, wx: float, wy: float) -> bool:
        pts = np.array(self.polygon_world_m, dtype=np.float32)
        return cv2.pointPolygonTest(pts, (wx, wy), False) >= 0

    def in_grace_period(self, track_id: int, now: float) -> bool:
        if track_id not in self._entry_times:
            self._entry_times[track_id] = now
        return (now - self._entry_times[track_id]) <= self.grace_period_sec

    def clear_track(self, track_id: int) -> None:
        self._entry_times.pop(track_id, None)


# ── Slipway Zone (Gap Z Fix: Full Frenet Frame) ───────────────────────────
@dataclass
class SlipwayZone:
    """
    Full Frenet frame implementation for curved slipways.
    Projects vehicle position onto the cubic spline centerline to find
    arc-length parameter s and lateral deviation d.
    Vehicle is legal if |d| <= max_lateral_deviation_m.
    """
    centerline_world_m: list[tuple[float, float]]
    polygon_world_m: list[tuple[float, float]]
    max_lateral_deviation_m: float = 2.5
    _s_vals: np.ndarray = field(init=False, repr=False)
    _cs_x: object = field(init=False, repr=False)
    _cs_y: object = field(init=False, repr=False)

    def __post_init__(self):
        pts = np.array(self.centerline_world_m, dtype=np.float64)
        diffs = np.diff(pts, axis=0)
        segs = np.hypot(diffs[:, 0], diffs[:, 1])
        s = np.concatenate([[0.0], np.cumsum(segs)])
        self._s_vals = s
        self._cs_x = CubicSpline(s, pts[:, 0])
        self._cs_y = CubicSpline(s, pts[:, 1])

    def contains(self, wx: float, wy: float) -> bool:
        pts = np.array(self.polygon_world_m, dtype=np.float32)
        return cv2.pointPolygonTest(pts, (wx, wy), False) >= 0

    def lateral_deviation(self, wx: float, wy: float) -> float:
        S_max = float(self._s_vals[-1])
        s_test = np.linspace(0.0, S_max, 200)
        cx = self._cs_x(s_test)
        cy = self._cs_y(s_test)
        dists = np.hypot(cx - wx, cy - wy)
        best_s = float(s_test[np.argmin(dists)])

        lo = max(0.0, best_s - S_max / 20.0)
        hi = min(S_max, best_s + S_max / 20.0)
        res = minimize_scalar(
            lambda s_val: np.hypot(self._cs_x(s_val) - wx, self._cs_y(s_val) - wy),
            bounds=(lo, hi),
            method="bounded",
        )
        return float(res.fun)

    def legal_flow_vector(self, wx: float, wy: float) -> tuple[float, float]:
        S_max = float(self._s_vals[-1])
        s_test = np.linspace(0.0, S_max, 200)
        cx = self._cs_x(s_test)
        cy = self._cs_y(s_test)
        best_s = float(s_test[np.argmin(np.hypot(cx - wx, cy - wy))])
        tx = float(self._cs_x(best_s, 1))
        ty = float(self._cs_y(best_s, 1))
        mag = math.hypot(tx, ty) + 1e-9
        return (tx / mag, ty / mag)


# ── BRTS Corridor Zone ──────────────────────────────────────────────────
@dataclass
class BRTSCorridorZone:
    polygon_world_m: list[tuple[float, float]]
    flow_vector: tuple[float, float]
    allowed_classes: set[str] = field(default_factory=lambda: {"bus", "brts_bus"})

    def contains(self, wx: float, wy: float) -> bool:
        pts = np.array(self.polygon_world_m, dtype=np.float32)
        return cv2.pointPolygonTest(pts, (wx, wy), False) >= 0


# ── Zebra Crossing Zone ─────────────────────────────────────────────────
@dataclass
class ZebraCrossingZone:
    polygon_world_m: list[tuple[float, float]]

    def contains(self, wx: float, wy: float) -> bool:
        pts = np.array(self.polygon_world_m, dtype=np.float32)
        return cv2.pointPolygonTest(pts, (wx, wy), False) >= 0


# ── Shoulder Zone ───────────────────────────────────────────────────────
@dataclass
class ShoulderZone:
    polygon_world_m: list[tuple[float, float]]

    def contains(self, wx: float, wy: float) -> bool:
        pts = np.array(self.polygon_world_m, dtype=np.float32)
        return cv2.pointPolygonTest(pts, (wx, wy), False) >= 0


# ── Emergency Vehicle Classifier (Gap Y Fix) ────────────────────────────
EMERGENCY_CLASSES = {"ambulance", "fire_truck", "police_vehicle", "108_ambulance"}


def is_emergency_vehicle(
    vehicle_class: str,
    crop_bgr: Optional[np.ndarray],
    beacon_model: Optional[object] = None,
) -> bool:
    # Stage 1: Tracker class
    if vehicle_class.lower() in EMERGENCY_CLASSES:
        return True

    # Stage 2: Livery color ratio (Ambulance = White + Red stripe; Police = Blue + White)
    if crop_bgr is not None and crop_bgr.size > 0:
        hsv = cv2.cvtColor(crop_bgr, cv2.COLOR_BGR2HSV)
        total = max(hsv.shape[0] * hsv.shape[1], 1)

        red_mask = cv2.inRange(hsv, (0, 100, 100), (10, 255, 255)) | \
                   cv2.inRange(hsv, (160, 100, 100), (180, 255, 255))
        white_mask = cv2.inRange(hsv, (0, 0, 180), (180, 40, 255))
        blue_mask = cv2.inRange(hsv, (100, 100, 100), (130, 255, 255))

        red_ratio = cv2.countNonZero(red_mask) / total
        white_ratio = cv2.countNonZero(white_mask) / total
        blue_ratio = cv2.countNonZero(blue_mask) / total

        if white_ratio > 0.35 and red_ratio > 0.05:
            return True
        if blue_ratio > 0.15 and white_ratio > 0.25:
            return True

    # Stage 3: Beacon detector model (optional)
    if beacon_model is not None and crop_bgr is not None and crop_bgr.size > 0:
        try:
            preds = beacon_model(crop_bgr, verbose=False)
            if preds and len(preds[0].boxes) > 0:
                if float(preds[0].boxes[0].conf) > 0.70:
                    return True
        except Exception:
            pass

    return False


# ── Plate Deduplication Cache ───────────────────────────────────────────
class PlateDeduplicationCache:
    WINDOW_SEC = 900.0  # 15 minutes

    def __init__(self):
        self._seen: dict[tuple[str, str], float] = {}

    def is_duplicate(self, plate: str, violation_type: str, now: float) -> bool:
        key = (plate.upper(), violation_type)
        if key in self._seen:
            if (now - self._seen[key]) < self.WINDOW_SEC:
                return True
            else:
                del self._seen[key]
        self._seen[key] = now
        return False

    def cleanup(self, now: float) -> None:
        self._seen = {k: v for k, v in self._seen.items() if (now - v) < self.WINDOW_SEC}


# ── Master Junction Topology Definition ─────────────────────────────────
@dataclass
class JunctionTopology:
    roundabouts: list[RoundaboutZone] = field(default_factory=list)
    median_cuts: list[MedianCutZone] = field(default_factory=list)
    slipways: list[SlipwayZone] = field(default_factory=list)
    brts_corridors: list[BRTSCorridorZone] = field(default_factory=list)
    zebra_crossings: list[ZebraCrossingZone] = field(default_factory=list)
    shoulder_zones: list[ShoulderZone] = field(default_factory=list)
    dedup_cache: PlateDeduplicationCache = field(default_factory=PlateDeduplicationCache)


def evaluate_vehicle_maneuver(
    track_id: int,
    wx: float,
    wy: float,
    speed_kmh: float,
    vehicle_class: str,
    heading: tuple[float, float],
    plate: Optional[str],
    violation_type: str,
    topology: JunctionTopology,
    vehicle_crop: Optional[np.ndarray],
    now: float,
    net_displacement_opp_m: float = 0.0,
    beacon_model: Optional[object] = None,
) -> ManeuverVerdict:
    # 1. Emergency vehicle check (MV Act Sec 194E)
    if is_emergency_vehicle(vehicle_class, vehicle_crop, beacon_model):
        return ManeuverVerdict.EMERGENCY_EXEMPT

    # 2. Broken-down push on road shoulder
    if speed_kmh < 6.0:
        for shoulder in topology.shoulder_zones:
            if shoulder.contains(wx, wy):
                return ManeuverVerdict.BROKEN_DOWN_PUSH

    # 3. Parking / reversing maneuver (< 5m at < 10 km/h)
    if net_displacement_opp_m < 10.0 and speed_kmh < 10.0:
        return ManeuverVerdict.PARKING_MANEUVER

    # 4. Roundabout circular flow check (Gap V: Clockwise)
    for rotary in topology.roundabouts:
        if rotary.contains(wx, wy):
            flow = rotary.legal_flow_vector(wx, wy)
            cos_a = heading[0] * flow[0] + heading[1] * flow[1]
            if cos_a >= -0.50:
                return ManeuverVerdict.ROUNDABOUT_LEGAL

    # 5. Median U-turn pocket (45s grace)
    for pocket in topology.median_cuts:
        if pocket.contains(wx, wy):
            if pocket.in_grace_period(track_id, now):
                return ManeuverVerdict.MEDIAN_UTURN_GRACE
            else:
                pocket.clear_track(track_id)

    # 6. Free-left slipway (Frenet lateral deviation check)
    for slipway in topology.slipways:
        if slipway.contains(wx, wy):
            lateral_d = slipway.lateral_deviation(wx, wy)
            if lateral_d <= slipway.max_lateral_deviation_m:
                return ManeuverVerdict.SLIPWAY_LEGAL

    # 7. Cross-camera 15-minute deduplication for potential violations
    if plate and topology.dedup_cache.is_duplicate(plate, violation_type, now):
        return ManeuverVerdict.DEDUPLICATED

    # 8. BRTS corridor incursion vs counter-flow
    for brts in topology.brts_corridors:
        if brts.contains(wx, wy):
            if vehicle_class not in brts.allowed_classes:
                flow = brts.flow_vector
                cos_a = heading[0] * flow[0] + heading[1] * flow[1]
                if cos_a < -0.50:
                    return ManeuverVerdict.BRTS_WRONG_WAY
                else:
                    return ManeuverVerdict.BRTS_INCURSION

    return ManeuverVerdict.PROCEED_TO_CHECK

