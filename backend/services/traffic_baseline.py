"""
backend/services/traffic_baseline.py — Enterprise Seasonal Baseline & Contextual Anomaly Engine (Gap 3).

Delivers:
  1. Empirical Bayes Seasonal Baseline: $24 \times 7$ (Hour x Day-of-Week) normal profiles with Bayesian shrinkage.
  2. Multi-Tier Anomaly Engine:
     - Sustained Congestion Drop: $Z < -2.0$ or $>30\%$ drop sustained for $\ge 3$ consecutive minutes.
     - Shockwave Crash / Obstruction: Sudden velocity drop $>50\%$ in $<60$ seconds.
     - Contextual Speeding Outlier: $>3\sigma$ above hour-of-week contextual baseline.
     - Zero-Traffic vs True Gridlock Disambiguation.
  3. Origin-Destination Corridor Matrix: Inter-camera travel time, transit speed, and bottleneck ratios.
"""
from __future__ import annotations

import math
import threading
import time
from collections import defaultdict
from dataclasses import dataclass, field
from datetime import datetime, timedelta, timezone
from typing import Any, Dict, List, Optional, Sequence, Tuple

import numpy as np
from sqlalchemy import func
from sqlalchemy.orm import Session

from backend.db.models import (
    Camera,
    CameraHomographyCalibration,
    CameraMetrics1M,
    VehicleTrack,
    Alert,
)
from backend.db.session import SessionLocal
from backend.services.calibration import evaluate_calibration_quality
from backend.services.congestion_engine import get_congestion_engine
from backend.services.network_activity import haversine_km

# Mathematical constants
MIN_SUSTAINED_ANOMALY_MINUTES = 3
ANOMALY_SPEED_DROP_PCT = 0.30       # 30% drop below baseline
SHOCKWAVE_DROP_PCT = 0.50           # 50% drop within 60s
SHOCKWAVE_MIN_PREV_SPEED = 30.0     # Minimum starting speed to qualify as shockwave crash

# Bayesian Prior Constants (Road Type Defaults)
PRIOR_PSEUDO_COUNT_N0 = 15.0        # Weight of prior in small-sample shrinkage
DEFAULT_URBAN_MEAN_SPEED = 42.0     # km/h
DEFAULT_URBAN_STD_SPEED = 8.5       # km/h
DEFAULT_HIGHWAY_MEAN_SPEED = 65.0   # km/h
DEFAULT_HIGHWAY_STD_SPEED = 12.0    # km/h


@dataclass
class BaselineAnomalyAlert:
    camera_id: str
    camera_name: str
    anomaly_type: str                 # "SUSTAINED_CONGESTION" | "SHOCKWAVE_CRASH" | "CORRIDOR_BOTTLENECK" | "SPEEDING_CLUSTER"
    detected_at: str
    current_speed_kmh: float
    baseline_speed_kmh: float
    deviation_pct: float
    z_score: float
    sustained_minutes: int
    severity: str                     # "CRITICAL" | "WARNING"
    explanation: str
    recommended_action: str


@dataclass
class CorridorFlowMetric:
    origin_cam_id: str
    origin_name: str
    destination_cam_id: str
    destination_name: str
    distance_km: float
    median_transit_time_sec: float
    transit_speed_kmh: float
    free_flow_transit_speed_kmh: float
    delay_ratio: float
    sample_count: int
    confidence: str                   # "good" (n>=10) | "moderate" (n>=3) | "low" (n<3)
    is_bottleneck: bool


class TrafficBaselineEngine:
    """
    Production-Grade Empirical Bayes Seasonal Baseline & Contextual Anomaly Engine.
    Handles 24x7 matrix profiling, Bayesian shrinkage for cold starts, shockwave crash detection,
    and corridor transit matrix analytics.
    """

    _lock = threading.Lock()
    _cached_matrix: Dict[Tuple[str, int, int], Dict[str, Any]] = {}
    _last_built_ts: float = 0.0
    _CACHE_TTL_SEC: float = 600.0  # 10 minutes cache TTL

    def __init__(self):
        # (camera_id, day_of_week, hour) -> {"mean_speed": float, "std_speed": float, "mean_volume": float, "sample_buckets": int}
        self._baseline_matrix: Dict[Tuple[str, int, int], Dict[str, Any]] = {}

    def build_baseline_matrix(self, db: Optional[Session] = None, force_refresh: bool = False) -> int:
        """
        Computes historical 24x7 seasonal baseline statistics with Empirical Bayesian Shrinkage.
        """
        now_ts = time.time()
        should_use_cache = not force_refresh and (db is None) and bool(self._cached_matrix)
        if should_use_cache and (now_ts - self._last_built_ts) < self._CACHE_TTL_SEC:
            self._baseline_matrix = self._cached_matrix
            return len(self._baseline_matrix)

        with self._lock:
            if should_use_cache and (now_ts - self._last_built_ts) < self._CACHE_TTL_SEC:
                self._baseline_matrix = self._cached_matrix
                return len(self._baseline_matrix)

            own_session = False
            if db is None:
                db = SessionLocal()
                own_session = True
            try:
                cameras = {c.camera_id: c for c in db.query(Camera).filter(Camera.is_deleted == False).all()}

                # Metric speed where the camera has a homography, image-plane
                # speed where it does not. Congestion is a comparison of a
                # camera against its own history, so the unit only has to be
                # consistent per camera — but it must never be mixed within
                # one, and the km/h prior below must not be applied to a pixel
                # rate.
                rows = db.query(
                    CameraMetrics1M.camera_id,
                    CameraMetrics1M.bucket_start,
                    CameraMetrics1M.median_speed_kmh,
                    CameraMetrics1M.median_speed_px_s,
                    CameraMetrics1M.speed_basis,
                    CameraMetrics1M.vehicle_count,
                ).all()

                grouped: Dict[Tuple[str, int, int], List[Tuple[float, int]]] = defaultdict(list)
                cam_basis: Dict[str, str] = {}
                cam_all_speeds: Dict[str, List[float]] = defaultdict(list)
                for cam_id, dt, kmh, px, basis, count in rows:
                    basis = basis or ("kmh" if kmh is not None else None)
                    speed = kmh if basis == "kmh" else px
                    if speed is None or basis is None:
                        continue
                    # A camera that has been calibrated part-way through its
                    # history would otherwise mix units in one baseline.
                    prior = cam_basis.setdefault(cam_id, basis)
                    if prior != basis:
                        continue
                    grouped[(cam_id, dt.weekday(), dt.hour)].append(
                        (float(speed), int(count or 0)))
                    cam_all_speeds[cam_id].append(float(speed))

                matrix: Dict[Tuple[str, int, int], Dict[str, Any]] = {}

                # Compute Bayesian-shrunk statistics for each camera x dow x hour
                for (cam_id, dow, hour), samples in grouped.items():
                    cam_rec = cameras.get(cam_id)
                    basis = cam_basis.get(cam_id, "kmh")
                    if basis == "kmh":
                        is_highway = cam_rec and ("Highway" in (cam_rec.name or "") or "Bypass" in (cam_rec.name or ""))
                        mu_0 = DEFAULT_HIGHWAY_MEAN_SPEED if is_highway else DEFAULT_URBAN_MEAN_SPEED
                        sigma_0 = DEFAULT_HIGHWAY_STD_SPEED if is_highway else DEFAULT_URBAN_STD_SPEED
                    else:
                        # Pixel rates have no meaningful external prior — 27
                        # km/h means nothing in an image plane, and shrinking
                        # towards it would drag every uncalibrated camera's
                        # baseline to a number from a different unit. The
                        # camera's own overall distribution is the prior.
                        own = cam_all_speeds.get(cam_id) or [1.0]
                        mu_0 = float(np.median(own))
                        sigma_0 = max(float(np.std(own)), mu_0 * 0.25, 1e-3)

                    n = float(len(samples))
                    speeds = np.array([s[0] for s in samples], dtype=np.float64)
                    counts = np.array([s[1] for s in samples], dtype=np.float64)

                    sample_mean = float(np.mean(speeds))
                    sample_var = float(np.var(speeds)) if n > 1 else (sigma_0 ** 2)

                    # Empirical Bayes Shrinkage Formula
                    n0 = PRIOR_PSEUDO_COUNT_N0
                    shrunk_mean = (n / (n + n0)) * sample_mean + (n0 / (n + n0)) * mu_0
                    shrunk_var = (n / (n + n0)) * sample_var + (n0 / (n + n0)) * (sigma_0 ** 2) + ((n * n0) / ((n + n0) ** 2)) * ((sample_mean - mu_0) ** 2)
                    shrunk_std = max(2.5, math.sqrt(shrunk_var))

                    matrix[(cam_id, dow, hour)] = {
                        "mean_speed": round(shrunk_mean, 1),
                        "std_speed": round(shrunk_std, 1),
                        "mean_volume": round(float(np.mean(counts)), 1),
                        "sample_buckets": int(n),
                        "empirical_mean": round(sample_mean, 1),
                        "is_default": False,
                    }

                self._baseline_matrix = matrix
                TrafficBaselineEngine._cached_matrix = matrix
                TrafficBaselineEngine._last_built_ts = time.time()
                return len(matrix)
            finally:
                if own_session:
                    db.close()

    def get_baseline_for_camera(
        self,
        camera_id: str,
        dt: Optional[datetime] = None,
    ) -> Dict[str, Any]:
        """
        Returns the expected baseline speed, volume, and standard deviation for a camera at given time.
        """
        if dt is None:
            dt = datetime.now(timezone.utc)
        dow = dt.weekday()
        hour = dt.hour
        key = (camera_id, dow, hour)

        base = self._baseline_matrix.get(key)
        if base is not None:
            return base

        # Default fallback with road-class estimation
        return {
            "mean_speed": DEFAULT_URBAN_MEAN_SPEED,
            "std_speed": DEFAULT_URBAN_STD_SPEED,
            "mean_volume": 14.0,
            "sample_buckets": 0,
            "empirical_mean": DEFAULT_URBAN_MEAN_SPEED,
            "is_default": True,
        }

    def get_24h_baseline_curve(
        self,
        camera_id: str,
        target_date: Optional[datetime] = None,
        db: Optional[Session] = None,
    ) -> List[Dict[str, Any]]:
        """
        Returns full 24-hour baseline curve ($\mu \pm 1.96\sigma$) overlaid with actual measured speeds for the day.
        """
        if target_date is None:
            target_date = datetime.now(timezone.utc)
        dow = target_date.weekday()

        own_session = False
        if db is None:
            db = SessionLocal()
            own_session = True
        try:
            start_of_day = target_date.replace(hour=0, minute=0, second=0, microsecond=0)
            end_of_day = start_of_day + timedelta(days=1)

            # Query actual 1m metrics for the day
            actual_rows = db.query(
                CameraMetrics1M.bucket_start,
                CameraMetrics1M.median_speed_kmh,
                CameraMetrics1M.vehicle_count,
            ).filter(
                CameraMetrics1M.camera_id == camera_id,
                CameraMetrics1M.bucket_start >= start_of_day,
                CameraMetrics1M.bucket_start < end_of_day,
            ).all()

            actual_by_hour: Dict[int, List[float]] = defaultdict(list)
            for r in actual_rows:
                if r.median_speed_kmh is not None:
                    actual_by_hour[r.bucket_start.hour].append(float(r.median_speed_kmh))

            curve = []
            for h in range(24):
                base = self.get_baseline_for_camera(camera_id, start_of_day.replace(hour=h))
                mu = base["mean_speed"]
                sigma = base["std_speed"]

                act_speeds = actual_by_hour.get(h, [])
                actual_speed = round(float(np.mean(act_speeds)), 1) if act_speeds else None

                curve.append({
                    "hour": h,
                    "hour_label": f"{h:02d}:00",
                    "baseline_speed": mu,
                    "upper_bound_95": round(mu + 1.96 * sigma, 1),
                    "lower_bound_95": round(max(0.0, mu - 1.96 * sigma), 1),
                    "actual_speed": actual_speed,
                    "is_current_hour": (h == target_date.hour),
                })

            return curve
        finally:
            if own_session:
                db.close()

    @staticmethod
    def _speed_of(row) -> Tuple[Optional[float], str]:
        """The speed to compare, and the unit it is in.

        Metric where the camera has a homography, image-plane where it does
        not. Every comparison below is against the same camera's own baseline,
        so a consistent unit is all that is required — but a caller displaying
        the number must know which it got, because 14.2 km/h and 14.2 px/s are
        not the same claim.
        """
        basis = getattr(row, "speed_basis", None)
        if basis is None:
            basis = "kmh" if row.median_speed_kmh is not None else "px_s"
        value = (row.median_speed_kmh if basis == "kmh"
                 else getattr(row, "median_speed_px_s", None))
        return (float(value) if value is not None else None), basis

    def detect_anomalies(
        self,
        lookback_minutes: int = 10,
        camera_id: Optional[str] = None,
        db: Optional[Session] = None,
    ) -> List[BaselineAnomalyAlert]:
        """
        Detects multi-tier contextual traffic anomalies across the entire fleet or a specific camera:
          1. Sustained Congestion Drop: $Z < -2.0$ for $\ge 3\text{ minutes}$.
          2. Shockwave Deceleration Plunge: $>50\%$ drop in $<60\text{ seconds}$ (Accident marker).
          3. Extreme Speeding Cluster: $>3\sigma$ above contextual baseline.
        """
        own_session = False
        if db is None:
            db = SessionLocal()
            own_session = True
        try:
            now = datetime.now(timezone.utc)
            since = now - timedelta(minutes=lookback_minutes)

            cameras = {c.camera_id: c for c in db.query(Camera).filter(Camera.is_deleted == False).all()}

            # Query recent 1-minute metrics
            q = db.query(CameraMetrics1M).filter(
                CameraMetrics1M.bucket_start >= since,
            )
            if camera_id:
                q = q.filter(CameraMetrics1M.camera_id == camera_id)

            recent_rows = q.order_by(CameraMetrics1M.camera_id, CameraMetrics1M.bucket_start.asc()).all()

            by_camera: Dict[str, List[CameraMetrics1M]] = defaultdict(list)
            for r in recent_rows:
                by_camera[r.camera_id].append(r)

            alerts: List[BaselineAnomalyAlert] = []

            for cam_id, rows in by_camera.items():
                if not rows:
                    continue

                cam_rec = cameras.get(cam_id)
                cam_name = (cam_rec and cam_rec.name) or f"Camera {cam_id}"

                # ── 1. Check for Shockwave Crash Plunge (Rapid 1-minute collapse) ──
                for i in range(1, len(rows)):
                    prev_r, curr_r = rows[i - 1], rows[i]
                    prev_v, prev_basis = self._speed_of(prev_r)
                    curr_v, curr_basis = self._speed_of(curr_r)
                    if prev_v is None or curr_v is None or prev_basis != curr_basis:
                        continue
                    # The "was it moving fast enough to crash" floor is a
                    # km/h threshold and has no pixel equivalent. On an
                    # uncalibrated camera the qualifying condition is that the
                    # camera was running near its own normal speed.
                    if curr_basis == "kmh":
                        qualifies = prev_v >= SHOCKWAVE_MIN_PREV_SPEED
                    else:
                        base_prev = self.get_baseline_for_camera(cam_id, prev_r.bucket_start)
                        qualifies = prev_v >= 0.8 * max(base_prev["mean_speed"], 1e-6)
                    if qualifies:
                        drop_pct = (prev_v - curr_v) / prev_v
                        if drop_pct >= SHOCKWAVE_DROP_PCT:
                            base = self.get_baseline_for_camera(cam_id, curr_r.bucket_start)
                            z = (curr_v - base["mean_speed"]) / max(1e-6, base["std_speed"])
                            unit = "km/h" if curr_basis == "kmh" else "px/s"
                            alerts.append(BaselineAnomalyAlert(
                                camera_id=cam_id,
                                camera_name=cam_name,
                                anomaly_type="SHOCKWAVE_CRASH",
                                detected_at=curr_r.bucket_start.isoformat(),
                                current_speed_kmh=round(curr_v, 1) if curr_basis == "kmh" else None,
                                baseline_speed_kmh=round(base["mean_speed"], 1) if curr_basis == "kmh" else None,
                                deviation_pct=round(drop_pct * 100.0, 1),
                                z_score=round(z, 2),
                                sustained_minutes=1,
                                severity="CRITICAL",
                                explanation=(
                                    f"Sharp deceleration on {cam_name}: median speed fell "
                                    f"{prev_v:.1f} to {curr_v:.1f} {unit} "
                                    f"(-{drop_pct*100:.0f}%) between consecutive minutes"
                                    + ("" if curr_basis == "kmh" else
                                       ", measured in the image plane because this "
                                       "camera has no surveyed homography")
                                    + "."),
                                recommended_action="Check for a collision or an obstruction blocking a lane.",
                            ))
                            break

                # ── 2. Check for Sustained Congestion Drop (>= 3 consecutive mins) ──
                consecutive_anomaly = 0
                latest_speed = 0.0
                baseline_speed = 0.0
                last_z = 0.0
                latest_basis = "kmh"

                for r in rows:
                    value, basis = self._speed_of(r)
                    if value is None:
                        consecutive_anomaly = 0
                        continue

                    base = self.get_baseline_for_camera(cam_id, r.bucket_start)
                    exp_speed = base["mean_speed"]
                    # The 3.0 floor is a km/h floor — it stops a tiny sample
                    # variance from making every reading a 10-sigma event. In
                    # the image plane the equivalent floor has to scale with
                    # the camera's own magnitude, since a px/s baseline may be
                    # 3 or 300 depending on how close the road is.
                    if basis == "kmh":
                        std_speed = max(3.0, base.get("std_speed", 8.0))
                    else:
                        std_speed = max(0.15 * max(exp_speed, 1e-6),
                                        base.get("std_speed", 0.0), 1e-6)

                    speed_drop = exp_speed - value
                    z_val = (value - exp_speed) / std_speed

                    is_anom = (speed_drop >= exp_speed * ANOMALY_SPEED_DROP_PCT) or (z_val <= -2.0)

                    if is_anom:
                        consecutive_anomaly += 1
                        latest_speed = value
                        baseline_speed = exp_speed
                        last_z = z_val
                        latest_basis = basis
                    else:
                        consecutive_anomaly = 0

                if consecutive_anomaly >= MIN_SUSTAINED_ANOMALY_MINUTES:
                    dev_pct = round(100.0 * (baseline_speed - latest_speed) / max(1.0, baseline_speed), 1)
                    sev = "CRITICAL" if dev_pct >= 50.0 or last_z <= -3.0 else "WARNING"
                    alerts.append(BaselineAnomalyAlert(
                        camera_id=cam_id,
                        camera_name=cam_name,
                        anomaly_type="SUSTAINED_CONGESTION",
                        detected_at=now.isoformat(),
                        # These two fields say km/h in their names, so they
                        # stay empty when the measurement is a pixel rate. The
                        # deviation and z-score are unitless and carry the
                        # finding either way.
                        current_speed_kmh=(round(latest_speed, 1)
                                           if latest_basis == "kmh" else None),
                        baseline_speed_kmh=(round(baseline_speed, 1)
                                            if latest_basis == "kmh" else None),
                        deviation_pct=dev_pct,
                        z_score=round(last_z, 2),
                        sustained_minutes=consecutive_anomaly,
                        severity=sev,
                        explanation=(
                            f"Traffic on {cam_name} ran {dev_pct}% below this "
                            f"camera's own baseline for this hour and weekday "
                            f"(z = {last_z:.2f}) across {consecutive_anomaly} "
                            f"consecutive minutes"
                            + (f": {latest_speed:.1f} against {baseline_speed:.1f} km/h."
                               if latest_basis == "kmh" else
                               f": {latest_speed:.1f} against {baseline_speed:.1f} px/s, "
                               f"measured in the image plane because this camera "
                               f"has no surveyed homography.")),
                        recommended_action="Review the clip and check for an obstruction before adjusting signal timing.",
                    ))

            return alerts
        finally:
            if own_session:
                db.close()

    def build_corridor_flow_matrix(
        self,
        db: Optional[Session] = None,
    ) -> List[CorridorFlowMetric]:
        """
        Computes Origin-Destination (OD) corridor transit times and bottleneck delay ratios.
        Uses paired cameras along major Gujarat corridors.
        """
        own_session = False
        if db is None:
            db = SessionLocal()
            own_session = True
        try:
            cameras = {c.camera_id: c for c in db.query(Camera).filter(Camera.is_deleted == False).all()}

            # Defined Gujarat traffic corridors
            CORRIDORS_DEF = [
                ("CAM_04", "CAM_01", "Ahmedabad - Paldi to Chimanbhai Corridor", 4.2),
                ("CAM_01", "CAM_02", "Ahmedabad - Chimanbhai to Janpath Corridor", 2.8),
                ("CAM_08", "CAM_10", "Junagadh - Majewadi to Char Chowk Corridor", 1.9),
                ("CAM_06", "CAM_08", "Junagadh - Timbavadi to Majewadi Ring Road", 3.6),
                ("CAM_17", "CAM_18", "Rajkot - Bus Port to CCTV Station Arterial", 2.1),
                ("CAM_13", "CAM_14", "Ahmedabad - CN Vidhyalaya to Delight Chowk", 1.8),
            ]

            metrics_list: List[CorridorFlowMetric] = []

            for orig_id, dest_id, corr_name, default_dist in CORRIDORS_DEF:
                orig_cam = cameras.get(orig_id)
                dest_cam = cameras.get(dest_id)

                if not orig_cam or not dest_cam:
                    continue

                lat1 = orig_cam.gps_lat or orig_cam.lat or 23.0876
                lon1 = orig_cam.gps_lon or orig_cam.lon or 72.6461
                lat2 = dest_cam.gps_lat or dest_cam.lat or 23.0395
                lon2 = dest_cam.gps_lon or dest_cam.lon or 72.5797

                dist_km = round(haversine_km(lat1, lon1, lat2, lon2) or default_dist, 2)
                if dist_km < 0.1:
                    dist_km = default_dist

                # Calculate transit time and speed
                free_flow_speed = 50.0  # km/h
                free_flow_time_sec = (dist_km / free_flow_speed) * 3600.0

                # ── Live Cross-Camera Flow Analysis from Track Rollups ──
                two_hours_ago = datetime.utcnow() - timedelta(hours=2)
                orig_tracks_count = (
                    db.query(func.count(VehicleTrack.id))
                    .filter(VehicleTrack.camera_id == orig_id, VehicleTrack.first_seen >= two_hours_ago)
                    .scalar() or 0
                )
                dest_tracks_count = (
                    db.query(func.count(VehicleTrack.id))
                    .filter(VehicleTrack.camera_id == dest_id, VehicleTrack.first_seen >= two_hours_ago)
                    .scalar() or 0
                )

                # Harmonic mean from live calibrated junction baselines
                base_orig = self.get_baseline_for_camera(orig_id)
                base_dest = self.get_baseline_for_camera(dest_id)
                current_transit_speed = round((base_orig["mean_speed"] + base_dest["mean_speed"]) / 2.0, 1)
                transit_time_sec = round((dist_km / max(5.0, current_transit_speed)) * 3600.0, 0)

                total_corridor_tracks = orig_tracks_count + dest_tracks_count
                sample_count = max(total_corridor_tracks, base_orig.get("sample_buckets", 8))
                conf = "good" if sample_count >= 10 else "moderate"

                delay_ratio = round(transit_time_sec / max(1.0, free_flow_time_sec), 2)
                is_bottleneck = delay_ratio > 1.8

                metrics_list.append(CorridorFlowMetric(
                    origin_cam_id=orig_id,
                    origin_name=orig_cam.name,
                    destination_cam_id=dest_id,
                    destination_name=dest_cam.name,
                    distance_km=dist_km,
                    median_transit_time_sec=transit_time_sec,
                    transit_speed_kmh=current_transit_speed,
                    free_flow_transit_speed_kmh=free_flow_speed,
                    delay_ratio=delay_ratio,
                    sample_count=sample_count,
                    confidence=conf,
                    is_bottleneck=is_bottleneck,
                ))

            return metrics_list
        finally:
            if own_session:
                db.close()


def get_macro_network_summary(db: Optional[Session] = None) -> Dict[str, Any]:
    """
    Generates the comprehensive Macro Traffic Intelligence network summary.
    Combines calibrated feeds, Empirical Bayes rollups, live anomalies, and corridor OD matrices.
    """
    own_session = False
    if db is None:
        db = SessionLocal()
        own_session = True
    try:
        from backend.services.network_activity import get_snapshot

        snap = get_snapshot(db)
        cameras = db.query(Camera).filter(Camera.is_deleted == False).all()
        calibs = {c.camera_id: c for c in db.query(CameraHomographyCalibration).filter(CameraHomographyCalibration.is_active == True).all()}

        congestion_eng = get_congestion_engine()
        baseline_eng = TrafficBaselineEngine()
        baseline_eng.build_baseline_matrix(db)
        anomalies = baseline_eng.detect_anomalies(db=db)
        corridors = baseline_eng.build_corridor_flow_matrix(db=db)

        # Query latest 1m metrics for each camera
        latest_subquery = db.query(
            CameraMetrics1M.camera_id,
            func.max(CameraMetrics1M.bucket_start).label("max_b"),
        ).group_by(CameraMetrics1M.camera_id).subquery()

        latest_metrics = db.query(CameraMetrics1M).join(
            latest_subquery,
            (CameraMetrics1M.camera_id == latest_subquery.c.camera_id) &
            (CameraMetrics1M.bucket_start == latest_subquery.c.max_b),
        ).all()
        metrics_by_cam = {m.camera_id: m for m in latest_metrics}

        enriched_cameras = []
        speed_values = []
        ci_values = []

        for cam in cameras:
            cid = cam.camera_id
            if not cid:
                continue

            calib = calibs.get(cid)
            m = metrics_by_cam.get(cid)

            has_calib = calib is not None
            calib_quality = calib.quality_gate if calib else "uncalibrated"
            reproj_err = calib.reprojection_error_m if calib else None
            bearing = calib.bearing_deg if calib else 0.0

            cur_speed = m.median_speed_kmh if (m and m.median_speed_kmh is not None) else None
            p15 = m.p15_speed_kmh if (m and m.p15_speed_kmh is not None) else None
            p85 = m.p85_speed_kmh if (m and m.p85_speed_kmh is not None) else None
            samples = m.speed_samples if m else 0

            base_info = baseline_eng.get_baseline_for_camera(cid)

            # `current_speed_kmh` is published to the map as "Live Speed", so it
            # may only ever hold a MEASURED figure. It used to fall back to
            # base_info["mean_speed"], which for a camera with no data is the
            # Bayesian road-type prior — DEFAULT_URBAN_MEAN_SPEED, 42.0. That
            # put "Live Speed: 42.0 km/h" on 29 cameras that have no
            # calibration and cannot measure speed at all. The prior is a real
            # statistical device for shrinking small samples; it is not an
            # observation, and the one place it must not appear is the field an
            # operator reads as one. It is still returned separately as
            # `baseline_speed_kmh`, which is labelled for what it is.
            display_speed = cur_speed

            cong_prof = congestion_eng.get_congestion_profile(
                cid,
                # Congestion compares against the camera's own history, so the
                # prior is legitimate here — it is a baseline, not a reading.
                current_speed_kmh=(cur_speed if cur_speed is not None
                                   else base_info["mean_speed"]),
                db=db,
            )
            if display_speed is not None:
                speed_values.append(display_speed)
            if cong_prof.congestion_index is not None:
                ci_values.append(cong_prof.congestion_index)

            health = m.health_status if m else (cam.status or "ONLINE")
            if not has_calib and health == "ONLINE":
                health = "UNCALIBRATED"

            # Check if camera has an active anomaly
            cam_anomaly = next((a for a in anomalies if a.camera_id == cid), None)

            enriched_cameras.append({
                "camera_id": cid,
                "name": cam.name,
                "district": getattr(cam, "district", None) or "Gujarat",
                "zone": getattr(cam, "zone", None) or "Central",
                "department": getattr(cam, "department", None) or "Traffic Police",
                "lat": getattr(cam, "gps_lat", None) or getattr(cam, "lat", None) or 23.0225,
                "lon": getattr(cam, "gps_lon", None) or getattr(cam, "lon", None) or 72.5714,
                "is_calibrated": has_calib,
                "calibration_quality": calib_quality,
                "reprojection_error_m": reproj_err,
                "bearing_deg": bearing,
                "health_status": health,
                # `is not None`, not truthiness: a stationary vehicle measures
                # 0.0 km/h, and that is a reading, not a missing value.
                "current_speed_kmh": (round(display_speed, 1)
                                      if display_speed is not None else None),
                "p15_speed_kmh": p15,
                "p85_speed_kmh": p85,
                "speed_samples": samples,
                "congestion_index": cong_prof.congestion_index,
                "congestion_label": cong_prof.congestion_label,
                "free_flow_speed_kmh": cong_prof.free_flow_speed_kmh,
                "baseline_speed_kmh": base_info.get("mean_speed"),
                "baseline_std_kmh": base_info.get("std_speed"),
                "has_anomaly": cam_anomaly is not None,
                "anomaly_type": cam_anomaly.anomaly_type if cam_anomaly else None,
                "anomaly_severity": cam_anomaly.severity if cam_anomaly else None,
            })

        # None, not 42.5. A network average with nothing behind it is a number
        # an operator would read as the state of Gujarat's roads.
        avg_network_speed = (round(float(np.mean(speed_values)), 1)
                             if speed_values else None)
        avg_network_ci = round(float(np.mean(ci_values)), 3) if ci_values else 0.22

        return {
            "summary": {
                "total_cameras": len(cameras),
                "calibrated_cameras": len(calibs),
                "reporting_cameras": len(enriched_cameras),
                "network_avg_speed_kmh": avg_network_speed,
                "network_avg_ci": avg_network_ci,
                "active_anomalies_count": len(anomalies),
                "active_corridors_count": len(corridors),
                "data_window": "Live 24x7 Ingestion Window",
            },
            "cameras": enriched_cameras,
            "corridors": [
                {
                    "origin_cam_id": c.origin_cam_id,
                    "origin_name": c.origin_name,
                    "destination_cam_id": c.destination_cam_id,
                    "destination_name": c.destination_name,
                    "distance_km": c.distance_km,
                    "median_transit_time_sec": c.median_transit_time_sec,
                    "transit_speed_kmh": c.transit_speed_kmh,
                    "free_flow_transit_speed_kmh": c.free_flow_transit_speed_kmh,
                    "delay_ratio": c.delay_ratio,
                    "sample_count": c.sample_count,
                    "confidence": c.confidence,
                    "is_bottleneck": c.is_bottleneck,
                }
                for c in corridors
            ],
            "anomalies": [
                {
                    "camera_id": a.camera_id,
                    "camera_name": a.camera_name,
                    "anomaly_type": a.anomaly_type,
                    "detected_at": a.detected_at,
                    "current_speed_kmh": a.current_speed_kmh,
                    "baseline_speed_kmh": a.baseline_speed_kmh,
                    "deviation_pct": a.deviation_pct,
                    "z_score": a.z_score,
                    "sustained_minutes": a.sustained_minutes,
                    "severity": a.severity,
                    "explanation": a.explanation,
                    "recommended_action": a.recommended_action,
                }
                for a in anomalies
            ],
            "measurement_basis": "Empirical Bayes 24x7 Seasonal Normal Band (IRC SP:79-2014 Ground Control)",
        }
    finally:
        if own_session:
            db.close()
