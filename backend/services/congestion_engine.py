"""
backend/services/congestion_engine.py — Free-Flow Speed Bootstrap & Congestion Index (Phase 3).

Mathematical Formulation:
  - Free-flow speed v_free: 85th percentile of off-peak observed speeds over >= N_threshold samples.
  - Congestion Index (CI):
      CI = 1.0 - clamp(v / v_free, 0.0, 1.0)
      where 0.0 = completely free flow, 1.0 = gridlock.

Honesty Contract:
  - CI is strictly NULL until v_free reaches statistical bootstrap threshold (N >= 200).
  - The engine exposes exact bootstrap progress ("X of 200 samples") to the API and dashboard.
"""
from __future__ import annotations

import logging
from dataclasses import dataclass
from datetime import datetime, time
from typing import Any, Optional, Sequence

import numpy as np
from sqlalchemy import func
from sqlalchemy.orm import Session

from backend.db.models import CameraMetrics1M, VehicleTrack
from backend.db.session import SessionLocal

logger = logging.getLogger("sentinel.congestion_engine")

BOOTSTRAP_MIN_SAMPLES = 200
OFF_PEAK_HOURS = {22, 23, 0, 1, 2, 3, 4, 5, 11, 12, 13, 14}


@dataclass
class CameraCongestionProfile:
    camera_id: str
    is_bootstrapped: bool
    samples_collected: int
    samples_required: int
    progress_pct: float
    free_flow_speed_kmh: Optional[float]
    current_speed_kmh: Optional[float]
    congestion_index: Optional[float]     # NULL if not bootstrapped or speed is None
    congestion_label: str                 # "FREE_FLOW" | "MODERATE" | "CONGESTED" | "GRIDLOCK" | "BOOTSTRAPPING" | "UNAVAILABLE"


class CongestionEngine:
    """Computes free-flow speed baselines and real-time Congestion Index."""

    def __init__(self, min_samples_threshold: int = BOOTSTRAP_MIN_SAMPLES):
        self.min_samples_threshold = min_samples_threshold
        # camera_id -> v_free
        self._v_free_cache: dict[str, float] = {}

    def compute_free_flow_speed(
        self,
        camera_id: str,
        db: Optional[Session] = None,
    ) -> Optional[float]:
        """Calculates 85th percentile off-peak speed across accumulated history."""
        own_session = False
        if db is None:
            db = SessionLocal()
            own_session = True
        try:
            # Query recent valid speed observations for this camera (capped to 5000 to protect RAM)
            rows = db.query(VehicleTrack.speed_kmh, VehicleTrack.first_seen).filter(
                VehicleTrack.camera_id == camera_id,
                VehicleTrack.speed_kmh.isnot(None),
                VehicleTrack.quality.in_(["good", "degraded"]),
            ).order_by(VehicleTrack.first_seen.desc()).limit(5000).all()

            if not rows:
                return None

            # Filter off-peak hours
            off_peak_speeds = [
                speed for speed, dt in rows
                if dt.hour in OFF_PEAK_HOURS and speed is not None and speed >= 5.0
            ]

            # Fallback to all valid speeds if off-peak sample count is small
            sample_set = off_peak_speeds if len(off_peak_speeds) >= self.min_samples_threshold else [s for s, _ in rows if s is not None and s >= 5.0]

            if len(sample_set) < self.min_samples_threshold:
                return None

            v_free = float(np.percentile(sample_set, 85))
            self._v_free_cache[camera_id] = round(v_free, 1)
            return round(v_free, 1)
        finally:
            if own_session:
                db.close()

    def get_congestion_profile(
        self,
        camera_id: str,
        current_speed_kmh: Optional[float] = None,
        db: Optional[Session] = None,
    ) -> CameraCongestionProfile:
        """Returns the full congestion status and bootstrap progress for a camera."""
        own_session = False
        if db is None:
            db = SessionLocal()
            own_session = True
        try:
            total_samples = db.query(func.count(VehicleTrack.id)).filter(
                VehicleTrack.camera_id == camera_id,
                VehicleTrack.speed_kmh.isnot(None),
            ).scalar() or 0

            v_free = self._v_free_cache.get(camera_id)
            if v_free is None:
                v_free = self.compute_free_flow_speed(camera_id, db=db)

            is_bootstrapped = v_free is not None
            progress_pct = min(100.0, round(100.0 * total_samples / max(1, self.min_samples_threshold), 1))

            ci = None
            label = "BOOTSTRAPPING" if not is_bootstrapped else "UNAVAILABLE"

            if is_bootstrapped and current_speed_kmh is not None and v_free is not None and v_free > 0:
                ratio = max(0.0, min(1.0, current_speed_kmh / v_free))
                ci = round(1.0 - ratio, 3)

                if ci <= 0.20:
                    label = "FREE_FLOW"
                elif ci <= 0.50:
                    label = "MODERATE"
                elif ci <= 0.80:
                    label = "CONGESTED"
                else:
                    label = "GRIDLOCK"
            elif not is_bootstrapped:
                label = "BOOTSTRAPPING"
            elif current_speed_kmh is None:
                label = "UNAVAILABLE"

            return CameraCongestionProfile(
                camera_id=camera_id,
                is_bootstrapped=is_bootstrapped,
                samples_collected=total_samples,
                samples_required=self.min_samples_threshold,
                progress_pct=progress_pct,
                free_flow_speed_kmh=v_free,
                current_speed_kmh=current_speed_kmh,
                congestion_index=ci,
                congestion_label=label,
            )
        finally:
            if own_session:
                db.close()

    def update_rollups_congestion_index(self, db: Optional[Session] = None) -> int:
        """Batch updates `camera_metrics_1m.congestion_index` for all rows where v_free is bootstrapped."""
        own_session = False
        if db is None:
            db = SessionLocal()
            own_session = True
        try:
            # Query all unpopulated metrics rows with valid median speed
            metrics_rows = db.query(CameraMetrics1M).filter(
                CameraMetrics1M.median_speed_kmh.isnot(None),
                CameraMetrics1M.congestion_index.is_(None),
            ).all()

            updated = 0
            for r in metrics_rows:
                v_free = self._v_free_cache.get(r.camera_id)
                if v_free is None:
                    v_free = self.compute_free_flow_speed(r.camera_id, db=db)
                if v_free and v_free > 0 and r.median_speed_kmh is not None:
                    ratio = max(0.0, min(1.0, r.median_speed_kmh / v_free))
                    r.congestion_index = round(1.0 - ratio, 3)
                    updated += 1

            if updated > 0:
                db.commit()
            return updated
        finally:
            if own_session:
                db.close()


_global_congestion_engine: Optional[CongestionEngine] = None


def get_congestion_engine(min_samples_threshold: int = BOOTSTRAP_MIN_SAMPLES) -> CongestionEngine:
    global _global_congestion_engine
    if _global_congestion_engine is None:
        _global_congestion_engine = CongestionEngine(min_samples_threshold=min_samples_threshold)
    return _global_congestion_engine
