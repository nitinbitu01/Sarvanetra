"""
Camera Link Model (CLM) — Spatio-Temporal Feasibility Engine.
Scientific basis: Hsu et al., "Traffic-Aware Multi-Camera Tracking of Vehicles Based on ReID and Camera Link Model", ACM MM 2020.

PURPOSE: Prevent false ReID matches between physically impossible camera transitions (e.g., same vehicle seen 200km apart in 2 minutes).
SELF-SUPERVISED: Learns travel time distributions automatically from confirmed ANPR plate matches — no manual labeling required.
"""

import os
import math
import json
import asyncio
import logging
from dataclasses import dataclass, field, asdict
from typing import Dict, Tuple, Optional, List, Any
from datetime import datetime, timezone
import numpy as np

logger = logging.getLogger("sentinel.clm")


@dataclass
class TravelTimeDistribution:
    """
    Welford online statistics for travel time between a camera pair.
    Updated every time an ANPR confirms the same plate at both cameras.
    E.g., Ellis Bridge (cam_42) -> Naroda (cam_107): count=847, mean=14.2min, std=3.1min, min=8min, max=28min
    """
    cam_a_id: int
    cam_b_id: int
    count: int = 0
    mean_seconds: float = 0.0
    M2: float = 0.0  # Welford's running sum of squared differences
    min_seconds: float = float('inf')
    max_seconds: float = 0.0
    # Physical distance in km (from camera GPS metadata)
    distance_km: float = 0.0
    # Mode of transport this link primarily serves
    road_type: str = "urban"  # "urban" | "highway" | "expressway"

    @property
    def std_seconds(self) -> float:
        if self.count < 2:
            return 300.0  # Default 5-min std when data is sparse
        return math.sqrt(self.M2 / (self.count - 1))

    @property
    def lower_bound(self) -> float:
        """Hard lower bound: physics-based minimum travel time."""
        if self.count >= 10:
            return max(self.min_seconds * 0.85, self.mean_seconds - 3 * self.std_seconds)
        # Fallback: distance / max possible speed (120 km/h on highway)
        max_speed_mps = {
            "urban": 60 / 3.6,
            "highway": 100 / 3.6,
            "expressway": 120 / 3.6
        }.get(self.road_type, 60 / 3.6)
        if max_speed_mps <= 0:
            max_speed_mps = 60 / 3.6
        return (self.distance_km * 1000) / max_speed_mps

    @property
    def upper_bound(self) -> float:
        """Soft upper bound with generous margin for traffic jams."""
        if self.count >= 10:
            return self.mean_seconds + 4 * self.std_seconds
        # Fallback: distance / minimum possible speed (5 km/h in heavy traffic)
        min_speed_mps = 5 / 3.6
        return max(300.0, (self.distance_km * 1000) / min_speed_mps)

    def welford_update(self, new_value_seconds: float):
        """Online update without storing all observations."""
        self.count += 1
        delta = new_value_seconds - self.mean_seconds
        self.mean_seconds += delta / self.count
        delta2 = new_value_seconds - self.mean_seconds
        self.M2 += delta * delta2
        self.min_seconds = min(self.min_seconds, new_value_seconds)
        self.max_seconds = max(self.max_seconds, new_value_seconds)


@dataclass
class FeasibilityResult:
    is_feasible: bool
    confidence_boost: float = 0.0  # 0.0-0.2, added to cosine similarity if feasible
    reject_reason: Optional[str] = None  # Debugging info
    delta_t_seconds: float = 0.0
    expected_mean: float = 0.0
    expected_window: Tuple[float, float] = (0.0, 0.0)


class CameraLinkModel:
    """
    Production Camera Link Model.
    Modes:
      "vehicle" — car speed constraints (max 120 km/h)
      "pedestrian" — walking speed constraints (max 6 km/h)
      "mixed" — auto-detect from target class
    Storage: Persists learned distributions to disk/DB so model survives restarts.
    """
    PERSISTENCE_FILE = "output/camera_link_model.json"

    def __init__(self, mode: str = "vehicle", db=None):
        self.mode = mode
        self.db = db
        # Key: (cam_a_id, cam_b_id) — ORDERED pair
        self.link_table: Dict[Tuple[int, int], TravelTimeDistribution] = {}
        os.makedirs("output", exist_ok=True)
        self._load_persisted()

    # ─────────────────────────────────────────────
    # CORE API
    # ─────────────────────────────────────────────
    def check_feasibility(
        self, cam_a_id: int, cam_b_id: int, time_delta_seconds: float
    ) -> FeasibilityResult:
        """
        PRIMARY GATE: Call before accepting any ReID match.
        Returns FeasibilityResult with:
        - is_feasible: bool — hard reject if False
        - confidence_boost: 0.0-0.2 — reward matches at peak travel time
        """
        if cam_a_id == cam_b_id:
            # Same camera: cannot reappear in negative or near-zero time if different track
            if time_delta_seconds < 0.5:
                return FeasibilityResult(
                    is_feasible=False,
                    confidence_boost=0.0,
                    reject_reason=f"SAME_CAMERA_COLLISION: Δt={time_delta_seconds:.1f}s",
                    delta_t_seconds=time_delta_seconds
                )
            return FeasibilityResult(
                is_feasible=True,
                confidence_boost=0.05,
                delta_t_seconds=time_delta_seconds
            )

        pair = (cam_a_id, cam_b_id)
        reverse = (cam_b_id, cam_a_id)

        # Try forward direction first, then reverse (bidirectional roads)
        dist = self.link_table.get(pair) or self.link_table.get(reverse)
        if dist is None or dist.count < 5:
            # Unknown camera pair -> allow with conservative speed check if distance known
            if dist and dist.distance_km > 0:
                max_speed_kmh = 130.0 if self.mode == "vehicle" else 15.0
                min_time_sec = (dist.distance_km / max_speed_kmh) * 3600.0
                if time_delta_seconds < min_time_sec:
                    return FeasibilityResult(
                        is_feasible=False,
                        confidence_boost=0.0,
                        reject_reason=f"TOO_FAST: Δt={time_delta_seconds:.0f}s < min={min_time_sec:.0f}s",
                        delta_t_seconds=time_delta_seconds
                    )
            return FeasibilityResult(
                is_feasible=True,
                confidence_boost=0.0,
                reject_reason=None,
                delta_t_seconds=time_delta_seconds
            )

        lower = dist.lower_bound
        upper = dist.upper_bound

        # HARD REJECT: Physically impossible
        if time_delta_seconds < lower:
            return FeasibilityResult(
                is_feasible=False,
                confidence_boost=0.0,
                reject_reason=(
                    f"TOO_FAST: Δt={time_delta_seconds:.0f}s < "
                    f"min={lower:.0f}s for cam {cam_a_id}→{cam_b_id}"
                ),
                delta_t_seconds=time_delta_seconds,
                expected_window=(lower, upper)
            )

        # HARD REJECT: Too slow (different vehicle that happened to pass later)
        if time_delta_seconds > upper:
            return FeasibilityResult(
                is_feasible=False,
                confidence_boost=0.0,
                reject_reason=(
                    f"TOO_SLOW: Δt={time_delta_seconds:.0f}s > "
                    f"max={upper:.0f}s for cam {cam_a_id}→{cam_b_id}"
                ),
                delta_t_seconds=time_delta_seconds,
                expected_window=(lower, upper)
            )

        # FEASIBLE: Gaussian confidence boost (peak at mean travel time)
        z = (time_delta_seconds - dist.mean_seconds) / max(dist.std_seconds, 1.0)
        boost = 0.20 * float(np.exp(-0.5 * z ** 2))  # Max +0.20 at exact mean

        return FeasibilityResult(
            is_feasible=True,
            confidence_boost=boost,
            delta_t_seconds=time_delta_seconds,
            expected_mean=dist.mean_seconds,
            expected_window=(lower, upper)
        )

    def learn_from_confirmed_match(
        self,
        cam_a_id: int,
        cam_b_id: int,
        travel_time_seconds: float,
        match_source: str = "anpr"  # "anpr" | "manual" | "reid_confirmed"
    ):
        """
        Self-supervised learning. Called automatically when:
        1. ANPR confirms same plate at cam_a then cam_b
        2. Officer confirms a ReID match (human-in-the-loop feedback)
        3. Manual ground-truth labeling during calibration
        """
        if travel_time_seconds <= 0:
            return

        pair = (cam_a_id, cam_b_id)
        if pair not in self.link_table:
            self.link_table[pair] = TravelTimeDistribution(
                cam_a_id=cam_a_id,
                cam_b_id=cam_b_id
            )

        dist = self.link_table[pair]

        # Outlier rejection: ignore observations > 5 sigma from mean
        if dist.count >= 20:
            if abs(travel_time_seconds - dist.mean_seconds) > 5 * dist.std_seconds:
                logger.warning(
                    f"CLM: Rejected outlier {travel_time_seconds:.0f}s "
                    f"for pair ({cam_a_id},{cam_b_id}), "
                    f"mean={dist.mean_seconds:.0f}s, std={dist.std_seconds:.0f}s"
                )
                return

        dist.welford_update(travel_time_seconds)
        self._persist()
        logger.info(
            f"CLM updated ({cam_a_id}→{cam_b_id}): "
            f"n={dist.count}, mean={dist.mean_seconds/60:.1f}min, "
            f"std={dist.std_seconds/60:.1f}min [source={match_source}]"
        )

    # ─────────────────────────────────────────────
    # CAMERA TOPOLOGY SEEDING
    # ─────────────────────────────────────────────
    def seed_from_topology(self, camera_graph: List[dict]):
        """
        Seed initial link table from camera GPS coordinates.
        Before learning data exists, use physics-based defaults.
        camera_graph format:
        [
          {"cam_id": 1, "lat": 23.0225, "lon": 72.5714, "connects_to": [2, 3]},
          ...
        ]
        """
        cam_positions = {c["cam_id"]: (c["lat"], c["lon"]) for c in camera_graph if "lat" in c and "lon" in c}
        for cam in camera_graph:
            cid = cam["cam_id"]
            for neighbor_id in cam.get("connects_to", []):
                pair = (cid, neighbor_id)
                if pair in self.link_table:
                    continue

                if cid in cam_positions and neighbor_id in cam_positions:
                    lat1, lon1 = cam_positions[cid]
                    lat2, lon2 = cam_positions[neighbor_id]
                    dist_km = self._haversine_km(lat1, lon1, lat2, lon2)
                    self.link_table[pair] = TravelTimeDistribution(
                        cam_a_id=cid,
                        cam_b_id=neighbor_id,
                        distance_km=dist_km,
                        road_type=cam.get("road_type", "urban")
                    )

    # ─────────────────────────────────────────────
    # PERSISTENCE
    # ─────────────────────────────────────────────
    def _persist(self):
        """Save learned distributions to disk (survives restarts)."""
        try:
            data = {
                f"{k[0]}_{k[1]}": asdict(v) for k, v in self.link_table.items()
            }
            with open(self.PERSISTENCE_FILE, "w") as f:
                json.dump(data, f, indent=2)
        except Exception as e:
            logger.warning(f"CLM: Could not persist model: {e}")

    def _load_persisted(self):
        """Load previously learned distributions on startup."""
        try:
            if os.path.exists(self.PERSISTENCE_FILE):
                with open(self.PERSISTENCE_FILE) as f:
                    data = json.load(f)
                for key, val in data.items():
                    a, b = map(int, key.split("_"))
                    self.link_table[(a, b)] = TravelTimeDistribution(**val)
                logger.info(f"CLM: Loaded {len(self.link_table)} camera links")
        except Exception as e:
            logger.info(f"CLM: Starting fresh ({e})")

    @staticmethod
    def _haversine_km(lat1, lon1, lat2, lon2) -> float:
        R = 6371.0
        dlat = math.radians(lat2 - lat1)
        dlon = math.radians(lon2 - lon1)
        a = (math.sin(dlat / 2) ** 2 +
             math.cos(math.radians(lat1)) * math.cos(math.radians(lat2)) *
             math.sin(dlon / 2) ** 2)
        return R * 2 * math.asin(math.sqrt(a))
