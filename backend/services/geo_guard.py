"""
backend/services/geo_guard.py — Spatial-Temporal Road Graph & Geographic Cluster Guard.

Key Capabilities:
  1. Geodesic Haversine Distance computation between camera coordinates.
  2. Velocity validation: V <= 120 km/h (rejects impossible teleportations).
  3. Minimum feasible transit time gate (prevents same-person 2km jumps in 10 seconds).
  4. Regional cluster boundaries (Ahmedabad, Junagadh, Navsari, Gandhinagar, etc.).
  5. Pre-search reachable cluster calculation to prune FAISS candidate sub-spaces.
"""
from __future__ import annotations

import logging
import math
from datetime import datetime
from typing import Any

import yaml

logger = logging.getLogger(__name__)

EARTH_RADIUS_KM = 6371.0
DEFAULT_MAX_VELOCITY_KMH = 120.0
INTER_CLUSTER_MIN_HOURS = 2.0


def haversine_distance(lat1: float, lon1: float, lat2: float, lon2: float) -> float:
    """Compute Haversine great-circle distance in kilometers between two GPS points."""
    phi1, phi2 = math.radians(lat1), math.radians(lat2)
    delta_phi = math.radians(lat2 - lat1)
    delta_lambda = math.radians(lon2 - lon1)

    a = (math.sin(delta_phi / 2.0) ** 2 +
         math.cos(phi1) * math.cos(phi2) * math.sin(delta_lambda / 2.0) ** 2)
    c = 2.0 * math.atan2(math.sqrt(a), math.sqrt(1.0 - a))
    return EARTH_RADIUS_KM * c


class GeoGuard:
    """Production Spatial-Temporal Validation Engine for Multi-Camera Tracking."""

    def __init__(self, config_path: str = "config.yaml") -> None:
        self.camera_map: dict[str, dict[str, Any]] = {}
        self.cluster_map: dict[str, str] = {}  # camera_id -> district/zone
        self._load_config(config_path)

    def _load_config(self, config_path: str) -> None:
        try:
            with open(config_path, "r", encoding="utf-8") as f:
                cfg = yaml.safe_load(f)
            
            cams = cfg.get("demo_cameras", [])
            for c in cams:
                cid = c.get("id")
                if not cid:
                    continue
                self.camera_map[cid] = {
                    "id": cid,
                    "name": c.get("name", cid),
                    "lat": float(c.get("lat", 23.0)),
                    "lon": float(c.get("lon", 72.5)),
                    "district": c.get("district", "Unknown"),
                    "zone": c.get("zone", "Default"),
                }
                self.cluster_map[cid] = c.get("district", "Unknown")
        except Exception as e:
            logger.warning("Failed to load camera GPS config from %s: %s", config_path, e)

    def get_camera(self, camera_id: str) -> dict[str, Any] | None:
        return self.camera_map.get(camera_id)

    def get_distance(self, cam_id_a: str, cam_id_b: str) -> float:
        if cam_id_a == cam_id_b:
            return 0.0
        c1 = self.camera_map.get(cam_id_a)
        c2 = self.camera_map.get(cam_id_b)
        if not c1 or not c2:
            return 0.0
        return haversine_distance(c1["lat"], c1["lon"], c2["lat"], c2["lon"])

    def is_transit_feasible(
        self,
        from_camera_id: str,
        to_camera_id: str,
        time_gap_seconds: float,
        v_max_kmh: float = DEFAULT_MAX_VELOCITY_KMH,
    ) -> tuple[bool, str]:
        """Validate if a suspect could physically travel between two cameras in time_gap_seconds.

        Returns:
            (is_valid: bool, reason: str)
        """
        if from_camera_id == to_camera_id:
            return True, "SAME_CAMERA"

        dist_km = self.get_distance(from_camera_id, to_camera_id)
        if dist_km <= 0.01:
            return True, "CO_LOCATED"

        time_gap_hours = max(time_gap_seconds, 1.0) / 3600.0
        speed_kmh = dist_km / time_gap_hours

        # 1. Cluster boundary check
        cluster_a = self.cluster_map.get(from_camera_id, "A")
        cluster_b = self.cluster_map.get(to_camera_id, "B")
        if cluster_a != cluster_b and time_gap_hours < INTER_CLUSTER_MIN_HOURS:
            return (
                False,
                f"INTER_CLUSTER_VIOLATION: {cluster_a}->{cluster_b} in {time_gap_seconds:.0f}s (< {INTER_CLUSTER_MIN_HOURS*3600:.0f}s)"
            )

        # 2. Maximum velocity check
        if speed_kmh > v_max_kmh:
            return (
                False,
                f"SPEED_VIOLATION: Required speed {speed_kmh:.1f} km/h exceeds limit {v_max_kmh} km/h (dist={dist_km:.2f}km, dt={time_gap_seconds:.0f}s)"
            )

        return True, f"VALID_TRANSIT: {speed_kmh:.1f} km/h (dist={dist_km:.2f}km)"

    def get_reachable_clusters(
        self, current_camera_id: str, time_gap_hours: float, v_max_kmh: float = DEFAULT_MAX_VELOCITY_KMH
    ) -> list[str]:
        """Compute all district clusters reachable from current_camera_id within time_gap_hours."""
        curr = self.camera_map.get(current_camera_id)
        if not curr:
            return []
        
        reachable = set([curr["district"]])
        if time_gap_hours >= INTER_CLUSTER_MIN_HOURS:
            max_range_km = time_gap_hours * v_max_kmh
            for cid, info in self.camera_map.items():
                if haversine_distance(curr["lat"], curr["lon"], info["lat"], info["lon"]) <= max_range_km:
                    reachable.add(info["district"])
        return list(reachable)


# Global singleton instance
_geo_guard: GeoGuard | None = None

def get_geo_guard() -> GeoGuard:
    global _geo_guard
    if _geo_guard is None:
        _geo_guard = GeoGuard()
    return _geo_guard
