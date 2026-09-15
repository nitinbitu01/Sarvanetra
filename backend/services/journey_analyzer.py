import logging
import math
from datetime import datetime, timedelta
from typing import Any

from backend.metrics.prometheus_metrics import JOURNEY_ANOMALIES

logger = logging.getLogger("sentinel.journey")

MAX_SPEED_KMH      = 150
SUSPICIOUS_VISITS  = 3
ANALYSIS_WINDOW_HR = 1
HISTORY_LIMIT      = 50


def haversine(lat1, lon1, lat2, lon2) -> float:
    R = 6371
    d_lat = math.radians(lat2 - lat1)
    d_lon = math.radians(lon2 - lon1)
    a = (math.sin(d_lat / 2) ** 2
         + math.cos(math.radians(lat1)) * math.cos(math.radians(lat2))
         * math.sin(d_lon / 2) ** 2)
    return R * 2 * math.asin(math.sqrt(a))


class JourneyAnalyzer:
    def __init__(self, camera_registry: dict = None):
        self.registry = camera_registry or {}
        self._in_memory_journeys: dict[str, list] = {}

    async def analyze(
        self, reid_id: str, new_event: dict
    ) -> list[dict[str, Any]]:
        anomalies = []
        history   = await self._load_history(reid_id)

        if not history:
            self._append_history(reid_id, new_event)
            return anomalies

        last = history[-1]

        # ── Speed Check ───────────────────────────────────────
        t_diff_h = max(
            (new_event["timestamp"] - last["timestamp"]).total_seconds() / 3600,
            1e-6,
        )
        dist_km = haversine(last["lat"], last["lon"], new_event["lat"], new_event["lon"])
        speed   = dist_km / t_diff_h

        if speed > MAX_SPEED_KMH and dist_km > 1.0:
            anomaly = {
                "type":     "impossible_speed",
                "severity": "high",
                "detail":   (
                    f"{dist_km:.1f} km in {t_diff_h*60:.0f} min "
                    f"= {speed:.0f} km/h — impossible velocity"
                ),
                "reid_id":      reid_id,
                "from_camera":  last["camera_id"],
                "to_camera":    new_event["camera_id"],
            }
            anomalies.append(anomaly)
            JOURNEY_ANOMALIES.labels(anomaly_type="impossible_speed").inc()

        # ── High-Crime Clustering ─────────────────────────────
        window_start = new_event["timestamp"] - timedelta(hours=ANALYSIS_WINDOW_HR)
        recent = [e for e in history if e["timestamp"] >= window_start]
        hc_visits = [
            e for e in recent
            if self.registry.get(e["camera_id"], {}).get("crime_level") == "high"
        ]
        if len(hc_visits) >= SUSPICIOUS_VISITS:
            anomaly = {
                "type":      "high_crime_clustering",
                "severity":  "medium",
                "detail":    f"{len(hc_visits)} high-crime locations visited in 1 hour",
                "reid_id":   reid_id,
                "locations": [e["camera_id"] for e in hc_visits],
            }
            anomalies.append(anomaly)
            JOURNEY_ANOMALIES.labels(anomaly_type="high_crime_clustering").inc()

        # ── Restricted Zone Approach ──────────────────────────
        cam_info = self.registry.get(new_event["camera_id"], {})
        if cam_info.get("is_restricted"):
            anomaly = {
                "type":     "restricted_zone_approach",
                "severity": "high",
                "detail":   f"Approached restricted zone: {cam_info.get('name', '')}",
                "reid_id":  reid_id,
            }
            anomalies.append(anomaly)
            JOURNEY_ANOMALIES.labels(anomaly_type="restricted_zone_approach").inc()

        self._append_history(reid_id, new_event)
        return anomalies

    def _append_history(self, reid_id: str, event: dict):
        if reid_id not in self._in_memory_journeys:
            self._in_memory_journeys[reid_id] = []
        self._in_memory_journeys[reid_id].append(event)
        if len(self._in_memory_journeys[reid_id]) > HISTORY_LIMIT:
            self._in_memory_journeys[reid_id].pop(0)

    async def _load_history(self, reid_id: str) -> list:
        return self._in_memory_journeys.get(reid_id, [])
