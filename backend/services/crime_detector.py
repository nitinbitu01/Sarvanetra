import asyncio
import logging
import math
import time
from datetime import datetime
from typing import Dict, List, Optional, Tuple
import numpy as np

logger = logging.getLogger("sentinel.crime_detector")


def haversine_km(lat1: float, lon1: float, lat2: float, lon2: float) -> float:
    R = 6371.0
    dlat = math.radians(lat2 - lat1)
    dlon = math.radians(lon2 - lon1)
    a = (
        math.sin(dlat / 2) ** 2
        + math.cos(math.radians(lat1))
        * math.cos(math.radians(lat2))
        * math.sin(dlon / 2) ** 2
    )
    c = 2 * math.atan2(math.sqrt(a), math.sqrt(1 - a))
    return R * c


class CrimeIncidentDetector:
    """
    7 Core Crime Detection Categories:
    1. WANTED_SUSPECT: Match against wanted criminal embeddings
    2. STOLEN_VEHICLE: Match license plate against stolen vehicle registry
    3. NIGHT_INTRUSION: Human presence in restricted zone during night hours (22:00-05:00)
    4. LOITERING: Person remaining stationary in monitored zone > 5 mins
    5. CROWD_SURGE: Sudden clustering of >= 8 persons (stampede/riot risk)
    6. ABANDONED_OBJECT: Unattended bag/luggage without owner > 2 mins
    7. IMPOSSIBLE_SPEED: Multi-camera velocity anomaly (> 150 km/h)
    """

    def __init__(self, config: dict, camera_registry: dict):
        self.config = config
        self.camera_registry = camera_registry

        # State memory
        self._track_origins: Dict[Tuple[str, int], float] = {}  # (cam, track_id) -> first_seen
        self._person_sightings: Dict[str, List[dict]] = {}       # reid_id -> list of {cam, lat, lon, time}
        self._unattended_objects: Dict[str, float] = {}          # obj_key -> first_unattended_time
        self._alert_cooldowns: Dict[str, float] = {}             # alert_key -> timestamp

    def is_night_time(self, ts: float) -> bool:
        dt = datetime.fromtimestamp(ts)
        night_cfg = self.config.get("danger_scoring", {}).get("night_hours", {"start": 22, "end": 5})
        start_hr = night_cfg.get("start", 22)
        end_hr = night_cfg.get("end", 5)
        return dt.hour >= start_hr or dt.hour < end_hr

    def evaluate_frame_events(
        self,
        camera_id: str,
        tracks: List[dict],
        plates_detected: List[dict],
        reid_results: List[dict],
        timestamp: float,
    ) -> List[dict]:
        """
        Evaluates a single frame and returns any detected crime alerts.
        """
        alerts = []
        cam_info = self.camera_registry.get(camera_id, {})

        # 1. WANTED SUSPECT MATCH
        wanted_alerts = self._check_wanted_suspects(camera_id, cam_info, reid_results, timestamp)
        alerts.extend(wanted_alerts)

        # 2. STOLEN VEHICLE ANPR
        vehicle_alerts = self._check_stolen_vehicles(camera_id, cam_info, plates_detected, timestamp)
        alerts.extend(vehicle_alerts)

        # 3. NIGHT INTRUSION AT RESTRICTED ZONES
        night_alerts = self._check_night_intrusion(camera_id, cam_info, tracks, timestamp)
        alerts.extend(night_alerts)

        # 4. SUSPICIOUS LOITERING
        loiter_alerts = self._check_loitering(camera_id, cam_info, tracks, timestamp)
        alerts.extend(loiter_alerts)

        # 5. CROWD SURGE / STAMPEDE RISK
        crowd_alerts = self._check_crowd_surge(camera_id, cam_info, tracks, timestamp)
        alerts.extend(crowd_alerts)

        # 6. ABANDONED HAZARDOUS OBJECT
        abandon_alerts = self._check_abandoned_objects(camera_id, cam_info, tracks, timestamp)
        alerts.extend(abandon_alerts)

        # 7. IMPOSSIBLE SPEED / IDENTITY SPOOFING
        speed_alerts = self._check_impossible_speed(camera_id, cam_info, reid_results, timestamp)
        alerts.extend(speed_alerts)

        return alerts

    def _check_wanted_suspects(
        self, camera_id: str, cam_info: dict, reid_results: List[dict], ts: float
    ) -> List[dict]:
        alerts = []
        for r in reid_results:
            if r.get("is_wanted"):
                reid_id = r.get("reid_id", "UNKNOWN")
                key = f"wanted_{camera_id}_{reid_id}"
                if ts - self._alert_cooldowns.get(key, 0) > 30:
                    self._alert_cooldowns[key] = ts
                    alerts.append({
                        "crime_type": "WANTED_SUSPECT",
                        "camera_id": camera_id,
                        "reid_id": reid_id,
                        "description": f"Wanted suspect ({reid_id}) spotted at {cam_info.get('name', camera_id)}",
                        "timestamp": ts,
                        "extra": {"match_score": r.get("match_score", 0.95)},
                    })
        return alerts

    def _check_stolen_vehicles(
        self, camera_id: str, cam_info: dict, plates_detected: List[dict], ts: float
    ) -> List[dict]:
        alerts = []
        for p in plates_detected:
            if p.get("is_stolen") or p.get("is_wanted"):
                plate = p.get("plate_text", "")
                key = f"stolen_{camera_id}_{plate}"
                if ts - self._alert_cooldowns.get(key, 0) > 30:
                    self._alert_cooldowns[key] = ts
                    alerts.append({
                        "crime_type": "STOLEN_VEHICLE",
                        "camera_id": camera_id,
                        "plate_text": plate,
                        "description": f"Stolen/Wanted vehicle [{plate}] detected at {cam_info.get('name', camera_id)}",
                        "timestamp": ts,
                        "extra": {"vehicle_type": p.get("vehicle_type", "car")},
                    })
        return alerts

    def _check_night_intrusion(
        self, camera_id: str, cam_info: dict, tracks: List[dict], ts: float
    ) -> List[dict]:
        alerts = []
        if cam_info.get("is_restricted") and self.is_night_time(ts):
            persons = [t for t in tracks if t.get("cls") == 0]
            if len(persons) > 0:
                key = f"night_intrusion_{camera_id}"
                if ts - self._alert_cooldowns.get(key, 0) > 45:
                    self._alert_cooldowns[key] = ts
                    alerts.append({
                        "crime_type": "NIGHT_INTRUSION",
                        "camera_id": camera_id,
                        "description": f"Night-time unauthorized entry ({len(persons)} person(s)) at restricted site {cam_info.get('name', camera_id)}",
                        "timestamp": ts,
                        "extra": {"persons_count": len(persons)},
                    })
        return alerts

    def _check_loitering(
        self, camera_id: str, cam_info: dict, tracks: List[dict], ts: float
    ) -> List[dict]:
        alerts = []
        loiter_thresh = self.config.get("ai_models", {}).get("behavior", {}).get("loitering_min_sec", 300)
        persons = [t for t in tracks if t.get("cls") == 0]

        for p in persons:
            t_id = p.get("track_id")
            k = (camera_id, t_id)
            if k not in self._track_origins:
                self._track_origins[k] = ts
            duration = ts - self._track_origins[k]

            if duration > loiter_thresh:
                key = f"loiter_{camera_id}_{t_id}"
                if ts - self._alert_cooldowns.get(key, 0) > 60:
                    self._alert_cooldowns[key] = ts
                    alerts.append({
                        "crime_type": "LOITERING",
                        "camera_id": camera_id,
                        "description": f"Suspicious loitering (ID {t_id} lingering > {int(duration)}s) at {cam_info.get('name', camera_id)}",
                        "timestamp": ts,
                        "extra": {"duration_sec": duration},
                    })
        return alerts

    def _check_crowd_surge(
        self, camera_id: str, cam_info: dict, tracks: List[dict], ts: float
    ) -> List[dict]:
        alerts = []
        thresh = self.config.get("ai_models", {}).get("behavior", {}).get("crowd_threshold", 8)
        persons = [t for t in tracks if t.get("cls") == 0]

        if len(persons) >= thresh:
            key = f"crowd_{camera_id}"
            if ts - self._alert_cooldowns.get(key, 0) > 40:
                self._alert_cooldowns[key] = ts
                alerts.append({
                    "crime_type": "CROWD_SURGE",
                    "camera_id": camera_id,
                    "description": f"Crowd surge warning ({len(persons)} persons clustered) at {cam_info.get('name', camera_id)}",
                    "timestamp": ts,
                    "extra": {"density": len(persons)},
                })
        return alerts

    def _check_abandoned_objects(
        self, camera_id: str, cam_info: dict, tracks: List[dict], ts: float
    ) -> List[dict]:
        alerts = []
        # Class 24, 26, 28 = backpack, handbag, suitcase in COCO
        objects = [t for t in tracks if t.get("cls") in [24, 26, 28, 7]]
        persons = [t for t in tracks if t.get("cls") == 0]

        abandon_thresh = self.config.get("ai_models", {}).get("behavior", {}).get("abandoned_sec", 120)

        for obj in objects:
            obj_id = obj.get("track_id")
            k = f"{camera_id}_{obj_id}"
            has_nearby_person = len(persons) > 0  # In production, check Euclidean distance to bbox
            if not has_nearby_person:
                if k not in self._unattended_objects:
                    self._unattended_objects[k] = ts
                elif ts - self._unattended_objects[k] > abandon_thresh:
                    key = f"abandoned_{camera_id}_{obj_id}"
                    if ts - self._alert_cooldowns.get(key, 0) > 60:
                        self._alert_cooldowns[key] = ts
                        alerts.append({
                            "crime_type": "ABANDONED_OBJECT",
                            "camera_id": camera_id,
                            "description": f"Unattended bag/object left stationary > 2 mins at {cam_info.get('name', camera_id)}",
                            "timestamp": ts,
                            "extra": {"abandoned_duration": ts - self._unattended_objects[k]},
                        })
            else:
                self._unattended_objects.pop(k, None)

        return alerts

    def _check_impossible_speed(
        self, camera_id: str, cam_info: dict, reid_results: List[dict], ts: float
    ) -> List[dict]:
        alerts = []
        max_speed = self.config.get("ai_models", {}).get("behavior", {}).get("max_speed_kmh", 150)
        lat = cam_info.get("lat")
        lon = cam_info.get("lon")
        if lat is None or lon is None:
            return alerts

        for r in reid_results:
            reid_id = r.get("reid_id")
            if not reid_id:
                continue

            history = self._person_sightings.get(reid_id, [])
            if history:
                prev = history[-1]
                if prev["cam"] != camera_id:
                    dist_km = haversine_km(prev["lat"], prev["lon"], lat, lon)
                    time_diff_hours = max((ts - prev["ts"]) / 3600.0, 0.0001)
                    speed_kmh = dist_km / time_diff_hours

                    if speed_kmh > max_speed and dist_km > 5.0:
                        key = f"speed_{reid_id}_{camera_id}"
                        if ts - self._alert_cooldowns.get(key, 0) > 60:
                            self._alert_cooldowns[key] = ts
                            alerts.append({
                                "crime_type": "IMPOSSIBLE_SPEED",
                                "camera_id": camera_id,
                                "reid_id": reid_id,
                                "description": f"Identity spoofing / impossible velocity: {reid_id} moved {dist_km:.1f}km at {speed_kmh:.0f} km/h between {prev['cam']} and {camera_id}",
                                "timestamp": ts,
                                "extra": {"calculated_speed_kmh": speed_kmh, "distance_km": dist_km},
                            })

            # Record sighting
            history.append({"cam": camera_id, "lat": lat, "lon": lon, "ts": ts})
            if len(history) > 20:
                history.pop(0)
            self._person_sightings[reid_id] = history

        return alerts
