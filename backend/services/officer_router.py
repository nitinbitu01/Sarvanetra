import asyncio
import logging
import math
import os
from datetime import datetime
from typing import Dict, Optional

from backend.metrics.prometheus_metrics import OFFICERS_DISPATCHED

logger = logging.getLogger("sentinel.router")


def haversine(lat1, lon1, lat2, lon2) -> float:
    R = 6371
    d = math.radians
    a = (math.sin(d(lat2 - lat1) / 2) ** 2
         + math.cos(d(lat1)) * math.cos(d(lat2))
         * math.sin(d(lon2 - lon1) / 2) ** 2)
    return R * 2 * math.asin(math.sqrt(a))


HINDI_SCRIPTS = {
    "stolen_vehicle_detected": "Sentinel Gujarat alert. Chori ki gaadi mili hai. Turant karyavahi karein.",
    "person_detected":         "Sentinel Gujarat alert. Suspected vyakti detect hua. Nazdeeki jagah pahunche.",
    "loitering":               "Sentinel Gujarat alert. Suspicious gatividhi — loitering detected.",
    "crowd_anomaly":           "Sentinel Gujarat alert. Asaadharan bheed mili hai.",
    "abandoned_object":        "Sentinel Gujarat alert. Abandoned vastu mila. Jaanch karein.",
    "restricted_zone_approach":"Sentinel Gujarat alert. Restricted kshetra mein pravesh ki koshish.",
    "impossible_speed":        "Sentinel Gujarat alert. Suspicious movement detected.",
}

# In-memory officer roster for zero-dependency dispatch
_OFFICERS = [
    {"id": "OFF_AHM_01", "name": "Inspector R.K. Patel", "badge": "AHM001", "phone": "+919876543210", "lat": 23.0395, "lon": 72.5797, "zone": "Central", "is_available": True},
    {"id": "OFF_AHM_02", "name": "SI M.D. Sharma", "badge": "AHM002", "phone": "+919876543211", "lat": 23.0225, "lon": 72.5714, "zone": "Central", "is_available": True},
    {"id": "OFF_RAJ_01", "name": "Inspector V.M. Mehta", "badge": "RAJ001", "phone": "+919876543213", "lat": 22.3039, "lon": 70.8022, "zone": "Saurashtra", "is_available": True},
    {"id": "OFF_KUT_01", "name": "Inspector S.B. Parmar", "badge": "KUT001", "phone": "+919876543216", "lat": 23.0853, "lon": 70.1337, "zone": "Kutch", "is_available": True},
    {"id": "OFF_SGJ_01", "name": "Inspector A.K. Desai", "badge": "SGJ001", "phone": "+919876543218", "lat": 20.9467, "lon": 72.9520, "zone": "South Gujarat", "is_available": True},
    {"id": "OFF_GAN_01", "name": "Inspector R.N. Rao", "badge": "GAN001", "phone": "+919876543220", "lat": 23.2156, "lon": 72.6369, "zone": "Capital", "is_available": True},
]


class OfficerRouter:
    async def dispatch(
        self,
        lat:          float,
        lon:          float,
        alert_type:   str,
        danger_score: float,
        zone:         str,
        evidence_path: str = None,
    ) -> Dict:
        officer = await self._find_nearest(lat, lon, zone)
        if not officer:
            logger.warning(f"No officer available in zone: {zone}")
            return {"id": None, "name": "Unassigned", "status": "unassigned"}

        await asyncio.gather(
            self._send_push(officer, lat, lon, alert_type, danger_score, evidence_path),
            self._send_tts(officer, alert_type, danger_score),
            self._mark_dispatched(officer["id"]),
        )

        OFFICERS_DISPATCHED.labels(zone=zone or "Gujarat").inc()
        return officer

    async def _find_nearest(self, lat: float, lon: float, zone: str) -> Optional[Dict]:
        available = [o for o in _OFFICERS if o.get("is_available", True)]
        if not available:
            # Fallback to any officer
            available = _OFFICERS

        nearest = min(
            available,
            key=lambda o: haversine(lat, lon, o.get("lat", 0), o.get("lon", 0)),
        )
        return {
            "id":          nearest["id"],
            "name":        nearest["name"],
            "badge":       nearest["badge"],
            "phone":       nearest["phone"],
            "lat":         nearest["lat"],
            "lon":         nearest["lon"],
            "distance_km": haversine(lat, lon, nearest.get("lat", 0), nearest.get("lon", 0)),
        }

    async def _send_push(self, officer, lat, lon, alert_type, score, evidence_path):
        payload = {
            "officer":       officer["id"],
            "title":         f"🚨 SENTINEL ALERT — Score: {score:.1f}",
            "body":          HINDI_SCRIPTS.get(alert_type, "Alert detected"),
            "maps_url":      f"https://maps.google.com/?q={lat},{lon}",
            "evidence":      evidence_path,
            "timestamp":     datetime.utcnow().isoformat(),
        }
        logger.info(
            f"Push → {officer['id']} ({officer['phone']}) | "
            f"{alert_type} | score={score:.1f}"
        )
        return payload

    async def _send_tts(self, officer, alert_type, score):
        script = HINDI_SCRIPTS.get(alert_type, f"Sentinel alert. Score {score:.0f}.")
        try:
            from gTTS import gTTS
            import tempfile
            tts = gTTS(text=script, lang="hi")
            with tempfile.NamedTemporaryFile(suffix=".mp3", delete=False) as f:
                tts.save(f.name)
            logger.info(f"TTS generated: {script[:45]}...")
        except Exception:
            pass

    async def _mark_dispatched(self, officer_id: str):
        for o in _OFFICERS:
            if o["id"] == officer_id:
                o["is_available"] = False
                break
