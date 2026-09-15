import datetime
import logging
from typing import Any, Dict

logger = logging.getLogger("sentinel.danger_scorer")


class DangerScorer:
    """
    SENTINEL IQ Danger Scorer
    Formula: Final = min(Base * TimeMult * ZoneMult * RestrictedMult * RepeatMult, 10.0)
    """

    def __init__(self, config: dict):
        self.cfg = config.get("danger_scoring", {})
        self.base_weights = self.cfg.get(
            "base_weights",
            {
                "wanted_person": 4.0,
                "WANTED_SUSPECT": 4.0,
                "stolen_vehicle": 3.5,
                "STOLEN_VEHICLE": 3.5,
                "night_intrusion": 3.0,
                "NIGHT_INTRUSION": 3.0,
                "loitering": 1.5,
                "LOITERING": 1.5,
                "crowd_anomaly": 2.0,
                "CROWD_SURGE": 2.0,
                "abandoned_object": 2.5,
                "ABANDONED_OBJECT": 2.5,
                "impossible_speed": 2.8,
                "IMPOSSIBLE_SPEED": 2.8,
            },
        )
        self.multipliers = self.cfg.get(
            "multipliers",
            {
                "night": 1.4,
                "high_crime": 1.6,
                "restricted": 1.9,
                "repeat_offender": 2.0,
            },
        )
        self.night_start = self.cfg.get("night_hours", {}).get("start", 22)
        self.night_end = self.cfg.get("night_hours", {}).get("end", 5)

    def is_night(self, dt: datetime.datetime) -> bool:
        return dt.hour >= self.night_start or dt.hour < self.night_end

    def compute(
        self, crime_type: str, cam: Any, timestamp_or_epoch: Any, extra: Dict = None
    ) -> Dict[str, Any]:
        """Synchronous computation"""
        extra = extra or {}
        if isinstance(timestamp_or_epoch, (int, float)):
            dt = datetime.datetime.fromtimestamp(timestamp_or_epoch)
        elif isinstance(timestamp_or_epoch, datetime.datetime):
            dt = timestamp_or_epoch
        else:
            dt = datetime.datetime.utcnow()

        # Base score
        crime_key = str(crime_type).lower().replace(" ", "_")
        base = self.base_weights.get(crime_type, self.base_weights.get(crime_key, 3.0))

        # Time multiplier
        time_mult = self.multipliers.get("night", 1.4) if self.is_night(dt) else 1.0

        # Zone crime multiplier
        crime_lvl = str(
            getattr(cam, "crime_level", None)
            or (cam.get("crime_level") if isinstance(cam, dict) else "medium")
        ).lower()
        zone_mult = self.multipliers.get("high_crime", 1.6) if crime_lvl == "high" else 1.0

        # Restricted multiplier
        is_restr = bool(
            getattr(cam, "is_restricted", False)
            or (cam.get("is_restricted", False) if isinstance(cam, dict) else False)
        )
        restr_mult = self.multipliers.get("restricted", 1.9) if is_restr else 1.0

        # Repeat offender multiplier
        repeat_mult = self.multipliers.get("repeat_offender", 2.0) if extra.get("is_repeat") else 1.0

        raw_score = base * time_mult * zone_mult * restr_mult * repeat_mult
        final_score = round(min(10.0, max(1.0, raw_score)), 1)

        # Severity mapping
        if final_score >= 9.0:
            severity = "critical"
        elif final_score >= 7.0:
            severity = "high"
        elif final_score >= 4.0:
            severity = "medium"
        else:
            severity = "low"

        return {
            "final_score": final_score,
            "severity": severity,
            "breakdown": {
                "base_score": base,
                "time_mult": time_mult,
                "zone_mult": zone_mult,
                "restr_mult": restr_mult,
                "repeat_mult": repeat_mult,
                "raw_score": raw_score,
            },
        }

    async def score(
        self, crime_type: str, cam: Any, timestamp: datetime.datetime, extra: Dict = None
    ) -> Dict[str, Any]:
        """Async compatibility wrapper"""
        return self.compute(crime_type, cam, timestamp, extra)
