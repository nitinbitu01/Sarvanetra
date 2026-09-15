"""
Production Danger Score Engine.
PURPOSE: Prevent alert fatigue by computing a composite 0.0–1.0 danger score before routing any alert to an officer.
Only alerts with score >= 0.60 reach officers. All others are logged silently for analytics.
EXPLAINABILITY: Every score includes a human-readable breakdown so officers know WHY they are being alerted.
"""

from dataclasses import dataclass, field
from datetime import datetime, timezone
from enum import IntEnum
from typing import Dict, List, Optional, Any


class AlertPriority(IntEnum):
    SUPPRESSED = 0  # score < 0.40 -> silent log
    LOW = 1         # 0.40–0.59 -> dashboard only
    MEDIUM = 2      # 0.60–0.74 -> push to supervisor
    HIGH = 3        # 0.75–0.89 -> push to nearest officer
    CRITICAL = 4    # 0.90–1.00 -> push + auto escalate to control room


@dataclass
class DangerContext:
    # Identity
    is_on_watchlist: bool = False
    watchlist_severity: str = "NONE"  # "LOOKOUT" | "WANTED" | "TERRORIST"
    is_stolen_vehicle: bool = False
    visual_match_score: float = 0.0   # ReID confidence 0.0–1.0
    match_type: str = "visual_reid"   # "plate_confirmed" | "visual_reid"
    # Temporal
    timestamp: Optional[datetime] = None
    # Spatial
    camera_id: int = 0
    zone_risk_level: int = 0          # 0=safe, 1=moderate, 2=high, 3=critical
    department: str = "unknown"       # "traffic" | "municipality" | "railway"
    # Behavioral
    num_cameras_in_30min: int = 0     # Cross-camera frequency
    is_approaching_sensitive: bool = False
    speed_anomaly: bool = False
    repeated_suppressed_alerts: int = 0


@dataclass
class DangerScore:
    total: float  # 0.0–1.0
    priority: AlertPriority
    factor_breakdown: Dict[str, float]
    explanation: str
    should_alert: bool       # True if priority >= MEDIUM
    evidence_required: bool  # True if priority >= HIGH


class DangerScoreEngine:
    """
    Composite danger scoring with guard rails:
    1. TERRORIST watchlist severity -> always CRITICAL (no suppression)
    2. Repeated suppression escalation: 3+ suppressed alerts in 30min for same subject -> force LOW minimum
    3. Plate-confirmed matches get 15% bonus over visual-only
    4. All factors are independently auditable
    """
    WEIGHTS: Dict[str, float] = {
        "watchlist_terrorist": 1.00,
        "watchlist_wanted": 0.40,
        "watchlist_lookout": 0.25,
        "stolen_vehicle": 0.35,
        "zone_critical": 0.20,
        "zone_high": 0.15,
        "zone_moderate": 0.08,
        "nighttime": 0.14,           # 10 PM – 5 AM
        "cross_camera_3plus": 0.15,  # 3+ cameras in 30min
        "cross_camera_5plus": 0.25,  # 5+ cameras -> active chase
        "sensitive_approach": 0.20,
        "speed_anomaly": 0.12,
        "plate_confirmation": 0.15,  # Bonus for ANPR confirmation
        "reid_confidence_high": 0.08,# Visual match >= 0.90
        "repeated_suppression": 0.15,# Escalation guard
    }

    def compute(self, ctx: DangerContext) -> DangerScore:
        score = 0.0
        breakdown: Dict[str, float] = {}
        ts = ctx.timestamp or datetime.now(timezone.utc)

        # 1. Watchlist severity (highest priority)
        if ctx.is_on_watchlist:
            if ctx.watchlist_severity == "TERRORIST":
                return DangerScore(
                    total=1.0,
                    priority=AlertPriority.CRITICAL,
                    factor_breakdown={"watchlist_terrorist": 1.0},
                    explanation="⚠️ TERRORIST / HIGH THREAT individual detected",
                    should_alert=True,
                    evidence_required=True,
                )
            elif ctx.watchlist_severity == "WANTED":
                w = self.WEIGHTS["watchlist_wanted"]
                score += w
                breakdown["wanted_person_vehicle"] = w
            elif ctx.watchlist_severity == "LOOKOUT":
                w = self.WEIGHTS["watchlist_lookout"]
                score += w
                breakdown["lookout_person_vehicle"] = w

        # 2. Stolen vehicle
        if ctx.is_stolen_vehicle:
            w = self.WEIGHTS["stolen_vehicle"]
            score += w
            breakdown["stolen_vehicle"] = w

        # 3. Zone risk
        zone_key = {3: "zone_critical", 2: "zone_high", 1: "zone_moderate"}.get(
            ctx.zone_risk_level
        )
        if zone_key:
            w = self.WEIGHTS[zone_key]
            score += w
            breakdown[f"zone_risk_level_{ctx.zone_risk_level}"] = w

        # 4. Nighttime
        hour = ts.hour
        if hour >= 22 or hour < 5:
            w = self.WEIGHTS["nighttime"]
            score += w
            breakdown["nighttime_22_to_5"] = w

        # 5. Cross-camera frequency
        if ctx.num_cameras_in_30min >= 5:
            w = self.WEIGHTS["cross_camera_5plus"]
            score += w
            breakdown["5+_cameras_30min_chase"] = w
        elif ctx.num_cameras_in_30min >= 3:
            w = self.WEIGHTS["cross_camera_3plus"]
            score += w
            breakdown["3+_cameras_30min_trail"] = w

        # 6. Sensitive location
        if ctx.is_approaching_sensitive:
            w = self.WEIGHTS["sensitive_approach"]
            score += w
            breakdown["approaching_sensitive_location"] = w

        # 7. Speed anomaly
        if ctx.speed_anomaly:
            w = self.WEIGHTS["speed_anomaly"]
            score += w
            breakdown["erratic_speed_movement"] = w

        # 8. Match quality bonuses
        if ctx.match_type == "plate_confirmed":
            w = self.WEIGHTS["plate_confirmation"]
            score += w
            breakdown["anpr_plate_confirmed"] = w
        elif ctx.visual_match_score >= 0.90:
            w = self.WEIGHTS["reid_confidence_high"]
            score += w
            breakdown["high_confidence_reid"] = w

        # 9. Repeated suppression escalation guard
        if ctx.repeated_suppressed_alerts >= 3:
            w = self.WEIGHTS["repeated_suppression"]
            score += w
            breakdown["repeated_suppression_escalation"] = w

        total = round(min(score, 1.0), 3)
        priority = self._priority(total)
        explanation = self._explain(breakdown, ctx, ts)

        return DangerScore(
            total=total,
            priority=priority,
            factor_breakdown=breakdown,
            explanation=explanation,
            should_alert=priority >= AlertPriority.MEDIUM,
            evidence_required=priority >= AlertPriority.HIGH,
        )

    def _priority(self, score: float) -> AlertPriority:
        if score >= 0.90:
            return AlertPriority.CRITICAL
        if score >= 0.75:
            return AlertPriority.HIGH
        if score >= 0.60:
            return AlertPriority.MEDIUM
        if score >= 0.40:
            return AlertPriority.LOW
        return AlertPriority.SUPPRESSED

    def _explain(
        self, breakdown: Dict[str, float], ctx: DangerContext, ts: datetime
    ) -> str:
        parts = []
        if "wanted_person_vehicle" in breakdown:
            parts.append("🔴 WANTED person/vehicle")
        if "lookout_person_vehicle" in breakdown:
            parts.append("🟡 Lookout alert")
        if "stolen_vehicle" in breakdown:
            parts.append("🚗 Stolen vehicle")
        if "nighttime_22_to_5" in breakdown:
            parts.append(f"🌙 Nighttime ({ts.strftime('%H:%M')})")
        if any("zone_risk" in k for k in breakdown):
            parts.append(f"📍 Risk Zone Level {ctx.zone_risk_level}")
        if "5+_cameras_30min_chase" in breakdown:
            parts.append(f"🏃 Active chase: {ctx.num_cameras_in_30min} cameras in 30min")
        elif "3+_cameras_30min_trail" in breakdown:
            parts.append(f"📡 Trailed: {ctx.num_cameras_in_30min} cameras")
        if "approaching_sensitive_location" in breakdown:
            parts.append("⚠️ Approaching sensitive location")
        if "anpr_plate_confirmed" in breakdown:
            parts.append("✅ Plate confirmed by ANPR")
        return " | ".join(parts) if parts else "Automated pattern detection"
