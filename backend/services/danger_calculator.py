"""
backend/services/danger_calculator.py — Mathematical Threat & Danger Score Engine.

Formula:
  danger_score = base_class_score * M_loiter * M_watchlist * M_crowd * M_zone * M_time

Severity Tiers:
  - CRITICAL: >= 9.0 (Immediate Siren / Red Alert)
  - HIGH:     7.0 to 8.9 (Orange Alert / Priority Dispatch)
  - MEDIUM:   4.0 to 6.9 (Yellow Advisory)
  - LOW:      < 4.0 (Informational)
"""
from __future__ import annotations

from datetime import datetime
from typing import Any


def calculate_danger_score(
    class_name: str = "person",
    dwell_time_seconds: float = 0.0,
    is_watchlist_match: bool = False,
    nearby_crowd_count: int = 1,
    is_restricted_zone: bool = False,
    is_high_crime_zone: bool = False,
    timestamp: datetime | None = None,
) -> dict[str, Any]:
    """Calculate multi-factor danger score and threat category tags.

    Returns:
        {
            "danger_score": float,
            "alert_level": str,
            "threat_tags": list[str],
            "multipliers": dict[str, float]
        }
    """
    # 1. Base Class Score
    base = 1.0
    if class_name in ("car", "motorcycle"):
        base = 0.8
    elif class_name in ("truck", "bus"):
        base = 1.2

    threat_tags = []

    # 2. Loitering Multiplier
    m_loiter = 1.0
    if dwell_time_seconds >= 300.0:
        m_loiter = 2.5
        threat_tags.append("persistent_loitering")
    elif dwell_time_seconds >= 120.0:
        m_loiter = 1.5
        threat_tags.append("mild_loitering")

    # 3. Watchlist Multiplier
    m_watchlist = 1.0
    if is_watchlist_match:
        m_watchlist = 3.0
        threat_tags.append("watchlist_suspect")

    # 4. Crowd Anomaly Multiplier
    m_crowd = 1.0
    if nearby_crowd_count >= 8:
        m_crowd = 2.0
        threat_tags.append("crowd_surge")

    # 5. Zone Multiplier
    m_zone = 1.0
    if is_restricted_zone:
        m_zone = 1.9
        threat_tags.append("restricted_zone_approach")
    elif is_high_crime_zone:
        m_zone = 1.6
        threat_tags.append("high_crime_cluster")

    # 6. Night Multiplier
    m_time = 1.0
    dt = timestamp or datetime.now()
    if dt.hour >= 22 or dt.hour < 5:
        m_time = 1.4
        threat_tags.append("night_activity")

    # Compute Total Score
    total_score = round(base * m_loiter * m_watchlist * m_crowd * m_zone * m_time, 2)

    # Classify Alert Level
    if total_score >= 9.0:
        level = "CRITICAL"
    elif total_score >= 7.0:
        level = "HIGH"
    elif total_score >= 4.0:
        level = "MEDIUM"
    else:
        level = "LOW"

    return {
        "danger_score": total_score,
        "alert_level": level,
        "threat_tags": threat_tags,
        "multipliers": {
            "base": base,
            "loitering": m_loiter,
            "watchlist": m_watchlist,
            "crowd": m_crowd,
            "zone": m_zone,
            "time": m_time,
        },
    }
