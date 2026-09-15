"""
backend/services/cad_dispatch.py — Dial-112 PCR Patrol CAD Dispatch & Telemetry (v15.0.0)

Closes Gap W: Real-time GPS ingestion, stale position detection (> 60s without telemetry),
and nearest-unit allocation using Haversine distance.
"""
from __future__ import annotations

import math
import time
from dataclasses import dataclass, field
from enum import Enum, auto
from typing import Optional


from backend.core.config import settings


class UnitStatus(Enum):
    AVAILABLE = auto()
    DISPATCHED = auto()
    EN_ROUTE = auto()
    ON_SCENE = auto()
    RESOLVED = auto()
    BUSY = auto()
    OFFLINE = auto()


@dataclass
class PCRUnit:
    unit_id: str
    call_sign: str
    city: str
    officer_name: str
    contact_number: str
    lat: float
    lon: float
    status: UnitStatus = UnitStatus.AVAILABLE
    last_gps_update: float = field(default_factory=time.monotonic)
    current_incident: Optional[str] = None

    @property
    def gps_is_fresh(self) -> bool:
        threshold = getattr(settings, "CAD_GPS_STALE_THRESHOLD_SEC", 60.0)
        return (time.monotonic() - self.last_gps_update) < threshold

    def update_gps(self, lat: float, lon: float) -> None:
        self.lat = lat
        self.lon = lon
        self.last_gps_update = time.monotonic()


@dataclass
class DispatchResult:
    unit_id: str
    call_sign: str
    officer_name: str
    contact: str
    distance_km: float
    eta_minutes: float
    status: str


def _haversine_km(lat1: float, lon1: float, lat2: float, lon2: float) -> float:
    R = 6371.0
    d_lat = math.radians(lat2 - lat1)
    d_lon = math.radians(lon2 - lon1)
    a = (
        math.sin(d_lat / 2.0) ** 2
        + math.cos(math.radians(lat1)) * math.cos(math.radians(lat2)) * math.sin(d_lon / 2.0) ** 2
    )
    return R * 2.0 * math.atan2(math.sqrt(a), math.sqrt(max(1.0 - a, 0.0)))


DEFAULT_FLEET: list[PCRUnit] = [
    PCRUnit("PCR-01", "Alpha-1", "Ahmedabad", "Insp. R. Patel", "+91-9900001001", 23.0225, 72.5714),
    PCRUnit("PCR-02", "Alpha-2", "Ahmedabad", "SI D. Shah", "+91-9900001002", 23.0400, 72.5500),
    PCRUnit("PCR-03", "Alpha-3", "Ahmedabad", "Insp. K. Mehta", "+91-9900001003", 23.0100, 72.5900),
    PCRUnit("PCR-04", "Bravo-1", "Ahmedabad", "SI V. Joshi", "+91-9900001004", 23.0600, 72.5300),
    PCRUnit("PCR-05", "Bravo-2", "Surat", "Insp. A. Desai", "+91-9900001005", 21.1702, 72.8311),
    PCRUnit("PCR-06", "Bravo-3", "Surat", "SI P. Trivedi", "+91-9900001006", 21.1800, 72.8200),
    PCRUnit("PCR-07", "Charlie-1", "Vadodara", "Insp. N. Rao", "+91-9900001007", 22.3072, 73.1812),
    PCRUnit("PCR-08", "Charlie-2", "Vadodara", "SI S. Kumar", "+91-9900001008", 22.3200, 73.1700),
    PCRUnit("PCR-09", "Delta-1", "Rajkot", "Insp. H. Bhatt", "+91-9900001009", 22.3039, 70.8022),
    PCRUnit("PCR-10", "Delta-2", "Rajkot", "SI M. Pandya", "+91-9900001010", 22.3150, 70.7900),
    PCRUnit("PCR-11", "Echo-1", "Gandhinagar", "Insp. C. Nair", "+91-9900001011", 23.2156, 72.6369),
    PCRUnit("PCR-12", "Echo-2", "Gandhinagar", "SI B. Sharma", "+91-9900001012", 23.2300, 72.6200),
]


class CADDispatcher:
    """Manages live PCR fleet telemetry and incident dispatching."""

    def __init__(self, fleet: Optional[list[PCRUnit]] = None):
        self.fleet: dict[str, PCRUnit] = {
            u.unit_id: u for u in (fleet or DEFAULT_FLEET)
        }

    def update_unit_gps(self, unit_id: str, lat: float, lon: float) -> bool:
        if unit_id in self.fleet:
            self.fleet[unit_id].update_gps(lat, lon)
            return True
        return False

    def update_unit_status(
        self, unit_id: str, status: UnitStatus, incident_id: Optional[str] = None
    ) -> bool:
        if unit_id not in self.fleet:
            return False
        self.fleet[unit_id].status = status
        self.fleet[unit_id].current_incident = incident_id
        return True

    def dispatch_nearest(
        self,
        incident_lat: float,
        incident_lon: float,
        severity: str = "HIGH",
        description: str = "",
        incident_id: str = "",
    ) -> Optional[DispatchResult]:
        urban_speed_kmh = getattr(settings, "CAD_URBAN_SPEED_KMH", 30.0)
        best_unit: Optional[PCRUnit] = None
        best_dist = float("inf")

        for unit in self.fleet.values():
            if unit.status != UnitStatus.AVAILABLE:
                continue
            if not unit.gps_is_fresh:
                continue

            dist = _haversine_km(incident_lat, incident_lon, unit.lat, unit.lon)
            if dist < best_dist:
                best_dist = dist
                best_unit = unit

        if best_unit is None:
            return None

        best_unit.status = UnitStatus.DISPATCHED
        best_unit.current_incident = incident_id
        eta_min = (best_dist / max(urban_speed_kmh, 1.0)) * 60.0

        return DispatchResult(
            unit_id=best_unit.unit_id,
            call_sign=best_unit.call_sign,
            officer_name=best_unit.officer_name,
            contact=best_unit.contact_number,
            distance_km=round(best_dist, 2),
            eta_minutes=round(eta_min, 1),
            status="DISPATCHED",
        )

    def update_status_by_name(
        self, unit_id: str, status_name: str, incident_id: Optional[str] = None
    ) -> bool:
        """Update unit status using a string representation of status."""
        try:
            status_enum = UnitStatus[status_name.upper()]
            return self.update_unit_status(unit_id, status_enum, incident_id)
        except KeyError:
            return False

    def auto_dispatch_alert(
        self,
        alert_id: str,
        lat: float,
        lon: float,
        severity: str = "HIGH",
        description: str = "",
    ) -> Optional[dict]:
        """Automatically dispatch the nearest available PCR patrol unit for a given alert."""
        res = self.dispatch_nearest(
            incident_lat=lat,
            incident_lon=lon,
            severity=severity,
            description=description,
            incident_id=alert_id,
        )
        if not res:
            return None
        return {
            "unit_id": res.unit_id,
            "call_sign": res.call_sign,
            "officer": res.officer_name,
            "contact": res.contact,
            "distance_km": res.distance_km,
            "eta_minutes": res.eta_minutes,
            "status": res.status,
        }

    def get_fleet_telemetry(self) -> list[dict]:
        return [
            {
                "unit_id": u.unit_id,
                "call_sign": u.call_sign,
                "city": u.city,
                "officer": u.officer_name,
                "contact": u.contact_number,
                "lat": u.lat,
                "lon": u.lon,
                "status": u.status.name,
                "gps_fresh": u.gps_is_fresh,
                "gps_age_sec": round(time.monotonic() - u.last_gps_update, 1),
                "incident": u.current_incident,
            }
            for u in self.fleet.values()
        ]


_dispatcher = CADDispatcher()


def get_dispatcher() -> CADDispatcher:
    return _dispatcher
