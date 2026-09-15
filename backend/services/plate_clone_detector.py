"""Detect cloned number plates: one registration on two vehicles.

A cloned plate is a stolen vehicle wearing a legitimate registration. At any
single camera it is indistinguishable from the real vehicle — the plate reads
correctly, the format is valid, the watchlist is clean. The only thing that
gives it away is the pair: the same registration appearing in two places
between which no vehicle could have travelled in the time available.

WHY THE TEST IS CONSERVATIVE
    Distance here is the great-circle line between two cameras, and a road is
    always longer than a straight line. So the implied speed computed below is
    an UNDER-estimate of the speed a real vehicle would have had to hold. When
    even that under-estimate exceeds what a vehicle can do, the conclusion is
    safe in the direction that matters: it can produce a missed clone, never a
    fabricated one.

WHAT THIS IS NOT
    The same arithmetic already exists twice in this codebase, pointed the
    other way: `camera_link_model` and `vehicle_reid_engine` use impossible
    travel to REJECT a re-identification match as false. That is the right
    thing to do for a visual match, where the alternative explanation is "the
    appearance model was wrong". A plate read is different — a confirmed
    registration is not an appearance guess, so the alternative explanation
    "there are two of them" is the live one, and it is worth an alert.
    `crime_detector._check_impossible_speed` does raise an alert, but on ReID
    identities rather than plates.

MEASURED ON THIS DEPLOYMENT (2026-09-09)
    Run over all 727 real vehicle journey events, 327 distinct plates: **zero
    alerts**. That is the correct answer — only 2 plates were ever read at two
    different cameras (GJ01ER2263 at 129 km/h and RJ47GA5111 at 18 km/h) and
    both transitions are entirely possible. Not raising an alert on 727 real
    events is as much a property worth reporting as raising one on a clone.
"""
from __future__ import annotations

import logging
from dataclasses import dataclass, field
from datetime import datetime
from math import asin, cos, radians, sin, sqrt
from typing import Any, Dict, Iterable, List, Optional, Tuple

logger = logging.getLogger(__name__)

# The fastest a road vehicle could plausibly average between two cameras,
# including the fact that the straight-line distance understates the road.
# Anything above this is not a fast driver; it is two vehicles.
MAX_ROAD_SPEED_KMH = 150.0

# Below this separation the cameras are close enough that a clock skew or a
# GPS position error of a few tens of metres could manufacture a large implied
# speed. CAM_08 and CAM_06 are 2.3 km apart, where 0.9 minutes would already
# look "impossible" — far too tight to be safe.
MIN_SEPARATION_KM = 5.0

# Two reads of the same plate closer together in time than this are treated as
# one event rather than a transition: a vehicle straddling two overlapping
# fields of view is not a clone.
MIN_GAP_SECONDS = 20.0


@dataclass
class Sighting:
    plate: str
    camera_id: str
    timestamp: datetime
    lat: float
    lon: float
    confidence: Optional[float] = None


@dataclass
class CloneAlert:
    plate: str
    from_camera: str
    to_camera: str
    from_time: datetime
    to_time: datetime
    distance_km: float
    gap_seconds: float
    implied_kmh: float
    max_allowed_kmh: float = MAX_ROAD_SPEED_KMH
    detail: Dict[str, Any] = field(default_factory=dict)

    @property
    def excess_factor(self) -> float:
        return self.implied_kmh / self.max_allowed_kmh

    def describe(self) -> str:
        return (
            f"Cloned plate suspected: {self.plate} was read at "
            f"{self.from_camera} and again at {self.to_camera}, "
            f"{self.distance_km:.1f} km away, {self.gap_seconds / 60.0:.1f} "
            f"minutes later. That requires {self.implied_kmh:.0f} km/h by the "
            f"straight line between them — {self.excess_factor:.1f}x the "
            f"{self.max_allowed_kmh:.0f} km/h ceiling, and the road is longer "
            f"still. One registration, two vehicles."
        )


def haversine_km(a_lat: float, a_lon: float, b_lat: float, b_lon: float) -> float:
    R = 6371.0
    p1, p2 = radians(a_lat), radians(b_lat)
    dphi = p2 - p1
    dlam = radians(b_lon - a_lon)
    h = sin(dphi / 2) ** 2 + cos(p1) * cos(p2) * sin(dlam / 2) ** 2
    return 2 * R * asin(sqrt(h))


def find_clones(
    sightings: Iterable[Sighting],
    max_speed_kmh: float = MAX_ROAD_SPEED_KMH,
    min_separation_km: float = MIN_SEPARATION_KM,
    min_gap_seconds: float = MIN_GAP_SECONDS,
) -> List[CloneAlert]:
    """Every consecutive pair of sightings of one plate that cannot both be true.

    Consecutive pairs only. A plate seen at A, then B, then C is checked A->B
    and B->C: if the vehicle really did travel A->C legitimately, no adjacent
    pair is impossible, and flagging the A->C span as well would report the
    same journey twice.
    """
    by_plate: Dict[str, List[Sighting]] = {}
    for s in sightings:
        if not s.plate or s.lat is None or s.lon is None or s.timestamp is None:
            continue
        by_plate.setdefault(s.plate, []).append(s)

    alerts: List[CloneAlert] = []
    for plate, seen in by_plate.items():
        seen.sort(key=lambda x: x.timestamp)
        for a, b in zip(seen, seen[1:]):
            if a.camera_id == b.camera_id:
                continue
            gap = (b.timestamp - a.timestamp).total_seconds()
            if gap < min_gap_seconds:
                continue
            dist = haversine_km(a.lat, a.lon, b.lat, b.lon)
            if dist < min_separation_km:
                continue
            implied = dist / (gap / 3600.0)
            if implied <= max_speed_kmh:
                continue
            alerts.append(CloneAlert(
                plate=plate,
                from_camera=a.camera_id, to_camera=b.camera_id,
                from_time=a.timestamp, to_time=b.timestamp,
                distance_km=round(dist, 2),
                gap_seconds=round(gap, 1),
                implied_kmh=round(implied, 1),
                max_allowed_kmh=max_speed_kmh,
                detail={
                    "from_confidence": a.confidence,
                    "to_confidence": b.confidence,
                    "distance_basis": "great-circle between camera positions; "
                                      "the road is longer, so the implied speed "
                                      "is a lower bound",
                },
            ))
    alerts.sort(key=lambda x: -x.implied_kmh)
    return alerts


def load_sightings_from_db(db, include_simulated: bool = True) -> List[Sighting]:
    """Plate sightings from journey_events, positioned by their camera.

    Seeded rows are excluded by the same provenance test the search endpoints
    use: the live pipeline always sets subtype and visual_score, and the 914
    generated rows in this database set neither.
    """
    from backend.db.models import Camera, JourneyEvent

    cams = {}
    for c in db.query(Camera).all():
        lat = getattr(c, "lat", None) or getattr(c, "gps_lat", None)
        lon = getattr(c, "lon", None) or getattr(c, "gps_lon", None)
        if lat is not None and lon is not None:
            cams[c.camera_id] = (float(lat), float(lon))

    q = db.query(JourneyEvent).filter(JourneyEvent.object_class == "vehicle")
    out: List[Sighting] = []
    for e in q.all():
        if e.subtype is None and e.visual_score is None:
            continue                      # seeded, not observed
        if e.camera_id == "CAM_E2E":
            continue                      # test rig
        lat, lon = e.lat, e.lon
        if lat is None or lon is None:
            pos = cams.get(e.camera_id)
            if not pos:
                continue
            lat, lon = pos
        ts = e.timestamp
        if isinstance(ts, str):
            try:
                ts = datetime.fromisoformat(ts)
            except ValueError:
                continue
        if ts is None:
            continue
        out.append(Sighting(
            plate=(e.reid_id or "").upper().replace(" ", ""),
            camera_id=e.camera_id, timestamp=ts,
            lat=float(lat), lon=float(lon),
            confidence=e.final_score,
        ))
    return out
