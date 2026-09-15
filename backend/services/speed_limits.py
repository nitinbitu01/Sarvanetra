"""Indian road speed limits, by road class and vehicle type.

WHY THIS IS NOT A HARDCODED NUMBER
  The journey view used a single 60 km/h overspeed threshold. That is wrong
  almost everywhere: a car at 78 km/h is speeding on an urban road and well
  inside the limit on a national highway, and a truck at 85 is speeding on a
  2-lane NH while a car beside it is not. One constant cannot express that.

  The table below is reference data — the MoRTH class limits that apply on
  Indian roads — rather than a tuning parameter. Encoding it is the opposite
  of hardcoding: it replaces one invented number with the actual rule.

THE HARD PART IS NOT THE TABLE, IT IS KNOWING WHICH ROW APPLIES
  Two facts are needed per leg and neither is currently recorded:

    road class    the cameras table has no road/highway/lane column at all
    vehicle type  JourneyEvent.subtype is NULL for all 914 rows, even though
                  the detector knows it (YOLO_CLASS_NAMES: car, motorcycle,
                  bus, truck)

  Guessing either one would be worse than the constant it replaces, because a
  wrong guess produces a confident overspeed flag against a driver who was
  obeying the limit. So neither is guessed:

    both known            the exact limit for that combination
    road known only       the HIGHEST limit on that road class, so a flag
                          fires only if the vehicle was speeding whatever it
                          was
    road unknown          the highest limit on ANY Indian road (120 km/h,
                          car on an access-controlled expressway). Above that
                          a vehicle is speeding on every road in the country,
                          so the flag stays true without knowing where it is

  The last case is deliberately hard to trigger. That is the point: an
  unflagged speeding vehicle is a missed detection, a wrongly flagged one is
  an accusation.

POPULATING ROAD CLASS
  config/camera_road_class.json maps camera id to road class and is meant to
  be filled in by whoever knows the sites. backend/scripts/
  suggest_camera_road_class.py proposes values from camera names and owning
  department for a human to confirm — a suggestion is not evidence, so
  nothing it writes is treated as known until reviewed.
"""
from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Dict, Optional, Tuple

logger = logging.getLogger(__name__)

PROJECT_ROOT = Path(__file__).resolve().parents[2]
ROAD_CLASS_CONFIG = PROJECT_ROOT / "config" / "camera_road_class.json"

# Road classes, most permissive first.
EXPRESSWAY = "EXPRESSWAY"
NH_DIVIDED = "NH_4LANE_DIVIDED"
NH_UNDIVIDED = "NH_2LANE"
STATE_HIGHWAY = "STATE_HIGHWAY"
URBAN = "URBAN"
UNKNOWN_ROAD = "UNKNOWN"

ROAD_CLASSES = (EXPRESSWAY, NH_DIVIDED, NH_UNDIVIDED, STATE_HIGHWAY, URBAN)

CAR = "car"
MOTORCYCLE = "motorcycle"
BUS = "bus"
TRUCK = "truck"
UNKNOWN_VEHICLE = "unknown"

# km/h. Where a range is published the LOWER bound would flag more drivers,
# so the upper bound is used — the same conservative direction as everything
# else here.
SPEED_LIMITS: Dict[str, Dict[str, float]] = {
    EXPRESSWAY:    {CAR: 120.0, BUS: 100.0, TRUCK: 90.0,  MOTORCYCLE: 80.0},
    NH_DIVIDED:    {CAR: 100.0, BUS: 100.0, TRUCK: 80.0,  MOTORCYCLE: 80.0},
    NH_UNDIVIDED:  {CAR:  80.0, BUS:  65.0, TRUCK: 65.0,  MOTORCYCLE: 80.0},
    STATE_HIGHWAY: {CAR:  80.0, BUS:  65.0, TRUCK: 65.0,  MOTORCYCLE: 60.0},
    URBAN:         {CAR:  70.0, BUS:  50.0, TRUCK: 50.0,  MOTORCYCLE: 50.0},
}

# The fastest anything is allowed to travel on any Indian road.
ABSOLUTE_MAX_KMH = max(max(v.values()) for v in SPEED_LIMITS.values())

_cache: Optional[Dict[str, str]] = None


def load_road_classes(force: bool = False) -> Dict[str, str]:
    """camera id -> road class, from config. Missing file means all unknown."""
    global _cache
    if _cache is not None and not force:
        return _cache
    data: Dict[str, str] = {}
    if ROAD_CLASS_CONFIG.is_file():
        try:
            raw = json.loads(ROAD_CLASS_CONFIG.read_text(encoding="utf-8"))
            for cam, entry in (raw.get("cameras") or {}).items():
                rc = (entry or {}).get("road_class")
                # Only a confirmed entry counts. A suggestion left unreviewed
                # is still unknown, however plausible it looks.
                if rc in ROAD_CLASSES and (entry or {}).get("confirmed") is True:
                    data[str(cam)] = rc
        except Exception as exc:                                  # noqa: BLE001
            logger.warning("Could not read %s: %s", ROAD_CLASS_CONFIG, exc)
    _cache = data
    return data


def road_class_for(camera_id) -> str:
    return load_road_classes().get(str(camera_id), UNKNOWN_ROAD)


def limit_for(road_class: Optional[str],
              vehicle_type: Optional[str]) -> Tuple[float, str]:
    """(limit km/h, how that limit was arrived at).

    The second element travels with the flag so the UI and the PDF can say
    which rule was applied, rather than presenting a bare number an officer
    cannot check.
    """
    rc = (road_class or UNKNOWN_ROAD).upper()
    vt = (vehicle_type or UNKNOWN_VEHICLE).lower()

    if rc not in SPEED_LIMITS:
        return ABSOLUTE_MAX_KMH, (
            f"Road class not configured for this camera, so the limit used is "
            f"{ABSOLUTE_MAX_KMH:.0f} km/h — the highest permitted on any "
            f"Indian road. A vehicle above it is speeding wherever it was."
        )
    table = SPEED_LIMITS[rc]
    if vt in table:
        return table[vt], (
            f"{table[vt]:.0f} km/h — {vt} on {rc.replace('_', ' ').lower()}."
        )
    highest = max(table.values())
    return highest, (
        f"Vehicle type unrecorded, so the limit used is {highest:.0f} km/h — "
        f"the highest for any vehicle on {rc.replace('_', ' ').lower()}."
    )


def classify_leg(speed_kmh: float, duration_min: float,
                 road_class: Optional[str], vehicle_type: Optional[str],
                 stopover_min_minutes: float = 10.0,
                 stopover_max_kmh: float = 10.0,
                 road_matched: bool = False) -> Dict[str, object]:
    """Flag one leg between two sightings.

    `road_matched` says which distance the speed was computed over, and the
    wording has to follow it or the caveat becomes false:

      False — the geodesic. The vehicle covered AT LEAST that in the time, so
              the true road speed was this or higher. Conservative in the right
              direction, which is why the reason says "at least".
      True  — the routed road distance. This is the distance a driver could
              actually have covered, so the figure is a real speed rather than
              a lower bound. It still assumes the fastest sensible route; a
              driver who went a longer way was going faster still.
    """
    limit, basis = limit_for(road_class, vehicle_type)
    if speed_kmh >= limit:
        qualifier = (
            "Measured over the routed road distance, not the straight line, so "
            "this is the speed the journey actually required — a longer route "
            "would mean a higher speed still."
            if road_matched else
            "Straight-line speed is a lower bound — the road is longer than "
            "the geodesic, so the true speed was higher."
        )
        reason = (
            f"{'' if road_matched else 'At least '}{speed_kmh:.0f} km/h "
            f"against a {limit:.0f} km/h limit. {basis} {qualifier}"
        )
        # Above the national maximum there is no lawful explanation left, and
        # at that point a second explanation becomes at least as likely as
        # reckless driving: the journey may not be one vehicle at all.
        #
        # Plate recognition on this deployment is 64.9% exact, so a misread
        # splices two different vehicles that happen to share a decoded plate
        # into a single route — and the "speed" is then the distance between
        # two unrelated cars divided by the gap between two unrelated
        # timestamps. That artefact looks exactly like a supercar.
        #
        # Both readings are surfaced rather than picking one, because an
        # officer sent after a vehicle on the strength of a splice is a worse
        # outcome than a caveat they had to read.
        if speed_kmh > ABSOLUTE_MAX_KMH:
            reason += (
                f" VERIFY BEFORE ACTING: {speed_kmh:.0f} km/h exceeds the "
                f"national maximum of {ABSOLUTE_MAX_KMH:.0f} km/h, so this is "
                f"either serious overspeeding or two different vehicles merged "
                f"by a misread plate. Check the evidence crops for both "
                f"sightings before treating it as one vehicle."
            )
        return {
            "flag": "OVERSPEED",
            "speed_limit_kmh": limit,
            "limit_basis": basis,
            "needs_verification": bool(speed_kmh > ABSOLUTE_MAX_KMH),
            "flag_reason": reason,
        }
    if duration_min >= stopover_min_minutes and speed_kmh <= stopover_max_kmh:
        return {
            "flag": "STOPOVER",
            "speed_limit_kmh": limit,
            "limit_basis": basis,
            "needs_verification": False,
            "flag_reason": (
                f"{duration_min:.0f} minutes to cover very little ground. The "
                f"vehicle spent that time somewhere between these cameras."
            ),
        }
    return {
        "flag": "NORMAL",
        "speed_limit_kmh": limit,
        "limit_basis": basis,
        "needs_verification": False,
        "flag_reason": "",
    }
