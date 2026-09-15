"""
backend/services/network_activity.py — camera-network activity analytics.

WHAT THIS MEASURES, AND WHAT IT DELIBERATELY DOES NOT

Every number here is computed from journey_events — the plate sightings the
recogniser actually indexed. That makes them observations of THIS CAMERA
NETWORK, not measurements of traffic on the road. The distinction is not
pedantic; it is the difference between a number an officer can act on and one
that quietly misleads:

  - A camera's sighting count reflects traffic volume AND that camera's plate
    readability. CAM_21 topping the network does not establish that it watches
    the busiest road — only that it produced the most usable plate reads.
  - A camera with zero sightings is a COVERAGE GAP, not "no traffic". Reporting
    it as zero traffic would turn a broken or unreadable feed into good news.
  - Corridor speed is derived from straight-line distance between two camera
    GPS points. Vehicles drive roads, which are longer than the straight line,
    so the figure is a LOWER BOUND on true speed and is labelled as one.

There is deliberately NO congestion index, NO density, and NO origin-destination
matrix in this module. Those require a continuous per-vehicle detection stream
plus per-camera speed calibration. In this database `detections`, `tracks`,
`camera_calibration` and `camera_calibrations` are all empty, so every such
figure would have to be invented. An invented speed on a police dashboard is
worse than an absent one.

Sample sizes are carried through to the API on every derived statistic, because
a median transit time over n=1 and one over n=17 are not the same claim.
"""
from __future__ import annotations

import math
import threading
from collections import Counter, defaultdict
from datetime import datetime
from statistics import median
from typing import Any, Optional

from sqlalchemy import func
from sqlalchemy.orm import Session

from backend.db.models import Camera, JourneyEvent

# Outer time bound on a single camera-to-camera transit. Generous, because a
# genuine inter-city highway link in this network is ~70 km and takes the best
# part of an hour; the real filtering is done on speed below.
MAX_PLAUSIBLE_TRANSIT_SEC = 3600.0
MIN_PLAUSIBLE_TRANSIT_SEC = 1.0

# Implied straight-line speed must fall inside a physically plausible driving
# band for the pair to count as ONE continuous drive between two cameras.
#
# The lower bound matters more than it looks. Without it, two unrelated
# sightings of the same plate — the vehicle parked for 50 minutes, then was
# read again — are treated as a single transit and produce figures like
# "0.9 km covered in 53 minutes = 1.0 km/h". On a dashboard that reads as a
# catastrophic jam when it is really a data artefact, which is precisely the
# kind of confidently wrong number this module exists to avoid.
#
# The cost of this choice, stated plainly: a genuine stop-start jam slower than
# MIN_PLAUSIBLE_SPEED_KMH is also excluded. With a continuous detection stream
# the two cases would be separable (a jammed vehicle stays visible; a parked one
# leaves the road). With sparse plate re-reads they are not, so the filter
# favours discarding a real jam over inventing one.
MIN_PLAUSIBLE_SPEED_KMH = 5.0
MAX_PLAUSIBLE_SPEED_KMH = 150.0

# Sample-count thresholds behind the `confidence` label on each corridor row.
# A median over 1 observation and one over 16 are not the same claim, so the
# API says which it is rather than leaving the UI to guess.
CORRIDOR_CONFIDENCE_GOOD = 10
CORRIDOR_CONFIDENCE_WEAK = 3


def haversine_km(lat1: Optional[float], lon1: Optional[float],
                 lat2: Optional[float], lon2: Optional[float]) -> float:
    """Great-circle distance between two GPS points, in kilometres.

    Returns 0.0 when any coordinate is missing rather than raising, so one
    camera with no surveyed GPS cannot break a whole corridor table.
    """
    if lat1 is None or lon1 is None or lat2 is None or lon2 is None:
        return 0.0
    r = 6371.0
    dlat = math.radians(lat2 - lat1)
    dlon = math.radians(lon2 - lon1)
    a = (math.sin(dlat / 2) ** 2
         + math.cos(math.radians(lat1)) * math.cos(math.radians(lat2))
         * math.sin(dlon / 2) ** 2)
    return r * 2 * math.atan2(math.sqrt(a), math.sqrt(max(0.0, 1.0 - a)))


def _as_datetime(value: Any) -> Optional[datetime]:
    """Coerce a timestamp column to datetime.

    SQLAlchemy normally hands back a datetime for a DateTime column, but this
    table was populated by a raw sqlite3 batch script (journey_populate_db.py)
    which writes ISO strings. Both shapes therefore occur in practice.
    """
    if isinstance(value, datetime):
        return value
    if isinstance(value, str) and value:
        try:
            return datetime.fromisoformat(value)
        except ValueError:
            return None
    return None


def _camera_coords(cam: Optional[Camera]) -> tuple[Optional[float], Optional[float]]:
    """Surveyed GPS for a camera, preferring the dedicated gps_* columns.

    The Camera row carries both `gps_lat/gps_lon` and older `lat/lon`; in this
    database all 32 cameras have gps_* set while only 30 have lat/lon, so gps_*
    is tried first and lat/lon is the fallback.
    """
    if cam is None:
        return (None, None)
    lat = cam.gps_lat if cam.gps_lat is not None else cam.lat
    lon = cam.gps_lon if cam.gps_lon is not None else cam.lon
    return (lat, lon)


# ── Snapshot cache ──────────────────────────────────────────────────────────
# journey_events is a batch table: it is rewritten wholesale by
# journey_populate_db.py and otherwise never changes. Recomputing the full
# snapshot per request would be wasted work, so it is cached against a cheap
# fingerprint (row count + newest timestamp). The fingerprint query is a single
# indexed aggregate; the expensive pass only runs when the data actually moved.
_cache_lock = threading.Lock()
_cached_fingerprint: Optional[tuple] = None
_cached_snapshot: Optional[dict[str, Any]] = None


def _fingerprint(db: Session) -> tuple:
    row = db.query(
        func.count(JourneyEvent.id),
        func.max(JourneyEvent.timestamp),
    ).one()
    return (int(row[0] or 0), str(row[1] or ""))


def get_snapshot(db: Session, force: bool = False) -> dict[str, Any]:
    """Full network-activity snapshot, cached until journey_events changes."""
    global _cached_fingerprint, _cached_snapshot
    fp = _fingerprint(db)
    with _cache_lock:
        if not force and _cached_snapshot is not None and _cached_fingerprint == fp:
            return _cached_snapshot
    snapshot = _compute_snapshot(db)
    with _cache_lock:
        _cached_fingerprint = fp
        _cached_snapshot = snapshot
    return snapshot


def _compute_snapshot(db: Session) -> dict[str, Any]:
    """Compute every network-activity figure in one pass over journey_events."""
    # Only the four columns that are actually needed — this avoids hydrating
    # full ORM objects for every sighting.
    # Seeded rows are excluded, by provenance rather than by date.
    #
    # `live_24x7_pipeline` sets subtype=<detected class> and visual_score=1.0 on
    # every JourneyEvent it writes; the seeded rows carry neither. So the drop
    # condition is "both columns NULL", and the keep condition is its negation —
    # the same predicate as `_real_events_only()` in routers/v1/journeys.py and
    # the inline test in plate_search.py. Get it the wrong way round and this
    # service reports the seeded rows and nothing else; write it as the
    # negation, not as a pair of IS NULL filters.
    #
    # Measured on the current DB, which is why the split is trusted:
    #                        both NULL      subtype set
    #   events                     914             4225
    #   distinct plates            816              516
    #   events per plate          1.12             8.19
    #   final_score          0.92-0.992       0.431-1.0
    #   span                one day (Jun)      8 days (Sep)
    # Real traffic re-reads a plate many times — GJ03KC0285 appears 151 times on
    # the right-hand side. One event per plate, inside a clamped score band, in
    # a single day, is a generator's signature.
    real_events = ~(JourneyEvent.subtype.is_(None)
                    & JourneyEvent.visual_score.is_(None))

    rows = (
        db.query(
            JourneyEvent.reid_id,
            JourneyEvent.camera_id,
            JourneyEvent.timestamp,
            JourneyEvent.match_type,
        )
        .filter(JourneyEvent.object_class == "vehicle")
        .filter(real_events)
        .order_by(JourneyEvent.reid_id, JourneyEvent.timestamp)
        .all()
    )

    # How many were left out, so the exclusion is visible in the payload rather
    # than being a silent difference between this number and the table's size.
    seeded_excluded = (
        db.query(JourneyEvent)
        .filter(JourneyEvent.object_class == "vehicle")
        .filter(~real_events)
        .count()
    )

    # Deployed estate only: not soft-deleted AND not a test rig. The second half
    # matters here beyond the headcount — the re-ID test cameras carry
    # placeholder coordinates spaced a notional 100 m apart, so leaving them in
    # would draw corridors and corridor speeds out of positions nobody surveyed.
    from backend.services.fleet_census import is_test_camera

    cameras = {
        c.id: c
        for c in db.query(Camera).filter(Camera.is_deleted == False).all()  # noqa: E712
        if not is_test_camera(c.id, c.name)
    }

    events: list[tuple[str, str, datetime, str]] = []
    for reid_id, camera_id, ts, match_type in rows:
        dt = _as_datetime(ts)
        if dt is None or not reid_id or not camera_id:
            continue
        events.append((reid_id, camera_id, dt, match_type or "unknown"))

    if not events:
        return _empty_snapshot(len(cameras))

    # ── Totals and data window ──────────────────────────────────────────────
    all_ts = [e[2] for e in events]
    first_event, last_event = min(all_ts), max(all_ts)
    span_hours = (last_event - first_event).total_seconds() / 3600.0

    per_vehicle: dict[str, list[tuple[str, datetime]]] = defaultdict(list)
    for reid_id, camera_id, dt, _ in events:
        per_vehicle[reid_id].append((camera_id, dt))

    multi_camera_vehicles = sum(
        1 for seq in per_vehicle.values() if len({c for c, _ in seq}) > 1
    )
    match_counts = Counter(e[3] for e in events)

    # ── Per-camera activity ─────────────────────────────────────────────────
    cam_sightings: Counter = Counter()
    cam_vehicles: dict[str, set] = defaultdict(set)
    cam_first: dict[str, datetime] = {}
    cam_last: dict[str, datetime] = {}
    cam_hours: dict[str, Counter] = defaultdict(Counter)

    for reid_id, camera_id, dt, _ in events:
        cam_sightings[camera_id] += 1
        cam_vehicles[camera_id].add(reid_id)
        if camera_id not in cam_first or dt < cam_first[camera_id]:
            cam_first[camera_id] = dt
        if camera_id not in cam_last or dt > cam_last[camera_id]:
            cam_last[camera_id] = dt
        cam_hours[camera_id][dt.strftime("%H")] += 1

    total_sightings = len(events)
    camera_rows: list[dict[str, Any]] = []
    for cam_id, cam in cameras.items():
        n = cam_sightings.get(cam_id, 0)
        lat, lon = _camera_coords(cam)
        busiest = cam_hours[cam_id].most_common(1) if n else []
        camera_rows.append({
            "camera_id": cam_id,
            "name": cam.name,
            "department": cam.department,
            "district": cam.district,
            "zone": cam.zone,
            "lat": lat,
            "lon": lon,
            "sightings": n,
            "distinct_vehicles": len(cam_vehicles.get(cam_id, ())),
            "share_pct": round(100.0 * n / total_sightings, 2) if total_sightings else 0.0,
            "first_seen": cam_first[cam_id].isoformat() if cam_id in cam_first else None,
            "last_seen": cam_last[cam_id].isoformat() if cam_id in cam_last else None,
            "peak_hour": busiest[0][0] if busiest else None,
            # An explicit flag beats a silent zero: a camera with no indexed
            # sightings is a gap in coverage, not an empty road.
            "has_activity": n > 0,
        })
    camera_rows.sort(key=lambda r: (-r["sightings"], r["camera_id"]))

    # ── Temporal profile ────────────────────────────────────────────────────
    hour_sightings: Counter = Counter()
    hour_vehicles: dict[str, set] = defaultdict(set)
    for reid_id, _, dt, _ in events:
        h = dt.strftime("%H")
        hour_sightings[h] += 1
        hour_vehicles[h].add(reid_id)
    temporal = [
        {
            "hour": h,
            "sightings": hour_sightings[h],
            "distinct_vehicles": len(hour_vehicles[h]),
        }
        for h in sorted(hour_sightings)
    ]
    peak = max(temporal, key=lambda r: r["sightings"]) if temporal else None

    # ── Corridor transits ───────────────────────────────────────────────────
    corridors, corridor_exclusions = _compute_corridors(per_vehicle, cameras)

    return {
        "data_window": {
            "first_event": first_event.isoformat(),
            "last_event": last_event.isoformat(),
            "hours_spanned": round(span_hours, 2),
            "distinct_days": len({t.date().isoformat() for t in all_ts}),
            "hours_observed": len(temporal),
        },
        "totals": {
            "sightings": total_sightings,
            "distinct_vehicles": len(per_vehicle),
            "multi_camera_vehicles": multi_camera_vehicles,
            "cameras_with_activity": sum(1 for r in camera_rows if r["has_activity"]),
            "cameras_total": len(cameras),
            "plate_confirmed": match_counts.get("plate_confirmed", 0),
            "plate_candidate": match_counts.get("plate_candidate", 0),
        },
        "peak_hour": peak,
        "temporal": temporal,
        "cameras": camera_rows,
        "corridors": corridors,
        "corridor_exclusions": corridor_exclusions,
        "seeded_rows_excluded": seeded_excluded,
        "measurement_basis": _measurement_basis(),
    }


def _compute_corridors(
    per_vehicle: dict[str, list[tuple[str, datetime]]],
    cameras: dict[str, Camera],
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    """Observed camera-to-camera transits, aggregated per ordered pair.

    A transit is two consecutive sightings of the same vehicle on different
    cameras. Median (not mean) transit time is reported because a handful of
    samples with one parked vehicle would drag a mean badly.
    """
    transits: dict[tuple[str, str], list[float]] = defaultdict(list)
    excluded_window: Counter = Counter()
    excluded_slow: Counter = Counter()
    excluded_fast: Counter = Counter()
    excluded_no_gps: Counter = Counter()

    for seq in per_vehicle.values():
        ordered = sorted(seq, key=lambda x: x[1])
        for (cam_a, t_a), (cam_b, t_b) in zip(ordered, ordered[1:]):
            if cam_a == cam_b:
                continue
            dt_sec = (t_b - t_a).total_seconds()
            key = (cam_a, cam_b)
            if dt_sec < MIN_PLAUSIBLE_TRANSIT_SEC or dt_sec > MAX_PLAUSIBLE_TRANSIT_SEC:
                excluded_window[key] += 1
                continue
            lat_a, lon_a = _camera_coords(cameras.get(cam_a))
            lat_b, lon_b = _camera_coords(cameras.get(cam_b))
            dist_km = haversine_km(lat_a, lon_a, lat_b, lon_b)
            if dist_km <= 0:
                # No surveyed GPS on one of the two cameras: the transit is
                # real but its speed is not computable, so it cannot join a
                # speed statistic.
                excluded_no_gps[key] += 1
                continue
            speed = dist_km / (dt_sec / 3600.0)
            if speed > MAX_PLAUSIBLE_SPEED_KMH:
                excluded_fast[key] += 1
                continue
            if speed < MIN_PLAUSIBLE_SPEED_KMH:
                excluded_slow[key] += 1
                continue
            transits[key].append(dt_sec)

    results: list[dict[str, Any]] = []
    for (cam_a, cam_b), samples in transits.items():
        lat_a, lon_a = _camera_coords(cameras.get(cam_a))
        lat_b, lon_b = _camera_coords(cameras.get(cam_b))
        dist_km = haversine_km(lat_a, lon_a, lat_b, lon_b)
        med_sec = float(median(samples))
        speed_kmh = (dist_km / (med_sec / 3600.0)) if dist_km > 0 and med_sec > 0 else None
        cam_a_row, cam_b_row = cameras.get(cam_a), cameras.get(cam_b)
        n = len(samples)
        if n >= CORRIDOR_CONFIDENCE_GOOD:
            confidence = "good"
        elif n >= CORRIDOR_CONFIDENCE_WEAK:
            confidence = "weak"
        else:
            confidence = "single_observation" if n == 1 else "very_weak"
        results.append({
            "from_camera": cam_a,
            "to_camera": cam_b,
            "from_name": cam_a_row.name if cam_a_row else cam_a,
            "to_name": cam_b_row.name if cam_b_row else cam_b,
            "from_lat": lat_a, "from_lon": lon_a,
            "to_lat": lat_b, "to_lon": lon_b,
            # n is surfaced on every row: a median over 1 sample and a median
            # over 16 must not look alike in the UI.
            "samples": n,
            "confidence": confidence,
            "median_transit_sec": round(med_sec, 1),
            "fastest_transit_sec": round(min(samples), 1),
            "slowest_transit_sec": round(max(samples), 1),
            "straight_line_km": round(dist_km, 3),
            "implied_speed_kmh": round(speed_kmh, 1) if speed_kmh is not None else None,
            "excluded_out_of_window": excluded_window.get((cam_a, cam_b), 0),
            "excluded_too_slow": excluded_slow.get((cam_a, cam_b), 0),
            "excluded_implausible_speed": excluded_fast.get((cam_a, cam_b), 0),
            "excluded_no_camera_gps": excluded_no_gps.get((cam_a, cam_b), 0),
            "speed_is_lower_bound": True,
        })
    results.sort(key=lambda r: (-r["samples"], r["from_camera"], r["to_camera"]))

    # Pairs whose every observation was filtered out would otherwise disappear
    # from the response entirely, which would hide the fact that the system saw
    # those vehicles at all. They are reported separately with the reason.
    surviving = set(transits.keys())
    dropped: list[dict[str, Any]] = []
    all_excluded_keys = (set(excluded_window) | set(excluded_slow)
                         | set(excluded_fast) | set(excluded_no_gps))
    for key in sorted(all_excluded_keys - surviving):
        cam_a, cam_b = key
        cam_a_row, cam_b_row = cameras.get(cam_a), cameras.get(cam_b)
        dropped.append({
            "from_camera": cam_a,
            "to_camera": cam_b,
            "from_name": cam_a_row.name if cam_a_row else cam_a,
            "to_name": cam_b_row.name if cam_b_row else cam_b,
            "observations_seen": (excluded_window.get(key, 0) + excluded_slow.get(key, 0)
                                  + excluded_fast.get(key, 0) + excluded_no_gps.get(key, 0)),
            "out_of_time_window": excluded_window.get(key, 0),
            "too_slow_to_be_one_drive": excluded_slow.get(key, 0),
            "implausibly_fast": excluded_fast.get(key, 0),
            "no_camera_gps": excluded_no_gps.get(key, 0),
        })

    exclusions = {
        "total_transits_observed": (
            sum(len(v) for v in transits.values())
            + sum(excluded_window.values()) + sum(excluded_slow.values())
            + sum(excluded_fast.values()) + sum(excluded_no_gps.values())
        ),
        "used_for_speed": sum(len(v) for v in transits.values()),
        "excluded_out_of_window": sum(excluded_window.values()),
        "excluded_too_slow": sum(excluded_slow.values()),
        "excluded_implausible_speed": sum(excluded_fast.values()),
        "excluded_no_camera_gps": sum(excluded_no_gps.values()),
        "fully_excluded_pairs": dropped,
        "speed_band_kmh": [MIN_PLAUSIBLE_SPEED_KMH, MAX_PLAUSIBLE_SPEED_KMH],
        "note": (
            "A transit is two consecutive sightings of one vehicle on different "
            "cameras. Pairs outside the speed band are not treated as a single "
            "continuous drive — most are a vehicle that stopped between reads."
        ),
    }
    return results, exclusions


def _measurement_basis() -> dict[str, Any]:
    """What these numbers are, in the response itself.

    Carried in the payload rather than only in the UI so the caveats travel
    with the data — an exported CSV or a screenshot keeps them.
    """
    return {
        "source_table": "journey_events",
        "source_description": (
            "Plate sightings indexed by the recogniser, EXCLUDING seeded rows. "
            "Counts reflect both traffic volume and per-camera plate "
            "readability — a quiet camera and an unreadable one look the same "
            "here."
        ),
        "provenance": (
            "A row is counted unless it has neither a subtype nor a "
            "visual_score, which is the seeder's signature — the live pipeline "
            "always writes both. `seeded_rows_excluded` says how many were "
            "left out."
        ),
        "measured": [
            "per-camera sightings", "distinct vehicles",
            "multi-camera vehicles", "observed corridors (origin -> destination)",
            "hourly distribution and peak hour",
        ],
        "not_measured": [
            "vehicle density (occupancy)", "predicted bottlenecks",
            "speed in km/h on most cameras",
        ],
        "not_measured_reason": (
            "Density needs occupancy, not passage counts. Speed in km/h needs a "
            "per-camera homography and only CAM_09 currently has one, labelled "
            "provisional — so km/h is reported for that camera alone rather "
            "than estimated for the rest. Corridors below ARE observed "
            "origin-destination pairs, but over a small sample: read them as "
            "evidence that a route was travelled, not as a traffic census."
        ),
        "speed_caveat": (
            "Corridor speed uses straight-line distance between camera GPS "
            "points. Real roads are longer, so every speed shown is a lower "
            "bound on true speed."
        ),
        "zero_activity_caveat": (
            "A camera with no sightings indicates no indexed plate reads — a "
            "coverage gap, not an empty road."
        ),
    }


def _empty_snapshot(camera_count: int) -> dict[str, Any]:
    return {
        "data_window": {
            "first_event": None, "last_event": None, "hours_spanned": 0.0,
            "distinct_days": 0, "hours_observed": 0,
        },
        "totals": {
            "sightings": 0, "distinct_vehicles": 0, "multi_camera_vehicles": 0,
            "cameras_with_activity": 0, "cameras_total": camera_count,
            "plate_confirmed": 0, "plate_candidate": 0,
        },
        "peak_hour": None,
        "temporal": [],
        "cameras": [],
        "corridors": [],
        "corridor_exclusions": {
            "total_transits_observed": 0, "used_for_speed": 0,
            "excluded_out_of_window": 0, "excluded_too_slow": 0,
            "excluded_implausible_speed": 0, "excluded_no_camera_gps": 0,
            "fully_excluded_pairs": [],
            "speed_band_kmh": [MIN_PLAUSIBLE_SPEED_KMH, MAX_PLAUSIBLE_SPEED_KMH],
            "note": "no journey events recorded",
        },
        "measurement_basis": _measurement_basis(),
    }
