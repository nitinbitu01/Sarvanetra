"""
backend/routing/seed.py — demo officers + routing cameras (Day 14).

Inserted at startup only when the officers table is empty, so a real roster
is never overwritten by demo data on restart.

COORDINATE NOTE
  These are the San Francisco coordinates from the Day 14 spec, chosen so
  haversine produces unambiguously different distances (>10x ratio between
  nearest and second-nearest) — which is what makes Test 1 visibly
  demonstrate the correct officer being picked rather than a coin flip.

  The pre-existing demo cameras in this project are in Gujarat (23.0N,
  72.5E). Routing resolves coordinates per alert via
  alerts.camera_id → cameras.gps_lat/gps_lon, so the two sets coexist without
  interfering: an alert from a Gujarat camera simply finds every SF officer
  ~10,000 km away and picks the nearest of them. To demo the intended
  distances, fire the alert from one of the two cameras seeded here.
"""
from __future__ import annotations

import logging

from sqlalchemy import text

logger = logging.getLogger(__name__)

DEMO_OFFICERS = [
    # id=1 — nearest to Main Entrance Cam
    {"id": 1, "name": "Officer Chen", "lat": 37.7749, "lng": -122.4194,
     "status": "AVAILABLE"},
    # id=2 — nearest to Parking Lot Cam
    {"id": 2, "name": "Officer Park", "lat": 37.7751, "lng": -122.4180,
     "status": "AVAILABLE"},
    # id=3 — third nearest to both; the escalation fallback
    {"id": 3, "name": "Officer Ramirez", "lat": 37.7740, "lng": -122.4210,
     "status": "AVAILABLE"},
    # id=4 — OFFLINE: must never be assigned even if geographically closest
    {"id": 4, "name": "Officer Thompson", "lat": 37.7760, "lng": -122.4220,
     "status": "OFFLINE"},
]

DEMO_ROUTING_CAMERAS = [
    # Chen 0.01 km | Park 0.15 km | Ramirez 0.18 km
    {"camera_id": "CAM-RT-A", "name": "Main Entrance Cam",
     "gps_lat": 37.7748, "gps_lon": -122.4195},
    # Park 0.01 km | Chen 0.15 km | Ramirez 0.22 km
    {"camera_id": "CAM-RT-B", "name": "Parking Lot Cam",
     "gps_lat": 37.7752, "gps_lon": -122.4179},
]


def seed_routing_demo_data() -> None:
    """Idempotent. Safe to call on every startup."""
    from backend.db.session import SessionLocal

    db = SessionLocal()
    try:
        existing = db.execute(text("SELECT COUNT(*) FROM officers")).scalar() or 0
        if existing == 0:
            for o in DEMO_OFFICERS:
                db.execute(text("""
                    INSERT OR IGNORE INTO officers
                        (id, name, lat, lng, status, current_alert_id, last_updated)
                    VALUES (:id, :name, :lat, :lng, :status, NULL, datetime('now'))
                """), o)
            logger.info("Seeded %d demo officers.", len(DEMO_OFFICERS))

        for cam in DEMO_ROUTING_CAMERAS:
            present = db.execute(text(
                "SELECT id FROM cameras WHERE camera_id = :cid"
            ), {"cid": cam["camera_id"]}).fetchone()
            if present is None:
                # consecutive_failures is NOT NULL (added Day 7 with a
                # server_default that a raw INSERT does not apply) — omitting
                # it fails the whole seed with an IntegrityError.
                # `id` is deliberately omitted so SQLite autoincrements the
                # integer surrogate key. Writing the string camera_id into
                # BOTH columns (as this did) only worked while Camera.id was a
                # string PK — against the integer PK it raises "datatype
                # mismatch", and because officers and cameras share this one
                # transaction, that rollback silently takes the four demo
                # officers with it and every alert routes to nobody.
                db.execute(text("""
                    INSERT INTO cameras
                        (camera_id, name, gps_lat, gps_lon, status,
                         is_deleted, consecutive_failures, created_at)
                    VALUES (:camera_id, :name, :gps_lat, :gps_lon,
                            'ONLINE', 0, 0, datetime('now'))
                """), cam)
                logger.info("Seeded routing demo camera %s.", cam["camera_id"])

        db.commit()
    except Exception as exc:
        db.rollback()
        logger.error("Routing seed failed: %s", exc, exc_info=True)
    finally:
        db.close()
