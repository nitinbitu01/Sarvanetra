"""
Seed database with:
- 30 cameras from config.yaml
- 8 watchlist plates (stolen/wanted)
- 8 patrol officers across Gujarat zones
- 3 system users (admin, operator, viewer)
- 3 wanted persons with dummy embeddings
"""

import asyncio
import json
import os
import sys
import uuid
import yaml
import numpy as np
from datetime import datetime

_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
if _ROOT not in sys.path:
    sys.path.insert(0, _ROOT)

try:
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8")
except Exception:
    pass

from backend.db.session import init_db, SessionLocal
from backend.db.models import (
    Camera, WatchlistPlate, WatchlistPerson,
    Officer, User, UserRole
)
from backend.auth.jwt_handler import hash_password


def seed_cameras(config: dict):
    db = SessionLocal()
    try:
        for cam in config.get("demo_cameras", []):
            cid = str(cam.get("id") or cam.get("camera_id"))
            existing = db.query(Camera).filter(Camera.camera_id == cid).first()
            if existing:
                existing.status = "ONLINE"
                existing.is_online = True
                continue
            lat = cam.get("lat") if cam.get("lat") is not None else cam.get("gps_lat")
            lon = cam.get("lon") if cam.get("lon") is not None else cam.get("gps_lon")
            db.add(Camera(
                camera_id=cid,
                name=cam.get("name", cid),
                url=cam.get("url", f"https://live.corp8.cloud/stream/{cid}"),
                lat=lat,
                lon=lon,
                gps_lat=lat,
                gps_lon=lon,
                district=cam.get("district", "Gujarat"),
                zone=cam.get("zone", "Central"),
                crime_level=cam.get("crime_level", "medium"),
                is_restricted=cam.get("is_restricted", False),
                department=cam.get("department", "Traffic Police"),
                status="ONLINE",
                is_online=True,
                created_at=datetime.utcnow(),
            ))
        db.commit()
    finally:
        db.close()
    print(f"✅ Seeded {len(config.get('demo_cameras', []))} cameras")


def seed_watchlist_plates():
    plates = [
        ("GJ05AB1234", "Armed robbery — Ahmedabad East", "stolen"),
        ("GJ01CD5678", "Murder suspect vehicle", "wanted"),
        ("GJ18EF9012", "Stolen — Rajkot district", "stolen"),
        ("GJ06GH3456", "Drug trafficking suspect", "suspect"),
        ("GJ09IJ7890", "Stolen — Surat district", "stolen"),
        ("GJ27KL2345", "Human trafficking suspect", "wanted"),
        ("GJ15MN6789", "Stolen — Gandhinagar", "stolen"),
        ("GJ03OP1234", "Terrorist watchlist", "wanted"),
    ]
    db = SessionLocal()
    try:
        for plate, reason, category in plates:
            if not db.query(WatchlistPlate).filter(WatchlistPlate.plate == plate).first():
                db.add(WatchlistPlate(
                    plate=plate,
                    plate_number=plate,
                    reason=reason,
                    category=category,
                    added_by="system",
                ))
        db.commit()
    finally:
        db.close()
    print(f"✅ Seeded {len(plates)} watchlist plates")


def seed_wanted_persons():
    persons = [
        ("REID_20260821_98A4", "Rahul Demo-Suspect", "Murder warrant", "critical"),
        ("REID_20260821_B7C2", "Unknown Suspect B", "Armed robbery", "high"),
        ("REID_20260821_D5E1", "Unknown Suspect C", "Drug trafficking", "high"),
    ]
    db = SessionLocal()
    try:
        for reid_id, name, reason, danger in persons:
            if not db.query(WatchlistPerson).filter(WatchlistPerson.reid_id == reid_id).first():
                dummy_vec = np.random.rand(512).astype(np.float32)
                dummy_vec /= np.linalg.norm(dummy_vec)
                db.add(WatchlistPerson(
                    reid_id=reid_id,
                    name=name,
                    embedding=json.dumps(dummy_vec.tolist()),
                    reason=reason,
                    danger_level=danger,
                ))
        db.commit()
    finally:
        db.close()
    print("✅ Seeded 3 wanted persons with embeddings")


def seed_officers():
    officers = [
        ("OFF_AHM_01", "Inspector Patel", "AHM001", "+919876543210",
         23.0395, 72.5797, "Central", "Ahmedabad"),
        ("OFF_AHM_02", "SI Sharma", "AHM002", "+919876543211",
         23.0225, 72.5714, "Central", "Ahmedabad"),
        ("OFF_RAJ_01", "Inspector Mehta", "RAJ001", "+919876543212",
         22.3039, 70.8022, "Saurashtra", "Rajkot"),
        ("OFF_RAJ_02", "SI Joshi", "RAJ002", "+919876543213",
         22.2965, 70.7984, "Saurashtra", "Rajkot"),
        ("OFF_KUT_01", "Inspector Parmar", "KUT001", "+919876543214",
         23.0853, 70.1337, "Kutch", "Kutch"),
        ("OFF_SGJ_01", "Inspector Desai", "SGJ001", "+919876543215",
         20.9467, 72.9520, "South Gujarat", "Navsari"),
        ("OFF_GAN_01", "SI Rao", "GAN001", "+919876543216",
         23.2156, 72.6369, "Capital", "Gandhinagar"),
        ("OFF_NGJ_01", "Inspector Shah", "NGJ001", "+919876543217",
         23.8589, 72.1265, "North Gujarat", "Patan"),
    ]
    db = SessionLocal()
    try:
        for oid, name, badge, phone, lat, lon, zone, district in officers:
            if not db.query(Officer).filter(Officer.badge_number == badge).first():
                db.add(Officer(
                    name=name, badge_number=badge, phone=phone,
                    current_lat=lat, current_lon=lon,
                    lat=lat, lng=lon,
                    zone=zone, district=district,
                    is_available=True,
                    status="AVAILABLE",
                    last_gps_update=datetime.utcnow(),
                ))
        db.commit()
    finally:
        db.close()
    print(f"✅ Seeded {len(officers)} patrol officers")


def seed_users():
    users = [
        ("USER_ADMIN_01", "System Administrator", "ADMIN001", "admin",
         "admin123", UserRole.admin, "All"),
        ("USER_OPS_01", "Control Room Operator", "OPS001", "operator",
         "operator123", UserRole.operator, "Central"),
        ("USER_VIEW_01", "Monitoring Viewer", "VIEW001", "viewer",
         "viewer123", UserRole.viewer, "All"),
    ]
    db = SessionLocal()
    try:
        for uid, name, badge, uname, password, role, zone in users:
            existing = db.query(User).filter((User.badge_number == badge) | (User.username == uname)).first()
            if not existing:
                hashed = hash_password(password)
                db.add(User(
                    id=uid, name=name, username=uname, badge_number=badge,
                    role=role.value, zone=zone,
                    password_hash=hashed,
                    hashed_password=hashed,
                    is_active=True,
                ))
            else:
                existing.hashed_password = hash_password(password)
                existing.password_hash = existing.hashed_password
        db.commit()
    finally:
        db.close()
    print("✅ Seeded 3 system users")
    print("   Admin:    ADMIN001 / admin123")
    print("   Operator: OPS001 / operator123")
    print("   Viewer:   VIEW001 / viewer123")


async def main():
    with open("config.yaml", "r", encoding="utf-8") as f:
        config = yaml.safe_load(f)

    print("\n🌱 Seeding Sentinel Gujarat database...\n")
    await init_db(drop_first=True)
    seed_cameras(config)
    seed_watchlist_plates()
    seed_wanted_persons()
    seed_officers()
    seed_users()
    print("\n🎉 Database seeding complete!")
    print(f"   Cameras:         30")
    print(f"   Watchlist plates: 8")
    print(f"   Wanted persons:   3")
    print(f"   Officers:         8")
    print(f"   System users:     3")


if __name__ == "__main__":
    asyncio.run(main())
