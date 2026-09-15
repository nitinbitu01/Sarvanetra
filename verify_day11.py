"""
verify_day11.py — Day 11 Alert Fatigue Management Verification.

Covers all 6 Checkpoints from the Day 11 prompt:
  1. Watchlist match across 3 cameras (same global_id, within seconds) ->
     exactly one primary alert in default feed (highest confidence wins),
     and the other 2 are merged into it and reachable via /alerts/{id}/merged.
  2. Tie-break fallback for alerts without confidence (loitering) ->
     earliest timestamp wins.
  3. 5+ mixed-type alerts within 100m in 2 minutes -> opens a single
     ZoneIncident using real haversine distance, referencing all 5+ alert IDs.
  4. Alerts with global_id=NULL are never dedup'd against each other.
  4a. Duplicate camera identities at one physical location (same GPS) cluster
     cleanly via haversine distance into a single incident.
  5. DB check: no Alert row is ever deleted or marked is_deleted by either feature.
  6. Out-of-scope check: load-balancing / officer roster was not built.

Run:
  python verify_day11.py
"""
from __future__ import annotations

import json
import os
import sys
import tempfile
from datetime import datetime, timedelta
from pathlib import Path

_ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(_ROOT))

_tmpdir = tempfile.mkdtemp(prefix="verify_day11_")
os.environ["DATABASE_URL"] = f"sqlite:///{_tmpdir}/verify_day11.db"
os.environ.pop("SENTINEL_SOURCE", None)

PASS: list[str] = []
FAIL: list[str] = []

ADMIN_PW = "verify-admin-pw"


def check(name: str, cond: bool, detail: str = "") -> None:
    (PASS if cond else FAIL).append(name)
    print(f"{'PASS' if cond else 'FAIL'} — {name}{(': ' + detail) if detail else ''}")


def run_checks() -> None:
    import asyncio
    from fastapi.testclient import TestClient
    from backend.auth.routes import pwd_context
    from backend.db.models import Alert, Base, Camera, User, ZoneIncident
    from backend.db.session import SessionLocal, engine
    from backend.services.alert_dedup import apply_dedup
    from backend.services.zone_incident_manager import check_zone_incident, haversine_meters
    from backend.main import app

    # Create tables
    Base.metadata.create_all(bind=engine)
    db = SessionLocal()

    # Seed Admin User and Cameras
    admin = User(
        username="admin_test",
        hashed_password=pwd_context.hash(ADMIN_PW),
        role="admin",
        is_active=True,
    )
    db.add(admin)

    # Real Ahmedabad area GPS locations
    # CAM 1, 2, 3: Near Gandhi Road (~30m apart)
    cam1 = Camera(name="Gandhi Rd Cam-1", camera_id="CAM-G1", zone="Old City", gps_lat=23.0225, gps_lon=72.5714, status="ONLINE")
    cam2 = Camera(name="Gandhi Rd Cam-2", camera_id="CAM-G2", zone="Old City", gps_lat=23.0227, gps_lon=72.5716, status="ONLINE")
    cam3 = Camera(name="Gandhi Rd Cam-3", camera_id="CAM-G3", zone="Old City", gps_lat=23.0226, gps_lon=72.5715, status="ONLINE")

    # CAM 4a, 4b, 4c: Duplicate identities at Hwy Junction (Exact same GPS coords)
    cam4a = Camera(name="Gandhinagar Hwy Cam-4", camera_id="CAM-HWY-A", zone="Highway", gps_lat=23.1000, gps_lon=72.6000, status="ONLINE")
    cam4b = Camera(name="Gandhinagar Hwy Cam-4", camera_id="CAM-HWY-B", zone="Highway", gps_lat=23.1000, gps_lon=72.6000, status="ONLINE")
    cam4c = Camera(name="Gandhinagar Hwy Cam-4", camera_id="CAM-HWY-C", zone="Highway", gps_lat=23.1000, gps_lon=72.6000, status="ONLINE")

    # CAM 5: Distant camera in Surat (~200km away)
    cam5 = Camera(name="Surat Station Cam-1", camera_id="CAM-SURAT", zone="Surat", gps_lat=21.2000, gps_lon=72.8300, status="ONLINE")

    db.add_all([cam1, cam2, cam3, cam4a, cam4b, cam4c, cam5])
    db.commit()

    client = TestClient(app)
    # Auth token
    login_resp = client.post("/api/v1/auth/login", json={"username": "admin_test", "password": ADMIN_PW})
    assert login_resp.status_code == 200, f"Login failed: {login_resp.text}"
    token = login_resp.json()["access_token"]
    headers = {"Authorization": f"Bearer {token}"}

    # ── Checkpoint 1: 3-camera Watchlist Match Dedup (Confidence tie-break) ──
    print("\n--- Checkpoint 1: 3-Camera Watchlist Dedup ---")
    t0 = datetime.utcnow()
    # Alert 1: Cam 1, confidence 0.82
    a1 = Alert(
        alert_type="WATCHLIST_FACE_MATCH",
        camera_id=cam1.id,
        global_id="GP-101",
        confidence=0.82,
        subject_label="Wanted Person A",
        created_at=t0,
    )
    db.add(a1)
    db.flush()
    p1 = apply_dedup(a1, db)
    db.commit()
    check("CP1.1: First alert is primary", p1.id == a1.id and a1.merged_into_alert_id is None)

    # Alert 2: Cam 2, confidence 0.95 (higher than Alert 1) -> should become primary, Alert 1 merged into Alert 2
    a2 = Alert(
        alert_type="WATCHLIST_FACE_MATCH",
        camera_id=cam2.id,
        global_id="GP-101",
        confidence=0.95,
        subject_label="Wanted Person A",
        created_at=t0 + timedelta(seconds=2),
    )
    db.add(a2)
    db.flush()
    p2 = apply_dedup(a2, db)
    db.commit()
    db.refresh(a1)
    db.refresh(a2)
    check("CP1.2: Higher confidence alert wins primary", p2.id == a2.id and a2.merged_into_alert_id is None)
    check("CP1.3: Lower confidence prior alert merged into new winner", a1.merged_into_alert_id == a2.id)

    # Alert 3: Cam 3, confidence 0.79 -> lower than Alert 2, merges into Alert 2
    a3 = Alert(
        alert_type="WATCHLIST_FACE_MATCH",
        camera_id=cam3.id,
        global_id="GP-101",
        confidence=0.79,
        subject_label="Wanted Person A",
        created_at=t0 + timedelta(seconds=5),
    )
    db.add(a3)
    db.flush()
    p3 = apply_dedup(a3, db)
    db.commit()
    db.refresh(a3)
    check("CP1.4: Lower confidence third alert loses and merges into Alert 2", a3.merged_into_alert_id == a2.id and p3.id == a2.id)

    # Verify via API: Default feed returns exactly a2, with merged_count = 2
    feed_resp = client.get("/api/v1/alerts", headers=headers)
    feed_data = feed_resp.json()
    feed_ids = [a["id"] for a in feed_data]
    check("CP1.5: Only primary alert (a2) in default feed", a2.id in feed_ids and a1.id not in feed_ids and a3.id not in feed_ids)
    a2_item = next(a for a in feed_data if a["id"] == a2.id)
    check("CP1.6: Primary alert has merged_count == 2", a2_item.get("merged_count") == 2)

    # Verify via /alerts/{id}/merged: returns a1 and a3
    merged_resp = client.get(f"/api/v1/alerts/{a2.id}/merged", headers=headers)
    merged_data = merged_resp.json()
    merged_ids = [m["id"] for m in merged_data]
    check("CP1.7: /alerts/{id}/merged returns both merged constituents", set(merged_ids) == {a1.id, a3.id})

    # ── Checkpoint 2: Loitering Tie-Break (No confidence -> Earliest timestamp wins) ──
    print("\n--- Checkpoint 2: Loitering Earliest-Timestamp Tie-Break ---")
    t_loiter = datetime.utcnow()
    # Alert L1: created at t_loiter
    l1 = Alert(
        alert_type="LOITERING",
        camera_id=cam1.id,
        global_id="GP-202",
        confidence=None,
        subject_label="Loitering Cam-1",
        created_at=t_loiter,
    )
    db.add(l1)
    db.flush()
    p_l1 = apply_dedup(l1, db)
    db.commit()

    # Alert L2: created at t_loiter + 10s (later timestamp)
    l2 = Alert(
        alert_type="LOITERING",
        camera_id=cam2.id,
        global_id="GP-202",
        confidence=None,
        subject_label="Loitering Cam-2",
        created_at=t_loiter + timedelta(seconds=10),
    )
    db.add(l2)
    db.flush()
    p_l2 = apply_dedup(l2, db)
    db.commit()
    db.refresh(l1)
    db.refresh(l2)

    check("CP2.1: Earliest timestamp wins (l1 is primary)", p_l2.id == l1.id and l1.merged_into_alert_id is None)
    check("CP2.2: Later alert l2 merged into l1", l2.merged_into_alert_id == l1.id)

    # Cross-type check: WATCHLIST alert for same global_id must NOT dedup with LOITERING
    w_other = Alert(
        alert_type="WATCHLIST_FACE_MATCH",
        camera_id=cam1.id,
        global_id="GP-202",
        confidence=0.88,
        subject_label="Watchlist hit",
        created_at=t_loiter + timedelta(seconds=12),
    )
    db.add(w_other)
    db.flush()
    p_w = apply_dedup(w_other, db)
    db.commit()
    check("CP2.3: No cross-alert-type dedup (different type not merged)", p_w.id == w_other.id and w_other.merged_into_alert_id is None)

    # ── Checkpoint 3: Spatial Zone Incident (5+ alerts, 100m, 2 min) ─────────
    print("\n--- Checkpoint 3: Spatial Zone Incident (5+ alerts within 100m) ---")
    t_zone = datetime.utcnow()
    # Distance between CAM-G1, CAM-G2, CAM-G3 is ~20-30m (< 100m)
    dist_g1_g2 = haversine_meters(cam1.gps_lat, cam1.gps_lon, cam2.gps_lat, cam2.gps_lon)
    check("CP3.1: Camera distance < 100m", dist_g1_g2 < 50.0, f"Distance is {dist_g1_g2:.1f}m")

    # Fire 5 mixed-type alerts at Gandhi Rd cameras within 2 minutes
    zone_alerts = [
        Alert(alert_type="LOITERING", camera_id=cam1.id, subject_label="Loiter 1", created_at=t_zone),
        Alert(alert_type="CROWD_ANOMALY", camera_id=cam2.id, subject_label="Crowd 1", created_at=t_zone + timedelta(seconds=5)),
        Alert(alert_type="ABANDONED_OBJECT", camera_id=cam3.id, subject_label="Bag 1", created_at=t_zone + timedelta(seconds=10)),
        Alert(alert_type="LOITERING", camera_id=cam1.id, subject_label="Loiter 2", created_at=t_zone + timedelta(seconds=15)),
        Alert(alert_type="CROWD_ANOMALY", camera_id=cam2.id, subject_label="Crowd 2", created_at=t_zone + timedelta(seconds=20)),
    ]

    for za in zone_alerts:
        db.add(za)
        db.flush()
        asyncio.run(check_zone_incident(za, db))
        db.commit()

    # Verify ZoneIncident created in DB
    incidents = db.query(ZoneIncident).filter(ZoneIncident.closed_at.is_(None)).all()
    check("CP3.2: ZoneIncident opened in DB", len(incidents) >= 1)
    gandhi_inc = next((inc for inc in incidents if inc.zone == "Old City"), None)
    check("CP3.3: Correct zone label on incident", gandhi_inc is not None)

    if gandhi_inc:
        ids_in_inc = json.loads(gandhi_inc.alert_ids)
        check("CP3.4: Incident references all 5+ constituent alert IDs", len(ids_in_inc) >= 5)

        # Distant camera (Surat, 200km away) alert must NOT attach to this incident
        surat_alert = Alert(alert_type="LOITERING", camera_id=cam5.id, subject_label="Surat loiter", created_at=t_zone + timedelta(seconds=25))
        db.add(surat_alert)
        db.flush()
        asyncio.run(check_zone_incident(surat_alert, db))
        db.commit()
        db.refresh(gandhi_inc)
        ids_after_surat = json.loads(gandhi_inc.alert_ids)
        check("CP3.5: Distant alert (200km) excluded by haversine distance", surat_alert.id not in ids_after_surat)

    # Verify Zone Incident API
    zi_resp = client.get("/api/v1/zone-incidents?expand_alerts=true", headers=headers)
    check("CP3.6: GET /zone-incidents returns HTTP 200", zi_resp.status_code == 200)
    zi_data = zi_resp.json()
    check("CP3.7: Open zone incident listed in API", any(z["zone"] == "Old City" for z in zi_data))

    # ── Checkpoint 4: global_id=NULL Alerts are Never Dedup'd ────────────────
    print("\n--- Checkpoint 4: global_id=NULL Alerts Never Dedup ---")
    t_null = datetime.utcnow()
    n1 = Alert(alert_type="LOITERING", camera_id=cam1.id, global_id=None, subject_label="Null ID 1", created_at=t_null)
    n2 = Alert(alert_type="LOITERING", camera_id=cam1.id, global_id=None, subject_label="Null ID 2", created_at=t_null + timedelta(seconds=2))
    db.add_all([n1, n2])
    db.flush()
    p_n1 = apply_dedup(n1, db)
    p_n2 = apply_dedup(n2, db)
    db.commit()
    db.refresh(n1)
    db.refresh(n2)

    check("CP4.1: First null-global_id alert is primary", p_n1.id == n1.id and n1.merged_into_alert_id is None)
    check("CP4.2: Second null-global_id alert is NOT dedup'd (remains primary)", p_n2.id == n2.id and n2.merged_into_alert_id is None)

    # ── Checkpoint 4a: Duplicate Camera IDs Collapse Cleanly via Haversine ────
    print("\n--- Checkpoint 4a: Duplicate Camera Identities Collapse Cleanly ---")
    # Fire 5 alerts across duplicate Hwy cameras (CAM-HWY-A, B, C)
    hwy_alerts = [
        Alert(alert_type="LOITERING", camera_id=cam4a.id, subject_label="Hwy 1", created_at=t_zone),
        Alert(alert_type="LOITERING", camera_id=cam4b.id, subject_label="Hwy 2", created_at=t_zone + timedelta(seconds=5)),
        Alert(alert_type="CROWD_ANOMALY", camera_id=cam4c.id, subject_label="Hwy 3", created_at=t_zone + timedelta(seconds=10)),
        Alert(alert_type="CROWD_ANOMALY", camera_id=cam4a.id, subject_label="Hwy 4", created_at=t_zone + timedelta(seconds=15)),
        Alert(alert_type="ABANDONED_OBJECT", camera_id=cam4b.id, subject_label="Hwy 5", created_at=t_zone + timedelta(seconds=20)),
    ]
    for ha in hwy_alerts:
        db.add(ha)
        db.flush()
        asyncio.run(check_zone_incident(ha, db))
        db.commit()

    hwy_incidents = db.query(ZoneIncident).filter(ZoneIncident.zone == "Highway", ZoneIncident.closed_at.is_(None)).all()
    check("CP4a.1: Exactly ONE ZoneIncident for duplicate camera location", len(hwy_incidents) == 1)
    if hwy_incidents:
        hwy_ids = json.loads(hwy_incidents[0].alert_ids)
        check("CP4a.2: All 5 alerts across duplicate cameras clustered in one incident", len(hwy_ids) == 5)

    # ── Checkpoint 5: No Alert Row is Ever Deleted ───────────────────────────
    print("\n--- Checkpoint 5: Evidence Integrity (No Alerts Deleted) ---")
    total_alerts_count = db.query(Alert).count()
    deleted_alerts_count = db.query(Alert).filter(Alert.is_deleted == True).count()
    check("CP5.1: Zero alerts marked is_deleted=True", deleted_alerts_count == 0)
    check("CP5.2: All alerts remain fully queryable in DB", total_alerts_count >= 15)

    # ── Checkpoint 6: Load-Balancing Feature was NOT Built (Out of Scope) ────
    print("\n--- Checkpoint 6: Scope Boundary Verification ---")
    # Verify no officer roster / load-balancing files or classes exist in backend/services
    services_dir = _ROOT / "backend" / "services"
    forbidden_terms = ["officer_roster", "load_balance_officer", "officer_shift", "max_10_per_hr"]
    found_forbidden = False
    for py_file in services_dir.glob("*.py"):
        text = py_file.read_text(encoding="utf-8").lower()
        for term in forbidden_terms:
            if term in text:
                print(f"Found forbidden term '{term}' in {py_file.name}")
                found_forbidden = True

    check("CP6.1: Load-balancing / officer roster explicitly not implemented", not found_forbidden)

    # Final Summary
    print("\n" + "=" * 60)
    print(f"VERIFICATION RESULTS: {len(PASS)} PASSED, {len(FAIL)} FAILED")
    print("=" * 60)
    if FAIL:
        sys.exit(1)


if __name__ == "__main__":
    run_checks()
