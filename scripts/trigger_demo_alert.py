#!/usr/bin/env python3
"""
scripts/trigger_demo_alert.py — Trigger a live threat alert in Sentinel Gujarat.
Inserts an alert and routes it to the nearest officer, instantly updating the Control Room dashboard.
"""
import sys
from pathlib import Path

# Add project root to sys.path
PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from backend.db.session import SessionLocal
from backend.db.models import Alert, Camera, Officer
from backend.routing.service import route_alert

def trigger_alert():
    db = SessionLocal()
    try:
        # Find camera
        cam = db.query(Camera).filter(Camera.is_deleted == False).first()
        cam_id = cam.id if cam else 1
        cam_name = cam.name if cam else "01 Chiman bhai Bridge"

        # Create threat alert
        alert = Alert(
            camera_id=cam_id,
            alert_type="FACE_WATCHLIST",
            subject_label="Wanted Suspect: Rahul (Gujarat Case #447)",
            confidence=0.92,
            danger_score=8.7,
            severity="critical",
            status="NEW",
            lifecycle_status="NEW",
            iq_contribution=8.7,
        )
        db.add(alert)
        db.commit()
        db.refresh(alert)
        print(f"✅ Alert #{alert.id} created on {cam_name} with SENTINEL IQ Score 8.7/10")

        # Route alert to nearest officer
        res = route_alert(alert.id, db)
        if isinstance(res, dict) and res.get("status") == "ROUTED":
            officer_id = res.get("officer_id")
            officer = db.query(Officer).filter(Officer.id == officer_id).first()
            officer_name = officer.name if officer else f"Officer #{officer_id}"
            print(f"🚨 GPS Auto-Routed to nearest officer: {officer_name} (Status: BUSY)")
        else:
            print("🚨 Alert active in Control Room dispatch queue.")

        print("\n🎉 Go to your browser dashboard at http://localhost:5173 — click Refresh or see the live alert card!")

    except Exception as e:
        print(f"❌ Error triggering alert: {e}")
        db.rollback()
    finally:
        db.close()

if __name__ == "__main__":
    trigger_alert()
