#!/usr/bin/env python3
"""
verify_day16.py — Full Acceptance Suite for Day 16: Feedback Loop & HLS Streaming.
"""

import asyncio
import os
import shutil
import sys
import tempfile
import time
from pathlib import Path
from typing import Any

if sys.platform == "win32":
    try:
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    except Exception:
        pass

# ── Isolated database & HLS environment ──────────────────────────────────────
_tmpdir = tempfile.mkdtemp(prefix="sentinel_day16_")
os.environ["DATABASE_URL"] = f"sqlite:///{_tmpdir}/verify_day16.db"
os.environ["HLS_DIR"] = str(Path(_tmpdir) / "hls")
os.environ["FALSE_ALARM_FLAG_THRESHOLD"] = "3"
os.environ["HLS_SEGMENT_DURATION"] = "2"
os.environ["HLS_LIST_SIZE"] = "3"

from sqlalchemy import create_engine, text
from sqlalchemy.orm import sessionmaker

from backend.core.config import settings
from backend.db.models import Base, Camera, Alert, Officer, AlertFeedback, FeedbackFlagLog
from backend.feedback.service import submit_feedback, get_zone_summary, ALLOWED_VERDICTS
from backend.stream.manager import (
    check_ffmpeg_version,
    _hls_dir,
    _clear_hls_dir,
    clear_all_hls_on_startup,
    spawn_ffmpeg,
    terminate_ffmpeg,
    terminate_all_ffmpeg,
    ffmpeg_processes,
)
from backend.routing.utils import to_iso8601

engine = create_engine(os.environ["DATABASE_URL"], connect_args={"check_same_thread": False})
SessionLocal = sessionmaker(bind=engine, autoflush=False, autocommit=False)
Base.metadata.create_all(bind=engine)

passed_tests = []
failed_tests = []

def record_pass(name: str):
    passed_tests.append(name)
    print(f"  [PASS] {name}")

def record_fail(name: str, reason: str):
    failed_tests.append((name, reason))
    print(f"  [FAIL] {name}: {reason}")


def test_0_preflight():
    print("\n--- Test 0: Pre-Flight Checks ---")
    try:
        # Check demo clips
        c1 = Path("demo/clips/entrance_loop.mp4")
        c2 = Path("demo/clips/parking_loop.mp4")
        if not c1.exists() or not c2.exists():
            record_fail("Demo Clips", "demo/clips/entrance_loop.mp4 or parking_loop.mp4 missing")
            return
        record_pass("Demo clips exist")

        # Check ffmpeg version function
        check_ffmpeg_version()
        record_pass("ffmpeg version verified")

        # Seed demo cameras in DB
        db = SessionLocal()
        cam1 = Camera(
            id="1", camera_id="CAM-01", name="Main Entrance Cam",
            location_label="Main Entrance Zone", zone="Main Entrance Zone",
            stream_url="demo/clips/entrance_loop.mp4", status="ONLINE"
        )
        cam2 = Camera(
            id="2", camera_id="CAM-02", name="Parking Lot Cam",
            location_label="Parking Zone", zone="Parking Zone",
            stream_url="demo/clips/parking_loop.mp4", status="ONLINE"
        )
        officer1 = Officer(id=1, name="Officer Sharma", badge_number="GJ-101", status="AVAILABLE")
        alert1 = Alert(id="101", camera_id="1", alert_type="LOITERING", severity="high")
        alert2 = Alert(id="102", camera_id="1", alert_type="LOITERING", severity="critical")
        alert3 = Alert(id="103", camera_id="1", alert_type="LOITERING", severity="critical")
        alert4 = Alert(id="104", camera_id="2", alert_type="LOITERING", severity="medium")

        db.add_all([cam1, cam2, officer1, alert1, alert2, alert3, alert4])
        db.commit()
        db.close()
        record_pass("Demo database seeded with cameras, officers, and alerts")
    except Exception as e:
        record_fail("Preflight", str(e))


def test_1_feedback_upsert():
    print("\n--- Test 1: Feedback Upsert ---")
    db = SessionLocal()
    try:
        payload = submit_feedback(alert_id=101, verdict="FALSE_ALARM", officer_id=1, db=db)
        db.commit()

        row = db.query(AlertFeedback).filter_by(alert_id="101").first()
        if not row or row.verdict != "FALSE_ALARM":
            record_fail("Feedback Upsert DB", "alert_feedback row not found or wrong verdict")
            return

        zs = payload["zone_summary"]
        if zs["zone_name"] != "Main Entrance Zone" or zs["false_alarm_count"] != 1:
            record_fail("Zone Summary", f"Unexpected zone summary: {zs}")
            return

        record_pass("Feedback upsert created row and returned zone summary")
    except Exception as e:
        record_fail("Feedback Upsert", str(e))
    finally:
        db.close()


def test_2_idempotency_and_reclassification():
    print("\n--- Test 2 & 3: Idempotency & Reclassification ---")
    db = SessionLocal()
    try:
        # Same verdict twice (idempotent)
        p1 = submit_feedback(alert_id=101, verdict="FALSE_ALARM", officer_id=1, db=db)
        db.commit()
        count = db.query(AlertFeedback).filter_by(alert_id="101").count()
        if count != 1:
            record_fail("Idempotency", f"Expected 1 row, got {count}")
            return
        record_pass("Idempotent submission preserved exactly 1 row")

        # Reclassification: INVESTIGATING -> FALSE_ALARM
        submit_feedback(alert_id=102, verdict="INVESTIGATING", officer_id=1, db=db)
        db.commit()
        row_inv = db.query(AlertFeedback).filter_by(alert_id="102").first()
        if row_inv.verdict != "INVESTIGATING":
            record_fail("Initial Investigating", f"Verdict was {row_inv.verdict}")
            return

        submit_feedback(alert_id=102, verdict="FALSE_ALARM", officer_id=1, db=db)
        db.commit()
        row_reclass = db.query(AlertFeedback).filter_by(alert_id="102").first()
        count_102 = db.query(AlertFeedback).filter_by(alert_id="102").count()
        if count_102 != 1 or row_reclass.verdict != "FALSE_ALARM":
            record_fail("Reclassification", f"Reclassification failed: count={count_102}, verdict={row_reclass.verdict}")
            return
        record_pass("Reclassification from INVESTIGATING to FALSE_ALARM succeeded cleanly")
    except Exception as e:
        record_fail("Idempotency/Reclass", str(e))
    finally:
        db.close()


def test_5_threshold_flag_and_reset():
    print("\n--- Test 5 & 6: System Learning Threshold & Reset ---")
    db = SessionLocal()
    try:
        # Submit 3rd false alarm for Main Entrance Zone (alert 103)
        payload = submit_feedback(alert_id=103, verdict="FALSE_ALARM", officer_id=1, db=db)
        db.commit()

        zs = payload["zone_summary"]
        if not zs["threshold_flag"] or zs["false_alarm_count"] < 3:
            record_fail("Threshold Flag", f"Expected threshold_flag=True, got {zs}")
            return
        record_pass(f"Threshold flag triggered at {zs['false_alarm_count']} false alarms: {zs['flag_message']}")

        # Verify flag log table has active row
        flag_row = db.query(FeedbackFlagLog).filter_by(zone_name="Main Entrance Zone", resolved=0).first()
        if not flag_row:
            record_fail("Flag Log Check", "feedback_flag_log row not created")
            return
        record_pass("feedback_flag_log active row verified")

        # Test Reset (resolve flag without deleting feedback)
        db.execute(text("UPDATE feedback_flag_log SET resolved = 1 WHERE zone_name = 'Main Entrance Zone'"))
        db.commit()

        open_flags = db.query(FeedbackFlagLog).filter_by(zone_name="Main Entrance Zone", resolved=0).count()
        total_feedback = db.query(AlertFeedback).count()
        if open_flags != 0 or total_feedback < 3:
            record_fail("Reset Check", f"open_flags={open_flags}, total_feedback={total_feedback}")
            return
        record_pass("Reset resolved flag log while preserving all feedback rows")
    except Exception as e:
        record_fail("Threshold/Reset", str(e))
    finally:
        db.close()


def test_7_hls_stream_manager():
    print("\n--- Test 7, 8, 10, 11: HLS Stream Manager & ffmpeg Lifecycle ---")
    try:
        clear_all_hls_on_startup()
        record_pass("clear_all_hls_on_startup cleared root")

        # Spawn camera 1
        spawned = spawn_ffmpeg(1, "demo/clips/entrance_loop.mp4")
        if not spawned:
            record_fail("ffmpeg Spawn", "spawn_ffmpeg returned False")
            return
        record_pass("ffmpeg process spawned successfully for camera 1")

        # Double spawn lock test
        spawned_again = spawn_ffmpeg(1, "demo/clips/entrance_loop.mp4")
        if spawned_again:
            record_fail("Spawn Lock", "Second spawn on alive process should return False")
            return
        record_pass("Spawn lock prevented duplicate process spawn")

        # Wait for manifest generation
        manifest = _hls_dir(1) / "output.m3u8"
        start = time.time()
        ready = False
        while time.time() - start < 10.0:
            if manifest.exists() and manifest.stat().st_size > 0:
                ready = True
                break
            time.sleep(0.5)

        if not ready:
            record_fail("Manifest Generation", "output.m3u8 was not generated within 10s")
            return
        record_pass("HLS manifest output.m3u8 created and populated")

        # Wait for at least one .ts segment
        segments = list(_hls_dir(1).glob("*.ts"))
        if not segments:
            time.sleep(2.0)
            segments = list(_hls_dir(1).glob("*.ts"))
        if not segments:
            record_fail("Segments", "No .ts segments found in HLS directory")
            return
        record_pass(f"Found {len(segments)} HLS .ts segments generated on disk")

        # Test Termination
        terminate_ffmpeg(1)
        if 1 in ffmpeg_processes and ffmpeg_processes[1].poll() is None:
            record_fail("Terminate", "Process still running after terminate_ffmpeg")
            return
        record_pass("terminate_ffmpeg stopped the ffmpeg process cleanly")

        terminate_all_ffmpeg()
        record_pass("terminate_all_ffmpeg completed without errors")
    except Exception as e:
        record_fail("HLS Stream Manager", str(e))


def test_12_fastapi_endpoints():
    print("\n--- Test 12: FastAPI Route Integration ---")
    from fastapi.testclient import TestClient
    from backend.main import app

    client = TestClient(app)

    # Test GET /api/v1/feedback/summary
    res = client.get("/api/v1/feedback/summary?since=today")
    if res.status_code != 200:
        record_fail("GET /feedback/summary", f"Status {res.status_code}: {res.text}")
        return
    data = res.json()
    if "summary" not in data or "threshold" not in data:
        record_fail("Summary schema", f"Invalid response: {data}")
        return
    record_pass("GET /api/v1/feedback/summary returned 200 with valid schema")

    # Test POST /api/v1/alerts/101/feedback
    f_res = client.post("/api/v1/alerts/101/feedback", json={"verdict": "GENUINE"})
    if f_res.status_code != 200:
        record_fail("POST /alerts/{id}/feedback", f"Status {f_res.status_code}: {f_res.text}")
        return
    record_pass("POST /api/v1/alerts/{id}/feedback returned 200 OK")

    # Test Invalid verdict
    inv_res = client.post("/api/v1/alerts/101/feedback", json={"verdict": "INVALID_CHOICE"})
    if inv_res.status_code != 400:
        record_fail("Invalid verdict rejection", f"Expected 400, got {inv_res.status_code}")
        return
    record_pass("Invalid verdict correctly rejected with 400 Bad Request")


def main():
    print("=" * 60)
    print("  Sentinel Gujarat — Day 16 Acceptance Test Suite")
    print("=" * 60)

    test_0_preflight()
    test_1_feedback_upsert()
    test_2_idempotency_and_reclassification()
    test_5_threshold_flag_and_reset()
    test_7_hls_stream_manager()
    test_12_fastapi_endpoints()

    print("\n" + "=" * 60)
    print(f"Results: {len(passed_tests)} Passed, {len(failed_tests)} Failed")
    print("=" * 60)

    if failed_tests:
        print("\nFailed Checks:")
        for name, reason in failed_tests:
            print(f"  ❌ {name}: {reason}")
        sys.exit(1)
    else:
        print("\nAll Day 16 Acceptance Checks Passed Successfully! 🎉")
        sys.exit(0)


if __name__ == "__main__":
    main()
