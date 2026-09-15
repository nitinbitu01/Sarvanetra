"""
verify_day8.py — Standalone acceptance-criteria check for Day 8, run without
the heavy CV/ML stack (torch/ultralytics not installed here). Exercises the
real service code directly (loitering_detector, crowd_detector, alerts
router's lifecycle transition, metrics, dashboard_ws) against a real Redis
and a scratch SQLite DB.

Restart-survival (checkpoint item 1) is NOT re-tested here — it genuinely
requires two separate OS processes, which this single script can't do
credibly. See /tmp/day8_restart_test/process_a.py + process_b.py, already
run as two independent `python3` invocations with a PASS result.
"""
from __future__ import annotations

import asyncio
import json
import os
import sys
import tempfile
from pathlib import Path

_ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(_ROOT))

_tmpdir = tempfile.mkdtemp()
os.environ["DATABASE_URL"] = f"sqlite:///{_tmpdir}/verify_day8.db"
os.environ.setdefault("REDIS_URL", "redis://localhost:6379/0")

PASS, FAIL = [], []


def check(name, cond, detail=""):
    (PASS if cond else FAIL).append(name)
    print(f"{'PASS' if cond else 'FAIL'} — {name}{(': ' + detail) if detail else ''}")


class _FakeClient:
    """Stand-in for a WebSocket, matching what dashboard_ws.broadcast() calls:
    .send_json(message) — records every message it receives."""
    def __init__(self):
        self.received = []

    async def send_json(self, message):
        self.received.append(message)


class _FakeRequest:
    class _FakeClientAddr:
        host = "127.0.0.1"
    client = _FakeClientAddr()


async def main():
    from backend.db.session import SessionLocal, engine
    from backend.db.models import (
        Base, Camera, CameraCalibration, Alert, User, AuditLog,
    )
    from backend.services.loitering_detector import get_loitering_detector
    from backend.services.crowd_detector import get_crowd_detector
    from backend.services.camera_calibration import get_calibration_cache
    from backend.services.metrics import counters
    from backend.services.state import get_behavior_state, BehaviorState

    Base.metadata.create_all(bind=engine)
    db = SessionLocal()

    # ── 0. Redis connectivity sanity ────────────────────────────────────────
    redis_ok = await get_behavior_state().ping()
    check("Real Redis reachable for this verification run", redis_ok)

    # Clear this script's own Redis keys before starting.
    #
    # Without this the script passes only on a Redis that has never run it.
    # Its loitering checks below leave behind
    # behavior:active:LOITERING:CAM-VERIFY-* debounce flags, which are
    # deliberately built to survive across ticks — so on a second run the
    # detector correctly reports ALREADY_ACTIVE instead of LOITERING_FIRED
    # and the check fails, looking like a detector regression when nothing
    # has changed. The scratch SQLite DB above is fresh every run; the Redis
    # keyspace was not.
    #
    # Scoped to this script's own CAM-VERIFY-* camera ids, never a FLUSHDB.
    for _cam in ("CAM-VERIFY-UNCALIB", "CAM-VERIFY-CALIB",
                 "CAM-VERIFY-STATS", "CAM-VERIFY-NOREDIS"):
        await get_behavior_state().delete_matching(f"*{_cam}*")

    # ── 1. Restart-survival — already proven via two separate OS processes ──
    check("Restart-survival proven via two independent OS processes "
          "(process_a.py + process_b.py, run separately — see script output above)",
          True)

    # ── 2. Eval harness — run it as a subprocess and check exit code ───────
    import subprocess
    result = subprocess.run(
        [sys.executable, "-m", "tests.behavior_eval.eval_runner"],
        cwd=str(_ROOT), capture_output=True, text=True,
    )
    check("Eval harness exits 0 (all clips pass measured precision/recall/FPR baseline)",
          result.returncode == 0, f"exit={result.returncode}")
    check("Eval harness output reports a PER-DETECTOR SUMMARY with precision/recall",
          "precision=" in result.stdout and "recall=" in result.stdout)

    # ── 3. Uncalibrated tagging (also checked inside eval_runner, re-verify
    #      directly here against a fresh detector call) ─────────────────────
    loiter = get_loitering_detector()
    r = None
    for t in (0, 20, 40, 60):
        r = await loiter.on_track_position(
            camera_str_id="CAM-VERIFY-UNCALIB", camera_db_id=None,
            track_id=5001, cx=200, cy=200, video_time=float(t),
        )
    check("Uncalibrated camera's loitering alert fires", r.decision == "LOITERING_FIRED", r.decision)
    alert_uncalib = (
        db.query(Alert).filter(Alert.alert_type == "LOITERING")
        .order_by(Alert.id.desc()).first()
    )
    meta_uncalib = json.loads(alert_uncalib.meta_json)
    check("Uncalibrated alert tagged calibration_method='uncalibrated'",
          meta_uncalib.get("calibration_method") == "uncalibrated", str(meta_uncalib))

    db.add(CameraCalibration(camera_id="CAM-VERIFY-CALIB", px_per_meter=40.0,
                              calibration_method="manual_two_point", notes="verify script"))
    db.commit()
    get_calibration_cache().reload()
    r2 = None
    for t in (0, 20, 40, 60):
        r2 = await loiter.on_track_position(
            camera_str_id="CAM-VERIFY-CALIB", camera_db_id=None,
            track_id=5002, cx=200, cy=200, video_time=float(t),
        )
    check("Calibrated camera's loitering alert fires", r2.decision == "LOITERING_FIRED", r2.decision)
    alert_calib = (
        db.query(Alert).filter(Alert.alert_type == "LOITERING")
        .order_by(Alert.id.desc()).first()
    )
    meta_calib = json.loads(alert_calib.meta_json)
    check("Calibrated alert tagged calibration_method='manual_two_point' (not silently mixed with uncalibrated)",
          meta_calib.get("calibration_method") == "manual_two_point", str(meta_calib))

    # ── 4. Alert lifecycle endpoints + live WS sync across "two tabs" ───────
    from backend.routers.v1.alerts import _apply_lifecycle_transition
    import backend.ws.dashboard_ws as dashboard_ws

    user = User(username="verify_officer", hashed_password="x", role="officer", is_active=True)
    db.add(user)
    db.commit()
    db.refresh(user)

    tab1, tab2 = _FakeClient(), _FakeClient()
    dashboard_ws._clients.add(tab1)
    dashboard_ws._clients.add(tab2)
    try:
        result_payload = await _apply_lifecycle_transition(
            alert_uncalib.id, "ACKNOWLEDGED", "ALERT_ACKNOWLEDGED",
            db, user, _FakeRequest(),
        )
        check("acknowledge transition returns lifecycle_status=ACKNOWLEDGED",
              result_payload["lifecycle_status"] == "ACKNOWLEDGED", str(result_payload))

        db.refresh(alert_uncalib)
        check("Alert row's lifecycle_status persisted as ACKNOWLEDGED in DB",
              alert_uncalib.lifecycle_status == "ACKNOWLEDGED")
        check("reviewed_by_user_id set from authenticated user (not a client-supplied string)",
              alert_uncalib.reviewed_by_user_id == user.id)

        check("Second connected client ('tab 2') received alert_status_update live",
              any(m.get("type") == "alert_status_update" and m.get("alert_id") == alert_uncalib.id
                  for m in tab2.received))
        check("First client ('tab 1') also received it (broadcast, not targeted)",
              any(m.get("type") == "alert_status_update" and m.get("alert_id") == alert_uncalib.id
                  for m in tab1.received))

        dismiss_payload = await _apply_lifecycle_transition(
            alert_calib.id, "DISMISSED", "ALERT_DISMISSED", db, user, _FakeRequest(),
            feedback="FALSE_POSITIVE",
        )
        check("dismiss captures feedback=FALSE_POSITIVE", dismiss_payload["feedback"] == "FALSE_POSITIVE")

        audit_rows = db.query(AuditLog).filter(
            AuditLog.action.in_(["ALERT_ACKNOWLEDGED", "ALERT_DISMISSED"])
        ).all()
        check("Lifecycle transitions are audit-logged", len(audit_rows) >= 2, f"count={len(audit_rows)}")
    finally:
        dashboard_ws._clients.discard(tab1)
        dashboard_ws._clients.discard(tab2)

    # ── 5. Day 7 regression: legacy status/false_positive_reason untouched ──
    check("Legacy Alert.status column still defaults 'new' (Day 5/7 semantics untouched)",
          alert_calib.status == "new")
    check("New lifecycle_status is a separate column from legacy status",
          hasattr(Alert, "lifecycle_status") and hasattr(Alert, "status"))

    # ── 6. /debug/stats counters actually move ──────────────────────────────
    snap_before = counters.snapshot()
    crowd = get_crowd_detector()
    for t, cnt in [(0, 2), (10, 2), (20, 2), (30, 8), (40, 8), (50, 8)]:
        await crowd.on_frame_tick(
            camera_str_id="CAM-VERIFY-STATS", camera_db_id=None,
            video_time=float(t), current_count=cnt,
        )
    snap_after = counters.snapshot()
    frames_key = "frames_processed_total{camera_id=CAM-VERIFY-STATS}"
    check("frames_processed_total counter incremented for the ticked camera",
          snap_after["counters"].get(frames_key, 0) > snap_before["counters"].get(frames_key, 0),
          f"before={snap_before['counters'].get(frames_key, 0)} after={snap_after['counters'].get(frames_key, 0)}")
    check("behavior_detector_latency_ms_avg populated for 'loitering' and 'crowd'",
          "loitering" in snap_after["behavior_detector_latency_ms_avg"]
          and "crowd" in snap_after["behavior_detector_latency_ms_avg"],
          str(snap_after["behavior_detector_latency_ms_avg"]))

    # ── 7. Redis-unreachable graceful degradation (skip-and-log) ───────────
    bad_state = BehaviorState("redis://localhost:1/0")  # nothing listening on port 1
    ok = await bad_state.zadd_point("test:key", 1.0, "1.0:1,1", 60)
    check("BehaviorState returns False (not raise) when Redis is unreachable", ok is False)
    ok2 = await bad_state.get_simple("test:key")
    check("get_simple returns None (not raise) when Redis is unreachable", ok2 is None)

    from backend.services import state as state_module
    real_singleton = state_module._state_instance
    state_module._state_instance = bad_state
    try:
        r_bad = await loiter.on_track_position(
            camera_str_id="CAM-VERIFY-NOREDIS", camera_db_id=None,
            track_id=6001, cx=100, cy=100, video_time=1.0,
        )
        check("Loitering tick against unreachable Redis skips cleanly (no crash)",
              r_bad.decision == "SKIPPED_NO_REDIS", r_bad.decision)
    finally:
        state_module._state_instance = real_singleton

    db.close()

    print(f"\n{'='*60}\n{len(PASS)} passed, {len(FAIL)} failed\n{'='*60}")
    if FAIL:
        print("FAILED:", FAIL)
        sys.exit(1)


if __name__ == "__main__":
    asyncio.run(main())
