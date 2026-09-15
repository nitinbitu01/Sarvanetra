"""
verify_multicamera_and_interlock.py — the two readiness blockers.

PART 1 — MULTI-CAMERA
  camera identity used to live in a module-level global
  (connection_manager._reid_camera_db_id), set once at startup. With two
  feeds running, every alert from the second was stamped with the FIRST
  camera's cameras.id. Per-camera IQ scores, zone incidents, cross-zone
  journeys and Day 14 officer dispatch all key off that id, so they produced
  confidently wrong answers rather than failing.

  A test that ran one camera would pass either way. This runs TWO
  ConnectionManagers concurrently and asserts each stamps its own id — the
  only shape that can tell the two implementations apart.

PART 2 — FACE-MATCH DISPATCH INTERLOCK
  WATCHLIST_FACE_MATCH_THRESHOLD has never been validated on real data.
  Day 14 made that operationally dangerous: a face match is CRITICAL, and
  CRITICAL auto-dispatches a named officer to a physical location.

  The interlock withholds ONLY the automatic dispatch. These tests assert
  both halves: that dispatch is withheld, and that everything else about the
  alert still works — because an interlock that silently suppressed the
  alert would be a worse failure than the one it prevents.

Run:  python verify_multicamera_and_interlock.py
"""
from __future__ import annotations

import asyncio
import os
import sys
import tempfile
from datetime import datetime
from pathlib import Path

_ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(_ROOT))

_tmpdir = tempfile.mkdtemp(prefix="verify_mc_")
os.environ["DATABASE_URL"] = f"sqlite:///{_tmpdir}/verify_mc.db"
os.environ["EVIDENCE_DIR"] = str(Path(_tmpdir) / "evidence")
os.environ.pop("SENTINEL_SOURCE", None)

PASS: list[str] = []
FAIL: list[str] = []


def check(name: str, cond: bool, detail: str = "") -> None:
    (PASS if cond else FAIL).append(name)
    print(f"{'PASS' if cond else 'FAIL'} — {name}{(': ' + detail) if detail else ''}")


async def main() -> None:
    from sqlalchemy import text

    from backend.db.models import Alert, Base, Camera
    from backend.db.session import SessionLocal, engine

    Base.metadata.create_all(bind=engine)
    db = SessionLocal()

    # Two cameras in different zones — the case the global broke.
    cam_a = Camera(camera_id="MC-CAM-A", name="North Gate", zone="North",
                   gps_lat=23.03, gps_lon=72.58, status="ONLINE")
    cam_b = Camera(camera_id="MC-CAM-B", name="South Gate", zone="South",
                   gps_lat=23.01, gps_lon=72.60, status="ONLINE")
    db.add_all([cam_a, cam_b])
    db.commit()
    db.refresh(cam_a)
    db.refresh(cam_b)

    # ── PART 1: multi-camera identity ───────────────────────────────────────
    import backend.connection_manager as cm_module
    from backend.connection_manager import ConnectionManager

    has_global = hasattr(cm_module, "_reid_camera_db_id")
    check("The single-camera global is gone from connection_manager",
          not has_global,
          "module still exposes _reid_camera_db_id" if has_global else "")

    cm_module.set_reid_ready()
    cm_module.set_behavior_ready()
    cm_module.set_abandoned_ready()

    mgr_a = ConnectionManager(camera_id="MC-CAM-A", track_expiry_seconds=5.0,
                              camera_db_id=cam_a.id)
    mgr_b = ConnectionManager(camera_id="MC-CAM-B", track_expiry_seconds=5.0,
                              camera_db_id=cam_b.id)

    check("Each ConnectionManager carries its OWN cameras.id as instance state",
          mgr_a._camera_db_id == cam_a.id and mgr_b._camera_db_id == cam_b.id
          and cam_a.id != cam_b.id,
          f"A={mgr_a._camera_db_id} B={mgr_b._camera_db_id}")

    # Constructing B must not retroactively change A — the exact failure the
    # module global produced (last writer won, for the whole process).
    check("Creating a second manager does NOT overwrite the first one's camera "
          "(the precise bug the module-level global caused)",
          mgr_a._camera_db_id == cam_a.id, str(mgr_a._camera_db_id))

    # Drive one real detection event through each manager concurrently and
    # confirm the tracks rows are attributed to the right camera.
    def make_event(cam_str: str, track_id: int) -> dict:
        return {
            "event_type": "person_track", "camera_id": cam_str,
            "track_id": track_id, "frame_number": 1, "timestamp": 1.0,
            "bbox": [10.0, 10.0, 60.0, 120.0], "confidence": 0.9,
        }

    ev_a = make_event("MC-CAM-A", 8001)
    ev_b = make_event("MC-CAM-B", 8002)
    await asyncio.gather(
        mgr_a.handle_detection_event(ev_a),
        mgr_b.handle_detection_event(ev_b),
    )

    rows = db.execute(text(
        "SELECT camera_id, track_id FROM tracks WHERE track_id IN (8001, 8002)"
    )).mappings().fetchall()
    by_track = {r["track_id"]: r["camera_id"] for r in rows}
    check("Concurrent events from two cameras create correctly-attributed "
          "tracks rows",
          by_track.get(8001) == "MC-CAM-A" and by_track.get(8002) == "MC-CAM-B",
          str(by_track))

    # ── Alerts fired via each manager's camera id land on the right camera ──
    from backend.services.loitering_detector import get_loitering_detector
    from backend.services.state import get_behavior_state
    from tests.in_memory_state import install as install_fake_state

    install_fake_state()
    check("Behavior state backend available for the loitering path",
          await get_behavior_state().ping())

    loiter = get_loitering_detector()
    for mgr, cam, tid in ((mgr_a, cam_a, 9101), (mgr_b, cam_b, 9102)):
        for t in (0, 20, 40, 60):
            await loiter.on_track_position(
                camera_str_id=mgr._camera_id, camera_db_id=mgr._camera_db_id,
                track_id=tid, cx=100, cy=100, video_time=float(t),
            )

    alerts = db.execute(text("""
        SELECT a.camera_id AS cam, a.subject_label AS label
        FROM   alerts a WHERE a.alert_type = 'LOITERING'
    """)).mappings().fetchall()
    cams_used = {r["cam"] for r in alerts}
    check("Loitering alerts from two cameras are stamped with DIFFERENT "
          "cameras.id values — under the old global both would carry the same",
          cams_used == {cam_a.id, cam_b.id}, str(sorted(c for c in cams_used if c)))

    # ── PART 2: face-match dispatch interlock ───────────────────────────────
    from backend.core.config import settings
    from backend.services.threshold_validation import face_threshold_validation_state

    state = face_threshold_validation_state()
    check("Face threshold is correctly reported as NOT validated "
          "(the only report present is synthetic)",
          state.validated is False, state.reason[:120])
    check("The unvalidated reason names the synthetic report rather than just "
          "saying 'missing'",
          "synthetic" in state.reason.lower() or "degenerate" in state.reason.lower()
          or "never been run" in state.reason.lower(), state.reason[:120])
    check("Interlock is ON by default (safe default, not opt-in)",
          settings.REQUIRE_VALIDATED_FACE_THRESHOLD_FOR_DISPATCH is True)

    # Seed officers so a dispatch WOULD succeed if it were attempted — this
    # is what makes the next assertion meaningful rather than vacuous.
    from backend.routing.seed import seed_routing_demo_data
    seed_routing_demo_data()
    avail = db.execute(text(
        "SELECT COUNT(*) FROM officers WHERE status='AVAILABLE'"
    )).scalar()
    check("Officers ARE available — so a withheld dispatch is the interlock "
          "acting, not an empty roster",
          avail > 0, f"available={avail}")

    from backend.services.evidence_capture import on_alert_fired

    face_alert = Alert(alert_type="WATCHLIST_FACE_MATCH", camera_id=cam_a.id,
                       subject_label="unvalidated-match", status="new",
                       lifecycle_status="OPEN", iq_contribution=7.0,
                       created_at=datetime.utcnow())
    db.add(face_alert)
    db.flush()
    face_id = face_alert.id
    await on_alert_fired(face_alert, db, "MC-CAM-A")
    db.commit()

    routed = db.execute(text(
        "SELECT * FROM routed_alerts WHERE alert_id = :a"
    ), {"a": face_id}).mappings().fetchone()
    check("INTERLOCK: an unvalidated face match does NOT auto-dispatch an "
          "officer to a physical location",
          routed is None, f"routed_alerts row exists: {dict(routed) if routed else ''}")

    db.refresh(face_alert)
    check("…but the alert STILL fires and is still classified CRITICAL "
          "(the interlock withholds dispatch, it does not suppress detection)",
          face_alert.severity == "critical", str(face_alert.severity))
    check("…and no officer was made BUSY by it",
          db.execute(text(
              "SELECT COUNT(*) FROM officers WHERE status='BUSY'"
          )).scalar() == 0)

    # A non-face CRITICAL alert must still dispatch normally — the interlock
    # is scoped to face matching, not a blanket dispatch kill-switch.
    other = Alert(alert_type="ABANDONED_OBJECT", camera_id=cam_a.id,
                  subject_label="still dispatches", status="new",
                  lifecycle_status="OPEN", iq_contribution=7.5,
                  created_at=datetime.utcnow())
    db.add(other)
    db.flush()
    other_id = other.id
    await on_alert_fired(other, db, "MC-CAM-A")
    db.commit()

    routed_other = db.execute(text(
        "SELECT * FROM routed_alerts WHERE alert_id = :a"
    ), {"a": other_id}).mappings().fetchone()
    check("A non-face CRITICAL alert still dispatches normally — the interlock "
          "is scoped to face matching, not a blanket kill-switch",
          routed_other is not None and routed_other["status"] == "ROUTED",
          str(routed_other["status"]) if routed_other else "no row")

    # ── The interlock must be SATISFIABLE, not permanently on ───────────────
    # An interlock that can never be cleared by doing the right work is just
    # a disabled feature with extra steps. Write a report with no synthetic
    # markers — the shape a real validation run produces — and confirm
    # dispatch resumes. The report is removed afterwards so this test cannot
    # leave behind a file that fakes validation for every later run.
    reports_dir = _ROOT / "reports"
    reports_dir.mkdir(exist_ok=True)
    fake_real = reports_dir / "face_watchlist_validation_99999999_999999.md"
    fake_real.write_text(
        "# Sentinel Gujarat - Face Watchlist Match Validation Report\n\n"
        "Labeled dataset.\n\n"
        "| **False Positive Rate** (FP / (FP+TN)) | **0.42%** |\n",
        encoding="utf-8",
    )
    try:
        from backend.services.threshold_validation import face_threshold_validation_state as _st
        st2 = _st()
        check("Interlock CLEARS when a non-synthetic validation report exists "
              "(it is satisfiable by real work, not a permanent off-switch)",
              st2.validated is True, st2.reason[:110])
        check("…and the measured false-positive rate is parsed out of the report",
              st2.fpr_at_threshold is not None
              and abs(st2.fpr_at_threshold - 0.0042) < 1e-6,
              str(st2.fpr_at_threshold))

        face2 = Alert(alert_type="WATCHLIST_FACE_MATCH", camera_id=cam_a.id,
                      subject_label="validated-match", status="new",
                      lifecycle_status="OPEN", iq_contribution=7.0,
                      created_at=datetime.utcnow())
        db.add(face2)
        db.flush()
        face2_id = face2.id
        await on_alert_fired(face2, db, "MC-CAM-A")
        db.commit()
        routed2 = db.execute(text(
            "SELECT * FROM routed_alerts WHERE alert_id = :a"
        ), {"a": face2_id}).mappings().fetchone()
        check("With the threshold validated, a face match DOES auto-dispatch",
              routed2 is not None and routed2["status"] == "ROUTED",
              str(routed2["status"]) if routed2 else "no row")
    finally:
        fake_real.unlink(missing_ok=True)

    st3 = face_threshold_validation_state()
    check("Removing the report restores the unvalidated state — validation is "
          "read from disk each time, never cached into a permanent yes",
          st3.validated is False, st3.reason[:90])

    db.close()

    # ── Readiness endpoint states it plainly ────────────────────────────────
    await _check_readiness_endpoint()

    print(f"\n{'=' * 70}\n{len(PASS)} passed, {len(FAIL)} failed\n{'=' * 70}")
    if FAIL:
        print("FAILED:", FAIL)
        sys.exit(1)


async def _check_readiness_endpoint() -> None:
    from starlette.testclient import TestClient

    from backend.auth.routes import pwd_context
    from backend.db.models import User
    from backend.db.session import SessionLocal

    db = SessionLocal()
    db.add(User(username="mc_admin", hashed_password=pwd_context.hash("pw"),
                role="admin", is_active=True))
    db.commit()
    db.close()

    import backend.main as backend_main

    with TestClient(backend_main.app) as client:
        tok = client.post("/api/v1/auth/login",
                          json={"username": "mc_admin", "password": "pw"}).json()["access_token"]
        h = {"Authorization": f"Bearer {tok}"}
        r = client.get("/api/v1/routing/dispatch-readiness", headers=h)
        check("GET /routing/dispatch-readiness returns 200",
              r.status_code == 200, str(r.status_code))
        body = r.json()
        check("Readiness endpoint reports face auto-dispatch as DISABLED",
              body.get("face_match_auto_dispatch_enabled") is False, str(body)[:160])
        check("…and explains why, so an operator can tell this apart from "
              "'no officer was available'",
              bool(body.get("note")) and bool(body["validation"]["reason"]),
              str(body.get("note"))[:120])


if __name__ == "__main__":
    asyncio.run(main())
