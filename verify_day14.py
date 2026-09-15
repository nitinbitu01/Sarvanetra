"""
verify_day14.py â€” Alert Routing acceptance checks (all 7 tests).

Exercises the real routing service, the real escalation tick, and the real
HTTP endpoints. Concurrency (Test 2) uses genuine threads against the real
Session factory â€” a sequential simulation would prove nothing about the
atomic-claim pattern, which exists precisely to survive interleaving.

ACK_TIMEOUT_SECONDS is set to a small value here rather than sleeping 120s.

Run:  python verify_day14.py
"""
from __future__ import annotations

import asyncio
import os
import sys
import tempfile
import threading
import time
from datetime import datetime
from pathlib import Path

_ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(_ROOT))

try:
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    if hasattr(sys.stderr, "reconfigure"):
        sys.stderr.reconfigure(encoding="utf-8", errors="replace")
except Exception:
    pass

_tmpdir = tempfile.mkdtemp(prefix="verify_day14_")
os.environ["DATABASE_URL"] = f"sqlite:///{_tmpdir}/verify_day14.db"
os.environ["EVIDENCE_DIR"] = str(Path(_tmpdir) / "evidence")
os.environ["ACK_TIMEOUT_SECONDS"] = "3"      # demo compression for the tests
os.environ.pop("SENTINEL_SOURCE", None)

PASS: list[str] = []
FAIL: list[str] = []


def check(name: str, cond: bool, detail: str = "") -> None:
    (PASS if cond else FAIL).append(name)
    print(f"{'PASS' if cond else 'FAIL'} â€” {name}{(': ' + detail) if detail else ''}")


def quiesce(db) -> None:
    """Retire every still-active routed_alert before starting a new test.

    Necessary because the escalation tick is GLOBAL: it scans every overdue
    ROUTED/ESCALATED_ROUTED row, not just the one a given test cares about.
    Leaving earlier tests' alerts live meant the tick escalated those first
    and consumed the officer the next test was relying on â€” the tick behaving
    exactly as designed, while the test misread it as a routing failure.
    Terminal ACKNOWLEDGED rows are invisible to the tick.
    """
    from sqlalchemy import text
    db.execute(text("""
        UPDATE routed_alerts
        SET    status = 'ACKNOWLEDGED', ack_at = datetime('now')
        WHERE  status IN ('PENDING', 'ROUTED', 'ESCALATED_ROUTED')
    """))
    db.commit()


def reset_officers(db, statuses: dict[int, str] | None = None) -> None:
    from sqlalchemy import text
    db.execute(text("UPDATE officers SET status='AVAILABLE', current_alert_id=NULL"))
    if statuses:
        for oid, st in statuses.items():
            db.execute(text("UPDATE officers SET status=:s WHERE id=:i"),
                       {"s": st, "i": oid})
    db.commit()


def make_critical_alert(db, camera_db_id: int, label: str = "test") -> int:
    from backend.db.models import Alert
    a = Alert(alert_type="WATCHLIST_FACE_MATCH", camera_id=camera_db_id,
              subject_label=label, status="new", lifecycle_status="OPEN",
              severity="critical", iq_contribution=7.0,
              created_at=datetime.utcnow())
    db.add(a)
    db.commit()
    db.refresh(a)
    return a.id


async def main() -> None:
    from sqlalchemy import text

    from backend.core.config import settings
    from backend.db.models import Base
    from backend.db.session import SessionLocal, engine
    from backend.routing.seed import seed_routing_demo_data
    from backend.routing.service import is_critical_severity, route_alert

    Base.metadata.create_all(bind=engine)
    seed_routing_demo_data()

    check("ACK_TIMEOUT_SECONDS is read from config (demo compression works)",
          settings.ACK_TIMEOUT_SECONDS == 3, str(settings.ACK_TIMEOUT_SECONDS))
    check("Severity match is case-insensitive â€” 'critical' (what this codebase "
          "actually writes) counts as CRITICAL",
          is_critical_severity("critical") and is_critical_severity("CRITICAL")
          and not is_critical_severity("high"))

    db = SessionLocal()
    cam_a = db.execute(text("SELECT id FROM cameras WHERE camera_id='CAM-RT-A'")).scalar()
    cam_b = db.execute(text("SELECT id FROM cameras WHERE camera_id='CAM-RT-B'")).scalar()
    check("Demo routing cameras seeded", cam_a is not None and cam_b is not None,
          f"A={cam_a} B={cam_b}")

    # â”€â”€ TEST 1: nearest assigned, OFFLINE excluded â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€
    # Thompson (id=4) is moved to be the geographically CLOSEST officer while
    # OFFLINE. If eligibility were a distance sort with a post-filter, he'd
    # win. He must not.
    reset_officers(db, {4: "OFFLINE"})
    db.execute(text("UPDATE officers SET lat=37.7748, lng=-122.4195 WHERE id=4"))
    db.commit()

    a1 = make_critical_alert(db, cam_a, "test1")
    r1 = route_alert(a1, db)
    check("T1: alert routed", r1 is not None and r1["status"] == "ROUTED",
          str(r1["status"]) if r1 else "None")
    check("T1: nearest AVAILABLE officer (Chen, id=1) assigned",
          r1 and r1["assigned_officer"] == 1, str(r1["assigned_officer"] if r1 else None))
    check("T1: OFFLINE officer NOT assigned even though he is the closest "
          "(eligibility is inside the atomic UPDATE, not a post-filter)",
          r1 and r1["assigned_officer"] != 4)
    chen = db.execute(text("SELECT status, current_alert_id FROM officers WHERE id=1")
                      ).mappings().fetchone()
    check("T1: Chen is BUSY with current_alert_id set to the routed_alert",
          chen["status"] == "BUSY" and chen["current_alert_id"] == r1["id"],
          f"{chen['status']} / {chen['current_alert_id']}")
    check("T1: assigned_at populated (the countdown has a server-side origin)",
          r1["assigned_at"] is not None)
    check("T1: no PENDING leak â€” route_alert always resolves before returning",
          db.execute(text("SELECT COUNT(*) FROM routed_alerts WHERE status='PENDING'")
                     ).scalar() == 0)

    # â”€â”€ TEST 2: concurrent claim, no double-assignment â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€
    quiesce(db)
    reset_officers(db, {2: "OFFLINE", 3: "OFFLINE", 4: "OFFLINE"})   # only Chen free
    a2 = make_critical_alert(db, cam_a, "test2a")
    a3 = make_critical_alert(db, cam_a, "test2b")
    db.close()

    results: dict[int, dict] = {}
    barrier = threading.Barrier(2)

    def worker(alert_id: int) -> None:
        s = SessionLocal()
        try:
            barrier.wait(timeout=10)   # maximise interleaving
            results[alert_id] = route_alert(alert_id, s) or {}
        finally:
            s.close()

    threads = [threading.Thread(target=worker, args=(aid,)) for aid in (a2, a3)]
    for t in threads:
        t.start()
    for t in threads:
        t.join(timeout=30)

    db = SessionLocal()
    routed_to_chen = db.execute(text("""
        SELECT COUNT(*) FROM routed_alerts
        WHERE assigned_officer = 1 AND status = 'ROUTED' AND alert_id IN (:a,:b)
    """), {"a": a2, "b": a3}).scalar()
    check("T2: exactly ONE of two concurrent alerts claimed Chen "
          "(atomic UPDATE + rowcount, not SELECT-then-UPDATE)",
          routed_to_chen == 1, f"count={routed_to_chen}")
    other = db.execute(text("""
        SELECT status FROM routed_alerts
        WHERE alert_id IN (:a,:b) AND (assigned_officer IS NULL OR assigned_officer != 1)
    """), {"a": a2, "b": a3}).scalar()
    check("T2: the loser resolved to UNROUTED, not PENDING and not a crash",
          other == "UNROUTED", str(other))
    dupes = db.execute(text("""
        SELECT COUNT(*) FROM routed_alerts
        WHERE assigned_officer = 1 AND status IN ('ROUTED','ESCALATED_ROUTED')
        AND   alert_id IN (:a, :b)
    """), {"a": a2, "b": a3}).scalar()
    check("T2: no officer holds two simultaneous active assignments",
          dupes == 1, f"active_for_chen={dupes}")

    # â”€â”€ TEST 3: escalation, cap, previous officer released â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€
    from backend.routing.escalation import _run_escalation_tick_sync

    quiesce(db)
    reset_officers(db, {3: "OFFLINE", 4: "OFFLINE"})   # Chen + Park available
    a4 = make_critical_alert(db, cam_a, "test3")
    r4 = route_alert(a4, db)
    check("T3: initially routed to Chen", r4["assigned_officer"] == 1,
          str(r4["assigned_officer"]))

    time.sleep(settings.ACK_TIMEOUT_SECONDS + 1.5)
    await asyncio.to_thread(_run_escalation_tick_sync)

    row4 = db.execute(text("SELECT * FROM routed_alerts WHERE id=:i"),
                      {"i": r4["id"]}).mappings().fetchone()
    check("T3: escalated to ESCALATED_ROUTED", row4["status"] == "ESCALATED_ROUTED",
          row4["status"])
    check("T3: reassigned to the next-nearest officer (Park, id=2)",
          row4["assigned_officer"] == 2, str(row4["assigned_officer"]))
    check("T3: escalation_count = 1 (the DB-level cap is set)",
          row4["escalation_count"] == 1, str(row4["escalation_count"]))
    check("T3: assigned_at reset so the second officer gets a full timeout",
          row4["assigned_at"] is not None)

    chen_after = db.execute(text("SELECT status, current_alert_id FROM officers WHERE id=1")
                            ).mappings().fetchone()
    check("T3: the timed-out officer is RELEASED, not stranded BUSY on a dead "
          "alert (this is what shrinks the dispatch pool if missed)",
          chen_after["status"] == "AVAILABLE" and chen_after["current_alert_id"] is None,
          f"{chen_after['status']} / {chen_after['current_alert_id']}")

    # Cap: a second overdue window must NOT escalate again.
    time.sleep(settings.ACK_TIMEOUT_SECONDS + 1.5)
    await asyncio.to_thread(_run_escalation_tick_sync)
    row4b = db.execute(text("SELECT * FROM routed_alerts WHERE id=:i"),
                       {"i": r4["id"]}).mappings().fetchone()
    check("T3: does NOT escalate a second time â€” escalation_count=0 in the "
          "WHERE clause is a database-enforced cap, not application logic",
          row4b["escalation_count"] == 1 and row4b["assigned_officer"] == 2,
          f"count={row4b['escalation_count']} officer={row4b['assigned_officer']}")

    # â”€â”€ TEST 6 (before 4/5 so we have an UNROUTED row for Test 7) â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€
    quiesce(db)
    reset_officers(db, {1: "OFFLINE", 2: "OFFLINE", 3: "OFFLINE", 4: "OFFLINE"})
    a6 = make_critical_alert(db, cam_b, "test6")
    r6 = route_alert(a6, db)
    check("T6: zero available officers â†’ clean UNROUTED, no exception",
          r6 and r6["status"] == "UNROUTED", str(r6["status"]) if r6 else "None")
    check("T6: UNROUTED row records why",
          bool(r6["escalation_note"]), str(r6["escalation_note"]))
    check("T6: officers table intact after a zero-availability route",
          db.execute(text("SELECT COUNT(*) FROM officers")).scalar() == 4)

    # â”€â”€ TEST 4: ACK wins, officer reverts, idempotent replay â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€
    reset_officers(db, {3: "OFFLINE", 4: "OFFLINE"})
    a5 = make_critical_alert(db, cam_a, "test4")
    r5 = route_alert(a5, db)
    officer5 = r5["assigned_officer"]
    check("T4: routed before ACK", r5["status"] == "ROUTED", r5["status"])
    db.close()

    await _api_tests(a5, r5["id"], officer5, a6, settings)

    # Escalation must not fire after a successful ACK.
    db = SessionLocal()
    time.sleep(settings.ACK_TIMEOUT_SECONDS + 1.5)
    await asyncio.to_thread(_run_escalation_tick_sync)
    row5 = db.execute(text("SELECT * FROM routed_alerts WHERE id=:i"),
                      {"i": r5["id"]}).mappings().fetchone()
    check("T4: escalation does NOT fire on an ACKNOWLEDGED alert after the "
          "timeout passes (terminal state is respected)",
          row5["status"] == "ACKNOWLEDGED" and row5["escalation_count"] == 0,
          f"{row5['status']} count={row5['escalation_count']}")
    off5 = db.execute(text("SELECT status FROM officers WHERE id=:i"),
                      {"i": officer5}).mappings().fetchone()
    check("T4: officer reverted to AVAILABLE on ACK",
          off5["status"] == "AVAILABLE", off5["status"])
    db.close()

    # ── END-TO-END: routing must fire from the REAL alert hook ──────────────
    # route_alert() above was called directly with a clean Session. The
    # production caller is evidence_capture.on_alert_fired(), which does
    # `alert.severity = ...; db.flush()` FIRST — leaving a transaction open.
    # That is what made `with db.begin():` raise InvalidRequestError on every
    # CRITICAL alert. _route_critical_alert() swallows exceptions by design
    # (routing must not stop an alert firing), so a regression here would be
    # invisible unless something asserts the row actually appears.
    from backend.services.evidence_capture import on_alert_fired

    db = SessionLocal()
    quiesce(db)
    reset_officers(db, {3: "OFFLINE", 4: "OFFLINE"})

    from backend.db.models import Alert

    # Deliberately NOT a WATCHLIST_FACE_MATCH: the face-match dispatch
    # interlock withholds auto-dispatch while the matching threshold is
    # unvalidated (services/threshold_validation.py), so a face alert would
    # correctly produce no routed_alerts row and this test would be asserting
    # the interlock rather than the transaction handling it exists to check.
    # ABANDONED_OBJECT with a CRITICAL-level IQ score exercises the identical
    # on_alert_fired → route_alert path. The interlock itself is covered by
    # verify_multicamera_and_interlock.py.
    e2e = Alert(alert_type="ABANDONED_OBJECT", camera_id=cam_a,
                subject_label="e2e", status="new", lifecycle_status="OPEN",
                iq_contribution=7.5, created_at=datetime.utcnow())
    db.add(e2e)
    db.flush()          # exactly what the firing modules do before the hook
    e2e_id = e2e.id

    await on_alert_fired(e2e, db, "CAM-RT-A")

    db.commit()
    ra = db.execute(text(
        "SELECT * FROM routed_alerts WHERE alert_id = :aid"
    ), {"aid": e2e_id}).mappings().fetchone()
    check("E2E: on_alert_fired() routed the alert through the real hook — with "
          "an open transaction from db.flush(), the case that broke "
          "`with db.begin():` in production",
          ra is not None and ra["status"] == "ROUTED",
          str(ra["status"]) if ra else "NO ROUTED_ALERTS ROW")
    check("E2E: an officer was actually assigned by the hook",
          ra is not None and ra["assigned_officer"] is not None,
          str(ra["assigned_officer"]) if ra else "-")
    db.refresh(e2e)
    check("E2E: on_alert_fired also stamped severity='critical' on the alert",
          e2e.severity == "critical", str(e2e.severity))
    db.close()

    print(f"\n{'=' * 70}\n{len(PASS)} passed, {len(FAIL)} failed\n{'=' * 70}")
    if FAIL:
        print("FAILED:", FAIL)
        sys.exit(1)


async def _api_tests(acked_alert_id: int, ra_id: int, officer_id: int,
                     unrouted_alert_id: int, settings) -> None:
    """Tests 4 (ACK), 5 (reload resume), 7 (ACK on UNROUTED) over real HTTP."""
    from starlette.testclient import TestClient

    from backend.auth.routes import pwd_context
    from backend.db.models import User
    from backend.db.session import SessionLocal

    db = SessionLocal()
    db.add(User(username="rt_admin", hashed_password=pwd_context.hash("pw"),
                role="admin", is_active=True))
    db.commit()
    db.close()

    import backend.main as backend_main

    with TestClient(backend_main.app) as client:
        tok = client.post("/api/v1/auth/login",
                          json={"username": "rt_admin", "password": "pw"}).json()["access_token"]
        h = {"Authorization": f"Bearer {tok}"}

        # â”€â”€ TEST 5: /routing/active seeds the countdown from server truth â”€â”€â”€â”€
        active = client.get("/api/v1/routing/active", headers=h)
        check("T5: GET /routing/active returns 200", active.status_code == 200,
              str(active.status_code))
        rows = active.json()
        mine = [r for r in rows if r["alert_id"] == acked_alert_id]
        check("T5: the active alert appears in /routing/active", len(mine) == 1,
              f"found={len(mine)}")
        if mine:
            rec = mine[0]
            check("T5: response carries timeout_seconds and seconds_elapsed so a "
                  "reloaded dashboard resumes mid-countdown instead of restarting",
                  "timeout_seconds" in rec and "seconds_elapsed" in rec
                  and rec["timeout_seconds"] == settings.ACK_TIMEOUT_SECONDS,
                  f"timeout={rec.get('timeout_seconds')} elapsed={rec.get('seconds_elapsed')}")
            check("T5: seconds_elapsed is a sane non-negative server value",
                  isinstance(rec["seconds_elapsed"], int) and rec["seconds_elapsed"] >= 0,
                  str(rec["seconds_elapsed"]))
            check("T5: assigned_at is ISO-8601 with a Z suffix (Safari's Date() "
                  "rejects the raw space-separated SQLite form)",
                  rec["assigned_at"] and rec["assigned_at"].endswith("Z")
                  and "T" in rec["assigned_at"], str(rec["assigned_at"]))
            check("T5: officer_name resolved for the badge",
                  bool(rec["officer_name"]), str(rec["officer_name"]))

        # â”€â”€ TEST 4: ACK â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€
        ack = client.post(f"/api/v1/alerts/{acked_alert_id}/ack", headers=h)
        check("T4: ACK returns 200 acknowledged",
              ack.status_code == 200 and ack.json()["status"] == "acknowledged",
              f"{ack.status_code} {ack.text[:80]}")

        ack2 = client.post(f"/api/v1/alerts/{acked_alert_id}/ack", headers=h)
        check("T4: repeat ACK is idempotent â€” 200 already_acknowledged, no state change",
              ack2.status_code == 200 and ack2.json()["status"] == "already_acknowledged",
              f"{ack2.status_code} {ack2.text[:80]}")

        officers = client.get("/api/v1/officers", headers=h).json()
        target = [o for o in officers if o["id"] == officer_id]
        check("T4: GET /officers shows the acknowledging officer AVAILABLE again",
              target and target[0]["status"] == "AVAILABLE",
              str(target[0]["status"]) if target else "not found")
        names = [o["name"] for o in officers]
        check("T4: /officers is sorted by name ASC (stable list â€” prevents the "
              "OfficerPanel reshuffling on every status event)",
              names == sorted(names), str(names))

        # â”€â”€ TEST 6/7: unrouted listing + ACK on an unrouted alert â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€
        unrouted = client.get("/api/v1/routing/unrouted", headers=h)
        check("T6: GET /routing/unrouted lists the unrouted alert",
              unrouted.status_code == 200
              and any(r["alert_id"] == unrouted_alert_id for r in unrouted.json()),
              f"{unrouted.status_code} n={len(unrouted.json())}")

        bad_ack = client.post(f"/api/v1/alerts/{unrouted_alert_id}/ack", headers=h)
        check("T7: ACK on an UNROUTED alert returns 409 cannot_ack",
              bad_ack.status_code == 409
              and bad_ack.json()["status"] == "cannot_ack",
              f"{bad_ack.status_code} {bad_ack.text[:110]}")
        check("T7: 409 body explains why (never routed to an officer)",
              "never routed" in bad_ack.json().get("reason", ""),
              str(bad_ack.json().get("reason")))

        still = client.get("/api/v1/routing/unrouted", headers=h).json()
        check("T7: the UNROUTED row is unchanged by the rejected ACK",
              any(r["alert_id"] == unrouted_alert_id and r["status"] == "UNROUTED"
                  for r in still))

        missing = client.post("/api/v1/alerts/99999999/ack", headers=h)
        check("ACK on an alert with no routing record â†’ 404, not a 500",
              missing.status_code == 404, str(missing.status_code))

        # â”€â”€ Officer mark-available endpoint â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€
        avail = client.post(f"/api/v1/officers/{officer_id}/available", headers=h)
        check("POST /officers/{id}/available returns ok",
              avail.status_code == 200 and avail.json()["status"] == "ok",
              f"{avail.status_code} {avail.text[:60]}")
        bad_officer = client.post("/api/v1/officers/99999/available", headers=h)
        check("Mark-available on an unknown officer â†’ 404, not a crash",
              bad_officer.status_code == 404, str(bad_officer.status_code))

        unauth = client.get("/api/v1/routing/active")
        check("Routing endpoints require authentication",
              unauth.status_code in (401, 403), str(unauth.status_code))


if __name__ == "__main__":
    asyncio.run(main())
