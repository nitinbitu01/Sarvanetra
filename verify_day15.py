"""
verify_day15.py — Push notification backend acceptance checks.

SCOPE — READ THIS BEFORE TRUSTING A PASS
  Tests 2, 3, 4, 6 and 10 of the Day 15 checkpoint require a physical phone,
  an HTTPS tunnel, and a real push service (FCM/Mozilla). None of that can be
  driven from here, and this file does NOT claim to cover them. What it does
  cover is everything on the server side of the relay:

    - subscriptions are officer-scoped and reject anonymous callers
    - the upsert makes re-subscribe idempotent instead of duplicating
    - fan-out targets the ASSIGNED officer, not everyone
    - every attempt writes a push_delivery_log row
    - rate limiting blocks the 3rd send in a window and logs it
    - a 410 from the push service deletes the subscription
    - the cleanup job removes stale rows
    - /alerts/{id}/detail returns one aggregated payload

  Actual delivery to a handset is verified by the manual checklist in
  BACKLOG.md, not by this script. A green run here means "the server did the
  right thing", not "the phone buzzed".

  webpush() is monkeypatched throughout: the real call would try to reach
  fcm.googleapis.com with a fabricated endpoint. The patch is on the SEND
  function only — targeting, logging, rate limiting and cleanup are all real.

Run:  python verify_day15.py
"""
from __future__ import annotations

import asyncio
import json
import os
import sys
import tempfile
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

_tmpdir = tempfile.mkdtemp(prefix="verify_day15_")
os.environ["DATABASE_URL"] = f"sqlite:///{_tmpdir}/verify_day15.db"
os.environ["EVIDENCE_DIR"] = str(Path(_tmpdir) / "evidence")
os.environ.pop("SENTINEL_SOURCE", None)

PASS: list[str] = []
FAIL: list[str] = []


def check(name: str, cond: bool, detail: str = "") -> None:
    (PASS if cond else FAIL).append(name)
    print(f"{'PASS' if cond else 'FAIL'} — {name}{(': ' + detail) if detail else ''}")


class FakePushResponse:
    def __init__(self, status_code: int):
        self.status_code = status_code


class FakeWebPushException(Exception):
    def __init__(self, message: str, status_code: int | None = None):
        super().__init__(message)
        self.response = FakePushResponse(status_code) if status_code else None


def main() -> None:
    from sqlalchemy import text

    from backend.core.config import settings
    from backend.db.models import Alert, Base, Camera, User
    from backend.db.session import SessionLocal, engine
    from backend.push import service as push_service

    Base.metadata.create_all(bind=engine)
    db = SessionLocal()

    # ── Fixtures ────────────────────────────────────────────────────────────
    cam = Camera(camera_id="PUSH-CAM", name="Gate Cam", zone="North",
                 gps_lat=23.03, gps_lon=72.58, status="ONLINE")
    db.add(cam)
    db.commit()
    db.refresh(cam)

    db.execute(text("""
        INSERT INTO officers (id, name, lat, lng, status, last_updated)
        VALUES (1, 'Officer Alpha', 23.03, 72.58, 'AVAILABLE', datetime('now')),
               (2, 'Officer Bravo', 23.05, 72.60, 'AVAILABLE', datetime('now'))
    """))
    db.commit()

    from backend.auth.routes import pwd_context
    db.add_all([
        User(username="Officer Alpha", hashed_password=pwd_context.hash("pw"),
             role="officer", is_active=True),
        User(username="admin_nobody", hashed_password=pwd_context.hash("pw"),
             role="admin", is_active=True),
    ])
    db.commit()

    def new_alert(label: str) -> Alert:
        a = Alert(alert_type="ABANDONED_OBJECT", camera_id=cam.id,
                  subject_label=label, status="new", lifecycle_status="OPEN",
                  severity="critical", iq_contribution=7.5,
                  created_at=datetime.utcnow())
        db.add(a)
        db.commit()
        db.refresh(a)
        return a

    def add_sub(officer_id: int, endpoint: str) -> int:
        db.execute(text("""
            INSERT INTO push_subscriptions
                (officer_id, endpoint, p256dh, auth, device_label,
                 created_at, last_seen_at)
            VALUES (:o, :e, 'fake-p256dh', 'fake-auth', 'test device',
                    datetime('now'), datetime('now'))
        """), {"o": officer_id, "e": endpoint})
        db.commit()
        return db.execute(text(
            "SELECT id FROM push_subscriptions WHERE endpoint = :e"
        ), {"e": endpoint}).scalar()

    # ── VAPID config sanity ─────────────────────────────────────────────────
    check("VAPID public key is a P-256 uncompressed point (87 base64url chars)",
          len(settings.VAPID_PUBLIC_KEY) == 87,
          f"len={len(settings.VAPID_PUBLIC_KEY)}")
    check("VAPID contact email configured (push services 400 without it, with "
          "an error that points at signing instead)",
          "@" in settings.VAPID_CONTACT_EMAIL, settings.VAPID_CONTACT_EMAIL)

    # ── TEST 9: rate limiting ───────────────────────────────────────────────
    sub_a = add_sub(1, "https://fcm.example/ep-alpha")

    sent_calls: list[dict] = []

    def fake_ok(**kwargs):
        sent_calls.append(kwargs)
        return FakePushResponse(201)

    push_service.reset_rate_limits()
    original_limit = settings.PUSH_RATE_LIMIT_PER_SUB_PER_MINUTE
    settings.PUSH_RATE_LIMIT_PER_SUB_PER_MINUTE = 2

    import pywebpush
    original_webpush = pywebpush.webpush
    pywebpush.webpush = fake_ok
    try:
        tallies = []
        for i in range(3):
            a = new_alert(f"rate-{i}")
            tallies.append(push_service.send_critical_push(
                a, {"assigned_officer": 1}, db))
    finally:
        pywebpush.webpush = original_webpush
        settings.PUSH_RATE_LIMIT_PER_SUB_PER_MINUTE = original_limit

    check("TEST 9: with the limit at 2/min, 3 alerts produce 2 sends",
          sum(t["sent"] for t in tallies) == 2,
          str([t["sent"] for t in tallies]))
    check("TEST 9: the 3rd is rate_limited, not silently dropped",
          sum(t["rate_limited"] for t in tallies) == 1,
          str([t["rate_limited"] for t in tallies]))

    log_rows = db.execute(text(
        "SELECT status, COUNT(*) c FROM push_delivery_log GROUP BY status"
    )).mappings().fetchall()
    counts = {r["status"]: r["c"] for r in log_rows}
    check("TEST 9: delivery log records 2 sent + 1 rate_limited — a blocked "
          "push is an audit event, not a no-op",
          counts.get("sent") == 2 and counts.get("rate_limited") == 1, str(counts))

    # ── Officer targeting ───────────────────────────────────────────────────
    sub_b = add_sub(2, "https://fcm.example/ep-bravo")
    push_service.reset_rate_limits()
    sent_calls.clear()

    pywebpush.webpush = fake_ok
    try:
        a = new_alert("targeting")
        tally = push_service.send_critical_push(a, {"assigned_officer": 1}, db)
    finally:
        pywebpush.webpush = original_webpush

    endpoints = [c["subscription_info"]["endpoint"] for c in sent_calls]
    check("Fan-out targets ONLY the assigned officer's device — broadcasting "
          "would leak a camera location to officers with no role in the incident",
          endpoints == ["https://fcm.example/ep-alpha"], str(endpoints))
    check("Tally reports one target", tally["targets"] == 1, str(tally))

    # Unassigned → all AVAILABLE officers, so somebody knows.
    push_service.reset_rate_limits()
    sent_calls.clear()
    pywebpush.webpush = fake_ok
    try:
        a = new_alert("unassigned")
        tally2 = push_service.send_critical_push(a, None, db)
    finally:
        pywebpush.webpush = original_webpush

    check("An UNASSIGNED critical alert notifies all AVAILABLE officers "
          "instead of nobody", tally2["targets"] == 2, str(tally2))
    payload = json.loads(sent_calls[0]["data"])
    check("…and the payload says assigned=false, so the phone cannot imply an "
          "assignment the routing layer did not make",
          payload["assigned"] is False, str(payload)[:120])

    # ── Payload contents ────────────────────────────────────────────────────
    check("Payload carries the live Day 10 score, not the legacy NULL `score` "
          "column", payload["score"] == 7.5, str(payload.get("score")))
    check("Payload carries identifiers only — no crop, clip or subject label "
          "transits the third-party push relay or a lock screen",
          not any(k in payload for k in ("crop_url", "clip_url", "subject_label")),
          str(sorted(payload.keys())))
    check("Payload timestamp is ISO-8601 with Z (Safari rejects the raw "
          "SQLite space form)",
          payload["created_at"].endswith("Z") and "T" in payload["created_at"],
          payload["created_at"])

    # ── Expired subscription (410) is deleted ───────────────────────────────
    def fake_410(**kwargs):
        raise FakeWebPushException("gone", 410)

    push_service.reset_rate_limits()
    pywebpush.webpush = fake_410
    try:
        a = new_alert("expired")
        tally3 = push_service.send_critical_push(a, {"assigned_officer": 1}, db)
    finally:
        pywebpush.webpush = original_webpush

    still_there = db.execute(text(
        "SELECT COUNT(*) FROM push_subscriptions WHERE id = :i"
    ), {"i": sub_a}).scalar()
    check("A 410 from the push service DELETES the subscription — retrying a "
          "permanently-retired endpoint would fail on every future alert",
          still_there == 0 and tally3["expired"] == 1,
          f"rows={still_there} tally={tally3}")

    expired_log = db.execute(text("""
        SELECT subscription_id, officer_id, status_code FROM push_delivery_log
        WHERE status = 'expired'
    """)).mappings().fetchone()
    check("…and the audit row SURVIVES that deletion (subscription_id NULL, "
          "officer_id retained) — the attempt is what an audit needs",
          expired_log is not None and expired_log["subscription_id"] is None
          and expired_log["officer_id"] == 1 and expired_log["status_code"] == 410,
          str(dict(expired_log)) if expired_log else "no row")

    # ── Unexpected failure is logged, never raised ──────────────────────────
    def fake_boom(**kwargs):
        raise RuntimeError("relay exploded")

    push_service.reset_rate_limits()
    pywebpush.webpush = fake_boom
    try:
        a = new_alert("boom")
        tally4 = push_service.send_critical_push(a, {"assigned_officer": 2}, db)
    finally:
        pywebpush.webpush = original_webpush

    check("An unexpected send error is logged as 'failed' and never raises "
          "into the alert pipeline", tally4["failed"] == 1, str(tally4))

    # ── TEST 8: stale subscription cleanup ──────────────────────────────────
    db.execute(text("""
        UPDATE push_subscriptions
        SET last_seen_at = datetime('now', '-31 days')
        WHERE id = :i
    """), {"i": sub_b})
    db.commit()
    fresh = add_sub(1, "https://fcm.example/ep-fresh")
    removed = push_service.cleanup_stale_subscriptions()
    check("TEST 8: cleanup removes a subscription unseen for 31 days",
          removed == 1, f"removed={removed}")
    check("TEST 8: …and leaves a recently-seen one alone",
          db.execute(text("SELECT COUNT(*) FROM push_subscriptions WHERE id = :i"),
                     {"i": fresh}).scalar() == 1)

    db.close()
    asyncio.run(_api_tests(cam.id))

    print(f"\n{'=' * 70}\n{len(PASS)} passed, {len(FAIL)} failed\n{'=' * 70}")
    print("NOTE: on-device delivery (checkpoint tests 2/3/4/6/10) requires a "
          "phone + HTTPS tunnel and is NOT covered here.")
    if FAIL:
        print("FAILED:", FAIL)
        sys.exit(1)


async def _api_tests(camera_db_id: int) -> None:
    from starlette.testclient import TestClient

    from backend.db.models import Alert
    from backend.db.session import SessionLocal

    db = SessionLocal()
    a = Alert(alert_type="WATCHLIST_FACE_MATCH", camera_id=camera_db_id,
              subject_label="detail test", status="new", lifecycle_status="OPEN",
              severity="critical", iq_contribution=7.0,
              created_at=datetime.utcnow())
    db.add(a)
    db.commit()
    db.refresh(a)
    alert_id = a.id
    db.close()

    import backend.main as backend_main

    with TestClient(backend_main.app) as client:
        # ── VAPID key endpoint is intentionally public ──────────────────────
        r = client.get("/api/v1/push/vapid-public-key")
        check("GET /push/vapid-public-key needs no auth (it authenticates the "
              "SERVER to the push service, not the client to us)",
              r.status_code == 200 and len(r.json()["public_key"]) == 87,
              str(r.status_code))

        # ── TEST 1: anonymous subscribe rejected ────────────────────────────
        body = {"endpoint": "https://fcm.example/anon",
                "keys": {"p256dh": "x", "auth": "y"}}
        r = client.post("/api/v1/push/subscribe", json=body)
        check("TEST 1: subscribing without auth is rejected — no anonymous "
              "device subscriptions",
              r.status_code in (401, 403), str(r.status_code))

        tok = client.post("/api/v1/auth/login",
                          json={"username": "Officer Alpha", "password": "pw"}
                          ).json()["access_token"]
        h = {"Authorization": f"Bearer {tok}"}

        r = client.post("/api/v1/push/subscribe", json=body, headers=h)
        check("TEST 1: authenticated subscribe succeeds and binds to an officer",
              r.status_code == 200 and r.json()["officer_id"] == 1,
              f"{r.status_code} {r.text[:110]}")
        sub_id = r.json()["subscription_id"]

        # Re-subscribe on the same device → upsert, not a duplicate row.
        body2 = dict(body, keys={"p256dh": "NEW", "auth": "NEW"})
        r = client.post("/api/v1/push/subscribe", json=body2, headers=h)
        check("Re-subscribing the same device UPDATES in place — tapping "
              "Enable Alerts twice must not create duplicate rows or "
              "duplicate buzzes",
              r.status_code == 200 and r.json()["subscription_id"] == sub_id,
              f"{r.json().get('subscription_id')} vs {sub_id}")

        db = SessionLocal()
        from sqlalchemy import text
        n = db.execute(text(
            "SELECT COUNT(*) FROM push_subscriptions WHERE endpoint = :e"
        ), {"e": body["endpoint"]}).scalar()
        keys = db.execute(text(
            "SELECT p256dh FROM push_subscriptions WHERE endpoint = :e"
        ), {"e": body["endpoint"]}).scalar()
        db.close()
        check("…exactly one row for that endpoint", n == 1, f"rows={n}")
        check("…with the refreshed keys stored", keys == "NEW", str(keys))

        # ── A user with no officers row must be refused, not guessed ────────
        atok = client.post("/api/v1/auth/login",
                           json={"username": "admin_nobody", "password": "pw"}
                           ).json()["access_token"]
        ah = {"Authorization": f"Bearer {atok}"}
        r = client.post("/api/v1/push/subscribe",
                        json={"endpoint": "https://fcm.example/orphan",
                              "keys": {"p256dh": "a", "auth": "b"}}, headers=ah)
        check("A login with no matching officers row is REFUSED (409) rather "
              "than attached to an arbitrary officer — guessing here would "
              "route one officer's alerts to another's phone",
              r.status_code == 409, f"{r.status_code} {r.text[:100]}")

        # ── Admin listing ───────────────────────────────────────────────────
        r = client.get("/api/v1/push/subscriptions", headers=ah)
        check("GET /push/subscriptions works for admin", r.status_code == 200,
              str(r.status_code))
        check("…and truncates endpoints so an ops screen does not display a "
              "device-unique token in full",
              all("endpoint" not in row for row in r.json())
              and all(row["endpoint_preview"].endswith("…") for row in r.json()),
              str(r.json())[:110])

        r = client.get("/api/v1/push/subscriptions", headers=h)
        check("…and is refused for a non-admin officer", r.status_code == 403,
              str(r.status_code))

        # ── Unsubscribe ─────────────────────────────────────────────────────
        r = client.request("DELETE", "/api/v1/push/subscribe",
                           json={"endpoint": body["endpoint"]}, headers=h)
        check("DELETE /push/subscribe removes the caller's own subscription",
              r.status_code == 200 and r.json()["removed"] == 1, r.text[:90])

        # ── /alerts/{id}/detail ─────────────────────────────────────────────
        r = client.get(f"/api/v1/alerts/{alert_id}/detail", headers=h)
        check("GET /alerts/{id}/detail returns 200", r.status_code == 200,
              str(r.status_code))
        d = r.json()
        check("Detail aggregates alert + camera + routing in ONE response "
              "(the card opens on a cold mobile start; three fetches would be "
              "three chances to hang)",
              all(k in d for k in ("alert_id", "camera_name", "routing",
                                    "score", "clip_url", "crop_url")),
              str(sorted(d.keys())))
        check("Detail prefers the live IQ score over the legacy NULL column",
              d["score"] == 7.0, str(d["score"]))
        check("Detail timestamp is ISO-8601 Z",
              d["created_at"].endswith("Z"), str(d["created_at"]))
        check("clip_url points at the AUTHENTICATED evidence endpoint, never a "
              "raw filesystem path (evidence must not become a public URL)",
              d["clip_url"] is None or d["clip_url"].startswith("/api/v1/evidence/"),
              str(d["clip_url"]))

        r = client.get("/api/v1/alerts/99999999/detail", headers=h)
        check("Unknown alert → 404, not a 500", r.status_code == 404,
              str(r.status_code))
        r = client.get(f"/api/v1/alerts/{alert_id}/detail")
        check("Alert detail requires auth", r.status_code in (401, 403),
              str(r.status_code))


if __name__ == "__main__":
    main()
