"""
verify_iq_api.py — End-to-end HTTP check of the SENTINEL IQ endpoints.

Exercises the real FastAPI app object through Starlette's TestClient, which
runs the full ASGI stack: lifespan startup, RequestIDMiddleware, CORS, the
slowapi rate limiter, JWT auth dependencies, and the routers themselves.
This is the layer verify_part_c.py does NOT cover — that one calls the
service functions directly, so a route that was never registered, an auth
dependency that rejects everyone, or a payload that fails JSON serialisation
would all have passed it.

Runs against a SCRATCH database with a known admin, never the real
output/sentinel.db — the real admin password is randomly generated at seed
time and is not something a test should be resetting.

Run:  python verify_iq_api.py
"""
from __future__ import annotations

import json
import os
import sys
import tempfile
from pathlib import Path

_ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(_ROOT))

_tmpdir = tempfile.mkdtemp(prefix="verify_iq_api_")
os.environ["DATABASE_URL"] = f"sqlite:///{_tmpdir}/verify_iq_api.db"
# No pipeline — this checks the API surface, not the detection loop.
os.environ.pop("SENTINEL_SOURCE", None)

PASS: list[str] = []
FAIL: list[str] = []

ADMIN_PW = "verify-admin-pw"
OFFICER_PW = "verify-officer-pw"


def check(name: str, cond: bool, detail: str = "") -> None:
    (PASS if cond else FAIL).append(name)
    print(f"{'PASS' if cond else 'FAIL'} — {name}{(': ' + detail) if detail else ''}")


def seed() -> None:
    """Create the scratch schema plus two users, a camera, and two scored alerts."""
    from backend.auth.routes import pwd_context
    from backend.db.models import Alert, Base, Camera, User
    from backend.db.session import SessionLocal, engine
    from backend.services.sentinel_iq import (
        IQContext, compute_alert_iq, set_night_override,
    )

    # Pin the night multiplier OFF so the arithmetic below is deterministic.
    # The night window is 20:00-06:00 IST; without this pin, running the
    # suite during an Indian evening turns the expected 3.0 + 7.0 = 10.0 into
    # 13.0 and the test "fails" against a perfectly correct engine. The
    # night-mode ENDPOINT is still exercised further down — that part sets
    # the override explicitly rather than relying on the ambient clock.
    set_night_override(False)

    Base.metadata.create_all(bind=engine)
    db = SessionLocal()
    try:
        db.add_all([
            User(username="verify_admin", hashed_password=pwd_context.hash(ADMIN_PW),
                 role="admin", is_active=True),
            User(username="verify_officer", hashed_password=pwd_context.hash(OFFICER_PW),
                 role="officer", is_active=True),
        ])
        cam = Camera(camera_id="CAM-API-01", name="API Verify Camera",
                     zone="Central", status="ONLINE")
        db.add(cam)
        db.commit()
        db.refresh(cam)

        # Two scored alerts, computed through the REAL engine so the payload
        # carries genuine stored breakdowns rather than hand-written JSON.
        for alert_type in ("LOITERING", "WATCHLIST_FACE_MATCH"):
            iq = compute_alert_iq(alert_type, IQContext(
                db=db, camera_db_id=cam.id, camera_str_id="CAM-API-01",
            ))
            db.add(Alert(
                alert_type=alert_type, camera_id=cam.id,
                subject_label=f"{alert_type} api-verify",
                status="new", lifecycle_status="OPEN",
                iq_contribution=iq.total if iq else None,
                iq_breakdown_json=iq.to_json() if iq else None,
                meta_json=json.dumps({"source": "verify_iq_api"}),
            ))

        # An UNSCORED alert — must be counted separately, never as 0.0.
        db.add(Alert(
            alert_type="PERIMETER_BREACH", camera_id=cam.id,
            subject_label="unscored roadmap-type alert",
            status="new", lifecycle_status="OPEN",
            iq_contribution=None, iq_breakdown_json=None,
        ))
        db.commit()
        return cam.id
    finally:
        db.close()


def main() -> None:
    from starlette.testclient import TestClient

    camera_id = seed()

    import backend.main as backend_main

    with TestClient(backend_main.app) as client:
        # ── Auth ────────────────────────────────────────────────────────────
        r = client.post("/api/v1/auth/login",
                        json={"username": "verify_admin", "password": ADMIN_PW})
        check("Admin login succeeds", r.status_code == 200, f"{r.status_code} {r.text[:120]}")
        admin_h = {"Authorization": f"Bearer {r.json()['access_token']}"}

        r = client.post("/api/v1/auth/login",
                        json={"username": "verify_officer", "password": OFFICER_PW})
        check("Officer login succeeds", r.status_code == 200, str(r.status_code))
        officer_h = {"Authorization": f"Bearer {r.json()['access_token']}"}

        # ── Auth gate on the score card ─────────────────────────────────────
        r = client.get("/api/v1/iq/cameras")
        check("GET /iq/cameras requires auth (401/403 unauthenticated)",
              r.status_code in (401, 403), str(r.status_code))

        # ── The score card itself ───────────────────────────────────────────
        r = client.get("/api/v1/iq/cameras?window_minutes=60", headers=admin_h)
        check("GET /iq/cameras returns 200 over real HTTP", r.status_code == 200,
              f"{r.status_code} {r.text[:200]}")
        body = r.json()

        cams = body.get("cameras", [])
        check("Score card lists the seeded camera", len(cams) == 1, json.dumps(cams)[:200])

        if cams:
            cam = cams[0]
            # LOITERING 3.0 + WATCHLIST_FACE_MATCH 7.0, no multipliers applied
            # (daytime, no journeys) = 10.0
            check("Aggregate is the sum of stored contributions (3.0 + 7.0 = 10.0)",
                  abs(cam["total_score"] - 10.0) < 1e-9, str(cam["total_score"]))
            check("Aggregate counts 2 scored alerts", cam["alert_count"] == 2,
                  str(cam["alert_count"]))
            check("Unscored alert is reported separately, NOT summed as 0.0",
                  cam["unscored_count"] == 1, str(cam["unscored_count"]))
            check("Each contribution carries its full stored breakdown",
                  all(c.get("breakdown") and "multipliers" in c["breakdown"]
                      for c in cam["contributions"]),
                  json.dumps(cam["contributions"])[:200])
            check("Camera identity resolved in the payload (not 'Unknown camera')",
                  cam["camera_name"] == "API Verify Camera" and cam["camera_zone"] == "Central",
                  f'{cam["camera_name"]} / {cam["camera_zone"]}')

        # ── PERIMETER_BREACH: label only, no number anywhere in the payload ──
        roadmap = body.get("roadmap_triggers", [])
        check("Response carries the roadmap trigger as a static label",
              any(t["alert_type"] == "PERIMETER_BREACH"
                  and t["status"] == "Roadmap — not yet live" for t in roadmap),
              json.dumps(roadmap))
        check("No numeric value appears on any roadmap trigger in the HTTP payload",
              not any(isinstance(v, (int, float)) for t in roadmap for v in t.values()),
              json.dumps(roadmap))
        # The whole serialized response must not attach a score to PERIMETER_BREACH
        # anywhere — including inside a camera's contribution list.
        scored_roadmap = [
            c for cm in cams for c in cm.get("contributions", [])
            if c.get("alert_type") == "PERIMETER_BREACH"
        ]
        check("PERIMETER_BREACH never appears with an iq_contribution anywhere "
              "in the response",
              not scored_roadmap,
              json.dumps(scored_roadmap) if scored_roadmap else "")

        # ── Single-camera endpoint ──────────────────────────────────────────
        r = client.get(f"/api/v1/iq/cameras/{camera_id}?window_minutes=60", headers=admin_h)
        check("GET /iq/cameras/{id} returns 200", r.status_code == 200, str(r.status_code))
        check("Single-camera total matches the list endpoint's total",
              abs(r.json()["camera"]["total_score"] - 10.0) < 1e-9,
              str(r.json()["camera"]["total_score"]))

        # ── Night mode: admin-only write ────────────────────────────────────
        r = client.post("/api/v1/admin/night-mode", json={"enabled": True}, headers=officer_h)
        check("POST /admin/night-mode is refused for a non-admin (403)",
              r.status_code == 403, f"{r.status_code} {r.text[:120]}")

        r = client.post("/api/v1/admin/night-mode", json={"enabled": True}, headers=admin_h)
        check("POST /admin/night-mode succeeds for admin", r.status_code == 200,
              f"{r.status_code} {r.text[:200]}")
        check("Response reports mode = manual_override",
              r.json().get("mode") == "manual_override", json.dumps(r.json()))

        r = client.get("/api/v1/admin/night-mode", headers=officer_h)
        check("GET /admin/night-mode is readable by a non-admin",
              r.status_code == 200 and r.json()["override"] is True, str(r.status_code))

        # ── Already-fired alerts are NOT rescored by the override ──────────
        r = client.get("/api/v1/iq/cameras?window_minutes=60", headers=admin_h)
        check("Flipping night mode does not retroactively change the aggregate",
              abs(r.json()["cameras"][0]["total_score"] - 10.0) < 1e-9,
              str(r.json()["cameras"][0]["total_score"]))

        # Release it so the process doesn't leave a forced override behind.
        client.post("/api/v1/admin/night-mode", json={"enabled": None}, headers=admin_h)
        r = client.get("/api/v1/admin/night-mode", headers=admin_h)
        check("Passing enabled=null returns to clock mode",
              r.json()["override"] is None and r.json()["mode"] == "clock",
              json.dumps(r.json()))

        # ── /alerts exposes the stored IQ fields for the feed badge ────────
        r = client.get("/api/v1/alerts?limit=50", headers=admin_h)
        check("GET /alerts returns 200", r.status_code == 200, str(r.status_code))
        alerts = r.json()
        scored = [a for a in alerts if a.get("iq_contribution") is not None]
        check("/alerts exposes iq_contribution for scored alerts", len(scored) == 2,
              str(len(scored)))
        check("/alerts exposes a parsed iq_breakdown object (not a raw JSON string)",
              all(isinstance(a.get("iq_breakdown"), dict) for a in scored),
              str([type(a.get("iq_breakdown")).__name__ for a in scored]))
        unscored = [a for a in alerts if a["alert_type"] == "PERIMETER_BREACH"]
        check("Unscored alert reports iq_contribution as null, not 0",
              len(unscored) == 1 and unscored[0]["iq_contribution"] is None,
              json.dumps(unscored)[:160])

    print(f"\n{'=' * 68}\n{len(PASS)} passed, {len(FAIL)} failed\n{'=' * 68}")
    if FAIL:
        print("FAILED:", FAIL)
        sys.exit(1)


if __name__ == "__main__":
    main()
