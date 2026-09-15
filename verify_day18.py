"""
verify_day18.py — Day 18 control-room backend endpoints.

Frontend assembly (single-socket guarantee, panel isolation, tile behaviour)
is covered by frontend/src/components/__tests__/ControlRoom.test.jsx — 21
tests under jsdom. This file covers the three endpoints those panels read.

Run:  python verify_day18.py
"""
from __future__ import annotations

import os
import sys
import tempfile
from datetime import datetime, timedelta
from pathlib import Path

_ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(_ROOT))

_tmpdir = tempfile.mkdtemp(prefix="verify_day18_")
os.environ["DATABASE_URL"] = f"sqlite:///{_tmpdir}/verify_day18.db"
os.environ["EVIDENCE_DIR"] = str(Path(_tmpdir) / "evidence")
os.environ.pop("SENTINEL_SOURCE", None)

PASS: list[str] = []
FAIL: list[str] = []


def check(name: str, cond: bool, detail: str = "") -> None:
    (PASS if cond else FAIL).append(name)
    print(f"{'PASS' if cond else 'FAIL'} — {name}{(': ' + detail) if detail else ''}")


def main() -> None:
    from starlette.testclient import TestClient

    from backend.auth.routes import pwd_context
    from backend.db.models import Alert, Base, Camera, User
    from backend.db.session import SessionLocal, engine

    Base.metadata.create_all(bind=engine)
    db = SessionLocal()

    cam = Camera(camera_id="D18-CAM", name="Gate", zone="North",
                 gps_lat=23.0, gps_lon=72.5, status="ONLINE")
    db.add(cam)
    db.add(User(username="d18_admin", hashed_password=pwd_context.hash("pw"),
                role="admin", is_active=True))
    db.commit()
    db.refresh(cam)

    now = datetime.utcnow()

    def mk(sev, when=None, deleted=False):
        db.add(Alert(alert_type="LOITERING", camera_id=cam.id,
                     subject_label="x", status="new", lifecycle_status="OPEN",
                     severity=sev, created_at=when or now, is_deleted=deleted))

    # Lowercase is what this system actually writes. Mixed case is what the
    # legacy raw-SQL ANPR path writes. Both must land in the same bucket.
    mk("critical"); mk("critical"); mk("CRITICAL")
    mk("high"); mk("medium"); mk("low")
    mk(None)                                   # unscored — must be counted
    mk("critical", when=now - timedelta(days=3))   # outside "today"
    mk("critical", deleted=True)                   # soft-deleted — excluded
    db.commit()
    db.close()

    import backend.main as backend_main

    with TestClient(backend_main.app) as client:
        tok = client.post("/api/v1/auth/login",
                          json={"username": "d18_admin", "password": "pw"}
                          ).json()["access_token"]
        h = {"Authorization": f"Bearer {tok}"}

        r = client.get("/api/v1/alerts/summary?since=today", headers=h)
        check("GET /alerts/summary returns 200", r.status_code == 200, str(r.status_code))
        d = r.json()

        check("Severity grouping is case-insensitive — 'critical' and "
              "'CRITICAL' land in ONE bucket, not two",
              d["CRITICAL"] == 3, f"CRITICAL={d.get('CRITICAL')}")
        check("HIGH / MEDIUM / LOW counted",
              d["HIGH"] == 1 and d["MEDIUM"] == 1 and d["LOW"] == 1,
              f"{d.get('HIGH')}/{d.get('MEDIUM')}/{d.get('LOW')}")
        check("Alerts with NULL severity are surfaced as UNSCORED, not dropped "
              "— dropping them would make the panel disagree with the feed",
              d["UNSCORED"] == 1, str(d.get("UNSCORED")))
        check("Soft-deleted alerts are excluded",
              d["total"] == 7, f"total={d['total']}")
        check("A 3-day-old alert is outside the 'today' window",
              d["CRITICAL"] == 3)
        check("since is ISO-8601 with Z",
              d["since"].endswith("Z") and "T" in d["since"], d["since"])

        r_all = client.get("/api/v1/alerts/summary?since=all", headers=h)
        check("since=all widens the window (picks up the 3-day-old alert)",
              r_all.json()["CRITICAL"] == 4, str(r_all.json()["CRITICAL"]))

        r_bad = client.get("/api/v1/alerts/summary?since=bogus", headers=h)
        check("An invalid `since` is rejected by validation, not silently "
              "coerced", r_bad.status_code == 422, str(r_bad.status_code))

        r = client.get("/api/v1/alerts/summary")
        check("Summary requires auth", r.status_code in (401, 403), str(r.status_code))

        # ── Review queue count ──────────────────────────────────────────────
        r = client.get("/api/v1/reid/review-queue/count", headers=h)
        check("GET /reid/review-queue/count returns a pending integer",
              r.status_code == 200 and isinstance(r.json()["pending"], int),
              r.text[:80])

        # ── Client error reporting ──────────────────────────────────────────
        r = client.post("/api/v1/client-error", json={
            "panel": "Live Map", "error": "tiles failed",
            "stack": "at LiveMap", "component": "<LiveMap>",
        })
        check("POST /client-error accepts a report WITHOUT auth — a panel can "
              "crash before auth settles, and a reporter that needs a session "
              "cannot report that case",
              r.status_code == 200, f"{r.status_code} {r.text[:80]}")
        check("…and always answers 200 so the client never treats error "
              "reporting as a failure path",
              r.json().get("status") == "recorded", r.text[:80])

        r = client.post("/api/v1/client-error", json={})
        check("An empty error report is tolerated, not rejected",
              r.status_code == 200, str(r.status_code))

    print(f"\n{'=' * 70}\n{len(PASS)} passed, {len(FAIL)} failed\n{'=' * 70}")
    if FAIL:
        print("FAILED:", FAIL)
        sys.exit(1)


if __name__ == "__main__":
    main()
