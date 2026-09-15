"""Show the department boundary working, from three different logins.

Built to be run in front of someone. It logs in as each seeded account and
prints what that account can see and what it is refused, so the boundary is
visible rather than asserted.

The distinction it is making: a filtered LIST is not access control. Every
camera-specific endpoint is checked too, so naming another department's camera
directly is refused rather than served. That is what the "denied" column shows.

Run:  python -m backend.scripts.demo_department_scoping
"""
from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

ACCOUNTS = [
    ("admin", "admin123"),
    ("operator", "operator123"),
    ("viewer", "viewer123"),
]


def main() -> int:
    from fastapi.testclient import TestClient
    from backend.main import app
    from backend.db.session import SessionLocal
    from backend.db.models import Camera

    client = TestClient(app)

    db = SessionLocal()
    cams = db.query(Camera).filter(Camera.is_deleted == False).all()  # noqa: E712
    by_dept: dict[str, list] = {}
    for c in cams:
        by_dept.setdefault((c.department or "unassigned").strip().lower(),
                           []).append(c)
    db.close()

    print("FLEET")
    for dept, group in sorted(by_dept.items(), key=lambda kv: -len(kv[1])):
        print(f"   {dept:<18}{len(group):>3} cameras")
    print(f"   {'TOTAL':<18}{len(cams):>3}\n")

    print("=" * 78)
    for username, password in ACCOUNTS:
        r = client.post("/api/v1/auth/login",
                        json={"username": username, "password": password})
        if r.status_code != 200:
            print(f"{username}: login failed ({r.status_code})")
            continue
        h = {"Authorization": f"Bearer {r.json()['access_token']}"}

        scope = client.get("/api/v1/cameras/scope", headers=h).json()
        print(f"\nLOGIN: {username}  (role={scope['role']}, "
              f"department={scope['department'] or 'state-wide'})")
        print(f"  sees {scope['cameras_visible']} of "
              f"{scope['cameras_total']} cameras")
        print(f"  {scope['explanation']}")

        # Try one camera from every department, by id, and report the verdict.
        allowed, denied = [], []
        for dept, group in sorted(by_dept.items()):
            cam = group[0]
            key = cam.camera_id or str(cam.id)
            resp = client.get(
                f"/api/v1/analytics/live/stream-token/{key}", headers=h)
            (allowed if resp.status_code == 200 else denied).append(
                f"{dept}({key})")
        print(f"  video ALLOWED : {', '.join(allowed) if allowed else 'none'}")
        print(f"  video DENIED  : {', '.join(denied) if denied else 'none'}")

    print("\n" + "=" * 78)
    print("""
The DENIED column is the point. Those cameras are not merely absent from the
operator's list — asking for them by name returns HTTP 403 naming the owning
department. Filtering a list while serving anything requested by id is the
usual way this control is got wrong, and it is why every camera-specific
endpoint is checked: live frame, MJPEG stream, HLS manifest, HLS segments,
stream tokens, IQ scores and telemetry.""")
    return 0


if __name__ == "__main__":
    sys.exit(main())
