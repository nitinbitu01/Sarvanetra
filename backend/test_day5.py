"""
backend/test_day5.py — Clean automated acceptance test suite for Day 5.
"""

import json
import urllib.request
import urllib.error

BASE = "http://localhost:8000"
PASS = "[PASS]"
FAIL = "[FAIL]"
results = []


def check(name: str, ok: bool, detail: str = ""):
    results.append((name, ok))
    print(f"{PASS if ok else FAIL} {name}" + (f": {detail}" if detail else ""))


def req(method: str, path: str, body=None, token=None):
    url = BASE + path
    data = json.dumps(body).encode() if body is not None else None
    headers = {"Content-Type": "application/json"}
    if token:
        headers["Authorization"] = f"Bearer {token}"
    req_obj = urllib.request.Request(url, data=data, headers=headers, method=method)
    try:
        with urllib.request.urlopen(req_obj, timeout=10) as r:
            raw = r.read()
            try:
                resp_data = json.loads(raw.decode("utf-8")) if raw else None
            except Exception:
                resp_data = raw.decode("utf-8", errors="ignore")
            return r.status, resp_data
    except urllib.error.HTTPError as e:
        raw = e.read()
        try:
            resp_data = json.loads(raw.decode("utf-8")) if raw else None
        except Exception:
            resp_data = raw.decode("utf-8", errors="ignore")
        return e.code, resp_data


def run_tests():
    print("=== Day 5 Acceptance Tests ===")

    # 1. Health check
    st, h = req("GET", "/api/v1/health")
    check("1. GET /api/v1/health returns 200 OK", st == 200 and h.get("status") == "ok", str(h))

    # 2. Auth protection
    st, _ = req("GET", "/api/v1/cameras")
    check("2. GET /api/v1/cameras without token returns 401", st == 401)

    # 3. Admin login
    st, login_data = req("POST", "/api/v1/auth/login", {"username": "admin", "password": "aNQdY7BD4RTsGI05iEiCrw"})
    check("3. POST /api/v1/auth/login succeeds", st == 200 and "access_token" in login_data)
    if st != 200:
        print("Login failed, aborting further tests.")
        return
    token = login_data["access_token"]
    check("   Admin role verified", login_data.get("role") == "admin")

    # 4. User profile /me
    st, me = req("GET", "/api/v1/auth/me", token=token)
    check("4. GET /api/v1/auth/me returns admin user", st == 200 and me.get("username") == "admin")

    # 5. Camera test connection
    st, test_res = req("POST", "/api/v1/cameras/test", {"ip_address": "192.168.1.50", "protocol": "RTSP"}, token=token)
    check("5. POST /api/v1/cameras/test connection probe", st == 200 and "success" in test_res)

    # 6. Add camera
    cam_payload = {
        "name": "Gandhinagar Hwy Cam-4",
        "ip_address": "10.10.4.12",
        "protocol": "RTSP",
        "zone": "Gandhinagar",
        "department": "Traffic Police",
        "risk_level": "HIGH",
        "gps_lat": 23.2156,
        "gps_lon": 72.6369,
    }
    st, cam = req("POST", "/api/v1/cameras", cam_payload, token=token)
    check("6. POST /api/v1/cameras creates new camera", st == 201 and cam.get("id") is not None, str(cam))
    cam_id = cam.get("id")

    # 7. List cameras
    st, cams = req("GET", "/api/v1/cameras", token=token)
    check("7. GET /api/v1/cameras includes new camera", isinstance(cams, list) and any(c.get("id") == cam_id for c in cams))

    # 8. Check audit log for camera creation & login
    st, audit = req("GET", "/api/v1/audit-log?limit=50", token=token)
    check("8. GET /api/v1/audit-log returns audit entries", st == 200 and isinstance(audit, list))
    check("   Audit log records CAMERA_ADD action", any(r.get("action") == "CAMERA_ADD" for r in audit))
    check("   Audit log records LOGIN action", any(r.get("action") == "LOGIN" for r in audit))

    # 9. Alert feed listing
    st, alerts = req("GET", "/api/v1/alerts?limit=10", token=token)
    check("9. GET /api/v1/alerts returns alert feed", st == 200 and isinstance(alerts, list))

    # 10. Soft-delete camera
    st, _ = req("DELETE", f"/api/v1/cameras/{cam_id}", token=token)
    check("10. DELETE /api/v1/cameras/{id} returns 204 No Content", st == 204)

    # 11. Verify soft-deleted camera is omitted from listing
    st, cams_after = req("GET", "/api/v1/cameras", token=token)
    check("11. Soft-deleted camera excluded from camera list", not any(c.get("id") == cam_id for c in cams_after))

    # 12. Check audit log for soft delete
    st, audit_after = req("GET", "/api/v1/audit-log?limit=50", token=token)
    check("12. Audit log records CAMERA_SOFT_DELETE action", any(r.get("action") == "CAMERA_SOFT_DELETE" for r in audit_after))

    # Final summary
    n_pass = sum(1 for _, ok in results if ok)
    n_fail = sum(1 for _, ok in results if not ok)
    print("=" * 60)
    print(f"FINAL SUMMARY: {n_pass} PASSED / {n_fail} FAILED")
    if n_fail == 0:
        print("ALL DAY 5 ACCEPTANCE CRITERIA VERIFIED PERFECTLY!")
    print("=" * 60)


if __name__ == "__main__":
    run_tests()
