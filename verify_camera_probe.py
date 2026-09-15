"""
verify_camera_probe.py — the camera connectivity test must probe, not guess.

BACKGROUND
──────────
test_camera_connection() used to be `random.random() < 0.85` with a
time.sleep() to fake plausible latency. It reported "RTSP connection to
<ip> succeeded" for addresses that need not exist, and POST /cameras/test
wrote that fabricated success into the audit log as fact.

The two properties that matter here are hard to test by asserting a single
happy path, so this checks them directly:

  1. DETERMINISM — the same input must give the same answer. A simulator
     with an 85% success rate passes any single-shot assertion 85% of the
     time, which is exactly why the old behaviour survived so long. This
     runs each case repeatedly and requires every result to agree.
  2. HONESTY — an unreachable address must fail, and a merely-open TCP port
     must NOT be reported as a verified camera stream.

Run:  python verify_camera_probe.py
"""
from __future__ import annotations

import os
import socket
import sys
import tempfile
import threading
from pathlib import Path

_ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(_ROOT))

_tmpdir = tempfile.mkdtemp(prefix="verify_camera_probe_")
os.environ["DATABASE_URL"] = f"sqlite:///{_tmpdir}/probe.db"
os.environ.pop("SENTINEL_SOURCE", None)

PASS: list[str] = []
FAIL: list[str] = []


def check(name: str, cond: bool, detail: str = "") -> None:
    (PASS if cond else FAIL).append(name)
    print(f"{'PASS' if cond else 'FAIL'} — {name}{(': ' + detail) if detail else ''}")


class _DummyTCPServer:
    """A socket that accepts connections but serves no media.

    This is the case the old simulator would have called a success ~85% of
    the time: something IS listening, but it is not a camera.
    """

    def __init__(self):
        self._sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        self._sock.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        self._sock.bind(("127.0.0.1", 0))
        self._sock.listen(5)
        self.port = self._sock.getsockname()[1]
        self._stop = threading.Event()
        self._thread = threading.Thread(target=self._serve, daemon=True)
        self._thread.start()

    def _serve(self):
        while not self._stop.is_set():
            try:
                self._sock.settimeout(0.5)
                conn, _ = self._sock.accept()
                conn.close()
            except (socket.timeout, OSError):
                continue

    def close(self):
        self._stop.set()
        try:
            self._sock.close()
        except OSError:
            pass


def _free_port() -> int:
    """A port number nothing is listening on."""
    s = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    s.bind(("127.0.0.1", 0))
    port = s.getsockname()[1]
    s.close()
    return port


def main() -> None:
    from backend.services.camera_connection import test_camera_connection

    # ── 1. No simulation left in the module ─────────────────────────────────
    src = (_ROOT / "backend" / "services" / "camera_connection.py").read_text(encoding="utf-8")
    body_start = src.find("def test_camera_connection")
    body = src[body_start:] if body_start != -1 else src
    has_random = "random." in body
    check("No `random` use remains in the probe implementation "
          "(the old 85%-success simulator is gone)",
          not has_random,
          "found a random.* call in test_camera_connection" if has_random else "")

    # ── 2. Unreachable port must FAIL, deterministically ────────────────────
    dead_port = _free_port()
    results = [
        test_camera_connection(f"127.0.0.1:{dead_port}", "RTSP")
        for _ in range(5)
    ]
    check("Unreachable address fails", all(not r.success for r in results),
          str([r.success for r in results]))
    check("Unreachable address gives the SAME answer every time "
          "(a 5-run disagreement is what exposed the old simulator)",
          len({r.success for r in results}) == 1,
          str([r.success for r in results]))
    check("Failure message names the actual cause, not a generic string",
          "refused" in results[0].message.lower()
          or "unreachable" in results[0].message.lower()
          or "timed out" in results[0].message.lower(),
          results[0].message)
    check("Unreachable address never claims stream_verified",
          not any(r.stream_verified for r in results))

    # ── 3. Open TCP port that is NOT a camera must not pass as verified ─────
    server = _DummyTCPServer()
    try:
        rtsp_results = [
            test_camera_connection(f"127.0.0.1:{server.port}", "RTSP")
            for _ in range(3)
        ]
        check("A listening port that serves no stream is NOT reported as a "
              "working RTSP camera (the old simulator's core failure)",
              all(not r.success for r in rtsp_results),
              str([r.success for r in rtsp_results]))
        check("…and the message explains that a listening port is not a camera",
              "reachable" in rtsp_results[0].message.lower()
              and "stream" in rtsp_results[0].message.lower(),
              rtsp_results[0].message)
        check("…deterministically across repeats",
              len({r.success for r in rtsp_results}) == 1)

        # ONVIF has no client library — must be explicit, not a fake pass.
        onvif = test_camera_connection(f"127.0.0.1:{server.port}", "ONVIF")
        check("ONVIF reports TCP reachability WITHOUT claiming the device was "
              "verified (stream_verified stays False)",
              onvif.success is True and onvif.stream_verified is False,
              f"success={onvif.success} stream_verified={onvif.stream_verified}")
        check("…and says so in the message rather than implying a full check",
              "not performed" in onvif.message.lower()
              or "not that it is a working" in onvif.message.lower(),
              onvif.message)
    finally:
        server.close()

    # ── 4. Real local file feed passes and is genuinely verified ────────────
    feed = Path(_tmpdir) / "feed.mp4"
    feed.write_bytes(b"\x00" * 2048)
    file_res = test_camera_connection(str(feed), "HTTP")
    check("An existing, non-empty local feed file passes and is marked verified",
          file_res.success and file_res.stream_verified,
          f"success={file_res.success} stream_verified={file_res.stream_verified}")

    empty = Path(_tmpdir) / "empty.mp4"
    empty.write_bytes(b"")
    empty_res = test_camera_connection(str(empty), "HTTP")
    check("An empty feed file FAILS (a file existing is not a working feed)",
          not empty_res.success, empty_res.message)

    # ── 5. Garbage input fails cleanly, no exception ────────────────────────
    for bad in ("", "   ", "not a real host at all", "999.999.999.999"):
        try:
            r = test_camera_connection(bad, "RTSP")
            check(f"Malformed address {bad!r} fails cleanly without raising",
                  r.success is False, r.message[:80])
        except Exception as exc:
            check(f"Malformed address {bad!r} fails cleanly without raising",
                  False, f"raised {type(exc).__name__}: {exc}")

    # ── 6. The endpoint surfaces stream_verified, and audits it ─────────────
    from starlette.testclient import TestClient

    from backend.auth.routes import pwd_context
    from backend.db.models import AuditLog, Base, User
    from backend.db.session import SessionLocal, engine

    Base.metadata.create_all(bind=engine)
    db = SessionLocal()
    db.add(User(username="probe_admin", hashed_password=pwd_context.hash("pw"),
                role="admin", is_active=True))
    db.commit()
    db.close()

    import backend.main as backend_main

    with TestClient(backend_main.app) as client:
        tok = client.post("/api/v1/auth/login",
                          json={"username": "probe_admin", "password": "pw"}).json()["access_token"]
        h = {"Authorization": f"Bearer {tok}"}
        r = client.post("/api/v1/cameras/test",
                        json={"ip_address": f"127.0.0.1:{dead_port}", "protocol": "RTSP"},
                        headers=h)
        check("POST /cameras/test returns 200", r.status_code == 200, str(r.status_code))
        payload = r.json()
        check("Endpoint reports failure for an unreachable camera "
              "(previously ~85% chance of a fabricated success)",
              payload.get("success") is False, str(payload))
        check("Endpoint exposes stream_verified so the UI can distinguish "
              "'port open' from 'working camera'",
              "stream_verified" in payload, str(payload))

        db = SessionLocal()
        row = (db.query(AuditLog)
               .filter(AuditLog.action == "CAMERA_TEST")
               .order_by(AuditLog.id.desc()).first())
        import json as _json
        details = _json.loads(row.details) if row and row.details else {}
        check("Audit log records the REAL outcome, including stream_verified",
              details.get("success") is False and "stream_verified" in details,
              str(details))
        db.close()

    print(f"\n{'=' * 70}\n{len(PASS)} passed, {len(FAIL)} failed\n{'=' * 70}")
    if FAIL:
        print("FAILED:", FAIL)
        sys.exit(1)


if __name__ == "__main__":
    main()
