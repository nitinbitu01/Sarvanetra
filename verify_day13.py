"""
verify_day13.py — Evidence Lock acceptance checks.

Covers all five checkpoints plus the robustness requirements. Uses a REAL
frame buffer fed with REAL frames, a REAL cv2 clip write, a REAL SHA-256 over
the finalised file, and a REAL fpdf2 PDF — nothing about the artefact path is
mocked. What IS controlled is time: frames are stamped with explicit
wall-clock values so a 90-second window can be exercised without the test
taking 90 seconds, and capture is invoked with wait_for_post=False because
the post-roll is already staged.

Run:  python verify_day13.py
"""
from __future__ import annotations

import asyncio
import hashlib
import os
import subprocess
import sys
import tempfile
import time
from datetime import datetime, timezone
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

_tmpdir = tempfile.mkdtemp(prefix="verify_day13_")
os.environ["DATABASE_URL"] = f"sqlite:///{_tmpdir}/verify_day13.db"
# Keep all evidence artefacts out of the real evidence/ tree.
os.environ["EVIDENCE_DIR"] = str(Path(_tmpdir) / "evidence")
os.environ.pop("SENTINEL_SOURCE", None)

import numpy as np  # noqa: E402

PASS: list[str] = []
FAIL: list[str] = []


def check(name: str, cond: bool, detail: str = "") -> None:
    (PASS if cond else FAIL).append(name)
    print(f"{'PASS' if cond else 'FAIL'} — {name}{(': ' + detail) if detail else ''}")


def _frame(i: int, w: int = 320, h: int = 240) -> np.ndarray:
    """A visually distinct frame so a human can tell ordering in the clip."""
    img = np.full((h, w, 3), (i * 7) % 255, dtype=np.uint8)
    img[10:40, 10:10 + max(1, (i % 20) * 12)] = (255, 255, 255)
    return img


def _epoch(dt: datetime) -> float:
    return dt.replace(tzinfo=timezone.utc).timestamp()


def stage_buffer(camera_id: str, event_dt: datetime, pre_s: float, post_s: float,
                 fps: float = 5.0) -> int:
    """Fill a camera's buffer with frames spanning [event-pre, event+post].

    Frames are injected with explicit wall_time values rather than by sleeping,
    so the full 90s window is exercised in milliseconds.
    """
    import cv2

    from backend.services.frame_buffer import BufferedFrame, get_frame_buffer_registry

    buf = get_frame_buffer_registry().get(camera_id)
    buf.clear()
    ev = _epoch(event_dt)
    n = int((pre_s + post_s) * fps)
    entries = []
    for i in range(n):
        t = ev - pre_s + (i / fps)
        ok, enc = cv2.imencode(".jpg", _frame(i), [int(cv2.IMWRITE_JPEG_QUALITY), 85])
        if ok:
            entries.append(BufferedFrame(wall_time=t, video_time=i / fps,
                                         frame_number=i, jpeg=enc.tobytes()))
    with buf._lock:                      # noqa: SLF001 — staging fixture
        buf._frames.extend(entries)      # noqa: SLF001
    return len(entries)


async def main() -> None:
    from backend.core.config import settings
    from backend.db.models import Alert, Base, Camera, Evidence, User
    from backend.db.session import SessionLocal, engine
    from backend.services import evidence_capture as ec

    Base.metadata.create_all(bind=engine)
    ec.ensure_evidence_dirs()

    check("Startup creates evidence/ and evidence/.tmp/",
          ec.evidence_root().exists() and (ec.evidence_root() / ".tmp").exists(),
          str(ec.evidence_root()))

    db = SessionLocal()
    cam = Camera(camera_id="CAM-EV-01", name="Evidence Cam", zone="Central", status="ONLINE")
    db.add(cam)
    db.add(User(username="ev_admin", hashed_password="x", role="admin", is_active=True))
    db.commit()
    db.refresh(cam)

    # ── Severity classification (the Step-0 gap) ────────────────────────────
    check("WATCHLIST_FACE_MATCH classifies as CRITICAL "
          "(previously severity was NULL for this type, so evidence would "
          "never have triggered for the headline case)",
          ec.classify_severity("WATCHLIST_FACE_MATCH", None) == "critical")
    check("A high IQ score alone reaches CRITICAL (the 'or score threshold' path)",
          ec.classify_severity("LOITERING", 7.8) == "critical")
    check("An ordinary loitering alert is NOT critical",
          ec.classify_severity("LOITERING", 3.0) != "critical")
    check("An already-set severity is preserved, not overwritten "
          "(legacy ANPR path sets its own)",
          ec.classify_severity("LOITERING", 9.9, existing="medium") == "medium")

    def make_alert(alert_type="WATCHLIST_FACE_MATCH", iq=7.0, when=None):
        a = Alert(alert_type=alert_type, camera_id=cam.id,
                  subject_label=f"{alert_type} test", status="new",
                  lifecycle_status="OPEN", iq_contribution=iq,
                  created_at=when or datetime.utcnow())
        db.add(a)
        db.commit()
        db.refresh(a)
        return a

    # ── CHECKPOINT 1: full capture produces clip + PDF ──────────────────────
    event = datetime.utcnow()
    alert = make_alert(when=event)
    staged = stage_buffer("CAM-EV-01", event,
                          settings.EVIDENCE_PRE_SECONDS, settings.EVIDENCE_POST_SECONDS)
    check("Frame buffer staged with a full pre+post window", staged > 400, f"frames={staged}")

    await ec.on_alert_fired(alert, db, "CAM-EV-01")
    db.refresh(alert)
    check("on_alert_fired assigned severity=critical to the alert row",
          alert.severity == "critical", str(alert.severity))

    # on_alert_fired schedules with the real POST sleep; drive it directly here.
    status = await ec.capture_evidence(alert.id, wait_for_post=False)
    check("Capture reports COMPLETE", status == "COMPLETE", status)

    row = db.query(Evidence).filter(Evidence.alert_id == alert.id).first()
    db.refresh(row)
    clip = ec.evidence_dir_for(alert.id) / "clip.mp4"
    pdf = ec.evidence_dir_for(alert.id) / "custody.pdf"
    check("clip.mp4 exists in evidence/{alert_id}/", clip.exists(), str(clip))
    check("custody.pdf exists in evidence/{alert_id}/", pdf.exists(), str(pdf))
    check("Folder is named by alert_id, never a timestamp",
          clip.parent.name == str(alert.id), clip.parent.name)
    check("Row status is COMPLETE", row.status == "COMPLETE", row.status)

    # ── CHECKPOINT 2: hash matches an independent computation ───────────────
    manual = hashlib.sha256(clip.read_bytes()).hexdigest()
    check("Stored SHA-256 matches an independent hash of the clip file",
          row.sha256 == manual, f"stored={row.sha256[:16]}… manual={manual[:16]}…")

    # Cross-check against the OS tool a judge would actually run.
    try:
        out = subprocess.run(["certutil", "-hashfile", str(clip), "SHA256"],
                             capture_output=True, text=True, timeout=60)
        os_hash = "".join(out.stdout.split("\n")[1].split()).lower() if out.returncode == 0 else ""
        check("Stored hash matches OS certutil/sha256sum output",
              os_hash == row.sha256.lower(), f"os={os_hash[:16]}…")
    except Exception as exc:
        check("Stored hash matches OS certutil/sha256sum output", False, f"tool error: {exc}")

    pdf_bytes = pdf.read_bytes()
    check("Custody PDF embeds the same SHA-256 (split across two lines)",
          row.sha256[:32].encode() in pdf_bytes and row.sha256[32:].encode() in pdf_bytes)
    check("Custody PDF names Sentinel IQ as generator",
          b"Sentinel IQ" in pdf_bytes)

    # ── The clip must actually be playable, not merely present ──────────────
    import cv2

    capr = cv2.VideoCapture(str(clip))
    ok, first = capr.read()
    nframes = int(capr.get(cv2.CAP_PROP_FRAME_COUNT))
    capr.release()
    check("Written clip is decodable and non-empty (a file that exists but "
          "will not play is not evidence)",
          ok and first is not None and nframes > 0, f"frames={nframes}")
    check("Clip frame count matches the metadata", nframes == row.frame_count,
          f"file={nframes} row={row.frame_count}")

    # Browser playability is a demo-critical property, not a nicety: OpenCV
    # silently falls back to mp4v (MPEG-4 Part 2) when the OpenH264 DLL is
    # absent, producing a valid file that Chrome refuses to play. Assert the
    # container really carries H.264 rather than trusting the fallback.
    head = clip.read_bytes()[:4096]
    is_h264 = b"avc1" in head or b"avcC" in head or b"H264" in head
    check("Clip is H.264 in the container — playable in a browser, not just VLC "
          "(mp4v would pass every other check here and still show black in the "
          "dashboard)",
          is_h264, f"codec_recorded={row.codec} avc1_box={b'avc1' in head}")

    # ── CHECKPOINT 4: partial pre-roll ──────────────────────────────────────
    event2 = datetime.utcnow()
    alert2 = make_alert(alert_type="WATCHLIST_FACE_MATCH", when=event2)
    staged2 = stage_buffer("CAM-EV-01", event2, pre_s=8.0,
                           post_s=settings.EVIDENCE_POST_SECONDS)
    check("Buffer staged with only 8s of pre-event history", staged2 > 0, f"frames={staged2}")

    st2 = await ec.capture_evidence(
        (await _ensure_row(ec, db, alert2, "CAM-EV-01")), wait_for_post=False)
    check("Partial pre-roll still completes — no error, no black padding",
          st2 == "COMPLETE", st2)

    row2 = db.query(Evidence).filter(Evidence.alert_id == alert2.id).first()
    db.refresh(row2)
    check("Metadata records the SHORTER actual pre-roll (~8s, not 30s)",
          row2.actual_pre_seconds is not None and 6.0 <= row2.actual_pre_seconds <= 9.5,
          f"actual_pre={row2.actual_pre_seconds}")
    check("Requested window is still recorded alongside it, so the clip is "
          "visibly truncated rather than looking complete",
          row2.requested_pre_seconds == settings.EVIDENCE_PRE_SECONDS,
          f"requested_pre={row2.requested_pre_seconds}")
    pdf2 = (ec.evidence_dir_for(alert2.id) / "custody.pdf").read_bytes()
    check("Custody PDF carries an explicit short-pre-roll NOTE",
          b"Less pre-event footage" in pdf2)

    # ── CHECKPOINT 5: two near-simultaneous CRITICAL alerts ─────────────────
    ev_a = datetime.utcnow()
    a1 = make_alert(when=ev_a)
    a2 = make_alert(when=ev_a)          # same second, same camera
    stage_buffer("CAM-EV-01", ev_a, settings.EVIDENCE_PRE_SECONDS,
                 settings.EVIDENCE_POST_SECONDS)

    id1 = await _ensure_row(ec, db, a1, "CAM-EV-01")
    id2 = await _ensure_row(ec, db, a2, "CAM-EV-01")
    s1, s2 = await asyncio.gather(
        ec.capture_evidence(id1, wait_for_post=False),
        ec.capture_evidence(id2, wait_for_post=False),
    )
    check("Both concurrent captures COMPLETE", s1 == "COMPLETE" and s2 == "COMPLETE",
          f"{s1}/{s2}")
    c1 = ec.evidence_dir_for(a1.id) / "clip.mp4"
    c2 = ec.evidence_dir_for(a2.id) / "clip.mp4"
    check("Two alerts in the same second get SEPARATE folders",
          c1.parent != c2.parent and c1.exists() and c2.exists(),
          f"{c1.parent.name} vs {c2.parent.name}")
    r1 = db.query(Evidence).filter(Evidence.alert_id == a1.id).first()
    r2 = db.query(Evidence).filter(Evidence.alert_id == a2.id).first()
    check("Neither clip is corrupt — both hash to their own stored digest",
          hashlib.sha256(c1.read_bytes()).hexdigest() == r1.sha256
          and hashlib.sha256(c2.read_bytes()).hexdigest() == r2.sha256)

    # ── Idempotency ─────────────────────────────────────────────────────────
    before = db.query(Evidence).count()
    dup = ec.create_pending_row(a1, db, "CAM-EV-01")
    after = db.query(Evidence).count()
    check("Re-triggering capture for an existing alert_id creates no second row",
          dup is None and before == after, f"{before} -> {after}")

    db.refresh(r1)
    hash_before = r1.sha256
    st_again = await ec.capture_evidence(a1.id, wait_for_post=False)
    db.refresh(r1)
    check("Re-running capture on a COMPLETE row is a no-op, not a re-write",
          st_again == "COMPLETE" and r1.sha256 == hash_before)

    # ── Atomic writes: nothing partial is ever left behind ──────────────────
    tmp_root = ec.evidence_root() / ".tmp"
    leftovers = [p for p in tmp_root.iterdir()] if tmp_root.exists() else []
    check("No temp artefacts left behind after successful captures",
          not leftovers, str([p.name for p in leftovers]))
    stray = [p.name for p in ec.evidence_dir_for(a1.id).iterdir()
             if p.suffix not in (".mp4", ".pdf")]
    check("Evidence folder holds only the finalised clip and PDF",
          not stray, str(stray))
    check("Temp dir is INSIDE the evidence root (same filesystem, so "
          "os.replace is genuinely atomic rather than a cross-device copy)",
          str(ec.temp_dir_for(1)).startswith(str(ec.evidence_root())))

    # ── Failure path: no frames for the window ──────────────────────────────
    lonely_event = datetime.utcnow()
    a_none = make_alert(when=lonely_event)
    from backend.services.frame_buffer import get_frame_buffer_registry
    get_frame_buffer_registry().get("CAM-EV-EMPTY").clear()
    id_none = await _ensure_row(ec, db, a_none, "CAM-EV-EMPTY")
    st_none = await ec.capture_evidence(id_none, wait_for_post=False)
    check("Empty buffer yields FAILED with a reason, not a crash",
          st_none == "FAILED", st_none)
    r_none = db.query(Evidence).filter(Evidence.alert_id == a_none.id).first()
    db.refresh(r_none)
    check("FAILED row records why", bool(r_none.failure_reason), r_none.failure_reason)

    # ── Startup sweep of interrupted captures ───────────────────────────────
    a_stuck = make_alert(when=datetime.utcnow())
    stuck_row = Evidence(alert_id=a_stuck.id, camera_str_id="CAM-EV-01",
                         status="PENDING", event_time=datetime.utcnow())
    db.add(stuck_row)
    db.commit()
    swept = ec.sweep_stale_pending()
    db.refresh(stuck_row)
    check("Startup sweep converts an interrupted PENDING row to FAILED "
          "(nothing stays 'processing' forever with no task alive)",
          swept >= 1 and stuck_row.status == "FAILED", f"swept={swept}")

    db.close()

    # ── CHECKPOINT 3: API states ────────────────────────────────────────────
    await _check_api(alert.id, a_stuck.id)

    print(f"\n{'=' * 70}\n{len(PASS)} passed, {len(FAIL)} failed\n{'=' * 70}")
    if FAIL:
        print("FAILED:", FAIL)
        sys.exit(1)


async def _ensure_row(ec, db, alert, camera_str_id) -> int:
    """Create the PENDING row via the real code path; return alert_id."""
    alert.severity = ec.classify_severity(alert.alert_type, alert.iq_contribution,
                                           alert.severity)
    db.commit()
    ec.create_pending_row(alert, db, camera_str_id)
    return alert.id


async def _check_api(complete_alert_id: int, failed_alert_id: int) -> None:
    from starlette.testclient import TestClient

    from backend.auth.routes import pwd_context
    from backend.db.models import Alert, Evidence, User
    from backend.db.session import SessionLocal

    db = SessionLocal()
    u = db.query(User).filter(User.username == "ev_admin").first()
    u.hashed_password = pwd_context.hash("pw")
    # A non-critical alert with no evidence row, to exercise the two
    # distinct 404s.
    plain = Alert(alert_type="LOITERING", subject_label="no evidence",
                  status="new", lifecycle_status="OPEN", created_at=datetime.utcnow())
    db.add(plain)
    pending_alert = Alert(alert_type="WATCHLIST_FACE_MATCH", subject_label="pending",
                          status="new", lifecycle_status="OPEN",
                          created_at=datetime.utcnow())
    db.add(pending_alert)
    db.commit()
    db.refresh(plain)
    db.refresh(pending_alert)
    plain_id, pending_id = plain.id, pending_alert.id
    db.close()

    import backend.main as backend_main

    with TestClient(backend_main.app) as client:
        # The PENDING row is staged INSIDE the client context, i.e. AFTER
        # lifespan startup has run. Creating it beforehand meant startup's
        # sweep_stale_pending() correctly reclassified it to FAILED before
        # the request was ever made — the sweep working as designed, not a
        # bug. A PENDING row only legitimately exists while the process that
        # created it is alive, which is exactly what this now models.
        _db = SessionLocal()
        _db.add(Evidence(alert_id=pending_id, camera_str_id="CAM-EV-01",
                         status="PENDING", event_time=datetime.utcnow()))
        _db.commit()
        _db.close()

        tok = client.post("/api/v1/auth/login",
                          json={"username": "ev_admin", "password": "pw"}).json()["access_token"]
        h = {"Authorization": f"Bearer {tok}"}

        r = client.get(f"/api/v1/evidence/{complete_alert_id}", headers=h)
        check("GET /evidence/{id} returns 200 for a COMPLETE capture",
              r.status_code == 200, str(r.status_code))
        body = r.json()
        check("Response carries hash, both file paths, and status",
              bool(body.get("sha256")) and body.get("clip_path")
              and body.get("pdf_path") and body.get("status") == "COMPLETE",
              str({k: body.get(k) for k in ("status", "clip_path", "pdf_path")}))
        check("Response distinguishes requested vs captured window",
              "requested_window" in body and "captured_window" in body)

        r404 = client.get("/api/v1/evidence/99999999", headers=h)
        check("Unknown alert_id → 404 with a clear message (not a 500)",
              r404.status_code == 404 and "does not exist" in r404.json()["detail"],
              f"{r404.status_code} {r404.text[:90]}")

        r_noev = client.get(f"/api/v1/evidence/{plain_id}", headers=h)
        check("Alert exists but has no evidence → 404 with a DIFFERENT message "
              "(an operator can tell 'no such alert' from 'nothing captured')",
              r_noev.status_code == 404 and "no evidence record" in r_noev.json()["detail"],
              f"{r_noev.status_code} {r_noev.text[:110]}")

        r_pend = client.get(f"/api/v1/evidence/{pending_id}", headers=h)
        check("PENDING evidence → 202 processing, never a crash",
              r_pend.status_code == 202 and r_pend.json()["status"] == "PENDING",
              f"{r_pend.status_code} {r_pend.text[:90]}")

        r_fail = client.get(f"/api/v1/evidence/{failed_alert_id}", headers=h)
        check("FAILED evidence → 200 with status+reason (a recorded failure is "
              "not an endpoint error)",
              r_fail.status_code == 200 and r_fail.json()["status"] == "FAILED",
              f"{r_fail.status_code}")

        r_unauth = client.get(f"/api/v1/evidence/{complete_alert_id}")
        check("Evidence requires authentication",
              r_unauth.status_code in (401, 403), str(r_unauth.status_code))

        r_clip = client.get(f"/api/v1/evidence/{complete_alert_id}/clip", headers=h)
        check("Clip downloads with video/mp4 content type",
              r_clip.status_code == 200
              and r_clip.headers.get("content-type", "").startswith("video/mp4"),
              f"{r_clip.status_code} {r_clip.headers.get('content-type')}")
        check("Downloaded clip bytes hash to the stored digest "
              "(what is served is what was hashed)",
              hashlib.sha256(r_clip.content).hexdigest() == body["sha256"])

        r_pdf = client.get(f"/api/v1/evidence/{complete_alert_id}/custody", headers=h)
        check("Custody PDF downloads as application/pdf",
              r_pdf.status_code == 200
              and r_pdf.headers.get("content-type", "").startswith("application/pdf"),
              f"{r_pdf.status_code} {r_pdf.headers.get('content-type')}")


if __name__ == "__main__":
    asyncio.run(main())
