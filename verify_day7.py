"""
verify_day7.py — Standalone acceptance-criteria check for Day 7.

This bypasses backend/main.py's app object (which imports PipelineBridge ->
main.run_detection_pipeline -> ultralytics/torch) and instead exercises the
Day 7 services directly: face_watchlist_matcher, face_embedder, camera_heartbeat.

EMBEDDER MODE — pinned, not inherited:
  This script originally assumed insightface was NOT installed, so that
  FaceEmbedder would fall back to stub mode on its own. That assumption was
  environment-dependent and eventually false: on a machine WITH insightface
  and the buffalo_l pack present, the real model loads, correctly returns
  None for the flat synthetic "face" crops below (they contain no face), and
  the script crashed on `encode_embedding(None)` — reporting a failure that
  said nothing about Day 7.

  The fix is to pin stub mode explicitly rather than depend on what happens
  to be installed. What this file verifies is the WATCHLIST MATCHING LOGIC
  (threshold behaviour, alert firing, audit, broadcast), which is exactly the
  layer above the embedder and is identical either way. Face-recognition
  ACCURACY is a separate question that a synthetic crop cannot answer at all
  — that belongs to scripts/reid_validate.py --mode face, and remains
  unvalidated pending real labeled pairs (see data/test_face_pairs/README.md).
"""
from __future__ import annotations

import asyncio
import os
import sys
import tempfile
from pathlib import Path

_ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(_ROOT))

# Use a throwaway SQLite DB so we never touch output/sentinel.db
_tmpdir = tempfile.mkdtemp()
os.environ["DATABASE_URL"] = f"sqlite:///{_tmpdir}/verify_day7.db"

import numpy as np

# Pin the embedder to stub mode BEFORE anything resolves the singleton.
# face_watchlist_matcher binds get_face_embedder at import time, so both the
# defining module and that call site are patched — see the module docstring
# for why this is pinned rather than inherited from the environment.
from backend.services import face_embedder as _fe_module
from backend.services.face_embedder import FaceEmbedder as _FaceEmbedder

_stub_embedder = _FaceEmbedder(stub_mode=True)
_fe_module.get_face_embedder = lambda: _stub_embedder  # type: ignore[assignment]

from backend.services import face_watchlist_matcher as _fwm_module  # noqa: E402

_fwm_module.get_face_embedder = lambda: _stub_embedder  # type: ignore[assignment]

# Redirect alert evidence into the scratch dir.
#
# resolve_watchlist_face() saves a snapshot for every match it fires, and the
# module constant points at the REAL output/alerts/watchlist_face/. Without
# this, each run of this script dropped another orphaned, read-only JPEG into
# the live evidence store with no DB row pointing at it — fake evidence
# accumulating in an evidence directory is not an acceptable side effect of
# running a test. (Same fix already applied to verify_part_a.py.)
_fwm_module._ALERT_EVIDENCE_DIR = Path(_tmpdir) / "alerts" / "watchlist_face"
_fwm_module._PROJECT_ROOT = Path(_tmpdir)

# _save_snapshot() returns a path RELATIVE to _PROJECT_ROOT, and the code that
# follows it (hash_file, and this script's own "snapshot_path valid file"
# assertion) resolves that relative path against the process CWD. In
# production those are the same directory. Redirecting the root without also
# moving CWD breaks that invariant and silently nulls evidence_hash — so move
# CWD too, keeping the relative-path contract intact instead of weakening the
# assertions to tolerate a broken one. sys.path and DATABASE_URL are already
# absolute, so nothing else here depends on the original CWD.
os.chdir(_tmpdir)

from backend.db.session import SessionLocal, engine
from backend.db.models import Base, Camera, WatchlistPerson, Alert, AuditLog
from backend.embedding_utils import encode_embedding
from backend.services.face_embedder import get_face_embedder, FACE_EMBEDDING_DIM
from backend.services.face_watchlist_matcher import (
    get_face_watchlist_matcher, resolve_watchlist_face,
)
from backend.services.camera_heartbeat import _run_heartbeat_cycle, check_camera_health
from backend.core.config import settings

PASS, FAIL = [], []

def check(name, cond, detail=""):
    (PASS if cond else FAIL).append(name)
    print(f"{'PASS' if cond else 'FAIL'} — {name}{(': ' + detail) if detail else ''}")


async def main():
    Base.metadata.create_all(bind=engine)
    db = SessionLocal()

    embedder = get_face_embedder()
    print(f"\nFaceEmbedder stub_mode={embedder.is_stub} (expected True — insightface not installed here)\n")

    # ── Seed a watchlist person with a KNOWN embedding ─────────────────────
    known_crop = np.full((128, 64, 3), 77, dtype=np.uint8)   # deterministic "face" crop A
    other_crop = np.full((128, 64, 3), 200, dtype=np.uint8)  # deterministic "face" crop B (different person)

    known_vec = embedder.extract(known_crop)
    wp = WatchlistPerson(
        name="Test Wanted Person", reason="TEST_CASE",
        face_embedding=encode_embedding(known_vec), embedding_dim=FACE_EMBEDDING_DIM,
        reference_photo_path="ref/test.jpg", active=True,
    )
    db.add(wp)
    db.commit()
    db.refresh(wp)

    matcher = get_face_watchlist_matcher()
    n = matcher.reload_watchlist()
    check("reload_watchlist() loads active entries", n == 1, f"loaded={n}")

    # ── Criterion 1-3: matching crop fires alert with correct fields ───────
    result = await resolve_watchlist_face(
        track_db_id=1, camera_str_id="CAM-TEST", camera_db_id=None, crop_bgr=known_crop,
    )
    check("Known face -> WATCHLIST_MATCH decision", result["decision"] == "WATCHLIST_MATCH", str(result))

    alert = db.query(Alert).filter(Alert.alert_type == "WATCHLIST_FACE_MATCH").first()
    check("Alert row created", alert is not None)
    if alert:
        check("subject_label set", bool(alert.subject_label), alert.subject_label)
        check("confidence matches similarity (>=threshold)", alert.confidence >= settings.WATCHLIST_FACE_MATCH_THRESHOLD, str(alert.confidence))
        check("evidence_hash non-null", alert.evidence_hash is not None)
        check("snapshot_path valid file", alert.snapshot_path and Path(alert.snapshot_path).exists())

    audit = db.query(AuditLog).filter(AuditLog.action == "WATCHLIST_FACE_MATCH_FIRED").first()
    check("AuditLog WATCHLIST_FACE_MATCH_FIRED row exists", audit is not None)
    if audit:
        import json
        details = json.loads(audit.details) if isinstance(audit.details, str) else audit.details
        check("Audit details has watchlist_person_id + similarity",
              details.get("watchlist_person_id") == wp.id and "similarity" in details, str(details))

    # ── Criterion 4: non-watchlist face -> no alert ─────────────────────────
    before = db.query(Alert).count()
    result2 = await resolve_watchlist_face(
        track_db_id=2, camera_str_id="CAM-TEST", camera_db_id=None, crop_bgr=other_crop,
    )
    after = db.query(Alert).count()
    check("Non-watchlist face -> NO_MATCH, no new alert", result2["decision"] == "NO_MATCH" and after == before, f"{result2}, before={before} after={after}")

    # ── Criterion 5: mark-false-positive preserves the alert ───────────────
    alert.status = "false_positive"
    alert.false_positive_reason = "Confirmed misidentification in test"
    from backend.services.audit_logger import log_audit
    log_audit(db, user=None, action="ALERT_MARKED_FALSE_POSITIVE", resource_type="alert",
               resource_id=alert.id, details={"reason": alert.false_positive_reason})
    db.commit()
    still_there = db.query(Alert).filter(Alert.id == alert.id).first()
    check("False-positive mark preserves the alert row (not deleted)",
          still_there is not None and still_there.false_positive_reason is not None)
    fp_audit = db.query(AuditLog).filter(AuditLog.action == "ALERT_MARKED_FALSE_POSITIVE").first()
    check("ALERT_MARKED_FALSE_POSITIVE audit row logged", fp_audit is not None)

    # ── Criterion 6: no face detected doesn't crash the pipeline ───────────
    # (stub mode never returns None, so we call extract() bypass logic directly
    #  by monkeypatching extract() to simulate the "no face" branch.)
    orig_extract = embedder.extract
    embedder.extract = lambda crop: None
    try:
        result3 = await resolve_watchlist_face(
            track_db_id=3, camera_str_id="CAM-TEST", camera_db_id=None, crop_bgr=known_crop,
        )
        check("No face detected -> NO_FACE decision, no crash", result3["decision"] == "NO_FACE", str(result3))
    finally:
        embedder.extract = orig_extract

    # ── Criterion 7-10: camera heartbeat ────────────────────────────────────
    feed_path = Path(_tmpdir) / "camera1_feed.mp4"
    feed_path.write_bytes(b"fake video bytes")  # non-empty file = "healthy"

    cam = Camera(name="Test Camera 1", ip_address=str(feed_path), status="OFFLINE")
    db.add(cam)
    db.commit()
    db.refresh(cam)

    healthy = await check_camera_health(cam)
    check("check_camera_health() true for existing non-empty file", healthy is True)

    await _run_heartbeat_cycle()
    db.refresh(cam)
    check("Camera flips OFFLINE -> ONLINE on healthy check", cam.status == "ONLINE", cam.status)
    check("last_heartbeat_at populated", cam.last_heartbeat_at is not None)

    online_audit_count_before = db.query(AuditLog).filter(AuditLog.action == "CAMERA_CAME_ONLINE").count()

    # Kill the feed (delete file) -> should require FAILURE_THRESHOLD consecutive fails
    feed_path.unlink()
    for i in range(settings.CAMERA_HEARTBEAT_FAILURE_THRESHOLD):
        await _run_heartbeat_cycle()
        db.refresh(cam)
        if i < settings.CAMERA_HEARTBEAT_FAILURE_THRESHOLD - 1:
            check(f"Debounce: still ONLINE after {i+1} failure(s) (threshold={settings.CAMERA_HEARTBEAT_FAILURE_THRESHOLD})",
                  cam.status == "ONLINE", cam.status)
    check(f"Camera flips to OFFLINE after {settings.CAMERA_HEARTBEAT_FAILURE_THRESHOLD} consecutive failures",
          cam.status == "OFFLINE", cam.status)

    offline_audit = db.query(AuditLog).filter(AuditLog.action == "CAMERA_WENT_OFFLINE").first()
    check("CAMERA_WENT_OFFLINE audit row logged on transition", offline_audit is not None)

    # ── Criterion 10 (no-noise): a cycle with no status change logs nothing new
    before_count = db.query(AuditLog).count()
    await _run_heartbeat_cycle()  # still offline, no file -> no NEW transition
    after_count = db.query(AuditLog).count()
    check("No new audit row when status doesn't change (no noise)", after_count == before_count, f"{before_count} -> {after_count}")

    # ── Criterion 9: concurrency (not a strict timing assertion here, just a
    #    structural check that check_camera_health is awaited via gather) ───
    import inspect
    src = inspect.getsource(check_camera_health)
    check("check_camera_health offloads blocking I/O via executor", "run_in_executor" in src)

    db.close()

    print(f"\n{'='*60}\n{len(PASS)} passed, {len(FAIL)} failed\n{'='*60}")
    if FAIL:
        print("FAILED:", FAIL)
        sys.exit(1)


if __name__ == "__main__":
    asyncio.run(main())
