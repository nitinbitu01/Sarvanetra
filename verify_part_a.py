"""
verify_part_a.py — Proof that the pipeline actually wires together end to end.

WHY THIS SCRIPT EXISTS, AND WHY IT IS NOT verify_day7.py
────────────────────────────────────────────────────────
verify_day7.py validated the face-watchlist matcher by calling
resolve_watchlist_face() directly. That was a legitimate unit-level test of
the matcher, but it could not have caught the actual defect: nothing in the
running system ever created a `tracks` DB row, so `event.get("db_track_id")`
was always None, so the dispatch gate in connection_manager.py that would
have called resolve_watchlist_face() in production could never evaluate
True. The matcher worked; the wiring did not exist.

So this script contains ZERO direct calls to resolve_watchlist_face() or to
resolve_identity(). A watchlist alert here can only appear if the whole chain
executed on its own:

    EventEmitter.emit()                 (real — attaches crop_bgr)
      → _BridgedQueue.put_nowait()      (real)
      → loop.call_soon_threadsafe()     (real — thread → asyncio hand-off)
      → PipelineBridge._drain_loop()    (real)
      → ConnectionManager
            .handle_detection_event()   (real — creates the tracks row,
                                          sets db_track_id/track_confirmed)
      → the Day 7 dispatch gate         (real — the thing that was dead)
      → resolve_watchlist_face()        (real — reached, not called)
      → Alert row + WebSocket broadcast (real)

WHAT IS SUBSTITUTED, AND WHY THAT IS HONEST
───────────────────────────────────────────
Two things, neither of which is the thing under test:

  1. YOLOv8 inference. run_detection_pipeline() is replaced with a stand-in
     that feeds pre-made ConfirmedTrack instances into a REAL EventEmitter,
     on a REAL producer thread, through the REAL bridge. Everything from
     EventEmitter onward is production code. What YOLO would contribute is
     the bounding boxes, and this test is not about box quality.

  2. The InsightFace model. The embedder is forced into its documented stub
     mode (deterministic hash-derived vectors) and the watchlist entry is
     seeded with the exact vector that stub produces for the test crop, so a
     match is guaranteed to be reachable. This proves the DISPATCH PATH
     fires, not that face recognition is accurate — accuracy is Part B's
     job, and every alert this produces carries is_stub=true in its metadata
     precisely so the two are never confused.

Run:  python verify_part_a.py
"""
from __future__ import annotations

import asyncio
import json
import os
import sys
import tempfile
from pathlib import Path

_ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(_ROOT))

_tmpdir = tempfile.mkdtemp(prefix="verify_part_a_")
os.environ["DATABASE_URL"] = f"sqlite:///{_tmpdir}/verify_part_a.db"

import numpy as np  # noqa: E402

PASS: list[str] = []
FAIL: list[str] = []


def check(name: str, cond: bool, detail: str = "") -> None:
    (PASS if cond else FAIL).append(name)
    print(f"{'PASS' if cond else 'FAIL'} — {name}{(': ' + detail) if detail else ''}")


CAMERA_STR_ID = "CAM-01"
TRACK_ID = 7701
BBOX = [100.0, 80.0, 220.0, 400.0]


def _make_frame() -> np.ndarray:
    """A deterministic BGR frame. Content is irrelevant — the stub embedder
    hashes whatever bytes it is given, and the real crop-extraction code
    still runs over it for real."""
    rng = np.random.default_rng(1234)
    return rng.integers(0, 255, size=(480, 640, 3), dtype=np.uint8)


async def main() -> None:
    from backend.db.models import Base, Alert, AuditLog, Camera, Track, WatchlistPerson
    from backend.db.session import SessionLocal, engine
    from backend.embedding_utils import encode_embedding

    Base.metadata.create_all(bind=engine)
    db = SessionLocal()

    # Pin the SENTINEL IQ night multiplier OFF for this run.
    #
    # Without this the IQ assertions below are time-dependent: the night
    # window is 20:00-06:00 IST, so running the suite during an Indian
    # evening silently turned the expected 7.0 into 9.1 (7.0 x 1.3) and the
    # test "failed" while the engine was behaving perfectly. A test whose
    # result depends on what time it is run is not a test. Night behaviour
    # is covered explicitly, with the override pinned ON, in verify_part_c.py.
    from backend.services.sentinel_iq import set_night_override
    set_night_override(False)

    # ── Setup: a registered camera ──────────────────────────────────────────
    camera = Camera(camera_id=CAMERA_STR_ID, name="Verify Camera",
                    zone="Central", status="ONLINE")
    db.add(camera)
    db.commit()
    db.refresh(camera)

    # ── Setup: force the face embedder into documented stub mode ────────────
    from backend.services import face_embedder as fe_module
    from backend.services import face_watchlist_matcher as fwm_module
    from backend.services.face_embedder import FaceEmbedder

    # Redirect alert evidence into the scratch dir. resolve_watchlist_face()
    # saves a snapshot for every match it fires, and the module constant
    # points at the REAL output/alerts/watchlist_face/ — so without this, each
    # run of this script drops another orphaned JPEG (read-only, by design)
    # into the live evidence store, with no DB row left pointing at it. Fake
    # evidence accumulating in an evidence directory is not an acceptable
    # side effect of running a test.
    fwm_module._ALERT_EVIDENCE_DIR = Path(_tmpdir) / "alerts" / "watchlist_face"
    fwm_module._PROJECT_ROOT = Path(_tmpdir)

    stub_embedder = FaceEmbedder(stub_mode=True)
    # face_watchlist_matcher.py binds `get_face_embedder` at import time, so
    # patching only the defining module would not reach the call site inside
    # resolve_watchlist_face(). Patch both. Nothing else about the matcher is
    # touched — this swaps the MODEL, not the dispatch path under test.
    fe_module.get_face_embedder = lambda: stub_embedder      # type: ignore[assignment]
    fwm_module.get_face_embedder = lambda: stub_embedder     # type: ignore[assignment]
    check("Face embedder is in stub mode for this run (accuracy NOT under test here)",
          stub_embedder.is_stub)

    # ── Setup: seed the watchlist with the exact vector this crop produces ──
    frame = _make_frame()
    x1, y1, x2, y2 = (int(v) for v in BBOX)
    expected_crop = frame[y1:y2, x1:x2].copy()
    target_embedding = stub_embedder.extract(expected_crop)
    check("Stub embedder returned an embedding for the test crop",
          target_embedding is not None)

    db.add(WatchlistPerson(
        name="VERIFY SUBJECT",
        reason="synthetic entry for pipeline wiring verification",
        face_embedding=encode_embedding(target_embedding),
        embedding_dim=int(target_embedding.shape[0]),
        reference_photo_path=None,
        active=True,
    ))
    db.commit()

    from backend.services.face_watchlist_matcher import get_face_watchlist_matcher
    loaded = get_face_watchlist_matcher().reload_watchlist()
    check("Watchlist loaded with the seeded entry", loaded == 1, f"entries={loaded}")

    # ── Setup: readiness flags, exactly as backend/main.py sets them ────────
    import backend.connection_manager as cm_module
    from backend.connection_manager import ConnectionManager

    # Camera identity is now per-ConnectionManager, not a module global —
    # set_reid_ready() only flips the readiness flag.
    cm_module.set_reid_ready()
    cm_module.set_face_watchlist_ready()
    # Behavior engine deliberately left NOT ready — Redis is not running in
    # this environment, and the Day 8 branch is not what Part A is about.

    manager = ConnectionManager(
        camera_id=CAMERA_STR_ID,
        track_expiry_seconds=2.0,
        track_update_interval_frames=15,
        camera_db_id=camera.id,
    )

    # ── The real bridge, with only YOLO inference stood in for ──────────────
    import backend.pipeline_bridge as pb_module
    from event_emitter import EventEmitter
    from tracker import ConfirmedTrack

    cfg = {
        "output": {
            "events_file": f"{_tmpdir}/events.jsonl",
            "crops_dir": f"{_tmpdir}/crops",
            "crop_save_interval_frames": 15,
        }
    }

    def _stand_in_pipeline(cfg, source, event_queue, stop_event, display, save_video):
        """Stands in for run_detection_pipeline. Uses the REAL EventEmitter
        against the REAL bridged queue — only the YOLO box source differs."""
        emitter = EventEmitter(cfg, event_queue=event_queue, camera_id=CAMERA_STR_ID)
        try:
            # Three frames for one track: frame 1 is the track's first
            # sighting (crop always saved), frames 2-3 exercise the
            # cached-PK / throttled-refresh path.
            for frame_number in (1, 2, 3):
                emitter.emit(
                    [ConfirmedTrack(
                        track_id=TRACK_ID, bbox=list(BBOX), confidence=0.91,
                        frame_number=frame_number, timestamp=float(frame_number) / 5.0,
                    )],
                    frame,
                )
        finally:
            emitter.close()

    original_pipeline = pb_module.run_detection_pipeline
    pb_module.run_detection_pipeline = _stand_in_pipeline
    try:
        bridge = pb_module.PipelineBridge(cfg=cfg, source="verify", connection_manager=manager)
        bridge.start(asyncio.get_running_loop())

        # Let the producer thread, the drain loop, and the fire-and-forget
        # dispatch tasks all complete.
        for _ in range(60):
            await asyncio.sleep(0.1)
            if db.query(Alert).filter(Alert.alert_type == "WATCHLIST_FACE_MATCH").first():
                break
        await asyncio.sleep(0.5)
        await bridge.stop()
    finally:
        pb_module.run_detection_pipeline = original_pipeline

    # ── 1. The taproot: a tracks row now exists ─────────────────────────────
    track_rows = db.query(Track).filter(
        Track.camera_id == CAMERA_STR_ID, Track.track_id == TRACK_ID
    ).all()
    check("A `tracks` DB row was created by the live pipeline "
          "(nothing in the system did this before)",
          len(track_rows) == 1, f"rows={len(track_rows)}")

    if track_rows:
        row = track_rows[0]
        check("tracks row has first_seen_frame populated", row.first_seen_frame == 1,
              str(row.first_seen_frame))
        check("tracks row has best_confidence populated",
              row.best_confidence is not None and abs(row.best_confidence - 0.91) < 1e-6,
              str(row.best_confidence))

    # ── 2. The JSONL contract is untouched by the crop enrichment ───────────
    jsonl_lines = [
        json.loads(l) for l in Path(cfg["output"]["events_file"]).read_text().splitlines() if l.strip()
    ]
    check("JSONL still written (one line per emitted event)", len(jsonl_lines) == 3,
          f"lines={len(jsonl_lines)}")
    expected_fields = {"event_type", "camera_id", "track_id", "frame_number",
                       "timestamp", "bbox", "confidence"}
    check("JSONL lines carry EXACTLY the seven contract fields — no crop_bgr, "
          "no db_track_id leaked into the file",
          all(set(l.keys()) == expected_fields for l in jsonl_lines),
          str(sorted(jsonl_lines[0].keys())) if jsonl_lines else "")

    # ── 3. THE PROOF: the alert fired without any test code calling the matcher ──
    alerts = db.query(Alert).filter(Alert.alert_type == "WATCHLIST_FACE_MATCH").all()
    check("WATCHLIST_FACE_MATCH alert fired through the real dispatch chain "
          "(no test code called resolve_watchlist_face)",
          len(alerts) == 1, f"alerts={len(alerts)}")

    if alerts:
        alert = alerts[0]
        check("Alert is bound to the resolved camera PK (not NULL)",
              alert.camera_id == camera.id, f"camera_id={alert.camera_id}")
        check("Alert.track_id is the `tracks` PK produced by the new persistence step",
              track_rows and alert.track_id == track_rows[0].id,
              f"alert.track_id={alert.track_id}")
        meta = json.loads(alert.meta_json) if alert.meta_json else {}
        check("Alert metadata honestly records is_stub=true for the stub embedder",
              meta.get("is_stub") is True, str(meta))

        # Part C rides on Part A: the IQ engine ran inside the real firing
        # path, not in a test harness. Base 7.0, daytime, no cross-zone
        # journey — so no multiplier applies and the total is the base.
        check("SENTINEL IQ scored this alert at fire time through the real path",
              alert.iq_contribution is not None and abs(alert.iq_contribution - 7.0) < 1e-9,
              str(alert.iq_contribution))
        breakdown = json.loads(alert.iq_breakdown_json) if alert.iq_breakdown_json else {}
        check("Stored IQ breakdown is traceable back to base + named multipliers",
              breakdown.get("base_score") == 7.0
              and {m["name"] for m in breakdown.get("multipliers", [])}
                  == {"Night", "Cross-zone travel"},
              str(breakdown))

    # ── 4. Fire-once semantics held across the three events ─────────────────
    check("Exactly one alert for three events — dispatch-once bookkeeping held",
          len(alerts) == 1, f"alerts={len(alerts)}")

    audit = db.query(AuditLog).filter(
        AuditLog.action == "WATCHLIST_FACE_MATCH_FIRED"
    ).all()
    check("Audit log row written for the fired match", len(audit) == 1, f"rows={len(audit)}")

    db.close()

    print(f"\n{'=' * 68}\n{len(PASS)} passed, {len(FAIL)} failed\n{'=' * 68}")
    if FAIL:
        print("FAILED:", FAIL)
        sys.exit(1)


if __name__ == "__main__":
    asyncio.run(main())
