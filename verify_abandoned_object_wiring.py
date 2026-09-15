"""
verify_abandoned_object_wiring.py — proof that the abandoned-object detector
is now actually reachable from the live pipeline.

WHY THIS EXISTS
───────────────
Before this fix: backend/main.py constructed AbandonedObjectDetector at
startup and logged something that read like activation, but nothing in
connection_manager.py or the detection loop ever called
on_object_update()/on_person_positions(). Every prior "green" test of this
detector (tests/abandoned_object_eval, verify_detector_regression.py) called
those methods directly — a legitimate unit test of the detector's logic, but
none of them could have caught "nothing wires this up in production", because
they never went near the wiring.

This script contains ZERO direct calls to on_object_update() or
on_person_positions(). An ABANDONED_OBJECT alert here can only appear if the
whole chain executed on its own:

    EventEmitter.emit() / emit_objects()   (real — person + object streams)
      → _BridgedQueue.put_nowait()          (real)
      → loop.call_soon_threadsafe()         (real — thread → asyncio hand-off)
      → PipelineBridge._drain_loop()        (real — routes by event_type)
      → ConnectionManager.handle_object_event()   (real — the thing that was
                                                     missing before Day 10)
      → proximity computed from real person positions in current_state
      → AbandonedObjectDetector.on_person_positions() / on_object_update()
      → Alert row + SENTINEL IQ contribution + audit log

WHAT IS SUBSTITUTED, AND WHY THAT IS HONEST
────────────────────────────────────────────
Only YOLO inference. run_detection_pipeline() is replaced with a stand-in
that feeds pre-made ConfirmedTrack objects — one person track, one suitcase
track — into the REAL EventEmitter, on a REAL producer thread, through the
REAL bridge. Everything from EventEmitter onward is production code.

REQUIRES A REAL REDIS at settings.REDIS_URL (default redis://localhost:6379/0)
— the detector's state layer refuses to fake this out, same posture as
tests/abandoned_object_eval. This script checks connectivity up front and
fails loudly, rather than silently passing on SKIPPED_NO_REDIS.

Run:  python verify_abandoned_object_wiring.py
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

_tmpdir = tempfile.mkdtemp(prefix="verify_abandoned_wiring_")
os.environ["DATABASE_URL"] = f"sqlite:///{_tmpdir}/verify_abandoned_wiring.db"

import numpy as np  # noqa: E402

PASS: list[str] = []
FAIL: list[str] = []


def check(name: str, cond: bool, detail: str = "") -> None:
    (PASS if cond else FAIL).append(name)
    print(f"{'PASS' if cond else 'FAIL'} — {name}{(': ' + detail) if detail else ''}")


CAMERA_STR_ID = "CAM-ABANDON-WIRING"
PERSON_TRACK_ID = 501
OBJECT_TRACK_ID_PIPELINE = 601   # pipeline-local id; connection_manager makes
                                  # "CAM-ABANDON-WIRING:obj:601" from this

# Object stays put the whole time (drift ~0, well under
# STATIC_MOVEMENT_PX_THRESHOLD). Person starts right next to it, then leaves.
OBJECT_BBOX = [480.0, 480.0, 520.0, 520.0]     # center (500, 500)
PERSON_BBOX_NEAR = [490.0, 490.0, 510.0, 510.0]  # center (500, 500) — inside radius


def _make_frame() -> np.ndarray:
    rng = np.random.default_rng(99)
    return rng.integers(0, 255, size=(480, 640, 3), dtype=np.uint8)


async def main() -> None:
    from backend.db.models import Alert, AuditLog, Base, Camera
    from backend.db.session import SessionLocal, engine
    from backend.services.state import get_behavior_state

    Base.metadata.create_all(bind=engine)
    db = SessionLocal()

    # ── Redis reachability — required, not faked ────────────────────────────
    redis_ok = await get_behavior_state().ping()
    check("Real Redis reachable (this script does not fake the state backend)",
          redis_ok)
    if not redis_ok:
        print("\n[FATAL] No Redis at settings.REDIS_URL — cannot exercise the "
              "real abandoned-object state machine. Start one with "
              "`docker run -p 6379:6379 redis` and retry.", file=sys.stderr)
        sys.exit(1)

    # Clear this script's own Redis keys before running.
    #
    # Same idempotency bug already fixed in tests/behavior_eval/eval_runner.py
    # and tests/abandoned_object_eval/eval_runner.py: this script uses a
    # fixed camera_id/object_track_id every run, and on_object_update()'s
    # "already fired" debounce is designed to survive across ticks — which
    # also means it survives across RUNS of this script. A second run finds
    # Redis already saying status=alert_fired from the previous run and the
    # detector correctly refuses to fire again, which looks exactly like the
    # wiring being broken. Scoped to this script's own CAMERA_STR_ID, never
    # a FLUSHDB.
    cleared = await get_behavior_state().delete_matching(f"*{CAMERA_STR_ID}*")
    print(f"Cleared {cleared} leftover key(s) from a previous run.\n")

    # ── Setup: a registered camera (so Alert.camera_id resolves, not NULL) ──
    camera = Camera(camera_id=CAMERA_STR_ID, name="Abandoned-Object Wiring Test",
                    zone="Central", status="ONLINE")
    db.add(camera)
    db.commit()
    db.refresh(camera)

    from backend.services.camera_calibration import get_calibration_cache
    get_calibration_cache().reload()   # no row for this camera → uncalibrated fallback

    # Pin the night multiplier OFF — the IQ assertion below expects the bare
    # base score of 5.0, and the night window (20:00-06:00 IST) would
    # otherwise make that 6.5 depending purely on what time the suite runs.
    from backend.services.sentinel_iq import set_night_override
    set_night_override(False)

    # ── Readiness flags — ONLY abandoned-object, to keep this test focused ──
    import backend.connection_manager as cm_module
    from backend.connection_manager import ConnectionManager

    cm_module.set_reid_ready()   # readiness flag only — camera identity is
                                 # per-ConnectionManager now, passed below.
    cm_module.set_abandoned_ready()

    manager = ConnectionManager(
        camera_id=CAMERA_STR_ID, track_expiry_seconds=5.0,
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
            "object_events_file": f"{_tmpdir}/object_events.jsonl",
            "crops_dir": f"{_tmpdir}/crops",
            "crop_save_interval_frames": 15,
        }
    }

    # Timeline: person present (within OWNERSHIP_RADIUS_PX) at t=0,2,4, then
    # leaves. Object present and static the whole time.
    #
    # first_seen_time resets on every tick where a person is "nearby" — which
    # requires BOTH bbox proximity AND a recent report (within
    # ABANDON_PROXIMITY_WINDOW_SEC, default 2.0, of THIS tick's video_time —
    # see handle_object_event). The person's last real report is t=4, so:
    #   t=6:  |6-4|=2  <= 2.0  → still "nearby" → first_seen_time resets to 6
    #   t=10: |10-4|=6 >  2.0  → no longer nearby → first_seen_time stays 6
    # From there span_sec = video_time - 6 needs to reach ABANDON_DURATION_SEC
    # (60.0 default) → drive past video_time=66. Final tick at 68 →
    # span_sec = 68 - 6 = 62.
    PERSON_TICKS = [0.0, 2.0, 4.0]
    OBJECT_TICKS = [0.0, 2.0, 4.0, 6.0, 10.0, 20.0, 30.0, 40.0, 50.0, 60.0, 68.0]
    EXPECTED_FIRST_SEEN_TIME = 6.0
    EXPECTED_SPAN_SEC = 68.0 - EXPECTED_FIRST_SEEN_TIME

    def _stand_in_pipeline(cfg, source, event_queue, stop_event, display, save_video):
        emitter = EventEmitter(cfg, event_queue=event_queue, camera_id=CAMERA_STR_ID)
        frame = _make_frame()
        try:
            frame_number = 0
            for vt in OBJECT_TICKS:
                if vt in PERSON_TICKS:
                    emitter.emit(
                        [ConfirmedTrack(
                            track_id=PERSON_TRACK_ID, bbox=list(PERSON_BBOX_NEAR),
                            confidence=0.90, frame_number=frame_number, timestamp=vt,
                        )],
                        frame,
                    )
                emitter.emit_objects(
                    [ConfirmedTrack(
                        track_id=OBJECT_TRACK_ID_PIPELINE, bbox=list(OBJECT_BBOX),
                        confidence=0.88, frame_number=frame_number, timestamp=vt,
                    )],
                    object_classes={OBJECT_TRACK_ID_PIPELINE: "suitcase"},
                    # Explicitly not baseline — this test is about wiring
                    # reachability, not main.py's warmup heuristic (which is
                    # simple enough to read directly; see config.yaml's
                    # abandoned_object.warmup_sec comment for what it does).
                    object_baseline={OBJECT_TRACK_ID_PIPELINE: False},
                )
                frame_number += 1
        finally:
            emitter.close()

    original_pipeline = pb_module.run_detection_pipeline
    pb_module.run_detection_pipeline = _stand_in_pipeline
    try:
        bridge = pb_module.PipelineBridge(cfg=cfg, source="verify", connection_manager=manager)
        bridge.start(asyncio.get_running_loop())

        for _ in range(80):
            await asyncio.sleep(0.1)
            if db.query(Alert).filter(Alert.alert_type == "ABANDONED_OBJECT").first():
                break
        await asyncio.sleep(0.5)
        await bridge.stop()
    finally:
        pb_module.run_detection_pipeline = original_pipeline

    # ── 1. Object events actually reached connection_manager's cache ────────
    expected_object_id = f"{CAMERA_STR_ID}:obj:{OBJECT_TRACK_ID_PIPELINE}"
    check("Object position cache holds the tracked suitcase "
          "(handle_object_event ran, not skipped)",
          expected_object_id in manager._object_positions,
          str(list(manager._object_positions.keys())))

    # ── 2. The object JSONL stream got the additive fields, person stream untouched ──
    obj_lines = [
        json.loads(l) for l in
        Path(cfg["output"]["object_events_file"]).read_text().splitlines() if l.strip()
    ]
    check("Object events written to the SEPARATE object_events_file",
          len(obj_lines) == len(OBJECT_TICKS), f"lines={len(obj_lines)}")
    check("Object event carries object_class and is_baseline additively",
          obj_lines and obj_lines[0].get("object_class") == "suitcase"
          and obj_lines[0].get("is_baseline") is False,
          str(obj_lines[0]) if obj_lines else "")

    person_lines = [
        json.loads(l) for l in
        Path(cfg["output"]["events_file"]).read_text().splitlines() if l.strip()
    ]
    check("Person events file untouched — exactly the 7 contract fields, "
          "no object-only keys leaked in",
          all(set(l.keys()) == {"event_type", "camera_id", "track_id", "frame_number",
                                 "timestamp", "bbox", "confidence"} for l in person_lines),
          str(sorted(person_lines[0].keys())) if person_lines else "")

    # ── 3. THE PROOF: alert fired without any test code calling the detector ──
    alerts = db.query(Alert).filter(Alert.alert_type == "ABANDONED_OBJECT").all()
    check("ABANDONED_OBJECT alert fired through the real dispatch chain "
          "(no test code called on_object_update/on_person_positions)",
          len(alerts) == 1, f"alerts={len(alerts)}")

    if alerts:
        alert = alerts[0]
        check("Alert is bound to the resolved camera PK (not NULL)",
              alert.camera_id == camera.id, f"camera_id={alert.camera_id}")
        meta = json.loads(alert.meta_json) if alert.meta_json else {}
        check("Alert metadata records the correct object_track_id "
              "(built as {camera}:obj:{pipeline_track_id})",
              meta.get("object_track_id") == expected_object_id,
              str(meta))
        check("Alert metadata records object_class='suitcase'",
              meta.get("object_class") == "suitcase", str(meta))

        # ── 4. SENTINEL IQ scored it through the real path too ──────────────
        check("SENTINEL IQ scored the alert at fire time (base 5.0, no "
              "multipliers expected here — daytime clock, no cross-zone "
              "journey for an object)",
              alert.iq_contribution is not None and abs(alert.iq_contribution - 5.0) < 1e-9,
              str(alert.iq_contribution))

    audit = db.query(AuditLog).filter(
        AuditLog.action == "ABANDONED_OBJECT_ALERT_FIRED"
    ).all()
    check("Audit log row written for the fired alert", len(audit) == 1, f"rows={len(audit)}")

    # ── 5. Proximity suppression actually happened, correctly bounded by
    #      video_time (not wall-clock) — the alert fired with the span_sec
    #      that only comes out right if first_seen_time reset all the way to
    #      t=6 (the last tick within ABANDON_PROXIMITY_WINDOW_SEC of the
    #      person's true last sighting at t=4) and NOT to t=0 (proximity
    #      suppression did nothing) or kept resetting past t=6 (the stale
    #      wall-clock-based bug this test caught during development — see
    #      handle_object_event's proximity computation comment) ─────────────
    if alerts:
        meta = json.loads(alerts[0].meta_json)
        check(f"Fired with span_sec={EXPECTED_SPAN_SEC:g} — the only value "
              "consistent with first_seen_time resetting to t=6 exactly "
              "(video_time-based proximity aging, not wall-clock)",
              meta.get("span_sec") == EXPECTED_SPAN_SEC, str(meta.get("span_sec")))
        check("Fired at video_time=68 (the tick that reaches "
              "ABANDON_DURATION_SEC from first_seen_time=6)",
              meta.get("video_time") == 68.0, str(meta.get("video_time")))

    db.close()

    print(f"\n{'=' * 68}\n{len(PASS)} passed, {len(FAIL)} failed\n{'=' * 68}")
    if FAIL:
        print("FAILED:", FAIL)
        sys.exit(1)


if __name__ == "__main__":
    asyncio.run(main())
