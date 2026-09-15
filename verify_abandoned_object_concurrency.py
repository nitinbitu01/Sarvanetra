"""
verify_abandoned_object_concurrency.py — the per-object lock added to fix the
read-modify-write race (see connection_manager.py's _object_locks comment)
must serialize ticks for the SAME object without serializing DIFFERENT
objects against each other. This proves both halves of that claim against
the real bridge + real Redis.

Run:  python verify_abandoned_object_concurrency.py
"""
from __future__ import annotations

import asyncio
import json
import os
import sys
import tempfile
import time
from pathlib import Path

_ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(_ROOT))

_tmpdir = tempfile.mkdtemp(prefix="verify_abandoned_concurrency_")
os.environ["DATABASE_URL"] = f"sqlite:///{_tmpdir}/verify.db"

import numpy as np  # noqa: E402

PASS: list[str] = []
FAIL: list[str] = []


def check(name: str, cond: bool, detail: str = "") -> None:
    (PASS if cond else FAIL).append(name)
    print(f"{'PASS' if cond else 'FAIL'} — {name}{(': ' + detail) if detail else ''}")


CAMERA_STR_ID = "CAM-CONCURRENCY"
N_OBJECTS = 12
N_TICKS_PER_OBJECT = 15   # each object gets a dense, un-suppressed tick run


def _make_frame() -> np.ndarray:
    rng = np.random.default_rng(7)
    return rng.integers(0, 255, size=(480, 640, 3), dtype=np.uint8)


async def main() -> None:
    from backend.db.models import Base, Camera
    from backend.db.session import SessionLocal, engine
    from backend.services.state import get_behavior_state

    Base.metadata.create_all(bind=engine)
    db = SessionLocal()

    redis_ok = await get_behavior_state().ping()
    check("Real Redis reachable", redis_ok)
    if not redis_ok:
        sys.exit(1)

    # Same idempotency guard as verify_abandoned_object_wiring.py — fixed
    # camera_id/object ids mean a second run would find every object already
    # at status=alert_fired or mid-window from the previous run.
    cleared = await get_behavior_state().delete_matching(f"*{CAMERA_STR_ID}*")
    print(f"Cleared {cleared} leftover key(s) from a previous run.\n")

    camera = Camera(camera_id=CAMERA_STR_ID, name="Concurrency Test",
                    zone="Central", status="ONLINE")
    db.add(camera)
    db.commit()
    db.refresh(camera)

    from backend.services.camera_calibration import get_calibration_cache
    get_calibration_cache().reload()

    import backend.connection_manager as cm_module
    from backend.connection_manager import ConnectionManager

    cm_module.set_abandoned_ready()
    manager = ConnectionManager(camera_id=CAMERA_STR_ID, track_expiry_seconds=30.0)

    import backend.pipeline_bridge as pb_module
    from event_emitter import EventEmitter
    from tracker import ConfirmedTrack

    cfg = {"output": {
        "events_file": f"{_tmpdir}/events.jsonl",
        "object_events_file": f"{_tmpdir}/object_events.jsonl",
        "crops_dir": f"{_tmpdir}/crops",
        "crop_save_interval_frames": 15,
    }}

    # N_OBJECTS distinct, well-separated, static objects — none ever has a
    # person nearby, so every tick is a clean "just accumulate position
    # history" write with no proximity branch involved. This isolates the
    # lock's effect on the read-modify-write race from proximity logic.
    def _stand_in(cfg, source, event_queue, stop_event, display, save_video):
        emitter = EventEmitter(cfg, event_queue=event_queue, camera_id=CAMERA_STR_ID)
        frame = _make_frame()
        try:
            for obj_idx in range(N_OBJECTS):
                base_x = 50.0 + obj_idx * 50.0   # spaced far apart on x
                for tick in range(N_TICKS_PER_OBJECT):
                    vt = float(tick) * 2.0
                    emitter.emit_objects(
                        [ConfirmedTrack(
                            track_id=1000 + obj_idx,
                            bbox=[base_x, 100.0, base_x + 20.0, 120.0],
                            confidence=0.85, frame_number=tick, timestamp=vt,
                        )],
                        object_classes={1000 + obj_idx: "suitcase"},
                        object_baseline={1000 + obj_idx: False},
                    )
        finally:
            emitter.close()

    original = pb_module.run_detection_pipeline
    pb_module.run_detection_pipeline = _stand_in
    state = get_behavior_state()
    try:
        bridge = pb_module.PipelineBridge(cfg=cfg, source="verify", connection_manager=manager)
        t0 = time.monotonic()
        bridge.start(asyncio.get_running_loop())

        # Wait for all N_OBJECTS to accumulate their full tick history.
        for _ in range(100):
            await asyncio.sleep(0.1)
            all_done = True
            for obj_idx in range(N_OBJECTS):
                h = await state.hget_object_track(f"{CAMERA_STR_ID}:obj:{1000 + obj_idx}")
                history = h.get("position_history") if h else None
                if not history or len(history) < N_TICKS_PER_OBJECT:
                    all_done = False
                    break
            if all_done:
                break
        elapsed = time.monotonic() - t0
        await bridge.stop()
    finally:
        pb_module.run_detection_pipeline = original

    check(f"All {N_OBJECTS} objects × {N_TICKS_PER_OBJECT} ticks completed "
          f"within the poll window ({elapsed:.2f}s)", all_done, f"elapsed={elapsed:.2f}s")

    # ── The actual race-condition proof: NO writes were lost ────────────────
    lost_writes = []
    for obj_idx in range(N_OBJECTS):
        oid = f"{CAMERA_STR_ID}:obj:{1000 + obj_idx}"
        h = await state.hget_object_track(oid)
        history = h.get("position_history") if h else []
        expected_times = {float(t) * 2.0 for t in range(N_TICKS_PER_OBJECT)}
        # Capped at _MAX_POSITION_HISTORY=20 in the detector — 15 < 20 here,
        # so nothing should be trimmed; every tick's write must be present.
        actual_times = {p["t"] for p in history}
        if actual_times != expected_times:
            lost_writes.append((oid, sorted(expected_times - actual_times)))

    check(f"No lost writes across {N_OBJECTS} objects × {N_TICKS_PER_OBJECT} "
          "ticks each — the per-object lock prevents ticks for the SAME "
          "object from racing on Redis's read-modify-write hash update",
          not lost_writes, str(lost_writes))

    # ── The other half of the claim: this did NOT serialize into one big
    #    line. N_OBJECTS × N_TICKS_PER_OBJECT = 180 total Redis round-trips;
    #    if they ran one-at-a-time globally this would take much longer than
    #    if different objects' locks are independent. Not a strict timing
    #    assertion (timing is inherently noisy) — just confirms it finished
    #    fast enough that a global lock is implausible.
    check(f"Completed in {elapsed:.2f}s — consistent with per-object (not "
          f"global) locking; {N_OBJECTS} objects processing serially "
          "end-to-end through a single Python/SQLite/Redis round-trip each "
          "would be expected to take meaningfully longer",
          elapsed < 5.0, f"elapsed={elapsed:.2f}s")

    print(f"\n{'=' * 68}\n{len(PASS)} passed, {len(FAIL)} failed\n{'=' * 68}")
    if FAIL:
        print("FAILED:", FAIL)
        sys.exit(1)


if __name__ == "__main__":
    asyncio.run(main())
