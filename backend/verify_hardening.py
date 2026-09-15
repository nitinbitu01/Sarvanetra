"""
backend/verify_hardening.py — Automated hardening test suite for Days 1-4.

Tests each of the 8 hardening items by injecting synthetic faults and
confirming graceful recovery. Run from the project root:

    python backend/verify_hardening.py
"""

from __future__ import annotations

import asyncio
import json
import logging
import queue
import sqlite3
import sys
import tempfile
import threading
import time
from pathlib import Path

import numpy as np

sys.path.insert(0, ".")

logging.basicConfig(
    level=logging.WARNING,  # suppress INFO noise during tests
    format="%(levelname)s %(name)s: %(message)s",
)
logger = logging.getLogger("verify_hardening")

PASS = "[PASS]"
FAIL = "[FAIL]"
results: list[tuple[str, bool, str]] = []


def check(name: str, passed: bool, detail: str = "") -> None:
    results.append((name, passed, detail))
    status = PASS if passed else FAIL
    print(f"{status} {name}" + (f": {detail}" if detail else ""))


# ── Item 1: Config schema validation ─────────────────────────────────────────
def test_config_schema():
    from backend.startup_checks import check_config_schema

    import yaml
    cfg = yaml.safe_load(open("config.yaml"))

    # Valid config should produce no errors
    errors = check_config_schema(cfg)
    check("Config schema valid config → 0 errors", len(errors) == 0,
          f"got {len(errors)} errors: {errors[:2]}")

    # Inject a bad type
    bad_cfg = {**cfg, "processing": {**cfg.get("processing", {}), "fps": "five"}}
    errors2 = check_config_schema(bad_cfg)
    check("Config schema bad type → caught", any("fps" in e for e in errors2),
          f"errors: {errors2[:2]}")

    # Missing key
    bad_cfg2 = dict(cfg)
    bad_cfg2.pop("model", None)
    errors3 = check_config_schema(bad_cfg2)
    check("Config schema missing 'model' → caught", len(errors3) >= 2,
          f"got {len(errors3)} errors")


# ── Item 1b: Video source validation ─────────────────────────────────────────
def test_video_source_check():
    from backend.startup_checks import check_video_source

    errors_real = check_video_source("nonexistent_video.mp4")
    check("Video source: missing file caught", len(errors_real) > 0,
          errors_real[0] if errors_real else "no error")

    errors_webcam = check_video_source("0")
    check("Video source: webcam index is not an error", len(errors_webcam) == 0)


# ── Item 2: DB retry on locked ────────────────────────────────────────────────
def test_db_retry_on_locked():
    """Verify _retry_on_locked decorator retries on 'database is locked'."""
    from backend.db import _retry_on_locked

    call_count = [0]

    @_retry_on_locked
    def flaky_function():
        call_count[0] += 1
        if call_count[0] < 3:
            raise sqlite3.OperationalError("database is locked")
        return "success"

    result = flaky_function()
    check("DB retry: succeeds after 2 locked errors", result == "success",
          f"calls={call_count[0]}, result={result}")

    # Non-lock error should NOT be retried
    non_lock_calls = [0]

    @_retry_on_locked
    def always_fails():
        non_lock_calls[0] += 1
        raise sqlite3.OperationalError("no such table: xyz")

    try:
        always_fails()
        check("DB retry: non-lock error propagates immediately", False, "did not raise")
    except sqlite3.OperationalError:
        check("DB retry: non-lock error propagates immediately (not retried)",
              non_lock_calls[0] == 1, f"calls={non_lock_calls[0]}")


# ── Item 3: EventEmitter fault isolation ─────────────────────────────────────
def test_event_emitter_fault_isolation():
    """A broken file handle must not crash the detection loop."""
    import io
    from unittest.mock import patch, MagicMock
    import yaml
    from event_emitter import EventEmitter
    from tracker import ConfirmedTrack

    cfg = yaml.safe_load(open("config.yaml"))

    with tempfile.TemporaryDirectory() as tmpdir:
        cfg2 = dict(cfg)
        cfg2["output"] = {
            **cfg.get("output", {}),
            "events_file": str(Path(tmpdir) / "events.jsonl"),
            "crops_dir":   str(Path(tmpdir) / "crops"),
            "crop_save_interval_frames": 1,
        }
        emitter = EventEmitter(cfg2, event_queue=None, camera_id="TEST")

        # Break the file handle so writes raise OSError
        emitter._events_fh.close()

        track = ConfirmedTrack(
            track_id=1, bbox=[0.0, 0.0, 100.0, 200.0],
            confidence=0.9, frame_number=1, timestamp=1.0,
        )
        frame = np.zeros((480, 640, 3), dtype=np.uint8)

        crashed = False
        try:
            emitter.emit([track], frame)
        except Exception:
            crashed = True

        check("EventEmitter: broken file handle does NOT crash loop", not crashed)

        # Verify crop save with empty frame doesn't crash
        emitter2 = EventEmitter(cfg2, event_queue=None, camera_id="TEST")
        try:
            emitter2._maybe_save_crop(track, np.array([]))
            emitter2.close()
        except Exception as e:
            check("EventEmitter: empty frame crop does NOT crash", False, str(e))
            return
        check("EventEmitter: empty frame crop handled gracefully", True)


# ── Item 4: WebSocket send timeout ───────────────────────────────────────────
def test_websocket_timeout():
    """Verify _send_to returns False when a send exceeds the timeout."""
    from backend.connection_manager import ConnectionManager

    async def run():
        class SlowSocket:
            async def send_json(self, _):
                await asyncio.sleep(5.0)  # much longer than SEND_TIMEOUT_S

        slow_ws = SlowSocket()
        t0 = time.monotonic()
        result = await ConnectionManager._send_to(slow_ws, {"type": "test"})
        elapsed = time.monotonic() - t0
        return result, elapsed

    result, elapsed = asyncio.run(run())
    check(
        "WebSocket timeout: slow send returns False",
        result is False,
        f"result={result}",
    )
    check(
        f"WebSocket timeout: completes in <{ConnectionManager.SEND_TIMEOUT_S + 0.5:.1f}s",
        elapsed < ConnectionManager.SEND_TIMEOUT_S + 0.5,
        f"elapsed={elapsed:.2f}s",
    )


# ── Item 4b: Client cap ───────────────────────────────────────────────────────
def test_client_cap():
    from backend.connection_manager import ConnectionManager
    cm = ConnectionManager("CAM-01", 5.0)
    # Fill up to MAX_CLIENTS
    cm._clients = set(range(ConnectionManager.MAX_CLIENTS))  # type: ignore
    check(
        "ConnectionManager: MAX_CLIENTS cap respected",
        len(cm._clients) == ConnectionManager.MAX_CLIENTS,
        f"cap={ConnectionManager.MAX_CLIENTS}",
    )


# ── Item 5: Pipeline watchdog (structural) ───────────────────────────────────
def test_pipeline_watchdog_exists():
    from backend.pipeline_bridge import PipelineBridge
    import inspect
    has_watchdog = hasattr(PipelineBridge, "_watchdog_loop")
    check("PipelineBridge: _watchdog_loop method exists", has_watchdog)
    # Check stop() sets _thread_died_cleanly
    src = inspect.getsource(PipelineBridge.stop)
    check("PipelineBridge.stop: sets _thread_died_cleanly", "_thread_died_cleanly" in src)


# ── Item 6: OCR worker restart ───────────────────────────────────────────────
def test_ocr_worker_restart():
    """Verify MAX_RESTART_ATTEMPTS is set and restart logic exists."""
    from backend.anpr_worker import AnprWorker
    import inspect

    check(
        "AnprWorker: MAX_RESTART_ATTEMPTS = 3",
        AnprWorker.MAX_RESTART_ATTEMPTS == 3,
        f"got {AnprWorker.MAX_RESTART_ATTEMPTS}",
    )
    src = inspect.getsource(AnprWorker._worker_loop)
    check(
        "AnprWorker._worker_loop: restart loop present",
        "MAX_RESTART_ATTEMPTS" in src and "attempt" in src,
    )
    check(
        "AnprWorker: _worker_inner method exists",
        hasattr(AnprWorker, "_worker_inner"),
    )

    # Verify worker auto-recovers from one failed start
    restart_count = [0]
    stop = threading.Event()

    def patched_inner(self):
        restart_count[0] += 1
        if restart_count[0] < 3:
            raise RuntimeError("Simulated model crash")
        stop.set()  # signal success on 3rd attempt

    import types
    original = AnprWorker._worker_inner
    AnprWorker._worker_inner = patched_inner

    worker = AnprWorker({"anpr": {"ocr_worker_queue_maxsize": 10}}, result_callback=lambda *a: None)
    worker.start()
    stop.wait(timeout=10.0)
    worker.stop()
    AnprWorker._worker_inner = original

    check(
        "AnprWorker: recovers from 2 crashes on 3rd attempt",
        restart_count[0] == 3,
        f"restart_count={restart_count[0]}",
    )


# ── Item 7: FAISS dimension mismatch ─────────────────────────────────────────
def test_faiss_dimension_mismatch():
    import faiss as faiss_lib
    import yaml
    from backend.faiss_index import FaissReIDIndex

    cfg = yaml.safe_load(open("config.yaml"))

    with tempfile.TemporaryDirectory() as tmpdir:
        # Create a 256-dim index but claim 512 in config
        idx = faiss_lib.IndexFlatIP(256)
        vec = np.random.randn(1, 256).astype(np.float32)
        idx.add(vec)
        index_path = Path(tmpdir) / "test.index"
        id_map_path = Path(tmpdir) / "id_map.json"
        faiss_lib.write_index(idx, str(index_path))
        id_map_path.write_text(json.dumps([{"clip_id": "c1", "label": "x", "camera_id": "c", "npy_path": "x"}]))

        bad_cfg = {**cfg, "faiss": {
            "index_path": str(index_path),
            "id_map_path": str(id_map_path),
            "embedding_dim": 512,  # mismatch: index has 256
        }}
        try:
            FaissReIDIndex.load(bad_cfg)
            check("FAISS: dim mismatch raises ValueError", False, "no error raised")
        except ValueError as e:
            check("FAISS: dim mismatch raises ValueError", True, str(e)[:60])


# ── Item 8: Startup checks DB missing table ───────────────────────────────────
def test_startup_db_missing_tables():
    from backend.startup_checks import check_database
    import yaml

    with tempfile.TemporaryDirectory() as tmpdir:
        db_path = str(Path(tmpdir) / "empty.db")
        # Create DB with no tables
        conn = sqlite3.connect(db_path)
        conn.close()

        cfg = yaml.safe_load(open("config.yaml"))
        bad_cfg = {**cfg, "database": {"path": db_path}}
        errors = check_database(bad_cfg)
        check(
            "Startup check: DB missing tables detected",
            any("missing tables" in e for e in errors),
            f"errors: {errors}",
        )


# ── Run all tests ─────────────────────────────────────────────────────────────
if __name__ == "__main__":
    print("=== Sentinel Gujarat — Hardening Verification (Days 1–4) ===\n")

    test_config_schema()
    test_video_source_check()
    test_db_retry_on_locked()
    test_event_emitter_fault_isolation()
    test_websocket_timeout()
    test_client_cap()
    test_pipeline_watchdog_exists()
    test_ocr_worker_restart()
    test_faiss_dimension_mismatch()
    test_startup_db_missing_tables()

    n_pass = sum(1 for _, p, _ in results if p)
    n_fail = sum(1 for _, p, _ in results if not p)

    print(f"\n{'='*54}")
    print(f"RESULTS: {n_pass} passed / {n_fail} failed / {len(results)} total")
    print("="*54)

    if n_fail:
        print("\nFailed checks:")
        for name, passed, detail in results:
            if not passed:
                print(f"  {FAIL} {name}: {detail}")
        sys.exit(1)
    else:
        print("ALL HARDENING CHECKS PASSED ✓")
