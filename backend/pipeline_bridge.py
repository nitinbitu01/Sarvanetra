"""
backend/pipeline_bridge.py — Bridges the blocking detection thread to the async world.

The core problem: YOLOv8 inference is CPU-bound and blocking. Running it
directly on FastAPI's async event loop would stall ALL WebSocket clients.
Solution:
  1. Run the pipeline in a daemon threading.Thread (producer).
  2. Events land in a stdlib queue.Queue (thread-safe).
  3. A small callback uses loop.call_soon_threadsafe() to push each event
     into an asyncio.Queue without blocking the thread.
  4. An asyncio drain task (runs on the event loop) pulls from the asyncio.Queue
     and calls ConnectionManager.handle_detection_event().
  5. A separate periodic expiry task calls ConnectionManager.check_and_expire_tracks()
     every second.

This pattern is correct and idiomatic. Do NOT simplify to run_in_executor
wrapping a loop — that re-enters the executor on every frame and loses state.
"""

from __future__ import annotations

import asyncio
import logging
import queue as stdlib_queue
import sys
import threading
from pathlib import Path
from typing import Any

logger = logging.getLogger(__name__)

# Add project root to sys.path so backend/ can import from root (detector.py etc.)
_PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))

from main import run_detection_pipeline  # noqa: E402  (after sys.path patch)


class PipelineBridge:
    """Starts the detection thread and wires its output into asyncio.

    Args:
        cfg: Full parsed config dict.
        source: Video file path or webcam index string.
        connection_manager: The ConnectionManager instance to feed events into.
    """

    def __init__(
        self,
        cfg: dict[str, Any],
        source: str,
        connection_manager: Any,  # ConnectionManager — avoid circular import typing
    ) -> None:
        self._cfg = cfg
        self._source = source
        self._manager = connection_manager

        # Thread-safe queue: pipeline thread → bridge callback
        self._thread_queue: stdlib_queue.Queue[dict] = stdlib_queue.Queue(maxsize=500)

        # Asyncio queue: bridge callback → drain task (on the event loop)
        self._async_queue: asyncio.Queue[dict] = asyncio.Queue(maxsize=500)

        # Stop signal for the pipeline thread
        self._stop_event = threading.Event()

        # Background thread handle
        self._thread: threading.Thread | None = None

        # Asyncio task handles
        self._drain_task: asyncio.Task | None = None
        self._expiry_task: asyncio.Task | None = None
        self._watchdog_task: asyncio.Task | None = None

        # Running state
        self._running = False
        self._thread_died_cleanly = False  # set True when stop() signals the thread

    # ──────────────────────────────────────────────────────────────────────
    # Lifecycle
    # ──────────────────────────────────────────────────────────────────────

    def start(self, loop: asyncio.AbstractEventLoop) -> None:
        """Start the pipeline thread and asyncio tasks.

        Args:
            loop: The running asyncio event loop (from FastAPI startup).
        """
        if self._running:
            logger.warning("PipelineBridge.start() called but already running.")
            return

        self._running = True
        self._loop = loop

        # Start the drain task (on the event loop)
        self._drain_task    = loop.create_task(self._drain_loop())
        self._expiry_task   = loop.create_task(self._expiry_loop())
        self._watchdog_task = loop.create_task(self._watchdog_loop())

        # Start the producer thread
        self._thread = threading.Thread(
            target=self._producer_target,
            name="detection-pipeline",
            daemon=True,
        )
        self._thread.start()
        logger.info("Pipeline thread started (source: %s).", self._source)

    async def stop(self) -> None:
        """Signal the pipeline thread to stop and cancel async tasks."""
        logger.info("Stopping pipeline bridge...")
        self._thread_died_cleanly = True  # suppress watchdog false-positive
        self._stop_event.set()

        for task in (self._drain_task, self._expiry_task, self._watchdog_task):
            if task and not task.done():
                task.cancel()

        if self._thread and self._thread.is_alive():
            self._thread.join(timeout=5.0)
            if self._thread.is_alive():
                logger.warning("Pipeline thread did not stop within 5s.")

        self._running = False
        logger.info("Pipeline bridge stopped.")

    @property
    def is_running(self) -> bool:
        return self._running and (self._thread is not None) and self._thread.is_alive()

    # ──────────────────────────────────────────────────────────────────────
    # Producer side (runs in background thread)
    # ──────────────────────────────────────────────────────────────────────

    def _producer_target(self) -> None:
        """Entry point for the detection pipeline thread.

        Calls run_detection_pipeline with an event_queue. The emitter inside
        puts events into `_thread_queue`. A callback moves them to
        `_async_queue` via call_soon_threadsafe.
        """
        def _on_event(event: dict) -> None:
            """Called from the thread — schedule async queue put on event loop."""
            try:
                self._loop.call_soon_threadsafe(
                    self._async_queue.put_nowait, event
                )
            except asyncio.QueueFull:
                logger.warning("Async queue full — dropping event for track %d",
                               event.get("track_id", "?"))
            except RuntimeError:
                # Loop is closed during shutdown
                pass

        # Wrap the thread_queue to intercept puts and call _on_event
        bridged_queue = _BridgedQueue(self._thread_queue, _on_event)

        try:
            run_detection_pipeline(
                cfg=self._cfg,
                source=self._source,
                event_queue=bridged_queue,
                stop_event=self._stop_event,
                display=False,
                save_video=False,
            )
        except Exception as exc:
            logger.exception("Unhandled exception in pipeline thread: %s", exc)
        finally:
            logger.info("Pipeline thread exiting.")

    # ──────────────────────────────────────────────────────────────────────
    # Consumer side (runs on the async event loop)
    # ──────────────────────────────────────────────────────────────────────

    async def _drain_loop(self) -> None:
        """Continuously drain events from the async queue and feed ConnectionManager.

        Day 10: routes by event_type. Person-track events (the original,
        undocumented-as-such default) go through handle_detection_event() —
        untouched, its docstring's "has camera_id, track_id, bbox, ..."
        contract still describes exactly what it receives. Object-track
        events (see event_emitter.py's emit_objects()) go through the new
        handle_object_event() instead, kept entirely separate rather than
        adding a branch inside handle_detection_event and having to carve
        object-shaped exceptions into a method documented as person-shaped.
        """
        logger.debug("Drain loop started.")
        try:
            while True:
                event = await self._async_queue.get()
                event_type = event.get("event_type", "person_track")
                logger.debug(
                    "Drain: processing %s track_id=%s", event_type, event.get("track_id")
                )
                try:
                    if event_type == "object_track":
                        await self._manager.handle_object_event(event)
                    else:
                        await self._manager.handle_detection_event(event)
                except Exception as exc:
                    logger.error("Event handler raised for %s: %s", event_type, exc)
        except asyncio.CancelledError:
            logger.debug("Drain loop cancelled.")

    async def _expiry_loop(self) -> None:
        """Periodically check for and broadcast expired tracks (every 1s)."""
        logger.debug("Expiry loop started.")
        try:
            while True:
                await asyncio.sleep(1.0)
                try:
                    await self._manager.check_and_expire_tracks()
                except Exception as exc:
                    logger.error("check_and_expire_tracks raised: %s", exc)
        except asyncio.CancelledError:
            logger.debug("Expiry loop cancelled.")

    async def _watchdog_loop(self) -> None:
        """Poll pipeline thread liveness every 5s; broadcast alert if it dies.

        If the detection thread exits unexpectedly (uncaught exception, OOM,
        model crash), clients would otherwise see a frozen dashboard with no
        indication that processing has stopped. This watchdog detects that and
        broadcasts a 'pipeline_dead' message so the UI can show an error state.
        """
        logger.debug("Watchdog loop started.")
        # Give the thread time to start before first check
        await asyncio.sleep(10.0)
        try:
            while True:
                await asyncio.sleep(5.0)
                if self._thread_died_cleanly:
                    break
                if self._thread is not None and not self._thread.is_alive():
                    logger.error(
                        "WATCHDOG: detection pipeline thread has died unexpectedly! "
                        "Dashboard is frozen. Restart the server to recover."
                    )
                    try:
                        await self._manager.broadcast({
                            "type": "pipeline_dead",
                            "camera_id": self._manager._camera_id,
                            "message": "Detection pipeline stopped unexpectedly. Restart the server.",
                        })
                    except Exception as exc:
                        logger.error("Watchdog broadcast failed: %s", exc)
                    break  # stop looping after alerting once
        except asyncio.CancelledError:
            logger.debug("Watchdog loop cancelled.")


class _BridgedQueue:
    """Minimal queue.Queue interface shim that calls a callback on every put().

    This allows run_detection_pipeline (which expects a stdlib queue.Queue)
    to transparently trigger the thread-to-async bridge callback on every event.
    """

    def __init__(
        self,
        inner: stdlib_queue.Queue,
        on_put_callback: Any,
    ) -> None:
        self._inner = inner
        self._callback = on_put_callback

    def put_nowait(self, item: dict) -> None:
        """Put item and immediately trigger the bridge callback."""
        self._callback(item)

    def put(self, item: dict, block: bool = True, timeout: float | None = None) -> None:
        """Put item and immediately trigger the bridge callback."""
        self._callback(item)

    # Pass through other queue.Queue attrs that EventEmitter might check
    @property
    def maxsize(self) -> int:
        return self._inner.maxsize
