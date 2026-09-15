"""backend/services/behaviour_bridge.py — run the async behaviour detectors from
the synchronous CCTV pipeline.

WHY A BRIDGE IS NEEDED
  The behaviour detectors (loitering, crowd, abandoned-object) are async: they
  await Redis-backed state. They were written for the API server, which is
  async throughout, and are called from connection_manager.py.

  main.py - the pipeline that actually decodes video and runs the detector - is
  synchronous, and never calls them. That is one half of why the alerts table
  held 332 rows without a single behavioural detection: the detectors lived in
  a process that sees WebSocket events, not one that sees frames.

  (The other half was infrastructure: no Redis server, an alerts.id PK the ORM
  could not populate, and a NOT NULL severity column. All three are fixed; a
  loitering and a crowd alert now persist. This connects them to real video.)

DESIGN - one loop, one thread, never block the pipeline
  A dedicated asyncio loop runs in a daemon thread. The pipeline submits
  coroutines to it with run_coroutine_threadsafe and does NOT wait for the
  result. Frame processing must not stall behind a detector: a dropped
  behaviour tick costs one sample from a 60-second window, whereas a stalled
  pipeline drops frames outright and corrupts every downstream measurement.

  asyncio.run() per call was the alternative and is wrong here - it builds and
  tears down a loop per frame, and the detectors hold state across calls that a
  fresh loop would strand.

BACKPRESSURE
  Submissions are counted and dropped past a ceiling. Without that, a slow
  detector would grow an unbounded queue of futures and the pipeline would leak
  memory for as long as it ran. Dropping is visible in the stats, not silent.
"""
from __future__ import annotations

import asyncio
import logging
import threading
from typing import Any, Coroutine

logger = logging.getLogger(__name__)

# Ceiling on coroutines in flight. Loitering samples one position per track per
# processed frame, so a busy camera submits tens per second; a few hundred
# pending means the detectors have fallen behind and shedding is correct.
_MAX_INFLIGHT = 400


class BehaviourBridge:
    """Submit async detector work from sync code without blocking it."""

    def __init__(self) -> None:
        self._loop: asyncio.AbstractEventLoop | None = None
        self._thread: threading.Thread | None = None
        self._inflight = 0
        self._lock = threading.Lock()
        self.submitted = 0
        self.dropped = 0
        self.failed = 0

    def start(self) -> None:
        if self._thread is not None:
            return
        ready = threading.Event()

        def _run() -> None:
            self._loop = asyncio.new_event_loop()
            asyncio.set_event_loop(self._loop)
            ready.set()
            self._loop.run_forever()

        self._thread = threading.Thread(target=_run, name="behaviour-loop",
                                        daemon=True)
        self._thread.start()
        ready.wait(timeout=5.0)
        logger.info("[behaviour] async bridge started")

    def submit(self, coro: Coroutine[Any, Any, Any]) -> bool:
        """Fire-and-forget. Returns False if shed or the loop is not running."""
        if self._loop is None or not self._loop.is_running():
            coro.close()
            return False
        with self._lock:
            if self._inflight >= _MAX_INFLIGHT:
                self.dropped += 1
                coro.close()
                return False
            self._inflight += 1
            self.submitted += 1

        fut = asyncio.run_coroutine_threadsafe(coro, self._loop)

        def _done(f: Any) -> None:
            with self._lock:
                self._inflight -= 1
            exc = f.exception()
            if exc is not None:
                self.failed += 1
                # Log once per distinct type; a broken detector would otherwise
                # emit one traceback per frame and bury everything else.
                logger.warning("[behaviour] detector raised %s: %s",
                               type(exc).__name__, exc)

        fut.add_done_callback(_done)
        return True

    def stats(self) -> dict[str, int]:
        with self._lock:
            return {"submitted": self.submitted, "dropped": self.dropped,
                    "failed": self.failed, "inflight": self._inflight}

    def stop(self, timeout: float = 3.0) -> None:
        if self._loop is None:
            return
        # Give queued detector work a moment to finish so a loitering window
        # that just completed still writes its alert.
        deadline = timeout
        while deadline > 0 and self._inflight > 0:
            threading.Event().wait(0.1)
            deadline -= 0.1
        self._loop.call_soon_threadsafe(self._loop.stop)
        if self._thread is not None:
            self._thread.join(timeout=2.0)
        logger.info("[behaviour] bridge stopped — %s", self.stats())


_bridge: BehaviourBridge | None = None


def get_bridge() -> BehaviourBridge:
    global _bridge
    if _bridge is None:
        _bridge = BehaviourBridge()
        _bridge.start()
    return _bridge


def feed_persons(camera_str_id: str, camera_db_id: int | str | None,
                 tracks: list, video_time: float) -> None:
    """Feed confirmed person tracks to loitering, and the count to crowd.

    `tracks` are ConfirmedTrack objects; the loitering window keys on the
    bbox centroid, which is what the detector expects for pixel-distance
    comparison against its radius.

    camera_db_id lands in Alert.camera_id, which the live schema declares
    VARCHAR(64) NOT NULL. Passing None there fails the insert outright - two
    correctly detected loitering events were lost that way before this
    defaulted to the string camera key. Callers with a real Camera row may
    still pass its id.
    """
    if camera_db_id is None:
        camera_db_id = camera_str_id
    from backend.services.crowd_detector import get_crowd_detector
    from backend.services.loitering_detector import get_loitering_detector

    bridge = get_bridge()
    loiter = get_loitering_detector()
    for t in tracks:
        x1, y1, x2, y2 = t.bbox
        bridge.submit(loiter.on_track_position(
            camera_str_id, camera_db_id, t.track_id,
            (x1 + x2) / 2.0, (y1 + y2) / 2.0, video_time))

    # One crowd sample per frame, not per track - the detector dedupes on
    # video_time, but submitting N identical ticks wastes N-1 slots of the
    # in-flight budget that loitering needs.
    bridge.submit(get_crowd_detector().on_frame_tick(
        camera_str_id, camera_db_id, video_time, len(tracks)))
