"""
backend/connection_manager.py — WebSocket client management for Sentinel Gujarat.

Responsibilities:
  - Track all active WebSocket connections
  - Send a snapshot of current state immediately on connect
  - Broadcast detection events and expiry notices to all clients
  - Maintain `current_state` — the in-memory truth of active confirmed tracks
  - Handle dead sockets gracefully (failed sends are silently dropped)

WebSocket message schemas (contract — Day 5+ extends, does not replace):

  # Sent once on connect:
  {"type": "snapshot", "camera_id": str, "tracks": [
      {"track_id": int, "bbox": [...], "confidence": float, "frame_number": int}
  ]}

  # On each new detection event:
  {"type": "detection_event", "camera_id": str, "track_id": int,
   "frame_number": int, "timestamp": float, "bbox": [...], "confidence": float}

  # When a track expires:
  {"type": "track_expired", "camera_id": str, "track_id": int}

Track persistence (added after Day 9 — the fix for a Day 6/7 foundation gap):
  This module is the ONLY place a `tracks` DB row is created from the live
  pipeline. Nothing else in the running system ever did (backend/db.py's
  raw-SQL upsert_track() has no production callers), which meant
  `event.get("db_track_id")` was always None, and BOTH the ReID dispatch gate
  and the face-watchlist dispatch gate below were structurally unreachable:
  they require a DB Track id that nothing produced. resolve_identity()'s
  `Track.id == track_db_id` lookup and Journey.local_track_id's FK to
  tracks.id are downstream of the same missing row.

  On first sighting of a (camera_id, track_id) pair we find-or-create that
  row and cache its PK for the rest of the track's life; last_seen_* is
  refreshed on a frame throttle (NOT per frame — that would be one DB write
  per person per frame). The event dict is then enriched with `db_track_id`
  and `track_confirmed` before the dispatch gates read them.
"""

from __future__ import annotations

import asyncio
import logging
import math
import time
from typing import Any

from fastapi import WebSocket

logger = logging.getLogger(__name__)

# Fallback frame cadence for last_seen_* refreshes when main.py doesn't pass
# one. Mirrors config.yaml's output.crop_save_interval_frames default so the
# DB-update cadence and the crop cadence stay the same number in both places.
DEFAULT_TRACK_UPDATE_INTERVAL_FRAMES = 15

# Flag: set to True from main.py lifespan once ReID services are ready.
# Guards against the ReID task firing before the index is initialized.
_reid_ready: bool = False
def set_reid_ready(camera_db_id: int | None = None) -> None:
    """Called from main.py lifespan once the ReID index is initialised.

    `camera_db_id` is accepted but IGNORED, and is retained only so older
    call sites don't break. It used to populate a module-level global that
    every detector dispatch read — which silently made the whole process
    single-camera: with two feeds running, every alert from the second one
    was tagged with the FIRST camera's database id. Per-camera IQ scores,
    zone incidents, cross-zone journeys and Day 14 officer dispatch all key
    off that id, so they produced confident wrong answers rather than
    failing. Camera identity now lives on the ConnectionManager instance
    that actually owns the feed.
    """
    global _reid_ready
    _reid_ready = True
    if camera_db_id is not None:
        logger.debug(
            "set_reid_ready(camera_db_id=%s) — argument ignored; camera identity "
            "is per-ConnectionManager now.", camera_db_id,
        )


# Day 7: same readiness-gating pattern, separate flag — face watchlist
# matching is a parallel branch to ReID, not a dependency of it, so it gets
# its own ready flag rather than piggybacking on _reid_ready.
_face_watchlist_ready: bool = False


def set_face_watchlist_ready() -> None:
    """Called from main.py lifespan once the face embedder + watchlist matcher are ready."""
    global _face_watchlist_ready
    _face_watchlist_ready = True


# Day 8: behavior engine (loitering + crowd anomaly) readiness flag. Same
# gating pattern as ReID/face-watchlist above, but this branch runs on EVERY
# event (not dispatch-once at confirmation) — see handle_detection_event.
_behavior_ready: bool = False


def set_behavior_ready() -> None:
    """Called from main.py lifespan once the behavior-engine state backend
    (Redis) connectivity has been checked at startup (best-effort — the
    engine still runs, and skips-and-logs per-tick, even if Redis is down;
    this flag just gates the branch from firing before main.py has finished
    its own startup sequence)."""
    global _behavior_ready
    _behavior_ready = True


# Day 10: abandoned-object detector readiness flag. Same gating pattern as
# behavior_ready — shares the same Redis backend and the same calibration
# cache, so it's safe to flip once both of those are confirmed ready.
_abandoned_ready: bool = False


def set_abandoned_ready() -> None:
    """Called from main.py lifespan once the abandoned-object detector's
    dependencies (Redis, calibration cache) are the same ones the behavior
    engine already confirmed — this flag exists as its own gate, not reused
    from _behavior_ready, so the two branches can be disabled independently
    if one detector needs to be pulled without touching the other."""
    global _abandoned_ready
    _abandoned_ready = True



# ──────────────────────────────────────────────────────────────────────────
# Track row persistence (sync — always called via asyncio.to_thread)
# ──────────────────────────────────────────────────────────────────────────
# These two functions do blocking SQLAlchemy work. handle_detection_event()
# runs on the drain loop (the async hot path that feeds every WebSocket
# client), so they are never called inline there — always through
# asyncio.to_thread, which is also what makes the "another writer won the
# race" retry below meaningful.


def crop_path_for(track_id: int, frame_number: int, camera_id: str = "") -> str:
    """Project-root-relative path of the crop event_emitter.py saved this frame.

    Mirrors event_emitter.py's convention EXACTLY:
        output/crops/{camera_id}/track_{track_id}/frame_{frame_number}.jpg

    The camera_id segment matters: BoT-SORT track ids are per-camera locals
    that restart at 1 on each feed, so without it two cameras' crops of
    DIFFERENT people share one track_N directory — and this function would
    hand the review queue a picture of the wrong person.

    Only call this when the event actually carried a `crop_bgr` — that is the
    signal the emitter's throttle fired and a file was written for this frame.
    Deriving it from an event without a crop would fabricate a path to a file
    that does not exist.
    """
    from event_emitter import _fs_safe

    cam_segment = _fs_safe(camera_id) if camera_id else "unknown_camera"
    return f"output/crops/{cam_segment}/track_{track_id}/frame_{frame_number}.jpg"


def _find_or_create_track_row(
    camera_str_id: str,
    track_id: int,
    frame_number: int,
    timestamp: float,
    confidence: float,
    crop_path: str | None = None,
) -> int | None:
    """Return the `tracks` PK for (camera_str_id, track_id), inserting if absent.

    Returns None on any DB failure — the caller degrades to the pre-existing
    behaviour (no db_track_id on the event, ReID/face dispatch skipped for
    that track) rather than crashing the detection stream over a DB blip.
    """
    from sqlalchemy.exc import IntegrityError

    from backend.db.models import Track
    from backend.db.session import SessionLocal

    db = SessionLocal()
    try:
        row = (
            db.query(Track)
            .filter(Track.camera_id == camera_str_id, Track.track_id == track_id)
            .first()
        )
        if row is not None:
            return row.id

        row = Track(
            camera_id=camera_str_id,
            track_id=track_id,
            first_seen_frame=frame_number,
            first_seen_timestamp=timestamp,
            last_seen_frame=frame_number,
            last_seen_timestamp=timestamp,
            best_confidence=confidence,
            # Populates ReIDReviewItem.crop_image_path downstream — that is
            # the "New Sighting" image an officer compares against in the
            # review queue. Left NULL, the review UI has nothing to show and
            # the whole side-by-side comparison degrades to two placeholders.
            best_crop_path=crop_path,
        )
        db.add(row)
        db.commit()
        logger.debug(
            "Created tracks row id=%d for %s:%d", row.id, camera_str_id, track_id
        )
        return row.id
    except IntegrityError:
        # The `tracks` table carries UNIQUE(camera_id, track_id). Losing this
        # race is not an error — re-read whatever the winner inserted.
        db.rollback()
        row = (
            db.query(Track)
            .filter(Track.camera_id == camera_str_id, Track.track_id == track_id)
            .first()
        )
        return row.id if row else None
    except Exception as exc:
        db.rollback()
        logger.error(
            "Failed to find-or-create tracks row for %s:%d — ReID and face "
            "watchlist dispatch will be skipped for this track: %s",
            camera_str_id, track_id, exc,
        )
        return None
    finally:
        db.close()


def _touch_track_row(
    db_track_id: int,
    frame_number: int,
    timestamp: float,
    confidence: float,
    crop_path: str | None = None,
) -> None:
    """Refresh last_seen_* (and best_confidence/best_crop_path, if this frame beat it).

    Throttled by the caller — see DEFAULT_TRACK_UPDATE_INTERVAL_FRAMES.
    """
    from backend.db.models import Track
    from backend.db.session import SessionLocal

    db = SessionLocal()
    try:
        row = db.query(Track).filter(Track.id == db_track_id).first()
        if row is None:
            return
        row.last_seen_frame = frame_number
        row.last_seen_timestamp = timestamp
        if row.best_confidence is None or confidence > row.best_confidence:
            row.best_confidence = confidence
            # best_crop_path must move WITH best_confidence, not independently —
            # otherwise "best crop" and "best confidence" describe different
            # frames and the review queue shows an image that isn't the one
            # the stored score refers to. Only overwrite when this frame
            # actually carried a crop; a higher-confidence frame that fell
            # outside the emitter's crop throttle keeps the previous path
            # rather than nulling out a real image.
            if crop_path is not None:
                row.best_crop_path = crop_path
        elif row.best_crop_path is None and crop_path is not None:
            # First crop to arrive for a track whose row was created before
            # the emitter's throttle had fired — backfill rather than leaving
            # the review queue with nothing to display.
            row.best_crop_path = crop_path
        db.commit()
    except Exception as exc:
        db.rollback()
        logger.debug("Failed to touch tracks row id=%d: %s", db_track_id, exc)
    finally:
        db.close()


class ConnectionManager:
    """Manages all active WebSocket client connections and in-memory track state.

    Thread safety: All methods are called from the async event loop only.
    Do NOT call from a background thread — use asyncio primitives instead.
    """

    # Max simultaneous clients: prevents runaway memory if dashboard is hammered
    MAX_CLIENTS = 50
    # Seconds before a slow WebSocket send is treated as dead and client removed
    SEND_TIMEOUT_S = 1.5

    def __init__(
        self,
        camera_id: str,
        track_expiry_seconds: float,
        track_update_interval_frames: int = DEFAULT_TRACK_UPDATE_INTERVAL_FRAMES,
        camera_db_id: int | None = None,
    ) -> None:
        self._camera_id = camera_id
        # The integer cameras.id for THIS feed. Instance state, deliberately
        # not a module global: one process can now run several
        # ConnectionManagers, one per camera, and each must stamp its own
        # camera on the alerts it produces.
        self._camera_db_id = camera_db_id
        self._expiry_seconds = track_expiry_seconds
        self._track_update_interval = max(1, int(track_update_interval_frames))

        # Active WebSocket connections
        self._clients: set[WebSocket] = set()

        # In-memory state: track_id → {track_id, bbox, confidence, frame_number,
        #                               last_seen_wall_time}
        # Updated on every detection_event, deleted on expiry.
        self.current_state: dict[int, dict[str, Any]] = {}

        # Track IDs already dispatched to ReID to avoid re-processing.
        # Cleared on track expiry. Uses the pipeline integer track_id (not DB id).
        self._reid_dispatched: set[int] = set()

        # Day 7: same dispatch-once bookkeeping for the face watchlist branch,
        # tracked separately from ReID so the two never interfere with each
        # other's re-processing state.
        self._face_dispatched: set[int] = set()

        # pipeline track_id → `tracks` table PK, populated on first sighting
        # and cleared on expiry. This is what makes db_track_id non-None on
        # emitted events, which is what makes the ReID and face-watchlist
        # dispatch gates below actually reachable.
        self._track_db_ids: dict[int, int] = {}

        # pipeline track_id → frame_number of the last last_seen_* DB refresh.
        self._track_touch_frame: dict[int, int] = {}

        # Strong refs to in-flight background tasks (DB writes, abandoned-
        # object ticks). asyncio only holds a weak reference to a task, so a
        # create_task() result that nothing keeps can be garbage-collected
        # mid-flight and silently never run.
        self._pending_tasks: set[asyncio.Task] = set()

        # Day 10: object_track_id ("{camera_id}:obj:{pipeline_track_id}") →
        # {cx, cy, last_seen_wall_time}. Mirrors current_state but for object
        # tracks — needed so handle_object_event can find "objects near this
        # person" and, separately, so a slow/stalled object stream can be
        # expired instead of accumulating forever. Unlike current_state, this
        # is NOT read by any WebSocket broadcast (this detector has no
        # dashboard live-view yet — it only produces Alert rows).
        self._object_positions: dict[str, dict[str, Any]] = {}

        # Day 10: one asyncio.Lock per object_track_id, serializing that
        # object's ticks through the detector. AbandonedObjectDetector.
        # on_object_update() does a Redis READ of the full object_track hash,
        # merges fields in Python, then WRITEs the full hash back — a
        # read-modify-write cycle, not an atomic single command. It is
        # written (and tested, in tests/abandoned_object_eval) assuming ticks
        # for one object arrive one at a time. Every OTHER detector
        # integration in this file (_run_reid, _run_behavior_tick, ...) fires
        # fully independent background tasks with no such assumption — this
        # one can't, without two ticks racing on the same hash and the
        # slower write silently discarding the faster one's fields (this was
        # found, not assumed: see verify_abandoned_object_wiring.py history —
        # under real threaded/queued delivery, only 2 of 11 ticks survived).
        # Distinct objects still process fully concurrently; only same-object
        # ticks are serialized. Cleared on expiry alongside _object_positions
        # so this cannot grow unbounded over a long-running deployment.
        self._object_locks: dict[str, asyncio.Lock] = {}

    # ──────────────────────────────────────────────────────────────────────
    # Connection lifecycle
    # ──────────────────────────────────────────────────────────────────────

    async def connect(self, websocket: WebSocket) -> None:
        """Accept a new WebSocket connection and send an immediate snapshot.

        Rejects connections beyond MAX_CLIENTS to prevent runaway memory.

        Args:
            websocket: Incoming WebSocket connection (not yet accepted).
        """
        if len(self._clients) >= self.MAX_CLIENTS:
            logger.warning(
                "WebSocket connection rejected: at capacity (%d/%d clients).",
                len(self._clients), self.MAX_CLIENTS,
            )
            await websocket.close(code=1013)  # 1013 = Try Again Later
            return

        await websocket.accept()
        self._clients.add(websocket)
        logger.info(
            "WebSocket client connected. Active clients: %d", len(self._clients)
        )

        # Build and send snapshot from current in-memory state
        snapshot = {
            "type": "snapshot",
            "camera_id": self._camera_id,
            "tracks": [
                {
                    "track_id": t["track_id"],
                    "bbox": t["bbox"],
                    "confidence": t["confidence"],
                    "frame_number": t["frame_number"],
                }
                for t in self.current_state.values()
            ],
        }
        await self._send_to(websocket, snapshot)
        logger.debug("Snapshot sent: %d active tracks.", len(self.current_state))

    async def disconnect(self, websocket: WebSocket) -> None:
        """Remove a client from the active set.

        Safe to call even if the socket is already closed.

        Args:
            websocket: The WebSocket to remove.
        """
        self._clients.discard(websocket)
        logger.info(
            "WebSocket client disconnected. Active clients: %d", len(self._clients)
        )

    # ──────────────────────────────────────────────────────────────────────
    # Event broadcasting
    # ──────────────────────────────────────────────────────────────────────

    async def handle_detection_event(self, event: dict[str, Any]) -> None:
        """Process an incoming detection event from the pipeline.

        Updates current_state and broadcasts to all clients.

        Args:
            event: A detection event dict from event_emitter (has camera_id,
                   track_id, bbox, confidence, frame_number, timestamp).
        """
        track_id: int = event["track_id"]
        camera_str_id: str = event.get("camera_id", self._camera_id)

        # Update in-memory state
        self.current_state[track_id] = {
            "track_id": track_id,
            "bbox": event["bbox"],
            "confidence": event["confidence"],
            "frame_number": event["frame_number"],
            "timestamp": event["timestamp"],
            "last_seen_wall_time": time.monotonic(),
        }

        # Broadcast detection_event to all clients
        message = {
            "type": "detection_event",
            "camera_id": event.get("camera_id", self._camera_id),
            "track_id": track_id,
            "frame_number": event["frame_number"],
            "timestamp": event["timestamp"],
            "bbox": event["bbox"],
            "confidence": event["confidence"],
        }
        await self.broadcast(message)

        # Also publish onto the AUTHENTICATED dashboard socket.
        #
        # self.broadcast() above only reaches this manager's own clients, i.e.
        # the legacy Day 2 /ws/detections endpoint — which is unauthenticated
        # AND bound to the single module-level `manager` global, so it carries
        # exactly one camera's events. Neither property is acceptable for the
        # multi-camera control room, so detections were invisible there.
        #
        # /ws/dashboard is token-gated and already carries every other live
        # event type, so the control room can render per-camera detections
        # over the ONE socket it already holds open, rather than opening a
        # second (unauthenticated) one.
        #
        # Volume is safe: this fires per CONFIRMED track event, measured at
        # ~0.5/sec across 8 live cameras — not per raw detection (~8/sec) and
        # not per frame.
        try:
            from backend.ws.dashboard_ws import broadcast as _dash_broadcast

            await _dash_broadcast(message)
        except Exception as exc:
            # A dashboard delivery problem must never take down the detection
            # path — this is a display concern, the DB write below is not.
            logger.debug("Dashboard broadcast of detection_event failed: %s", exc)

        # ── Track row persistence — enrich the event for the gates below ─────
        # Placed AFTER the broadcast on purpose: the first event for a track
        # costs a DB round-trip (find-or-create), and putting that ahead of
        # the broadcast would delay every dashboard's first sight of a new
        # person by that round-trip for no benefit — the broadcast payload
        # does not carry db_track_id. It must still come BEFORE the two
        # dispatch gates, which read db_track_id/track_confirmed off this
        # same dict and were structurally unreachable until it did.
        await self._ensure_track_row(camera_str_id, event)

        # ── Day 6: Trigger ReID for newly confirmed tracks ───────────────────────────
        # Fire-and-forget via asyncio.create_task() so it NEVER blocks the
        # detection loop. The ReID resolution runs completely off the hot path.
        if (
            _reid_ready
            and track_id not in self._reid_dispatched
            and event.get("track_confirmed", False)   # set by _ensure_track_row above
        ):
            crop_bgr = event.get("crop_bgr")          # np.ndarray or None (throttled)
            db_track_id = event.get("db_track_id")    # `tracks` PK

            # Mark dispatched INSIDE the guard, not before it. Marking first
            # burned the one dispatch attempt on any event that happened to
            # arrive without a crop — the track was then permanently flagged
            # as handled and never re-tried on the next event that did carry
            # one. `crop_bgr` is throttled by design, so that is a normal
            # event, not an edge case.
            if crop_bgr is not None and db_track_id is not None:
                self._reid_dispatched.add(track_id)
                asyncio.create_task(
                    self._run_reid(
                        db_track_id=int(db_track_id),
                        camera_str_id=camera_str_id,
                        crop_bgr=crop_bgr,
                        confidence=float(event["confidence"]),
                        camera_db_id=self._camera_db_id,
                    ),
                    name=f"reid-track-{track_id}",
                )

        # ── Day 7: Trigger face watchlist matching for newly confirmed tracks ────
        # Parallel branch to ReID above, not sequential with it — both answer
        # different questions about the same confirmed track and neither
        # depends on the other's result. Fire-and-forget for the same reason:
        # it must never block the detection loop.
        if (
            _face_watchlist_ready
            and track_id not in self._face_dispatched
            and event.get("track_confirmed", False)
        ):
            crop_bgr = event.get("crop_bgr")
            db_track_id = event.get("db_track_id")

            # Same dispatch-marking fix as the ReID gate above.
            if crop_bgr is not None and db_track_id is not None:
                self._face_dispatched.add(track_id)
                asyncio.create_task(
                    self._run_face_watchlist(
                        db_track_id=int(db_track_id),
                        camera_str_id=camera_str_id,
                        crop_bgr=crop_bgr,
                        camera_db_id=self._camera_db_id,
                    ),
                    name=f"facewl-track-{track_id}",
                )

        # ── Day 8: Behavior engine (loitering + crowd anomaly) ────────────
        # Unlike ReID/face-watchlist above, this branch runs on EVERY event,
        # not dispatch-once at first confirmation — loitering needs a
        # continuous per-frame position stream for the track's whole visible
        # lifetime, and crowd counting needs a live per-frame count. No extra
        # confirmation gating is needed here: every event reaching this point
        # already corresponds to a confirmed track (event_emitter.py only
        # ever emits ConfirmedTrack instances — see its module docstring),
        # regardless of whether the (apparently not currently populated)
        # `track_confirmed` field above is set on this particular event.
        # Fire-and-forget for the same reason as ReID/face-watchlist: must
        # never block the detection hot path.
        if _behavior_ready:
            bbox = event.get("bbox")
            if bbox and len(bbox) == 4:
                cx = (bbox[0] + bbox[2]) / 2.0
                cy = (bbox[1] + bbox[3]) / 2.0
                asyncio.create_task(
                    self._run_behavior_tick(
                        camera_str_id=camera_str_id,
                        camera_db_id=self._camera_db_id,
                        track_id=track_id,
                        cx=cx, cy=cy,
                        video_time=float(event["timestamp"]),
                        current_count=len(self.current_state),
                    ),
                    name=f"behavior-track-{track_id}",
                )

    async def _ensure_track_row(self, camera_str_id: str, event: dict[str, Any]) -> None:
        """Attach `db_track_id` + `track_confirmed` to this event.

        On first sighting of a (camera_id, track_id) pair, find-or-create the
        `tracks` row and cache its PK for the rest of the track's life. On
        every later event, reuse the cached PK and refresh last_seen_* on a
        frame throttle.

        The find-or-create is awaited (the very first event for a track must
        carry a db_track_id, otherwise the dispatch gates skip it); the
        throttled refresh is fire-and-forget, since nothing downstream reads
        last_seen_* synchronously.
        """
        track_id: int = event["track_id"]
        frame_number: int = event["frame_number"]
        timestamp = float(event["timestamp"])
        confidence = float(event["confidence"])

        # A crop on the event means event_emitter.py's throttle fired and a
        # file was written for THIS frame, at the path its convention defines.
        # No crop on the event means no file — deriving a path anyway would
        # point the review queue at something that isn't there.
        crop_path = (
            crop_path_for(track_id, frame_number, camera_str_id)
            if event.get("crop_bgr") is not None
            else None
        )

        db_track_id = self._track_db_ids.get(track_id)

        if db_track_id is None:
            db_track_id = await asyncio.to_thread(
                _find_or_create_track_row,
                camera_str_id, track_id, frame_number, timestamp, confidence,
                crop_path,
            )
            if db_track_id is None:
                # DB unavailable. Leave the event unenriched — the gates below
                # will skip this track exactly as they did before this fix,
                # rather than the whole detection stream failing.
                return
            self._track_db_ids[track_id] = db_track_id
            self._track_touch_frame[track_id] = frame_number
        else:
            last_touch = self._track_touch_frame.get(track_id, frame_number)
            throttle_elapsed = (frame_number - last_touch) >= self._track_update_interval
            # Also touch on any frame that carried a crop, even off-throttle.
            # The crop throttle and this DB throttle share an interval but not
            # a guaranteed phase (a failed crop extraction desynchronises
            # them), and a crop that lands between touches would otherwise
            # never reach best_crop_path — leaving the review queue with no
            # image for that track. Crops are already throttled upstream, so
            # this adds no meaningful write load.
            if throttle_elapsed or crop_path is not None:
                self._track_touch_frame[track_id] = frame_number
                self._spawn_background(
                    asyncio.to_thread(
                        _touch_track_row,
                        db_track_id, frame_number, timestamp, confidence,
                        crop_path,
                    ),
                    name=f"track-touch-{track_id}",
                )

        event["db_track_id"] = db_track_id
        # Every event reaching the emitter already represents a confirmed
        # track (event_emitter.py only ever receives ConfirmedTrack instances
        # — see its module docstring), so the meaningful question this flag
        # answers downstream is "is this event carrying a usable DB identity
        # yet", and it is set exactly when that becomes true.
        event["track_confirmed"] = True

    def _spawn_background(self, coro: Any, name: str) -> None:
        """Fire-and-forget a background task, keeping a strong reference.

        Used for DB writes (track touch) and, since Day 10, abandoned-object
        detector ticks — anything that must run off the hot path without
        being silently garbage-collected mid-flight.
        """
        task = asyncio.create_task(coro, name=name)
        self._pending_tasks.add(task)
        task.add_done_callback(self._pending_tasks.discard)

    async def check_and_expire_tracks(self) -> None:
        """Expire any track that hasn't been updated within `_expiry_seconds`.

        Called periodically by the pipeline_bridge expiry task.
        Expired tracks are removed from current_state and announced to clients.
        """
        now = time.monotonic()
        expired_ids = [
            tid
            for tid, state in self.current_state.items()
            if now - state["last_seen_wall_time"] > self._expiry_seconds
        ]

        for tid in expired_ids:
            state = self.current_state[tid]

            # Final last_seen_* write before dropping the cached PK, so the
            # `tracks` row ends up reflecting where the track actually ended
            # rather than wherever the last throttled refresh happened to land.
            db_track_id = self._track_db_ids.pop(tid, None)
            self._track_touch_frame.pop(tid, None)
            if db_track_id is not None:
                self._spawn_background(
                    asyncio.to_thread(
                        _touch_track_row,
                        db_track_id,
                        int(state["frame_number"]),
                        float(state["timestamp"]),
                        float(state["confidence"]),
                    ),
                    name=f"track-final-touch-{tid}",
                )

            del self.current_state[tid]
            self._reid_dispatched.discard(tid)  # allow re-processing if track returns
            self._face_dispatched.discard(tid)
            logger.debug("Track %d expired — broadcasting track_expired.", tid)
            await self.broadcast({
                "type": "track_expired",
                "camera_id": self._camera_id,
                "track_id": tid,
            })

        if expired_ids:
            logger.info("Expired %d tracks: %s", len(expired_ids), expired_ids)

        # Day 10: expire stale object positions the same way, on the same
        # 1s-interval task (pipeline_bridge._expiry_loop already calls this
        # method every second — no new periodic task needed). No broadcast:
        # object tracks have no dashboard live-view, they only ever produce
        # Alert rows via the detector, which manages its own Redis state
        # independently of this in-memory cache.
        expired_objects = [
            oid for oid, state in self._object_positions.items()
            if now - state["last_seen_wall_time"] > self._expiry_seconds
        ]
        for oid in expired_objects:
            del self._object_positions[oid]
            self._object_locks.pop(oid, None)
        if expired_objects:
            logger.debug("Expired %d stale object position(s): %s",
                        len(expired_objects), expired_objects)

    async def broadcast(self, message: dict[str, Any]) -> None:
        """Send a message to all active WebSocket clients.

        Dead clients are silently removed — one bad connection cannot block
        the broadcast for all other healthy clients.

        Args:
            message: JSON-serialisable dict to send.
        """
        dead: list[WebSocket] = []
        for ws in list(self._clients):
            success = await self._send_to(ws, message)
            if not success:
                dead.append(ws)

        for ws in dead:
            self._clients.discard(ws)
            logger.info("Removed dead WebSocket client. Active clients: %d", len(self._clients))

    # ──────────────────────────────────────────────────────────────────────
    # Internal helpers
    # ──────────────────────────────────────────────────────────────────────

    @staticmethod
    async def _send_to(websocket: WebSocket, message: dict[str, Any]) -> bool:
        """Attempt to send a JSON message to a single client with a timeout.

        Uses asyncio.wait_for with SEND_TIMEOUT_S so a hung or slow client
        cannot block the broadcast loop for all other clients.

        Args:
            websocket: Target WebSocket.
            message: Dict to serialise and send.

        Returns:
            True on success, False if the send failed or timed out.
        """
        try:
            await asyncio.wait_for(
                websocket.send_json(message),
                timeout=ConnectionManager.SEND_TIMEOUT_S,
            )
            return True
        except asyncio.TimeoutError:
            logger.warning(
                "WebSocket send timed out after %.1fs — dropping client.",
                ConnectionManager.SEND_TIMEOUT_S,
            )
            return False
        except Exception:
            return False

    @property
    def active_client_count(self) -> int:
        """Number of currently connected WebSocket clients."""
        return len(self._clients)

    @property
    def pipeline_has_active_tracks(self) -> bool:
        """True if any tracks are currently in state (not all expired)."""
        return len(self.current_state) > 0

    # ──────────────────────────────────────────────────────────────────
    # ReID integration (Day 6)
    # ──────────────────────────────────────────────────────────────────

    @staticmethod
    async def _run_reid(
        db_track_id: int,
        camera_str_id: str,
        crop_bgr,               # np.ndarray
        confidence: float,
        camera_db_id: int | None = None,
    ) -> None:
        """Background ReID task fired via asyncio.create_task().

        Completely off the detection hot path. Failures are logged but
        never propagate to the caller.
        """
        try:
            from backend.services.reid_matcher import resolve_identity
            result = await resolve_identity(
                track_db_id=db_track_id,
                camera_str_id=camera_str_id,
                camera_db_id=camera_db_id,
                crop_bgr=crop_bgr,
                best_confidence=confidence,
            )
            logger.debug(
                "ReID result for db_track=%d: decision=%s gp_id=%s",
                db_track_id,
                result.get("decision"),
                result.get("global_person_id"),
            )
        except Exception as exc:
            logger.error(
                "Background ReID task failed for db_track=%d: %s",
                db_track_id, exc,
            )

    @staticmethod
    async def _run_face_watchlist(
        db_track_id: int,
        camera_str_id: str,
        crop_bgr,               # np.ndarray
        camera_db_id: int | None = None,
    ) -> None:
        """Background face-watchlist task fired via asyncio.create_task().

        Completely off the detection hot path, parallel to _run_reid. Failures
        are logged but never propagate to the caller.
        """
        try:
            from backend.services.face_watchlist_matcher import resolve_watchlist_face
            result = await resolve_watchlist_face(
                track_db_id=db_track_id,
                camera_str_id=camera_str_id,
                camera_db_id=camera_db_id,
                crop_bgr=crop_bgr,
            )
            logger.debug(
                "Face watchlist result for db_track=%d: decision=%s",
                db_track_id, result.get("decision"),
            )
        except Exception as exc:
            logger.error(
                "Background face watchlist task failed for db_track=%d: %s",
                db_track_id, exc,
            )

    # ──────────────────────────────────────────────────────────────────
    # Behavior engine integration (Day 8)
    # ──────────────────────────────────────────────────────────────────

    @staticmethod
    async def _run_behavior_tick(
        camera_str_id: str,
        camera_db_id: int | None,
        track_id: int,
        cx: float,
        cy: float,
        video_time: float,
        current_count: int,
    ) -> None:
        """Background behavior-engine task fired via asyncio.create_task().

        Runs loitering (per-track) and crowd (per-camera) concurrently — they
        are independent detectors that both consume this same event, neither
        depends on the other's result. Completely off the detection hot path;
        failures are logged but never propagate to the caller.
        """
        try:
            from backend.services.loitering_detector import get_loitering_detector
            from backend.services.crowd_detector import get_crowd_detector

            loiter_task = get_loitering_detector().on_track_position(
                camera_str_id=camera_str_id, camera_db_id=camera_db_id,
                track_id=track_id, cx=cx, cy=cy, video_time=video_time,
            )
            crowd_task = get_crowd_detector().on_frame_tick(
                camera_str_id=camera_str_id, camera_db_id=camera_db_id,
                video_time=video_time, current_count=current_count,
            )
            loiter_result, crowd_result = await asyncio.gather(
                loiter_task, crowd_task, return_exceptions=True,
            )
            if isinstance(loiter_result, Exception):
                logger.error("Loitering tick failed for track=%d: %s", track_id, loiter_result)
            if isinstance(crowd_result, Exception):
                logger.error("Crowd tick failed for camera=%s: %s", camera_str_id, crowd_result)
        except Exception as exc:
            logger.error(
                "Background behavior-engine task failed for track=%d: %s", track_id, exc,
            )

    # ──────────────────────────────────────────────────────────────────
    # Abandoned-object detector integration (Day 10)
    # ──────────────────────────────────────────────────────────────────
    #
    # This is the wiring that was missing: previously the detector was
    # constructed at startup and never called (see abandoned_object_
    # detector.py's module docstring, pre-Day-10). Object-track events
    # arrive via pipeline_bridge._drain_loop routing event_type=="object_track"
    # here instead of to handle_detection_event() — see that method's Day 10
    # comment for why they're kept fully separate.

    async def handle_object_event(self, event: dict[str, Any]) -> None:
        """Process one object-track event from the pipeline.

        Runs on EVERY object event, like the Day 8 behavior-engine branch —
        the detector needs a continuous per-frame position stream for each
        object's whole visible lifetime, not a one-shot dispatch at
        confirmation. Two things happen, in order, both inside the single
        background task spawned at the end of this method:

          1. Proximity: is any currently-tracked PERSON within
             OWNERSHIP_RADIUS_PX of this object AND recently reported (within
             ABANDON_PROXIMITY_WINDOW_SEC of THIS event's video_time, not
             merely still unexpired by wall-clock — current_state's own
             expiry is wall-clock-based and can lag video_time badly on a
             fast/looped feed; see the comment at the computation below)?
             Computed synchronously here (cheap — in-memory only, no I/O)
             from self.current_state. If yes, on_person_positions() is
             called BEFORE on_object_update() in the background task — same
             ordering the eval harness enforces, and for the same reason:
             on_object_update
             reads the proximity marker on the way through its own logic, so
             a stale marker read before this tick's proximity is recorded
             would suppress the exact tick meant to refresh it.
          2. on_object_update() — the actual static/unattended/duration
             check that can fire an alert.

        Args:
            event: An object_track event from event_emitter.emit_objects()
                   (event_type, camera_id, track_id, frame_number, timestamp,
                   bbox, confidence, object_class, is_baseline).
        """
        if not _abandoned_ready:
            return

        bbox = event.get("bbox")
        if not bbox or len(bbox) != 4:
            return

        pipeline_track_id: int = event["track_id"]
        camera_str_id: str = event.get("camera_id", self._camera_id)
        object_track_id = f"{camera_str_id}:obj:{pipeline_track_id}"

        cx = (bbox[0] + bbox[2]) / 2.0
        cy = (bbox[1] + bbox[3]) / 2.0
        video_time = float(event["timestamp"])
        object_class = event.get("object_class", "object")
        is_baseline = bool(event.get("is_baseline", False))

        self._object_positions[object_track_id] = {
            "cx": cx, "cy": cy, "last_seen_wall_time": time.monotonic(),
        }

        # ── Proximity: cheap, synchronous, in-memory only ────────────────
        from backend.core.config import settings
        from backend.services.camera_calibration import get_px_per_meter

        px_per_meter, _calibration_method = get_px_per_meter(camera_str_id)
        radius_px = settings.OWNERSHIP_RADIUS_METERS * px_per_meter

        # A person entry counts as "nearby" only if BOTH its bbox distance
        # AND its recency (in video_time, not wall-clock) qualify.
        #
        # current_state's only staleness gate is wall-clock (check_and_
        # expire_tracks(), track_expiry_seconds — default 2s of REAL time).
        # On a live camera feed wall-clock and video_time move together, so
        # that alone would be fine. But main.py can process video far faster
        # than real-time (config.yaml's loop_video, or catching up after a
        # stall), and this method's own video_time can then advance by
        # minutes while wall-clock barely moves at all — a departed person's
        # LAST reported position would still sit in current_state, unexpired,
        # for the whole run, making every object tick see "someone nearby"
        # forever. That is exactly the wall-clock-vs-video_time class of bug
        # already fixed once in abandoned_object_detector.on_object_update
        # (see the comment there) — this is the same mistake made again one
        # layer up, caught by verify_abandoned_object_wiring.py rather than
        # by a slow feed in the field. Reusing ABANDON_PROXIMITY_WINDOW_SEC
        # keeps "how stale can a position be and still count" on the ONE
        # clock (video_time) and the ONE tunable, rather than adding a second.
        someone_nearby = False
        for person in self.current_state.values():
            pbbox = person.get("bbox")
            p_timestamp = person.get("timestamp")
            if not pbbox or len(pbbox) != 4 or p_timestamp is None:
                continue
            if abs(video_time - float(p_timestamp)) > settings.ABANDON_PROXIMITY_WINDOW_SEC:
                continue
            pcx = (pbbox[0] + pbbox[2]) / 2.0
            pcy = (pbbox[1] + pbbox[3]) / 2.0
            if math.hypot(cx - pcx, cy - pcy) <= radius_px:
                someone_nearby = True
                break

        object_lock = self._object_locks.setdefault(object_track_id, asyncio.Lock())

        self._spawn_background(
            self._run_abandoned_object_tick(
                camera_str_id=camera_str_id,
                camera_db_id=self._camera_db_id,
                object_track_id=object_track_id,
                object_class=object_class,
                cx=cx, cy=cy,
                video_time=video_time,
                is_baseline=is_baseline,
                someone_nearby=someone_nearby,
                object_lock=object_lock,
            ),
            name=f"abandoned-obj-{object_track_id}",
        )

    @staticmethod
    async def _run_abandoned_object_tick(
        camera_str_id: str,
        camera_db_id: int | None,
        object_track_id: str,
        object_class: str,
        cx: float,
        cy: float,
        video_time: float,
        is_baseline: bool,
        someone_nearby: bool,
        object_lock: asyncio.Lock,
    ) -> None:
        """Background abandoned-object task fired via asyncio.create_task().

        Completely off the detection hot path; failures are logged but never
        propagate to the caller — same posture as every other detector
        integration in this file.

        `object_lock` serializes this against every OTHER tick for the SAME
        object_track_id (see its allocation in handle_object_event for why —
        the detector's on_object_update() is a Redis read-modify-write, not
        an atomic operation, and was breaking under concurrent ticks for one
        object before this lock was added). Ticks for DIFFERENT objects hold
        different locks and run fully concurrently; this is a per-object
        gate, not a global one.
        """
        try:
            from backend.services.abandoned_object_detector import get_abandoned_object_detector
            detector = get_abandoned_object_detector()

            async with object_lock:
                # Proximity marker refreshed BEFORE the object tick reads it —
                # see handle_object_event's docstring for why the order matters.
                if someone_nearby:
                    await detector.on_person_positions(
                        camera_str_id=camera_str_id,
                        object_track_ids_with_nearby_person=[object_track_id],
                        video_time=video_time,
                    )

                result = await detector.on_object_update(
                    camera_str_id=camera_str_id,
                    camera_db_id=camera_db_id,
                    object_track_id=object_track_id,
                    object_class=object_class,
                    cx=cx, cy=cy,
                    video_time=video_time,
                    is_baseline=is_baseline,
                )
            logger.debug(
                "Abandoned-object tick for %s: decision=%s",
                object_track_id, result.decision,
            )
        except Exception as exc:
            logger.error(
                "Background abandoned-object task failed for %s: %s",
                object_track_id, exc,
            )
