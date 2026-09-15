"""
event_emitter.py — Structured output for Sentinel Gujarat.

Day 2 changes:
  - Accepts optional `event_queue: queue.Queue` — events pushed to queue
    in addition to JSONL, enabling the FastAPI bridge to consume them.
  - Accepts `camera_id: str` — added to every event (Day 2 schema extension).
  - `emit()` accepts `override_timestamp` — used by the loop-video path in
    main.py to ensure monotonic timestamps across video loop restarts.

Output format contract (DO NOT change field names — Day 2+ reads these):
  {
      "event_type":   "person_track",
      "camera_id":    str,
      "track_id":     int,
      "frame_number": int,
      "timestamp":    float,
      "bbox":         [x1, y1, x2, y2],
      "confidence":   float
  }

No "confirmed" field — its presence implies confirmation. Do not re-add it.

Crop path convention (backend/connection_manager.crop_path_for mirrors this
EXACTLY - change both together or Track.best_crop_path points at nothing):
  output/crops/{camera_id}/track_{track_id}/frame_{frame_number}.jpg

The camera_id segment is load-bearing: BoT-SORT track ids are per-camera
locals that restart at 1 on every feed, so without it two cameras' crops of
different people land in the same track_N directory.

In-memory queue enrichment (additive — JSONL is NOT affected):
  Events pushed to `event_queue` may carry ONE extra key beyond the seven
  above:

      "crop_bgr":  np.ndarray (BGR person crop) — present only on the frames
                   where a crop was already being extracted anyway (see the
                   crop_save_interval_frames throttle), absent otherwise.

  This is the same crop object already written to disk, handed on by
  reference rather than re-extracted — the consumer (backend/connection_
  manager.py) runs in the same process, so there is no serialization
  boundary to cross. It is deliberately NOT attached on every frame: a
  480x640x3 uint8 crop is ~0.9 MB, and a 500-slot queue holding one per
  event would be several hundred MB of resident memory.

  The JSONL line is written BEFORE enrichment and always contains exactly
  the seven contract fields — a numpy array is not JSON-serialisable and
  must never reach json.dumps(). The remaining pipeline-set fields
  (`db_track_id`, `track_confirmed`) are added downstream by
  connection_manager.py, which is what actually creates the DB Track row.

Day 10 addition — object-track events (separate contract, own JSONL file):
  emit_objects() feeds the abandoned-object detector (backend/services/
  abandoned_object_detector.py), which — before this — was constructed at
  startup but never called: nothing fed it a track. This is a SEPARATE
  event stream, not a variant of the person_track contract above:

  {
      "event_type":   "object_track",
      "camera_id":    str,
      "track_id":     int,
      "frame_number": int,
      "timestamp":    float,
      "bbox":         [x1, y1, x2, y2],
      "confidence":   float,
      "object_class": str,     # "backpack" | "handbag" | "suitcase"
      "is_baseline":  bool     # True = present at pipeline warm-up,
                                # can never fire (see config.yaml's
                                # abandoned_object.warmup_sec)
  }

  Written to its own JSONL file (output.object_events_file) rather than
  interleaved into events.jsonl, so nothing that already parses
  events.jsonl expecting the seven-field person_track shape has to change
  or filter. Queue-pushed the same way person events are, for
  connection_manager.py to consume — no crop enrichment, this detector
  never looks at pixels, only positions.
"""

from __future__ import annotations

import json
import logging
import queue as stdlib_queue
from pathlib import Path
from typing import Any

import cv2
import numpy as np

from tracker import ConfirmedTrack

logger = logging.getLogger(__name__)


def _fs_safe(camera_id: str) -> str:
    """Make a camera id safe to use as a single path segment.

    Camera ids come from config and are normally plain ("CAM_01"), but a
    stray "/" or ".." would otherwise write crops outside the crops dir.
    """
    cleaned = "".join(
        ch if (ch.isalnum() or ch in "-_") else "_" for ch in str(camera_id)
    ).strip("._")
    return cleaned or "unknown_camera"


class EventEmitter:
    """Writes JSONL events and throttled crops for confirmed tracks.

    Args:
        cfg: Full parsed config dict.
        event_queue: Optional thread-safe queue for server mode.
                     When provided, every event is also put() here
                     so the FastAPI bridge can consume it in real-time.
                     Pass None for CLI / JSONL-only mode.
        camera_id: Camera identifier string, added to every event.
    """

    def __init__(
        self,
        cfg: dict[str, Any],
        event_queue: "stdlib_queue.Queue[dict] | None" = None,
        camera_id: str = "CAM-01",
    ) -> None:
        output_cfg = cfg["output"]
        self._events_file = Path(output_cfg["events_file"])
        self._crops_dir = Path(output_cfg["crops_dir"])
        self._crop_interval = output_cfg["crop_save_interval_frames"]
        self._event_queue = event_queue
        self._camera_id = camera_id

        # Day 10: object-track events go to their own JSONL file, not
        # interleaved into events_file — see module docstring. Falls back to
        # a sibling file next to events_file if output.object_events_file
        # isn't set, so an older config.yaml without the new key still works.
        object_events_path = output_cfg.get("object_events_file")
        self._object_events_file = (
            Path(object_events_path) if object_events_path
            else self._events_file.with_name("object_events.jsonl")
        )

        self._first_crop_saved: set[int] = set()
        self._last_crop_frame: dict[int, int] = {}

        self._ensure_dirs()
        self._events_fh = self._events_file.open("a", encoding="utf-8")
        self._object_events_fh = self._object_events_file.open("a", encoding="utf-8")
        logger.info(
            "EventEmitter ready. Events → %s | Object events → %s | camera_id: %s | queue: %s",
            self._events_file,
            self._object_events_file,
            camera_id,
            "enabled" if event_queue is not None else "disabled (JSONL only)",
        )
        logger.info(
            "Crops → %s (interval: every %d processed frames)",
            self._crops_dir, self._crop_interval,
        )

    def _ensure_dirs(self) -> None:
        self._events_file.parent.mkdir(parents=True, exist_ok=True)
        self._object_events_file.parent.mkdir(parents=True, exist_ok=True)
        self._crops_dir.mkdir(parents=True, exist_ok=True)

    # ──────────────────────────────────────────────────────────────────────
    # Public API
    # ──────────────────────────────────────────────────────────────────────

    def emit(
        self,
        confirmed_tracks: list[ConfirmedTrack],
        frame: np.ndarray,
        override_timestamp: float | None = None,
    ) -> None:
        """Process all confirmed tracks for one frame.

        Writes one JSON line per track, pushes to queue if provided,
        and saves crops where throttle allows.

        Args:
            confirmed_tracks: Output from TrackStateManager.update().
            frame: Current BGR frame from OpenCV (used for crop extraction).
            override_timestamp: When set, replaces track.timestamp in the
                emitted event. Used by loop-video mode to provide monotonic
                timestamps across video restart boundaries.
        """
        for track in confirmed_tracks:
            ts = override_timestamp if override_timestamp is not None else track.timestamp
            event = self._build_event(track, ts)

            # JSONL first, with exactly the seven contract fields. Enrichment
            # below must never reach the file — see module docstring.
            self._write_jsonl(event, track.track_id)

            crop = self._maybe_save_crop(track, frame)
            if crop is not None:
                event["crop_bgr"] = crop

            self._enqueue_event(event, track.track_id)

    def emit_objects(
        self,
        confirmed_objects: list[ConfirmedTrack],
        object_classes: dict[int, str],
        object_baseline: dict[int, bool],
        override_timestamp: float | None = None,
    ) -> None:
        """Process all confirmed object tracks for one frame.

        Separate from emit() — writes to object_events_file (not events_file)
        with the object_track event_type. See module docstring for the
        contract. No crop handling: the abandoned-object detector consumes
        only positions (cx, cy, video_time), never pixels.

        Args:
            confirmed_objects: Output from a TrackStateManager fed only
                object-class detections (see detector.OBJECT_CLASS_IDS).
            object_classes: track_id -> COCO class name ("backpack" etc.),
                looked up by the caller from detector.OBJECT_CLASS_NAMES.
            object_baseline: track_id -> is_baseline, decided once per track
                by the caller (main.py) from when it was first observed
                relative to pipeline start — this class has no notion of
                "pipeline start" itself and does not compute the flag.
            override_timestamp: Same monotonic-timestamp override as emit(),
                for the loop-video path.
        """
        for track in confirmed_objects:
            ts = override_timestamp if override_timestamp is not None else track.timestamp
            event: dict[str, Any] = {
                "event_type": "object_track",
                "camera_id": self._camera_id,
                "track_id": track.track_id,
                "frame_number": track.frame_number,
                "timestamp": round(ts, 4),
                "bbox": [round(v, 2) for v in track.bbox],
                "confidence": round(track.confidence, 4),
                "object_class": object_classes.get(track.track_id, "object"),
                "is_baseline": bool(object_baseline.get(track.track_id, False)),
            }

            try:
                self._object_events_fh.write(json.dumps(event) + "\n")
                self._object_events_fh.flush()
            except (OSError, ValueError) as exc:
                logger.error(
                    "EventEmitter: object JSONL write failed for track %d: %s",
                    track.track_id, exc,
                )

            if self._event_queue is not None:
                try:
                    self._event_queue.put_nowait(event)
                except stdlib_queue.Full:
                    logger.warning(
                        "Event queue full — dropping object event for track %d",
                        track.track_id,
                    )

    def close(self) -> None:
        """Flush and close the events file. Call at pipeline shutdown."""
        self._events_fh.flush()
        self._events_fh.close()
        self._object_events_fh.flush()
        self._object_events_fh.close()
        logger.info(
            "EventEmitter closed. Events written to %s and %s",
            self._events_file, self._object_events_file,
        )

    # ──────────────────────────────────────────────────────────────────────
    # Internal helpers
    # ──────────────────────────────────────────────────────────────────────

    def _build_event(self, track: ConfirmedTrack, timestamp: float) -> dict[str, Any]:
        """Build the seven-field contract event dict for one confirmed track."""
        return {
            "event_type": "person_track",
            "camera_id": self._camera_id,
            "track_id": track.track_id,
            "frame_number": track.frame_number,
            "timestamp": round(timestamp, 4),
            "bbox": [round(v, 2) for v in track.bbox],
            "confidence": round(track.confidence, 4),
        }

    def _write_jsonl(self, event: dict[str, Any], track_id: int) -> None:
        """Append one JSON event line.

        Errors here (disk full, closed file handle) must NOT propagate to the
        detection loop — a logging failure should never stop person tracking.

        Must be called BEFORE any in-memory enrichment: `crop_bgr` is a numpy
        array and json.dumps() would raise on it.
        """
        try:
            self._events_fh.write(json.dumps(event) + "\n")
            self._events_fh.flush()  # ensure event survives a crash
        except (OSError, ValueError) as exc:
            # ValueError: "I/O operation on closed file" — treat same as OSError
            logger.error(
                "EventEmitter: JSONL write failed for track %d (disk full / I/O error): %s",
                track_id, exc,
            )

    def _enqueue_event(self, event: dict[str, Any], track_id: int) -> None:
        """Push the (possibly crop-enriched) event to the real-time queue."""
        if self._event_queue is None:
            return
        try:
            self._event_queue.put_nowait(event)
        except stdlib_queue.Full:
            logger.warning(
                "Event queue full — dropping event for track %d", track_id
            )

    def _maybe_save_crop(
        self, track: ConfirmedTrack, frame: np.ndarray
    ) -> np.ndarray | None:
        """Save a crop JPEG if throttle conditions are met.

        Guarded: crop-save failures must not propagate to the detection loop.

        Returns:
            The extracted BGR crop when this call was inside the throttle
            window and extraction succeeded, else None. The caller attaches it
            to the queued event so downstream consumers (ReID, face watchlist)
            get a crop without re-reading it back off disk. Returned even when
            cv2.imwrite() fails — an unwritable disk is no reason to also
            withhold a perfectly good in-memory crop from the live consumers.
        """
        if frame is None or frame.size == 0:
            return None

        is_first = track.track_id not in self._first_crop_saved
        last_saved = self._last_crop_frame.get(track.track_id, -self._crop_interval)
        interval_elapsed = (track.frame_number - last_saved) >= self._crop_interval

        if not (is_first or interval_elapsed):
            return None

        try:
            crop = self._extract_crop(frame, track.bbox)
            if crop is None:
                logger.debug(
                    "Track %d frame %d: crop extraction failed (invalid bbox).",
                    track.track_id, track.frame_number,
                )
                return None

            # Camera id is part of the path, NOT just the track id.
            #
            # BoT-SORT track ids are LOCAL to one camera and restart from 1 on
            # every feed, so with more than one camera running,
            # output/crops/track_1/ received crops of a different person from
            # each camera, interleaved into one directory. Two consequences,
            # both silent:
            #   * Track.best_crop_path could point at a crop of someone from
            #     another camera entirely - the image an officer compares in
            #     the review queue.
            #   * Any ReID validation set built from these folders is
            #     corrupt by construction, because "same folder" no longer
            #     means "same person".
            # It also made cross-camera pair extraction impossible: nothing
            # in the path recorded which camera a crop came from.
            crop_path = (
                self._crops_dir
                / _fs_safe(self._camera_id)
                / f"track_{track.track_id}"
                / f"frame_{track.frame_number}.jpg"
            )
            crop_path.parent.mkdir(parents=True, exist_ok=True)

            success = cv2.imwrite(str(crop_path), crop)
            if success:
                logger.debug("Crop saved: %s", crop_path)
            else:
                logger.warning(
                    "cv2.imwrite returned False for track %d — crop not saved: %s",
                    track.track_id, crop_path,
                )

            # Advance the throttle regardless of imwrite success: this call
            # DID do the extraction work and DID hand a crop to the live
            # consumers, so it counts as this track's crop for this interval.
            # (Pre-existing behaviour only advanced on imwrite success, which
            # meant a full disk turned the throttle off and re-extracted a
            # crop on every single frame.)
            self._first_crop_saved.add(track.track_id)
            self._last_crop_frame[track.track_id] = track.frame_number
            return crop
        except OSError as exc:
            logger.error(
                "Crop save I/O error for track %d frame %d: %s",
                track.track_id, track.frame_number, exc,
            )
            return None
        except Exception as exc:
            logger.warning(
                "Unexpected error saving crop for track %d: %s",
                track.track_id, exc,
            )
            return None

    @staticmethod
    def _extract_crop(frame: np.ndarray, bbox: list[float]) -> np.ndarray | None:
        """Extract and return the person crop from the frame."""
        h, w = frame.shape[:2]
        x1, y1, x2, y2 = (int(round(v)) for v in bbox)
        x1, y1 = max(0, x1), max(0, y1)
        x2, y2 = min(w, x2), min(h, y2)
        if x2 <= x1 or y2 <= y1:
            return None
        return frame[y1:y2, x1:x2].copy()
