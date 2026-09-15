"""
backend/services/frame_buffer.py — per-camera rolling frame buffer (Day 13).

Feeds evidence_capture.py, which needs frames from BEFORE an alert fired —
by which time the detection loop has long since discarded them. This keeps a
bounded, time-ordered window of recent frames per camera.

WHY JPEG, NOT RAW
  A 640x480 BGR frame is 921 KB. The window has to span PRE+POST plus slack
  (~105s), and at 5 fps that is ~525 frames — roughly 480 MB PER CAMERA held
  resident, permanently, whether or not anything ever fires. JPEG-encoded at
  q85 the same frame is ~40 KB, so the same window costs ~21 MB. The clip is
  re-encoded to video on capture anyway, so nothing is lost by not holding
  raw pixels; the only cost is a decode pass at capture time (a few hundred
  frames, once, off the hot path).

  Encoding happens on the producer side, on the detection thread. At 5 fps
  that is ~2-5 ms per frame against a 200 ms budget.

WHY WALL-CLOCK IS THE SLICING KEY
  Each entry carries BOTH a wall-clock timestamp and the pipeline's
  video_time. Slicing is done on wall-clock because that is the only clock
  every alert has: Alert.created_at is set on every row, whereas video_time
  appears only in some detectors' meta_json (loitering/crowd/abandoned record
  it; the face-watchlist path does not). One key that works for all four
  beats four special cases. video_time is retained alongside purely for
  diagnostics — it is what the detector logs speak in.

  Note this makes the buffer correct for LIVE feeds. A looped or
  faster-than-real-time file source advances video_time faster than
  wall-clock; the buffer still yields a coherent, correctly-ordered clip of
  what was recently processed, which is what evidence needs.

SIZING IS DERIVED, NOT CONFIGURED
  capacity_seconds = PRE + POST + MARGIN. Exposing an independent buffer-size
  setting would let it drift below PRE+POST, at which point the pre-roll has
  already been evicted by the time we slice (POST seconds after the event)
  and every clip silently loses its lead-in.

THREAD SAFETY
  The producer is the detection thread (main.run_detection_pipeline); the
  consumer is an asyncio task. deque.append is atomic under the GIL, but a
  slice is not — an eviction mid-iteration would raise or tear. All access
  goes through a lock, and the consumer copies out the entries it needs
  rather than holding the lock across decode/encode work.
"""
from __future__ import annotations

import logging
import threading
import time
from collections import deque
from dataclasses import dataclass
from typing import Any

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class BufferedFrame:
    """One buffered frame. `jpeg` is encoded bytes, not a numpy array."""
    wall_time: float      # time.time() at capture — the slicing key
    video_time: float     # pipeline video clock — diagnostics only
    frame_number: int
    jpeg: bytes


class CameraFrameBuffer:
    """Bounded, time-ordered ring buffer of recent frames for ONE camera."""

    def __init__(self, camera_id: str, capacity_seconds: float, jpeg_quality: int = 85,
                 max_width: int = 0) -> None:
        self.camera_id = camera_id
        self.capacity_seconds = capacity_seconds
        self.jpeg_quality = jpeg_quality
        # Downscale before JPEG encoding when the frame is wider than this.
        # 0 disables (keep native resolution).
        #
        # WHY THIS EXISTS: add_frame() runs on the DETECTION thread for every
        # processed frame, and cv2.imencode at 1080p was measured at 16.2 ms -
        # MORE than the YOLO detect+track pass itself (11.7 ms). That put a
        # ~28 ms floor under every frame and was the single largest limit on
        # how many cameras could run at once, while looking like a GPU problem.
        # Encoding at 960x540 costs 4.3 ms, freeing ~12 ms per frame.
        self.max_width = int(max_width or 0)
        self._frames: deque[BufferedFrame] = deque()
        self._lock = threading.Lock()
        self._dropped_encode_failures = 0

    def add_frame(self, frame: Any, video_time: float, frame_number: int) -> bool:
        """Encode and append one frame, evicting anything older than the window.

        Called from the detection thread on every processed frame. Never
        raises — a buffering failure must not take down detection.

        Returns True if the frame was buffered.
        """
        if frame is None or getattr(frame, "size", 0) == 0:
            return False
        try:
            import cv2

            # Downscale first when configured. Evidence clips stay legible at
            # 960px wide, and this is the difference between ~16 ms and
            # ~4 ms of detection-thread time per frame.
            if self.max_width and frame.shape[1] > self.max_width:
                scale = self.max_width / float(frame.shape[1])
                frame = cv2.resize(
                    frame,
                    (self.max_width, max(1, int(round(frame.shape[0] * scale)))),
                    interpolation=cv2.INTER_AREA,
                )

            ok, encoded = cv2.imencode(
                ".jpg", frame, [int(cv2.IMWRITE_JPEG_QUALITY), int(self.jpeg_quality)]
            )
            if not ok:
                self._dropped_encode_failures += 1
                return False

            entry = BufferedFrame(
                wall_time=time.time(),
                video_time=float(video_time),
                frame_number=int(frame_number),
                jpeg=encoded.tobytes(),
            )
            with self._lock:
                self._frames.append(entry)
                self._evict_locked(entry.wall_time)
            return True
        except Exception as exc:
            self._dropped_encode_failures += 1
            logger.debug("Frame buffer encode failed for %s: %s", self.camera_id, exc)
            return False

    def _evict_locked(self, now: float) -> None:
        """Drop frames older than the window. Caller must hold the lock."""
        cutoff = now - self.capacity_seconds
        frames = self._frames
        while frames and frames[0].wall_time < cutoff:
            frames.popleft()

    def slice_range(self, start_wall: float, end_wall: float) -> list[BufferedFrame]:
        """Return buffered frames with start_wall <= wall_time <= end_wall.

        Returns whatever actually exists — possibly fewer than requested, or
        empty. Deliberately does NOT pad, error, or synthesise frames: a clip
        that begins 8 seconds before the event because that is all the buffer
        held is honest; one padded to 30s with black frames is not.
        """
        with self._lock:
            return [f for f in self._frames if start_wall <= f.wall_time <= end_wall]

    def stats(self) -> dict[str, Any]:
        with self._lock:
            n = len(self._frames)
            oldest = self._frames[0].wall_time if n else None
            newest = self._frames[-1].wall_time if n else None
            nbytes = sum(len(f.jpeg) for f in self._frames)
        return {
            "camera_id": self.camera_id,
            "frames": n,
            "span_seconds": round(newest - oldest, 2) if n >= 2 else 0.0,
            "bytes": nbytes,
            "megabytes": round(nbytes / (1024 * 1024), 2),
            "capacity_seconds": self.capacity_seconds,
            "encode_failures": self._dropped_encode_failures,
        }

    def clear(self) -> None:
        with self._lock:
            self._frames.clear()


class FrameBufferRegistry:
    """Process-wide registry of per-camera buffers, created on demand."""

    def __init__(self) -> None:
        self._buffers: dict[str, CameraFrameBuffer] = {}
        self._lock = threading.Lock()

    @staticmethod
    def capacity_seconds() -> float:
        """PRE + POST + MARGIN — derived so it can never be too small.

        The slice happens POST seconds after the event, so the buffer must
        still hold event-time-minus-PRE at that moment.
        """
        from backend.core.config import settings

        return (
            settings.EVIDENCE_PRE_SECONDS
            + settings.EVIDENCE_POST_SECONDS
            + settings.EVIDENCE_BUFFER_MARGIN_SECONDS
        )

    def get(self, camera_id: str) -> CameraFrameBuffer:
        with self._lock:
            buf = self._buffers.get(camera_id)
            if buf is None:
                from backend.core.config import settings

                buf = CameraFrameBuffer(
                    camera_id=camera_id,
                    max_width=getattr(settings, "EVIDENCE_BUFFER_MAX_WIDTH", 0),
                    capacity_seconds=self.capacity_seconds(),
                    jpeg_quality=settings.EVIDENCE_JPEG_QUALITY,
                )
                self._buffers[camera_id] = buf
                logger.info(
                    "Frame buffer created for camera %s (window=%.0fs, jpeg_q=%d).",
                    camera_id, buf.capacity_seconds, buf.jpeg_quality,
                )
            return buf

    def all_stats(self) -> list[dict[str, Any]]:
        with self._lock:
            buffers = list(self._buffers.values())
        return [b.stats() for b in buffers]


_registry: FrameBufferRegistry | None = None


def get_frame_buffer_registry() -> FrameBufferRegistry:
    global _registry
    if _registry is None:
        _registry = FrameBufferRegistry()
    return _registry


def buffer_frame(camera_id: str, frame: Any, video_time: float, frame_number: int) -> bool:
    """Convenience entry point for the detection loop."""
    return get_frame_buffer_registry().get(camera_id).add_frame(
        frame, video_time, frame_number
    )
