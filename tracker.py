"""
tracker.py — Track state management and 3-frame confirmation for Sentinel Gujarat.

This is the ONLY file that contains custom tracking logic. It owns the
TrackStateManager and the 3-consecutive-frame confirmation state machine.

Everything downstream (detector.py, event_emitter.py, main.py) only ever
receives ConfirmedTrack objects from this module — they have no visibility
into the confirmation rule, the counter state, or any unconfirmed activity.

Confirmation rule (deliberate and strict):
  - A track must appear in min_confirmation_frames *consecutive* processed
    frames with no gaps.
  - If a track is absent from tracker output for even 1 processed frame,
    its consecutive_frame_count resets to 0. This prevents a flickering
    false detection from eventually accumulating enough frames to confirm.
  - Once confirmed=True, confirmation is NOT revoked on future absences.
    BoT-SORT maintains IDs across brief occlusion via Kalman prediction,
    so a confirmed track reappearing under the same ID remains confirmed.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import Any

from detector import RawTrackResult

logger = logging.getLogger(__name__)


@dataclass
class ConfirmedTrack:
    """A track that has passed the 3-consecutive-frame confirmation rule.

    Only ConfirmedTrack instances are ever handed to event_emitter.py.
    """
    track_id: int
    bbox: list[float]       # [x1, y1, x2, y2]
    confidence: float
    frame_number: int
    timestamp: float        # frame_number / source_fps (video time, not wall-clock)


@dataclass
class _TrackState:
    """Internal per-track state. Never exposed outside this module."""
    consecutive_frame_count: int = 0
    confirmed: bool = False
    last_seen_frame: int = -1


class TrackStateManager:
    """Maintains per-track confirmation state across processed frames.

    Args:
        min_confirmation_frames: Number of consecutive processed frames a
            track must appear in before it is considered confirmed.
    """

    def __init__(self, min_confirmation_frames: int) -> None:
        self._min_frames = min_confirmation_frames
        # Internal state dict — keyed by track_id
        self._states: dict[int, _TrackState] = {}

    # ──────────────────────────────────────────────────────────────────────
    # Public API
    # ──────────────────────────────────────────────────────────────────────

    def update(
        self,
        raw_tracks: list[RawTrackResult],
        frame_number: int,
        source_fps: float,
    ) -> list[ConfirmedTrack]:
        """Apply the confirmation state machine and return confirmed tracks.

        Called once per processed frame.

        Args:
            raw_tracks: Raw detector output for this frame (may be empty).
            frame_number: Current raw frame index (from source video).
            source_fps: Video source FPS, used to compute timestamps.

        Returns:
            List of ConfirmedTrack for every track that is currently
            confirmed *and* visible in this frame. Empty list if none.
        """
        current_ids = {t.track_id for t in raw_tracks}

        # ── Step 1: Update state for every track seen in this frame ──────
        for raw in raw_tracks:
            state = self._states.setdefault(raw.track_id, _TrackState())
            state.consecutive_frame_count += 1
            state.last_seen_frame = frame_number

            # ── Confirmation gate ─────────────────────────────────────────
            if not state.confirmed and state.consecutive_frame_count >= self._min_frames:
                state.confirmed = True
                logger.info(
                    "Track %d confirmed after %d consecutive frames.",
                    raw.track_id,
                    state.consecutive_frame_count,
                )

        # ── Step 2: Reset counters for any track absent this frame ───────
        # Deliberate: even 1 missing processed frame resets an unconfirmed
        # track's streak to 0. Confirmed tracks are not affected — they stay
        # confirmed (BoT-SORT re-associates them via Kalman on reappearance).
        for track_id, state in self._states.items():
            if track_id not in current_ids:
                if not state.confirmed:
                    if state.consecutive_frame_count > 0:
                        logger.debug(
                            "Track %d absent — resetting consecutive_frame_count "
                            "from %d to 0 (was unconfirmed).",
                            track_id,
                            state.consecutive_frame_count,
                        )
                    state.consecutive_frame_count = 0
                # Confirmed tracks: counter not reset, confirmation preserved.

        # ── Step 3: Build confirmed output list ──────────────────────────
        timestamp = frame_number / source_fps if source_fps > 0 else 0.0
        confirmed: list[ConfirmedTrack] = []

        for raw in raw_tracks:
            state = self._states[raw.track_id]
            if state.confirmed:
                confirmed.append(
                    ConfirmedTrack(
                        track_id=raw.track_id,
                        bbox=raw.bbox,
                        confidence=raw.confidence,
                        frame_number=frame_number,
                        timestamp=timestamp,
                    )
                )

        logger.debug(
            "Frame %d: %d raw tracks → %d confirmed tracks.",
            frame_number, len(raw_tracks), len(confirmed),
        )
        return confirmed

    def get_state_snapshot(self) -> dict[int, dict[str, Any]]:
        """Return a read-only snapshot of current state for diagnostics.

        Not used in the main pipeline — available for testing / debugging.
        """
        return {
            tid: {
                "consecutive_frame_count": s.consecutive_frame_count,
                "confirmed": s.confirmed,
                "last_seen_frame": s.last_seen_frame,
            }
            for tid, s in self._states.items()
        }
