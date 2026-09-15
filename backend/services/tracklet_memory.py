"""
backend/services/tracklet_memory.py — Re-ID Buffer Preservation & State Management.

Preserves the 40-frame majority buffer and cooldown status across occlusions and tracker ID swaps
using 48-dim HSV color histograms over a 3.0s time-to-live cache.
"""
from __future__ import annotations

import time
from collections import deque
from dataclasses import dataclass, field
from typing import Any, Optional

import cv2
import numpy as np

REID_SIM_THRESHOLD = 0.85
REID_MAX_AGE_SEC = 3.0
WINDOW_SIZE = 40
VIOLATION_COOLDOWN = 30.0
WARMUP_FRAMES = 10


@dataclass
class TrackState:
    track_id: int
    frame_count: int = 0
    frame_buffer: deque = field(default_factory=lambda: deque(maxlen=WINDOW_SIZE))
    _cooldown_until: float = 0.0
    plate_cache: Any = None

    def __post_init__(self):
        from .plate_cache import PlateCache

        if self.plate_cache is None:
            self.plate_cache = PlateCache()

    def push(self, n_riders: int) -> None:
        self.frame_count += 1
        self.frame_buffer.append(1 if n_riders >= 3 else 0)

    @property
    def majority_ratio(self) -> float:
        if not self.frame_buffer:
            return 0.0
        return sum(self.frame_buffer) / len(self.frame_buffer)

    def is_warmed_up(self) -> bool:
        return self.frame_count >= WARMUP_FRAMES

    def is_violation(self) -> bool:
        return (
            len(self.frame_buffer) >= (WINDOW_SIZE // 2)
            and self.majority_ratio >= 0.65
        )

    def can_alert(self, now: float) -> bool:
        return now >= self._cooldown_until

    def record_alert(self, now: float) -> None:
        self._cooldown_until = now + VIOLATION_COOLDOWN


@dataclass
class EvictedSnapshot:
    track_id: int
    hist: np.ndarray  # 48-dim HSV histogram unit vector
    evicted_at: float
    frame_buffer: deque
    plate_cache: Any
    cooldown_until: float


class TrackletMemory:
    def __init__(self):
        self._live: dict[int, TrackState] = {}
        self._evicted: list[EvictedSnapshot] = []

    @staticmethod
    def _hist(crop: np.ndarray) -> np.ndarray:
        if crop is None or crop.size == 0:
            return np.zeros(48, dtype=np.float32)
        hsv = cv2.cvtColor(crop, cv2.COLOR_BGR2HSV)
        vecs = [cv2.calcHist([hsv], [ch], None, [16], [0, 256]).flatten() for ch in range(3)]
        v = np.concatenate(vecs).astype(np.float32)
        n = np.linalg.norm(v)
        return v / n if n > 1e-6 else v

    def get_or_create(
        self, track_id: int, crop: np.ndarray, now: float
    ) -> TrackState:
        if track_id in self._live:
            return self._live[track_id]

        h = self._hist(crop)
        best_snap: Optional[EvictedSnapshot] = None
        best_sim = 0.0

        for snap in self._evicted:
            if now - snap.evicted_at > REID_MAX_AGE_SEC:
                continue
            sim = float(np.dot(h, snap.hist))
            if sim >= REID_SIM_THRESHOLD and sim > best_sim:
                best_sim = sim
                best_snap = snap

        if best_snap:
            self._evicted.remove(best_snap)
            state = TrackState(track_id=track_id)
            state.frame_buffer = best_snap.frame_buffer
            state.plate_cache = best_snap.plate_cache
            state._cooldown_until = best_snap.cooldown_until
            state.frame_count = len(best_snap.frame_buffer)
        else:
            state = TrackState(track_id=track_id)

        self._live[track_id] = state
        return state

    def evict(self, track_id: int, crop: np.ndarray, now: float) -> None:
        if track_id not in self._live:
            return
        s = self._live.pop(track_id)
        self._evicted.append(
            EvictedSnapshot(
                track_id=track_id,
                hist=self._hist(crop),
                evicted_at=now,
                frame_buffer=s.frame_buffer,
                plate_cache=s.plate_cache,
                cooldown_until=s._cooldown_until,
            )
        )
        self._evicted = [
            e for e in self._evicted if now - e.evicted_at <= REID_MAX_AGE_SEC
        ]
