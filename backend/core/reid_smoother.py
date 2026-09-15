"""
backend/core/reid_smoother.py — Thread-Safe Multi-View Temporal ReID Smoother.

Key Capabilities:
  1. Per-track thread-safe LRU FIFO buffer (deque maxlen=5).
  2. L2-normalized weighted temporal centroid pooling:
       e_track = sum(w_i * e_i) / ||sum(w_i * e_i)||
  3. Adaptive consensus gating: handles early tracklets (<3 frames) safely by
     routing them to review queue rather than premature auto-merge.
  4. Automatic TTL eviction of inactive tracks to prevent memory leaks.
"""
from __future__ import annotations

import logging
import time
from collections import deque
from dataclasses import dataclass, field
from threading import Lock
from typing import Any

import numpy as np

from backend.embedding_utils import normalize_l2

logger = logging.getLogger(__name__)

DEFAULT_BUFFER_SIZE = 5
DEFAULT_TTL_SECONDS = 120.0
DECAY_LAMBDA = 0.05


@dataclass
class TrackBuffer:
    embeddings: deque[np.ndarray] = field(default_factory=lambda: deque(maxlen=DEFAULT_BUFFER_SIZE))
    timestamps: deque[float] = field(default_factory=lambda: deque(maxlen=DEFAULT_BUFFER_SIZE))
    lock: Lock = field(default_factory=Lock)
    last_updated: float = field(default_factory=time.time)


def consensus_vote(
    similarities: list[float],
    theta: float,
    s_geo: float = 1.0,
    min_votes: int = 3,
) -> bool:
    """Adaptive consensus voting across multi-frame candidate similarities.

    Rules:
      1. If s_geo != 1.0 (teleportation violation) -> immediately REJECT.
      2. If fewer than 3 frames exist (<3 frames) -> return False (route to review, no auto-merge).
      3. Otherwise -> requires at least min_votes (or len(similarities)) to exceed theta.
    """
    if s_geo != 1.0:
        return False

    if len(similarities) < 3:
        return False

    votes = sum(1 for s in similarities if s >= theta)
    required = min(min_votes, len(similarities))
    return votes >= required


class ReidSmoother:
    """Thread-safe multi-view temporal smoother for cross-camera ReID tracklets."""

    def __init__(self, ttl_seconds: float = DEFAULT_TTL_SECONDS) -> None:
        self._buffers: dict[int, TrackBuffer] = {}
        self._global_lock = Lock()
        self.ttl = ttl_seconds

    def add_frame(self, track_id: int, embedding: np.ndarray, timestamp: float | None = None) -> None:
        """Add an L2-normalized embedding to the tracklet's rolling buffer."""
        ts = timestamp if timestamp is not None else time.time()
        norm_vec = normalize_l2(embedding)

        with self._global_lock:
            if track_id not in self._buffers:
                self._buffers[track_id] = TrackBuffer()
            buf = self._buffers[track_id]

        with buf.lock:
            buf.embeddings.append(norm_vec)
            buf.timestamps.append(ts)
            buf.last_updated = time.time()

    def get_centroid_embedding(self, track_id: int) -> np.ndarray | None:
        """Compute the weighted temporal centroid vector for a given track."""
        with self._global_lock:
            buf = self._buffers.get(track_id)
            if not buf:
                return None

        with buf.lock:
            if not buf.embeddings:
                return None

            now = time.time()
            weighted_sum = np.zeros_like(buf.embeddings[0], dtype=np.float32)
            
            for emb, ts in zip(buf.embeddings, buf.timestamps):
                weight = float(np.exp(-DECAY_LAMBDA * max(0.0, now - ts)))
                weighted_sum += weight * emb

            return normalize_l2(weighted_sum)

    def get_history_similarities(self, track_id: int, candidate_embedding: np.ndarray) -> list[float]:
        """Compute cosine similarity between each historical frame in the buffer and a candidate vector."""
        with self._global_lock:
            buf = self._buffers.get(track_id)
            if not buf:
                return []

        cand_norm = normalize_l2(candidate_embedding)
        with buf.lock:
            return [float(np.dot(emb, cand_norm)) for emb in buf.embeddings]

    def get_frame_count(self, track_id: int) -> int:
        with self._global_lock:
            buf = self._buffers.get(track_id)
            if not buf:
                return 0
        with buf.lock:
            return len(buf.embeddings)

    def evict_stale(self) -> int:
        """Evict track buffers that have been inactive longer than TTL. Returns number evicted."""
        now = time.time()
        evicted = 0
        with self._global_lock:
            stale_keys = [k for k, v in self._buffers.items() if now - v.last_updated > self.ttl]
            for k in stale_keys:
                del self._buffers[k]
                evicted += 1
        return evicted


# Global singleton instance
_smoother: ReidSmoother | None = None

def get_reid_smoother() -> ReidSmoother:
    global _smoother
    if _smoother is None:
        _smoother = ReidSmoother()
    return _smoother
