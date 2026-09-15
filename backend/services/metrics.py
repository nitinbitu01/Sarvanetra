"""
backend/services/metrics.py — Lightweight counters + rolling latency for Day 8
observability (§4). NOT a Prometheus client — that's explicitly out of scope
for this pass (see Day 8 prompt §7). This is a plain in-process store exposed
via GET /api/v1/debug/stats, which a later pass can swap for real Prometheus
metrics without changing detector code (they only ever call `counters.incr`
and `counters.observe_latency`).

Deliberately process-local (not Redis-backed): counters resetting on restart
is an acceptable trade for this pass — they're for "is this thing ticking
right now" observability, not an audit trail. The audit_log table already
covers anything that needs to survive a restart.

Thread/async safety: incr() and observe_latency() are called from many
concurrent asyncio.create_task() detector ticks. Plain dict/list mutation is
safe here because CPython's GIL makes each individual dict increment and
list append atomic — no explicit lock needed for this access pattern.
"""
from __future__ import annotations

import time
from collections import defaultdict, deque
from typing import Any


class MetricsRegistry:
    def __init__(self, latency_window: int = 200) -> None:
        # Simple named counters, e.g. "frames_processed_total{camera_id=CAM-01}"
        # collapsed to a flat key "frames_processed_total:CAM-01" — no label
        # sets/cardinality machinery, this is a placeholder for real Prometheus.
        self._counters: dict[str, int] = defaultdict(int)
        # Rolling window of recent latencies per detector, for a simple avg.
        self._latencies: dict[str, deque[float]] = defaultdict(lambda: deque(maxlen=latency_window))
        self._started_at = time.monotonic()

    def incr(self, name: str, labels: dict[str, str] | None = None, amount: int = 1) -> None:
        key = self._key(name, labels)
        self._counters[key] += amount

    def observe_latency(self, detector: str, latency_ms: float) -> None:
        self._latencies[detector].append(latency_ms)

    def snapshot(self) -> dict[str, Any]:
        latency_avgs = {
            detector: round(sum(vals) / len(vals), 2)
            for detector, vals in self._latencies.items()
            if vals
        }
        return {
            "uptime_seconds": round(time.monotonic() - self._started_at, 1),
            "counters": dict(sorted(self._counters.items())),
            "behavior_detector_latency_ms_avg": latency_avgs,
        }

    @staticmethod
    def _key(name: str, labels: dict[str, str] | None) -> str:
        if not labels:
            return name
        label_str = ",".join(f"{k}={v}" for k, v in sorted(labels.items()))
        return f"{name}{{{label_str}}}"


# Module-level singleton — imported as `from backend.services.metrics import counters`
counters = MetricsRegistry()
