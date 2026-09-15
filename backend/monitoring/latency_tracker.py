"""
backend/monitoring/latency_tracker.py — Per-Stage Latency & Budget Monitoring.

Tracks execution time for each pipeline stage and alerts if stage budgets are violated.
"""
from __future__ import annotations

import logging
import time
from contextlib import contextmanager
from typing import Iterator

logger = logging.getLogger(__name__)

STAGE_BUDGETS_MS = {
    "frame_enhance": 4.0,
    "reid": 2.0,
    "rider_count": 5.0,
    "helmet_detect": 3.0,
    "total_pipeline": 20.0,
}


class LatencyTracker:
    def __init__(self, camera_id: str):
        self.camera_id = camera_id
        self._latencies: dict[str, list[float]] = {}

    @contextmanager
    def measure(self, stage_name: str) -> Iterator[None]:
        t0 = time.perf_counter()
        try:
            yield
        finally:
            elapsed_ms = (time.perf_counter() - t0) * 1000.0
            if stage_name not in self._latencies:
                self._latencies[stage_name] = []
            self._latencies[stage_name].append(elapsed_ms)
            if len(self._latencies[stage_name]) > 300:
                self._latencies[stage_name].pop(0)

            budget = STAGE_BUDGETS_MS.get(stage_name, 25.0)
            if elapsed_ms > budget * 1.5:
                logger.warning(
                    "Stage '%s' exceeded latency budget for %s: %.2fms > %.2fms",
                    stage_name,
                    self.camera_id,
                    elapsed_ms,
                    budget,
                )

    def get_stage_p95(self, stage_name: str) -> float:
        vals = self._latencies.get(stage_name, [])
        if not vals:
            return 0.0
        return float(sorted(vals)[int(0.95 * len(vals))])
