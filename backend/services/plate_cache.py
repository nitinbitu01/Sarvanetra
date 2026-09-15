"""
backend/services/plate_cache.py — Sampled License Plate Cache.

Samples plate crop every 10 frames and retains the highest confidence reading.
"""
from dataclasses import dataclass, field
from typing import Optional

from .anpr import PlateResult, read_plate

SAMPLE_EVERY_N = 10


@dataclass
class PlateCache:
    best: Optional[PlateResult] = None
    _frame_counter: int = field(default=0, repr=False)

    def should_sample(self) -> bool:
        self._frame_counter += 1
        return (self._frame_counter % SAMPLE_EVERY_N) == 0

    def update(self, crop_path: str, min_conf: float = 0.75) -> Optional[PlateResult]:
        """Synchronous update for robust in-loop and test execution."""
        result = read_plate(crop_path, min_conf)
        if result.text and (self.best is None or result.confidence > self.best.confidence):
            self.best = result
        return result

    async def update_async(self, crop_path: str, min_conf: float = 0.75) -> Optional[PlateResult]:
        """Async update for non-blocking worker queues."""
        return self.update(crop_path, min_conf)

    def get_best(self) -> Optional[PlateResult]:
        return self.best
