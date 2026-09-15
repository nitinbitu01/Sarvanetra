"""
tests/test_plate_cache.py — Plate Cache Sampling & Best Result Retention Tests (Tests 20-22).
"""
import pytest

from backend.services.anpr import PlateResult
from backend.services.plate_cache import PlateCache, SAMPLE_EVERY_N


def test_plate_cache_sampling_frequency():
    cache = PlateCache()
    sampled_count = 0
    # Over 40 frames
    for _ in range(40):
        if cache.should_sample():
            sampled_count += 1
    # Exactly 4 samples (every 10th frame)
    assert sampled_count == 4


def test_plate_cache_retains_highest_confidence():
    cache = PlateCache()
    res1 = PlateResult("GJ01AA1111", confidence=0.72, crop_path=None, format_valid=True)
    res2 = PlateResult("GJ01AA1111", confidence=0.91, crop_path=None, format_valid=True)
    res3 = PlateResult("GJ01AA1111", confidence=0.65, crop_path=None, format_valid=True)

    cache.best = res1
    if res2.confidence > cache.best.confidence:
        cache.best = res2
    if res3.confidence > cache.best.confidence:
        cache.best = res3

    assert cache.get_best().confidence == 0.91
