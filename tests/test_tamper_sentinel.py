"""
tests/test_tamper_sentinel.py — Async Camera Tampering Sentinel Tests (v15.0.0).
"""
import numpy as np
import pytest

from backend.monitoring.tamper_sentinel import (
    TamperSentinel,
    TamperType,
    _check_sync,
)


def test_blur_defocus_detection():
    # Gaussian blur -> low Laplacian variance
    blurred = np.full((100, 100, 3), 128, dtype=np.uint8)
    res = _check_sync(blurred, None, None, "CAM_01", 100.0)
    assert res.tamper_type in (TamperType.LENS_BLUR_DEFOCUS, TamperType.CAMERA_OCCLUSION)
    assert res.is_healthy is False


def test_glare_blinding_detection():
    # Saturated white frame (> 85% > 245)
    glare = np.full((100, 100, 3), 250, dtype=np.uint8)
    res = _check_sync(glare, None, None, "CAM_01", 100.0)
    assert res.tamper_type == TamperType.LENS_BLINDING_GLARE
    assert res.is_healthy is False


def test_camera_occlusion_detection():
    # Dark black covered lens (mean < 10)
    dark = np.full((100, 100, 3), 5, dtype=np.uint8)
    res = _check_sync(dark, None, None, "CAM_01", 100.0)
    assert res.tamper_type == TamperType.CAMERA_OCCLUSION
    assert res.is_healthy is False


@pytest.mark.asyncio
async def test_async_1_in_30_sampling():
    alerts = []

    async def mock_broadcast(payload):
        alerts.append(payload)

    sentinel = TamperSentinel("CAM_01", ws_broadcast_fn=mock_broadcast)
    dark = np.full((50, 50, 3), 5, dtype=np.uint8)

    # Frame 1 to 29: Skipped by sampling
    for _ in range(29):
        r = await sentinel.check_async(dark, now=10.0)

    assert len(alerts) == 0

    # Frame 30: Executes check -> fires alert
    r = await sentinel.check_async(dark, now=10.0)
    assert r is not None
    assert r.is_healthy is False
    assert len(alerts) == 1
