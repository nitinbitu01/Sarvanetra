"""
backend/monitoring/tamper_sentinel.py — Async Camera Tampering & Optical Diagnostics Sentinel (v15.0.0)

Closes Gap X: All tamper checks run on a 1-in-30 frame sample in a dedicated background
executor — NEVER blocking the detection pipeline.
"""
from __future__ import annotations

import asyncio
import time
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass
from enum import Enum, auto
from typing import Optional

import cv2
import numpy as np

try:
    from skimage.metrics import structural_similarity as _ssim
    _skimage_ok = True
except ImportError:
    _skimage_ok = False

TAMPER_SAMPLE_EVERY_N = 30
_tamper_executor = ThreadPoolExecutor(max_workers=1, thread_name_prefix="tamper")


class TamperType(Enum):
    HEALTHY = auto()
    LENS_BLUR_DEFOCUS = auto()    # Laplacian var < 15.0
    LENS_BLINDING_GLARE = auto()  # Overexposed > 85%
    CAMERA_OCCLUSION = auto()     # Mean luminance < 10.0
    CAMERA_PTZ_SHIFT = auto()     # SSIM vs keyframe < 0.60
    FROZEN_FEED = auto()          # Temporal diff < 0.02


@dataclass
class TamperResult:
    tamper_type: TamperType
    metric_value: float
    threshold: float
    camera_id: str
    timestamp: float
    is_healthy: bool


def _check_sync(
    frame_bgr: np.ndarray,
    reference_frame: Optional[np.ndarray],
    prev_frame: Optional[np.ndarray],
    camera_id: str,
    now: float,
) -> TamperResult:
    gray = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2GRAY)

    # 1. Occlusion: near-black or covered lens
    mean_lum = float(gray.mean())
    if mean_lum < 10.0:
        return TamperResult(TamperType.CAMERA_OCCLUSION, mean_lum, 10.0, camera_id, now, False)

    # 2. Blinding / glare
    overexp_ratio = float(np.sum(gray > 245) / max(gray.size, 1))
    if overexp_ratio > 0.85:
        return TamperResult(TamperType.LENS_BLINDING_GLARE, overexp_ratio, 0.85, camera_id, now, False)

    # 3. Defocus / blur
    lap_var = float(cv2.Laplacian(gray, cv2.CV_64F).var())
    if lap_var < 15.0:
        return TamperResult(TamperType.LENS_BLUR_DEFOCUS, lap_var, 15.0, camera_id, now, False)


    # 4. Frozen feed
    if prev_frame is not None:
        prev_gray = cv2.cvtColor(prev_frame, cv2.COLOR_BGR2GRAY).astype(np.float32)
        curr_gray = gray.astype(np.float32)
        diff = float(np.abs(curr_gray - prev_gray).mean()) / 255.0
        if diff < 0.005:
            return TamperResult(TamperType.FROZEN_FEED, diff, 0.005, camera_id, now, False)

    # 5. PTZ displacement / camera shifted
    if reference_frame is not None and _skimage_ok:
        ref_gray = cv2.cvtColor(
            cv2.resize(reference_frame, (gray.shape[1], gray.shape[0])),
            cv2.COLOR_BGR2GRAY,
        )
        ssim_val = float(_ssim(gray, ref_gray))
        if ssim_val < 0.60:
            return TamperResult(TamperType.CAMERA_PTZ_SHIFT, ssim_val, 0.60, camera_id, now, False)

    return TamperResult(TamperType.HEALTHY, 1.0, 1.0, camera_id, now, True)


class TamperSentinel:
    """Per-camera async tamper monitor with 1/30 sampling."""

    def __init__(self, camera_id: str, ws_broadcast_fn=None):
        self.camera_id = camera_id
        self._frame_counter = 0
        self._reference: Optional[np.ndarray] = None
        self._prev_frame: Optional[np.ndarray] = None
        self._last_result: Optional[TamperResult] = None
        self._ws_broadcast = ws_broadcast_fn

    async def check_async(self, frame_bgr: np.ndarray, now: float) -> Optional[TamperResult]:
        self._frame_counter += 1
        if (self._frame_counter % TAMPER_SAMPLE_EVERY_N) != 0:
            return self._last_result

        if self._reference is None:
            self._reference = frame_bgr.copy()

        loop = asyncio.get_event_loop()
        result = await loop.run_in_executor(
            _tamper_executor,
            _check_sync,
            frame_bgr,
            self._reference,
            self._prev_frame,
            self.camera_id,
            now,
        )
        self._prev_frame = frame_bgr.copy()
        self._last_result = result

        if not result.is_healthy and self._ws_broadcast:
            try:
                await self._ws_broadcast({
                    "type": "CAMERA_HEALTH_ALERT",
                    "camera_id": self.camera_id,
                    "tamper_type": result.tamper_type.name,
                    "metric": result.metric_value,
                    "threshold": result.threshold,
                    "timestamp": now,
                })
            except Exception:
                pass

        return result
