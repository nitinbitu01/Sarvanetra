"""
backend/services/plate_utils_v2.py

Lighting-aware CRNN input preprocessor.
Replaces the fixed CLAHE+resize in _preprocess_plate_strip().

Changes:
  • Auto-classifies lighting from crop brightness (or accepts explicit label)
  • Night pipeline: gamma → CLAHE → denoise → adaptive threshold → morph
  • Overexposed pipeline: darken → CLAHE
  • Day pipeline: mild CLAHE → light denoise
  • All pipelines output (target_h × target_w) uint8 grayscale
    compatible with existing CRNN input spec.
"""
from __future__ import annotations

import cv2
import numpy as np


class PlatePreprocessorV2:
    """Adaptive CRNN input preprocessor. Thread-safe (no mutable state)."""

    def __init__(self, target_h: int = 32, target_w: int = 128):
        self.target_h = target_h
        self.target_w = target_w

    # ── Public API ────────────────────────────────────────────────────────────

    def preprocess(
        self,
        plate_crop: np.ndarray,
        lighting: str = "auto",
    ) -> np.ndarray | None:
        """
        Returns (target_h × target_w) uint8 grayscale ready for CRNN.
        lighting: "auto" | "day" | "twilight" | "night_moderate" |
                  "night_dark" | "night" | "overexposed"
        """
        if plate_crop is None or plate_crop.size == 0:
            return None

        # Resolve "auto"
        if lighting == "auto":
            lighting = self._classify_lighting(plate_crop)

        if lighting in ("night_dark", "night_moderate", "night"):
            processed = self._night_pipeline(plate_crop)
        elif lighting == "overexposed":
            processed = self._overexposed_pipeline(plate_crop)
        else:   # day / twilight / unknown
            processed = self._day_pipeline(plate_crop)

        if processed is None:
            return None

        return cv2.resize(
            processed, (self.target_w, self.target_h),
            interpolation=cv2.INTER_AREA,
        )

    # ── Lighting classifier ───────────────────────────────────────────────────

    def _classify_lighting(self, crop: np.ndarray) -> str:
        gray = cv2.cvtColor(crop, cv2.COLOR_BGR2GRAY) if crop.ndim == 3 else crop
        mean = float(np.mean(gray))
        if mean < 80:
            return "night"
        if mean > 200:
            return "overexposed"
        return "day"

    # ── Preprocessing pipelines ───────────────────────────────────────────────

    def _day_pipeline(self, crop: np.ndarray) -> np.ndarray:
        gray     = cv2.cvtColor(crop, cv2.COLOR_BGR2GRAY) if crop.ndim == 3 else crop
        enhanced = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(4, 4)).apply(gray)
        return cv2.fastNlMeansDenoising(enhanced, h=8)

    def _night_pipeline(self, crop: np.ndarray) -> np.ndarray:
        # 1. Gamma lift
        gamma = 0.5
        lut   = np.array(
            [((i / 255.0) ** (1.0 / gamma)) * 255 for i in range(256)],
            dtype=np.uint8,
        )
        boosted  = cv2.LUT(crop if crop.ndim == 3 else cv2.cvtColor(crop, cv2.COLOR_GRAY2BGR), lut)
        gray     = cv2.cvtColor(boosted, cv2.COLOR_BGR2GRAY)

        # 2. Strong CLAHE
        enhanced = cv2.createCLAHE(clipLimit=5.0, tileGridSize=(4, 4)).apply(gray)

        # 3. Denoise
        denoised = cv2.fastNlMeansDenoising(enhanced, h=15)

        # 4. Adaptive threshold (handles headlight hotspots)
        binary   = cv2.adaptiveThreshold(
            denoised, 255,
            cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
            cv2.THRESH_BINARY,
            blockSize=15, C=4,
        )

        # 5. Morphological close (fill broken character strokes)
        kernel  = cv2.getStructuringElement(cv2.MORPH_RECT, (2, 2))
        cleaned = cv2.morphologyEx(binary, cv2.MORPH_CLOSE, kernel)
        return cleaned

    def _overexposed_pipeline(self, crop: np.ndarray) -> np.ndarray:
        gray    = cv2.cvtColor(crop, cv2.COLOR_BGR2GRAY) if crop.ndim == 3 else crop
        gamma   = 1.8
        lut     = np.array(
            [((i / 255.0) ** gamma) * 255 for i in range(256)],
            dtype=np.uint8,
        )
        darkened = cv2.LUT(gray, lut)
        return cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8)).apply(darkened)