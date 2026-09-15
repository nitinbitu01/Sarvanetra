"""
backend/services/frame_selector.py

Quality gate for plate crops before CRNN inference.
Replaces the raw Laplacian threshold in the old engine.

Score = weighted composite of:
  • Sharpness  (Laplacian variance)   — 50%
  • Contrast   (std of intensity)     — 30%
  • Brightness (proximity to ideal)   — 20%
  • Size check (minimum px)           — hard gate

Changes vs old MIN_STRIP_SHARPNESS approach:
  • 3 quality dimensions instead of 1
  • Explicit minimum-size gate prevents 10×5 phantom crops
  • Returns numeric score for downstream logging
"""
from __future__ import annotations

import cv2
import numpy as np


class PlateFrameSelector:
    """Scores plate crops for quality. Only best crops reach CRNN."""

    def __init__(
        self,
        min_sharpness: float = 15.0,
        min_size:      tuple[int, int] = (12, 4),
    ):
        self.min_sharpness = min_sharpness
        self.min_size      = min_size   # (width, height)

    # ── Individual metrics ────────────────────────────────────────────────────

    def compute_sharpness(self, crop: np.ndarray) -> float:
        gray = cv2.cvtColor(crop, cv2.COLOR_BGR2GRAY) if crop.ndim == 3 else crop
        return float(cv2.Laplacian(gray, cv2.CV_64F).var())

    def compute_contrast(self, crop: np.ndarray) -> float:
        gray = cv2.cvtColor(crop, cv2.COLOR_BGR2GRAY) if crop.ndim == 3 else crop
        return float(np.std(gray))

    def compute_brightness_score(self, crop: np.ndarray) -> float:
        """
        Returns 1.0 for ideal brightness (100–180), lower otherwise.
        Penalises both too-dark (night noise) and overexposed (blown-out).
        """
        gray = cv2.cvtColor(crop, cv2.COLOR_BGR2GRAY) if crop.ndim == 3 else crop
        mean = float(np.mean(gray))
        if 100 <= mean <= 180:
            return 1.0
        if 60  <= mean < 100 or 180 < mean <= 220:
            return 0.7
        return 0.3

    # ── Composite score ───────────────────────────────────────────────────────

    def score(self, crop: np.ndarray) -> float:
        """
        Returns quality score in [0, 100].
        Hard-zero if crop is smaller than min_size.
        """
        if crop is None or crop.size == 0:
            return 0.0

        h, w = crop.shape[:2]
        if w < self.min_size[0] or h < self.min_size[1]:
            return 0.0

        sharpness  = self.compute_sharpness(crop)
        contrast   = self.compute_contrast(crop)
        brightness = self.compute_brightness_score(crop)

        raw = (
            sharpness  * 0.50
            + contrast * 0.30
            + brightness * 20.0 * 0.20   # scale brightness to ~0-20 range
        )
        return min(float(raw), 100.0)

    def is_acceptable(self, crop: np.ndarray) -> bool:
        return self.score(crop) >= self.min_sharpness