"""
backend/scripts/night_enhancer_v2.py

Replaces night_enhancer.py.
Changes vs old process_frame_adaptive():
  • 4-tier classification: day / twilight / night_moderate / night_dark
  • All LUTs precomputed at __init__ — zero per-frame allocation
  • Day mode applies mild CLAHE (old code did nothing for day frames)
  • Night-dark uses double-pass CLAHE + bilateral + unsharp mask
  • Returns (enhanced_frame, lighting_label) for downstream logging
  • Thread-safe: all state is read-only after __init__
"""
from __future__ import annotations

import cv2
import numpy as np


class AdaptiveNightEnhancer:
    """
    Auto-detects lighting condition and applies the appropriate
    enhancement pipeline. No manual switching required.
    """

    # Brightness thresholds (mean pixel value of centre crop)
    _T_DARK     = 40    # below → night_dark
    _T_MODERATE = 70    # below → night_moderate
    _T_TWILIGHT = 110   # below → twilight
                        # above → day

    def __init__(self):
        # Precompute all gamma LUTs once — reused every frame
        self._luts: dict[float, np.ndarray] = {}
        for gamma in (0.4, 0.5, 0.6, 0.7, 0.8, 1.0, 1.8):
            inv   = 1.0 / gamma
            table = np.array(
                [((i / 255.0) ** inv) * 255 for i in range(256)],
                dtype=np.uint8,
            )
            self._luts[gamma] = table

        # Precompute sharpening kernel
        self._sharpen_kernel = np.array(
            [[-0.5, -0.5, -0.5],
             [-0.5,  5.0, -0.5],
             [-0.5, -0.5, -0.5]],
            dtype=np.float32,
        )

    # ── Public API ────────────────────────────────────────────────────────────

    def enhance(self, frame: np.ndarray) -> tuple[np.ndarray, str]:
        """
        Entry point. Returns (enhanced_frame, lighting_condition).
        lighting_condition ∈ {"day","twilight","night_moderate","night_dark"}
        """
        brightness = self._brightness(frame)
        condition  = self._classify(brightness)

        if condition == "day":
            out = self._day(frame)
        elif condition == "twilight":
            out = self._twilight(frame)
        elif condition == "night_moderate":
            out = self._night(frame, gamma=0.6)
        else:
            out = self._night_dark(frame)

        return out, condition

    # ── Brightness estimation ─────────────────────────────────────────────────

    def _brightness(self, frame: np.ndarray) -> float:
        """Fast: uses centre 1/3 crop to avoid black borders."""
        h, w = frame.shape[:2]
        crop = frame[h // 3: 2 * h // 3, w // 3: 2 * w // 3]
        gray = cv2.cvtColor(crop, cv2.COLOR_BGR2GRAY)
        return float(np.mean(gray))

    def _classify(self, b: float) -> str:
        if b < self._T_DARK:
            return "night_dark"
        if b < self._T_MODERATE:
            return "night_moderate"
        if b < self._T_TWILIGHT:
            return "twilight"
        return "day"

    # ── Per-condition pipelines ───────────────────────────────────────────────

    def _day(self, frame: np.ndarray) -> np.ndarray:
        """Mild CLAHE on L channel — preserves natural colour."""
        lab = cv2.cvtColor(frame, cv2.COLOR_BGR2LAB)
        l, a, b = cv2.split(lab)
        l = cv2.createCLAHE(clipLimit=1.5, tileGridSize=(8, 8)).apply(l)
        return cv2.cvtColor(cv2.merge([l, a, b]), cv2.COLOR_LAB2BGR)

    def _twilight(self, frame: np.ndarray) -> np.ndarray:
        """Moderate CLAHE + gentle gamma lift."""
        lab = cv2.cvtColor(frame, cv2.COLOR_BGR2LAB)
        l, a, b = cv2.split(lab)
        l = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(8, 8)).apply(l)
        enhanced = cv2.cvtColor(cv2.merge([l, a, b]), cv2.COLOR_LAB2BGR)
        return cv2.LUT(enhanced, self._luts[0.8])

    def _night(self, frame: np.ndarray, gamma: float = 0.6) -> np.ndarray:
        """Gamma lift → aggressive CLAHE → bilateral denoise."""
        corrected = cv2.LUT(frame, self._luts[gamma])
        lab = cv2.cvtColor(corrected, cv2.COLOR_BGR2LAB)
        l, a, b = cv2.split(lab)
        l = cv2.createCLAHE(clipLimit=4.0, tileGridSize=(4, 4)).apply(l)
        enhanced  = cv2.cvtColor(cv2.merge([l, a, b]), cv2.COLOR_LAB2BGR)
        return cv2.bilateralFilter(enhanced, 5, 50, 50)

    def _night_dark(self, frame: np.ndarray) -> np.ndarray:
        """Aggressive gamma → double-pass CLAHE → bilateral → unsharp."""
        # 1. Gamma
        corrected = cv2.LUT(frame, self._luts[0.4])
        # 2. Double CLAHE
        lab = cv2.cvtColor(corrected, cv2.COLOR_BGR2LAB)
        l, a, b = cv2.split(lab)
        l = cv2.createCLAHE(clipLimit=5.0, tileGridSize=(4, 4)).apply(l)
        l = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(8, 8)).apply(l)
        enhanced  = cv2.cvtColor(cv2.merge([l, a, b]), cv2.COLOR_LAB2BGR)
        # 3. Heavy bilateral
        denoised  = cv2.bilateralFilter(enhanced, 9, 75, 75)
        # 4. Unsharp mask (sharpening kernel)
        return cv2.filter2D(denoised, -1, self._sharpen_kernel)