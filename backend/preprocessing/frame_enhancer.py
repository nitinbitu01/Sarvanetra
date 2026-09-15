"""
backend/preprocessing/frame_enhancer.py — Weather Regime Detection & Image Enhancement.

Provides:
  - WeatherRegime (NORMAL, NIGHT_IR, RAIN, FOG)
  - enhance_frame: Inspects contrast, luminance, and gradient energy to apply
    conditional CLAHE, bilateral derain, or LAB-space gamma dehazing.
"""
from dataclasses import dataclass
from enum import Enum, auto

import cv2
import numpy as np


class WeatherRegime(Enum):
    NORMAL = auto()
    NIGHT_IR = auto()
    RAIN = auto()
    FOG = auto()


@dataclass
class EnhancedFrame:
    frame: np.ndarray
    regime: WeatherRegime
    enhancement_applied: str
    michelson_contrast: float
    mean_luminance: float


def detect_regime(frame_bgr: np.ndarray) -> tuple[WeatherRegime, float, float]:
    """Detects environmental regime from luminance, contrast, and edge energy."""
    small = cv2.resize(frame_bgr, (160, 90))
    gray = cv2.cvtColor(small, cv2.COLOR_BGR2GRAY)
    b, g, r = cv2.split(small.astype(np.float32))
    ch_std = float(np.std([b.mean(), g.mean(), r.mean()]))
    mean_lum = float(gray.mean())
    gmin, gmax = float(gray.min()), float(gray.max())
    michelson = (gmax - gmin) / (gmax + gmin + 1e-6)

    if ch_std < 4.0 and mean_lum < 80:
        return WeatherRegime.NIGHT_IR, michelson, mean_lum
    if michelson < 0.25:
        return WeatherRegime.FOG, michelson, mean_lum
    sobel_y = cv2.Sobel(gray, cv2.CV_32F, 0, 1, ksize=3)
    if float(np.abs(sobel_y).mean()) > 18.0:
        return WeatherRegime.RAIN, michelson, mean_lum
    return WeatherRegime.NORMAL, michelson, mean_lum


def _enhance_night_ir(frame: np.ndarray) -> tuple[np.ndarray, str]:
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    clahe = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(8, 8))
    enhanced = cv2.cvtColor(clahe.apply(gray), cv2.COLOR_GRAY2BGR)
    return enhanced, "CLAHE_night_ir"


def _enhance_fog(frame: np.ndarray) -> tuple[np.ndarray, str]:
    inv = 255 - frame
    lab = cv2.cvtColor(inv, cv2.COLOR_BGR2LAB)
    l, a, b = cv2.split(lab)
    clahe = cv2.createCLAHE(clipLimit=2.5, tileGridSize=(8, 8))
    lab_eq = cv2.merge([clahe.apply(l), a, b])
    dehazed = 255 - cv2.cvtColor(lab_eq, cv2.COLOR_LAB2BGR)
    gamma = 1.5
    lut = np.array(
        [min(255, int(((i / 255.0) ** (1.0 / gamma)) * 255)) for i in range(256)],
        dtype=np.uint8,
    )
    return cv2.LUT(dehazed, lut), "fog_dehaze_gamma"


def _enhance_rain(frame: np.ndarray) -> tuple[np.ndarray, str]:
    return cv2.bilateralFilter(frame, d=7, sigmaColor=50, sigmaSpace=50), "bilateral_derain"


def enhance_frame(frame_bgr: np.ndarray) -> EnhancedFrame:
    """Enhances frame according to detected weather regime."""
    regime, michelson, mean_lum = detect_regime(frame_bgr)
    if regime == WeatherRegime.NIGHT_IR:
        enhanced, method = _enhance_night_ir(frame_bgr)
    elif regime == WeatherRegime.FOG:
        enhanced, method = _enhance_fog(frame_bgr)
    elif regime == WeatherRegime.RAIN:
        enhanced, method = _enhance_rain(frame_bgr)
    else:
        enhanced, method = frame_bgr, "none"
    return EnhancedFrame(
        frame=enhanced,
        regime=regime,
        enhancement_applied=method,
        michelson_contrast=michelson,
        mean_luminance=mean_lum,
    )
