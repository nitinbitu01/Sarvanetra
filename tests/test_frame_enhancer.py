"""
tests/test_frame_enhancer.py — Frame Enhancer & Weather Regime Unit Tests (Tests 1-5).
"""
import cv2
import numpy as np
import pytest

from backend.preprocessing.frame_enhancer import (
    WeatherRegime,
    detect_regime,
    enhance_frame,
)


def test_night_ir_detection():
    # Grayscale-like, low luminance image -> NIGHT_IR
    frame = np.full((480, 640, 3), 40, dtype=np.uint8)
    e = enhance_frame(frame)
    assert e.regime == WeatherRegime.NIGHT_IR
    assert "CLAHE" in e.enhancement_applied


def test_fog_detection():
    # Low contrast washed-out image -> FOG (Michelson contrast < 0.25)
    frame = np.full((480, 640, 3), 150, dtype=np.uint8)
    frame[:50, :50] = 160  # tiny range: (160-150)/(160+150) = 10/310 = 0.032 < 0.25
    e = enhance_frame(frame)
    assert e.regime == WeatherRegime.FOG
    assert "dehaze" in e.enhancement_applied


def test_rain_detection():
    # Image with high vertical gradient energy -> RAIN
    frame = np.zeros((480, 640, 3), dtype=np.uint8)
    # Vertical streaks
    for y in range(0, 480, 4):
        frame[y : y + 2, :] = 255
    e = enhance_frame(frame)
    assert e.regime == WeatherRegime.RAIN
    assert "derain" in e.enhancement_applied


def test_clahe_improves_contrast():
    # Dark image with faint pattern
    frame = np.full((480, 640, 3), 30, dtype=np.uint8)
    frame[100:200, 100:200] = 50
    e = enhance_frame(frame)
    assert e.regime == WeatherRegime.NIGHT_IR
    # Contrast after enhancement should be >= original
    orig_grad = np.std(cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY))
    enh_grad = np.std(cv2.cvtColor(e.frame, cv2.COLOR_BGR2GRAY))
    assert enh_grad >= orig_grad


def test_no_double_enhancement_normal():
    # Normal colorful daytime image
    frame = np.zeros((480, 640, 3), dtype=np.uint8)
    frame[:, :, 0] = np.tile(np.linspace(0, 255, 640, dtype=np.uint8), (480, 1))
    frame[:, :, 1] = np.tile(np.linspace(50, 200, 480, dtype=np.uint8).reshape(-1, 1), (1, 640))
    frame[:, :, 2] = 120
    e = enhance_frame(frame)
    assert e.regime == WeatherRegime.NORMAL
    assert e.enhancement_applied == "none"
