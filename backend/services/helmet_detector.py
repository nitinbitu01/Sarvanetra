"""
backend/services/helmet_detector.py — Per-Rider Head-Crop Helmet Classifier.

Extracts the upper 35% of each assigned rider bounding box and performs YOLOv8 helmet classification.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

import cv2
import numpy as np

try:
    from ultralytics import YOLO as _YOLO

    _yolo_ok = True
except ImportError:
    _yolo_ok = False

HELMET_CLASS_ID = 0
NO_HELMET_CLASS_ID = 1
HELMET_CONF_THRESH = 0.60


@dataclass
class HelmetResult:
    rider_index: int
    has_helmet: bool
    confidence: float
    label: str  # "Helmet" | "No Helmet" | "Unknown"


_helmet_model = None


def load_helmet_model(weights: str) -> None:
    global _helmet_model
    if _yolo_ok:
        _helmet_model = _YOLO(weights)


def detect_helmets(
    frame_bgr: np.ndarray,
    rider_boxes: list[tuple[float, float, float, float]],
) -> list[HelmetResult]:
    """Classifies helmet presence for each rider by inspecting their head region crop."""
    results = []
    for idx, (x1, y1, x2, y2) in enumerate(rider_boxes):
        hb = y2 - y1
        x1_i, y1_i = max(0, int(x1)), max(0, int(y1))
        x2_i, y2_i = min(frame_bgr.shape[1], int(x2)), min(frame_bgr.shape[0], int(y1 + (hb * 0.35)))

        if (x2_i <= x1_i) or (y2_i <= y1_i) or (frame_bgr is None) or (_helmet_model is None):
            results.append(HelmetResult(idx, False, 0.0, "Unknown"))
            continue

        head = frame_bgr[y1_i:y2_i, x1_i:x2_i]
        if head is None or head.size == 0:
            results.append(HelmetResult(idx, False, 0.0, "Unknown"))
            continue

        preds = _helmet_model(head, verbose=False)
        if not preds or not len(preds[0].boxes):
            results.append(HelmetResult(idx, False, 0.0, "Unknown"))
            continue

        best = max(preds[0].boxes, key=lambda b: float(b.conf))
        conf = float(best.conf)
        cls = int(best.cls)
        if conf < HELMET_CONF_THRESH:
            results.append(HelmetResult(idx, False, conf, "Unknown"))
        elif cls == HELMET_CLASS_ID:
            results.append(HelmetResult(idx, True, conf, "Helmet"))
        else:
            results.append(HelmetResult(idx, False, conf, "No Helmet"))
    return results


def count_no_helmet(helmet_results: list[HelmetResult]) -> int:
    return sum(1 for r in helmet_results if not r.has_helmet and r.label != "Unknown")
