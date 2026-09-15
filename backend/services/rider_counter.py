"""
backend/services/rider_counter.py — Tri-Modal Rider Counting Engine.

Paths:
  - Path A: Hungarian Bipartite Matching (IoU / Overlap >= 0.20 + Centroid Distance + Confidence)
  - Path B: Contrast-Normalized (CLAHE) Head-Peak Projection with Background Subtraction
  - Path C: Haar Upper Body Cascade as Tiebreaker
  - Fusion: If Path A == Path B -> Result = Path A; Else -> Median(A, B, C)
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Optional

import cv2
import numpy as np
from scipy.optimize import linear_sum_assignment
from scipy.signal import find_peaks

# ── Constants ──────────────────────────────────────────────────────────
MIN_PERSON_BIKE_IOU = 0.20
HUN_IOU_W = 0.60
HUN_DIST_W = 0.30
HUN_CONF_W = 0.10
HEAD_SLICE_RATIO = 0.40
PEAK_PROM_RATIO = 0.20
PEAK_SEP_RATIO = 0.15
SIDE_ANGLE_THRESH_DEG = 30.0

_haar: Optional[Any] = None
_haar_attempted = False


def load_haar() -> None:
    global _haar, _haar_attempted
    _haar_attempted = True
    try:
        cascade_cls = getattr(cv2, "CascadeClassifier", None)
        if cascade_cls is not None and hasattr(cv2, "data") and hasattr(cv2.data, "haarcascades"):
            path = cv2.data.haarcascades + "haarcascade_upperbody.xml"
            _haar = cascade_cls(path)
    except Exception:
        _haar = None


@dataclass
class PersonBox:
    x1: float
    y1: float
    x2: float
    y2: float
    confidence: float

    @property
    def area(self) -> float:
        return max(0.0, self.x2 - self.x1) * max(0.0, self.y2 - self.y1)

    @property
    def centroid(self) -> tuple[float, float]:
        return ((self.x1 + self.x2) / 2.0, (self.y1 + self.y2) / 2.0)


@dataclass
class BikeBox:
    x1: float
    y1: float
    x2: float
    y2: float

    @property
    def width(self) -> float:
        return max(0.0, self.x2 - self.x1)

    @property
    def height(self) -> float:
        return max(0.0, self.y2 - self.y1)

    @property
    def centroid(self) -> tuple[float, float]:
        return ((self.x1 + self.x2) / 2.0, (self.y1 + self.y2) / 2.0)


def iou(p: PersonBox, b: BikeBox) -> float:
    """Calculates overlap between person box and bike bounding box."""
    ix1 = max(p.x1, b.x1)
    iy1 = max(p.y1, b.y1)
    ix2 = min(p.x2, b.x2)
    iy2 = min(p.y2, b.y2)
    inter = max(0.0, ix2 - ix1) * max(0.0, iy2 - iy1)
    if inter == 0.0:
        return 0.0
    overlap_person = inter / max(p.area, 1e-6)
    union = p.area + (b.width * b.height) - inter
    standard_iou = inter / max(union, 1e-6)
    return max(standard_iou, overlap_person)


def _path_a(persons: list[PersonBox], bike: BikeBox) -> int:
    candidates = [p for p in persons if iou(p, bike) >= MIN_PERSON_BIKE_IOU]
    if not candidates:
        return 0
    n = len(candidates)
    cost = np.zeros((n, n), dtype=np.float32)
    diag = np.hypot(bike.width, bike.height) + 1e-6
    bx, by = bike.centroid
    for i, p in enumerate(candidates):
        iou_s = iou(p, bike)
        px, py = p.centroid
        dist_s = float(np.clip(1.0 - np.hypot(px - bx, py - by) / diag, 0, 1))
        score = (HUN_IOU_W * iou_s) + (HUN_DIST_W * dist_s) + (HUN_CONF_W * p.confidence)
        cost[i, i] = 1.0 - score
    ri, ci = linear_sum_assignment(cost)
    return sum(1 for r, c in zip(ri, ci) if cost[r, c] < 0.70)


def _path_b(
    frame_bgr: np.ndarray,
    bike: BikeBox,
    camera_angle_deg: float = 0.0,
) -> int:
    H, W = frame_bgr.shape[:2]
    x1 = int(np.clip(bike.x1, 0, W))
    y1 = int(np.clip(bike.y1, 0, H))
    x2 = int(np.clip(bike.x2, 0, W))
    y2 = int(np.clip(bike.y2, 0, H))
    crop = frame_bgr[y1:y2, x1:x2]
    if crop.size == 0:
        return 0
    bh, bw = crop.shape[:2]

    gray = cv2.cvtColor(crop, cv2.COLOR_BGR2GRAY).astype(np.uint8)
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(4, 4))
    norm = clahe.apply(gray).astype(np.float32)
    blur = cv2.GaussianBlur(norm, (0, 0), sigmaX=max(bw / 4.0, 1.0))
    sub = np.clip(norm - blur + 128.0, 0.0, 255.0)

    sl = sub[: int(bh * HEAD_SLICE_RATIO), :]
    if sl.shape[0] == 0:
        return 0

    if abs(camera_angle_deg) >= SIDE_ANGLE_THRESH_DEG:
        profile = sl.sum(axis=0)
    else:
        profile = sl.sum(axis=0)

    p_min, p_max = float(profile.min()), float(profile.max())
    if (p_max - p_min) < 1.0:
        return 0
    profile_n = (profile - p_min) / (p_max - p_min + 1e-6)

    min_sep = max(1, int(PEAK_SEP_RATIO * bw))
    peaks, _ = find_peaks(profile_n, prominence=PEAK_PROM_RATIO, distance=min_sep)
    return int(len(peaks))


def _path_c(frame_bgr: np.ndarray, bike: BikeBox) -> int:
    global _haar, _haar_attempted
    if not _haar_attempted:
        load_haar()
    if _haar is None or not hasattr(_haar, "detectMultiScale"):
        return 0
    try:
        if _haar.empty():
            return 0
        H, W = frame_bgr.shape[:2]
        x1 = int(np.clip(bike.x1, 0, W))
        y1 = int(np.clip(bike.y1, 0, H))
        x2 = int(np.clip(bike.x2, 0, W))
        y2 = int(np.clip(bike.y2, 0, H))
        crop = frame_bgr[y1:y2, x1:x2]
        if crop.size == 0:
            return 0
        gray = cv2.cvtColor(crop, cv2.COLOR_BGR2GRAY)
        dets = _haar.detectMultiScale(
            gray,
            scaleFactor=1.05,
            minNeighbors=3,
            minSize=(15, 20),
            maxSize=(int(bike.width), int(bike.height)),
        )
        return len(dets) if len(dets) else 0
    except Exception:
        return 0


def count_riders(
    frame_bgr: np.ndarray,
    bike: BikeBox,
    persons: list[PersonBox],
    camera_angle_deg: float = 0.0,
) -> tuple[int, str]:
    """
    Returns (rider_count, method_description).
    Fusion logic:
      - Paths A and B agree -> use result directly (high confidence)
      - Paths A and B disagree -> median(A, B, C) via Haar tiebreaker
    """
    n_a = _path_a(persons, bike)
    n_b = _path_b(frame_bgr, bike, camera_angle_deg)
    if n_a == n_b:
        return n_a, f"ab_agree({n_a})"
    n_c = _path_c(frame_bgr, bike)
    result = int(np.median([n_a, n_b, n_c]))
    return result, f"haar_tiebreak(A={n_a},B={n_b},C={n_c}->{result})"
