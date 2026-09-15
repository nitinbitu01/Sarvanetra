"""
backend/services/indian_vehicle_refiner.py — Indian Traffic Vehicle Disambiguation & Hierarchy Refiner.

Solves the standard COCO-YOLO limitation where Indian-specific vehicles:
  - Auto-Rickshaws / Tuk-Tuks / Chhakdas are falsely classified as 'truck' or 'car'
  - Scooters (step-through) vs Motorcycles are lumped together
  - Tata Ace / Mini-Tempos (Chhota Hathi) are called 'truck'
  - Duplicate overlapping class boxes (e.g., 'bus' + 'truck' on same vehicle) are merged via NMS.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import List, Tuple
import numpy as np


@dataclass
class RefinedDetection:
    box: Tuple[int, int, int, int]  # (x1, y1, x2, y2)
    cls_name: str
    confidence: float
    indian_vehicle_type: str        # 'AUTO_RICKSHAW', 'SCOOTER', 'MOTORCYCLE', 'BUS', 'CAR', 'TEMPO', 'PEDESTRIAN'


def compute_iou(b1: Tuple[int, int, int, int], b2: Tuple[int, int, int, int]) -> float:
    x1 = max(b1[0], b2[0])
    y1 = max(b1[1], b2[1])
    x2 = min(b1[2], b2[2])
    y2 = min(b1[3], b2[3])

    inter_w = max(0, x2 - x1)
    inter_h = max(0, y2 - y1)
    inter_area = inter_w * inter_h

    area1 = (b1[2] - b1[0]) * (b1[3] - b1[1])
    area2 = (b2[2] - b2[0]) * (b2[3] - b2[1])
    union = area1 + area2 - inter_area
    if union <= 0:
        return 0.0
    return inter_area / union


def refine_indian_detections(
    raw_boxes: list[dict],
    image_shape: Tuple[int, int],
    min_confidence: float = 0.35,
) -> List[RefinedDetection]:
    """
    Takes raw YOLO detections, removes duplicate overlaps, and refines COCO classes
    to exact Indian traffic taxonomy.
    """
    img_h, img_w = image_shape[:2]

    # 1. Filter low-confidence noise and static background clutter
    filtered = [b for b in raw_boxes if b["conf"] >= min_confidence]

    # 2. Non-Maximum Suppression across overlapping categories (e.g. bus vs truck on same bus)
    filtered.sort(key=lambda x: x["conf"], reverse=True)
    keep: list[dict] = []
    for candidate in filtered:
        b_cand = candidate["xyxy"]
        overlap = False
        for chosen in keep:
            b_chosen = chosen["xyxy"]
            if compute_iou(b_cand, b_chosen) > 0.45:
                overlap = True
                break
        if not overlap:
            keep.append(candidate)

    # 3. Disambiguate Indian Vehicle Classes
    refined: List[RefinedDetection] = []
    for item in keep:
        x1, y1, x2, y2 = item["xyxy"]
        coco_cls = item["cls_name"].lower()
        conf = item["conf"]
        bw = x2 - x1
        bh = y2 - y1
        aspect_ratio = bw / max(bh, 1)

        final_cls = coco_cls.upper()

        # A. Auto-Rickshaw / Chhakda Disambiguation
        # COCO often classifies 3-wheelers as 'truck' or 'car' due to cargo bed / 3 wheels
        is_border = (x1 <= 15 or x2 >= img_w - 15)
        if coco_cls in ("truck", "car"):
            # Only reclassify if not a truncated border box and has typical 3-wheeler size & aspect ratio
            if not is_border and bw >= 120 and (0.65 <= aspect_ratio <= 1.15) and (bh < img_h * 0.45):
                final_cls = "AUTO-RICKSHAW"
            elif coco_cls == "truck" and not is_border and bh < img_h * 0.35:
                final_cls = "TEMPO / TATA ACE"
            elif coco_cls == "truck" and bh >= img_h * 0.35:
                final_cls = "HEAVY TRUCK"
            elif coco_cls == "car":
                final_cls = "CAR"
            elif coco_cls == "truck":
                final_cls = "TRUCK"


        # B. Bus Disambiguation
        elif coco_cls == "bus":
            final_cls = "BUS"

        # C. Two-Wheeler Disambiguation: Scooter vs Motorcycle
        elif coco_cls in ("motorcycle", "bicycle"):
            if coco_cls == "bicycle":
                final_cls = "BICYCLE"
            else:
                # Scooters (e.g. Activa) have wide step-through profile (AR > 1.3 when sideways)
                if aspect_ratio >= 1.25:
                    final_cls = "SCOOTER"
                else:
                    final_cls = "MOTORCYCLE"

        # D. Pedestrian
        elif coco_cls == "person":
            final_cls = "PEDESTRIAN"

        refined.append(
            RefinedDetection(
                box=(x1, y1, x2, y2),
                cls_name=coco_cls,
                confidence=conf,
                indian_vehicle_type=final_cls,
            )
        )

    return refined
