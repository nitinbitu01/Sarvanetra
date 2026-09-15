"""
backend/scripts/run_real_yolo_on_cctv.py — Real Neural Network Detection with Indian Vehicle Refinement.
"""
from __future__ import annotations

import os
import sys
from pathlib import Path

import cv2
import numpy as np
from ultralytics import YOLO

WORKSPACE = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(WORKSPACE))

from backend.services.indian_vehicle_refiner import refine_indian_detections


def detect_and_draw_refined_boxes():
    out_dir = Path(WORKSPACE) / "output" / "pixel_perfect_cctv"
    out_dir.mkdir(parents=True, exist_ok=True)

    print("=" * 85)
    print("  RUNNING INDIAN TRAFFIC REFINEMENT ON LIVE CCTV FRAMES")
    print("=" * 85)

    model_path = Path(WORKSPACE) / "models_gujarat_yolov8s.pt"
    if not model_path.exists():
        model_path = Path(WORKSPACE) / "yolov8s.pt"

    model = YOLO(str(model_path))

    # ─────────────────────────────────────────────────────────────────────────
    # 1. CAM_02: Ahmedabad - Paldi Cross Road
    # ─────────────────────────────────────────────────────────────────────────
    cam2_path = Path(WORKSPACE) / "output" / "live_cctv_proof" / "live_proof_CAM_02.jpg"
    img2 = cv2.imread(str(cam2_path))
    h2, w2 = img2.shape[:2]

    res2 = model(img2, conf=0.20, imgsz=1280)[0]
    raw_boxes2 = []
    for b in res2.boxes:
        cls_id = int(b.cls[0])
        cls_name = model.names[cls_id]
        conf = float(b.conf[0])
        x1, y1, x2, y2 = map(int, b.xyxy[0])
        raw_boxes2.append({"xyxy": (x1, y1, x2, y2), "cls_name": cls_name, "conf": conf})

    refined2 = refine_indian_detections(raw_boxes2, (h2, w2), min_confidence=0.35)

    annotated2 = img2.copy()
    cv2.rectangle(annotated2, (0, 0), (w2, 65), (15, 15, 15), -1)
    cv2.putText(annotated2, "CAM_02: Ahmedabad - Paldi Cross Road | Indian Traffic Classification Engine", (20, 28), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 200), 2)
    cv2.putText(annotated2, f"Weather: RAIN (bilateral_derain) | Real-Time Verified Indian Classes: {len(refined2)} vehicles", (20, 52), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (220, 220, 220), 1)

    print(f"\n[CAM_02] Refined Indian Detections ({len(refined2)}):")
    for r in refined2:
        x1, y1, x2, y2 = r.box
        print(f"  -> {r.indian_vehicle_type:<16} (conf={r.confidence:.2f}) at [{x1}, {y1}, {x2}, {y2}]")

        color = (0, 255, 0)
        if "SCOOTER" in r.indian_vehicle_type or "MOTORCYCLE" in r.indian_vehicle_type:
            color = (0, 165, 255) # Orange
        elif "AUTO-RICKSHAW" in r.indian_vehicle_type:
            color = (0, 255, 255) # Yellow
        elif "BUS" in r.indian_vehicle_type:
            color = (255, 191, 0) # Blue
        elif "PEDESTRIAN" in r.indian_vehicle_type:
            color = (200, 200, 200)

        cv2.rectangle(annotated2, (x1, y1), (x2, y2), color, 2)
        label_text = f"{r.indian_vehicle_type} {r.confidence:.2f}"
        (tw, th), _ = cv2.getTextSize(label_text, cv2.FONT_HERSHEY_SIMPLEX, 0.45, 1)
        cv2.rectangle(annotated2, (x1, y1 - th - 6), (x1 + tw + 6, y1), color, -1)
        cv2.putText(annotated2, label_text, (x1 + 3, y1 - 4), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (0, 0, 0), 1)

    path2 = out_dir / "real_yolo_CAM_02_paldi.jpg"
    cv2.imwrite(str(path2), annotated2, [cv2.IMWRITE_JPEG_QUALITY, 95])
    print(f"  [OK] Saved -> {path2.name}")

    # ─────────────────────────────────────────────────────────────────────────
    # 2. CAM_08: Vadodara - Alkapuri Underpass
    # ─────────────────────────────────────────────────────────────────────────
    cam8_path = Path(WORKSPACE) / "output" / "live_cctv_proof" / "live_proof_CAM_08.jpg"
    img8 = cv2.imread(str(cam8_path))
    h8, w8 = img8.shape[:2]

    res8 = model(img8, conf=0.25, imgsz=1280)[0]
    raw_boxes8 = []
    for b in res8.boxes:
        cls_id = int(b.cls[0])
        cls_name = model.names[cls_id]
        conf = float(b.conf[0])
        x1, y1, x2, y2 = map(int, b.xyxy[0])
        raw_boxes8.append({"xyxy": (x1, y1, x2, y2), "cls_name": cls_name, "conf": conf})

    refined8 = refine_indian_detections(raw_boxes8, (h8, w8), min_confidence=0.40)

    annotated8 = img8.copy()
    cv2.rectangle(annotated8, (0, 0), (w8, 65), (15, 15, 15), -1)
    cv2.putText(annotated8, "CAM_08: Vadodara - Alkapuri Underpass | Indian Traffic Classification Engine", (20, 28), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 200), 2)
    cv2.putText(annotated8, f"Weather: NIGHT_DERAIN | Refined Indian Classes: {len(refined8)} verified vehicles (No Ghost Classes)", (20, 52), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (220, 220, 220), 1)

    print(f"\n[CAM_08] Refined Indian Detections ({len(refined8)}):")
    for r in refined8:
        x1, y1, x2, y2 = r.box
        print(f"  -> {r.indian_vehicle_type:<16} (conf={r.confidence:.2f}) at [{x1}, {y1}, {x2}, {y2}]")

        color = (0, 255, 0)
        if "SCOOTER" in r.indian_vehicle_type or "MOTORCYCLE" in r.indian_vehicle_type:
            color = (0, 165, 255)
        elif "AUTO-RICKSHAW" in r.indian_vehicle_type:
            color = (0, 255, 255)
        elif "BUS" in r.indian_vehicle_type:
            color = (255, 191, 0)
        elif "TEMPO" in r.indian_vehicle_type:
            color = (255, 100, 100)
        elif "PEDESTRIAN" in r.indian_vehicle_type:
            color = (200, 200, 200)

        cv2.rectangle(annotated8, (x1, y1), (x2, y2), color, 2)
        label_text = f"{r.indian_vehicle_type} {r.confidence:.2f}"
        (tw, th), _ = cv2.getTextSize(label_text, cv2.FONT_HERSHEY_SIMPLEX, 0.45, 1)
        cv2.rectangle(annotated8, (x1, y1 - th - 6), (x1 + tw + 6, y1), color, -1)
        cv2.putText(annotated8, label_text, (x1 + 3, y1 - 4), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (0, 0, 0), 1)

    path8 = out_dir / "real_yolo_CAM_08_alkapuri.jpg"
    cv2.imwrite(str(path8), annotated8, [cv2.IMWRITE_JPEG_QUALITY, 95])
    print(f"  [OK] Saved -> {path8.name}")


if __name__ == "__main__":
    detect_and_draw_refined_boxes()
