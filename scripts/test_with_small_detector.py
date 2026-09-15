import sys
from pathlib import Path
import cv2
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from ultralytics import YOLO
from backend.services.anpr_engine import get_anpr_engine

engine = get_anpr_engine()
small_model = YOLO(str(ROOT / "models" / "plate_detector" / "plate_v4_small.pt"))
v1_model = YOLO(str(ROOT / "runs" / "detect" / "runs" / "plate" / "plate_v1" / "weights" / "best.pt"))

# Assign small_model to plate_detector_small
engine.plate_detector_small = small_model
# Also try assigning to plate_detector
engine.plate_detector = small_model

crops = sorted(list((ROOT / "scratch" / "motorcycle_crops").glob("*.jpg")))
print(f"Testing with plate_v4_small as detector on {len(crops)} crops...")

successes = 0
found_boxes = 0

for cpath in crops:
    img = cv2.imread(str(cpath))
    if img is None:
        continue
    h, w = img.shape[:2]
    bbox = [0, 0, w, h]
    
    # 1. Test extraction
    prep, raw_bgr, q = engine.extract_plate_candidate(img, bbox, cls_id=3)
    if raw_bgr is not None:
        found_boxes += 1
        cw, ch = raw_bgr.shape[1], raw_bgr.shape[0]
        # 2. Test recognition
        res = engine.process_vehicle_track(img, bbox, cls_id=3, track_id=hash(cpath.name) % 100000)
        plate_str = res.get("plate") if res else None
        conf = res.get("confidence") if res else 0.0
        raw = res.get("raw_read") if res else None
        print(f"  FOUND BOX: {cpath.name} -> crop {cw}x{ch} (q={q:.1f}) | OCR: {plate_str} (conf={conf}, raw={raw})")
        if plate_str:
            successes += 1
    else:
        print(f"  NO BOX: {cpath.name} ({w}x{h})")

print(f"\nResult: Boxes found: {found_boxes}/{len(crops)} | OCR Successes: {successes}/{len(crops)}")
