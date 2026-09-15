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

engine.plate_detector = small_model

# Let's test the 3 crops that found boxes
test_files = [
    "CAM_06_f140_moto_175x254.jpg",
    "CAM_06_f140_moto_223x310.jpg",
    "CAM_06_f180_moto_175x287.jpg",
]

for fname in test_files:
    p = ROOT / "scratch" / "motorcycle_crops" / fname
    img = cv2.imread(str(p))
    h, w = img.shape[:2]
    prep, raw_bgr, q = engine.extract_plate_candidate(img, [0, 0, w, h], cls_id=3)
    if raw_bgr is not None:
        print(f"\n{fname}: Plate crop size {raw_bgr.shape[1]}x{raw_bgr.shape[0]}")
        # Test recognition directly bypassing line 1086
        plate_str, conf, state = engine.recognize_plate(prep, raw_bgr_crop=raw_bgr)
        print(f"  Direct recognize_plate -> plate={plate_str}, conf={conf}, state={state}")
        # Also test with ensemble / EasyOCR
        if engine._ensemble is not None:
            raw_text, econf, src = engine._ensemble.recognize(raw_bgr, prep)
            print(f"  Ensemble recognize -> text={raw_text}, conf={econf}, src={src}")
