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
det_small = YOLO(str(ROOT / "models" / "plate_detector" / "plate_v4_small.pt"))
engine.plate_detector_small = det_small
engine.plate_detector = det_small

# Load all 25 crops
crops = sorted(list((ROOT / "scratch" / "motorcycle_crops").glob("*.jpg")))
print(f"=== TESTING VERIFIED PIPELINE ON {len(crops)} MOTORCYCLE CROPS ===")

reads = []

for cpath in crops:
    img = cv2.imread(str(cpath))
    if img is None:
        continue
    h, w = img.shape[:2]
    bbox = [0, 0, w, h]
    
    # Simulate process_vehicle_track
    enhanced, _ = engine._enhance_frame(img)
    
    # 1. extract
    off_y = int(h * 0.30)
    search_crop = enhanced[off_y:, :]
    
    box = engine.detect_plate_bbox(search_crop, cls_id=3)
    if box is None:
        continue
        
    px1 = max(0, int(round(box[0])))
    py1 = max(0, int(round(box[1])) + off_y)
    px2 = min(w, int(round(box[2])))
    py2 = min(h, int(round(box[3])) + off_y)
    pw, ph = px2 - px1, py2 - py1
    
    if (pw / max(1, ph)) >= 2.0 and ph <= 18:
        py2 = min(h, py2 + int(ph * 1.5))
        ph = py2 - py1
        
    pad_w = int(0.08 * pw)
    pad_h = int(0.10 * ph)
    cx1 = max(0, px1 - pad_w)
    cy1 = max(0, py1 - pad_h)
    cx2 = min(w, px2 + pad_w)
    cy2 = min(h, py2 + pad_h)
    raw_bgr = img[cy1:cy2, cx1:cx2].copy()
    
    native_w = raw_bgr.shape[1]
    
    # Floor check: 12px for motorcycles
    if native_w < 12:
        continue
        
    prep = cv2.resize(
        cv2.cvtColor(raw_bgr, cv2.COLOR_BGR2GRAY),
        (engine._rec_hw[1], engine._rec_hw[0]),
        interpolation=cv2.INTER_AREA
    )
    
    plate_str, conf, state = engine.recognize_plate(prep, raw_bgr_crop=raw_bgr)
    if plate_str:
        reads.append((cpath.name, plate_str, conf, state, f"{raw_bgr.shape[1]}x{raw_bgr.shape[0]}"))

print(f"\nRESULTS: {len(reads)} / {len(crops)} crops successfully decoded to Indian plates:")
for r in reads:
    print(f"  {r[0]:30s} -> Plate: {r[1]:12s} | Conf: {r[2]:.2f} | State: {r[3]} | Size: {r[4]}")
