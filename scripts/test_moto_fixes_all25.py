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
# 1. Use plate_v4_small for small / motorcycle plates
small_model = YOLO(str(ROOT / "models" / "plate_detector" / "plate_v4_small.pt"))
engine.plate_detector_small = small_model

crops = sorted(list((ROOT / "scratch" / "motorcycle_crops").glob("*.jpg")))
print(f"Testing {len(crops)} real motorcycle crops from fleet cameras...")

success_count = 0
box_count = 0

for cpath in crops:
    img = cv2.imread(str(cpath))
    if img is None:
        continue
    h, w = img.shape[:2]
    bbox = [0, 0, w, h]
    
    # Run extraction with motorcycle adjustments
    offset_y = int(h * 0.30)
    search_crop = img[offset_y:, :]
    
    # Detect using small detector with conf=0.12
    res = small_model.predict(search_crop, conf=0.12, verbose=False, device=engine.device)
    if res and res[0].boxes is not None and len(res[0].boxes) > 0:
        box_count += 1
        # pick best box
        b = max(res[0].boxes, key=lambda x: float(x.conf[0]))
        xyxy = b.xyxy[0].cpu().numpy()
        det_c = float(b.conf[0])
        px1 = max(0, int(round(xyxy[0])))
        py1 = max(0, int(round(xyxy[1])) + offset_y)
        px2 = min(w, int(round(xyxy[2])))
        py2 = min(h, int(round(xyxy[3])) + offset_y)
        pw, ph = px2 - px1, py2 - py1
        
        # Two-line expansion
        if (pw / max(1, ph)) >= 2.2 and ph <= 18:
            py2 = min(h, py2 + int(ph * 1.5))
            ph = py2 - py1
            
        pad_w = int(0.08 * pw)
        pad_h = int(0.10 * ph)
        cx1 = max(0, px1 - pad_w)
        cy1 = max(0, py1 - pad_h)
        cx2 = min(w, px2 + pad_w)
        cy2 = min(h, py2 + pad_h)
        raw_bgr = img[cy1:cy2, cx1:cx2].copy()
        
        # Recognize plate
        prep = cv2.resize(
            cv2.cvtColor(raw_bgr, cv2.COLOR_BGR2GRAY) if raw_bgr.ndim == 3 else raw_bgr,
            (engine._rec_hw[1], engine._rec_hw[0]),
            interpolation=cv2.INTER_AREA
        )
        plate_str, conf, state = engine.recognize_plate(prep, raw_bgr_crop=raw_bgr)
        if plate_str:
            success_count += 1
            print(f"  [SUCCESS] {cpath.name:30s} -> Plate: {plate_str:12s} | Conf: {conf:.2f} | State: {state} | Crop: {raw_bgr.shape[1]}x{raw_bgr.shape[0]} (det_c={det_c:.2f})")
        else:
            print(f"  [NO-OCR]  {cpath.name:30s} -> Crop: {raw_bgr.shape[1]}x{raw_bgr.shape[0]} (det_c={det_c:.2f})")
    else:
        print(f"  [NO-BOX]  {cpath.name:30s} ({w}x{h})")

print(f"\nSummary: Detections: {box_count}/{len(crops)} | Confirmed Reads: {success_count}/{len(crops)}")
