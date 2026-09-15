import sys
from pathlib import Path
import cv2
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from ultralytics import YOLO

m4 = YOLO(str(ROOT / "models" / "plate_detector" / "plate_v4_small.pt"))
m1 = YOLO(str(ROOT / "runs" / "detect" / "runs" / "plate" / "plate_v1" / "weights" / "best.pt"))

crops = sorted(list((ROOT / "scratch" / "motorcycle_crops").glob("*.jpg")))

print(f"{'Crop Name':30s} | {'YOLO Boxes (m4 @ conf=0.10)':35s} | {'YOLO Boxes (m1 @ conf=0.10)':35s}")
print("-" * 110)

for cpath in crops:
    img = cv2.imread(str(cpath))
    if img is None:
        continue
    h, w = img.shape[:2]
    offset_y = int(h * 0.35)
    search_crop = img[offset_y:, :]
    
    r4 = m4.predict(search_crop, conf=0.10, verbose=False, device="cuda")
    r1 = m1.predict(search_crop, conf=0.10, verbose=False, device="cuda")
    
    boxes4 = []
    if r4 and r4[0].boxes is not None:
        for b in r4[0].boxes:
            conf = float(b.conf[0])
            xyxy = [round(float(x), 1) for x in b.xyxy[0]]
            pw = xyxy[2] - xyxy[0]
            ph = xyxy[3] - xyxy[1]
            ar = pw / max(1, ph)
            boxes4.append(f"c={conf:.2f},{pw:.0f}x{ph:.0f},ar={ar:.1f}")
            
    boxes1 = []
    if r1 and r1[0].boxes is not None:
        for b in r1[0].boxes:
            conf = float(b.conf[0])
            xyxy = [round(float(x), 1) for x in b.xyxy[0]]
            pw = xyxy[2] - xyxy[0]
            ph = xyxy[3] - xyxy[1]
            ar = pw / max(1, ph)
            boxes1.append(f"c={conf:.2f},{pw:.0f}x{ph:.0f},ar={ar:.1f}")
            
    b4_str = "; ".join(boxes4) if boxes4 else "None"
    b1_str = "; ".join(boxes1) if boxes1 else "None"
    print(f"{cpath.name:30s} | {b4_str:35s} | {b1_str:35s}")
