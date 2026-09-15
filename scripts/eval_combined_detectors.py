import sys
from pathlib import Path
import cv2

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from ultralytics import YOLO

m1 = YOLO(str(ROOT / "runs" / "detect" / "runs" / "plate" / "plate_v1" / "weights" / "best.pt"))
m4 = YOLO(str(ROOT / "models" / "plate_detector" / "plate_v4_small.pt"))

crops = sorted(list((ROOT / "scratch" / "motorcycle_crops").glob("*.jpg")))
total = len(crops)
either_hits = 0

for cpath in crops:
    img = cv2.imread(str(cpath))
    if img is None:
        continue
    h, w = img.shape[:2]
    search_img = img[int(h * 0.30):, :]
    
    r1 = m1.predict(search_img, conf=0.12, verbose=False, device="cuda")
    r4 = m4.predict(search_img, conf=0.12, verbose=False, device="cuda")
    
    b1 = len(r1[0].boxes) if r1 and r1[0].boxes is not None else 0
    b4 = len(r4[0].boxes) if r4 and r4[0].boxes is not None else 0
    
    if b1 > 0 or b4 > 0:
        either_hits += 1
        b_best = []
        if b1 > 0:
            for b in r1[0].boxes:
                b_best.append(('m1', float(b.conf[0]), [round(float(x),1) for x in b.xyxy[0]]))
        if b4 > 0:
            for b in r4[0].boxes:
                b_best.append(('m4', float(b.conf[0]), [round(float(x),1) for x in b.xyxy[0]]))
        print(f"HIT: {cpath.name} ({w}x{h}): {b_best}")
    else:
        print(f"MISS: {cpath.name} ({w}x{h})")

print(f"\nCombined Recall: {either_hits}/{total} ({either_hits/total*100:.1f}%)")
