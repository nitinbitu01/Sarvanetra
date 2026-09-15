import sys
from pathlib import Path
import cv2
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from ultralytics import YOLO

p_v3 = ROOT / "runs" / "detect" / "runs" / "plate" / "plate_v3_ft" / "weights" / "best.pt"
p_v4 = ROOT / "models" / "plate_detector" / "plate_v4_small.pt"

print(f"Loading v3: {p_v3}")
m_v3 = YOLO(str(p_v3))
print(f"Loading v4: {p_v4}")
m_v4 = YOLO(str(p_v4))

crops = sorted(list((ROOT / "scratch" / "motorcycle_crops").glob("*.jpg")))
print(f"Total motorcycle crops: {len(crops)}")

for conf in [0.15, 0.25]:
    print(f"\n================ CONF = {conf} ================")
    v3_hits = 0
    v4_hits = 0
    for cpath in crops:
        img = cv2.imread(str(cpath))
        if img is None:
            continue
        h, w = img.shape[:2]
        
        # Lower portion where plate usually sits on motorcycle
        search_img = img[int(h * 0.30):, :]
        
        r_v3 = m_v3.predict(search_img, conf=conf, verbose=False, device="cuda")
        r_v4 = m_v4.predict(search_img, conf=conf, verbose=False, device="cuda")
        
        b3 = len(r_v3[0].boxes) if r_v3 and r_v3[0].boxes is not None else 0
        b4 = len(r_v4[0].boxes) if r_v4 and r_v4[0].boxes is not None else 0
        if b3 > 0:
            v3_hits += 1
        if b4 > 0:
            v4_hits += 1
        
        # print details for first few
        if crops.index(cpath) < 5:
            info3 = [(float(b.conf[0]), [round(float(x),1) for x in b.xyxy[0]]) for b in r_v3[0].boxes] if b3 else []
            info4 = [(float(b.conf[0]), [round(float(x),1) for x in b.xyxy[0]]) for b in r_v4[0].boxes] if b4 else []
            print(f"  {cpath.name} ({w}x{h}): v3_boxes={info3} | v4_boxes={info4}")

    print(f"Summary @ conf={conf}: v3 detections={v3_hits}/{len(crops)}, v4 detections={v4_hits}/{len(crops)}")
