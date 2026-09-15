import sys
from pathlib import Path
import cv2

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from ultralytics import YOLO

small_det = YOLO(str(ROOT / "models" / "plate_detector" / "plate_v4_small.pt"))
crops = sorted(list((ROOT / "scratch" / "motorcycle_crops").glob("*.jpg")))

for frac in [0.15, 0.20, 0.25, 0.30, 0.35]:
    hits = 0
    for cpath in crops:
        img = cv2.imread(str(cpath))
        if img is None:
            continue
        h, w = img.shape[:2]
        search = img[int(h * frac):, :]
        res = small_det.predict(search, conf=0.12, verbose=False, device="cuda")
        if res and res[0].boxes is not None and len(res[0].boxes) > 0:
            hits += 1
    print(f"Offset fraction {frac:.2f} -> {hits}/{len(crops)} detections")
