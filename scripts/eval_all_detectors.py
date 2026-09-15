import sys
from pathlib import Path
import cv2

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from ultralytics import YOLO

candidate_models = [
    ROOT / "models" / "plate_detector" / "plate_v4_small.pt",
    ROOT / "runs" / "detect" / "output" / "detector_train" / "plate_v4" / "weights" / "best.pt",
    ROOT / "runs" / "detect" / "output" / "detector_train" / "plate_v4_aligned" / "weights" / "best.pt",
    ROOT / "runs" / "detect" / "runs" / "plate" / "plate_v3_ft" / "weights" / "best.pt",
    ROOT / "runs" / "detect" / "runs" / "plate" / "plate_v2" / "weights" / "best.pt",
    ROOT / "runs" / "detect" / "runs" / "plate" / "plate_v1" / "weights" / "best.pt",
]

crops = sorted(list((ROOT / "scratch" / "motorcycle_crops").glob("*.jpg")))
print(f"Total motorcycle crops: {len(crops)}")

for m_path in candidate_models:
    if not m_path.exists():
        continue
    try:
        model = YOLO(str(m_path))
        hits_15 = 0
        hits_25 = 0
        hits_10 = 0
        for cpath in crops:
            img = cv2.imread(str(cpath))
            if img is None:
                continue
            h, w = img.shape[:2]
            search_img = img[int(h * 0.30):, :]
            
            # Predict
            res = model.predict(search_img, conf=0.10, verbose=False, device="cuda")
            if res and res[0].boxes is not None and len(res[0].boxes) > 0:
                confs = [float(b.conf[0]) for b in res[0].boxes]
                hits_10 += 1
                if any(c >= 0.15 for c in confs):
                    hits_15 += 1
                if any(c >= 0.25 for c in confs):
                    hits_25 += 1
        print(f"Model: {m_path.name:25s} (from {m_path.parent.parent.name}) -> hits@0.10: {hits_10:2d}/{len(crops)} | hits@0.15: {hits_15:2d}/{len(crops)} | hits@0.25: {hits_25:2d}/{len(crops)}")
    except Exception as exc:
        print(f"Model: {m_path.name} failed with {exc}")
