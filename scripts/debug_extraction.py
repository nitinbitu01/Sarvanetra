import sys
from pathlib import Path
import cv2

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from ultralytics import YOLO
from backend.services.anpr_engine import get_anpr_engine

engine = get_anpr_engine()
engine.plate_detector = YOLO(str(ROOT / "models" / "plate_detector" / "plate_v4_small.pt"))

fname = "CAM_06_f150_moto_127x201.jpg"
p = ROOT / "scratch" / "motorcycle_crops" / fname
img = cv2.imread(str(p))
h, w = img.shape[:2]

print(f"Testing {fname} ({w}x{h}):")

# Check detect_plate_bbox directly
offset_y = int(h * 0.35)
search_crop = img[offset_y:, :]
box = engine.detect_plate_bbox(search_crop, cls_id=3)
print(f"  detect_plate_bbox result: {box}")

# Check extract_plate_candidate
prep, raw_bgr, q = engine.extract_plate_candidate(img, [0, 0, w, h], cls_id=3)
print(f"  extract_plate_candidate: raw_bgr is None? {raw_bgr is None}, quality={q}")
if raw_bgr is not None:
    print(f"  raw_bgr shape: {raw_bgr.shape}")
