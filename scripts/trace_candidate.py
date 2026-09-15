import sys
from pathlib import Path
import cv2
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from ultralytics import YOLO
from backend.services.anpr_engine import (
    get_anpr_engine, MIN_VEHICLE_BBOX_PX, PLATE_DET_CONF, PLATE_DET_IMGSZ,
    MIN_PLATE_WIDTH_PX, MIN_PLATE_HEIGHT_PX, MIN_PLATE_ASPECT_RATIO,
    MAX_PLATE_ASPECT_RATIO, FUSION_FLOOR_PX, MIN_STRIP_QUALITY
)

engine = get_anpr_engine()
detector = YOLO(str(ROOT / "models" / "plate_detector" / "plate_v4_small.pt"))

fname = "CAM_06_f150_moto_127x201.jpg"
img = cv2.imread(str(ROOT / "scratch" / "motorcycle_crops" / fname))
h_frame, w_frame = img.shape[:2]
bbox = [0, 0, w_frame, h_frame]
cls_id = 3

print("Tracing extract_plate_candidate...")

x1, y1, x2, y2  = [int(v) for v in bbox]
bw = x2 - x1
bh = y2 - y1
print(f"bw={bw}, bh={bh}")

vehicle_crop = img[y1:y2, x1:x2]
raw_vehicle_crop = vehicle_crop
vcrop_h = vehicle_crop.shape[0]

offset_y = int(vcrop_h * 0.35)
print(f"offset_y={offset_y}")

plate_search_crop = raw_vehicle_crop[offset_y:, :]
raw_bgr_crop = None

# Step 1: detect_plate_bbox
h_s, w_s = plate_search_crop.shape[:2]
print(f"plate_search_crop shape: {w_s}x{h_s}")

det_conf = min(PLATE_DET_CONF, 0.18) if cls_id == 3 else PLATE_DET_CONF
print(f"det_conf={det_conf}")

res = detector.predict(plate_search_crop, imgsz=PLATE_DET_IMGSZ, conf=det_conf, verbose=False, device=engine.device)
print(f"predict found {len(res[0].boxes)} boxes")

plate_box = engine.detect_plate_bbox(plate_search_crop, cls_id=cls_id)
print(f"engine.detect_plate_bbox returned: {plate_box}")

if plate_box is not None:
    px1 = max(0,  int(round(plate_box[0])))
    py1 = max(0,  int(round(plate_box[1])) + offset_y)
    px2 = min(bw, int(round(plate_box[2])))
    py2 = min(bh, int(round(plate_box[3])) + offset_y)
    pw, ph = px2 - px1, py2 - py1
    print(f"px1={px1}, py1={py1}, px2={px2}, py2={py2}, pw={pw}, ph={ph}")

    if cls_id in (3, 5, 7) and (pw / max(1, ph)) >= 2.4 and ph <= 18:
        py2 = min(bh, py2 + int(ph * 1.5))
        ph = py2 - py1
        print(f"After expansion: py2={py2}, ph={ph}")

    min_w = 14 if cls_id == 3 else MIN_PLATE_WIDTH_PX
    min_h = 4 if cls_id == 3 else MIN_PLATE_HEIGHT_PX
    min_ar = 1.05 if cls_id == 3 else MIN_PLATE_ASPECT_RATIO
    print(f"min_w={min_w}, min_h={min_h}, min_ar={min_ar}")

    if pw >= min_w and ph >= min_h:
        aspect_ratio = pw / ph if ph > 0 else 0
        print(f"aspect_ratio={aspect_ratio}")
        if min_ar <= aspect_ratio <= MAX_PLATE_ASPECT_RATIO:
            pad_w = int(0.08 * pw)
            pad_h = int(0.10 * ph)
            cx1 = max(0, px1 - pad_w)
            cy1 = max(0, py1 - pad_h)
            cx2 = min(bw, px2 + pad_w)
            cy2 = min(bh, py2 + pad_h)
            raw_bgr_crop = raw_vehicle_crop[cy1:cy2, cx1:cx2].copy()
            print(f"raw_bgr_crop created! shape={raw_bgr_crop.shape}")
        else:
            print(f"REJECTED by aspect ratio: {aspect_ratio} not in [{min_ar}, {MAX_PLATE_ASPECT_RATIO}]")
    else:
        print(f"REJECTED by size: pw={pw} >= {min_w}? ph={ph} >= {min_h}?")
else:
    print("plate_box is None!")

if raw_bgr_crop is None:
    print("EXIT: raw_bgr_crop is None")
else:
    native_w = raw_bgr_crop.shape[1]
    floor_px = 14 if cls_id == 3 else FUSION_FLOOR_PX
    print(f"native_w={native_w}, floor_px={floor_px}")
    quality = engine._frame_selector.score(raw_bgr_crop)
    min_q = 6.0 if cls_id == 3 else MIN_STRIP_QUALITY
    print(f"quality={quality}, min_q={min_q}")
