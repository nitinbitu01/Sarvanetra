import sys
from pathlib import Path
import cv2
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from backend.services.anpr_engine import get_anpr_engine

engine = get_anpr_engine()

test_crops = [
    "CAM_06_f140_moto_118x196.jpg",
    "CAM_06_f150_moto_127x201.jpg",
    "CAM_06_f150_moto_132x149.jpg",
    "CAM_06_f160_moto_141x225.jpg",
    "CAM_06_f180_moto_175x287.jpg",
    "CAM_06_f100_moto_127x202.jpg",
]

# We need the plate detector to extract the exact plate crop
from ultralytics import YOLO
det = YOLO(str(ROOT / "models" / "plate_detector" / "plate_v4_small.pt"))

print("Inspecting OCR on motorcycle crops...")

for fname in test_crops:
    img = cv2.imread(str(ROOT / "scratch" / "motorcycle_crops" / fname))
    h, w = img.shape[:2]
    offset_y = int(h * 0.30)
    search_crop = img[offset_y:, :]
    res = det.predict(search_crop, conf=0.12, verbose=False, device="cuda")
    if not (res and res[0].boxes is not None and len(res[0].boxes) > 0):
        continue
    b = max(res[0].boxes, key=lambda x: float(x.conf[0]))
    xyxy = b.xyxy[0].cpu().numpy()
    px1, py1 = max(0, int(round(xyxy[0]))), max(0, int(round(xyxy[1])) + offset_y)
    px2, py2 = min(w, int(round(xyxy[2]))), min(h, int(round(xyxy[3])) + offset_y)
    pw, ph = px2 - px1, py2 - py1
    
    # Raw box (Line 1 only)
    raw_l1 = img[py1:py2, px1:px2]
    # Two-line expanded box (Line 1 + Line 2)
    py2_exp = min(h, py2 + int(ph * 1.5))
    raw_both = img[py1:py2_exp, px1:px2]
    
    print(f"\n--- {fname} (Det Box: {pw}x{ph}, Exp: {pw}x{py2_exp-py1}) ---")
    
    # 1. Test CRNN on Line 1 only (resized to 128x32)
    p_l1 = cv2.resize(cv2.cvtColor(raw_l1, cv2.COLOR_BGR2GRAY), (128, 32))
    t_l1, c_l1 = engine._crnn_infer(p_l1)
    print(f"  Line 1 CRNN: text='{t_l1}', conf={c_l1:.2f}")
    
    # 2. Test CRNN on Both Lines direct
    p_both = cv2.resize(cv2.cvtColor(raw_both, cv2.COLOR_BGR2GRAY), (128, 32))
    t_both, c_both = engine._crnn_infer(p_both)
    print(f"  Both Lines Direct CRNN: text='{t_both}', conf={c_both:.2f}")
    
    # 3. Test Unstacking (Top half + Bot half side-by-side)
    h_b, w_b = raw_both.shape[:2]
    sy = int(h_b * 0.48)
    top_h, bot_h = raw_both[:sy, :], raw_both[sy:, :]
    if top_h.size > 0 and bot_h.size > 0:
        th_r = cv2.resize(top_h, (64, 32), interpolation=cv2.INTER_CUBIC)
        bh_r = cv2.resize(bot_h, (64, 32), interpolation=cv2.INTER_CUBIC)
        stitched = np.hstack([th_r, bh_r])
        p_stitch = cv2.resize(cv2.cvtColor(stitched, cv2.COLOR_BGR2GRAY), (128, 32))
        t_st, c_st = engine._crnn_infer(p_stitch)
        print(f"  Unstacked/Stitched CRNN: text='{t_st}', conf={c_st:.2f}")
        
    # 4. Test EasyOCR / Ensemble if available
    try:
        import easyocr
        reader = easyocr.Reader(['en'], gpu=True)
        # Upscale 3x for CRAFT
        up = cv2.resize(raw_both, (w_b * 3, h_b * 3), interpolation=cv2.INTER_CUBIC)
        e_res = reader.readtext(up)
        print(f"  EasyOCR (3x upscaled): {[(item[1], round(item[2], 2)) for item in e_res]}")
    except Exception as exc:
        print(f"  EasyOCR error: {exc}")
