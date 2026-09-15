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
det = YOLO(str(ROOT / "models" / "plate_detector" / "plate_v4_small.pt"))

crops = sorted(list((ROOT / "scratch" / "motorcycle_crops").glob("*.jpg")))
print(f"Testing geometric 2-wheeler expansion (AR=1.4) on {len(crops)} crops...")

confirmed = 0

for cpath in crops:
    img = cv2.imread(str(cpath))
    if img is None:
        continue
    h, w = img.shape[:2]
    offset_y = int(h * 0.30)
    search_crop = img[offset_y:, :]
    
    res = det.predict(search_crop, conf=0.12, verbose=False, device=engine.device)
    if not (res and res[0].boxes is not None and len(res[0].boxes) > 0):
        continue
    b = max(res[0].boxes, key=lambda x: float(x.conf[0]))
    xyxy = b.xyxy[0].cpu().numpy()
    det_c = float(b.conf[0])
    px1 = max(0, int(round(xyxy[0])))
    py1 = max(0, int(round(xyxy[1])) + offset_y)
    px2 = min(w, int(round(xyxy[2])))
    py2 = min(h, int(round(xyxy[3])) + offset_y)
    pw, ph = px2 - px1, py2 - py1
    
    # Check if this box is thin (Line 1 only, AR >= 2.0)
    if (pw / max(1, ph)) >= 2.0:
        # Full 2-wheeler plate has AR ~ 1.4 -> full height is pw / 1.4
        target_ph = int(pw / 1.40)
        # Expand downwards
        py2 = min(h, py1 + target_ph)
        ph = py2 - py1
        
    pad_w = int(0.08 * pw)
    pad_h = int(0.08 * ph)
    cx1 = max(0, px1 - pad_w)
    cy1 = max(0, py1 - pad_h)
    cx2 = min(w, px2 + pad_w)
    cy2 = min(h, py2 + pad_h)
    raw_bgr = img[cy1:cy2, cx1:cx2].copy()
    
    prep = cv2.resize(
        cv2.cvtColor(raw_bgr, cv2.COLOR_BGR2GRAY) if raw_bgr.ndim == 3 else raw_bgr,
        (engine._rec_hw[1], engine._rec_hw[0]),
        interpolation=cv2.INTER_AREA
    )
    plate_str, conf, state = engine.recognize_plate(prep, raw_bgr_crop=raw_bgr)
    if plate_str:
        confirmed += 1
        print(f"  [SUCCESS] {cpath.name:30s} -> {plate_str:12s} (conf={conf:.2f}, state={state}) | {raw_bgr.shape[1]}x{raw_bgr.shape[0]}")
    else:
        # Try unstacking with split at 45%
        h_r, w_r = raw_bgr.shape[:2]
        sy = int(h_r * 0.45)
        top_h, bot_h = raw_bgr[:sy, :], raw_bgr[sy:, :]
        if top_h.size > 0 and bot_h.size > 0:
            th_r = cv2.resize(top_h, (64, 32), interpolation=cv2.INTER_CUBIC)
            bh_r = cv2.resize(bot_h, (64, 32), interpolation=cv2.INTER_CUBIC)
            stitched = np.hstack([th_r, bh_r])
            p_s = cv2.resize(cv2.cvtColor(stitched, cv2.COLOR_BGR2GRAY), (engine._rec_hw[1], engine._rec_hw[0]))
            t_s, c_s = engine._crnn_infer(p_s)
            from backend.scripts.indian_plate_grammar import decode_plate
            dec = decode_plate(t_s, local_state="GJ", apply_prior=False)
            if dec.get("plate"):
                confirmed += 1
                print(f"  [STITCH-SUCCESS] {cpath.name:30s} -> {dec['plate']:12s} (conf={c_s:.2f}) from raw '{t_s}'")
            else:
                print(f"  [FAIL] {cpath.name:30s} -> raw='{t_s}' ({raw_bgr.shape[1]}x{raw_bgr.shape[0]})")

print(f"\nTotal Confirmed: {confirmed}")
