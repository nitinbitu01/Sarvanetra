import cv2
import json
from pathlib import Path
from ultralytics import YOLO
import easyocr
import sys

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from backend.services.anpr_engine import get_anpr_engine

def analyze_crops():
    det = YOLO("models/plate_detector/plate_v4_small.pt")
    anpr = get_anpr_engine()
    reader = easyocr.Reader(["en"], gpu=True, verbose=False)

    crops = sorted(list(Path("scratch/motorcycle_crops").glob("*.jpg")))
    crops = [c for c in crops if not c.name.startswith("annotated_")]
    print(f"Analyzing {len(crops)} motorcycle crops...", flush=True)

    found = 0
    reads = 0
    for p in crops:
        img = cv2.imread(str(p))
        h, w = img.shape[:2]
        
        # Test 1: Full crop prediction
        res = det.predict(img, imgsz=320, conf=0.15, verbose=False, device="cuda:0")[0]
        boxes = res.boxes
        if boxes is not None and len(boxes) > 0:
            found += 1
            for b in boxes:
                conf = float(b.conf[0].item())
                xyxy = [int(v) for v in b.xyxy[0].tolist()]
                bw, bh = xyxy[2] - xyxy[0], xyxy[3] - xyxy[1]
                ar = bw / max(1, bh)
                
                # Extract plate candidate
                pad_w = int(0.1 * bw)
                pad_h = int(0.15 * bh)
                px1 = max(0, xyxy[0] - pad_w)
                py1 = max(0, xyxy[1] - pad_h)
                px2 = min(w, xyxy[2] + pad_w)
                py2 = min(h, xyxy[3] + pad_h)
                plate_crop = img[py1:py2, px1:px2]
                
                # Test EasyOCR with vertical sorting
                easy_res = reader.readtext(plate_crop, detail=1, allowlist="ABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789")
                sorted_boxes = sorted(easy_res, key=lambda r: (r[0][0][1] + r[0][2][1]) / 2.0)
                easy_text = "".join(r[1] for r in sorted_boxes).upper().replace(" ", "")
                easy_conf = max([r[2] for r in sorted_boxes], default=0.0)
                
                # Test CRNN directly
                crnn_prep = cv2.resize(cv2.cvtColor(plate_crop, cv2.COLOR_BGR2GRAY), (128, 32))
                crnn_text, crnn_conf = anpr._crnn_infer(crnn_prep)
                
                # Test Unstacked CRNN if 2-line (AR <= 2.2)
                unstacked_text = ""
                unstacked_conf = 0.0
                if ar <= 2.2 and bh >= 10:
                    split_y = int(plate_crop.shape[0] * 0.48)
                    t_half = plate_crop[:split_y, :]
                    b_half = plate_crop[split_y:, :]
                    th_r = cv2.resize(t_half, (int(t_half.shape[1] * (32 / max(1, t_half.shape[0]))), 32))
                    bh_r = cv2.resize(b_half, (int(b_half.shape[1] * (32 / max(1, b_half.shape[0]))), 32))
                    stitched = np.hstack([th_r, bh_r])
                    s_prep = cv2.resize(cv2.cvtColor(stitched, cv2.COLOR_BGR2GRAY), (128, 32))
                    unstacked_text, unstacked_conf = anpr._crnn_infer(s_prep)
                
                if easy_text or crnn_text or unstacked_text:
                    reads += 1
                
                print(f"{p.name} [{w}x{h}]: Box {bw}x{bh} (conf={conf:.2f}, AR={ar:.2f}, rel_y={(xyxy[1]/h):.2f})", flush=True)
                print(f"   -> EasyOCR: '{easy_text}' (conf={easy_conf:.2f})", flush=True)
                print(f"   -> CRNN:    '{crnn_text}' (conf={crnn_conf:.2f})", flush=True)
                if unstacked_text:
                    print(f"   -> Unstack: '{unstacked_text}' (conf={unstacked_conf:.2f})", flush=True)

    print(f"\nResults: {found}/{len(crops)} plates detected, {reads} OCR reads", flush=True)

if __name__ == "__main__":
    import numpy as np
    analyze_crops()
