import cv2
import sys
import numpy as np
from pathlib import Path
from ultralytics import YOLO

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from backend.services.anpr_engine import get_anpr_engine

def unstack_two_line_plate(crop_bgr: np.ndarray) -> np.ndarray:
    """Split a 2-line plate into top and bottom halves, resize, and stack horizontally."""
    h, w = crop_bgr.shape[:2]
    split_y = int(h * 0.48)
    top_half = crop_bgr[:split_y, :]
    bot_half = crop_bgr[split_y:, :]
    target_h = 32
    tw = max(10, int(top_half.shape[1] * (target_h / max(1, top_half.shape[0]))))
    bw = max(10, int(bot_half.shape[1] * (target_h / max(1, bot_half.shape[0]))))
    top_resized = cv2.resize(top_half, (tw, target_h), interpolation=cv2.INTER_CUBIC)
    bot_resized = cv2.resize(bot_half, (bw, target_h), interpolation=cv2.INTER_CUBIC)
    return np.hstack([top_resized, bot_resized])

def evaluate_motorcycles():
    anpr = get_anpr_engine()
    yolo = YOLO("yolov8n.pt")
    
    # Test cameras with rich 2-wheeler traffic
    for cam in ["CAM_06", "CAM_21", "CAM_22", "CAM_18"]:
        clips = sorted(list(Path(f"data/clips/{cam}").glob("*0830*.mp4"))) or sorted(list(Path(f"data/clips/{cam}").glob("*.mp4")))
        if not clips:
            continue
        clip = clips[0]
        cap = cv2.VideoCapture(str(clip))
        print(f"\n==================== Testing Motorcycles on {cam} ({clip.name}) ====================", flush=True)
        
        motos_seen = 0
        plates_found = 0
        two_line_candidates = 0
        
        for f_idx in range(0, 500, 10):
            cap.set(cv2.CAP_PROP_POS_FRAMES, f_idx)
            ret, frame = cap.read()
            if not ret:
                break
            res = yolo(frame, imgsz=640, verbose=False)[0]
            for box in res.boxes:
                cls = int(box.cls[0].item())
                if cls == 3:  # motorcycle
                    motos_seen += 1
                    xyxy = [int(v) for v in box.xyxy[0].tolist()]
                    w, h = xyxy[2] - xyxy[0], xyxy[3] - xyxy[1]
                    vcrop = frame[xyxy[1]:xyxy[3], xyxy[0]:xyxy[2]]
                    if vcrop.size == 0 or w < 20 or h < 20:
                        continue
                    
                    # Motorcycle search window: lower 45% of crop (avoid rider torso/helmet)
                    offset_y = int(h * 0.40)
                    search_crop = vcrop[offset_y:, :]
                    if search_crop.size == 0:
                        continue
                    
                    # Run plate detector with relaxed aspect ratio
                    pbox = anpr.detect_plate_bbox(search_crop)
                    if pbox is not None:
                        plates_found += 1
                        pw = pbox[2] - pbox[0]
                        ph = pbox[3] - pbox[1]
                        ar = pw / max(1, ph)
                        px1 = max(0, int(pbox[0]))
                        py1 = max(0, int(pbox[1]))
                        px2 = min(search_crop.shape[1], int(pbox[2]))
                        py2 = min(search_crop.shape[0], int(pbox[3]))
                        raw_plate = search_crop[py1:py2, px1:px2]
                        
                        is_2line = ar < 1.8
                        if is_2line:
                            two_line_candidates += 1
                        
                        # Test normal OCR
                        p_res = anpr.process_vehicle_track(frame, xyxy, cls, track_id=900000 + motos_seen)
                        plate_read = p_res.get("plate") if p_res else None
                        conf = p_res.get("confidence", 0.0) if p_res else 0.0
                        
                        print(f"[{cam} f{f_idx}] MOTO {w}x{h} -> Plate {pw:.1f}x{ph:.1f} (AR={ar:.2f}, 2-line={is_2line}) -> Read: {plate_read} (conf={conf:.2f})", flush=True)

        cap.release()
        print(f"SUMMARY {cam}: motos={motos_seen}, plates_found={plates_found}, 2-line={two_line_candidates}", flush=True)

if __name__ == "__main__":
    evaluate_motorcycles()
