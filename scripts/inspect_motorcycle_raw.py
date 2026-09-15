import cv2
import sys
from pathlib import Path
from ultralytics import YOLO

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

def inspect_motorcycle_crops():
    det = YOLO("models/plate_detector/plate_v4_small.pt")
    yolo = YOLO("yolov8n.pt")
    
    for cam in ["CAM_06", "CAM_21", "CAM_22"]:
        clips = sorted(list(Path(f"data/clips/{cam}").glob("*.mp4")))
        if not clips:
            continue
        cap = cv2.VideoCapture(str(clips[0]))
        print(f"\n==================== Probing {cam} ({clips[0].name}) ====================", flush=True)
        
        motos_checked = 0
        raw_dets_found = 0
        
        for f_idx in range(0, 300, 10):
            cap.set(cv2.CAP_PROP_POS_FRAMES, f_idx)
            ret, frame = cap.read()
            if not ret:
                break
            res = yolo(frame, imgsz=640, verbose=False)[0]
            for box in res.boxes:
                if int(box.cls[0].item()) == 3:  # motorcycle
                    motos_checked += 1
                    xyxy = [int(v) for v in box.xyxy[0].tolist()]
                    w, h = xyxy[2] - xyxy[0], xyxy[3] - xyxy[1]
                    vcrop = frame[xyxy[1]:xyxy[3], xyxy[0]:xyxy[2]]
                    if vcrop.size == 0 or w < 20 or h < 20:
                        continue
                    
                    # Predict without any size/AR filter, conf=0.05
                    pred = det.predict(vcrop, imgsz=320, conf=0.05, verbose=False, device="cuda:0")
                    if pred and pred[0].boxes is not None and len(pred[0].boxes) > 0:
                        for b in pred[0].boxes:
                            raw_dets_found += 1
                            c = float(b.conf[0].item())
                            bx = b.xyxy[0].tolist()
                            bw = bx[2] - bx[0]
                            bh = bx[3] - bx[1]
                            ar = bw / max(1e-3, bh)
                            print(f"[{cam} f{f_idx}] Moto {w}x{h} -> RAW DET: conf={c:.2f}, box={bw:.1f}x{bh:.1f} (AR={ar:.2f}), rel_y={(bx[1]/h):.2f}", flush=True)
        cap.release()
        print(f"SUMMARY {cam}: motos={motos_checked}, raw_dets={raw_dets_found}", flush=True)

if __name__ == "__main__":
    inspect_motorcycle_crops()
