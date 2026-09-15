"""
Find exact frames and coordinates for CAM_08 and CAM_07 plates.
"""
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

import cv2
from ultralytics import YOLO
from backend.services.anpr_engine import get_anpr_engine

def find_in_clip(clip_path, max_frames=250, stride=2):
    print(f"\n--- Scanning {clip_path} ---")
    cap = cv2.VideoCapture(str(clip_path))
    engine = get_anpr_engine()
    yolo = YOLO("yolov8s.pt")
    
    fr = 0
    results = []
    while fr < max_frames:
        ok, frame = cap.read()
        if not ok:
            break
        fr += 1
        if fr % stride != 0:
            continue
        res = yolo.predict(frame, verbose=False, conf=0.35)
        if not res or res[0].boxes is None:
            continue
        for i in range(len(res[0].boxes)):
            cls_id = int(res[0].boxes.cls[i].item())
            if cls_id not in (2, 3, 5, 7):
                continue
            vbox = [float(x) for x in res[0].boxes.xyxy[i].cpu().numpy()]
            out = engine.process_vehicle_track(frame, vbox, cls_id, track_id=300000 + i)
            if out and out.get("plate"):
                plate = out.get("plate")
                conf = float(out.get("confidence", 0.0))
                print(f"Frame {fr}: plate={plate}, conf={conf:.3f}, cls={cls_id}, bbox={[int(x) for x in vbox]}")
                results.append({
                    "frame_idx": fr,
                    "plate": plate,
                    "conf": conf,
                    "cls_id": cls_id,
                    "vbox": vbox
                })
    cap.release()
    return results

if __name__ == "__main__":
    c8 = find_in_clip(ROOT / "demo" / "clips" / "cam08_demo_loop.mp4", max_frames=200, stride=2)
    c7 = find_in_clip(ROOT / "data" / "clips" / "CAM_07" / "CAM_07_0830.mp4", max_frames=120, stride=2)
