import cv2
import sys
import os
from pathlib import Path
from ultralytics import YOLO

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from backend.services.anpr_engine import get_anpr_engine

def test_all_clips():
    anpr = get_anpr_engine()
    yolo = YOLO("yolov8n.pt")

    for cam in ["CAM_21", "CAM_22"]:
        cam_dir = Path(f"data/clips/{cam}")
        clips = sorted(list(cam_dir.glob("*.mp4")))
        print(f"\n==================== {cam}: Found {len(clips)} clips ====================", flush=True)
        for clip in clips:
            cap = cv2.VideoCapture(str(clip))
            frames_checked = 0
            vehicles_found = 0
            plate_detections = 0
            ocr_reads = 0
            
            # Sample 50 frames
            for idx in range(0, 750, 15):
                cap.set(cv2.CAP_PROP_POS_FRAMES, idx)
                ret, frame = cap.read()
                if not ret:
                    break
                frames_checked += 1
                res = yolo(frame, imgsz=640, verbose=False)[0]
                for box in res.boxes:
                    cls = int(box.cls[0].item())
                    if cls in [2, 3, 5, 7]:
                        vehicles_found += 1
                        xyxy = [int(v) for v in box.xyxy[0].tolist()]
                        w, h = xyxy[2] - xyxy[0], xyxy[3] - xyxy[1]
                        vcrop = frame[xyxy[1]:xyxy[3], xyxy[0]:xyxy[2]]
                        if vcrop.size == 0 or w < 30 or h < 30:
                            continue
                        
                        pbox = anpr.detect_plate_bbox(vcrop)
                        if pbox is not None:
                            plate_detections += 1
                            p_res = anpr.process_vehicle_track(frame, xyxy, cls, track_id=vehicles_found)
                            if p_res and p_res.get("plate"):
                                ocr_reads += 1
            cap.release()
            print(f"  Clip {clip.name:18s}: frames={frames_checked:2d}, veh={vehicles_found:3d}, plate_dets={plate_detections:2d}, ocr_reads={ocr_reads:2d}", flush=True)

if __name__ == "__main__":
    test_all_clips()

