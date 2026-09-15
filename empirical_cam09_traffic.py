import cv2
import numpy as np
import os
import time

os.environ["SENTINEL_PLATE_DET_CONF"] = "0.15"
os.environ["SENTINEL_MIN_PLATE_ASPECT_RATIO"] = "1.1"
os.environ["SENTINEL_SINGLE_FRAME_FLOOR_PX"] = "16"
os.environ["SENTINEL_FUSION_FLOOR_PX"] = "14"

from backend.services.anpr_engine import ANPREngine
from ultralytics import YOLO

yolo = YOLO("models/yolov8s.pt")
anpr = ANPREngine()

video_path = "data/clips/CAM_09/CAM_09_0100.mp4"
cap = cv2.VideoCapture(video_path)

# Skip to where traffic is moving (e.g. 200 seconds in, which is where worker_w5 was)
fps = cap.get(cv2.CAP_PROP_FPS) or 25.0
cap.set(cv2.CAP_PROP_POS_FRAMES, int(200 * fps))

print(f"Analyzing CAM_09 from 200s for 600 frames (24 seconds)...")

track_history = {} # tid -> {'cls': str, 'max_area': 0, 'frames': 0, 'plate': None, 'conf': 0.0}

for f_idx in range(600):
    ret, frame = cap.read()
    if not ret:
        break

    res = yolo.track(frame, persist=True, classes=[2, 3, 5, 7], verbose=False)
    boxes = res[0].boxes
    if boxes is None or boxes.id is None:
        continue

    for box in boxes:
        tid = int(box.id[0])
        cls_id = int(box.cls[0])
        cls_name = yolo.names[cls_id]
        xyxy = box.xyxy[0].cpu().numpy().astype(int)
        w = xyxy[2] - xyxy[0]
        h = xyxy[3] - xyxy[1]
        area = w * h

        if tid not in track_history:
            track_history[tid] = {'cls': cls_name, 'max_area': area, 'frames': 1, 'plate': None, 'conf': 0.0}
        else:
            track_history[tid]['frames'] += 1
            track_history[tid]['max_area'] = max(track_history[tid]['max_area'], area)

        # Only evaluate vehicles that are large enough to be in the camera view (w >= 30, h >= 30)
        if w >= 30 and h >= 30:
            bbox = [xyxy[0], xyxy[1], xyxy[2], xyxy[3]]
            p_res = anpr.process_vehicle_track(frame, bbox, cls_id, tid)
            if p_res and p_res.get("detected") and p_res.get("plate"):
                p_text = p_res.get("plate")
                p_conf = float(p_res.get("confidence", 0.0))
                if p_conf > track_history[tid]['conf']:
                    track_history[tid]['plate'] = p_text
                    track_history[tid]['conf'] = p_conf

cap.release()

# Filter vehicles that actually passed through the frame (e.g. visible for >= 10 frames and max_area >= 2500 px)
significant_vehicles = {k: v for k, v in track_history.items() if v['frames'] >= 10 and v['max_area'] >= 2500}
plates_found = {k: v for k, v in significant_vehicles.items() if v['plate'] is not None}
high_conf_plates = {k: v for k, v in plates_found.items() if v['conf'] >= 0.80}

print("\n" + "="*60)
print(f"EMPIRICAL TRAFFIC ANALYSIS ON CAM_09 (600 frames):")
print(f"Total Raw ByteTrack IDs:          {len(track_history)}")
print(f"Actual Significant Vehicles Passing: {len(significant_vehicles)}")
for tid, v in significant_vehicles.items():
    print(f"  Vehicle #{tid:4d} ({v['cls']:10s}, {v['frames']:3d} frames, max_area={v['max_area']:6d}px): Plate={v['plate']} (conf={v['conf']:.1%})")
print("-" * 60)
print(f"Vehicles with Plate Detected:       {len(plates_found)} / {len(significant_vehicles)}")
if significant_vehicles:
    det_rate = len(plates_found) / len(significant_vehicles) * 100
    rec_rate = len(high_conf_plates) / len(significant_vehicles) * 100
    print(f"Plate Detection Rate:               {det_rate:.1f}%")
    print(f"Plate Recognition Rate (>=80% conf):{rec_rate:.1f}%")
print("="*60)
