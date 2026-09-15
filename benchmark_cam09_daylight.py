import os
import cv2
import time
import numpy as np

# Set environment before imports
os.environ["SENTINEL_PLATE_DET_CONF"] = "0.15"
os.environ["SENTINEL_MIN_PLATE_ASPECT_RATIO"] = "1.1"
os.environ["SENTINEL_SINGLE_FRAME_FLOOR_PX"] = "16"
os.environ["SENTINEL_FUSION_FLOOR_PX"] = "14"

from backend.services.anpr_engine import ANPREngine
from ultralytics import YOLO

print("Loading models...")
yolo_veh = YOLO("models/yolov8s.pt")
anpr = ANPREngine()

clip_path = "data/clips/CAM_09/CAM_09_0830.mp4"
cap = cv2.VideoCapture(clip_path)

total_vehicles_seen = set()
vehicle_tracks = {} # track_id -> dict

print(f"Testing on {clip_path}...")
frame_idx = 0

while frame_idx < 300:
    ret, frame = cap.read()
    if not ret:
        break
    frame_idx += 1
    if frame_idx % 2 != 0:
        continue # step at ~12.5 fps

    results = yolo_veh.track(frame, persist=True, classes=[2, 3, 5, 7], verbose=False)
    boxes = results[0].boxes
    if boxes is None or boxes.id is None:
        continue

    for box in boxes:
        tid = int(box.id[0])
        cls = int(box.cls[0])
        cls_name = yolo_veh.names[cls]
        xyxy = box.xyxy[0].cpu().numpy().astype(int)
        x1, y1, x2, y2 = xyxy
        vw = x2 - x1
        vh = y2 - y1
        if vw < 30 or vh < 30:
            continue

        total_vehicles_seen.add(tid)
        bbox = [int(x1), int(y1), int(x2), int(y2)]

        res = anpr.process_vehicle_track(frame, bbox, cls, tid)
        if res and res.get("detected"):
            plate = res.get("plate", "")
            conf = res.get("confidence", 0)
            is_locked = res.get("is_locked", False)
            if tid not in vehicle_tracks:
                vehicle_tracks[tid] = {"plate": plate, "conf": conf, "cls": cls_name, "locked": is_locked}
                print(f"  [FRAME {frame_idx:3d}] Veh #{tid} ({cls_name}): Plate='{plate}', Conf={conf:.1%}, Locked={is_locked}")
            else:
                if conf > vehicle_tracks[tid]["conf"]:
                    vehicle_tracks[tid]["plate"] = plate
                    vehicle_tracks[tid]["conf"] = conf
                    vehicle_tracks[tid]["locked"] = is_locked

cap.release()

readable_vehicles = len(vehicle_tracks)
total_seen = len(total_vehicles_seen)
high_conf = sum(1 for v in vehicle_tracks.values() if v["conf"] >= 0.80)

print("\n" + "="*55)
print(f"CAM_09 BENCHMARK RESULTS (CAM_09_0830.mp4):")
print(f"Total Unique Vehicles Tracked: {total_seen}")
print(f"Vehicles with Plate Detected:  {readable_vehicles}")
print(f"Plates with >= 80% Confidence: {high_conf}")
if readable_vehicles > 0:
    print(f"Recognition Accuracy (>=80% conf): {high_conf / readable_vehicles * 100:.1f}%")
print("="*55)
