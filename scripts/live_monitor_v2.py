# sentinel gujarat/scripts/live_monitor_v2.py

import cv2
import threading
import time
import queue
from collections import defaultdict
from datetime import datetime

# Import your existing models
import sys
sys.path.append("sentinel gujarat/backend")
from ultralytics import YOLO
from services.anpr_engine import ANPREngine  # Your existing CRNN
from scripts.night_enhancer_v2 import AdaptiveNightEnhancer
from services.plate_utils_v2 import PlatePreprocessorV2
from services.anpr_track_aggregator_v2 import TemporalPlateVoter
from services.frame_selector import PlateFrameSelector

# ─── CONFIG ─────────────────────────────────────────
NUM_CAMERAS = 30
STREAM_BASE = "https://cctv.corp8.cloud/cam{:02d}/index.m3u8"
PROCESS_FPS = 5          # Process 5 frames/sec per camera
BATCH_SIZE = 8            # GPU batch inference
CONFIDENCE_THRESHOLD = 0.55
WHITELISTED_CAMS = [6, 7, 8, 9, 10, 18, 21, 27]  # Priority cameras

# ─── LOAD MODELS (ONCE, SHARED) ────────────────────
vehicle_model = YOLO("sentinel gujarat/yolov8s.pt")
plate_model = YOLO("runs/detect/runs/plate/plate_v3_ft/weights/best.pt")
enhancer = AdaptiveNightEnhancer()
preprocessor = PlatePreprocessorV2()
selector = PlateFrameSelector()

# ─── SHARED DETECTION LOG ──────────────────────────
detection_log = queue.Queue()


class CameraWorker(threading.Thread):
    """One worker per camera — lightweight frame grabber"""
    
    def __init__(self, cam_id, cookies=None):
        super().__init__(daemon=True)
        self.cam_id = cam_id
        self.url = STREAM_BASE.format(cam_id)
        self.latest_frame = None
        self.lock = threading.Lock()
        self.running = True
        self.cookies = cookies
        self.fps_actual = 0
    
    def run(self):
        while self.running:
            try:
                cap = cv2.VideoCapture(self.url)
                cap.set(cv2.CAP_PROP_BUFFERSIZE, 2)
                
                frame_count = 0
                start_time = time.time()
                
                while self.running:
                    ret, frame = cap.read()
                    if not ret:
                        break
                    
                    frame_count += 1
                    elapsed = time.time() - start_time
                    
                    if elapsed >= 1.0:
                        self.fps_actual = frame_count / elapsed
                        frame_count = 0
                        start_time = time.time()
                    
                    # Skip frames to hit target FPS
                    stream_fps = cap.get(cv2.CAP_PROP_FPS) or 25
                    skip = max(1, int(stream_fps / PROCESS_FPS))
                    
                    if frame_count % skip != 0:
                        continue
                    
                    with self.lock:
                        self.latest_frame = frame.copy()
                
                cap.release()
                time.sleep(2)  # Reconnect delay
                
            except Exception as e:
                print(f"[CAM{self.cam_id:02d}] Error: {e}")
                time.sleep(5)
    
    def get_frame(self):
        with self.lock:
            return self.latest_frame


class ANPRProcessor(threading.Thread):
    """
    GPU inference thread — pulls frames from ALL cameras,
    batches them, runs detection + recognition.
    """
    
    def __init__(self, camera_workers):
        super().__init__(daemon=True)
        self.workers = camera_workers
        self.voters = {cam_id: TemporalPlateVoter() for cam_id in range(1, 31)}
        self.running = True
    
    def run(self):
        while self.running:
            batch_frames = []
            batch_cam_ids = []
            
            # Collect latest frames from all cameras
            for cam_id, worker in self.workers.items():
                frame = worker.get_frame()
                if frame is not None:
                    batch_frames.append(frame)
                    batch_cam_ids.append(cam_id)
            
            if not batch_frames:
                time.sleep(0.1)
                continue
            
            # ── STAGE 0: Night Enhancement ──
            enhanced_frames = []
            lighting_conditions = []
            for frame in batch_frames:
                enhanced, condition = enhancer.enhance(frame)
                enhanced_frames.append(enhanced)
                lighting_conditions.append(condition)
            
            # ── STAGE 1: Vehicle Detection (BATCHED) ──
            vehicle_results = vehicle_model(
                enhanced_frames,
                conf=0.4,
                classes=[2, 3, 5, 7],  # car, motorcycle, bus, truck
                verbose=False
            )
            
            # Process each camera's results
            for idx, (cam_id, frame, v_result, lighting) in enumerate(
                zip(batch_cam_ids, enhanced_frames, vehicle_results, lighting_conditions)
            ):
                self._process_camera_frame(cam_id, frame, v_result, lighting)
            
            time.sleep(0.05)  # ~20 cycles/sec
    
    def _process_camera_frame(self, cam_id, frame, vehicle_result, lighting):
        for vbox in vehicle_result.boxes:
            vx1, vy1, vx2, vy2 = map(int, vbox.xyxy[0])
            track_id = int(vbox.id[0]) if vbox.id is not None else -1
            
            # Crop vehicle region (lower 60% for plate)
            vh = vy2 - vy1
            vehicle_crop = frame[vy1 + int(vh * 0.4):vy2, vx1:vx2]
            
            if vehicle_crop.size == 0:
                continue
            
            # ── STAGE 2: Plate Detection ──
            plate_results = plate_model(vehicle_crop, conf=0.5, verbose=False)
            
            for pbox in plate_results[0].boxes:
                px1, py1, px2, py2 = map(int, pbox.xyxy[0])
                plate_crop = vehicle_crop[py1:py2, px1:px2]
                
                if plate_crop.size == 0:
                    continue
                
                # ── QUALITY CHECK ──
                quality = selector.score(plate_crop)
                if quality < 30:
                    continue  # Skip garbage crops
                
                # ── STAGE 3: Preprocess ──
                processed = preprocessor.preprocess(plate_crop, lighting)
                
                # ── STAGE 4: CRNN Recognition ──
                # Use your existing CRNN inference
                plate_text, crnn_conf = self._run_crnn(processed)
                
                if not plate_text or crnn_conf < 0.3:
                    continue
                
                # ── STAGE 5: Grammar Fix ──
                from scripts.indian_plate_grammar import fix_plate
                plate_text = fix_plate(plate_text)
                
                # ── STAGE 6: Temporal Vote ──
                voter = self.voters[cam_id]
                voter.add_reading(
                    track_id=track_id,
                    plate_text=plate_text,
                    confidence=crnn_conf,
                    quality_score=quality
                )
                
                # Try to get consensus
                final_plate, final_conf = voter.vote(track_id)
                
                if final_plate and final_conf >= CONFIDENCE_THRESHOLD:
                    detection_log.put({
                        'cam_id': cam_id,
                        'plate': final_plate,
                        'confidence': round(final_conf, 3),
                        'lighting': lighting,
                        'quality': quality,
                        'timestamp': datetime.now().isoformat(),
                        'num_reads': len(voter.tracks.get(track_id, []))
                    })
                    
                    status = "🌙" if "night" in lighting else "☀️"
                    print(f"{status} [CAM{cam_id:02d}] {final_plate} "
                          f"| conf={final_conf:.2f} | reads={len(voter.tracks.get(track_id, []))}")
    
    def _run_crnn(self, processed_plate):
        """Hook into your existing CRNN from anpr_engine.py"""
        # Replace with your actual CRNN inference call
        # Example:
        # return anpr_engine.recognize(processed_plate)
        pass


# ─── MAIN LAUNCHER ──────────────────────────────────
def main():
    print("=" * 60)
    print("  SENTINEL GUJARAT — 30-Camera Real-Time ANPR")
    print("=" * 60)
    
    # Launch frame grabbers for all 30 cameras
    workers = {}
    for cam_id in range(1, NUM_CAMERAS + 1):
        w = CameraWorker(cam_id)
        w.start()
        workers[cam_id] = w
        print(f"  [CAM{cam_id:02d}] Stream worker started")
    
    time.sleep(5)  # Let streams connect
    
    # Launch GPU processor
    processor = ANPRProcessor(workers)
    processor.start()
    print("\n🚀 ANPR Processor running...\n")
    
    # Log detections
    try:
        while True:
            if not detection_log.empty():
                det = detection_log.get()
                # Save to DB, trigger alerts, etc.
                log_detection(det)
            time.sleep(0.01)
    except KeyboardInterrupt:
        print("\nShutting down...")


def log_detection(det):
    """Save to SQLite + check watchlist"""
    import sqlite3
    conn = sqlite3.connect("sentinel_detections.db")
    c = conn.cursor()
    c.execute('''CREATE TABLE IF NOT EXISTS detections (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        cam_id INTEGER, plate TEXT, confidence REAL,
        lighting TEXT, quality REAL, timestamp TEXT
    )''')
    c.execute("INSERT INTO detections (cam_id,plate,confidence,lighting,quality,timestamp) VALUES (?,?,?,?,?,?)",
              (det['cam_id'], det['plate'], det['confidence'],
               det['lighting'], det['quality'], det['timestamp']))
    conn.commit()
    conn.close()


if __name__ == "__main__":
    main()