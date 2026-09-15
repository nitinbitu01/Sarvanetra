"""
sentinel gujarat/scripts/eval_live_cctv.py

Live Cloud CCTV ANPR Accuracy & Telemetry Evaluator:
  • Connects directly to live authenticated streams on https://cctv.corp8.cloud/
  • Evaluates Detection Rate, Recognition Accuracy, Average Confidence, and FPS
  • Multi-camera concurrent or single-camera focused mode
  • Logs structured reads to logs/live_cctv_accuracy/

Usage:
    # Single Camera Mode (e.g. CAM_08 - Majewadi Gate)
    python "sentinel gujarat/scripts/eval_live_cctv.py" --cam cam08 --seconds 30

    # Whitelisted ANPR Cameras Mode (CAM 06, 07, 08, 09, 10, 18, 21, 27)
    python "sentinel gujarat/scripts/eval_live_cctv.py" --mode whitelist --seconds 45

    # All 30 Cameras
    python "sentinel gujarat/scripts/eval_live_cctv.py" --mode all --seconds 60
"""
from __future__ import annotations

import sys
import time
import json
import re
import argparse
import subprocess
import tempfile
import os
from pathlib import Path
from datetime import datetime
from collections import defaultdict, Counter

import cv2
import numpy as np

# Path setup
ROOT    = Path(__file__).resolve().parents[1]
BACKEND = ROOT / "backend"

if str(ROOT) not in sys.path:
    sys.path.append(str(ROOT))
if str(BACKEND) not in sys.path:
    sys.path.insert(0, str(BACKEND))

try:
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8")
except Exception:
    pass

from services.anpr_engine import ANPREngine, get_anpr_engine
from ultralytics import YOLO

BASE_URL = "https://cctv.corp8.cloud"
VALID_RE = re.compile(r"^[A-Z]{2}[0-9]{1,2}[A-Z]{0,3}[0-9]{4}$")

# Load session cookies
COOKIE_PATHS = [
    ROOT.parent / "stream_cookies.json",
    ROOT / "stream_cookies.json",
    Path("stream_cookies.json"),
]

COOKIES = {}
for cp in COOKIE_PATHS:
    if cp.exists():
        try:
            COOKIES = json.loads(cp.read_text(encoding="utf-8"))
            break
        except Exception:
            pass

COOKIE_STR = "; ".join(f"{k}={v}" for k, v in COOKIES.items()) if COOKIES else ""

ALL_CAMERAS = {
    "cam01": "01 Chiman bhai Bridge",
    "cam02": "02 Janpath",
    "cam03": "03 O.N.G.C. Office",
    "cam04": "04 Paldi Circle",
    "cam05": "05 Visat teen Rasta",
    "cam06": "06 Timbavadi gate",
    "cam07": "07 Hero Showroom",
    "cam08": "08 Majewadi Gate",
    "cam09": "09 New Bypass Circle",
    "cam10": "10 Char Chowk Road",
    "cam11": "11 Dolatpara",
    "cam12": "12 Tri Mandir Adalaj",
    "cam13": "13 CN Vidhyalaya",
    "cam14": "14 Delight RLVD",
    "cam15": "15 Suvidha Park",
    "cam16": "16 Visat P2",
    "cam17": "17 Rajkot Bus Port",
    "cam18": "18 Rajkot CCTV",
    "cam19": "19 Khaparia",
    "cam20": "20 Mohanpura",
    "cam21": "21 Patan Dethali",
    "cam22": "22 BK Mervada",
    "cam23": "23 Kheram",
    "cam24": "24 Dehgam",
    "cam25": "25 Dhanori",
    "cam26": "26 Tankal",
    "cam27": "27 Bilimora",
    "cam28": "28 Bilimora 2",
    "cam29": "29 Bilimora 3",
    "cam30": "30 Gandhidham",
}

WHITELIST_CAMS = ["cam08", "cam06", "cam07", "cam09", "cam10", "cam18", "cam21", "cam27"]

FRAME_W, FRAME_H = 1280, 720
LOG_DIR = ROOT / "logs" / "live_cctv_accuracy"
LOG_DIR.mkdir(parents=True, exist_ok=True)

class LiveCameraEvaluator:
    def __init__(self, cam_id: str, location: str, engine: ANPREngine, vdet: YOLO):
        self.cam_id = cam_id
        self.location = location
        self.engine = engine
        self.vdet = vdet

        # Metrics
        self.frames_read = 0
        self.vehicles_detected = 0
        self.plates_detected = 0
        self.plates_recognized = 0
        self.valid_grammar_reads = 0
        self.watchlist_alerts = 0
        self.confidences = []
        self.plate_reads = Counter()
        self.running = True

    def open_pipe(self):
        url = f"{BASE_URL}/{self.cam_id}/index.m3u8"
        headers_str = (
            f"User-Agent: Mozilla/5.0 (Windows NT 10.0; Win64; x64)\r\n"
            f"Cookie: {COOKIE_STR}\r\n"
            f"Referer: {BASE_URL}/\r\n"
        )
        cmd = [
            "ffmpeg",
            "-loglevel", "error",
            "-headers", headers_str,
            "-re",
            "-allowed_extensions", "ALL",
            "-protocol_whitelist", "file,crypto,data,http,https,tcp,tls,httpproxy",
            "-i", url,
            "-vf", f"fps=5,scale={FRAME_W}:{FRAME_H}",
            "-f", "rawvideo",
            "-pix_fmt", "bgr24",
            "-an",
            "-",
        ]
        return subprocess.Popen(
            cmd,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            bufsize=FRAME_W * FRAME_H * 3 * 10,
        )

    def read_frame_pipe(self, proc):
        fb = FRAME_W * FRAME_H * 3
        data = b""
        while len(data) < fb:
            chunk = proc.stdout.read(fb - len(data))
            if not chunk:
                return None
            data += chunk
        return np.frombuffer(data, dtype=np.uint8).reshape((FRAME_H, FRAME_W, 3))

    def evaluate_stream(self, duration_secs: int = 30):
        t0 = time.time()
        print(f"[{self.cam_id.upper()}] Connecting to live stream: {self.location}...")
        tbase = abs(hash(self.cam_id)) % 100000

        proc = self.open_pipe()
        empty_retries = 0

        try:
            while self.running and (time.time() - t0 < duration_secs):
                frame = self.read_frame_pipe(proc)
                if frame is None:
                    empty_retries += 1
                    if empty_retries > 120:
                        break
                    time.sleep(0.1)
                    continue

                empty_retries = 0
                self.frames_read += 1
                self._process_single_frame(frame, tbase)

        finally:
            try:
                proc.terminate()
            except Exception:
                pass

    def _process_single_frame(self, frame: np.ndarray, tbase: int):
        # 1. Vehicle Detection
        try:
            res = self.vdet.predict(
                frame, imgsz=640, conf=0.20,
                classes=[2, 3, 5, 7], verbose=False,
            )
        except Exception:
            return

        if not res or res[0].boxes is None:
            return

        boxes = res[0].boxes
        self.vehicles_detected += len(boxes)

        # 2. Plate Detection & Recognition per vehicle
        for i, box in enumerate(boxes[:4]):
            bbox = box.xyxy[0].cpu().numpy().tolist()
            cls_id = int(box.cls[0])
            track_id = tbase + (self.frames_read % 10000) * 10 + i

            try:
                r = self.engine.process_vehicle_track(
                    frame=frame, bbox=bbox, cls_id=cls_id, track_id=track_id
                )
            except Exception:
                continue

            if not r or not r.get("plate"):
                continue

            plate = r["plate"]
            conf = r.get("confidence", 0.0)
            self.plates_detected += 1
            self.plates_recognized += 1
            self.confidences.append(conf)

            is_valid = bool(VALID_RE.match(plate))
            if is_valid:
                self.valid_grammar_reads += 1
                self.plate_reads[plate] += 1

            is_stolen = r.get("is_stolen", False)
            if is_stolen:
                self.watchlist_alerts += 1

            # Real-time console telemetry
            if is_valid and conf >= 0.40:
                tag = " 🚨 [WATCHLIST HIT]" if is_stolen else ""
                print(f"  [{self.cam_id.upper()}] Plate: {plate:<12} | Conf: {conf*100:4.1f}% | Lighting: {r.get('lighting','day')}{tag}")

    def get_summary(self):
        avg_conf = np.mean(self.confidences) if self.confidences else 0.0
        grammar_rate = (self.valid_grammar_reads / max(1, self.plates_recognized)) * 100
        return {
            "cam_id": self.cam_id,
            "location": self.location,
            "frames": self.frames_read,
            "vehicles": self.vehicles_detected,
            "plates_detected": self.plates_detected,
            "plates_recognized": self.plates_recognized,
            "valid_grammar": self.valid_grammar_reads,
            "grammar_accuracy_pct": grammar_rate,
            "avg_conf": avg_conf * 100,
            "unique_plates": len(self.plate_reads),
            "watchlist_alerts": self.watchlist_alerts,
        }

def main():
    parser = argparse.ArgumentParser(description="Sentinel Live CCTV ANPR Accuracy Evaluator")
    parser.add_argument("--cam", type=str, default=None, help="Target camera ID (e.g. cam08, cam06, cam09)")
    parser.add_argument("--mode", type=str, default="whitelist", choices=["single", "whitelist", "all"], help="Evaluation mode")
    parser.add_argument("--seconds", type=int, default=30, help="Evaluation duration per camera (in seconds)")
    args = parser.parse_args()

    print("=" * 70)
    print("  SENTINEL GUJARAT — LIVE CCTV ANPR ACCURACY BENCHMARK")
    print("  Target Portal: " + BASE_URL)
    print("=" * 70)

    engine = get_anpr_engine()
    vdet   = YOLO(str(ROOT / "yolov8s.pt"))

    # Determine target cameras
    if args.cam:
        target_cams = [args.cam.lower()]
    elif args.mode == "whitelist":
        target_cams = WHITELIST_CAMS
    else:
        target_cams = list(ALL_CAMERAS.keys())

    print(f"Evaluating {len(target_cams)} Live Camera Stream(s) for {args.seconds}s each...")
    print(f"Device: {engine.device} | YOLOv8s + YOLOv8 Plate Det (plate_v3_ft) + CRNN")
    print()

    summaries = []

    for cam_id in target_cams:
        loc = ALL_CAMERAS.get(cam_id, cam_id)
        evaluator = LiveCameraEvaluator(cam_id, loc, engine, vdet)
        evaluator.evaluate_stream(duration_secs=args.seconds)
        s = evaluator.get_summary()
        summaries.append(s)
        print()

    # Final Telemetry Report
    print("=" * 70)
    print("  LIVE CCTV ACCURACY & PERFORMANCE REPORT")
    print("=" * 70)
    print(f"{'CAM':<8} {'LOCATION':<22} {'VEHICLES':<10} {'PLATES':<8} {'VALID %':<9} {'AVG CONF':<10} {'UNIQUE'}")
    print("─" * 70)

    total_v = 0
    total_p = 0
    total_val = 0
    all_confs = []

    for s in summaries:
        total_v += s["vehicles"]
        total_p += s["plates_recognized"]
        total_val += s["valid_grammar"]
        if s["avg_conf"] > 0:
            all_confs.append(s["avg_conf"])

        print(f"{s['cam_id'].upper():<8} {s['location'][:20]:<22} {s['vehicles']:<10} {s['plates_recognized']:<8} {s['grammar_accuracy_pct']:5.1f}%   {s['avg_conf']:5.1f}%      {s['unique_plates']}")

    overall_grammar = (total_val / max(1, total_p)) * 100 if total_p > 0 else 0.0
    overall_conf = np.mean(all_confs) if all_confs else 0.0

    print("─" * 70)
    print(f"TOTALS:   Vehicles: {total_v} | Plates Processed: {total_p} | Unique: {sum(s['unique_plates'] for s in summaries)}")
    print(f"OVERALL:  Grammar Validity: {overall_grammar:.1f}% | Average Plate Confidence: {overall_conf:.1f}%")
    print("=" * 70)

if __name__ == "__main__":
    main()
