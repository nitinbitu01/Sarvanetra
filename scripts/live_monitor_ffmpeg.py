"""
sentinel gujarat/scripts/live_monitor_ffmpeg.py

Production 30-Camera Concurrent Live CCTV ANPR Monitor:
  • Connects concurrently to all 30 live Gujarat CCTV cameras on https://cctv.corp8.cloud/
  • Uses real-time FFmpeg pipe with authenticated HLS decryption headers
  • Runs YOLOv8 vehicle detection, license plate localization, CRNN OCR + Indian plate grammar
  • Multi-frame temporal consensus voting and real-time Watchlist / Stolen vehicle alerts
  • Logs structured reads to logs/live_anpr/ and prints real-time detections to terminal
"""
from __future__ import annotations

import sys
import cv2
import re
import json
import time
import subprocess
import threading
import numpy as np
from pathlib import Path
from datetime import datetime
from ultralytics import YOLO

ROOT    = Path(__file__).resolve().parent.parent
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

from services.anpr_engine import get_anpr_engine

# ── Config ────────────────────────────────────────────────────────────────────
BASE    = "https://cctv.corp8.cloud"
VALID   = re.compile(r"^[A-Z]{2}[0-9]{1,2}[A-Z]{0,3}[0-9]{4}$")
LOG_DIR = ROOT / "logs" / "live_anpr"
LOG_DIR.mkdir(parents=True, exist_ok=True)

# Cookie load
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

# Target frame size
FRAME_W = 1280
FRAME_H = 720

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

print_lock = threading.Lock()
running = True

def open_ffmpeg_pipe(cam_id: str):
    """
    Open real-time authenticated FFmpeg pipe for HLS stream.
    """
    url = f"{BASE}/{cam_id}/index.m3u8"
    headers_str = (
        f"User-Agent: Mozilla/5.0 (Windows NT 10.0; Win64; x64)\r\n"
        f"Cookie: {COOKIE_STR}\r\n"
        f"Referer: {BASE}/\r\n"
    )

    cmd = [
        "ffmpeg",
        "-loglevel", "error",
        "-headers", headers_str,
        "-re",
        "-allowed_extensions", "ALL",
        "-protocol_whitelist", "file,crypto,data,http,https,tcp,tls,httpproxy",
        "-i", url,
        "-vf", f"fps=3,scale={FRAME_W}:{FRAME_H}",
        "-f", "rawvideo",
        "-pix_fmt", "bgr24",
        "-an",
        "-",
    ]

    return subprocess.Popen(
        cmd,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        bufsize=FRAME_W * FRAME_H * 3 * 5,
    )

def read_frame(proc) -> np.ndarray:
    """Read a raw BGR frame from FFmpeg pipe."""
    frame_bytes = FRAME_W * FRAME_H * 3
    data = b""
    while len(data) < frame_bytes:
        chunk = proc.stdout.read(frame_bytes - len(data))
        if not chunk:
            return None
        data += chunk

    return np.frombuffer(data, dtype=np.uint8).reshape((FRAME_H, FRAME_W, 3))

def camera_worker(cam_id: str, location: str, engine, vdet):
    log_file = LOG_DIR / f"{cam_id}.jsonl"
    tbase    = abs(hash(cam_id)) % 100000

    with print_lock:
        print(f"[{cam_id.upper()}] Connected -> {location}")

    while running:
        proc = open_ffmpeg_pipe(cam_id)
        frame_idx = 0
        empty_cnt = 0

        try:
            while running:
                frame = read_frame(proc)
                if frame is None:
                    empty_cnt += 1
                    if empty_cnt > 20:
                        break
                    time.sleep(0.1)
                    continue

                empty_cnt  = 0
                frame_idx += 1

                # ── Vehicle detection ─────────────────────────────────────────
                try:
                    res = vdet.predict(
                        frame, imgsz=640, conf=0.20,
                        classes=[2, 3, 5, 7], verbose=False,
                    )
                except Exception:
                    continue

                if not res or res[0].boxes is None:
                    continue

                # ── ANPR on detected vehicles ─────────────────────────────────
                for i, box in enumerate(res[0].boxes[:3]):
                    bbox     = box.xyxy[0].cpu().numpy().tolist()
                    cls_id   = int(box.cls[0])
                    track_id = tbase + (frame_idx % 10000) * 10 + i

                    try:
                        r = engine.process_vehicle_track(
                            frame    = frame,
                            bbox     = bbox,
                            cls_id   = cls_id,
                            track_id = track_id,
                        )
                    except Exception:
                        continue

                    if not r or not r.get("plate"):
                        continue

                    plate = r["plate"]
                    conf  = r.get("confidence", 0.0)
                    votes = r.get("total_votes", 1)

                    # Quality filters
                    if conf < 0.45:
                        continue
                    if not VALID.match(plate):
                        continue

                    is_stolen = r.get("is_stolen", False)
                    ts_str    = datetime.now().strftime("%H:%M:%S")

                    # Console print
                    with print_lock:
                        tag = " 🚨 [WATCHLIST HIT / STOLEN]" if is_stolen else ""
                        print(f"[{ts_str}] [{cam_id.upper():<6}] Plate: {plate:<12} | Conf: {conf*100:4.1f}% | Loc: {location[:18]:<18}{tag}")

                    # Structured JSON log
                    rec = {
                        "ts":     datetime.now().isoformat(),
                        "cam":    cam_id,
                        "loc":    location,
                        "plate":  plate,
                        "conf":   round(conf, 3),
                        "votes":  votes,
                        "stolen": is_stolen,
                        "alert":  r.get("alert_reason") or "",
                    }

                    try:
                        with open(log_file, "a", encoding="utf-8") as f:
                            f.write(json.dumps(rec) + "\n")
                    except Exception:
                        pass

        finally:
            try:
                proc.terminate()
            except Exception:
                pass

        if running:
            time.sleep(4)

def main():
    global running
    print("=" * 65)
    print("  SENTINEL GUJARAT — 30-CAMERA CONCURRENT LIVE ANPR MONITOR")
    print(f"  Target Portal: {BASE}")
    print(f"  Log Directory: {LOG_DIR}")
    print("  Press Ctrl+C in terminal to stop")
    print("=" * 65)

    engine = get_anpr_engine()
    vdet   = YOLO(str(ROOT / "yolov8s.pt"))

    threads = []
    for cam_id, loc in ALL_CAMERAS.items():
        t = threading.Thread(
            target=camera_worker,
            args=(cam_id, loc, engine, vdet),
            daemon=True,
        )
        t.start()
        threads.append(t)
        time.sleep(0.08)  # Stagger thread startup slightly

    print(f"\n✅ All {len(threads)} live camera workers active and monitoring.\n")

    try:
        while True:
            time.sleep(1)
    except KeyboardInterrupt:
        print("\nStopping all 30 live camera monitor threads...")
        running = False

if __name__ == "__main__":
    main()