"""
test_seeking_12h.py — Test seeking across the 12-hour video stream for Camera 14.
"""
import subprocess
import time
from pathlib import Path
import cv2

url = "https://live.corp8.cloud/stream/14"

print("[*] Testing OpenCV seek...")
cap = cv2.VideoCapture(url)
total_frames = cap.get(cv2.CAP_PROP_FRAME_COUNT)
fps = cap.get(cv2.CAP_PROP_FPS)
print(f"OpenCV total frames: {total_frames}, fps: {fps}")

# Test seeking to 1000s, 5000s, 15000s, 30000s via ffmpeg
test_seconds = [100, 3600, 7200, 14400, 21600, 28800, 36000, 42000]
out_dir = Path("data/test_12h_seeking")
out_dir.mkdir(parents=True, exist_ok=True)

for s in test_seconds:
    h = s // 3600
    m = (s % 3600) // 60
    sec = s % 60
    time_str = f"{h:02d}:{m:02d}:{sec:02d}"
    out_file = out_dir / f"test_seek_{s}s.jpg"
    print(f"[*] Testing ffmpeg seek to {time_str} ({s}s)...")
    cmd = [
        "ffmpeg", "-y",
        "-ss", str(s),
        "-i", url,
        "-vframes", "1",
        "-q:v", "2",
        str(out_file)
    ]
    t0 = time.time()
    res = subprocess.run(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE, timeout=15)
    t1 = time.time()
    if out_file.exists() and out_file.stat().st_size > 0:
        print(f"  [+] Success at {time_str}! File size: {out_file.stat().st_size} bytes, Time taken: {t1-t0:.2f}s")
    else:
        print(f"  [-] Failed at {time_str}. Return code: {res.returncode}")
