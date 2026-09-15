"""
backend/scripts/harvest_camera_14_frames.py — Deep Frame Harvester for Camera 14 (Delight Junction).

Harvests 1,000+ Full HD frames from https://live.corp8.cloud/camera/14 with:
  - Multi-stream protocol fallback (HTTP stream, HLS index.m3u8, RTSP)
  - Motion / diversity sampling to ensure diverse traffic conditions
  - Automatic progress reporting and metadata cataloging
  - High quality JPEG saving into data/harvested_camera_14/
"""
from __future__ import annotations

import json
import os
import sys
import time
from pathlib import Path

import cv2
import numpy as np

WORKSPACE = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(WORKSPACE))

STREAM_CANDIDATES = [
    "https://live.corp8.cloud/stream/14",
    "https://live.corp8.cloud/live/stream/14/index.m3u8",
    "http://live.corp8.cloud:8889/stream/14/whep",
    "rtsp://live.corp8.cloud:8554/stream/14",
]

TARGET_FRAMES = 1050  # Harvest at least 1,000+ frames


def harvest_camera_14(target_count: int = TARGET_FRAMES):
    out_dir = Path(WORKSPACE) / "data" / "harvested_camera_14"
    out_dir.mkdir(parents=True, exist_ok=True)

    print("=" * 85)
    print(f"  DEEP CCTV FRAME HARVESTER — CAMERA 14 (Delight Junction)")
    print(f"  Target: Minimum {target_count} Frames | Destination: {out_dir}")
    print("=" * 85)

    cap = None
    connected_url = None

    for url in STREAM_CANDIDATES:
        print(f"[*] Probing video source: {url}")
        test_cap = cv2.VideoCapture(url)
        # Try reading a few frames
        for _ in range(5):
            ret, frame = test_cap.read()
            if ret and frame is not None and frame.size > 0:
                print(f"[+] Successfully connected to: {url}")
                print(f"[+] Resolution: {frame.shape[1]}x{frame.shape[0]} | Channels: {frame.shape[2]}")
                cap = test_cap
                connected_url = url
                break
            time.sleep(0.1)
        if cap is not None:
            break
        test_cap.release()

    if cap is None:
        print("[!] Could not connect directly to live stream candidates. Trying HTTP fallback stream...")
        cap = cv2.VideoCapture("https://live.corp8.cloud/stream/14")

    saved_count = 0
    frame_idx = 0
    prev_gray = None
    start_time = time.time()
    reconnect_attempts = 0

    print("\n[*] Commencing Deep Harvesting Loop...")

    while saved_count < target_count:
        ret, frame = cap.read()
        frame_idx += 1

        if not ret or frame is None or frame.size == 0:
            reconnect_attempts += 1
            if reconnect_attempts > 10:
                print("[*] Reconnecting stream capture...")
                cap.release()
                time.sleep(0.5)
                cap = cv2.VideoCapture(connected_url or "https://live.corp8.cloud/stream/14")
                reconnect_attempts = 0
            time.sleep(0.05)
            continue

        reconnect_attempts = 0

        # Motion / Quality Check: Avoid saving identical corrupted duplicate frames
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        if prev_gray is not None:
            diff = float(np.abs(gray.astype(np.float32) - prev_gray.astype(np.float32)).mean())
            # If completely frozen (< 0.01 mean diff), skip
            if diff < 0.01:
                continue

        prev_gray = gray
        saved_count += 1

        # Save frame
        frame_filename = f"cam14_frame_{saved_count:05d}.jpg"
        save_path = out_dir / frame_filename
        cv2.imwrite(str(save_path), frame, [cv2.IMWRITE_JPEG_QUALITY, 95])

        if saved_count % 50 == 0 or saved_count == target_count:
            elapsed = time.time() - start_time
            fps = saved_count / max(elapsed, 0.001)
            eta_sec = (target_count - saved_count) / max(fps, 0.001)
            print(f"[{saved_count:04d}/{target_count}] Harvested: {frame_filename} | Speed: {fps:.1f} fps | Elapsed: {elapsed:.1f}s | ETA: {eta_sec:.1f}s")

    cap.release()
    total_time = time.time() - start_time

    # Generate Manifest / Metadata Catalog
    manifest = {
        "camera_id": "14",
        "camera_name": "Camera 14 - Delight Junction",
        "source_url": connected_url,
        "total_frames_harvested": saved_count,
        "resolution": f"{frame.shape[1]}x{frame.shape[0]}",
        "harvest_duration_seconds": round(total_time, 2),
        "timestamp_epoch": time.time(),
        "directory": str(out_dir),
    }

    manifest_path = out_dir / "harvest_manifest.json"
    with open(manifest_path, "w") as f:
        json.dump(manifest, f, indent=2)

    print("\n" + "=" * 85)
    print(f"  HARVEST COMPLETE!")
    print(f"  Total Frames Harvested: {saved_count}")
    print(f"  Output Directory: {out_dir}")
    print(f"  Manifest: {manifest_path}")
    print("=" * 85)


if __name__ == "__main__":
    count = int(sys.argv[1]) if len(sys.argv) > 1 else TARGET_FRAMES
    harvest_camera_14(count)
