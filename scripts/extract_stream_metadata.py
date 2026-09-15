import os
import csv
import cv2
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
clips_dir = ROOT / "data" / "clips"
print(f"Scanning clips in: {clips_dir}")

mp4_files = list(clips_dir.rglob("*.mp4"))
print(f"Found {len(mp4_files)} MP4 video streams.")

rows = []
for mp4 in sorted(mp4_files):
    cap = cv2.VideoCapture(str(mp4))
    if cap.isOpened():
        w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        fps = round(cap.get(cv2.CAP_PROP_FPS), 2)
        count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        duration_sec = round(count / max(1, fps), 2)
        cap.release()
        
        rows.append({
            "Camera": mp4.stem,
            "Resolution": f"{w}x{h}",
            "FPS": fps,
            "Total_Frames": count,
            "Duration_Sec": duration_sec,
            "Path": str(mp4.relative_to(ROOT.parent))
        })

out_csv = Path("stream_metadata.csv")
with open(out_csv, "w", newline="", encoding="utf-8") as f:
    writer = csv.DictWriter(f, fieldnames=["Camera", "Resolution", "FPS", "Total_Frames", "Duration_Sec", "Path"])
    writer.writeheader()
    writer.writerows(rows)

print(f"Successfully extracted technical specs for {len(rows)} camera streams into stream_metadata.csv")
