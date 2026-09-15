import cv2
import glob
import os

clips = sorted(glob.glob("data/clips/CAM_09/*.mp4"))
print(f"Found {len(clips)} CAM_09 clips:")
for c in clips:
    cap = cv2.VideoCapture(c)
    n_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    fps = cap.get(cv2.CAP_PROP_FPS)
    w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    dur_s = n_frames / fps if fps > 0 else 0
    cap.release()
    print(f"  {os.path.basename(c)}: {n_frames} frames, {fps:.1f} fps, {dur_s:.1f}s ({dur_s/60:.1f}m), {w}x{h}")
