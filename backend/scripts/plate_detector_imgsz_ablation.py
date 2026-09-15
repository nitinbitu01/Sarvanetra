"""Does a larger detector input find more plates?

Live measurement put capture rate at 70.6% and decode rate at 100%: every
plate the detector locates is decoded, and the entire loss is plates never
found. That makes the detector's input resolution the lever, because a plate
70 px wide in a 1920-wide frame is 23 px after the frame is scaled to 640, and
a 23 px box is close to what the detector can resolve at all.

This measures, on real frames from the cameras that are actually failing:

    how many plate regions the detector finds at each input size
    how wide the crops it returns are, since accuracy tracks crop width
      (0% below 70 px, 25% at 70-100, 51.7% at 100-140)
    what each size costs per frame

Frames come from the cameras with the worst measured capture rate, because a
change that only helps CAM_08 (88.9%) does not move the fleet.

Run:  python -m backend.scripts.plate_detector_imgsz_ablation
"""
from __future__ import annotations

import sys
import time
from collections import defaultdict
from pathlib import Path

import cv2
import numpy as np

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from ultralytics import YOLO                                       # noqa: E402

PLATE_MODEL = ROOT / "runs/detect/runs/plate/plate_v3_ft/weights/best.pt"
CLIPS = ROOT / "data" / "clips"

# Worst and best measured capture rates, so the effect can be seen on the
# cameras that need it and checked for regression on the one that does not.
CAMERAS = ["CAM_05", "CAM_10", "CAM_06", "CAM_14", "CAM_08"]
FRAMES_PER_CAM = 24
SIZES = (640, 960, 1280, 1600)
CONF = 0.15          # matches PLATE_DET_CONF in anpr_engine


def sample_frames(cam: str, n: int) -> list:
    clips = sorted((CLIPS / cam).glob("*.mp4"))
    if not clips:
        return []
    cap = cv2.VideoCapture(str(clips[0]))
    total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT) or 0)
    frames = []
    if total > n:
        # Spread across the clip rather than taking a run of near-identical
        # consecutive frames.
        for idx in np.linspace(total * 0.1, total * 0.9, n).astype(int):
            cap.set(cv2.CAP_PROP_POS_FRAMES, int(idx))
            ok, f = cap.read()
            if ok and f is not None:
                frames.append(f)
    else:
        while len(frames) < n:
            ok, f = cap.read()
            if not ok:
                break
            frames.append(f)
    cap.release()
    return frames


def main() -> int:
    if not PLATE_MODEL.is_file():
        print(f"plate detector not found at {PLATE_MODEL}")
        return 1

    model = YOLO(str(PLATE_MODEL))
    frames_by_cam = {c: sample_frames(c, FRAMES_PER_CAM) for c in CAMERAS}
    frames_by_cam = {c: f for c, f in frames_by_cam.items() if f}
    total_frames = sum(len(f) for f in frames_by_cam.values())
    print(f"{total_frames} frames from {len(frames_by_cam)} cameras\n", flush=True)

    results: dict[int, dict] = {}
    per_cam: dict[int, dict] = defaultdict(dict)

    for size in SIZES:
        found = 0
        widths: list[float] = []
        t0 = time.time()
        for cam, frames in frames_by_cam.items():
            cam_found = 0
            cam_widths = []
            for fr in frames:
                res = model.predict(fr, imgsz=size, conf=CONF, verbose=False,
                                    device="cuda:0")
                b = res[0].boxes
                if b is not None and len(b):
                    xyxy = b.xyxy.cpu().numpy()
                    cam_found += len(xyxy)
                    # Width in the SOURCE frame, which is what the recogniser
                    # eventually crops — not width at the detector's input
                    # scale, which would flatter a larger imgsz automatically.
                    cam_widths.extend((xyxy[:, 2] - xyxy[:, 0]).tolist())
            found += cam_found
            widths.extend(cam_widths)
            per_cam[size][cam] = {
                "found": cam_found,
                "frames": len(frames),
                "median_w": float(np.median(cam_widths)) if cam_widths else 0.0,
            }
        dt = (time.time() - t0) / max(total_frames, 1) * 1000
        results[size] = {
            "found": found,
            "per_frame": found / max(total_frames, 1),
            "median_w": float(np.median(widths)) if widths else 0.0,
            # The share of crops that land in a band the recogniser can
            # actually read. Finding more plates helps only if they are big
            # enough to decode.
            "pct_over_100px": (100.0 * sum(1 for w in widths if w >= 100)
                               / max(len(widths), 1)),
            "ms_per_frame": dt,
        }
        print(f"imgsz {size:>5}: {found:>4} plates  "
              f"{found/max(total_frames,1):.2f}/frame  "
              f"median width {results[size]['median_w']:>5.0f}px  "
              f"{results[size]['pct_over_100px']:>5.1f}% over 100px  "
              f"{dt:>6.1f} ms/frame", flush=True)

    base = results[SIZES[0]]
    print(f"\n{'imgsz':>7}{'plates':>9}{'vs 640':>9}{'median w':>11}"
          f"{'>=100px':>10}{'ms/frame':>11}")
    print("-" * 60)
    for size in SIZES:
        r = results[size]
        delta = (r["found"] - base["found"]) / max(base["found"], 1) * 100
        print(f"{size:>7}{r['found']:>9}{delta:>+8.0f}%"
              f"{r['median_w']:>10.0f}px{r['pct_over_100px']:>9.1f}%"
              f"{r['ms_per_frame']:>10.1f}")

    print(f"\n{'camera':<10}" + "".join(f"{s:>10}" for s in SIZES)
          + "   plates found")
    print("-" * (10 + 10 * len(SIZES) + 16))
    for cam in frames_by_cam:
        row = "".join(f"{per_cam[s][cam]['found']:>10}" for s in SIZES)
        print(f"{cam:<10}{row}")

    print("""
Read the two right-hand columns together. More detections only help if the
crops are large enough to read: accuracy is 0% below 70 px and 51.7% at
100-140 px, so a size that finds more plates but leaves them under 70 px has
added work, not reads. The cost column is per frame per camera — at 27
cameras it multiplies.""")
    return 0


if __name__ == "__main__":
    sys.exit(main())
