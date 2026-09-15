"""Detector input size, measured the way the engine actually runs it.

The first ablation ran the plate detector over whole frames and found that
1280 more than doubled detections. That result does not transfer, because the
engine does not work that way: extract_plate_candidate crops the vehicle first
and runs the detector on that crop at PLATE_DET_IMGSZ = 320.

A vehicle crop is a few hundred pixels across, so 320 may already be
upsampling it — in which case a larger value buys nothing and costs time. Or
the crop may be larger than 320 on close vehicles, in which case the plate is
being shrunk before the detector ever sees it. Which of those is happening is
an empirical question about this fleet's crop sizes, and it decides whether
there is anything to gain.

This measures on real vehicle crops taken through the real vehicle detector,
so the inputs are the ones the ANPR engine sees in production.

Run:  python -m backend.scripts.plate_det_imgsz_on_crops
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
VEHICLE_MODEL = ROOT / "yolov8s.pt"
CLIPS = ROOT / "data" / "clips"

CAMERAS = ["CAM_05", "CAM_10", "CAM_06", "CAM_14", "CAM_08", "CAM_04"]
FRAMES_PER_CAM = 20
SIZES = (320, 480, 640, 960)
PLATE_CONF = 0.15
VEHICLE_CONF = 0.35
VEHICLE_CLASSES = [2, 3, 5, 7]


def sample_frames(cam: str, n: int) -> list:
    clips = sorted((CLIPS / cam).glob("*.mp4"))
    if not clips:
        return []
    cap = cv2.VideoCapture(str(clips[0]))
    total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT) or 0)
    out = []
    for idx in np.linspace(total * 0.1, total * 0.9, n).astype(int):
        cap.set(cv2.CAP_PROP_POS_FRAMES, int(idx))
        ok, f = cap.read()
        if ok and f is not None:
            out.append(f)
    cap.release()
    return out


def main() -> int:
    vdet = YOLO(str(VEHICLE_MODEL))
    pdet = YOLO(str(PLATE_MODEL))

    # Build the crop set once: exactly what extract_plate_candidate receives.
    crops: list[tuple[str, np.ndarray]] = []
    for cam in CAMERAS:
        for fr in sample_frames(cam, FRAMES_PER_CAM):
            res = vdet.predict(fr, conf=VEHICLE_CONF, classes=VEHICLE_CLASSES,
                               verbose=False, device="cuda:0", imgsz=640)
            b = res[0].boxes
            if b is None or not len(b):
                continue
            for xyxy in b.xyxy.cpu().numpy():
                x1, y1, x2, y2 = (int(v) for v in xyxy)
                x1, y1 = max(0, x1), max(0, y1)
                x2, y2 = min(fr.shape[1], x2), min(fr.shape[0], y2)
                if x2 - x1 >= 40 and y2 - y1 >= 40:
                    crops.append((cam, fr[y1:y2, x1:x2]))

    if not crops:
        print("no vehicle crops produced")
        return 1

    widths = [c.shape[1] for _, c in crops]
    print(f"{len(crops)} vehicle crops from {len(CAMERAS)} cameras")
    print(f"crop width: median {int(np.median(widths))}px, "
          f"p10 {int(np.percentile(widths,10))}, "
          f"p90 {int(np.percentile(widths,90))}\n")
    over = 100.0 * sum(1 for w in widths if w > 320) / len(widths)
    print(f"{over:.0f}% of crops are wider than the current imgsz of 320 — "
          f"those are downscaled before the detector sees them\n", flush=True)

    per_cam_found: dict[int, dict] = defaultdict(lambda: defaultdict(int))
    print(f"{'imgsz':>7}{'plates':>9}{'vs 320':>9}{'median w':>11}"
          f"{'>=100px':>10}{'ms/crop':>10}")
    print("-" * 58)

    base_found = None
    for size in SIZES:
        found = 0
        pw: list[float] = []
        # Warm up so the first size is not charged for CUDA init.
        pdet.predict(crops[0][1], imgsz=size, conf=PLATE_CONF, verbose=False,
                     device="cuda:0")
        t0 = time.time()
        for cam, crop in crops:
            r = pdet.predict(crop, imgsz=size, conf=PLATE_CONF, verbose=False,
                             device="cuda:0")
            b = r[0].boxes
            if b is not None and len(b):
                found += 1                      # one plate per vehicle
                best = int(b.conf.argmax())
                xy = b.xyxy[best].cpu().numpy()
                pw.append(float(xy[2] - xy[0]))
                per_cam_found[size][cam] += 1
        dt = (time.time() - t0) / len(crops) * 1000
        if base_found is None:
            base_found = found
        delta = (found - base_found) / max(base_found, 1) * 100
        med = float(np.median(pw)) if pw else 0.0
        pct100 = 100.0 * sum(1 for w in pw if w >= 100) / max(len(pw), 1)
        print(f"{size:>7}{found:>9}{delta:>+8.0f}%{med:>10.0f}px"
              f"{pct100:>9.1f}%{dt:>9.1f}", flush=True)

    print(f"\n{'camera':<10}" + "".join(f"{s:>9}" for s in SIZES))
    print("-" * (10 + 9 * len(SIZES)))
    for cam in CAMERAS:
        if any(per_cam_found[s].get(cam) for s in SIZES):
            print(f"{cam:<10}" + "".join(f"{per_cam_found[s].get(cam,0):>9}"
                                          for s in SIZES))

    print("""
The plate crop width column is what decides whether a change is worth making:
accuracy is 0% below 70 px and 51.7% at 100-140 px, so finding more plates
that are too small to read adds cost without adding reads. Cost is per
vehicle crop, and a busy frame holds several.""")
    return 0


if __name__ == "__main__":
    sys.exit(main())
