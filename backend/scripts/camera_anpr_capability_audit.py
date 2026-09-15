"""Why does ANPR fail on a given camera — motion blur, or geometry?

Two failures look identical in the capture rate and need opposite fixes:

  motion blur   shutter too slow for the traffic speed. Vehicles smear while
                the static background stays sharp. Fixed by locking the
                shutter and adding light, without moving anything.

  geometry      the plate is foreshortened or not facing the lens at all.
                The image is perfectly sharp and still unreadable. Fixed only
                by re-aiming, lowering, or re-siting the camera.

The discriminator is sharpness of the MOVING vehicle measured against the
STATIC background in the same frame. Same sensor, same lens, same exposure, so
the only difference is motion. A camera whose vehicles are much softer than
its own background is blur-limited. A camera whose vehicles are as sharp as
its background, and still yields no plates, is geometry-limited.

This matters because the two fixes cost very different amounts, and a capture
rate alone cannot tell them apart. It also names which cameras are worth
running the plate pipeline on at all.

Run:  python -m backend.scripts.camera_anpr_capability_audit
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import cv2
import numpy as np

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
if str(ROOT / "backend") not in sys.path:
    sys.path.insert(0, str(ROOT / "backend"))

CLIPS = ROOT / "data" / "clips"
FRAMES_PER_CAM = 14
VEHICLE_CLASSES = [2, 3, 5, 7]


def sharpness(img: np.ndarray) -> float:
    """Laplacian variance on a size-normalised patch.

    Normalising the patch first matters: a larger crop otherwise scores higher
    for having more pixels rather than more detail, which would confound the
    very comparison this function exists to make.
    """
    if img is None or img.size == 0:
        return 0.0
    g = (cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
         if img.ndim == 3 and img.shape[2] >= 3 else img)
    g = cv2.resize(g, (128, 128), interpolation=cv2.INTER_AREA)
    return float(cv2.Laplacian(g, cv2.CV_32F).var())


def blur_anisotropy(img: np.ndarray) -> float:
    """Directional blur within one crop: vertical-edge energy / horizontal.

    An earlier version of this audit compared a vehicle's sharpness against
    random background patches. That was wrong: patches landed on sky and bare
    tarmac, which carry no texture and score near zero whatever the exposure,
    producing sharpness ratios of 100 and above that measured the emptiness of
    the background rather than anything about the vehicle.

    Motion blur has a signature that needs no external reference, because it
    is DIRECTIONAL. Traffic moves broadly horizontally, and a slow shutter
    smears along that axis — which destroys vertical edges (the ones a
    horizontal gradient detects) while leaving horizontal edges intact. So the
    ratio of vertical-edge energy to horizontal-edge energy falls below 1 when
    the exposure is too long, and sits near 1 on a sharp vehicle.

    Self-contained, exposure-independent, and it cannot be fooled by a
    featureless background.
    """
    if img is None or img.size == 0:
        return 1.0
    g = (cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
         if img.ndim == 3 and img.shape[2] >= 3 else img)
    g = cv2.resize(g, (128, 128), interpolation=cv2.INTER_AREA).astype(np.float32)
    gx = cv2.Sobel(g, cv2.CV_32F, 1, 0, ksize=3)   # responds to VERTICAL edges
    gy = cv2.Sobel(g, cv2.CV_32F, 0, 1, ksize=3)   # responds to HORIZONTAL edges
    ex, ey = float(np.mean(gx ** 2)), float(np.mean(gy ** 2))
    return ex / max(ey, 1e-6)


def main() -> int:
    from ultralytics import YOLO
    from services.anpr_engine import get_anpr_engine, SINGLE_FRAME_FLOOR_PX

    engine = get_anpr_engine()
    vdet = YOLO(str(ROOT / "yolov8s.pt"))

    results = []
    for cam_dir in sorted(d for d in CLIPS.iterdir() if d.is_dir()):
        clips = sorted(cam_dir.glob("*.mp4"))
        if not clips:
            continue
        cap = cv2.VideoCapture(str(clips[0]))
        total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT) or 0)
        if total < 60:
            cap.release()
            continue

        veh_sharp, aniso = [], []
        n_veh = n_plate = n_readable = 0
        for idx in np.linspace(total * 0.15, total * 0.85,
                               FRAMES_PER_CAM).astype(int):
            cap.set(cv2.CAP_PROP_POS_FRAMES, int(idx))
            ok, frame = cap.read()
            if not ok or frame is None:
                continue
            H, W = frame.shape[:2]
            res = vdet.predict(frame, conf=0.35, classes=VEHICLE_CLASSES,
                               verbose=False, device="cuda:0", imgsz=1280)
            boxes = res[0].boxes
            if boxes is None or not len(boxes):
                continue
            enh, light = engine._enhance_frame(frame)
            occupied = []
            for xyxy in boxes.xyxy.cpu().numpy():
                x1, y1, x2, y2 = (int(v) for v in xyxy)
                x1, y1 = max(0, x1), max(0, y1)
                x2, y2 = min(W, x2), min(H, y2)
                if x2 - x1 < 60 or y2 - y1 < 60:
                    continue
                occupied.append((x1, y1, x2, y2))
                n_veh += 1
                veh_sharp.append(sharpness(frame[y1:y2, x1:x2]))
                aniso.append(blur_anisotropy(frame[y1:y2, x1:x2]))
                _, raw, _ = engine.extract_plate_candidate(
                    enh, [x1, y1, x2, y2], 2, lighting=light, raw_frame=frame)
                if raw is not None:
                    n_plate += 1
                    if raw.shape[1] >= SINGLE_FRAME_FLOOR_PX:
                        n_readable += 1

        cap.release()

        if not veh_sharp or n_veh < 3:
            continue
        v_med = float(np.median(veh_sharp))
        a_med = float(np.median(aniso))
        results.append({
            "camera": cam_dir.name, "vehicles": n_veh,
            "plate_boxes": n_plate, "readable": n_readable,
            "veh_sharp": v_med, "aniso": a_med,
            "capture": 100.0 * n_readable / max(n_veh, 1),
        })

    results.sort(key=lambda r: -r["capture"])
    print(f"{'camera':<9}{'veh':>5}{'plate':>7}{'readable':>10}{'capture':>9}"
          f"{'sharpness':>11}{'V/H edges':>11}  diagnosis")
    print("-" * 88)
    for r in results:
        # A vehicle much softer than its own background is smeared by motion.
        # Both sharp and still no plate points at where the camera is aimed.
        if r["ratio"] < 0.55:
            dx = "MOTION BLUR — lock shutter"
        elif r["capture"] < 5.0:
            dx = "GEOMETRY — re-aim / lower"
        elif r["capture"] < 25.0:
            dx = "marginal — tighten FOV"
        else:
            dx = "WORKING — use as reference"
        print(f"{r['camera']:<9}{r['vehicles']:>5}{r['plate_boxes']:>7}"
              f"{r['readable']:>10}{r['capture']:>8.1f}%{r['veh_sharp']:>11.0f}"
              f"{r['bg_sharp']:>10.0f}{r['ratio']:>8.2f}  {dx}")

    blur = [r for r in results if r["ratio"] < 0.55]
    geom = [r for r in results if r["ratio"] >= 0.55 and r["capture"] < 5.0]
    work = [r for r in results if r["capture"] >= 25.0]
    print(f"""
{len(blur)} cameras blur-limited, {len(geom)} geometry-limited, {len(work)} working.

Blur-limited cameras are the cheap ones: locking the shutter and adding
illumination fixes them without touching the mount. Geometry-limited cameras
need physical work, and no shutter or lens change rescues them.""")

    out = ROOT / "output" / "camera_anpr_capability.json"
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(results, indent=1), encoding="utf-8")
    print(f"\nwrote {out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
