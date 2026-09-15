"""backend/scripts/collect_plate_scale.py — plate observations paired with the
ground row they were seen at, for a scale that does not depend on vehicle width.

WHY A PLATE IS A BETTER RULER THAN A VEHICLE
  `estimate_height` scales from regulated vehicle widths, and it fails on the
  cameras that most need it. The reason is geometric: the detector's box is
  axis-aligned, so a vehicle crossing obliquely presents a box spanning part of
  its LENGTH as well as its width. The box is too wide, the implied camera
  height too low — the first run of that estimator returned 0.41 m for CAM_18.
  The existing guard is a cone test that keeps only near-axial observations,
  and on CAM_01 and CAM_05 it rejects 100% of them.

  An Indian number plate is 500 x 120 mm and every car carries the same one, so
  it is a far tighter ruler than a width that ranges 1.75-2.60 m by class. More
  importantly it is PLANAR: seen obliquely it foreshortens rather than
  inflating, and its own aspect ratio says by how much. A plate detected at
  4.17:1 is square-on; one at 3.0:1 is turned away, and can be corrected or
  dropped on that evidence alone. A vehicle box carries no such signal.

WHY THIS NEEDS A NEW PASS
  Two fields are needed together and no existing artifact has both:

    plate pixel width   `journey_index/records.jsonl` has a real one
                        (`plate_w`, CV 0.64, 34-423 px)
    ground-contact row  `road_geometry` has it, as the vehicle box bottom

  They cannot be joined: the two were produced by separate tracking runs with
  independent track-id spaces, and a join test matched 0-2%. And
  `plate_corpus`'s `plate_px_est` is not a measurement at all — it is a flat
  0.28 x vehicle width (CV 0.0004), so using it would have been circular.

WHAT IS RECORDED
  Per detected plate: its pixel width and height, the aspect ratio, and the
  bottom row of the vehicle that contains it. Nothing is fitted here — the same
  separation of gathering from fitting the other collectors keep.

Run:  python -m backend.scripts.collect_plate_scale [camera ...]
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

from ultralytics import YOLO                                     # noqa: E402
from backend.services.live_24x7_pipeline import (                # noqa: E402
    YOLO_CONF_THRESHOLD, YOLO_IMGSZ,
)

OUT = ROOT / "output" / "plate_scale"
DET_PLATE = ROOT / "models" / "plate_detector" / "plate_v4_small.pt"

SAMPLE_FPS = 5
SECONDS = 120
VEHICLE_CLASSES = (2, 5, 7)
PLATE_CONF = 0.30
# Below this the plate is too small for its width to mean anything; the
# measured accuracy curve is already ~0% under 40 px.
MIN_PLATE_W = 24


def collect(cam: str, veh, plate, clip: Path) -> dict:
    cap = cv2.VideoCapture(str(clip))
    native = cap.get(cv2.CAP_PROP_FPS) or 25.0
    stride = max(1, int(round(native / SAMPLE_FPS)))
    want = int(SECONDS * native)

    obs, frame_size = [], None
    i = 0
    while i < want:
        ok, frame = cap.read()
        if not ok:
            break
        i += 1
        if i % stride:
            continue
        if frame_size is None:
            frame_size = [int(frame.shape[1]), int(frame.shape[0])]

        res = veh.predict(frame, imgsz=YOLO_IMGSZ, conf=YOLO_CONF_THRESHOLD,
                          classes=list(VEHICLE_CLASSES), verbose=False)
        b = res[0].boxes
        if b is None or not len(b):
            continue
        for xy, cl in zip(b.xyxy.cpu().numpy(), b.cls.cpu().numpy()):
            x1, y1, x2, y2 = (int(xy[0]), int(xy[1]), int(xy[2]), int(xy[3]))
            if x2 - x1 < 60 or y2 - y1 < 40:
                continue
            crop = frame[max(0, y1):y2, max(0, x1):x2]
            if crop.size == 0:
                continue
            pr = plate.predict(crop, conf=PLATE_CONF, verbose=False)[0]
            if pr.boxes is None or not len(pr.boxes):
                continue
            # Largest plate in this vehicle.
            best = max(pr.boxes.xyxy.cpu().numpy(), key=lambda q: q[2] - q[0])
            pw, ph = float(best[2] - best[0]), float(best[3] - best[1])
            if pw < MIN_PLATE_W or ph < 6:
                continue
            obs.append({
                # Ground contact: the bottom-centre of the VEHICLE box is where
                # it meets the road plane. The plate sits above it by a roughly
                # constant amount, which the fitter corrects for.
                "u": (x1 + x2) / 2.0, "v": float(y2),
                "plate_w": round(pw, 1), "plate_h": round(ph, 1),
                "aspect": round(pw / max(ph, 1e-6), 2),
                "veh_w": round(float(x2 - x1), 1),
                "cls": int(cl), "frame": i,
            })
    cap.release()
    return {"camera": cam, "clip": clip.name, "frame_size": frame_size,
            "observations": obs}


def main() -> int:
    cams = sys.argv[1:] or None
    veh = YOLO("yolov8s.pt")
    plate = YOLO(str(DET_PLATE))
    OUT.mkdir(parents=True, exist_ok=True)

    print(f"{'camera':<9}{'clips':>7}{'plates':>9}{'med w':>8}{'med aspect':>12}"
          f"{'square-on':>11}  verdict")
    print("-" * 72)

    for cam_dir in sorted((ROOT / "data" / "clips").iterdir()):
        if not cam_dir.is_dir() or not cam_dir.name.startswith("CAM_"):
            continue
        if cams and cam_dir.name not in cams:
            continue
        clips = sorted(cam_dir.glob("*.mp4"))
        if not clips:
            continue

        merged, frame_size, used = [], None, []
        for clip in clips:
            part = collect(cam_dir.name, veh, plate, clip)
            if not part.get("frame_size"):
                continue
            if frame_size is None:
                frame_size = part["frame_size"]
            elif part["frame_size"] != frame_size:
                continue
            merged.extend(part["observations"])
            used.append(clip.name)

        if frame_size is None:
            continue
        ws = sorted(o["plate_w"] for o in merged)
        asp = sorted(o["aspect"] for o in merged)
        # 4.17 is the real aspect of a 500x120 plate. Within 15% of it means
        # the plate is close to square-on and its width is its true width.
        square = sum(1 for a in asp if 3.5 <= a <= 4.9)
        verdict = ("usable" if square >= 40 else
                   "marginal" if square >= 15 else "too few square-on plates")
        print(f"{cam_dir.name:<9}{len(used):>7}{len(merged):>9}"
              f"{(ws[len(ws)//2] if ws else 0):>8.0f}"
              f"{(asp[len(asp)//2] if asp else 0):>12.2f}{square:>11}  {verdict}")

        (OUT / f"{cam_dir.name}.json").write_text(json.dumps(
            {"camera": cam_dir.name, "clips": used, "frame_size": frame_size,
             "observations": merged}), encoding="utf-8")

    print(f"\nwrote {OUT}")
    print("Nothing is fitted here. `fit_height_from_plates` consumes this.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
