"""backend/scripts/collect_vehicle_edges.py — the observations a SECOND
vanishing point can be estimated from, without a second traffic stream.

WHY THIS EXISTS
  fit_camera_calibration derives focal length from two orthogonal vanishing
  points, and gets the second one from a second stream of traffic travelling
  in a different direction. Measured on this fleet after pooling every clip:

      CAM_02 [95]        CAM_04 [89]        CAM_21 [50]      one stream only
      CAM_08 [25, 2, 1]                                      second too small
      CAM_05 [83, 54, 1] CAM_01 [32, 6, 2, 2]                two streams, but
                                                             oblique view kills
                                                             the height step

  Six cameras reached a plausible pole height and none reached a metric ground
  plane. Most of this network watches single-direction roads, so no amount of
  tuning produces a second stream that is not there.

THE STANDARD ANSWER
  Dubská, Herout & Sochor, "Fully Automatic Roadside Camera Calibration for
  Traffic Surveillance" (IEEE T-ITS 2014), take the second vanishing point from
  the vehicles themselves rather than from a second stream. A car is a box: its
  bumper line, roofline edge and window frames run ACROSS the direction of
  travel. Accumulated over many vehicles those edges meet at the vanishing
  point of the transverse direction — which is the orthogonal partner VP1 needs.

  This script only gathers the edge segments and reports whether they look
  usable. It fits nothing, for the same reason collect_road_geometry fits
  nothing: it is better to find out that a camera's vehicles are 40 px wide and
  carry no measurable edges than to produce a focal length for all 22.

WHAT IS COLLECTED
  Per camera, line segments found inside vehicle boxes, in full-frame pixel
  coordinates, with the box that produced them. Segments lying along the
  direction of travel are NOT filtered here — that needs VP1, which the fitter
  owns. Filtering is the fitter's job; collecting is this script's.

Run:  python -m backend.scripts.collect_vehicle_edges [camera ...]
"""
from __future__ import annotations

import json
import sys
from collections import defaultdict
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

OUT = ROOT / "output" / "vehicle_edges"

# Matches collect_road_geometry so the two sets describe the same footage.
SAMPLE_FPS = 5
SECONDS = 120
VEHICLE_CLASSES = (2, 5, 7)

# A vehicle smaller than this carries no edge long enough to fix a direction.
# 90 px across is roughly where a car's roofline survives Canny on this fleet.
MIN_BOX_W = 90
# Segment length as a fraction of the box width. Short segments are texture —
# door handles, shadows, number plates — and they vote for nothing in
# particular while outnumbering the real structural lines.
MIN_SEG_FRAC = 0.28
# Cap per box so one lorry covered in edges cannot outvote fifty cars.
MAX_SEG_PER_BOX = 6


def edges_in_box(gray: np.ndarray, x1: int, y1: int, x2: int, y2: int) -> list:
    """Line segments inside one vehicle box, in full-frame coordinates."""
    w, h = x2 - x1, y2 - y1
    if w < MIN_BOX_W or h < 24:
        return []
    roi = gray[max(0, y1):y2, max(0, x1):x2]
    if roi.size == 0:
        return []

    # Contrast-normalise the crop: a dark car at dusk and a white van at noon
    # otherwise need different Canny thresholds, and a fixed pair suits neither.
    roi = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(4, 4)).apply(roi)
    med = float(np.median(roi))
    lo = int(max(10, 0.66 * med))
    hi = int(min(255, 1.33 * med))
    ed = cv2.Canny(roi, lo, hi, L2gradient=True)

    min_len = max(12.0, MIN_SEG_FRAC * w)
    segs = cv2.HoughLinesP(ed, 1, np.pi / 180.0, threshold=int(min_len * 0.6),
                           minLineLength=int(min_len), maxLineGap=4)
    if segs is None:
        return []

    # HoughLinesP returns (N, 1, 4) on most builds and (N, 4) on some.
    # Reshaping covers both rather than assuming one.
    segs = np.asarray(segs).reshape(-1, 4)

    out = []
    for s in segs:
        sx1, sy1, sx2, sy2 = (float(s[0]) + x1, float(s[1]) + y1,
                              float(s[2]) + x1, float(s[3]) + y1)
        out.append({"p": [round(sx1, 1), round(sy1, 1),
                          round(sx2, 1), round(sy2, 1)],
                    "len": round(float(np.hypot(sx2 - sx1, sy2 - sy1)), 1),
                    "box_w": round(float(w), 1)})
    out.sort(key=lambda r: -r["len"])
    return out[:MAX_SEG_PER_BOX]


def collect(cam: str, model, clip: Path) -> dict:
    cap = cv2.VideoCapture(str(clip))
    native = cap.get(cv2.CAP_PROP_FPS) or 25.0
    stride = max(1, int(round(native / SAMPLE_FPS)))
    want = int(SECONDS * native)

    segs, frame_size, boxes = [], None, 0
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
        res = model.predict(frame, imgsz=YOLO_IMGSZ, conf=YOLO_CONF_THRESHOLD,
                            classes=list(VEHICLE_CLASSES), verbose=False)
        b = res[0].boxes
        if b is None or not len(b):
            continue
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        for xy in b.xyxy.cpu().numpy():
            x1, y1, x2, y2 = (int(xy[0]), int(xy[1]), int(xy[2]), int(xy[3]))
            found = edges_in_box(gray, x1, y1, x2, y2)
            if found:
                boxes += 1
                segs.extend(found)
    cap.release()
    return {"camera": cam, "clip": clip.name, "frame_size": frame_size,
            "segments": segs, "boxes_with_edges": boxes}


def main() -> int:
    cams = sys.argv[1:] or None
    model = YOLO("yolov8s.pt")
    OUT.mkdir(parents=True, exist_ok=True)

    print(f"{'camera':<9}{'clips':>7}{'boxes':>8}{'segments':>10}"
          f"{'med len':>9}{'med box w':>11}  verdict")
    print("-" * 70)

    for cam_dir in sorted((ROOT / "data" / "clips").iterdir()):
        if not cam_dir.is_dir() or not cam_dir.name.startswith("CAM_"):
            continue
        if cams and cam_dir.name not in cams:
            continue
        clips = sorted(cam_dir.glob("*.mp4"))
        if not clips:
            continue

        merged, frame_size, boxes, used = [], None, 0, []
        for clip in clips:
            part = collect(cam_dir.name, model, clip)
            if not part.get("frame_size"):
                continue
            if frame_size is None:
                frame_size = part["frame_size"]
            elif part["frame_size"] != frame_size:
                continue
            merged.extend(part["segments"])
            boxes += part["boxes_with_edges"]
            used.append(clip.name)

        if frame_size is None:
            continue
        lens = sorted(s["len"] for s in merged)
        bws = sorted(s["box_w"] for s in merged)
        # VP2 from a few dozen segments is noise. The paper accumulates
        # thousands; 300 is the floor where a RANSAC intersection is worth
        # attempting at all.
        verdict = ("usable" if len(merged) >= 300 else
                   "marginal" if len(merged) >= 120 else "too few edges")
        print(f"{cam_dir.name:<9}{len(used):>7}{boxes:>8}{len(merged):>10}"
              f"{(lens[len(lens)//2] if lens else 0):>9.0f}"
              f"{(bws[len(bws)//2] if bws else 0):>11.0f}  {verdict}")

        (OUT / f"{cam_dir.name}.json").write_text(json.dumps(
            {"camera": cam_dir.name, "clips": used, "frame_size": frame_size,
             "boxes_with_edges": boxes, "segments": merged}), encoding="utf-8")

    print(f"\nwrote {OUT}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
