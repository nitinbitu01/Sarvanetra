"""Does frame enhancement help or hurt the plate detector?

The engine enhances every frame before anything else runs:

    enhanced_frame, lighting = self._enhance_frame(frame)
    preprocessed, raw_bgr, quality = self.extract_plate_candidate(
        enhanced_frame, bbox, cls_id, lighting=lighting)

so the plate detector searches the enhanced image, not the original. By day
that is mild CLAHE on the L channel; on a dark frame it is gamma 0.6, a
double pass of CLAHE, a bilateral filter and an unsharp mask.

The same mismatch was already measured one stage later and cost a great deal.
The 64x256 recogniser was trained on raw crops, and feeding it the night
enhancement chain took exact match from 42.2% to 18.1%. The detector was
trained on harvested frames, which were also raw, so it is worth asking
whether it is paying the same price — and this is the earlier stage, where a
miss means no read is attempted at all rather than a wrong one.

Measured per camera, since the answer should differ between a bright junction
and an IR night view, and the fleet has both.

Run:  python -m backend.scripts.plate_det_enhancement_eval
"""
from __future__ import annotations

import sys
from collections import defaultdict
from pathlib import Path

import cv2
import numpy as np

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
if str(ROOT / "backend") not in sys.path:
    sys.path.insert(0, str(ROOT / "backend"))

from ultralytics import YOLO                                       # noqa: E402
from scripts.night_enhancer_v2 import AdaptiveNightEnhancer        # noqa: E402

PLATE_MODEL = ROOT / "runs/detect/runs/plate/plate_v3_ft/weights/best.pt"
VEHICLE_MODEL = ROOT / "yolov8s.pt"
CLIPS = ROOT / "data" / "clips"

# Day junctions and the two IR night views, so the effect can be separated.
CAMERAS = ["CAM_08", "CAM_04", "CAM_06", "CAM_02", "CAM_24", "CAM_29"]
FRAMES_PER_CAM = 20
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
    enhancer = AdaptiveNightEnhancer()

    per_cam = defaultdict(lambda: {"crops": 0, "raw": 0, "enh": 0,
                                   "raw_w": [], "enh_w": [], "light": ""})

    for cam in CAMERAS:
        for fr in sample_frames(cam, FRAMES_PER_CAM):
            enhanced, lighting = enhancer.enhance(fr)
            per_cam[cam]["light"] = lighting

            # Vehicles are located once, on the raw frame, so both branches
            # search exactly the same regions and only the pixels differ.
            res = vdet.predict(fr, conf=VEHICLE_CONF, classes=VEHICLE_CLASSES,
                               verbose=False, device="cuda:0", imgsz=640)
            b = res[0].boxes
            if b is None or not len(b):
                continue
            for xyxy in b.xyxy.cpu().numpy():
                x1, y1, x2, y2 = (int(v) for v in xyxy)
                x1, y1 = max(0, x1), max(0, y1)
                x2, y2 = min(fr.shape[1], x2), min(fr.shape[0], y2)
                if x2 - x1 < 40 or y2 - y1 < 40:
                    continue
                per_cam[cam]["crops"] += 1

                for tag, src in (("raw", fr), ("enh", enhanced)):
                    crop = src[y1:y2, x1:x2]
                    r = pdet.predict(crop, imgsz=320, conf=PLATE_CONF,
                                     verbose=False, device="cuda:0")
                    bb = r[0].boxes
                    if bb is not None and len(bb):
                        per_cam[cam][tag] += 1
                        best = int(bb.conf.argmax())
                        xy = bb.xyxy[best].cpu().numpy()
                        per_cam[cam][f"{tag}_w"].append(float(xy[2] - xy[0]))

    print(f"{'camera':<9}{'lighting':<16}{'crops':>7}{'raw':>8}{'enhanced':>10}"
          f"{'delta':>9}")
    print("-" * 60)
    tot = {"crops": 0, "raw": 0, "enh": 0}
    for cam in CAMERAS:
        d = per_cam[cam]
        if not d["crops"]:
            continue
        raw_pct = 100 * d["raw"] / d["crops"]
        enh_pct = 100 * d["enh"] / d["crops"]
        tot["crops"] += d["crops"]
        tot["raw"] += d["raw"]
        tot["enh"] += d["enh"]
        print(f"{cam:<9}{d['light']:<16}{d['crops']:>7}{raw_pct:>7.1f}%"
              f"{enh_pct:>9.1f}%{enh_pct-raw_pct:>+8.1f}")

    if tot["crops"]:
        raw_pct = 100 * tot["raw"] / tot["crops"]
        enh_pct = 100 * tot["enh"] / tot["crops"]
        print("-" * 60)
        print(f"{'ALL':<9}{'':<16}{tot['crops']:>7}{raw_pct:>7.1f}%"
              f"{enh_pct:>9.1f}%{enh_pct-raw_pct:>+8.1f}")

        print(f"""
A negative delta means enhancement is losing plates the detector would
otherwise have found, and a plate missed here is never read at all — this is
the capture-rate stage, upstream of any recognition. A positive delta on the
dark cameras and a negative one by day would say the enhancement should be
applied conditionally rather than to every frame.""")
    return 0


if __name__ == "__main__":
    sys.exit(main())
