"""What does sampling at 1 fps cost the tracker?

The throughput measurement settled how many frames per second 27 cameras can
share on one GPU: about 38 in aggregate, so 1 fps each with headroom. That is
a real constraint, but quoting it alone would hide the consequence. Tracking
associates detections between consecutive samples, and the further a vehicle
moves between them the harder that is — at 1 fps a car at 40 km/h travels 11
metres and may leave the frame entirely between samples.

So: run the same footage at several sampling rates and compare what survives.
The measure is fragmentation. A vehicle that crosses the view should be one
track; if it becomes three, the journey it belongs to is broken into three
pieces and the count of "tracks" goes up while the information goes down.

Run:  python -m backend.scripts.bench_tracking_rate
"""
from __future__ import annotations

import sys
from collections import Counter
from pathlib import Path

import cv2
import numpy as np

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from ultralytics import YOLO                                     # noqa: E402
from backend.services.live_24x7_pipeline import (                # noqa: E402
    DETECT_INPUT_SIZE, YOLO_CONF_THRESHOLD, YOLO_IMGSZ,
    YOLO_VEHICLE_CLASSES,
)
from backend.services.reid_embedder import get_embedder          # noqa: E402
from backend.services.tracker import BoTSORTTracker              # noqa: E402

# Busy junction and open road: fragmentation depends on how fast things cross.
CAMERAS = ["CAM_08", "CAM_04", "CAM_01", "CAM_21"]
WINDOW_SECONDS = 40.0


def run(frames: list, fps: int, model, emb) -> dict:
    tracker = BoTSORTTracker(frame_rate=fps)
    lengths: Counter = Counter()
    n_det = 0

    for frame in frames:
        scaled = cv2.resize(frame, DETECT_INPUT_SIZE)
        sx = frame.shape[1] / DETECT_INPUT_SIZE[0]
        sy = frame.shape[0] / DETECT_INPUT_SIZE[1]
        res = model.predict(scaled, verbose=False, conf=YOLO_CONF_THRESHOLD,
                            classes=YOLO_VEHICLE_CLASSES, device="cuda:0",
                            imgsz=YOLO_IMGSZ)
        dets, crops = [], []
        b = res[0].boxes
        if b is not None and len(b):
            h, w = frame.shape[:2]
            for xy, cf, cl in zip(b.xyxy.cpu().numpy(), b.conf.cpu().numpy(),
                                  b.cls.cpu().numpy()):
                x1, y1 = float(xy[0]) * sx, float(xy[1]) * sy
                x2, y2 = float(xy[2]) * sx, float(xy[3]) * sy
                dets.append({"bbox": [x1, y1, x2, y2], "cls": int(cl),
                             "conf": float(cf)})
                ix1, iy1 = max(0, int(x1)), max(0, int(y1))
                ix2, iy2 = min(w, int(x2)), min(h, int(y2))
                crops.append(frame[iy1:iy2, ix1:ix2] if ix2 > ix1 and iy2 > iy1
                             else np.zeros((8, 8, 3), np.uint8))
        n_det += len(dets)
        feats = emb.extract_batch(crops) if crops else None
        for t in tracker.update(dets, frame, feats):
            lengths[t["track_id"]] += 1

    if not lengths:
        return {"tracks": 0, "detections": n_det, "mean_len": 0.0,
                "singletons": 0, "usable": 0}
    vals = list(lengths.values())
    return {
        "tracks": len(lengths),
        "detections": n_det,
        "mean_len": float(np.mean(vals)),
        # A track seen once is a detection that was never associated with
        # anything; it cannot establish direction, speed or a journey leg.
        "singletons": sum(1 for v in vals if v == 1),
        # Three sightings is the minimum for a direction and a speed.
        "usable": sum(1 for v in vals if v >= 3),
    }


def main() -> int:
    model = YOLO("yolov8s.pt")
    emb = get_embedder()

    print(f"{'camera':<9}{'fps':>4}{'frames':>8}{'dets':>7}{'tracks':>8}"
          f"{'mean len':>10}{'1-frame':>9}{'>=3 frames':>12}")
    print("-" * 67)

    totals: dict[int, Counter] = {}
    for cam in CAMERAS:
        clips = sorted((ROOT / "data/clips" / cam).glob("*_0730.mp4")) or \
            sorted((ROOT / "data/clips" / cam).glob("*.mp4"))
        if not clips:
            continue
        cap = cv2.VideoCapture(str(clips[0]))
        native = cap.get(cv2.CAP_PROP_FPS) or 25.0
        raw = []
        need = int(WINDOW_SECONDS * native)
        while len(raw) < need:
            ok, f = cap.read()
            if not ok:
                break
            raw.append(f)
        cap.release()
        if len(raw) < native * 5:
            continue

        for fps in (1, 2, 5):
            stride = max(1, int(round(native / fps)))
            frames = raw[::stride]
            r = run(frames, fps, model, emb)
            totals.setdefault(fps, Counter()).update(r)
            print(f"{cam:<9}{fps:>4}{len(frames):>8}{r['detections']:>7}"
                  f"{r['tracks']:>8}{r['mean_len']:>10.1f}"
                  f"{r['singletons']:>9}{r['usable']:>12}")
        print()

    print("=" * 67)
    print(f"{'':<9}{'fps':>4}{'':>8}{'dets':>7}{'tracks':>8}"
          f"{'':>10}{'1-frame':>9}{'>=3 frames':>12}")
    for fps in sorted(totals):
        t = totals[fps]
        frag = t["singletons"] / max(1, t["tracks"])
        print(f"{'all':<9}{fps:>4}{'':>8}{t['detections']:>7}{t['tracks']:>8}"
              f"{'':>10}{t['singletons']:>9}{t['usable']:>12}"
              f"   {frag:.0%} never associated")

    print("""
Read the last two columns together. Tracks that appear in one frame only were
never matched to anything and carry no direction or speed; tracks of three or
more frames are the ones a journey can be built from. A sampling rate that
raises the first while lowering the second is producing more rows and less
information.""")
    return 0


if __name__ == "__main__":
    sys.exit(main())
