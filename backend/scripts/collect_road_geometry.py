"""Collect the observations a camera calibration can be estimated from.

There is no survey team and no ground control points for this fleet, so a
homography has to come from what the footage itself shows. Two things in it
carry geometry:

  vehicle trajectories    every vehicle travels along the road, so the lines
                          their image tracks sweep out all meet at the road's
                          vanishing point. That fixes the direction of travel
                          in the image and, with it, the horizon.

  vehicle widths          an auto-rickshaw is 1.3-1.4 m across and a bus is
                          capped at 2.6 m by the Central Motor Vehicles Rules.
                          A known real width observed at a known image height
                          fixes the metric scale that the vanishing point
                          alone leaves undetermined.

This script only gathers and reports; it fits nothing. The point is to see
whether the observations are good enough before building on them — a camera
with 6 usable tracks, or one pointed at an indoor waiting hall, cannot be
calibrated from its own footage, and it is better to know which those are than
to produce a matrix for all 27.

Run:  python -m backend.scripts.collect_road_geometry [camera ...]
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
    DETECT_INPUT_SIZE, YOLO_CONF_THRESHOLD, YOLO_IMGSZ,
)
from backend.services.tracker import BoTSORTTracker              # noqa: E402

OUT = ROOT / "output" / "road_geometry"

# Sampled at 5 fps for this purpose rather than the pipeline's 1: calibration
# is computed once offline, and denser tracks give straighter lines.
SAMPLE_FPS = 5
SECONDS = 120
MIN_TRACK_POINTS = 6

# Vehicle classes with a width that regulation pins down, and the width in
# metres. Motorcycles and bicycles are excluded: their apparent width depends
# on rider posture and lean, which is exactly the variance a scale estimate
# cannot absorb.
CLASS_WIDTH_M = {
    2: ("car", 1.75),        # Indian compact/sedan mean track width
    5: ("bus", 2.60),        # CMVR maximum, and city buses are built to it
    7: ("truck", 2.45),      # rigid goods vehicle body width
}
DETECT_CLASSES = sorted(CLASS_WIDTH_M)


def collect(cam: str, model, clip: Path) -> dict:
    cap = cv2.VideoCapture(str(clip))
    native = cap.get(cv2.CAP_PROP_FPS) or 25.0
    stride = max(1, int(round(native / SAMPLE_FPS)))
    want = int(SECONDS * native)

    tracker = BoTSORTTracker(frame_rate=SAMPLE_FPS)
    tracks: dict[int, list] = defaultdict(list)
    frame_size = None
    i = 0
    while i < want:
        ok = cap.grab()
        if not ok:
            break
        i += 1
        if i % stride:
            continue
        ok, frame = cap.retrieve()
        if not ok or frame is None:
            break
        if frame_size is None:
            frame_size = (frame.shape[1], frame.shape[0])

        scaled = cv2.resize(frame, DETECT_INPUT_SIZE)
        sx = frame.shape[1] / DETECT_INPUT_SIZE[0]
        sy = frame.shape[0] / DETECT_INPUT_SIZE[1]
        res = model.predict(scaled, verbose=False, conf=YOLO_CONF_THRESHOLD,
                            classes=DETECT_CLASSES, device="cuda:0",
                            imgsz=YOLO_IMGSZ)
        dets = []
        b = res[0].boxes
        if b is not None and len(b):
            for xy, cf, cl in zip(b.xyxy.cpu().numpy(), b.conf.cpu().numpy(),
                                  b.cls.cpu().numpy()):
                dets.append({"bbox": [float(xy[0]) * sx, float(xy[1]) * sy,
                                      float(xy[2]) * sx, float(xy[3]) * sy],
                             "cls": int(cl), "conf": float(cf)})
        for t in tracker.update(dets, frame, None):
            x1, y1, x2, y2 = t["bbox"]
            tracks[t["track_id"]].append({
                # Ground contact point: the bottom-centre of the box is where
                # the vehicle meets the road plane, which is the only point on
                # it a ground-plane homography can map.
                "u": (x1 + x2) / 2.0, "v": y2,
                "w": x2 - x1, "h": y2 - y1,
                "cls": t["cls"], "frame": i,
            })
    cap.release()

    long_tracks = {k: v for k, v in tracks.items()
                   if len(v) >= MIN_TRACK_POINTS}
    return {"camera": cam, "clip": clip.name, "frame_size": frame_size,
            "tracks": {str(k): v for k, v in long_tracks.items()},
            "n_tracks_all": len(tracks)}


def straightness(points: list) -> float:
    """How close a track is to a straight line, as a fraction of its length.

    A vehicle on a road travels straight; a track that wanders is either a
    turning vehicle or an association error, and neither should contribute to
    a vanishing point.
    """
    p = np.array([[q["u"], q["v"]] for q in points], dtype=float)
    d = p - p.mean(0)
    if len(p) < 3:
        return 1.0
    _, s, _ = np.linalg.svd(d, full_matrices=False)
    span = float(np.hypot(*(p[-1] - p[0])))
    return float(s[1] / max(span, 1e-6))


def main() -> int:
    cams = sys.argv[1:] or None
    model = YOLO("yolov8s.pt")
    OUT.mkdir(parents=True, exist_ok=True)

    print(f"{'camera':<9}{'tracks':>8}{'usable':>8}{'straight':>10}"
          f"{'pts/track':>11}{'width obs':>11}  verdict")
    print("-" * 74)

    summary = []
    for cam_dir in sorted((ROOT / "data" / "clips").iterdir()):
        if not cam_dir.is_dir() or not cam_dir.name.startswith("CAM_"):
            continue
        if cams and cam_dir.name not in cams:
            continue
        clips = sorted(cam_dir.glob("*.mp4"))
        if not clips:
            continue

        # EVERY clip, not just one.
        #
        # This used to take clips[0] — a single 120-second window per camera —
        # and that single choice is what starved everything downstream.
        # Measured on the resulting files: CAM_08 produced 94 tracks of which
        # only 8 were straight enough to contribute a line, and 69 points
        # reached the width estimator against a 20-observation floor. Seven
        # cameras therefore ended at "insufficient_width_observations" and none
        # reached a metric ground plane.
        #
        # There are 117 clips on disk and 27 had ever been read. Pooling a
        # camera's clips multiplies its usable tracks without loosening a
        # single quality gate, which is the difference between more evidence
        # and a lower standard of evidence.
        #
        # Tracker ids restart per clip, so keys are scoped by clip name;
        # without that, two different vehicles in two clips merge into one
        # track and the line through them is meaningless.
        merged: dict = {}
        frame_size = None
        n_all = 0
        used = []
        for clip in clips:
            part = collect(cam_dir.name, model, clip)
            if not part.get("frame_size"):
                continue
            # A camera whose clips differ in resolution cannot pool them: the
            # vanishing point and every pixel width would be in mixed units.
            if frame_size is None:
                frame_size = part["frame_size"]
            elif part["frame_size"] != frame_size:
                print(f"  [!] {cam_dir.name}: {clip.name} is "
                      f"{part['frame_size']} not {frame_size} — skipped")
                continue
            stem = clip.stem
            for k, v in part["tracks"].items():
                merged[f"{stem}#{k}"] = v
            n_all += part.get("n_tracks_all", 0)
            used.append(clip.name)

        if not merged:
            continue
        data = {"camera": cam_dir.name, "clips": used, "clip": used[0],
                "frame_size": frame_size, "tracks": merged,
                "n_tracks_all": n_all}
        tr = data["tracks"]
        straight = {k: v for k, v in tr.items() if straightness(v) < 0.03}
        widths = sum(1 for v in straight.values() for q in v
                     if q["cls"] in CLASS_WIDTH_M)
        pts = (np.mean([len(v) for v in straight.values()])
               if straight else 0.0)

        # A vanishing point needs several straight tracks crossing at a clear
        # angle; scale needs a decent count of known-width observations.
        if len(straight) >= 8 and widths >= 40:
            verdict = "calibratable"
        elif len(straight) >= 4:
            verdict = "marginal"
        else:
            verdict = "too few tracks"

        print(f"{cam_dir.name:<9}{data['n_tracks_all']:>8}{len(tr):>8}"
              f"{len(straight):>10}{pts:>11.1f}{widths:>11}  {verdict}")

        (OUT / f"{cam_dir.name}.json").write_text(
            json.dumps(data, indent=1), encoding="utf-8")
        summary.append({"camera": cam_dir.name, "clip": data["clip"],
                        "frame_size": data["frame_size"],
                        "tracks_all": data["n_tracks_all"],
                        "tracks_usable": len(tr),
                        "tracks_straight": len(straight),
                        "width_observations": widths,
                        "verdict": verdict})

    (OUT / "summary.json").write_text(json.dumps(summary, indent=1),
                                      encoding="utf-8")
    ok = sum(1 for s in summary if s["verdict"] == "calibratable")
    print(f"\n{ok}/{len(summary)} cameras have enough road geometry to attempt "
          f"a calibration")
    print(f"wrote {OUT}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
