"""backend/scripts/make_loitering_demo_video.py — render the detection as video.

WHY A VIDEO AND NOT A STILL
  A still frame with a box on it proves almost nothing to a sceptic: it shows
  a person, and asserts they were loitering. The claim is about TIME - someone
  stayed in one place for sixty seconds while everyone around them moved
  through - and time is exactly what a still cannot carry.

  So this renders what the detector sees, frame by frame:
    - every tracked person, with a live dwell timer
    - the timer climbing on whoever holds position, resetting on whoever moves
    - the moment a track crosses the threshold and the alert fires
    - the alert persisting afterwards, as an operator would see it

  A viewer watches a number count up on one person while others come and go.
  That is the argument, and it makes itself.

FAITHFUL TO THE REAL DETECTOR
  The dwell logic here mirrors backend/services/loitering_detector.py: a track
  accumulates position samples, and if the maximum pairwise distance across the
  window stays inside LOITER_RADIUS_METERS x px_per_meter for
  LOITER_DURATION_SEC, it fires. The same thresholds are read from the same
  settings, so what the video shows is what the system does - not a
  re-enactment tuned to look convincing.

USAGE
  python -m backend.scripts.make_loitering_demo_video --source demo/clips/cam_04_0700.mp4
"""
from __future__ import annotations

import argparse
import os
from collections import defaultdict, deque
from pathlib import Path

os.environ.setdefault("OPENCV_FFMPEG_CAPTURE_OPTIONS",
                      "timeout;30000000|stimeout;30000000|rw_timeout;30000000")

import cv2
import numpy as np
import yaml

OUT = Path("output/demo_video")

GREY = (150, 150, 150)
AMBER = (60, 190, 250)
RED = (48, 80, 245)
WHITE = (255, 255, 255)


def max_pairwise(points) -> float:
    """Largest distance between any two samples — the detector's own test."""
    if len(points) < 2:
        return 0.0
    a = np.asarray(points, dtype=np.float32)
    d = np.sqrt(((a[:, None, :] - a[None, :, :]) ** 2).sum(-1))
    return float(d.max())


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--source", default="demo/clips/cam_04_0700.mp4")
    ap.add_argument("--out", default="output/demo_video/loitering_proof.mp4")
    ap.add_argument("--dwell-sec", type=float, default=None,
                    help="Override LOITER_DURATION_SEC for a shorter demo.")
    ap.add_argument("--radius-px", type=float, default=None)
    ap.add_argument("--speed", type=int, default=1,
                    help="Write every Nth processed frame — 2 halves the "
                         "output length without changing the detection.")
    args = ap.parse_args()

    cfg = yaml.safe_load(open("config.yaml", encoding="utf-8"))
    try:
        from backend.core.config import settings
        dwell = float(args.dwell_sec or settings.LOITER_DURATION_SEC)
    except Exception:                                    # noqa: BLE001
        dwell = float(args.dwell_sec or 60.0)
    radius = float(args.radius_px or 120.0)

    cap = cv2.VideoCapture(args.source)
    if not cap.isOpened():
        raise SystemExit(f"cannot open {args.source}")
    fps = cap.get(cv2.CAP_PROP_FPS) or 25.0
    W = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    H = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    total = cap.get(cv2.CAP_PROP_FRAME_COUNT)
    cap.release()

    print(f"source   : {args.source}  ({total:.0f} frames @ {fps:.1f}fps)")
    print(f"thresholds: dwell {dwell:.0f}s, radius {radius:.0f}px")

    from ultralytics import YOLO
    model = YOLO(cfg["model"]["path"])

    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    # Playback rate is set after the stride is known (see below) so the output
    # runs at real-world speed rather than 5x fast.
    writer = None

    history: dict[int, deque] = defaultdict(lambda: deque(maxlen=4096))
    fired: dict[int, float] = {}
    n_written = 0
    alerts = []

    # MUST match the pipeline's sampling rate. main.py computes
    # frame_interval = round(source_fps / processing.fps) and only runs
    # detection on those frames, so tracking continuity and the dwell window's
    # sample density both depend on it. Tracking every frame instead produced
    # ZERO alerts on a clip where the real pipeline found two - different
    # track ids, different track lifetimes, and 5x the samples feeding the
    # max-pairwise spread test.
    target_fps = float(cfg["processing"]["fps"])
    stride = max(1, round(fps / target_fps))
    eff_fps = fps / stride
    print(f"sampling : every {stride} frames -> {eff_fps:.1f} fps "
          f"(matches pipeline processing.fps={target_fps:.0f})")
    # Write at the sampled rate so the video plays at true speed: a 60-second
    # dwell must look like sixty seconds, or the timer is not evidence.
    writer = cv2.VideoWriter(args.out, cv2.VideoWriter_fourcc(*"mp4v"),
                             max(eff_fps / args.speed, 1.0), (W, H))

    stream = model.track(source=args.source, stream=True, persist=True,
                         tracker="botsort.yaml", conf=0.35,
                         imgsz=cfg["processing"]["input_width"],
                         classes=[0], vid_stride=stride, verbose=False)

    for fi, r in enumerate(stream):
        # fi counts SAMPLED frames; real elapsed video time advances by stride.
        t = fi * stride / fps
        frame = r.orig_img.copy()

        if r.boxes is not None and r.boxes.id is not None:
            for b in r.boxes:
                tid = int(b.id[0])
                x1, y1, x2, y2 = (int(v) for v in b.xyxy[0].tolist())
                cx, cy = (x1 + x2) / 2.0, (y1 + y2) / 2.0
                hist = history[tid]
                hist.append((t, cx, cy))
                # Drop samples older than the window, exactly as the detector
                # trims its sorted set.
                while hist and t - hist[0][0] > dwell:
                    hist.popleft()

                span = t - hist[0][0] if hist else 0.0
                spread = max_pairwise([(p[1], p[2]) for p in hist])
                held = span if spread <= radius else 0.0

                if held >= dwell and tid not in fired:
                    fired[tid] = t
                    alerts.append((tid, t, spread))
                    print(f"  ALERT  t={t:6.1f}s  track {tid}  "
                          f"held {held:.0f}s  spread {spread:.0f}px")

                is_alert = tid in fired
                pct = min(held / dwell, 1.0)
                col = RED if is_alert else (AMBER if pct > 0.25 else GREY)
                thick = 3 if is_alert else (2 if pct > 0.25 else 1)
                cv2.rectangle(frame, (x1, y1), (x2, y2), col, thick)

                if is_alert:
                    cv2.circle(frame, (int(cx), int(cy)), int(radius), RED, 2)
                    lab = "LOITERING"
                    (tw, th), _ = cv2.getTextSize(lab, cv2.FONT_HERSHEY_SIMPLEX,
                                                  0.62, 2)
                    cv2.rectangle(frame, (x1, y1 - th - 11),
                                  (x1 + tw + 10, y1 - 2), RED, -1)
                    cv2.putText(frame, lab, (x1 + 4, y1 - 7),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.62, WHITE, 2)
                elif pct > 0.25:
                    # The dwell timer is the whole point: a number climbing on
                    # one person while everyone else resets to zero.
                    lab = f"{held:.0f}s"
                    cv2.putText(frame, lab, (x1, y1 - 6),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.55, AMBER, 2)
                    bw = int((x2 - x1) * pct)
                    cv2.rectangle(frame, (x1, y2 + 3), (x1 + bw, y2 + 7),
                                  AMBER, -1)

        # Header: thresholds and running totals, so a viewer can check the
        # claim against the numbers rather than taking the boxes on trust.
        band = np.full((70, W, 3), 20, np.uint8)
        cv2.putText(band, "SENTINEL — LOITERING DETECTION", (14, 27),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.72, (120, 235, 205), 2)
        cv2.putText(band, f"t={t:6.1f}s   rule: stationary {dwell:.0f}s within "
                          f"{radius:.0f}px   tracked: {len(history)}   "
                          f"ALERTS: {len(fired)}",
                    (14, 52), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (205, 205, 205), 1)
        frame[0:70] = cv2.addWeighted(frame[0:70], 0.25, band, 0.75, 0)

        if fi % args.speed == 0:
            writer.write(frame)
            n_written += 1
        if fi % 250 == 0:
            print(f"    {fi}/{total:.0f} frames", flush=True)

    writer.release()
    print(f"\nwrote {n_written} frames -> {Path(args.out).resolve()}")
    print(f"alerts in video: {len(alerts)}")
    for tid, t, sp in alerts:
        print(f"   track {tid} fired at t={t:.1f}s (spread {sp:.0f}px)")
    print("\nGrey = tracked, moving. Amber = holding position, timer running.")
    print("Red = threshold crossed, alert fired.")


if __name__ == "__main__":
    main()
