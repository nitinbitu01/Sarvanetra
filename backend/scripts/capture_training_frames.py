"""backend/scripts/capture_training_frames.py — build a detection dataset
from the live Gujarat cameras.

WHY FRAMES AND NOT THE PIPELINE
  The running pipeline saves CROPS of confirmed tracks. Detector training
  needs whole FRAMES with every object labelled, so this captures frames
  directly and independently of detection/tracking. It also means a camera
  that produces no confirmed tracks (most of them, most of the time) still
  contributes training data.

CAPTURE STRATEGY
  Cameras are opened one at a time, not all at once. Opening 30 concurrently
  was measured at 872s and produced corrupt h264 in places; sequential opens
  are slower per camera but far more reliable, and a camera that fails is
  skipped rather than stalling the rest.

  Frames are spaced by --every-nth reads so consecutive samples are not
  near-duplicates of the same instant, which would inflate the dataset
  without adding information.

USAGE
  python -m backend.scripts.capture_training_frames --per-camera 20
  python -m backend.scripts.capture_training_frames --cameras CAM_01,CAM_02
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import os

# Bound how long FFmpeg will sit on an unresponsive stream. Must be set
# BEFORE cv2 is imported - OpenCV reads this when it initialises its FFmpeg
# backend, not per-capture.
#
# cv2.VideoCapture(url) itself takes no timeout argument and will block
# indefinitely on a server that accepts the TCP connection then goes quiet.
# These corp8 endpoints do exactly that: a 30-way concurrent probe took
# 872s, and several cameras open but never deliver a frame. Without this,
# one bad camera stalls the entire sequential capture run.
os.environ.setdefault(
    "OPENCV_FFMPEG_CAPTURE_OPTIONS",
    "timeout;15000000|stimeout;15000000|rw_timeout;15000000",  # microseconds
)

import cv2
import yaml


def load_cameras(only: set[str] | None) -> list[dict]:
    cfg = yaml.safe_load(open("config.yaml", encoding="utf-8"))
    cams = []
    for c in cfg.get("demo_cameras", []):
        if c.get("enabled") is False:
            continue
        if only and c["id"] not in only:
            continue
        if not c.get("url"):
            continue
        cams.append(c)
    return cams


def capture_one(cam: dict, out_root: Path, want: int, every_nth: int,
                open_timeout: float) -> dict:
    cam_id = cam["id"]
    out_dir = out_root / cam_id
    out_dir.mkdir(parents=True, exist_ok=True)

    t_open = time.perf_counter()
    cap = cv2.VideoCapture(cam["url"])
    open_secs = time.perf_counter() - t_open
    if not cap.isOpened():
        cap.release()
        return {"camera": cam_id, "saved": 0, "status": "could not open",
                "open_secs": round(open_secs, 1)}

    saved = 0
    reads = 0
    t0 = time.perf_counter()
    # Bound the whole attempt: a stream that opens but dribbles frames must
    # not hold the run hostage - it just contributes fewer samples.
    while saved < want and (time.perf_counter() - t0) < open_timeout:
        ok, frame = cap.read()
        if not ok or frame is None:
            continue
        reads += 1
        if reads % every_nth:
            continue
        cv2.imwrite(str(out_dir / f"{cam_id}_{saved:04d}.jpg"), frame)
        saved += 1
    cap.release()

    return {
        "camera": cam_id,
        "saved": saved,
        "status": "ok" if saved else "opened but delivered no frames",
        "open_secs": round(open_secs, 1),
        "district": cam.get("district"),
        "resolution": f"{frame.shape[1]}x{frame.shape[0]}" if saved else None,
    }


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--per-camera", type=int, default=20)
    ap.add_argument("--every-nth", type=int, default=8,
                    help="Keep 1 of every N frames read, so samples are "
                         "spread in time rather than near-duplicates.")
    ap.add_argument("--timeout", type=float, default=45.0,
                    help="Max seconds to spend on one camera.")
    ap.add_argument("--cameras", type=str, default="",
                    help="Comma-separated camera ids; default is all enabled.")
    ap.add_argument("--out", type=Path, default=Path("data/detection_train/images"))
    args = ap.parse_args()

    only = {c.strip() for c in args.cameras.split(",") if c.strip()} or None
    cams = load_cameras(only)
    if not cams:
        print("No cameras matched.", file=sys.stderr)
        sys.exit(1)

    print(f"Capturing up to {args.per_camera} frame(s) from {len(cams)} camera(s), "
          f"sequentially.\n")
    args.out.mkdir(parents=True, exist_ok=True)

    report = []
    for i, cam in enumerate(cams, 1):
        # One camera must never end the run. These streams fail in varied
        # and surprising ways (open-but-silent, mid-stream codec errors,
        # abrupt resets); an unhandled exception on camera 15 previously
        # discarded the frames already captured from the first 14 and never
        # wrote the report.
        try:
            r = capture_one(cam, args.out, args.per_camera, args.every_nth,
                            args.timeout)
        except Exception as exc:
            r = {"camera": cam["id"], "saved": 0, "open_secs": 0.0,
                 "district": cam.get("district"), "resolution": None,
                 "status": f"EXCEPTION {type(exc).__name__}: {exc}"[:120]}
        report.append(r)
        print(f"  [{i:>2}/{len(cams)}] {r['camera']:<8} "
              f"saved={r['saved']:<3} open={r['open_secs']:>5}s  {r['status']}",
              flush=True)   # flush: piped stdout otherwise buffers the whole run

    total = sum(r["saved"] for r in report)
    live = sum(1 for r in report if r["saved"] > 0)
    (args.out.parent / "capture_report.json").write_text(json.dumps(report, indent=2))

    print()
    print(f"TOTAL frames      : {total}")
    print(f"cameras producing : {live}/{len(cams)}")
    print(f"images            : {args.out}")
    print(f"report            : {args.out.parent / 'capture_report.json'}")
    if total == 0:
        sys.exit(1)


if __name__ == "__main__":
    main()
