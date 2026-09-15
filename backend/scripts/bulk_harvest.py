"""backend/scripts/bulk_harvest.py — scale the detection dataset from
hundreds of frames to thousands, WITHOUT collecting near-duplicates.

THE MISTAKE THIS AVOIDS
  "More frames" is not the goal - more INFORMATION is. These are fixed
  cameras: the background never moves, so two frames half a second apart are
  ~95% identical and teach the model almost nothing new. A dataset of 10,000
  frames grabbed from one 40-second window represents 40 seconds of reality,
  and training on it mostly teaches "this exact traffic arrangement".

MEASURED ON THESE FEEDS (stream/2, 30fps, 12h recording)
  read rate            123.7 fps = 4.13x realtime
                       -> a full 12h recording traverses in ~2.9 hours
  frame difference     0.5s gap -> 10.4     2s -> 28.0
  vs temporal gap      3s   gap -> 33.6    10s -> 37.4
                       -> difference saturates around 3s; closer spacing is
                          mostly redundancy, wider spacing buys little more

  So: sample ~3 seconds of FOOTAGE apart (not wall-clock apart), and reject
  anything that still looks like the previous keeper.

TWO-LAYER DEDUPLICATION
  1. Temporal: skip N native frames between candidates (cheap, coarse).
  2. Perceptual: compare a downscaled grayscale of each candidate against the
     last frame KEPT for that camera; reject below --min-diff. This catches
     the case temporal spacing misses entirely - a stopped scene at 3am where
     nothing moves for minutes and every sample is the same empty road.

USAGE
  python -m backend.scripts.bulk_harvest --target-per-camera 300
  python -m backend.scripts.bulk_harvest --cameras CAM_01,CAM_02 --workers 4
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import threading
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

os.environ.setdefault(
    "OPENCV_FFMPEG_CAPTURE_OPTIONS",
    "timeout;20000000|stimeout;20000000|rw_timeout;20000000",
)

import cv2
import numpy as np
import yaml

OUT_ROOT = Path("data/detection_train/images")
MANIFEST = Path("data/detection_train/bulk_manifest.json")
_lock = threading.Lock()


def thumb(frame) -> np.ndarray:
    """Small grayscale signature for cheap perceptual comparison."""
    return cv2.cvtColor(cv2.resize(frame, (160, 90)), cv2.COLOR_BGR2GRAY).astype(np.int16)


def brightness_label(frame) -> tuple[str, float]:
    mean = float(cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY).mean())
    if mean < 60:
        return "dark", mean
    if mean > 180:
        return "bright", mean
    return "mid", mean


def harvest_camera(cam: dict, args) -> dict:
    cam_id = cam["id"]
    out_dir = OUT_ROOT / cam_id
    out_dir.mkdir(parents=True, exist_ok=True)

    rec = {"camera": cam_id, "district": cam.get("district"),
           "saved": 0, "rejected_similar": 0, "read": 0,
           "brightness": {"dark": 0, "mid": 0, "bright": 0}, "status": "ok"}

    cap = cv2.VideoCapture(cam["url"])
    if not cap.isOpened():
        cap.release()
        rec["status"] = "could not open"
        return rec

    native = cap.get(cv2.CAP_PROP_FPS) or 30.0
    # Convert "seconds of footage" into "native frames to skip".
    skip = max(1, int(round(native * args.gap_seconds)))

    last_keep: np.ndarray | None = None
    t0 = time.perf_counter()
    stamp = int(time.time())

    try:
        while (rec["saved"] < args.target_per_camera
               and (time.perf_counter() - t0) < args.timeout):
            ok, frame = cap.read()
            if not ok or frame is None:
                continue
            rec["read"] += 1
            if rec["read"] % skip:
                continue

            t = thumb(frame)
            if last_keep is not None:
                diff = float(np.mean(np.abs(t - last_keep)))
                if diff < args.min_diff:
                    # Scene has not changed enough to be worth a training
                    # sample - e.g. an empty road at night.
                    rec["rejected_similar"] += 1
                    continue

            label, mean = brightness_label(frame)
            name = f"{cam_id}_bulk{stamp}_{rec['saved']:04d}_{label}.jpg"
            cv2.imwrite(str(out_dir / name), frame)
            rec["brightness"][label] += 1
            rec["saved"] += 1
            last_keep = t
    except Exception as exc:
        rec["status"] = f"EXCEPTION {type(exc).__name__}"
    finally:
        cap.release()

    rec["elapsed_s"] = round(time.perf_counter() - t0, 1)
    with _lock:
        print(f"  {cam_id:<9} saved={rec['saved']:<4} "
              f"dup_rejected={rec['rejected_similar']:<4} "
              f"{rec['elapsed_s']:>5.0f}s  {rec['status']}", flush=True)
    return rec


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--target-per-camera", type=int, default=300)
    ap.add_argument("--gap-seconds", type=float, default=3.0,
                    help="Seconds of FOOTAGE between candidates. Measured "
                         "sweet spot is ~3s; below that frames are largely "
                         "redundant on a fixed camera.")
    ap.add_argument("--min-diff", type=float, default=8.0,
                    help="Reject a candidate if its mean abs difference from "
                         "the last KEPT frame is below this. Catches static "
                         "scenes that temporal spacing alone cannot.")
    ap.add_argument("--timeout", type=float, default=420.0,
                    help="Max seconds per camera.")
    ap.add_argument("--workers", type=int, default=6,
                    help="Concurrent cameras. Kept modest: a 30-way "
                         "concurrent probe on these endpoints took 872s and "
                         "produced corrupt h264.")
    ap.add_argument("--cameras", default="")
    args = ap.parse_args()

    cfg = yaml.safe_load(open("config.yaml", encoding="utf-8"))
    only = {c.strip() for c in args.cameras.split(",") if c.strip()} or None
    cams = [c for c in cfg.get("demo_cameras", [])
            if c.get("enabled") is not False and c.get("url")
            and (not only or c["id"] in only)]
    if not cams:
        print("No cameras matched.", file=sys.stderr)
        sys.exit(1)

    print(f"Harvesting up to {args.target_per_camera} frame(s) from "
          f"{len(cams)} camera(s), {args.workers} at a time.")
    print(f"  spacing {args.gap_seconds}s of footage | "
          f"perceptual min-diff {args.min_diff}\n")

    with ThreadPoolExecutor(max_workers=args.workers) as ex:
        report = list(ex.map(lambda c: harvest_camera(c, args), cams))

    saved = sum(r["saved"] for r in report)
    dups = sum(r["rejected_similar"] for r in report)
    light = {"dark": 0, "mid": 0, "bright": 0}
    for r in report:
        for k in light:
            light[k] += r["brightness"][k]

    MANIFEST.parent.mkdir(parents=True, exist_ok=True)
    MANIFEST.write_text(json.dumps(report, indent=2))

    total_on_disk = sum(1 for _ in OUT_ROOT.rglob("*.jpg"))
    print()
    print(f"saved this run       : {saved}")
    print(f"near-duplicates cut  : {dups}")
    print(f"cameras producing    : {sum(1 for r in report if r['saved'])}/{len(cams)}")
    print(f"brightness           : {light}")
    print(f"dataset total on disk: {total_on_disk}")
    if light["dark"] == 0:
        print()
        print("No dark frames captured. These recordings are 12h long and "
              "likely contain night footage, but reading always starts at "
              "the beginning of the file - reaching later material needs "
              "seeking (slow) or a much longer --timeout.")


if __name__ == "__main__":
    main()
