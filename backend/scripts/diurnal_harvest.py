"""backend/scripts/diurnal_harvest.py — collect training frames across
lighting conditions, not just whichever hour you happened to run capture.

!! READ THIS BEFORE TRUSTING THE TIME BINS !!
  The corp8 endpoints are NOT live streams. Measured directly:
      1,294,057 frames / 29.95 fps = exactly 12.00 hours
      HTTP 206 Partial Content, Accept-Ranges: bytes, Content-Type video/mp4
  They are 12-hour recorded MP4 files served over HTTP.

  So WALL-CLOCK BINNING DOES NOT WORK on this source: connecting at 03:00
  does not yield night footage, it yields whatever part of the recording the
  server hands you. A harvest run at 03:09 was binned "night" for all 231
  frames while the MEASURED brightness was 27 dark / 204 mid / 0 bright.

  Trust the `brightness` field, not the `bin` field, on this deployment.
  The bin is retained because it is correct for a genuinely live camera,
  and this script should still be right if these feeds are ever replaced
  with real ones.

  Getting true lighting variety from a recorded file means SEEKING to
  different offsets within it (cv2.CAP_PROP_POS_FRAMES), not waiting for the
  clock. Seeking does work here but is slow - each seek forces a fresh
  byte-range fetch and re-buffer - so budget minutes per sample, not
  seconds.

WHY THIS MATTERS MORE THAN DATASET SIZE
  The existing detection training set was 352 frames captured in one
  sitting, so every frame shares one sun angle and one exposure. A detector
  trained on that learns afternoon Gujarat and degrades at dusk and night -
  which is exactly when loitering and watchlist alerts matter most.

  Doubling frames from the same hour adds far less than the same number of
  frames spread across lighting conditions. This harvests into explicit
  bins and reports coverage, so gaps are visible instead of assumed away.

HOW IT WORKS
  Each run captures ONE bin - whichever the clock currently falls in - and
  appends to the dataset. Run it at different times of day (or leave it on
  a schedule) and coverage accumulates. It never silently overwrites an
  earlier bin.

  Bins are wall-clock, but the ACTUAL brightness of each captured frame is
  measured and recorded too, because clock time is a proxy: an overcast
  monsoon afternoon can be darker than a lit street at night. The report
  shows both so you can see when they disagree.

USAGE
  python -m backend.scripts.diurnal_harvest --per-camera 20
  python -m backend.scripts.diurnal_harvest --status      # coverage only
"""
from __future__ import annotations

import argparse
import json
import os
import sys
from collections import defaultdict
from datetime import datetime
from pathlib import Path

os.environ.setdefault(
    "OPENCV_FFMPEG_CAPTURE_OPTIONS",
    "timeout;15000000|stimeout;15000000|rw_timeout;15000000",
)

import cv2
import numpy as np
import yaml

# Wall-clock bins. Boundaries are deliberately coarse - the point is spread,
# not precision.
BINS = [
    ("night", 0, 6),
    ("morning", 6, 11),
    ("midday", 11, 16),
    ("evening", 16, 20),
    ("night", 20, 24),
]
ROOT = Path("data/detection_train/images")
MANIFEST = Path("data/detection_train/diurnal_manifest.json")


def current_bin(hour: int | None = None) -> str:
    h = datetime.now().hour if hour is None else hour
    for name, lo, hi in BINS:
        if lo <= h < hi:
            return name
    return "night"


def brightness_label(frame) -> tuple[str, float]:
    """Measured brightness, independent of the clock.

    Recorded alongside the time bin because they can disagree - a monsoon
    afternoon can be darker than a lit road at midnight, and the detector
    cares about the pixels, not the hour.
    """
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    mean = float(gray.mean())
    if mean < 60:
        return "dark", mean
    if mean > 180:
        return "bright", mean
    return "mid", mean


def load_manifest() -> dict:
    if MANIFEST.is_file():
        try:
            return json.loads(MANIFEST.read_text())
        except json.JSONDecodeError:
            print("WARNING: diurnal_manifest.json unreadable - starting fresh",
                  file=sys.stderr)
    return {"frames": []}


def show_status(man: dict) -> None:
    by_bin = defaultdict(int)
    by_light = defaultdict(int)
    by_cam = defaultdict(int)
    for f in man["frames"]:
        by_bin[f["bin"]] += 1
        by_light[f["brightness"]] += 1
        by_cam[f["camera"]] += 1

    print(f"harvested frames: {len(man['frames'])} across {len(by_cam)} camera(s)\n")
    print("by time bin (target: some coverage in every bin)")
    for name in ("night", "morning", "midday", "evening"):
        n = by_bin.get(name, 0)
        flag = "  <-- MISSING" if n == 0 else ""
        print(f"  {name:<9} {n:>5}{flag}")
    print("\nby measured brightness")
    for k in ("dark", "mid", "bright"):
        print(f"  {k:<9} {by_light.get(k, 0):>5}")

    # Brightness is the metric that actually matters for the detector, and on
    # this deployment it is the ONLY trustworthy one - the corp8 feeds are
    # recorded files, so the time bin says nothing about the pixels. Judge
    # coverage on brightness first.
    dark, bright = by_light.get("dark", 0), by_light.get("bright", 0)
    total = max(len(man["frames"]), 1)
    print()
    if dark == 0:
        print("NO dark frames captured. A detector trained without them will")
        print("degrade at night - when loitering and watchlist alerts matter")
        print("most. On a recorded source, seek to a different offset in the")
        print("file rather than waiting for the clock (see module docstring).")
    elif dark / total < 0.15:
        print(f"Only {dark}/{total} frames are dark ({dark/total*100:.0f}%).")
        print("Low-light coverage is thin; the model will be weakest exactly")
        print("where alerting matters most.")
    else:
        print(f"Low-light coverage: {dark}/{total} ({dark/total*100:.0f}%) - reasonable.")

    if bright == 0:
        print("No bright/high-glare frames either - harsh midday sun is a")
        print("distinct failure mode (blown highlights, hard shadows).")

    missing = [b for b in ("night", "morning", "midday", "evening")
               if by_bin.get(b, 0) == 0]
    if missing:
        print(f"\n(time bins with no frames: {', '.join(missing)} - only")
        print(" meaningful if these are genuinely live cameras)")


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--per-camera", type=int, default=20)
    ap.add_argument("--every-nth", type=int, default=8)
    ap.add_argument("--timeout", type=float, default=40.0)
    ap.add_argument("--status", action="store_true",
                    help="Show coverage and exit without capturing.")
    args = ap.parse_args()

    man = load_manifest()
    if args.status:
        show_status(man)
        return

    bin_name = current_bin()
    print(f"harvesting into time bin: {bin_name} "
          f"(local hour {datetime.now().hour})\n")

    cfg = yaml.safe_load(open("config.yaml", encoding="utf-8"))
    cams = [c for c in cfg.get("demo_cameras", [])
            if c.get("enabled") is not False and c.get("url")]

    seen = {f["path"] for f in man["frames"]}
    added = 0

    for i, cam in enumerate(cams, 1):
        cam_id = cam["id"]
        out_dir = ROOT / cam_id
        out_dir.mkdir(parents=True, exist_ok=True)
        saved = 0
        try:
            cap = cv2.VideoCapture(cam["url"])
            if not cap.isOpened():
                cap.release()
                print(f"  [{i:>2}/{len(cams)}] {cam_id:<8} could not open",
                      flush=True)
                continue
            import time as _t
            t0 = _t.perf_counter()
            reads = 0
            while saved < args.per_camera and (_t.perf_counter() - t0) < args.timeout:
                ok, frame = cap.read()
                if not ok or frame is None:
                    continue
                reads += 1
                if reads % args.every_nth:
                    continue
                light, mean = brightness_label(frame)
                # Bin is part of the filename so a later run in a different
                # bin cannot overwrite this one.
                name = f"{cam_id}_{bin_name}_{int(_t.time())}_{saved:03d}.jpg"
                path = out_dir / name
                cv2.imwrite(str(path), frame)
                rel = str(path).replace("\\", "/")
                if rel not in seen:
                    man["frames"].append({
                        "path": rel, "camera": cam_id, "bin": bin_name,
                        "brightness": light, "mean_luma": round(mean, 1),
                        "captured_at": datetime.now().isoformat(timespec="seconds"),
                    })
                    seen.add(rel)
                saved += 1
                added += 1
            cap.release()
        except Exception as exc:
            print(f"  [{i:>2}/{len(cams)}] {cam_id:<8} EXCEPTION "
                  f"{type(exc).__name__}", flush=True)
            continue
        print(f"  [{i:>2}/{len(cams)}] {cam_id:<8} saved={saved}", flush=True)

    MANIFEST.parent.mkdir(parents=True, exist_ok=True)
    MANIFEST.write_text(json.dumps(man, indent=2))
    print(f"\nadded {added} frame(s) this run")
    print()
    show_status(man)


if __name__ == "__main__":
    main()
