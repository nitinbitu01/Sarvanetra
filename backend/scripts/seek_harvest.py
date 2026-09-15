"""backend/scripts/seek_harvest.py — harvest DAYLIGHT frames by seeking into
the recording with ffmpeg, instead of always reading from the start.

THE GAP THIS FILLS
  Every earlier harvest read sequentially from frame 0, so the entire
  dataset comes from the first minutes of each file. Measured result:
      4,292 frames = 650 dark / 3,059 mid / 0 BRIGHT
  Zero bright frames means the detector has never seen harsh sun, blown
  highlights or hard shadows - precisely the conditions it will then fail in.

WHAT IS ACTUALLY IN THESE FILES (verified, not assumed)
  Burned-in start timestamp : 13-06-2026 21:00:00
  Duration                  : 12.00 h  (1,291,811 frames / 29.90 fps)
  True size                 : 2.55 GB  (from Content-Range)
  => the recordings span 21:00 -> 09:00 the NEXT day.

  Confirmed empirically: seeking to 36,288 s (10.08 h) produced a frame
  stamped 14-06-2026 07:04:42 in full daylight. So midday sun and evening
  twilight are NOT present; the only daylight is roughly 06:00-09:00.

WHY ffmpeg AND NOT cv2
  cv2.VideoCapture.set(POS_FRAMES/POS_MSEC) on these URLs hung for 20+
  minutes and never returned. `ffmpeg -ss <t> -i <url>` does an index-based
  INPUT seek (note: -ss must come BEFORE -i) and completed the same jump in
  284 s. The server is genuinely slow - measured 0.07-0.13 MB/s via range
  requests - so the seek is expensive either way, but ffmpeg actually
  finishes.

  Because one seek costs ~284 s, each invocation extracts a RUN of frames
  rather than a single image, amortising that cost over many samples.

USAGE
  python -m backend.scripts.seek_harvest --times 06:45,08:15
  python -m backend.scripts.seek_harvest --cameras CAM_04,CAM_05 --window 120
"""
from __future__ import annotations

import argparse
import json
import shutil
import subprocess
import sys
import tempfile
import threading
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import cv2
import numpy as np
import yaml

OUT_ROOT = Path("data/detection_train/images")
MANIFEST = Path("data/detection_train/seek_manifest.json")
FILE_START_HOUR = 21.0        # verified from the burned-in clock overlay
FILE_DURATION_H = 12.0        # verified: frames/fps on two cameras
_lock = threading.Lock()


def clock_to_offset_s(hhmm: str) -> float | None:
    """Wall-clock time -> seconds into the recording, or None if outside it."""
    h, m = (int(x) for x in hhmm.split(":"))
    t = h + m / 60.0
    off = (t - FILE_START_HOUR) % 24.0
    if off > FILE_DURATION_H:
        return None
    return off * 3600.0


def thumb(frame) -> np.ndarray:
    return cv2.cvtColor(cv2.resize(frame, (160, 90)), cv2.COLOR_BGR2GRAY).astype(np.int16)


def brightness_label(frame) -> tuple[str, float]:
    """Bucket a frame by measured luminance.

    The old thresholds (dark <60, bright >180) reported 'bright: 0' for a
    harvest that had in fact captured real daylight: a sunlit 07:00 street
    measures ~143, while a night street under sodium lamps measures ~100.
    Both landed in 'mid', so the one bucket held two completely different
    lighting regimes and the summary read as though nothing had been gained.

    A 'day' band at 130-180 separates them. Verified against harvested
    frames: 14-06-2026 07:00:52 daylight = 143.5 luma; 13-06-2026 21:00
    night = ~100.
    """
    mean = float(cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY).mean())
    if mean < 60:
        return "dark", mean
    if mean > 180:
        return "bright", mean          # blown highlights / harsh glare
    if mean >= 130:
        return "day", mean             # ordinary daylight
    return "mid", mean                 # lit night / dusk / overcast


def harvest_one(cam: dict, hhmm: str, args) -> dict:
    cam_id = cam["id"]
    offset = clock_to_offset_s(hhmm)
    rec = {"camera": cam_id, "district": cam.get("district"), "clock": hhmm,
           "offset_s": offset, "saved": 0, "rejected_similar": 0,
           "extracted": 0, "elapsed_s": None,
           "brightness": {"dark": 0, "mid": 0, "day": 0, "bright": 0}, "status": "ok"}
    if offset is None:
        rec["status"] = f"{hhmm} is outside the 21:00-09:00 recording"
        return rec

    out_dir = OUT_ROOT / cam_id
    out_dir.mkdir(parents=True, exist_ok=True)
    tmp = Path(tempfile.mkdtemp(prefix=f"seek_{cam_id}_"))
    t0 = time.perf_counter()

    try:
        cmd = [
            "ffmpeg", "-hide_banner", "-loglevel", "error",
            "-rw_timeout", "300000000",       # 300s; server runs 0.07-0.13 MB/s
            "-ss", f"{offset:.0f}",           # BEFORE -i => fast input seek
            "-i", cam["url"],
            "-t", str(args.window),
            "-vf", f"fps=1/{args.gap_seconds}",
            "-q:v", "2",
            # Some cameras (CAM_23/27/28) deliver MJPEG whose colour range
            # ffmpeg calls non-standard, and it then REFUSES TO WRITE rather
            # than warn: "Non full-range YUV is non-standard, set
            # strict_std_compliance to at most unofficial". That killed 6
            # otherwise-fine jobs. Naming the JPEG pixel format explicitly
            # (and relaxing compliance) makes the encode proceed - this is
            # an output-side quirk, nothing wrong with the source footage.
            "-pix_fmt", "yuvj420p",
            "-strict", "unofficial",
            "-y", str(tmp / "f_%04d.jpg"),
        ]
        proc = subprocess.run(cmd, capture_output=True, text=True,
                              timeout=args.timeout)
        if proc.returncode != 0:
            rec["status"] = f"ffmpeg rc={proc.returncode}: {proc.stderr.strip()[:120]}"

        # Perceptual dedup against the last KEPT frame - a fixed camera
        # produces near-identical frames and they teach the model nothing.
        last_keep = None
        stamp = int(time.time())
        for p in sorted(tmp.glob("*.jpg")):
            frame = cv2.imread(str(p))
            if frame is None:
                continue
            rec["extracted"] += 1
            t = thumb(frame)
            if last_keep is not None:
                if float(np.mean(np.abs(t - last_keep))) < args.min_diff:
                    rec["rejected_similar"] += 1
                    continue
            label, _ = brightness_label(frame)
            name = (f"{cam_id}_seek{stamp}_{hhmm.replace(':', '')}"
                    f"_{rec['saved']:04d}_{label}.jpg")
            shutil.copy(str(p), str(out_dir / name))
            rec["brightness"][label] += 1
            rec["saved"] += 1
            last_keep = t
    except subprocess.TimeoutExpired:
        rec["status"] = f"TIMEOUT after {args.timeout}s"
    except Exception as exc:
        rec["status"] = f"EXCEPTION {type(exc).__name__}: {exc}"
    finally:
        shutil.rmtree(tmp, ignore_errors=True)

    rec["elapsed_s"] = round(time.perf_counter() - t0, 1)
    with _lock:
        b = rec["brightness"]
        print(f"  {cam_id:<9} {hhmm}  saved={rec['saved']:<3} "
              f"(of {rec['extracted']}, dup {rec['rejected_similar']})  "
              f"dark/mid/day/bright="
              f"{b['dark']}/{b['mid']}/{b['day']}/{b['bright']}  "
              f"{rec['elapsed_s']:>5.0f}s  {rec['status']}", flush=True)
    return rec


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--times", default="06:45,08:15",
                    help="Wall-clock targets. Files run 21:00-09:00, so only "
                         "06:00-09:00 is daylight. Times outside are skipped.")
    ap.add_argument("--window", type=int, default=240,
                    help="Seconds of FOOTAGE to pull per seek. Larger "
                         "amortises the ~284s seek over more frames.")
    ap.add_argument("--gap-seconds", type=float, default=4.0,
                    help="Seconds between extracted frames.")
    ap.add_argument("--min-diff", type=float, default=8.0)
    ap.add_argument("--timeout", type=float, default=1500.0,
                    help="Hard cap per (camera, time) job.")
    ap.add_argument("--workers", type=int, default=4,
                    help="Kept low: a 30-way concurrent probe on these "
                         "endpoints took 872s and produced corrupt h264.")
    ap.add_argument("--cameras", default="")
    args = ap.parse_args()

    if not shutil.which("ffmpeg"):
        print("ffmpeg not found on PATH - required for seeking.", file=sys.stderr)
        sys.exit(1)

    times = [t.strip() for t in args.times.split(",") if t.strip()]
    skipped = [t for t in times if clock_to_offset_s(t) is None]
    times = [t for t in times if clock_to_offset_s(t) is not None]
    for t in skipped:
        print(f"SKIP {t}: outside the 21:00-09:00 recording window")
    if not times:
        print("No requested time falls inside the recording.", file=sys.stderr)
        sys.exit(1)

    cfg = yaml.safe_load(open("config.yaml", encoding="utf-8"))
    only = {c.strip() for c in args.cameras.split(",") if c.strip()} or None
    cams = [c for c in cfg.get("demo_cameras", [])
            if c.get("enabled") is not False and c.get("url")
            and (not only or c["id"] in only)]
    if not cams:
        print("No cameras matched.", file=sys.stderr)
        sys.exit(1)

    jobs = [(c, t) for t in times for c in cams]
    print(f"Seek-harvest: {len(cams)} camera(s) x {len(times)} time(s) "
          f"= {len(jobs)} job(s), {args.workers} at a time")
    print(f"  targets {times}  window {args.window}s  gap {args.gap_seconds}s")
    print(f"  NOTE: each seek costs ~280s on this server (0.07-0.13 MB/s)\n")

    with ThreadPoolExecutor(max_workers=args.workers) as ex:
        report = list(ex.map(lambda j: harvest_one(j[0], j[1], args), jobs))

    saved = sum(r["saved"] for r in report)
    light = {"dark": 0, "mid": 0, "day": 0, "bright": 0}
    for r in report:
        for k in light:
            light[k] += r["brightness"][k]

    MANIFEST.parent.mkdir(parents=True, exist_ok=True)
    MANIFEST.write_text(json.dumps(report, indent=2))

    total_disk = sum(1 for _ in OUT_ROOT.rglob("*.jpg"))
    print()
    print(f"saved this run       : {saved}")
    print(f"brightness           : {light}")
    print(f"jobs producing       : {sum(1 for r in report if r['saved'])}/{len(jobs)}")
    print(f"dataset total on disk: {total_disk}")
    print("\nNext: re-run autolabelling + training to use the new frames:")
    print("  python -m backend.scripts.autolabel_frames --sahi")
    print("  python -m backend.scripts.train_detector")


if __name__ == "__main__":
    main()
