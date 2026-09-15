"""backend/scripts/make_demo_clips.py — pull recorded clips for demo fallback.

WHY THIS EXISTS
  The live feeds are not dependable. Measured over one working day the
  corp8 server returned 502 twice, and at its worst delivered 0.07-0.13 MB/s
  where simply OPENING a stream took 175-280s. A live demo in that state is
  impossible, and preflight_check.py's own docstring already notes roughly
  one run in four produces zero detections even when the server is healthy.

  backend/main.py supports SENTINEL_SOURCES="CAM_ID=path.mp4" to run the
  real pipeline against a file instead of a stream. That path existed but
  had never been exercised. An untested fallback is not a fallback.

WHICH CLIPS AND WHY
  CAM_01  the wrong-way calibrated camera (config/traffic/CAM_01.json), so
          the traffic-intelligence feature can be demonstrated offline. Its
          homography is a property of the fixed viewpoint, not the lighting,
          so a daylight clip works with a calibration derived at night.
  CAM_04  highest confirmed-track count in the last GO preflight (372).
  CAM_02  second highest (188), different district.

  Daylight is chosen deliberately: it shows the v3 model where it improved
  most (F1 0.681 -> 0.801 on held-out daylight frames) and reads better on
  a projector than a night scene full of headlight glare.

USAGE
  python -m backend.scripts.make_demo_clips
  python -m backend.scripts.make_demo_clips --cameras CAM_01 --seconds 90
"""
from __future__ import annotations

import argparse
import shutil
import subprocess
import sys
import time
from pathlib import Path

import yaml

OUT_DIR = Path("demo/clips")
FILE_START_HOUR = 21.0
FILE_DURATION_H = 12.0


def clock_to_offset_s(hhmm: str) -> float | None:
    h, m = (int(x) for x in hhmm.split(":"))
    off = ((h + m / 60.0) - FILE_START_HOUR) % 24.0
    return None if off > FILE_DURATION_H else off * 3600.0


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--cameras", default="CAM_01,CAM_04,CAM_02")
    ap.add_argument("--time", default="07:00",
                    help="Wall-clock point to cut from. Recording spans "
                         "21:00-09:00, so daylight is 06:00-09:00.")
    ap.add_argument("--seconds", type=int, default=90)
    ap.add_argument("--timeout", type=float, default=1500.0)
    args = ap.parse_args()

    if not shutil.which("ffmpeg"):
        print("ffmpeg not found on PATH.", file=sys.stderr)
        sys.exit(1)

    offset = clock_to_offset_s(args.time)
    if offset is None:
        print(f"{args.time} is outside the 21:00-09:00 recording.", file=sys.stderr)
        sys.exit(1)

    cfg = yaml.safe_load(open("config.yaml", encoding="utf-8"))
    urls = {c["id"]: c["url"] for c in cfg.get("demo_cameras", [])
            if c.get("enabled") is not False and c.get("url")}

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    wanted = [c.strip() for c in args.cameras.split(",") if c.strip()]
    # Keep (camera_id, path) pairs. Parsing the id back out of the filename
    # is fragile - "cam_01_0700".split("_0")[0] gives "cam", not "CAM_01".
    made: list[tuple[str, Path]] = []

    for cam in wanted:
        if cam not in urls:
            print(f"  {cam}: not in config, skipping", flush=True)
            continue
        dst = OUT_DIR / f"{cam.lower()}_{args.time.replace(':', '')}.mp4"
        print(f"  {cam}: cutting {args.seconds}s from {args.time} "
              f"(offset {offset:.0f}s)...", flush=True)
        t0 = time.perf_counter()
        cmd = [
            "ffmpeg", "-hide_banner", "-loglevel", "error",
            "-rw_timeout", "300000000",
            "-ss", f"{offset:.0f}",       # BEFORE -i: index-based input seek
            "-i", urls[cam],
            "-t", str(args.seconds),
            # Re-encode rather than -c copy: a stream copy starts at the
            # nearest keyframe and can emit a file whose first frames are
            # undecodable, which is exactly the surprise you do not want
            # when the fallback is being used under pressure.
            "-c:v", "libx264", "-preset", "veryfast", "-crf", "23",
            "-pix_fmt", "yuv420p", "-an",
            "-y", str(dst),
        ]
        try:
            p = subprocess.run(cmd, capture_output=True, text=True,
                               timeout=args.timeout)
            el = time.perf_counter() - t0
            if p.returncode != 0 or not dst.is_file() or dst.stat().st_size < 10_000:
                print(f"    FAILED rc={p.returncode} "
                      f"{p.stderr.strip()[:140]}", flush=True)
                continue
            mb = dst.stat().st_size / 1_048_576
            print(f"    ok  {mb:.1f} MB in {el:.0f}s -> {dst}", flush=True)
            made.append((cam, dst))
        except subprocess.TimeoutExpired:
            print(f"    TIMEOUT after {args.timeout}s", flush=True)

    print()
    if not made:
        print("No clips produced. The server may be too degraded right now; "
              "retry when preflight_check.py reports GO.")
        sys.exit(1)

    print(f"{len(made)} clip(s) written to {OUT_DIR}/")
    print("\nRun the pipeline against them instead of the live feeds:")
    srcs = ",".join(f"{cam}={p.as_posix()}" for cam, p in made)
    print(f'  $env:SENTINEL_SOURCES="{srcs}"')
    print("  uvicorn backend.main:app")
    print("\nOr verify the fallback end to end:")
    print(f'  $env:SENTINEL_SOURCES="{srcs}"; python preflight_check.py 3 60')


if __name__ == "__main__":
    main()
