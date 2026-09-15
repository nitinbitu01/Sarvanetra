"""backend/scripts/plate_mine_corpus.py — ANPR Phase 1: mine the plate corpus.

WHAT THIS PRODUCES
  Vehicle crops large enough to carry a readable plate, cut from the SOURCE
  resolution frame and GROUPED BY TRACK. The grouping is the point: one vehicle
  seen across N consecutive frames is a single multi-frame fusion group, and
  fusion is the only stage of the ANPR pipeline that adds real information
  rather than merely revealing it. A pile of ungrouped single frames would be
  worth far less.

WHY IT SWEEPS INSTEAD OF SAMPLING
  Readable plates are RARE - fleet-wide only ~8.6% of four-wheelers reach the
  100px vendor floor and ~27% reach 50px (see plate_camera_survey.py). Sampling
  40 frames per camera finds almost none of them. These clips are on local
  disk, so they can be swept exhaustively, offline, at any compute cost, and
  re-swept whenever the thresholds change. That is what harvested clips are
  FOR, and it cannot be done against a live stream.

SCREENING AT LOW RESOLUTION - deliberate
  Detection runs at imgsz=640, not 1280. We are only looking for vehicles wide
  enough to carry a plate (>=178px), and objects that large are trivial to
  detect at 640. This is ~3-4x faster than screening at full resolution. The
  CROP is always taken from the full-resolution source frame, so no plate
  detail is lost - only the screening pass is cheap.

QUALITY METADATA
  Every crop records variance-of-Laplacian (sharpness). Phase 3 fuses only the
  sharpest 5-8 frames of each track: averaging blurry frames into sharp ones
  degrades the result, so more frames is not better - sharper frames are.

RESUMABLE
  Completed clips are recorded in the manifest and skipped on re-run. Safe to
  interrupt and restart.

USAGE
  python -m backend.scripts.plate_mine_corpus
  python -m backend.scripts.plate_mine_corpus --cameras CAM_27,CAM_06 --max-tracks 500
"""
from __future__ import annotations

import argparse
import json
import os
import re
import shutil
import sys
import time
from pathlib import Path

os.environ.setdefault("OPENCV_FFMPEG_CAPTURE_OPTIONS",
                      "timeout;30000000|stimeout;30000000|rw_timeout;30000000")

import cv2
import numpy as np
import yaml

# Ranked by measured plate-pixel yield (plate_camera_survey.py, output/
# plate_camera_survey.json). The remaining ~19 cameras produced no vehicles
# above the floor and are not worth the sweep time.
DEFAULT_CAMS = ["CAM_27", "CAM_06", "CAM_09", "CAM_25", "CAM_18",
                "CAM_07", "CAM_21", "CAM_08", "CAM_02", "CAM_10"]

PLATE_FRAC = 0.28                 # Indian plate 500mm on a ~1800mm vehicle
CAR, MOTO, BUS, TRUCK = 1, 2, 3, 4
FOURWHEEL = {CAR, BUS, TRUCK}

# Detector that produced the current run's crops, recorded on every manifest
# row. Set in main() from config.yaml.
_MODEL_TAG = "unknown"

# Abort guards. An unattended overnight run must fail loudly and early rather
# than spend hours producing nothing - which is exactly what happened when a
# CUDA fault on clip 2 let clips 3-43 "complete" in seconds, each writing no
# crops, and the run reported 43 clips swept.
# Errors do NOT abort the run. A bad clip is skipped, a CUDA fault triggers a
# context rebuild, and only a genuinely unrecoverable state gives up - one
# broken file must never cost the remaining forty clips.
MAX_CONSECUTIVE_ERRORS = 3    # pause and retry rather than abort
MAX_EMPTY_STREAK = 8          # expected yield is ~100-2000 crops per clip
MIN_FREE_GB = 3.0

# Exit code meaning "the process is wedged but the WORK is fine - restart me".
# Completed clips are recorded, so a restart resumes instead of repeating.
EXIT_RESTART = 75

OUT_ROOT = Path("output/plate_corpus")
MANIFEST = OUT_ROOT / "manifest.jsonl"
DONE_FILE = OUT_ROOT / "clips_done.json"


def sharpness(img) -> float:
    """Variance of Laplacian. Higher = sharper. Used to pick fusion frames."""
    g = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    return float(cv2.Laplacian(g, cv2.CV_64F).var())


class MotionGate:
    """Reject STATIONARY tracks - they are parked cars, not passing traffic.

    WHY THIS EXISTS
      The first sweep produced exactly 25 crops from exactly 1 track on three
      consecutive clips. Diagnosis: a single parked car, box width identical
      (376px) across all 400 frames tested, tracked continuously and mined
      repeatedly. The camera survey had the same flaw - it sampled that one
      parked vehicle 40 times and reported CAM_27 as the highest-yield camera
      in the fleet.

    WHY PARKED CARS ARE WORTHLESS HERE
      The corpus needs DISTINCT plates. A thousand crops of one parked car is
      one plate, and fusing frames of a static object gains nothing: multi-
      frame super-resolution works precisely because sub-pixel jitter samples
      the plate on slightly different grids as the vehicle moves. A perfectly
      still vehicle contributes the same samples over and over.

    THE TEST
      Track the centroid. A track must travel at least `min_travel_px` from
      where it was first seen before any of its crops are kept. Verdicts are
      cached per track so the check costs nothing after it passes.
    """

    def __init__(self, min_travel_px: float = 80.0) -> None:
        self._first: dict[int, tuple[float, float]] = {}
        self._moving: set[int] = set()
        self.min_travel = min_travel_px

    def passes(self, tid: int, cx: float, cy: float) -> bool:
        if tid in self._moving:
            return True
        if tid not in self._first:
            self._first[tid] = (cx, cy)
            return False
        ox, oy = self._first[tid]
        if (cx - ox) ** 2 + (cy - oy) ** 2 >= self.min_travel ** 2:
            self._moving.add(tid)
            return True
        return False

    @property
    def n_seen(self) -> int:
        return len(self._first)

    @property
    def n_moving(self) -> int:
        return len(self._moving)


def load_done() -> set[str]:
    if DONE_FILE.is_file():
        try:
            return set(json.loads(DONE_FILE.read_text()))
        except Exception:
            return set()
    return set()


def save_done(done: set[str]) -> None:
    DONE_FILE.write_text(json.dumps(sorted(done), indent=0))


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--cameras", default="ALL",
                    help="Comma-separated camera ids, or ALL. Defaults to ALL: "
                         "the earlier per-camera ranking counted parked cars "
                         "and was close to inverted for the cameras that "
                         "matter, so pre-filtering on it would skip good "
                         "cameras. The per-clip yield printed below IS the "
                         "ranking - no separate survey pass is needed.")
    ap.add_argument("--times", default="0630,0730,0830",
                    help="Clip time suffixes to mine. Defaults to dawn+daylight: "
                         "night is 40%% of the footage and plates need IR "
                         "illumination this fleet does not have, so 0100/2200 "
                         "are deprioritised until daylight is exhausted.")
    ap.add_argument("--min-plate-px", type=float, default=50.0,
                    help="Keep vehicles whose implied plate is at least this "
                         "wide. 50px is the fusion+SR tier; 70 is 'readable "
                         "with enhancement'; 100 is the vendor recommendation.")
    ap.add_argument("--screen-imgsz", type=int, default=640,
                    help="Detection size for SCREENING only. Crops are always "
                         "taken from the full-resolution source frame.")
    ap.add_argument("--max-frames-per-track", type=int, default=60,
                    help="Fusion needs only the sharpest handful; 60 is ample "
                         "headroom and bounds disk use.")
    ap.add_argument("--max-tracks-per-clip", type=int, default=400)
    ap.add_argument("--conf", type=float, default=0.35)
    ap.add_argument("--min-travel-px", type=float, default=80.0,
                    help="A track must move at least this far from where it "
                         "was first seen before its crops are kept. Rejects "
                         "parked cars, which otherwise dominate the corpus "
                         "and contribute exactly one plate.")
    ap.add_argument("--include-moto", action="store_true",
                    help="Motorcycles carry two-row plates with ~45mm "
                         "characters and need ~1.5x more width; excluded by "
                         "default so they do not inflate the corpus.")
    args = ap.parse_args()

    min_veh_w = args.min_plate_px / PLATE_FRAC
    if args.cameras.strip().upper() == "ALL":
        cams = sorted(d.name for d in Path("data/clips").iterdir() if d.is_dir())
    else:
        cams = [c.strip() for c in args.cameras.split(",") if c.strip()]

    cfg = yaml.safe_load(open("config.yaml", encoding="utf-8"))
    from ultralytics import YOLO
    model = YOLO(cfg["model"]["path"])
    global _MODEL_TAG
    _MODEL_TAG = Path(cfg["model"]["path"]).name

    OUT_ROOT.mkdir(parents=True, exist_ok=True)
    done = load_done()
    keep_cls = FOURWHEEL | ({MOTO} if args.include_moto else set())

    # Split on commas OR whitespace: PowerShell turns an unquoted a,b,c into an
    # array and joins it with spaces before the process ever sees it, so
    # "0630,0730,0830" can arrive as "0630 0730 0830".
    times = [t for t in re.split(r"[,\s]+", args.times.strip()) if t]
    clips: list[Path] = []
    for cam in cams:
        d = Path("data/clips") / cam
        if not d.is_dir():
            continue
        for c in sorted(d.glob("*.mp4")):
            if not times or c.stem.split("_")[-1] in times:
                clips.append(c)

    todo = [c for c in clips if str(c) not in done]
    print(f"cameras          : {len(cams)}")
    print(f"clips total      : {len(clips)}  (already done: {len(clips)-len(todo)})")
    print(f"min plate width  : {args.min_plate_px:.0f}px "
          f"-> vehicle >= {min_veh_w:.0f}px")
    print(f"screening imgsz  : {args.screen_imgsz} (crops from source res)")
    print(f"detector         : {cfg['model']['path']}\n", flush=True)

    manifest = MANIFEST.open("a", encoding="utf-8")
    t_start = time.time()
    tot_crops = tot_tracks = 0
    consecutive_errors = 0
    empty_streak = 0
    n_errors = 0
    n_ok = 0

    for ci, clip in enumerate(todo, 1):
        cam = clip.parent.name
        cap = cv2.VideoCapture(str(clip))
        n_frames = cap.get(cv2.CAP_PROP_FRAME_COUNT)
        cap.release()
        print(f"[{ci}/{len(todo)}] {cam}/{clip.name}  ({n_frames:.0f} frames)",
              flush=True)

        per_track: dict[int, int] = {}
        gate = MotionGate(args.min_travel_px)
        clip_crops = 0
        t0 = time.time()

        try:
            # persist=True keeps BoT-SORT track IDs stable across the clip, so
            # frames of the same vehicle land in the same fusion group.
            stream = model.track(source=str(clip), stream=True, persist=True,
                                 tracker="botsort.yaml", conf=args.conf,
                                 imgsz=args.screen_imgsz, verbose=False,
                                 quantize=16)
            for fi, r in enumerate(stream):
                if r.boxes is None or r.boxes.id is None:
                    continue
                frame = r.orig_img
                H, W = frame.shape[:2]
                for b in r.boxes:
                    if int(b.cls[0]) not in keep_cls:
                        continue
                    tid = int(b.id[0])
                    if per_track.get(tid, 0) >= args.max_frames_per_track:
                        continue
                    if len(per_track) >= args.max_tracks_per_clip and tid not in per_track:
                        continue
                    x1, y1, x2, y2 = (int(v) for v in b.xyxy[0].tolist())
                    vw = x2 - x1
                    if vw < min_veh_w:
                        continue
                    # Parked cars pass every size check forever. Require the
                    # track to have actually travelled before keeping it.
                    if not gate.passes(tid, (x1 + x2) / 2, (y1 + y2) / 2):
                        continue

                    pad = int(vw * 0.06)
                    cx1, cy1 = max(0, x1 - pad), max(0, y1 - pad)
                    cx2, cy2 = min(W, x2 + pad), min(H, y2 + pad)
                    crop = frame[cy1:cy2, cx1:cx2]
                    if crop.size == 0:
                        continue

                    tdir = OUT_ROOT / cam / clip.stem / f"track_{tid:05d}"
                    tdir.mkdir(parents=True, exist_ok=True)
                    fp = tdir / f"f{fi:06d}.jpg"
                    cv2.imwrite(str(fp), crop, [cv2.IMWRITE_JPEG_QUALITY, 96])

                    manifest.write(json.dumps({
                        # Which detector produced this crop. Clips 1-27 were
                        # mined with the YOLOv8s Gujarat model; anything mined
                        # later may come from a different one. Without this a
                        # future "CAM_11 yielded less than CAM_09" comparison
                        # cannot tell a camera difference from a model change.
                        "model": _MODEL_TAG,
                        "camera": cam,
                        "clip": clip.stem,
                        "track": tid,
                        "frame": fi,
                        "path": str(fp).replace("\\", "/"),
                        "veh_w": vw,
                        "plate_px_est": round(vw * PLATE_FRAC, 1),
                        "sharpness": round(sharpness(crop), 1),
                        "cls": int(b.cls[0]),
                    }) + "\n")
                    per_track[tid] = per_track.get(tid, 0) + 1
                    clip_crops += 1
        except Exception as e:                       # noqa: BLE001
            msg = str(e)
            print(f"    ERROR on {clip.name}: {msg}", file=sys.stderr, flush=True)
            consecutive_errors += 1
            n_errors += 1

            # RECOVER, DO NOT GIVE UP.
            # A CUDA context that has faulted poisons every later GPU call in
            # the process - a previous run hit an illegal memory access on
            # clip 2 and then "swept" clips 3-43 in seconds, each writing
            # nothing. But the right response is to REBUILD the context and
            # carry on, not to abandon 40 clips of work. Freeing the cache and
            # reloading the model recovers most faults; when it does not, the
            # process exits with EXIT_RESTART so a supervisor can restart it,
            # and because completed clips are recorded it resumes rather than
            # repeating work.
            if "CUDA" in msg or "cuda" in msg:
                print("    CUDA fault — attempting recovery "
                      "(cache flush + model reload)...", file=sys.stderr,
                      flush=True)
                try:
                    import torch
                    torch.cuda.empty_cache()
                    torch.cuda.synchronize()
                    model = YOLO(cfg["model"]["path"])
                    print("    recovered — continuing with the next clip.",
                          file=sys.stderr, flush=True)
                    continue
                except Exception as exc2:            # noqa: BLE001
                    print(f"    recovery failed ({exc2}). Exiting with "
                          f"EXIT_RESTART so a supervisor can restart the "
                          f"process; {len(done)} clips are recorded and will "
                          f"be skipped on resume.", file=sys.stderr, flush=True)
                    manifest.close()
                    save_done(done)
                    sys.exit(EXIT_RESTART)

            # Any other error: skip this clip and keep going. One unreadable
            # file must not cost the remaining forty.
            if consecutive_errors >= MAX_CONSECUTIVE_ERRORS:
                print(f"    {consecutive_errors} in a row — pausing 20s in "
                      f"case this is transient (disk, decoder), then "
                      f"continuing.", file=sys.stderr, flush=True)
                time.sleep(20)
                consecutive_errors = 0
            continue
        consecutive_errors = 0

        # A clip that yields nothing is not an error, but a RUN of them means
        # the detector is returning the wrong classes (e.g. a COCO model where
        # the 8-class Gujarat order was assumed) rather than the footage being
        # quiet.
        if clip_crops == 0:
            empty_streak += 1
            if empty_streak >= MAX_EMPTY_STREAK:
                # THE worst case the user described: not an error on one clip,
                # but every clip silently producing nothing. That is a
                # configuration fault (wrong class indices from a COCO
                # checkpoint, say) and no amount of continuing will fix it.
                print(f"\n*** {empty_streak} clips in a row produced ZERO "
                      f"crops. Expected yield is ~100-2000 per clip, so this "
                      f"is a configuration fault rather than quiet footage — "
                      f"most likely model.path pointing at a bare COCO "
                      f"checkpoint whose class indices differ from this "
                      f"project's 8-class order.", file=sys.stderr)
                print(f"*** Stopping. {len(done)} clips recorded; fix the "
                      f"config and rerun to resume.", file=sys.stderr,
                      flush=True)
                break
        else:
            empty_streak = 0

        # Disk guard: the corpus grows ~100MB per clip and the machine had
        # ~20GB free. Running the volume to zero mid-run risks corrupting the
        # manifest as well as losing the remaining clips.
        try:
            free_gb = shutil.disk_usage(".").free / 2**30
            if free_gb < MIN_FREE_GB:
                print(f"\n*** only {free_gb:.1f}GB free — stopping before the "
                      f"disk fills. {len(done)} clips recorded; free space and "
                      f"rerun to resume.", file=sys.stderr, flush=True)
                break
        except Exception:                            # noqa: BLE001
            pass

        manifest.flush()
        done.add(str(clip))
        save_done(done)
        n_ok += 1
        tot_crops += clip_crops
        tot_tracks += len(per_track)
        dt = time.time() - t0
        print(f"    -> {clip_crops} crops from {len(per_track)} moving tracks "
              f"({gate.n_seen - gate.n_moving} stationary rejected) "
              f"in {dt/60:.1f} min", flush=True)

    manifest.close()
    mins = (time.time() - t_start) / 60
    print("\n" + "=" * 60)
    print(f"clips OK      : {n_ok}")
    print(f"clips skipped : {n_errors}  (errors, recovered and moved on)")
    print(f"clips swept   : {len(todo)}")
    print(f"tracks        : {tot_tracks}")
    print(f"crops         : {tot_crops}")
    print(f"elapsed       : {mins:.1f} min")
    print(f"manifest      : {MANIFEST}")
    print("\nNEXT (Phase 2): bootstrap plate boxes over these crops with a")
    print("pretrained plate detector, hand-verify a few hundred, fine-tune")
    print("YOLOv8n at imgsz=320.")


if __name__ == "__main__":
    main()
