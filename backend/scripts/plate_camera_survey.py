"""backend/scripts/plate_camera_survey.py — WHICH cameras can support ANPR?

THE QUESTION
  Earlier feasibility tests (docs/ANPR_FEASIBILITY.md) showed plates are
  unreadable for the TYPICAL vehicle: median width 120px -> ~5px characters.
  But those tests never isolated the CLOSEST vehicles, and plate readability
  is not uniform across a fleet - it depends entirely on how close traffic
  passes a given camera.

  Real ANPR deployments do not read plates on every camera. They read them at
  one or two chokepoints where vehicles pass close and slow. So the useful
  question is not "can this fleet read plates" but "which camera can, and for
  what fraction of vehicles".

THE GEOMETRY (Indian plate: 500mm x 120mm, characters ~70mm; car ~1800mm)
      plate width      = 0.28 x vehicle width
      character height = 0.039 x vehicle width

      characters >= 20px (OCR floor)     -> vehicle >= ~510px wide
      characters >= 25px (reliable)      -> vehicle >= ~640px wide

  Note this assumes a SINGLE-ROW plate. Two-row plates (most motorcycles,
  some cars) have ~45mm characters and need roughly 1.5x more vehicle width,
  so motorcycle counts here are optimistic and are reported separately.

WHAT THIS OUTPUTS
  Per camera: the vehicle-width distribution and the count/percentage of
  vehicles clearing each threshold. Cameras are ranked by ANPR viability so
  the cascade can be pointed at the ones that can actually support it.

USAGE
  python -m backend.scripts.plate_camera_survey
  python -m backend.scripts.plate_camera_survey --frames-per-cam 60
"""
from __future__ import annotations

import argparse
import json
import os
from collections import defaultdict
from pathlib import Path

os.environ.setdefault("OPENCV_FFMPEG_CAPTURE_OPTIONS",
                      "timeout;30000000|stimeout;30000000|rw_timeout;30000000")

import cv2
import numpy as np
import yaml

# Thresholds are expressed in PLATE width, not character height, because that
# is the unit every ANPR vendor specifies against and it makes our numbers
# directly comparable to theirs:
#   Plate Recognizer  - recommends ~100px plate width; read correctly down to
#                       30px in controlled tests (clean, frontal, sharp)
#   industry general  - 100-150px across the full plate
# An earlier revision of this script demanded 20px CHARACTERS (= ~143px plate,
# ~510px vehicle), which is ~40% stricter than the vendor recommendation and
# understated how much of the fleet is usable. Report a spectrum instead of
# one arbitrary line.
#
# Indian plate 500mm wide on a ~1800mm car -> plate = 0.28 x vehicle width.
PLATE_FRAC = 0.28
CHAR_PER_PX = 0.039                      # character height per px of vehicle width

# (label, required plate width px, what it corresponds to)
THRESHOLDS = [
    ("100px plate (vendor rec)", 100),
    ("70px  plate (w/ enhance)", 70),
    ("50px  plate (fusion+SR)", 50),
]


def veh_width_for_plate(plate_px: float) -> int:
    """Vehicle width in px needed to yield a plate of the given width."""
    return int(plate_px / PLATE_FRAC)

CAR, MOTO, BUS, TRUCK = 1, 2, 3, 4
FOURWHEEL = {CAR, BUS, TRUCK}
OUT_JSON = Path("output/plate_camera_survey.json")


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--frames-per-cam", type=int, default=40)
    ap.add_argument("--time", default="0830", help="Daylight clip suffix.")
    ap.add_argument("--conf", type=float, default=0.40)
    args = ap.parse_args()

    cfg = yaml.safe_load(open("config.yaml", encoding="utf-8"))
    from ultralytics import YOLO
    model = YOLO(cfg["model"]["path"])

    clip_root = Path("data/clips")
    cams = sorted(d.name for d in clip_root.iterdir() if d.is_dir())
    print(f"surveying {len(cams)} cameras for plate-pixel yield\n")

    stats: dict[str, dict] = {}

    for cam in cams:
        # Prefer the requested daylight time; fall back to any clip.
        clip = clip_root / cam / f"{cam}_{args.time}.mp4"
        if not clip.is_file():
            found = sorted(clip_root.joinpath(cam).glob("*.mp4"))
            if not found:
                continue
            clip = found[0]

        cap = cv2.VideoCapture(str(clip))
        total = cap.get(cv2.CAP_PROP_FRAME_COUNT)
        if total <= 0:
            cap.release()
            continue
        step = max(1, int(total / max(args.frames_per_cam, 1)))

        widths4: list[int] = []
        widths2: list[int] = []
        idx = taken = 0
        while taken < args.frames_per_cam:
            ok, frame = cap.read()
            if not ok:
                break
            idx += 1
            if idx % step:
                continue
            taken += 1
            r = model(frame, imgsz=cfg["processing"]["input_width"],
                      conf=args.conf, verbose=False, quantize=16)[0]
            if r.boxes is None:
                continue
            for b in r.boxes:
                c = int(b.cls[0])
                if c not in FOURWHEEL and c != MOTO:
                    continue
                x1, _, x2, _ = b.xyxy[0].tolist()
                w = int(x2 - x1)
                (widths4 if c in FOURWHEEL else widths2).append(w)
        cap.release()

        if not widths4 and not widths2:
            continue
        a4 = np.array(widths4) if widths4 else np.array([0])
        per_thresh = {}
        for label, plate_px in THRESHOLDS:
            need = veh_width_for_plate(plate_px)
            n = int((a4 >= need).sum())
            per_thresh[label] = {
                "plate_px": plate_px,
                "veh_px_needed": need,
                "n": n,
                "pct": n / max(len(widths4), 1) * 100,
            }
        stats[cam] = {
            "clip": clip.name,
            "n_4wheel": len(widths4),
            "n_moto": len(widths2),
            "median": float(np.median(a4)),
            "median_plate_px": float(np.median(a4)) * PLATE_FRAC,
            "p90": float(np.percentile(a4, 90)),
            "p99": float(np.percentile(a4, 99)),
            "max": int(a4.max()),
            "thresholds": per_thresh,
        }
        s = stats[cam]
        cells = "  ".join(
            f"{lbl.split()[0]}:{per_thresh[lbl]['pct']:>5.1f}%"
            for lbl, _ in THRESHOLDS)
        print(f"  {cam:<8} n={s['n_4wheel']:<5} med={s['median']:>5.0f}px "
              f"(plate {s['median_plate_px']:>4.0f}px) max={s['max']:>5} | {cells}",
              flush=True)

    if not stats:
        print("no cameras yielded vehicles")
        return

    OUT_JSON.parent.mkdir(parents=True, exist_ok=True)
    OUT_JSON.write_text(json.dumps(stats, indent=2))

    vendor = THRESHOLDS[0][0]
    ranked = sorted(stats.items(),
                    key=lambda kv: kv[1]["thresholds"][vendor]["pct"],
                    reverse=True)
    print("\n" + "=" * 74)
    print("ANPR VIABILITY RANKING (% of 4-wheelers yielding a usable plate)")
    print("=" * 74)
    print(f"  {'camera':<9} " + "  ".join(f"{l.split()[0]:>10}" for l, _ in THRESHOLDS)
          + "   max plate")
    for cam, s in ranked[:12]:
        cells = "  ".join(f"{s['thresholds'][l]['pct']:>9.1f}%" for l, _ in THRESHOLDS)
        print(f"  {cam:<9} {cells}   {s['max']*PLATE_FRAC:>6.0f}px")

    tot4 = sum(s["n_4wheel"] for s in stats.values())
    print(f"\nfleet-wide ({tot4} four-wheelers sampled):")
    for label, plate_px in THRESHOLDS:
        n = sum(s["thresholds"][label]["n"] for s in stats.values())
        print(f"  {label:<26} {n:>5} ({n/max(tot4,1)*100:>4.1f}%)  "
              f"needs vehicle >= {veh_width_for_plate(plate_px)}px")

    print("\nNEXT: point the cascade at the top-ranked cameras, and use")
    print("multi-frame fusion to pull the 50-70px plate rows up to readable.")
    print(f"saved -> {OUT_JSON}")


if __name__ == "__main__":
    main()
