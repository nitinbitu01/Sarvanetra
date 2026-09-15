"""backend/scripts/plate_traffic_survey.py — camera ranking that counts only
DISTINCT MOVING vehicles.

WHY THE FIRST SURVEY WAS WRONG
  plate_camera_survey.py sampled every ~750th frame and counted every vehicle
  box it saw. That silently counted PARKED cars once per sample. CAM_27 was
  ranked the best camera in the fleet at 51.8% - diagnosis showed a single
  stationary vehicle, box width identical (376px) across 400 consecutive
  frames, counted 40 times. Several other "high yield" cameras are likely
  inflated the same way.

WHAT THIS DOES DIFFERENTLY
  1. Processes CONSECUTIVE frames so BoT-SORT can maintain track identity
     (sparse sampling makes tracking impossible - that is why the original
     survey could not have done this).
  2. Requires each track to TRAVEL before counting it, rejecting parked cars.
  3. Counts each vehicle ONCE, using its best (largest) observed width, rather
     than once per frame it appears in.

  The result answers the question that actually matters for ANPR: how many
  distinct vehicles pass this camera, and how many of them get close enough
  to carry a readable plate.

USAGE
  python -m backend.scripts.plate_traffic_survey
  python -m backend.scripts.plate_traffic_survey --frames 2000
"""
from __future__ import annotations

import argparse
import json
import os
from pathlib import Path

os.environ.setdefault("OPENCV_FFMPEG_CAPTURE_OPTIONS",
                      "timeout;30000000|stimeout;30000000|rw_timeout;30000000")

import numpy as np
import yaml

from backend.scripts.plate_mine_corpus import MotionGate, PLATE_FRAC

FOURWHEEL = {1, 3, 4}
OUT_JSON = Path("output/plate_traffic_survey.json")
THRESHOLDS = [("100px", 100), ("70px", 70), ("50px", 50)]


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--frames", type=int, default=1500,
                    help="Consecutive frames per camera (~1 min at 25fps). "
                         "Must be consecutive for tracking to work.")
    ap.add_argument("--time", default="0830")
    ap.add_argument("--imgsz", type=int, default=640)
    ap.add_argument("--conf", type=float, default=0.35)
    ap.add_argument("--min-travel-px", type=float, default=80.0)
    args = ap.parse_args()

    cfg = yaml.safe_load(open("config.yaml", encoding="utf-8"))
    from ultralytics import YOLO
    model = YOLO(cfg["model"]["path"])

    clip_root = Path("data/clips")
    cams = sorted(d.name for d in clip_root.iterdir() if d.is_dir())
    print(f"surveying {len(cams)} cameras, {args.frames} consecutive frames each")
    print("counting DISTINCT MOVING vehicles (parked cars rejected)\n", flush=True)

    stats: dict[str, dict] = {}

    for cam in cams:
        clip = clip_root / cam / f"{cam}_{args.time}.mp4"
        if not clip.is_file():
            found = sorted((clip_root / cam).glob("*.mp4"))
            if not found:
                continue
            clip = found[0]

        gate = MotionGate(args.min_travel_px)
        best_w: dict[int, float] = {}      # track id -> largest width seen
        n = 0
        try:
            stream = model.track(source=str(clip), stream=True, persist=True,
                                 tracker="botsort.yaml", conf=args.conf,
                                 imgsz=args.imgsz, verbose=False, quantize=16)
            for r in stream:
                n += 1
                if n > args.frames:
                    break
                if r.boxes is None or r.boxes.id is None:
                    continue
                for b in r.boxes:
                    if int(b.cls[0]) not in FOURWHEEL:
                        continue
                    tid = int(b.id[0])
                    x1, y1, x2, y2 = b.xyxy[0].tolist()
                    if not gate.passes(tid, (x1 + x2) / 2, (y1 + y2) / 2):
                        continue
                    w = x2 - x1
                    best_w[tid] = max(best_w.get(tid, 0.0), w)
        except Exception as e:                        # noqa: BLE001
            print(f"  {cam:<8} ERROR: {e}", flush=True)
            continue

        moving = len(best_w)
        if moving == 0:
            print(f"  {cam:<8} 0 moving vehicles "
                  f"({gate.n_seen} stationary rejected)", flush=True)
            stats[cam] = {"moving": 0, "stationary": gate.n_seen}
            continue

        arr = np.array(list(best_w.values()))
        per = {}
        for label, plate_px in THRESHOLDS:
            need = plate_px / PLATE_FRAC
            k = int((arr >= need).sum())
            per[label] = {"n": k, "pct": k / moving * 100}
        stats[cam] = {
            "moving": moving,
            "stationary": gate.n_seen - gate.n_moving,
            "median_w": float(np.median(arr)),
            "max_w": float(arr.max()),
            "thresholds": per,
        }
        cells = "  ".join(f"{l}:{per[l]['n']:>3}({per[l]['pct']:>4.0f}%)"
                          for l, _ in THRESHOLDS)
        print(f"  {cam:<8} {moving:>4} moving "
              f"({stats[cam]['stationary']:>3} parked)  "
              f"med={stats[cam]['median_w']:>4.0f}px | {cells}", flush=True)

    OUT_JSON.parent.mkdir(parents=True, exist_ok=True)
    OUT_JSON.write_text(json.dumps(stats, indent=2))

    live = {c: s for c, s in stats.items() if s.get("moving")}
    ranked = sorted(live.items(),
                    key=lambda kv: kv[1]["thresholds"]["70px"]["n"], reverse=True)
    print("\n" + "=" * 70)
    print("TRUE ANPR RANKING - by COUNT of distinct moving vehicles >=70px plate")
    print("(count matters more than percentage: a camera with 2 vehicles at")
    print(" 100% yields 2 plates; one with 200 at 20% yields 40)")
    print("=" * 70)
    for cam, s in ranked[:15]:
        t = s["thresholds"]
        print(f"  {cam:<8} {t['70px']['n']:>4} vehicles >=70px  "
              f"({t['70px']['pct']:>4.0f}% of {s['moving']:>4} moving)  "
              f"100px: {t['100px']['n']:>3}")

    tot_moving = sum(s["moving"] for s in live.values())
    for label, _ in THRESHOLDS:
        k = sum(s["thresholds"][label]["n"] for s in live.values())
        print(f"\nfleet {label:>5} plate: {k} of {tot_moving} moving vehicles "
              f"({k/max(tot_moving,1)*100:.1f}%)")
    print(f"\nPer {args.frames} frames (~1 min) per camera. Multiply by clip")
    print("length and clip count to project total corpus size.")
    print(f"saved -> {OUT_JSON}")


if __name__ == "__main__":
    main()
