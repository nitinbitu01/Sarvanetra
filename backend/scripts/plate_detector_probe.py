"""backend/scripts/plate_detector_probe.py — does the trained detector find plates
the bootstrap MISSED?

THE TEST THAT MATTERS
  plate_v1 scored mAP50 0.992 on its validation split. That number is close to
  meaningless on its own: the labels came from the bootstrap pipeline, and the
  val split is drawn from the same cameras and conditions as the training set.
  A model that perfectly reproduces the bootstrap - including everywhere the
  bootstrap was blind - scores exactly that.

  The honest question is whether the detector GENERALISES. The label harvest
  was dominated by one camera:

      CAM_09 276 boxes | CAM_10 38 | CAM_07 37 | CAM_08 34
      CAM_06 7 | CAM_02 5 | CAM_04 4 | CAM_01 2 | CAM_03 1 | CAM_05 0

  So probe the STARVED cameras. If the detector fires on CAM_05 - which
  contributed no training labels at all - it has learned what a plate looks
  like rather than where CAM_09 puts them. That is what makes the bootstrap
  loop work: round 2 harvests from cameras round 1 could not see.

WHAT IS REPORTED
  Detection rate per camera on crops the bootstrap never labelled, plus
  annotated images. The images are the evidence - a detection rate means
  nothing until you have looked at what it is firing on.

USAGE
  python -m backend.scripts.plate_detector_probe
  python -m backend.scripts.plate_detector_probe --cameras CAM_05,CAM_01 --per-cam 60
"""
from __future__ import annotations

import argparse
import json
from collections import defaultdict
from pathlib import Path

import cv2
import numpy as np

CORPUS = Path("output/plate_corpus")
MANIFEST = CORPUS / "manifest.jsonl"
OUT = Path("output/plate_probe")
WEIGHTS = "runs/detect/runs/plate/plate_v1/weights/best.pt"

# Cameras that contributed almost nothing to training, worst first.
STARVED = ["CAM_05", "CAM_03", "CAM_01", "CAM_04", "CAM_02", "CAM_06"]


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--weights", default=WEIGHTS)
    ap.add_argument("--cameras", default=",".join(STARVED))
    ap.add_argument("--per-cam", type=int, default=80,
                    help="Best crops per camera to probe (one per track).")
    ap.add_argument("--conf", type=float, default=0.25)
    ap.add_argument("--min-plate-px", type=float, default=45.0)
    ap.add_argument("--save-per-cam", type=int, default=8)
    args = ap.parse_args()

    if not Path(args.weights).is_file():
        raise SystemExit(f"weights not found: {args.weights}")

    rows = [json.loads(l) for l in MANIFEST.open(encoding="utf-8")]
    best: dict[tuple, dict] = {}
    for r in rows:
        k = (r["camera"], r["clip"], r["track"])
        s = r["sharpness"] * (r["plate_px_est"] ** 0.5)
        if k not in best or s > best[k]["_s"]:
            best[k] = {**r, "_s": s}

    cams = [c.strip() for c in args.cameras.split(",") if c.strip()]
    from ultralytics import YOLO
    model = YOLO(args.weights)
    OUT.mkdir(parents=True, exist_ok=True)
    print(f"detector : {args.weights}")
    print(f"probing  : {cams}\n", flush=True)

    stats = defaultdict(lambda: {"n": 0, "hit": 0, "boxes": 0})
    widths = defaultdict(list)

    for cam in cams:
        pool = [r for r in best.values()
                if r["camera"] == cam and r["plate_px_est"] >= args.min_plate_px]
        pool.sort(key=lambda r: -r["plate_px_est"])
        pool = pool[:args.per_cam]
        saved = 0
        for r in pool:
            img = cv2.imread(r["path"])
            if img is None:
                continue
            stats[cam]["n"] += 1
            res = model(img, imgsz=640, conf=args.conf, verbose=False)[0]
            H, W = img.shape[:2]
            # GEOMETRIC GATE at inference. A plate is 0.12-0.48 of the vehicle
            # it is bolted to (500mm on an 1800-2500mm body) and is always
            # wide, never square. Detector v1 fired on the burned-in station
            # caption at ~0.35 of the FRAME width with the wrong aspect - this
            # rejects that class of error even if the model still proposes it,
            # so a caption can never reach the recogniser.
            keep = []
            for b in (res.boxes or []):
                x1, y1, x2, y2 = (int(v) for v in b.xyxy[0].tolist())
                ratio = (x2 - x1) / max(W, 1)
                ar = (x2 - x1) / max(y2 - y1, 1)
                if 0.12 <= ratio <= 0.48 and 1.8 <= ar <= 7.0:
                    keep.append((x1, y1, x2, y2, float(b.conf[0])))
                else:
                    stats[cam]["gated"] = stats[cam].get("gated", 0) + 1
            nb = len(keep)
            if nb:
                stats[cam]["hit"] += 1
                stats[cam]["boxes"] += nb
                vis = img.copy()
                for x1, y1, x2, y2, cf in keep:
                    widths[cam].append(x2 - x1)
                    cv2.rectangle(vis, (x1, y1), (x2, y2), (0, 255, 0), 2)
                    cv2.putText(vis, f"{cf:.2f}", (x1, max(12, y1 - 4)),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.45, (0, 255, 0), 1)
                if saved < args.save_per_cam:
                    cv2.imwrite(str(OUT / f"{cam}_{saved:02d}_"
                                    f"{r['plate_px_est']:.0f}px.jpg"), vis)
                    saved += 1
        d = stats[cam]
        w = widths[cam]
        wtxt = (f"  box w median {np.median(w):.0f}px" if w else "")
        print(f"  {cam:<8} {d['hit']:>3}/{d['n']:<4} "
              f"({d['hit']/max(d['n'],1)*100:>5.1f}%)  "
              f"{d['boxes']} boxes  {d.get('gated',0)} gated{wtxt}", flush=True)

    tot_n = sum(d["n"] for d in stats.values())
    tot_h = sum(d["hit"] for d in stats.values())
    print("\n" + "=" * 58)
    print(f"starved cameras: {tot_h}/{tot_n} crops with a detection "
          f"({tot_h/max(tot_n,1)*100:.1f}%)")
    print("=" * 58)
    print("\nBootstrap OCR found almost nothing on these cameras. If this rate")
    print("is materially higher, the detector generalised and round 2 of the")
    print("loop will rebalance the training set.")
    print(f"\nannotated crops -> {OUT}  -- LOOK AT THESE. A high rate on")
    print("bumpers and badges is worse than a low rate on plates.")


if __name__ == "__main__":
    main()
