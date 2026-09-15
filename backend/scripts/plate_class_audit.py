"""backend/scripts/plate_class_audit.py — does the plate detector work on trucks
and buses, or only on cars?

WHY THIS IS MEASURED RATHER THAN INFERRED
  The detector's training set contains 715 car images and 5 truck images, which
  looks like a decisive answer: it has barely seen a truck. But 54 of the
  human-labelled plates ARE on trucks, and those crops were cut by this same
  detector during harvesting - so it evidently fires on trucks sometimes.

  Those two facts cannot both be summarised by a guess. Trucks and buses are
  2,386 of the 5,594 mined vehicles - 43% of the traffic - so whether the
  detector covers them decides whether nearly half the pipeline's input is
  being silently dropped. That is worth ten minutes of measurement.

WHAT IS REPORTED
  Detection rate per vehicle class on vehicles the detector was never trained
  on, plus the plate width it finds, because a class where plates are found but
  are systematically smaller will fail later at the recogniser instead - a
  different problem with a different fix.

USAGE
  python -m backend.scripts.plate_class_audit --per-class 150
"""
from __future__ import annotations

import argparse
import json
import random
import re
from collections import defaultdict
from pathlib import Path

CORPUS = Path("output/plate_corpus")
DET = Path("data/plate_detect_v2/images")
WEIGHTS = Path("runs/detect/runs/plate_v3/recovered/weights/best.pt")
NAMES = {0: "person", 1: "car", 2: "motorcycle", 3: "bus", 4: "truck"}
NAME_RE = re.compile(r"^(CAM[_-][\w-]+?)_t(\d+)_")


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--per-class", type=int, default=150,
                    help="Vehicles sampled per class.")
    ap.add_argument("--frames", type=int, default=5,
                    help="Sharpest frames tried per vehicle, as the harvest "
                         "does - judging on one random frame would understate "
                         "every class equally but by different amounts.")
    ap.add_argument("--conf", type=float, default=0.30)
    ap.add_argument("--weights", default=str(WEIGHTS))
    args = ap.parse_args()

    # Vehicles already used to train the detector, excluded from the sample.
    trained = set()
    if DET.is_dir():
        for split in ("train", "val"):
            for p in (DET / split).glob("*.jpg"):
                m = NAME_RE.match(p.stem)
                if m:
                    trained.add((m.group(1), int(m.group(2))))
    print(f"detector trained on : {len(trained)} vehicles")

    tracks = defaultdict(list)
    cls_of = {}
    for line in (CORPUS / "manifest.jsonl").open(encoding="utf-8"):
        r = json.loads(line)
        k = (r["camera"], r["track"])
        if k in trained:
            continue
        tracks[k].append(r)
        cls_of[k] = r.get("cls")

    by_cls = defaultdict(list)
    for k, rows in tracks.items():
        by_cls[cls_of[k]].append((k, rows))
    print("unseen vehicles available:",
          {NAMES.get(c, c): len(v) for c, v in sorted(by_cls.items())})

    from ultralytics import YOLO
    model = YOLO(args.weights)
    print(f"detector            : {args.weights}\n", flush=True)

    rng = random.Random(11)
    results = {}
    for cls in sorted(by_cls):
        pool = by_cls[cls]
        rng.shuffle(pool)
        pool = pool[:args.per_class]
        hits = 0
        widths = []
        for k, rows in pool:
            rows.sort(key=lambda r: -(r.get("sharpness", 0) ** 0.5
                                      * r.get("plate_px_est", 0)))
            found = False
            for fr in rows[:args.frames]:
                res = model.predict(fr["path"], conf=args.conf, verbose=False)[0]
                if not len(res.boxes):
                    continue
                b = res.boxes.xyxy.cpu().numpy()
                w = (b[:, 2] - b[:, 0])
                h = (b[:, 3] - b[:, 1])
                keep = [(ww, hh) for ww, hh in zip(w, h)
                        if ww >= 34 and hh >= 9 and 1.8 <= ww / max(hh, 1) <= 7.5]
                if keep:
                    found = True
                    widths.append(max(ww for ww, _ in keep))
                    break
            hits += found
        results[cls] = (len(pool), hits, widths)
        print(f"  {NAMES.get(cls, cls):<12} sampled {len(pool):>4}  "
              f"found {hits:>4}  rate {hits/max(len(pool),1)*100:>5.1f}%",
              flush=True)

    print("\n" + "=" * 62)
    print("PLATE DETECTION BY VEHICLE CLASS  (vehicles never trained on)")
    print("=" * 62)
    print(f"{'class':<12} {'sampled':>8} {'found':>7} {'rate':>7} "
          f"{'median width':>13}")
    print("-" * 62)
    for cls, (n, hits, widths) in sorted(results.items()):
        widths.sort()
        med = widths[len(widths) // 2] if widths else 0
        print(f"{NAMES.get(cls, cls):<12} {n:>8} {hits:>7} "
              f"{hits/max(n,1)*100:>6.1f}% {med:>12.0f}px")
    print("=" * 62)
    car = results.get(1, (1, 0, []))
    car_rate = car[1] / max(car[0], 1)
    for cls in (4, 3):
        if cls not in results:
            continue
        n, hits, _ = results[cls]
        r = hits / max(n, 1)
        gap = (car_rate - r) * 100
        print(f"\n{NAMES[cls]} vs car: {gap:+.1f} points")
        if gap > 15:
            print(f"  The detector materially under-covers {NAMES[cls]}s. "
                  f"Adding\n  {NAMES[cls]} plate boxes to its training set is "
                  f"the fix, not a\n  recogniser change.")
        else:
            print(f"  Comparable to cars - {NAMES[cls]}s are NOT being "
                  f"systematically\n  missed, whatever the training mix "
                  f"suggests.")


if __name__ == "__main__":
    main()
