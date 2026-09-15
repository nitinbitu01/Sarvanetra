"""backend/scripts/expand_crops_detector.py — multiply the labelled set by
re-cutting every frame of each labelled vehicle with the plate detector.

THE ARITHMETIC THIS EXPLOITS
  A human reads one crop and supplies one string. But the corpus holds up to
  25 frames of that same vehicle, and the plate in every one of them carries
  the SAME string - already known, at no further human cost. Each frame is a
  genuinely different view: different distance, angle, motion blur, exposure.
  That is the variation a recogniser needs, and it is free.

WHY THIS SUPERSEDES expand_labelled_crops.py
  The earlier version located plates with EasyOCR and used only its box. That
  worked, but EasyOCR reports a TEXT region, which is not a plate region: it
  clips the outer glyph where contrast falls away, and it wanders onto the
  dealer sticker. Worse, it only reports at all when it thinks it can read
  something, so the blurriest frames of a vehicle - the ones that would teach
  the recogniser most - were silently skipped.

  The detector fires on plate shape regardless of legibility, so the hard
  frames come through, and its box is the plate's own extent, so framing is
  consistent between training crops and what the live pipeline will produce.

THE SPLIT MUST STILL BE BY VEHICLE
  Every view produced here inherits its vehicle's (camera, track). Training
  code groups by that key, so all views of one plate land on the same side of
  the train/test split. Without that the model would train on one frame of a
  plate and be tested on another frame of the same plate - memorisation
  scored as accuracy, which has already invalidated one detector run in this
  project.

USAGE
  python -m backend.scripts.expand_crops_detector
  python -m backend.scripts.expand_crops_detector --max-per-track 12
"""
from __future__ import annotations

import argparse
import json
from collections import defaultdict
from pathlib import Path

import cv2

CORPUS = Path("output/plate_corpus")
REAL = Path("data/plate_real")
WEIGHTS = Path("runs/detect/runs/plate_v3/recovered/weights/best.pt")


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--weights", default=str(WEIGHTS))
    ap.add_argument("--conf", type=float, default=0.30)
    ap.add_argument("--min-width", type=float, default=32.0,
                    help="Lower than the harvest floor on purpose: the label "
                         "is already known here, so a crop too blurry for a "
                         "person to read is still a valid training example.")
    ap.add_argument("--max-per-track", type=int, default=12,
                    help="Cap views per vehicle. Beyond this the extra frames "
                         "are near-duplicates and let one plate dominate.")
    ap.add_argument("--verified", default="verified_all.jsonl")
    args = ap.parse_args()

    truth = {}
    for line in (REAL / args.verified).open(encoding="utf-8"):
        v = json.loads(line)
        if v.get("text"):
            truth[(v["camera"], v["track"])] = v["text"]
    print(f"labelled vehicles      : {len(truth)}")

    have: dict[tuple, set] = defaultdict(set)
    lab_path = REAL / "labels.jsonl"
    if lab_path.is_file():
        for line in lab_path.open(encoding="utf-8"):
            r = json.loads(line)
            have[(r["camera"], r["track"])].add(r["file"])
    print(f"crops already in set   : {sum(len(v) for v in have.values())}")

    frames: dict[tuple, list] = defaultdict(list)
    for line in (CORPUS / "manifest.jsonl").open(encoding="utf-8"):
        r = json.loads(line)
        k = (r["camera"], r["track"])
        if k in truth:
            frames[k].append(r)
    print(f"corpus frames available: {sum(len(v) for v in frames.values())}\n",
          flush=True)

    from ultralytics import YOLO
    model = YOLO(args.weights)

    out = lab_path.open("a", encoding="utf-8")
    added = 0
    for i, (key, rows) in enumerate(sorted(frames.items()), 1):
        cam, track = key
        text = truth[key]
        # Sharpest first, so the per-vehicle cap keeps the most informative
        # views rather than whichever frames happen to come first.
        rows.sort(key=lambda r: -r.get("sharpness", 0))
        kept = len(have[key])
        for r in rows:
            if kept >= args.max_per_track:
                break
            name = f"{cam}_t{track}_x{Path(r['path']).stem}.jpg"
            if name in have[key]:
                continue
            img = cv2.imread(r["path"])
            if img is None:
                continue
            res = model.predict(img, conf=args.conf, verbose=False)[0]
            if not len(res.boxes):
                continue
            h, w = img.shape[:2]
            best = None
            for b in res.boxes.xyxy.cpu().numpy():
                x1, y1, x2, y2 = b
                bw, bh = x2 - x1, y2 - y1
                if bw < args.min_width or bh < 8:
                    continue
                if not (1.8 <= bw / max(bh, 1) <= 7.5):
                    continue
                if best is None or bw > best[0]:
                    best = (bw, (x1, y1, x2, y2))
            if best is None:
                continue
            x1, y1, x2, y2 = best[1]
            # Same margins as the harvest, so expanded views and harvested
            # crops are framed identically - a mismatch here would show up as
            # the model performing differently on the two sources.
            mx, my = (x2 - x1) * 0.08, (y2 - y1) * 0.20
            crop = img[int(max(0, y1 - my)):int(min(h, y2 + my)),
                       int(max(0, x1 - mx)):int(min(w, x2 + mx))]
            if crop.size == 0 or crop.shape[1] < 24:
                continue
            cv2.imwrite(str(REAL / "images" / name), crop,
                        [cv2.IMWRITE_JPEG_QUALITY, 96])
            out.write(json.dumps({
                "file": name, "text": text, "state": "",
                "agreement": 1.0, "votes": 1,
                "camera": cam, "track": track, "expanded": True,
            }) + "\n")
            have[key].add(name)
            kept += 1
            added += 1
        if i % 50 == 0:
            print(f"  {i}/{len(frames)} vehicles  |  +{added} views",
                  flush=True)
    out.close()

    per = {k: len(v) for k, v in have.items() if k in truth}
    n = max(len(per), 1)
    total = sum(per.values())
    print("\n" + "=" * 58)
    print(f"views added    : {added}")
    print(f"training crops : {total} across {len(per)} vehicles "
          f"({total/n:.1f} per vehicle)")
    print(f"single-view    : {sum(1 for v in per.values() if v == 1)} vehicles")
    print("=" * 58)
    print("\nNo new labels were created - these are additional VIEWS of plates")
    print("a human already read. Views inherit their vehicle key, so the")
    print("train/test split by vehicle keeps every view of one plate on one"
          " side.")


if __name__ == "__main__":
    main()
