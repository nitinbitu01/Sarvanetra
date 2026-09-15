"""backend/scripts/plate_harvest_labels.py — turn OCR reads into a plate-detector
training set.

THE BOOTSTRAP LOOP
  Phase 2 localises plates with a GEOMETRIC GUESS - crop the lower 55% of the
  vehicle and hope the plate is in there. That is why coverage sits at 12.5%:
  the recogniser is handed bumper, badge, shadow and road along with the plate,
  and small glyphs do not survive that.

  A trained detector fixes it, but training needs labelled boxes and nobody
  wants to draw thousands by hand. The way out: WHEN OCR SUCCEEDS, IT HAS
  PROVEN WHERE THE PLATE IS. A read that decodes to a structurally valid Indian
  plate is a high-precision label - wrong text can still be a right box, and
  for a DETECTOR only the box matters.

  So: run the weak pipeline over everything, keep the boxes it earns, train a
  real detector on them, and the detector then finds the plates the guess
  missed. Each round labels the next.

WHY THIS BEATS HAND-LABELLING HERE
  Hand-labelling 2,000 plates is ~2 days of tedious work and produces exactly
  2,000 boxes. This produces boxes across every camera, distance, angle and
  lighting condition in the corpus, for the cost of GPU time, and it is
  unbiased by whichever crops a human found easy to see.

PRECISION OVER RECALL, DELIBERATELY
  Only reads that decode to valid plate grammar are kept. Missing a plate here
  costs nothing - the detector will learn it from its neighbours. A WRONG box
  teaches the detector to fire on bumpers, and that error compounds through
  every later stage.

OUTPUT
  data/plate_detect/{images,labels}/{train,val}  - YOLO format, single class.
  Boxes are mapped back from the upscaled band into ORIGINAL crop coordinates,
  because that is the frame the detector will run on in production.

USAGE
  python -m backend.scripts.plate_harvest_labels
  python -m backend.scripts.plate_harvest_labels --max-frames 12 --max-tracks 0
"""
from __future__ import annotations

import argparse
import json
import random
import shutil
from collections import defaultdict
from pathlib import Path

import cv2

from backend.scripts.indian_plate_grammar import decode_plate

CORPUS = Path("output/plate_corpus")
MANIFEST = CORPUS / "manifest.jsonl"
DATASET = Path("data/plate_detect")
ALLOW = "ABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789"
OSD = {"BRIDGE", "BHAI", "CHIMAN", "CSITMS", "CSI", "BRIDC", "GSRTC",
       "CARRIAGE", "EXPERIENCE", "SOMNATH", "CHITRA"}

BAND_Y0, BAND_Y1 = 0.42, 0.97
BAND_X0, BAND_X1 = 0.08, 0.92
BAND_MIN_W = 360


def band_and_scale(img):
    """Return (band, x_off, y_off, scale) so boxes can be mapped back."""
    h, w = img.shape[:2]
    x0, y0 = int(w * BAND_X0), int(h * BAND_Y0)
    b = img[y0:int(h * BAND_Y1), x0:int(w * BAND_X1)]
    if b.size == 0 or b.shape[1] < 20:
        return None, 0, 0, 1.0
    s = 1.0
    if b.shape[1] < BAND_MIN_W:
        s = BAND_MIN_W / b.shape[1]
        b = cv2.resize(b, None, fx=s, fy=s, interpolation=cv2.INTER_CUBIC)
    return b, x0, y0, s


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--min-plate-px", type=float, default=45.0)
    ap.add_argument("--max-tracks", type=int, default=0, help="0 = all")
    ap.add_argument("--max-frames", type=int, default=12,
                    help="Frames per track to attempt. Labels need coverage "
                         "of conditions, not every frame of every vehicle.")
    ap.add_argument("--conf", type=float, default=0.12)
    ap.add_argument("--val-frac", type=float, default=0.15)
    ap.add_argument("--include-heavy", action="store_true", default=True,
                    help="Keep buses/trucks: their plates are rarer but the "
                         "detector must handle them, and a box is a box.")
    args = ap.parse_args()

    rows = [json.loads(l) for l in MANIFEST.open(encoding="utf-8")]
    tracks: dict[tuple, list[dict]] = defaultdict(list)
    for r in rows:
        tracks[(r["camera"], r["clip"], r["track"])].append(r)

    elig = []
    for k, frames in tracks.items():
        best = max(frames, key=lambda r: r["plate_px_est"])
        if best["plate_px_est"] < args.min_plate_px:
            continue
        if not args.include_heavy and best["cls"] != 1:
            continue
        frames.sort(key=lambda r: -(r["sharpness"] * r["plate_px_est"]))
        elig.append((k, frames[:args.max_frames]))
    elig.sort(key=lambda t: -max(f["plate_px_est"] for f in t[1]))
    if args.max_tracks:
        elig = elig[:args.max_tracks]

    total = sum(len(f) for _, f in elig)
    print(f"tracks eligible : {len(elig)}")
    print(f"frames to scan  : {total}")
    print("keeping only boxes whose text decodes to valid plate grammar\n",
          flush=True)

    import easyocr
    print("loading EasyOCR...", flush=True)
    reader = easyocr.Reader(["en"], gpu=True, verbose=False)

    for split in ("train", "val"):
        for kind in ("images", "labels"):
            d = DATASET / kind / split
            if d.exists():
                shutil.rmtree(d)
            d.mkdir(parents=True, exist_ok=True)

    random.seed(1337)
    n_lab = n_img = 0
    per_cam: dict[str, int] = defaultdict(int)
    sizes = []

    for i, (key, frames) in enumerate(elig, 1):
        for fr in frames:
            img = cv2.imread(fr["path"])
            if img is None:
                continue
            band, xo, yo, s = band_and_scale(img)
            if band is None:
                continue
            boxes = []
            for box, txt, c in reader.readtext(band, allowlist=ALLOW,
                                               batch_size=8):
                if c < args.conf:
                    continue
                cleaned = "".join(ch for ch in txt.upper() if ch.isalnum())
                if len(cleaned) < 8 or any(t in cleaned for t in OSD):
                    continue
                d = decode_plate(cleaned)
                if not (d["plate"] and d["score"] >= 0.85):
                    continue
                xs = [p[0] for p in box]
                ys = [p[1] for p in box]
                # Map the band-space box back to ORIGINAL crop coordinates:
                # undo the upscale, then re-add the band offset.
                x1 = xo + min(xs) / s
                x2 = xo + max(xs) / s
                y1 = yo + min(ys) / s
                y2 = yo + max(ys) / s
                H, W = img.shape[:2]
                # Pad slightly: OCR boxes hug the glyphs, a detector should
                # learn the plate PLATE including its border.
                pw, ph = (x2 - x1) * 0.06, (y2 - y1) * 0.18
                x1, x2 = max(0, x1 - pw), min(W, x2 + pw)
                y1, y2 = max(0, y1 - ph), min(H, y2 + ph)
                if x2 - x1 < 12 or y2 - y1 < 5:
                    continue
                # GEOMETRIC SANITY - a plate is a fixed fraction of the
                # vehicle that carries it (500mm plate on an 1800-2500mm
                # vehicle). Detector v1 was trained without this and learned
                # to fire on the burned-in station caption at 350px wide on a
                # ~1000px vehicle - four times any real plate. The ratio is
                # physics, so enforce it at label time and the error cannot be
                # learned in the first place.
                ratio = (x2 - x1) / max(W, 1)
                if not (0.12 <= ratio <= 0.48):
                    continue
                # Plates are wide, never tall or square.
                ar = (x2 - x1) / max(y2 - y1, 1)
                if not (1.8 <= ar <= 7.0):
                    continue
                boxes.append((x1, y1, x2, y2))
                sizes.append(x2 - x1)

            if not boxes:
                continue
            split = "val" if random.random() < args.val_frac else "train"
            stem = f"{key[0]}_{key[1]}_t{key[2]}_f{fr['frame']}"
            cv2.imwrite(str(DATASET / "images" / split / f"{stem}.jpg"), img,
                        [cv2.IMWRITE_JPEG_QUALITY, 95])
            H, W = img.shape[:2]
            with (DATASET / "labels" / split / f"{stem}.txt").open("w") as f:
                for x1, y1, x2, y2 in boxes:
                    f.write(f"0 {(x1+x2)/2/W:.6f} {(y1+y2)/2/H:.6f} "
                            f"{(x2-x1)/W:.6f} {(y2-y1)/H:.6f}\n")
            n_img += 1
            n_lab += len(boxes)
            per_cam[key[0]] += len(boxes)

        if i % 100 == 0:
            print(f"  {i}/{len(elig)} tracks | {n_img} images, "
                  f"{n_lab} boxes", flush=True)

    yaml_p = DATASET / "plate.yaml"
    yaml_p.write_text(
        f"path: {DATASET.resolve().as_posix()}\n"
        "train: images/train\nval: images/val\n\nnc: 1\nnames: [plate]\n")

    print("\n" + "=" * 58)
    print(f"images : {n_img}")
    print(f"boxes  : {n_lab}")
    if sizes:
        import numpy as np
        a = np.array(sizes)
        print(f"box width: median {np.median(a):.0f}px  "
              f"p10 {np.percentile(a,10):.0f}  p90 {np.percentile(a,90):.0f}")
    print(f"per camera: {dict(sorted(per_cam.items(), key=lambda kv:-kv[1]))}")
    print(f"\ndataset -> {yaml_p}")
    print("\nNEXT: train YOLOv8n on this, then re-run reading with the trained")
    print("detector in place of the geometric guess.")


if __name__ == "__main__":
    main()
