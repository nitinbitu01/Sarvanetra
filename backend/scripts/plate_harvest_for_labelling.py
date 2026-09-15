"""backend/scripts/plate_harvest_for_labelling.py — collect plate crops for humans
to label, using OCR only to FIND plates, never to read them.

THE DISTINCTION THAT MATTERS
  EasyOCR is measured at 20% exact-match on these plates - too unreliable to
  produce labels. But it is good at LOCALISATION: it reliably reports WHERE a
  text region sits, even when it reads that region wrongly. Those are separate
  skills, and the earlier pseudo-label attempt failed by trusting the weak one.

  So this keeps every plate-SHAPED text region regardless of what OCR thinks it
  says, and hands the crop to a person. The earlier harvest discarded any crop
  whose text failed Indian plate grammar, which threw away perfectly legible
  plates simply because OCR misread them - 98 vehicles kept out of 2,009.

WHY THIS MATTERS NOW
  Phase A proved the pipeline: 65 hand-labelled vehicles took the recogniser
  from 23.1% to 76.9% exact-match on held-out plates. The only limit left is
  label count. Several hundred more should raise both the accuracy and the
  confidence in the figure - 13 held-out plates is a thin sample to quote.

GEOMETRIC FILTERING replaces grammar filtering
  A plate region is wide, not tall (aspect 2:1 to 7:1), sits in the lower part
  of the vehicle, and spans a sane fraction of it. Those constraints are
  physical and hold whatever OCR reads. They also reject the burned-in station
  caption, which is far wider than any plate.

USAGE
  python -m backend.scripts.plate_harvest_for_labelling --target 500
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
DONE = Path("data/plate_real/plate_labels_verified.jsonl")
OUT = Path("data/plate_label_batch2")
ALLOW = "ABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789"
OSD = {"BRIDGE", "BHAI", "CHIMAN", "CSITMS", "CSI", "BRIDC", "JANPATH", "PTZ"}


def band_of(img):
    h, w = img.shape[:2]
    b = img[int(h * 0.40):int(h * 0.98), int(w * 0.06):int(w * 0.94)]
    if b.size == 0 or b.shape[1] < 20:
        return None
    if b.shape[1] < 380:
        f = 380 / b.shape[1]
        b = cv2.resize(b, None, fx=f, fy=f, interpolation=cv2.INTER_CUBIC)
    return b


def plate_shaped(box, bw: int, bh: int, min_w: int = 40) -> bool:
    """Physical constraints only - no assumption about what the text says.

    min_w is loosened for large batches: a narrower crop is often still
    readable by a person, and a labeller can skip in three seconds. Being
    strict here cost yield - one pass kept only 122 of 2,901 candidates, and
    small label increments have measurably failed to move accuracy.
    """
    xs = [p[0] for p in box]
    ys = [p[1] for p in box]
    w, h = max(xs) - min(xs), max(ys) - min(ys)
    if w < min_w or h < 10:
        return False
    ar = w / max(h, 1)
    if not (2.0 <= ar <= 7.0):
        return False
    if w > bw * 0.80:                 # the station caption spans the frame
        return False
    if (min(ys) + max(ys)) / 2 < bh * 0.20:
        return False
    return True


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--target", type=int, default=500)
    ap.add_argument("--min-plate-px", type=float, default=45.0)
    ap.add_argument("--frames-per-track", type=int, default=6,
                    help="Only the sharpest few - one good crop per vehicle is "
                         "what a labeller needs.")
    ap.add_argument("--conf", type=float, default=0.05,
                    help="Very low: we want the BOX, not the reading.")
    ap.add_argument("--min-box-w", type=int, default=40,
                    help="Narrowest text region to keep, in band pixels. "
                         "Lower it for a bigger batch - unreadable ones cost "
                         "the labeller three seconds to skip, whereas a plate "
                         "never harvested is a training example lost.")
    ap.add_argument("--classes", default="1",
                    help="Detector class ids to accept. '1' = cars only; "
                         "'1,3,4' adds buses and trucks.")
    ap.add_argument("--exclude", default="",
                    help="Extra verified-label files whose vehicles to skip, "
                         "comma separated.")
    args = ap.parse_args()
    args.classes = {int(c) for c in args.classes.replace(",", " ").split()}

    already = set()
    srcs = [DONE] + [Path(p.strip()) for p in args.exclude.split(",") if p.strip()]
    for s in srcs:
        if not s.is_file():
            continue
        for l in s.open(encoding="utf-8"):
            v = json.loads(l)
            already.add((v["camera"], v["track"]))
    print(f"already labelled : {len(already)} vehicles (skipping these)")

    rows = [json.loads(l) for l in MANIFEST.open(encoding="utf-8")]
    tracks: dict[tuple, list[dict]] = defaultdict(list)
    for r in rows:
        tracks[(r["camera"], r["clip"], r["track"])].append(r)

    elig = []
    for k, frames in tracks.items():
        if (k[0], k[2]) in already:
            continue
        best = max(frames, key=lambda r: r["plate_px_est"])
        # Batch 3 widens the net: batch 2 scanned 1,944 vehicles for only 151
        # crops, so the car-only / 45px pool is close to exhausted. Buses and
        # trucks DO occasionally present a readable plate when they pass
        # front-on, and a labeller can skip the ones that do not - a skipped
        # crop costs three seconds, whereas a plate never harvested is a
        # training example lost for good.
        if best["cls"] not in args.classes or best["plate_px_est"] < args.min_plate_px:
            continue
        frames.sort(key=lambda r: -(r["sharpness"] * r["plate_px_est"]))
        elig.append((k, frames[:args.frames_per_track]))
    elig.sort(key=lambda t: -max(f["sharpness"] for f in t[1]))
    print(f"candidate vehicles: {len(elig)}\n", flush=True)

    import easyocr
    print("loading EasyOCR (localisation only)...", flush=True)
    reader = easyocr.Reader(["en"], gpu=True, verbose=False)

    (OUT / "images").mkdir(parents=True, exist_ok=True)
    lab = (OUT / "labels.jsonl").open("w", encoding="utf-8")

    kept = 0
    scanned = 0
    for k, frames in elig:
        if kept >= args.target:
            break
        scanned += 1
        # One crop per vehicle: the sharpest frame that yields a plate-shaped
        # region. Labelling several frames of one plate adds effort, not data.
        best_crop = None
        best_score = -1.0
        best_txt = ""
        for fr in frames:
            img = cv2.imread(fr["path"])
            if img is None:
                continue
            b = band_of(img)
            if b is None:
                continue
            bh, bw = b.shape[:2]
            for box, txt, cf in reader.readtext(b, allowlist=ALLOW, batch_size=8):
                if cf < args.conf or not plate_shaped(box, bw, bh, args.min_box_w):
                    continue
                cleaned = "".join(c for c in txt.upper() if c.isalnum())
                if len(cleaned) < 6 or any(t in cleaned for t in OSD):
                    continue
                xs = [p[0] for p in box]
                ys = [p[1] for p in box]
                x1, x2 = int(min(xs)), int(max(xs))
                y1, y2 = int(min(ys)), int(max(ys))
                px, py = int((x2 - x1) * 0.05), int((y2 - y1) * 0.22)
                crop = b[max(0, y1 - py):min(bh, y2 + py),
                         max(0, x1 - px):min(bw, x2 + px)]
                if crop.size == 0 or crop.shape[1] < 40:
                    continue
                # Prefer the sharpest crop, not the highest OCR confidence -
                # a human reads sharp pixels, and OCR confidence is unreliable.
                g = cv2.cvtColor(crop, cv2.COLOR_BGR2GRAY)
                sc = float(cv2.Laplacian(g, cv2.CV_64F).var())
                if sc > best_score:
                    best_score, best_crop, best_txt = sc, crop.copy(), cleaned
        if best_crop is None:
            continue
        name = f"{k[0]}_{k[1]}_t{k[2]}.jpg"
        cv2.imwrite(str(OUT / "images" / name), best_crop,
                    [cv2.IMWRITE_JPEG_QUALITY, 96])
        lab.write(json.dumps({"file": name, "camera": k[0], "track": k[2],
                              "text": best_txt, "agreement": 0.0, "votes": 1,
                              "sharpness": round(best_score, 1)}) + "\n")
        kept += 1
        if kept % 50 == 0:
            print(f"  {kept}/{args.target} kept  ({scanned} vehicles scanned)",
                  flush=True)

    lab.close()
    print("\n" + "=" * 56)
    print(f"crops for labelling : {kept}")
    print(f"vehicles scanned    : {scanned}")
    print(f"saved -> {OUT}")
    print("\nThese are NOT labelled - the text field is only an OCR draft, and")
    print("it is wrong ~80% of the time. It exists so the labeller corrects")
    print("rather than types.")


if __name__ == "__main__":
    main()
