"""backend/scripts/expand_labelled_crops.py — more training views, no new labels.

THE WASTE THIS RECOVERS
  199 vehicles have been hand-labelled, but only 85 of them contribute more
  than one crop to training - batches 2 to 4 copied a single best frame per
  vehicle to keep the labelling list short. The other frames were never
  deleted; they are still in output/plate_corpus, up to 25 per vehicle.

  A plate photographed across a dozen frames gives a dozen genuinely different
  views of the SAME known string: different distance, angle, motion blur,
  lighting. That is exactly the variation a recogniser needs, and the label is
  already known. Multiplying the training set this way costs GPU time and no
  human time at all.

WHY THE CROPS MUST BE RE-CUT, NOT COPIED
  The corpus holds VEHICLE crops; the recogniser trains on tight PLATE crops.
  So each frame is re-processed: take the lower-vehicle band, let OCR locate
  the text region, and cut to it. OCR's READING is discarded entirely - it is
  wrong most of the time - and only its BOX is used. Localisation is the one
  thing it does reliably, and the ground-truth text comes from the human label.

KEEPING THE HELD-OUT SET HONEST
  New views are grouped under their existing (camera, track), so the
  train/test split by track still puts every view of one plate on the same
  side. Without that, the model would train on one frame of a plate and be
  tested on another frame of it - memorisation scored as accuracy.

USAGE
  python -m backend.scripts.expand_labelled_crops
"""
from __future__ import annotations

import argparse
import json
from collections import defaultdict
from pathlib import Path

import cv2

CORPUS = Path("output/plate_corpus")
REAL = Path("data/plate_real")
ALLOW = "ABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789"
OSD = {"BRIDGE", "BHAI", "CHIMAN", "CSITMS", "CSI", "BRIDC", "JANPATH", "PTZ"}


def band_of(img):
    h, w = img.shape[:2]
    b = img[int(h * 0.42):int(h * 0.97), int(w * 0.08):int(w * 0.92)]
    if b.size == 0 or b.shape[1] < 20:
        return None
    if b.shape[1] < 360:
        f = 360 / b.shape[1]
        b = cv2.resize(b, None, fx=f, fy=f, interpolation=cv2.INTER_CUBIC)
    return b


def plate_shaped(box, bw: int, bh: int) -> bool:
    xs = [p[0] for p in box]
    ys = [p[1] for p in box]
    w, h = max(xs) - min(xs), max(ys) - min(ys)
    if w < 36 or h < 11:
        return False
    if not (2.0 <= w / max(h, 1) <= 7.0):
        return False
    if w > bw * 0.80:                      # the burned-in caption spans the frame
        return False
    return (min(ys) + max(ys)) / 2 >= bh * 0.18


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--max-per-track", type=int, default=10,
                    help="Cap views per vehicle. Beyond this the extra frames "
                         "are near-duplicates of ones already kept, and let "
                         "one vehicle dominate the training set.")
    ap.add_argument("--conf", type=float, default=0.06,
                    help="Low on purpose: only the BOX is used, never the text.")
    args = ap.parse_args()

    truth = {}
    for l in (REAL / "verified_all.jsonl").open(encoding="utf-8"):
        v = json.loads(l)
        truth[(v["camera"], v["track"])] = v["text"]
    print(f"hand-labelled vehicles : {len(truth)}")

    have: dict[tuple, set] = defaultdict(set)
    for l in (REAL / "labels.jsonl").open(encoding="utf-8"):
        r = json.loads(l)
        have[(r["camera"], r["track"])].add(r["file"])
    print(f"crops already in training set : "
          f"{sum(len(v) for v in have.values())}")

    # Every corpus frame belonging to a labelled vehicle.
    frames: dict[tuple, list] = defaultdict(list)
    for l in (CORPUS / "manifest.jsonl").open(encoding="utf-8"):
        r = json.loads(l)
        k = (r["camera"], r["track"])
        if k in truth:
            frames[k].append(r)
    total_avail = sum(len(v) for v in frames.values())
    print(f"corpus frames for those vehicles : {total_avail}\n", flush=True)

    import easyocr
    print("loading EasyOCR (localisation only)...", flush=True)
    reader = easyocr.Reader(["en"], gpu=True, verbose=False)

    added = 0
    skipped_dupe = 0
    out = (REAL / "labels.jsonl").open("a", encoding="utf-8")

    for i, (key, rows) in enumerate(sorted(frames.items()), 1):
        cam, track = key
        text = truth[key]
        # Sharpest first, so the cap keeps the most legible views.
        rows.sort(key=lambda r: -r.get("sharpness", 0))
        kept = len(have[key])
        for r in rows:
            if kept >= args.max_per_track:
                break
            name = f"{cam}_{Path(r['path']).parent.parent.name}_t{track}_x{Path(r['path']).stem}.jpg"
            if name in have[key]:
                skipped_dupe += 1
                continue
            img = cv2.imread(r["path"])
            if img is None:
                continue
            b = band_of(img)
            if b is None:
                continue
            bh, bw = b.shape[:2]
            best = None
            best_area = 0
            for box, txt, cf in reader.readtext(b, allowlist=ALLOW, batch_size=8):
                if cf < args.conf or not plate_shaped(box, bw, bh):
                    continue
                cleaned = "".join(c for c in txt.upper() if c.isalnum())
                if any(t in cleaned for t in OSD):
                    continue
                xs = [p[0] for p in box]
                ys = [p[1] for p in box]
                area = (max(xs) - min(xs)) * (max(ys) - min(ys))
                if area > best_area:
                    best_area, best = area, (xs, ys)
            if best is None:
                continue
            xs, ys = best
            x1, x2 = int(min(xs)), int(max(xs))
            y1, y2 = int(min(ys)), int(max(ys))
            px, py = int((x2 - x1) * 0.05), int((y2 - y1) * 0.22)
            crop = b[max(0, y1 - py):min(bh, y2 + py),
                     max(0, x1 - px):min(bw, x2 + px)]
            if crop.size == 0 or crop.shape[1] < 36:
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
        if i % 25 == 0:
            print(f"  {i}/{len(frames)} vehicles  |  +{added} views",
                  flush=True)

    out.close()
    per_track = {k: len(v) for k, v in have.items()}
    n = len(per_track)
    print("\n" + "=" * 56)
    print(f"views added   : {added}")
    print(f"training crops: {sum(per_track.values())} across {n} vehicles "
          f"({sum(per_track.values())/max(n,1):.1f} per vehicle)")
    print(f"single-view   : {sum(1 for v in per_track.values() if v == 1)} "
          f"vehicles still have only one")
    print("\nLabels are unchanged — these are additional VIEWS of plates a "
          "human already read.")


if __name__ == "__main__":
    main()
