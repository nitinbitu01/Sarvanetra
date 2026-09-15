"""backend/scripts/plate_harvest_detector.py — harvest plate crops with a plate
DETECTOR instead of EasyOCR's text boxes.

WHAT THIS REPLACES, AND WHY
  Every crop labelled so far was found this way: take a fixed band of the
  vehicle (rows 40-98%), upscale it, and ask EasyOCR to report text regions.
  That works, but badly, in three separate ways:

    yield      1,944 vehicles scanned produced 151 crops (7.8%). EasyOCR only
               reports a region when it thinks it can read something, so every
               plate too blurry for OCR was never even offered to the human -
               and those are exactly the examples the recogniser needs most.
    framing    a text box is not a plate box. It clips the first and last
               glyph when contrast falls off, and it wanders onto the dealer
               sticker below. The recogniser then trains on inconsistent
               framing, which it must waste capacity absorbing.
    geometry   the band is a guess about where plates sit. Trucks, autos and
               anything not framed like a hatchback fall outside it.

  A detector trained on plates has none of those failure modes: it fires on
  plate SHAPE, whether or not the glyphs are readable, and it returns the
  plate's own extent.

THE CROP IS DELIBERATELY LOOSE
  The box is expanded before cutting. A detector trained on tight boxes tends
  to shave the outermost glyph, and a recogniser cannot recover a character
  that was cropped away, whereas it copes easily with a few pixels of bumper.
  Asymmetric margins - more horizontally than vertically - match where that
  clipping actually happens.

USAGE
  python -m backend.scripts.plate_harvest_detector --target 1500
  python -m backend.scripts.plate_harvest_detector --target 800 --min-width 45
"""
from __future__ import annotations

import argparse
import json
from collections import defaultdict
from pathlib import Path

import cv2

CORPUS = Path("output/plate_corpus")
REAL = Path("data/plate_real")
WEIGHTS = Path("runs/plate_v3/recovered/weights/best.pt")


def already_labelled() -> set:
    """Vehicles a human has already read - never offer these again."""
    done = set()
    for name in ("verified_all.jsonl", "plate_labels_verified.jsonl",
                 "verified_batch2.jsonl", "verified_batch3.jsonl",
                 "verified_batch4.jsonl", "verified_batch5.jsonl"):
        p = REAL / name
        if not p.is_file():
            continue
        for line in p.open(encoding="utf-8"):
            v = json.loads(line)
            done.add((v["camera"], v["track"]))
    return done


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--out", default="data/plate_label_batch6")
    ap.add_argument("--weights", default=str(WEIGHTS))
    ap.add_argument("--target", type=int, default=1500,
                    help="Vehicles to prepare for labelling.")
    ap.add_argument("--conf", type=float, default=0.35)
    ap.add_argument("--min-width", type=float, default=40.0,
                    help="Narrowest detected plate to keep, in frame pixels. "
                         "Below roughly 40px a person cannot read it either, "
                         "so harvesting it only wastes labelling time.")
    ap.add_argument("--frames-per-vehicle", type=int, default=6,
                    help="Candidate frames examined per vehicle; the single "
                         "best detection is what gets saved.")
    ap.add_argument("--cameras", default="",
                    help="Restrict to these cameras, comma separated.")
    args = ap.parse_args()

    out = Path(args.out)
    (out / "images").mkdir(parents=True, exist_ok=True)

    done = already_labelled()
    print(f"already labelled : {len(done)} vehicles (skipped)")

    only = {c.strip() for c in args.cameras.replace(",", " ").split() if c.strip()}
    tracks = defaultdict(list)
    for line in (CORPUS / "manifest.jsonl").open(encoding="utf-8"):
        r = json.loads(line)
        if (r["camera"], r["track"]) in done:
            continue
        if only and r["camera"] not in only:
            continue
        tracks[(r["camera"], r["clip"], r["track"])].append(r)
    print(f"candidate vehicles: {len(tracks)}")

    # Sharpest frames first: the detector fires on blurry plates too, but a
    # human still has to read the result, and sharpness is the best cheap
    # predictor of that.
    ordered = []
    for k, frames in tracks.items():
        frames.sort(key=lambda r: -(r.get("sharpness", 0) * r.get("plate_px_est", 0)))
        ordered.append((k, frames[:args.frames_per_vehicle]))
    ordered.sort(key=lambda t: -max(f.get("sharpness", 0) for f in t[1]))

    from ultralytics import YOLO
    model = YOLO(args.weights)
    print(f"detector         : {args.weights}\n", flush=True)

    lab = (out / "labels.jsonl").open("w", encoding="utf-8")
    kept = scanned = 0
    widths = []
    per_cam = defaultdict(int)

    for key, frames in ordered:
        if kept >= args.target:
            break
        scanned += 1
        cam, clip, track = key

        best = None                     # (width, crop)
        for fr in frames:
            img = cv2.imread(fr["path"])
            if img is None:
                continue
            res = model.predict(img, conf=args.conf, verbose=False)[0]
            if not len(res.boxes):
                continue
            h, w = img.shape[:2]
            for b, cf in zip(res.boxes.xyxy.cpu().numpy(),
                             res.boxes.conf.cpu().numpy()):
                x1, y1, x2, y2 = b
                bw, bh = x2 - x1, y2 - y1
                if bw < args.min_width or bh < 10:
                    continue
                if not (1.8 <= bw / max(bh, 1) <= 7.5):
                    continue
                # Loose crop - see the module docstring. Horizontal margin is
                # larger because that is the edge glyphs get shaved from.
                mx, my = bw * 0.08, bh * 0.20
                cx1, cy1 = int(max(0, x1 - mx)), int(max(0, y1 - my))
                cx2, cy2 = int(min(w, x2 + mx)), int(min(h, y2 + my))
                crop = img[cy1:cy2, cx1:cx2]
                if crop.size == 0 or crop.shape[1] < 30:
                    continue
                if best is None or bw > best[0]:
                    best = (float(bw), crop.copy(), float(cf))

        if best is None:
            continue
        width, crop, conf = best
        name = f"{cam}_{clip}_t{track}.jpg"
        cv2.imwrite(str(out / "images" / name), crop,
                    [cv2.IMWRITE_JPEG_QUALITY, 96])
        lab.write(json.dumps({
            "file": name, "camera": cam, "clip": clip, "track": track,
            "text": "", "plate_w": round(width, 1), "det_conf": round(conf, 3),
        }) + "\n")
        kept += 1
        widths.append(width)
        per_cam[cam] += 1
        if kept % 100 == 0:
            print(f"  {kept}/{args.target} kept  ({scanned} vehicles scanned, "
                  f"{kept/max(scanned,1)*100:.0f}% yield)", flush=True)

    lab.close()
    widths.sort()
    print("\n" + "=" * 60)
    print(f"crops for labelling : {kept}")
    print(f"vehicles scanned    : {scanned}")
    print(f"YIELD               : {kept/max(scanned,1)*100:.1f}%"
          f"   (EasyOCR band heuristic: 7.8%)")
    if widths:
        print(f"plate width         : median {widths[len(widths)//2]:.0f}px  "
              f"p10 {widths[int(len(widths)*0.1)]:.0f}px  "
              f"p90 {widths[int(len(widths)*0.9)]:.0f}px")
    print("\nper camera:")
    for c, n in sorted(per_cam.items(), key=lambda kv: -kv[1]):
        print(f"  {c:<12} {n:>5}")
    print("=" * 60)
    print(f"\nsaved -> {out}")
    print("The text field is EMPTY on purpose. No OCR draft is offered here:")
    print("a wrong draft biases the reader toward accepting it, and EasyOCR is")
    print("wrong about 90% of the time on these crops.")


if __name__ == "__main__":
    main()
