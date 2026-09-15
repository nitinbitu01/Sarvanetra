"""backend/scripts/plate_harvest_multiview.py — offer several frames of each
vehicle, chosen for sharpness rather than size.

THE BUG THIS FIXES
  The previous harvest kept, per vehicle, the frame with the WIDEST detected
  plate. That sounds right and is wrong: the widest box is the frame where the
  vehicle is closest, which on a fixed-exposure CCTV camera is also the frame
  with the most motion blur. The harvest was therefore systematically handing
  the labeller the blurriest view of every plate.

  It showed. A human worked through all 672 crops and could read 77 - eleven
  percent. Re-rendering ten of the failures with sharpness-weighted frames
  made four of the nine real plates plainly readable, including one that is
  crisp in all four views and should never have been a failure.

WHY FOUR VIEWS AND NOT ONE
  Nobody reads a plate off a still. Motion blur, glare and occlusion land on
  different characters in different frames, so a reader combines them: the
  third character is legible in frame 1, the seventh in frame 3. Presenting a
  single frame discards that, and it is free to keep - the frames are already
  in the corpus and were mined months ago.

WHAT THIS DOES NOT DO
  It does not enhance, sharpen or upscale for training. Views are saved as
  cut, so the recogniser trains on the same distribution the live pipeline
  produces. Any display-side scaling belongs in the labelling tool, where it
  helps the human without changing the data.

USAGE
  python -m backend.scripts.plate_harvest_multiview --target 900
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

PLATE_CAPABLE = {"CAM_06", "CAM_07", "CAM_08", "CAM_09", "CAM_10",
                 "CAM_11", "CAM_18", "CAM_21", "CAM_27"}


def already_done() -> set:
    done = set()
    for name in ("verified_all.jsonl", "plate_labels_verified.jsonl",
                 "verified_batch2.jsonl", "verified_batch3.jsonl",
                 "verified_batch4.jsonl", "verified_batch5.jsonl",
                 "verified_batch6.jsonl"):
        p = REAL / name
        if not p.is_file():
            continue
        for line in p.open(encoding="utf-8"):
            v = json.loads(line)
            if v.get("text"):
                done.add((v["camera"], v["track"]))
    return done


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--out", default="data/plate_label_batch7")
    ap.add_argument("--weights", default=str(WEIGHTS))
    ap.add_argument("--target", type=int, default=900)
    ap.add_argument("--views", type=int, default=4)
    ap.add_argument("--conf", type=float, default=0.30)
    ap.add_argument("--min-width", type=float, default=40.0)
    ap.add_argument("--scan-frames", type=int, default=12,
                    help="Frames examined per vehicle before picking --views. "
                         "Higher costs time and finds sharper frames.")
    ap.add_argument("--all-cameras", action="store_true",
                    help="Do not restrict to the plate-capable cameras.")
    args = ap.parse_args()

    out = Path(args.out)
    (out / "images").mkdir(parents=True, exist_ok=True)

    done = already_done()
    print(f"already labelled : {len(done)} vehicles (skipped)")

    tracks = defaultdict(list)
    for line in (CORPUS / "manifest.jsonl").open(encoding="utf-8"):
        r = json.loads(line)
        if (r["camera"], r["track"]) in done:
            continue
        if not args.all_cameras and r["camera"] not in PLATE_CAPABLE:
            continue
        tracks[(r["camera"], r["clip"], r["track"])].append(r)
    print(f"candidate vehicles: {len(tracks)}")

    ordered = []
    for k, frames in tracks.items():
        # Sharpness FIRST, size second. The previous harvest sorted on plate
        # width alone and so preferred the closest, most motion-blurred frame
        # of every vehicle.
        frames.sort(key=lambda r: -(r.get("sharpness", 0) ** 0.5
                                    * r.get("plate_px_est", 0)))
        ordered.append((k, frames[:args.scan_frames]))
    ordered.sort(key=lambda t: -max(f.get("sharpness", 0) for f in t[1]))

    from ultralytics import YOLO
    model = YOLO(args.weights)
    print(f"detector         : {args.weights}\n", flush=True)

    lab = (out / "labels.jsonl").open("w", encoding="utf-8")
    kept = scanned = 0
    for key, frames in ordered:
        if kept >= args.target:
            break
        scanned += 1
        cam, clip, track = key
        views = []
        for fr in frames:
            if len(views) >= args.views:
                break
            img = cv2.imread(fr["path"])
            if img is None:
                continue
            res = model.predict(img, conf=args.conf, verbose=False)[0]
            if not len(res.boxes):
                continue
            h, w = img.shape[:2]
            best = None
            for b, cf in zip(res.boxes.xyxy.cpu().numpy(),
                             res.boxes.conf.cpu().numpy()):
                x1, y1, x2, y2 = b
                bw, bh = x2 - x1, y2 - y1
                if bw < args.min_width or bh < 10:
                    continue
                if not (1.8 <= bw / max(bh, 1) <= 7.5):
                    continue
                if best is None or bw > best[0]:
                    best = (bw, (x1, y1, x2, y2), float(cf))
            if best is None:
                continue
            bw, (x1, y1, x2, y2), cf = best
            mx, my = bw * 0.08, (y2 - y1) * 0.20
            crop = img[int(max(0, y1 - my)):int(min(h, y2 + my)),
                       int(max(0, x1 - mx)):int(min(w, x2 + mx))]
            if crop.size == 0 or crop.shape[1] < 30:
                continue
            name = f"{cam}_{clip}_t{track}_v{len(views)}.jpg"
            cv2.imwrite(str(out / "images" / name), crop,
                        [cv2.IMWRITE_JPEG_QUALITY, 96])
            # float() before round(): the box comes back as numpy float32,
            # which json cannot serialise and which round() preserves.
            views.append({"file": name, "plate_w": round(float(bw), 1),
                          "det_conf": round(float(cf), 3),
                          "sharpness": round(float(fr.get("sharpness", 0)), 1)})
        if not views:
            continue
        lab.write(json.dumps({
            "camera": cam, "clip": clip, "track": track, "text": "",
            "views": views,
            "agreement": max(v["det_conf"] for v in views),
            "plate_w": max(v["plate_w"] for v in views),
        }) + "\n")
        kept += 1
        if kept % 100 == 0:
            print(f"  {kept}/{args.target} vehicles  "
                  f"({scanned} scanned, {kept/max(scanned,1)*100:.0f}% yield)",
                  flush=True)
    lab.close()

    print("\n" + "=" * 58)
    print(f"vehicles offered : {kept}")
    print(f"vehicles scanned : {scanned}")
    print(f"yield            : {kept/max(scanned,1)*100:.1f}%")
    print(f"saved -> {out}")
    print("=" * 58)
    print("\nEach entry carries up to "
          f"{args.views} views of ONE plate. The labeller reads the")
    print("plate once, from whichever views make it legible.")


if __name__ == "__main__":
    main()
