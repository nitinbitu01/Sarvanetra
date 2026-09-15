"""backend/scripts/recover_plate_boxes.py — turn 804 already-labelled plate
crops into 804 detector training boxes, with no human effort.

WHY THIS EXISTS
  The audit showed the existing plate detector is worthless on unseen vehicles
  (8% detection rate, 0% on most cameras) because it was trained on 392 frames
  of which 89% of the val set was also in train. It memorised. A real detector
  needs real boxes, and lots of them.

  Those boxes already exist implicitly. Every crop in data/plate_real/images
  was CUT OUT of a corpus frame by an earlier harvest, and a human then read
  the plate in it - so each crop is a confirmed plate, and its position in the
  source frame is a confirmed box. The harvest simply never wrote the
  coordinates down. Template matching recovers them.

THE SCALE PROBLEM, AND WHY THE MATCH IS EXACT RATHER THAN A SEARCH
  Crops were not cut from the frame directly. The harvest took a band of the
  vehicle (rows 40-98%, columns 6-94%), upscaled it to a fixed width, and cut
  from that. So a crop is larger than the region it came from, by a factor the
  harvest computed as 380/band_width.

  That factor is recoverable from the frame's own dimensions, so the crop can
  be scaled back to frame resolution BEFORE matching, rather than searching
  over scales. That turns an ambiguous multi-scale search into one exact
  correlation per frame, which is both faster and far less likely to land on
  the wrong rectangle.

WHICH FRAME DID THE CROP COME FROM?
  The filename names the vehicle but usually not the frame, so every frame of
  that vehicle is tried and the best correlation wins. A crop genuinely cut
  from one of them correlates near 1.0; anything below --min-score is dropped
  rather than guessed, because a wrong box teaches the detector to fire on
  whatever it landed on.

USAGE
  python -m backend.scripts.recover_plate_boxes
  python -m backend.scripts.recover_plate_boxes --min-score 0.75 --out data/plate_detect_v2
"""
from __future__ import annotations

import argparse
import json
import re
from collections import defaultdict
from pathlib import Path

import cv2
import numpy as np

REAL = Path("data/plate_real")
CORPUS = Path("output/plate_corpus")

# Both harvest naming schemes:
#   CAM_08_CAM_08_0730_t34394_00.jpg      (batch 1, index suffix)
#   CAM_08_CAM_08_0730_t34394.jpg         (batches 2-5, one crop per vehicle)
#   CAM_08_CAM_08_0730_t34394_xf000014.jpg (expanded views, frame named)
NAME = re.compile(r"^(CAM[_-][\w-]+?)_(CAM[_-][\w-]+?_\d+)_t(\d+)"
                  r"(?:_x(f\d+)|_(\d+))?$")

# A fourth scheme, and by far the most common one - 3138 of 4701 crops:
#   CAM_04_t21644_xf001973.jpg            (camera, track, frame; NO clip)
#
# The clip segment is simply absent. NAME requires it, so every one of these
# failed to parse and was skipped, which is why the first run recovered 796
# boxes out of a possible 4701 and reported 3905 "unparsed name". They are not
# damaged or ambiguous - the harvest that wrote them just did not repeat the
# clip id it had already encoded in the track number.
#
# They are recoverable because track ids are allocated from a global counter
# rather than per clip, so (camera, track) identifies a vehicle on its own and
# the clip can be looked up rather than parsed.
NAME_NOCLIP = re.compile(r"^(CAM[_-][\w-]+?)_t(\d+)(?:_x(f\d+))?$")

# The two band geometries used by the harvest scripts. Both are tried and the
# better correlation wins - guessing wrong would silently shift every box.
BANDS = [
    (0.40, 0.98, 0.06, 0.94, 380.0),   # plate_harvest_for_labelling.py
    (0.42, 0.97, 0.08, 0.92, 360.0),   # expand_labelled_crops.py
]


def parse(stem: str):
    """(camera, clip, track, frame_tag). clip is None when the name omits it."""
    m = NAME.match(stem)
    if m:
        return m.group(1), m.group(2), int(m.group(3)), m.group(4)
    m = NAME_NOCLIP.match(stem)
    if m:
        return m.group(1), None, int(m.group(2)), m.group(3)
    return None


def match_in_frame(frame, crop, band):
    """Best (score, box) for this crop in this frame, in FRAME pixel coords."""
    y0f, y1f, x0f, x1f, target_w = band
    h, w = frame.shape[:2]
    y0, y1 = int(h * y0f), int(h * y1f)
    x0, x1 = int(w * x0f), int(w * x1f)
    b = frame[y0:y1, x0:x1]
    if b.size == 0 or b.shape[1] < 20:
        return -1.0, None

    # The upscale the harvest applied, so the crop can be brought back to
    # frame scale instead of searching over scales.
    f = target_w / b.shape[1] if b.shape[1] < target_w else 1.0
    ch, cw = crop.shape[:2]
    th, tw = max(1, int(round(ch / f))), max(1, int(round(cw / f)))
    if th >= b.shape[0] or tw >= b.shape[1] or th < 6 or tw < 12:
        return -1.0, None
    tpl = cv2.resize(crop, (tw, th), interpolation=cv2.INTER_AREA)

    res = cv2.matchTemplate(b, tpl, cv2.TM_CCOEFF_NORMED)
    _, score, _, loc = cv2.minMaxLoc(res)
    return float(score), (x0 + loc[0], y0 + loc[1], tw, th)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--out", default="data/plate_detect_v2")
    ap.add_argument("--min-score", type=float, default=0.70,
                    help="Correlation floor. A true match scores near 1.0; "
                         "below this the box is a guess and is dropped.")
    ap.add_argument("--val-frac", type=float, default=0.20)
    args = ap.parse_args()

    out = Path(args.out)
    for s in ("train", "val"):
        (out / "images" / s).mkdir(parents=True, exist_ok=True)
        (out / "labels" / s).mkdir(parents=True, exist_ok=True)

    truth = set()
    for line in (REAL / "verified_all.jsonl").open(encoding="utf-8"):
        v = json.loads(line)
        truth.add((v["camera"], v["track"]))
    print(f"human-verified vehicles : {len(truth)}")

    crops = []
    for line in (REAL / "labels.jsonl").open(encoding="utf-8"):
        r = json.loads(line)
        if (r["camera"], r["track"]) in truth:
            crops.append(r)
    print(f"crops to locate         : {len(crops)}")

    # Every corpus frame, indexed by the vehicle it belongs to.
    frames_of = defaultdict(list)
    # Second index for the names that omit the clip. Track ids come from a
    # global counter, so (camera, track) already identifies one vehicle; this
    # index just spares those crops from being discarded.
    frames_by_track = defaultdict(list)
    for line in (CORPUS / "manifest.jsonl").open(encoding="utf-8"):
        r = json.loads(line)
        frames_of[(r["camera"], r["clip"], r["track"])].append(r)
        frames_by_track[(r["camera"], r["track"])].append(r)
    print(f"corpus vehicles indexed : {len(frames_of)}")
    print(f"  also by (camera,track): {len(frames_by_track)}\n", flush=True)

    # If a track id ever repeated across clips on one camera, the fallback
    # index would merge two different vehicles and the correlation would then
    # pick a frame from the wrong one. Checked rather than assumed.
    ambiguous = sum(1 for k, v in frames_by_track.items()
                    if len({r["clip"] for r in v}) > 1)
    if ambiguous:
        print(f"  WARNING: {ambiguous} (camera,track) keys span several clips; "
              f"correlation still decides, but the candidate set is wider\n")

    found = []
    unparsed = no_frames = weak = 0
    for i, r in enumerate(crops, 1):
        p = parse(Path(r["file"]).stem)
        if not p:
            unparsed += 1
            continue
        cam, clip, track, frame_tag = p
        cands = (frames_of.get((cam, clip, track), []) if clip
                 else frames_by_track.get((cam, track), []))
        if not cands:
            no_frames += 1
            continue
        # An expanded crop names its own frame - no search needed.
        if frame_tag:
            cands = [c for c in cands if Path(c["path"]).stem == frame_tag] or cands

        crop = cv2.imread(str(REAL / "images" / r["file"]))
        if crop is None:
            continue

        best = (-1.0, None, None)
        for c in cands:
            frame = cv2.imread(c["path"])
            if frame is None:
                continue
            for band in BANDS:
                sc, box = match_in_frame(frame, crop, band)
                if sc > best[0]:
                    best = (sc, box, c["path"])
        if best[0] < args.min_score or best[1] is None:
            weak += 1
            continue
        found.append({"score": best[0], "box": best[1], "frame": best[2],
                      "camera": cam, "track": track, "text": r["text"]})
        if i % 100 == 0:
            print(f"  {i}/{len(crops)} crops  |  {len(found)} located",
                  flush=True)

    print(f"\nlocated       : {len(found)}")
    print(f"weak match    : {weak} (dropped rather than guessed)")
    print(f"no frames     : {no_frames}")
    print(f"unparsed name : {unparsed}")
    if found:
        s = sorted(f["score"] for f in found)
        print(f"match score   : median {s[len(s)//2]:.3f}  "
              f"min {s[0]:.3f}  max {s[-1]:.3f}")

    # One box per FRAME - a frame can host several located crops of the same
    # vehicle, and YOLO wants them in a single label file.
    per_frame = defaultdict(list)
    for f in found:
        per_frame[f["frame"]].append(f)

    # SPLIT BY VEHICLE. Splitting by frame is what produced the 89% leak in the
    # previous detector: consecutive frames of one plate landed on both sides,
    # and the resulting 0.995 mAP measured memorisation.
    vehicles = sorted({(f["camera"], f["track"]) for f in found})
    import random
    random.Random(4242).shuffle(vehicles)
    n_val = max(1, int(len(vehicles) * args.val_frac))
    val_vehicles = set(vehicles[:n_val])

    counts = {"train": 0, "val": 0}
    boxes = {"train": 0, "val": 0}
    for frame_path, items in per_frame.items():
        img = cv2.imread(frame_path)
        if img is None:
            continue
        h, w = img.shape[:2]
        # A frame belongs to exactly one vehicle in this corpus layout, so the
        # split stays clean.
        key = (items[0]["camera"], items[0]["track"])
        split = "val" if key in val_vehicles else "train"

        name = (f"{items[0]['camera']}_t{items[0]['track']}_"
                f"{Path(frame_path).stem}")
        lines = []
        for it in items:
            x, y, bw, bh = it["box"]
            lines.append(f"0 {(x+bw/2)/w:.6f} {(y+bh/2)/h:.6f} "
                         f"{bw/w:.6f} {bh/h:.6f}")
        if not lines:
            continue
        cv2.imwrite(str(out / "images" / split / f"{name}.jpg"), img,
                    [cv2.IMWRITE_JPEG_QUALITY, 95])
        (out / "labels" / split / f"{name}.txt").write_text(
            "\n".join(lines) + "\n", encoding="utf-8")
        counts[split] += 1
        boxes[split] += len(lines)

    # Sidecar for verify_recovered_boxes: the YOLO label files carry only
    # normalised coordinates, and checking that a box is really on a plate
    # needs the plate text and the frame it came from.
    with (out / "boxes.jsonl").open("w", encoding="utf-8") as fh:
        for f in found:
            fh.write(json.dumps({
                "frame": f["frame"], "box": [int(v) for v in f["box"]],
                "text": f["text"], "camera": f["camera"],
                "track": f["track"], "score": round(float(f["score"]), 4),
            }) + "\n")

    (out / "plate.yaml").write_text(
        f"path: {out.resolve().as_posix()}\n"
        f"train: images/train\nval: images/val\n\nnc: 1\nnames: [plate]\n",
        encoding="utf-8")

    print("\n" + "=" * 60)
    print(f"train : {counts['train']} images, {boxes['train']} boxes")
    print(f"val   : {counts['val']} images, {boxes['val']} boxes")
    print(f"split : by VEHICLE ({len(vehicles)-n_val} train / {n_val} val)")
    print(f"wrote : {out}/plate.yaml")
    print("=" * 60)
    print("\nEvery box here sits on a plate a human already read, so these are")
    print("ground truth, not pseudo-labels. Verify a sample visually before")
    print("training - a systematically shifted box would be invisible in the")
    print("numbers above and would teach the detector the wrong target.")


if __name__ == "__main__":
    main()
