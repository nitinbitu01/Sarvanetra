"""backend/scripts/autolabel_frames.py — pseudo-label captured frames with a
larger detector, to fine-tune the smaller production one.

THE TECHNIQUE, AND ITS HARD LIMIT
  This is self-training / knowledge distillation: run a BIGGER, slower model
  (yolov8x) over the Gujarat frames, keep only high-confidence boxes, and
  train the smaller production model (yolov8s) on those labels. It adapts
  the deployed model to this deployment's camera angles, lighting, and
  vehicle mix without anyone drawing a single box by hand.

  THE LIMIT, STATED PLAINLY: this can only teach yolov8s what yolov8x
  already sees. Objects BOTH models miss stay missed forever, and any
  systematic error yolov8x makes on Gujarat scenes gets baked in and
  amplified. It is not a substitute for human-labelled ground truth, and a
  model trained this way must NOT be described as "validated on Gujarat
  footage" - it has been *adapted* to it, which is a different and weaker
  claim.

  Nothing here can add a class COCO does not have. Auto-rickshaws - which
  are everywhere in Gujarat traffic - are not a COCO class, so they will
  keep being labelled "car"/"motorcycle" or missed. Adding that class needs
  real hand-labelling.

CONFIDENCE FLOOR
  --conf defaults to 0.5, deliberately higher than the 0.45 the pipeline
  runs at. A pseudo-label is training TRUTH: a wrong box teaches the student
  a wrong answer, so precision matters more here than recall.

USAGE
  python -m backend.scripts.autolabel_frames
  python -m backend.scripts.autolabel_frames --teacher yolov8m.pt --conf 0.6
"""
from __future__ import annotations

import argparse
import random
import shutil
import sys
from collections import Counter
from pathlib import Path

# COCO ids the pipeline cares about -> contiguous class indices for training.
#
# BAG CLASSES (24/26/28) ADDED.
#   The teacher (yolov8x) has always detected backpacks, handbags and
#   suitcases - they were being discarded here before the student ever saw
#   them. The consequence was not a weaker model but an impossible feature:
#   abandoned_object_detector.py is wired into the pipeline and asks "is
#   there a bag with no person near it?", while the detector knew only
#   person/car/motorcycle/bus/truck. It could never fire, and the alerts
#   table shows exactly that - 48 alerts, all anpr_uncertain, zero
#   abandoned-object in the system's entire history.
#
#   Keeping the labels costs nothing extra at teacher time: the same SAHI
#   pass already produces them.
#
#   Expect lower accuracy on these than on people. Measured object sizes put
#   a backpack at ~28px at source, ~19px after the imgsz=1280 downscale -
#   near YOLO's practical floor, which is why the P2 (stride-4) head and
#   blur augmentation go in alongside this change rather than after it.
KEEP = {
    0: "person", 2: "car", 3: "motorcycle", 5: "bus", 7: "truck",
    24: "backpack", 26: "handbag", 28: "suitcase",
}
CLASS_ORDER = [0, 2, 3, 5, 7, 24, 26, 28]
COCO_TO_IDX = {c: i for i, c in enumerate(CLASS_ORDER)}


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--images", type=Path, default=Path("data/detection_train/images"))
    ap.add_argument("--out", type=Path, default=Path("data/detection_train"))
    ap.add_argument("--teacher", default="yolov8x.pt")
    ap.add_argument("--conf", type=float, default=0.5)
    ap.add_argument("--imgsz", type=int, default=1280)
    ap.add_argument("--val-split", type=float, default=0.2)
    ap.add_argument("--sahi", action="store_true",
                    help="Slice each frame into overlapping tiles before "
                         "detection (SAHI). Finds more small/distant objects "
                         "because they occupy more pixels per inference pass. "
                         "MUCH slower - measured 16-26x on this hardware - "
                         "which is acceptable HERE precisely because labelling "
                         "is offline and one-off. Do NOT enable slicing in the "
                         "live pipeline: it collapsed camera capacity from ~45 "
                         "to 1-2 in benchmarking, and per-camera frame rate is "
                         "what gates track confirmation.")
    ap.add_argument("--slice", type=int, default=640,
                    help="SAHI tile size. Smaller finds more small objects "
                         "and costs more time (512px: +17% small objects, "
                         "25.6x slower; 640px: +6%, 16.2x slower).")
    args = ap.parse_args()

    frames = [
        p for p in sorted(args.images.rglob("*.jpg"))
        if "train" not in p.parts and "val" not in p.parts
    ]
    if not frames:
        print(f"No images under {args.images}. Run capture_training_frames first.",
              file=sys.stderr)
        sys.exit(1)
    print(f"Auto-labelling {len(frames)} frame(s) with {args.teacher} "
          f"(conf>={args.conf}, imgsz={args.imgsz})")

    # Two teacher backends with a common interface: given an image path,
    # return [(coco_class_id, xc, yc, w, h) ...] normalised to the frame.
    if args.sahi:
        from sahi import AutoDetectionModel
        from sahi.predict import get_sliced_prediction
        import cv2 as _cv2

        sahi_model = AutoDetectionModel.from_pretrained(
            model_type="ultralytics", model_path=args.teacher,
            confidence_threshold=args.conf, device="cuda:0",
        )
        print(f"  slicing enabled: {args.slice}px tiles, 20% overlap")

        def detect(img_path: Path):
            img = _cv2.imread(str(img_path))
            if img is None:
                return []
            h, w = img.shape[:2]
            res = get_sliced_prediction(
                img, sahi_model,
                slice_height=args.slice, slice_width=args.slice,
                overlap_height_ratio=0.2, overlap_width_ratio=0.2, verbose=0,
            )
            out = []
            for o in res.object_prediction_list:
                cid = o.category.id
                if cid not in COCO_TO_IDX:
                    continue
                bb = o.bbox
                out.append((cid,
                            ((bb.minx + bb.maxx) / 2) / w,
                            ((bb.miny + bb.maxy) / 2) / h,
                            (bb.maxx - bb.minx) / w,
                            (bb.maxy - bb.miny) / h))
            return out
    else:
        from ultralytics import YOLO
        teacher = YOLO(args.teacher)

        def detect(img_path: Path):
            r = teacher(str(img_path), imgsz=args.imgsz, conf=args.conf,
                        classes=CLASS_ORDER, device="cuda", verbose=False)[0]
            out = []
            for b in r.boxes:
                cid = int(b.cls[0])
                if cid not in COCO_TO_IDX:
                    continue
                x, y, bw, bh = b.xywhn[0].tolist()
                out.append((cid, x, y, bw, bh))
            return out

    # YOLO training layout: images/{train,val} + labels/{train,val}
    rng = random.Random(0)
    shuffled = frames[:]
    rng.shuffle(shuffled)
    n_val = max(1, int(len(shuffled) * args.val_split))
    split = {p: ("val" if i < n_val else "train") for i, p in enumerate(shuffled)}

    # CLEAR the split dirs before rewriting them.
    #
    # These used to be created with exist_ok=True and then written into,
    # leaving whatever a PREVIOUS run had put there. Because the split is a
    # fresh shuffle each time, a frame that was in val last run could land in
    # train this run while its stale val copy survived - so the same image
    # ended up in BOTH sets. Measured after one such run: 4,972 train +
    # 1,801 val for 5,372 source frames, with 1,401 images present in both.
    #
    # A model then trains on part of its own validation set and every score
    # is inflated by memorisation - silently, and in the flattering
    # direction, which is the hardest kind of bug to notice.
    for sub in ("train", "val"):
        for kind in ("images", "labels"):
            d = args.out / kind / sub
            if d.exists():
                shutil.rmtree(d)
            d.mkdir(parents=True, exist_ok=True)

    counts = Counter()
    empty = 0
    for i, img_path in enumerate(frames, 1):
        sub = split[img_path]
        lines = []
        for cid, x, y, w, h in detect(img_path):
            lines.append(f"{COCO_TO_IDX[cid]} {x:.6f} {y:.6f} {w:.6f} {h:.6f}")
            counts[KEEP[cid]] += 1

        # A frame with no detections is still a valid NEGATIVE example - an
        # empty label file teaches "nothing here", which reduces false
        # positives on empty road scenes. Dropping them would bias the
        # student toward always finding something.
        if not lines:
            empty += 1

        dst_img = args.out / "images" / sub / img_path.name
        if img_path.resolve() != dst_img.resolve():
            shutil.copy2(img_path, dst_img)
        (args.out / "labels" / sub / f"{img_path.stem}.txt").write_text(
            "\n".join(lines))

        if i % 25 == 0 or i == len(frames):
            print(f"  {i}/{len(frames)} labelled")

    data_yaml = args.out / "gujarat.yaml"
    data_yaml.write_text(
        f"path: {args.out.resolve().as_posix()}\n"
        "train: images/train\n"
        "val: images/val\n"
        f"nc: {len(CLASS_ORDER)}\n"
        f"names: {[KEEP[c] for c in CLASS_ORDER]}\n"
    )

    print()
    print(f"labels written   : {sum(counts.values())} boxes")
    for k, v in counts.most_common():
        print(f"  {k:<12} {v}")
    print(f"empty frames     : {empty} (kept as negatives)")
    print(f"train/val split  : {len(frames)-n_val}/{n_val}")
    print(f"dataset yaml     : {data_yaml}")
    print()
    print("Next: python -m backend.scripts.train_detector")


if __name__ == "__main__":
    main()
