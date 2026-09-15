"""backend/scripts/train_detector.py — fine-tune the production detector on
auto-labelled Gujarat frames.

WHAT THIS PRODUCES, AND HOW IT MUST BE DESCRIBED
  A yolov8s checkpoint ADAPTED to this deployment's cameras - their angles,
  lighting, resolution and traffic mix. That is a real improvement and worth
  doing.

  It is NOT a model "validated on Gujarat footage". The labels came from a
  larger model (see autolabel_frames.py), not from humans, so the ceiling is
  whatever the teacher already saw. Any claim of measured accuracy on this
  deployment still requires hand-labelled ground truth, which nothing here
  produces.

WHY FINE-TUNE RATHER THAN TRAIN FROM SCRATCH
  A few hundred frames is nowhere near enough to learn detection from
  nothing. Starting from COCO-pretrained yolov8s keeps everything the model
  already knows about people and vehicles and only nudges it toward this
  domain - which is also why the learning rate below is deliberately low.

EVALUATION CAVEAT
  The val split is auto-labelled by the SAME teacher, so val mAP measures
  "how well the student imitates the teacher", not real accuracy. Treat a
  high number as evidence that training converged, nothing more.

USAGE
  python -m backend.scripts.train_detector
  python -m backend.scripts.train_detector --epochs 50 --batch 8
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--data", type=Path,
                    default=Path("data/detection_train/gujarat.yaml"))
    ap.add_argument("--base", default="yolo26n.pt",
                    help="Starting weights — YOLO26 Nano (yolo26n.pt).")
    ap.add_argument("--epochs", type=int, default=40)
    ap.add_argument("--batch", type=int, default=4,
                    help="Small by default: 1280px images are VRAM-hungry, "
                         "and an OOM mid-run wastes the whole session.")
    ap.add_argument("--imgsz", type=int, default=1280,
                    help="Must match the deployed inference size "
                         "(processing.input_width in config.yaml). Training "
                         "at a different scale than you infer at is a "
                         "classic silent accuracy loss.")
    ap.add_argument("--lr", type=float, default=0.001,
                    help="Low: this is a domain nudge, not from-scratch "
                         "training. A high LR would wash out the COCO "
                         "pretraining that does most of the work.")
    ap.add_argument("--name", default="gujarat_yolo26n")
    ap.add_argument("--p2", action="store_true",
                    help="Train from yolov8-p2.yaml: adds a stride-4 "
                         "detection head. YOLOv8's finest default head is "
                         "stride-8, so each cell covers 8x8 px and a 19px "
                         "backpack spans barely two cells. P2 quadruples the "
                         "cells available to small objects. Costs ~20%% "
                         "compute, and CANNOT load yolov8s.pt weights - the "
                         "head count differs - so this trains the backbone "
                         "from the COCO-pretrained yaml instead.")
    ap.add_argument("--blur-aug", action="store_true",
                    help="Enable albumentations blur/noise/compression "
                         "augmentation. The dataset contains NO blurred "
                         "frames (measured: min sharpness 199, 0%% below "
                         "100) because the harvester rejects them - correct, "
                         "since the teacher labels blur badly, but it leaves "
                         "the student blur-naive. Augmenting sharp images "
                         "keeps the labels correct while showing the model "
                         "degraded pixels.")
    args = ap.parse_args()

    if not args.data.is_file():
        print(f"Dataset not found: {args.data}\n"
              "Run capture_training_frames then autolabel_frames first.",
              file=sys.stderr)
        sys.exit(1)

    import torch
    from ultralytics import YOLO

    if not torch.cuda.is_available():
        print("WARNING: CUDA unavailable — this will be extremely slow on CPU.",
              file=sys.stderr)

    base = args.base
    if args.p2:
        # A P2 model has four detection heads where yolov8s.pt has three, so
        # the pretrained checkpoint cannot be loaded directly. Build from the
        # architecture yaml and transfer what does match.
        #
        # The SCALE LETTER IS REQUIRED. Plain 'yolov8-p2.yaml' carries no
        # scale, so ultralytics silently defaults to nano:
        #     WARNING no model scale passed. Assuming scale='n'
        #     -> 3,354,144 params
        # against the deployed yolov8s at 11,166,560. Training that would
        # regress person/vehicle accuracy AND make any v4-vs-v5 comparison
        # meaningless, while looking like a successful run.
        #
        # Derived from --base so it tracks the deployed model rather than
        # being hardcoded: yolov8s.pt -> yolov8s-p2.yaml
        #     yolov8s-p2.yaml -> 10,884,336 params, strides [4, 8, 16, 32]
        stem = Path(args.base).stem            # 'yolov8s'
        base = f"{stem}-p2.yaml"

    print(f"Fine-tuning {base} on {args.data}")
    print(f"  epochs={args.epochs} batch={args.batch} imgsz={args.imgsz} lr={args.lr}")
    if args.p2:
        print("  P2 stride-4 head ENABLED (small-object detection)")
    if args.blur_aug:
        print("  blur/noise/JPEG augmentation ENABLED")

    model = YOLO(base)
    if args.p2:
        # Transfer the COCO backbone weights that DO match. Without this the
        # P2 model starts from random init and needs far more data than 4,784
        # frames to reach a useful place.
        try:
            model.load(args.base)
            print(f"  transferred matching weights from {args.base}")
        except Exception as exc:
            print(f"  WARNING: could not transfer {args.base} weights ({exc}); "
                  f"training from scratch will need more epochs", file=sys.stderr)
    model.train(
        data=str(args.data),
        epochs=args.epochs,
        batch=args.batch,
        imgsz=args.imgsz,
        lr0=args.lr,
        device=0 if torch.cuda.is_available() else "cpu",
        project="output/detector_train",
        name=args.name,
        exist_ok=True,
        # Augmentation tuned for fixed CCTV: these cameras never flip or
        # rotate, so heavy geometric augmentation would teach viewpoints
        # that cannot occur. Colour/exposure jitter DOES matter - it stands
        # in for time of day and weather.
        fliplr=0.0,
        flipud=0.0,
        degrees=0.0,
        hsv_h=0.015,
        hsv_s=0.7,
        hsv_v=0.4,
        # Mosaic OFF, for two independent reasons.
        #
        # Memory: mosaic stitches 4 images into a canvas of 2*imgsz, i.e.
        # 2560x2560x3 here. With the default 8 dataloader workers each
        # holding those buffers, a 60-epoch run died with
        # "Unable to allocate 18.8 MiB for an array with shape (2560,2560,3)".
        #
        # Correctness: these are FIXED cameras. Mosaic invents composite
        # scenes with four unrelated backgrounds and objects at scales and
        # positions that cannot occur in a static junction view. For domain
        # adaptation to specific cameras, that teaches the wrong prior.
        mosaic=0.0,
        # 8 workers (the default) is tuned for small images. At 1280px the
        # per-worker buffers are what exhausted RAM above.
        workers=2,
        patience=15,
        # Scale jitter matters MORE than usual here. Bags sit at ~19px after
        # the imgsz downscale, right at the detector's floor, so showing the
        # model the same object across a range of sizes is one of the few
        # cheap ways to improve small-object recall.
        scale=0.5 if not args.p2 else 0.6,
        # Ultralytics applies albumentations (Blur, MedianBlur, ToGray, CLAHE)
        # automatically when the package is importable. It is installed for
        # this run; `augment` keeps the standard pipeline on.
        augment=bool(args.blur_aug),
    )

    out = Path("output/detector_train") / args.name / "weights" / "best.pt"
    print()
    print(f"Best weights: {out}")
    print()
    print("To deploy, set config.yaml model.path to that file. Re-run")
    print("preflight_check.py afterwards: a fine-tuned model changes both")
    print("detection counts AND per-frame cost, which affects camera capacity.")
    print()
    print("REMINDER: labels were machine-generated. This model is ADAPTED to")
    print("Gujarat cameras, not VALIDATED on them.")


if __name__ == "__main__":
    main()
