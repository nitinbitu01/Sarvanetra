"""backend/scripts/eval_detector.py — did the fine-tune actually help?

WHAT IS BEING MEASURED
  Both the baseline (yolov8s.pt as shipped) and the fine-tuned checkpoint
  are scored against the SAME held-out val split, whose labels came from the
  yolov8x teacher. So the number answers exactly one question:

      "How closely does this model reproduce what the big model sees
       on Gujarat footage?"

  That is the correct measure for distillation, and it is NOT the same as
  real-world accuracy. Both models are being graded against a machine's
  opinion. If the teacher systematically misses something on these cameras,
  a model that also misses it scores perfectly here.

  A gain therefore means "the student absorbed teacher knowledge", which is
  a real and useful result - not "the system is now X% accurate on Gujarat
  CCTV", which would require hand-labelled ground truth.

USAGE
  python -m backend.scripts.eval_detector
  python -m backend.scripts.eval_detector --tuned path/to/best.pt
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path


# autolabel_frames.py's COCO -> contiguous mapping. Must stay in step with it.
COCO_TO_IDX = {0: 0, 2: 1, 3: 2, 5: 3, 7: 4}
IOU_MATCH = 0.5


def _iou(a, b) -> float:
    ix1, iy1 = max(a[0], b[0]), max(a[1], b[1])
    ix2, iy2 = min(a[2], b[2]), min(a[3], b[3])
    iw, ih = max(0.0, ix2 - ix1), max(0.0, iy2 - iy1)
    inter = iw * ih
    union = ((a[2]-a[0])*(a[3]-a[1]) + (b[2]-b[0])*(b[3]-b[1]) - inter)
    return inter / union if union > 0 else 0.0


def _load_labels(stem: str, lbl_dir: Path, w: int, h: int) -> list:
    f = lbl_dir / f"{stem}.txt"
    out = []
    if not f.is_file():
        return out
    for line in f.read_text().splitlines():
        if not line.strip():
            continue
        c, x, y, bw, bh = line.split()
        c = int(c); x, y, bw, bh = map(float, (x, y, bw, bh))
        out.append((c, (x-bw/2)*w, (y-bh/2)*h, (x+bw/2)*w, (y+bh/2)*h))
    return out


def _score(weights: str, args, from_coco: bool, half: bool = False) -> dict:
    """Precision/recall/F1 against teacher labels, in a common class space.

    half=True runs FP16 inference (CUDA only) - used to confirm a precision
    change doesn't quietly cost accuracy, not just to measure speed.

    Uses quantize=16 rather than the deprecated half= kwarg: half= routes
    through get_cfg's arg-merge on every single call and logs a deprecation
    warning per call (not deduped) - harmless once, unusable at hundreds of
    calls per eval run.
    """
    from ultralytics import YOLO
    import yaml as _yaml

    root = Path(_yaml.safe_load(args.data.read_text())["path"])
    imgs = sorted((root / "images" / "val").glob("*.jpg"))
    lbl_dir = root / "labels" / "val"

    quantize_kwargs = {"quantize": 16} if half else {}
    model = YOLO(weights)
    tp = fp = fn = 0
    for img in imgs:
        r = model(str(img), imgsz=args.imgsz, conf=0.45, verbose=False, **quantize_kwargs)[0]
        h, w = r.orig_shape
        gt = _load_labels(img.stem, lbl_dir, w, h)

        preds = []
        for b in r.boxes:
            c = int(b.cls[0])
            if from_coco:
                if c not in COCO_TO_IDX:
                    continue
                c = COCO_TO_IDX[c]
            elif c not in set(COCO_TO_IDX.values()):
                continue
            preds.append((c, *b.xyxy[0].tolist()))

        used: set[int] = set()
        for p in preds:
            best, best_i = 0.0, -1
            for i, g in enumerate(gt):
                if i in used or g[0] != p[0]:
                    continue
                v = _iou(p[1:], g[1:])
                if v > best:
                    best, best_i = v, i
            if best >= IOU_MATCH:
                tp += 1
                used.add(best_i)
            else:
                fp += 1
        fn += len(gt) - len(used)

    prec = tp / (tp + fp) if (tp + fp) else 0.0
    rec = tp / (tp + fn) if (tp + fn) else 0.0
    f1 = 2*prec*rec/(prec+rec) if (prec+rec) else 0.0
    return {"tp": tp, "fp": fp, "fn": fn,
            "precision": prec, "recall": rec, "f1": f1}


def find_tuned(name: str) -> Path | None:
    """Ultralytics may nest runs under runs/detect/<project>/<name>, so search
    rather than assume one layout."""
    for pat in (
        f"**/{name}/weights/best.pt",
        f"**/{name}*/weights/best.pt",
    ):
        hits = sorted(Path(".").glob(pat), key=lambda p: p.stat().st_mtime,
                      reverse=True)
        if hits:
            return hits[0]
    return None


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--data", type=Path,
                    default=Path("data/detection_train/gujarat.yaml"))
    ap.add_argument("--baseline", default="yolov8s.pt")
    ap.add_argument("--tuned", default="")
    ap.add_argument("--name", default="gujarat_yolov8s")
    ap.add_argument("--imgsz", type=int, default=1280)
    args = ap.parse_args()

    if not args.data.is_file():
        print(f"Dataset not found: {args.data}", file=sys.stderr)
        sys.exit(1)

    tuned = Path(args.tuned) if args.tuned else find_tuned(args.name)
    if not tuned or not tuned.is_file():
        print(f"Fine-tuned weights not found (looked for '{args.name}').\n"
              "Run train_detector first, or pass --tuned explicitly.",
              file=sys.stderr)
        sys.exit(1)

    from ultralytics import YOLO

    print(f"val set : {args.data}")
    print(f"baseline: {args.baseline}")
    print(f"tuned   : {tuned}\n")

    # DO NOT use ultralytics val() to compare these two models.
    #
    # The training set remaps classes to
    #     [person, car, motorcycle, bus, truck] = 0..4
    # but stock yolov8s emits COCO indices (car=2, motorcycle=3, bus=5,
    # truck=7). Only 'person' lines up. Scored naively, every car/bike/bus/
    # truck the BASELINE correctly found counts as a wrong-class error,
    # while the fine-tuned model - trained on the remapped indices - is
    # scored correctly. That produced an apparent +411% mAP50 gain when the
    # honest figure was +18% F1.
    #
    # So: map both models into a common class space first, then match
    # predictions to teacher labels by IoU.
    results = {}
    for label, weights, from_coco in (
        ("baseline", args.baseline, True),
        ("fine-tuned", str(tuned), False),
    ):
        results[label] = _score(weights, args, from_coco)

    print()
    print(f"{'model':<14} {'TP':>5} {'FP':>5} {'FN':>5} "
          f"{'prec':>7} {'recall':>7} {'F1':>7}")
    print("-" * 56)
    for label in ("baseline", "fine-tuned"):
        v = results[label]
        print(f"{label:<14} {v['tp']:>5} {v['fp']:>5} {v['fn']:>5} "
              f"{v['precision']:>7.3f} {v['recall']:>7.3f} {v['f1']:>7.3f}")

    print()
    b, t = results["baseline"]["f1"], results["fine-tuned"]["f1"]
    print(f"F1: {b:.3f} -> {t:.3f}  ({(t-b)/max(b, 1e-9)*100:+.1f}%)")
    print()
    gain = t - b
    if gain > 0.02:
        print("The fine-tune moved the model measurably closer to the teacher")
        print("on Gujarat footage. Deploy by pointing config.yaml model.path")
        print("at the tuned weights, then re-run preflight_check.py - a new")
        print("model changes both detection counts and per-frame cost.")
    elif gain > 0:
        print("Marginal gain. Worth more captured frames before deploying;")
        print("352 frames is a small dataset for domain adaptation.")
    else:
        print("No gain. Do NOT deploy this checkpoint - the baseline is at")
        print("least as good, and shipping a tuned model that is not better")
        print("adds risk for nothing.")

    print()
    print("REMINDER: labels are machine-generated. This measures agreement")
    print("with yolov8x, not real-world accuracy.")


if __name__ == "__main__":
    main()
