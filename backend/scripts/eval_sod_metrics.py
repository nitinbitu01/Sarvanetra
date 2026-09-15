"""backend/scripts/eval_sod_metrics.py — precision/recall/F1 split by OBJECT SIZE.

WHY SIZE-SPLIT AND NOT AGGREGATE
  An aggregate score averages over object sizes, so a model that is excellent
  on buses and completely blind to small objects still looks good. That is
  not hypothetical here: this project reported F1 0.803 for weeks while the
  detector could not see bags at all, because cars and buses carried the
  average.

  COCO size buckets, computed on the ORIGINAL image (not the downscaled
  inference size), so the numbers are comparable to published work:
      small   area <  32*32  = 1024 px^2
      medium  1024 <= area < 96*96 = 9216 px^2
      large   area >= 9216 px^2

WHAT IS BEING MEASURED
  Agreement with the SAHI teacher's labels, in a common class space, matched
  by IoU >= 0.5 - the same yardstick eval_detector.py uses. Not human ground
  truth. A model that reproduces the teacher's blind spots scores well here.

TWO MODELS, SAME FRAMES
  Both models are scored on identical images so the delta is attributable to
  the model rather than to the sample. Classes present in one model but not
  the other (the bag classes) are reported separately rather than silently
  penalising the older model for predictions it cannot make.

USAGE
  python -m backend.scripts.eval_sod_metrics
  python -m backend.scripts.eval_sod_metrics --b models_gujarat_yolov8s.pt --a <new.pt>
"""
from __future__ import annotations

import argparse
import sys
from collections import defaultdict
from pathlib import Path

SMALL_MAX = 32 * 32          # 1024 px^2
MEDIUM_MAX = 96 * 96         # 9216 px^2
IOU_MATCH = 0.5


def bucket(area: float) -> str:
    if area < SMALL_MAX:
        return "small"
    if area < MEDIUM_MAX:
        return "medium"
    return "large"


def _iou(a, b) -> float:
    ix1, iy1 = max(a[0], b[0]), max(a[1], b[1])
    ix2, iy2 = min(a[2], b[2]), min(a[3], b[3])
    iw, ih = max(0.0, ix2 - ix1), max(0.0, iy2 - iy1)
    inter = iw * ih
    union = (a[2] - a[0]) * (a[3] - a[1]) + (b[2] - b[0]) * (b[3] - b[1]) - inter
    return inter / union if union > 0 else 0.0


def load_labels(stem: str, lbl_dir: Path, w: int, h: int):
    f = lbl_dir / f"{stem}.txt"
    out = []
    if not f.is_file():
        return out
    for line in f.read_text().splitlines():
        if not line.strip():
            continue
        c, x, y, bw, bh = line.split()
        c = int(c)
        x, y, bw, bh = map(float, (x, y, bw, bh))
        out.append((c, (x - bw / 2) * w, (y - bh / 2) * h,
                    (x + bw / 2) * w, (y + bh / 2) * h))
    return out


def score(weights: str, imgs, lbl_dir: Path, imgsz: int, n_classes: int):
    """Per-size-bucket and per-class TP/FP/FN."""
    from ultralytics import YOLO

    model = YOLO(weights)
    known = set(range(len(model.names)))
    size_stat = defaultdict(lambda: {"tp": 0, "fp": 0, "fn": 0})
    cls_stat = defaultdict(lambda: {"tp": 0, "fp": 0, "fn": 0})

    for img in imgs:
        r = model(str(img), imgsz=imgsz, conf=0.25, verbose=False, quantize=16)[0]
        h, w = r.orig_shape
        gt = load_labels(img.stem, lbl_dir, w, h)

        preds = []
        if r.boxes is not None:
            for b in r.boxes:
                c = int(b.cls[0])
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
                g = gt[best_i]
                bk = bucket((g[3] - g[1]) * (g[4] - g[2]))
                size_stat[bk]["tp"] += 1
                cls_stat[g[0]]["tp"] += 1
                used.add(best_i)
            else:
                bk = bucket((p[3] - p[1]) * (p[4] - p[2]))
                size_stat[bk]["fp"] += 1
                cls_stat[p[0]]["fp"] += 1

        for i, g in enumerate(gt):
            if i in used:
                continue
            # A class this model was never trained on cannot be a miss for it.
            if g[0] not in known:
                continue
            bk = bucket((g[3] - g[1]) * (g[4] - g[2]))
            size_stat[bk]["fn"] += 1
            cls_stat[g[0]]["fn"] += 1

    return size_stat, cls_stat


def prf(d):
    tp, fp, fn = d["tp"], d["fp"], d["fn"]
    p = tp / (tp + fp) if (tp + fp) else 0.0
    r = tp / (tp + fn) if (tp + fn) else 0.0
    f = 2 * p * r / (p + r) if (p + r) else 0.0
    return p, r, f


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--data", type=Path,
                    default=Path("data/detection_train/gujarat.yaml"))
    ap.add_argument("--b", dest="before", default="models_gujarat_yolov8s.pt",
                    help="Currently deployed model.")
    ap.add_argument("--a", dest="after",
                    default="runs/detect/output/detector_train/gujarat_v5/weights/best.pt")
    ap.add_argument("--imgsz", type=int, default=1280)
    ap.add_argument("--limit", type=int, default=0,
                    help="Score only the first N val images (0 = all).")
    args = ap.parse_args()

    import yaml as _yaml
    root = Path(_yaml.safe_load(args.data.read_text())["path"])
    imgs = sorted((root / "images" / "val").glob("*.jpg"))
    if args.limit:
        imgs = imgs[:args.limit]
    lbl = root / "labels" / "val"
    names = _yaml.safe_load(args.data.read_text())["names"]

    print(f"val images : {len(imgs)}")
    print(f"before     : {args.before}")
    print(f"after      : {args.after}\n")

    results = {}
    for tag, w in (("before", args.before), ("after", args.after)):
        print(f"scoring {tag}...", flush=True)
        results[tag] = score(w, imgs, lbl, args.imgsz, len(names))

    print()
    print(f"{'':<9} {'':<8} {'TP':>6} {'FP':>6} {'FN':>6} "
          f"{'prec':>7} {'recall':>7} {'F1':>7}")
    print("-" * 62)
    for bk in ("small", "medium", "large"):
        for tag in ("before", "after"):
            d = results[tag][0][bk]
            p, r, f = prf(d)
            print(f"{bk if tag=='before' else '':<9} {tag:<8} {d['tp']:>6} "
                  f"{d['fp']:>6} {d['fn']:>6} {p:>7.3f} {r:>7.3f} {f:>7.3f}")
        print()

    print("--- AP-style F1 by size (the headline numbers) ---")
    for bk, label in (("small", "AP_S  (<32px)"), ("medium", "AP_M  (32-96px)"),
                      ("large", "AP_L  (>96px)")):
        fb = prf(results["before"][0][bk])[2]
        fa = prf(results["after"][0][bk])[2]
        delta = (fa - fb) / max(fb, 1e-9) * 100
        print(f"  {label:<18} {fb:.3f} -> {fa:.3f}   ({delta:+.1f}%)")

    print("\n--- per class (after) ---")
    for ci, cname in enumerate(names):
        d = results["after"][1][ci]
        if d["tp"] + d["fp"] + d["fn"] == 0:
            continue
        p, r, f = prf(d)
        print(f"  {cname:<12} TP {d['tp']:>5}  FN {d['fn']:>5}  "
              f"P {p:.3f}  R {r:.3f}  F1 {f:.3f}")

    print("\nREMINDER: scored against SAHI-teacher labels, not human ground")
    print("truth. This measures agreement with the teacher on unseen frames.")


if __name__ == "__main__":
    main()
