"""backend/scripts/eval_by_daypart.py — did the DAYLIGHT frames actually help?

WHY AN AGGREGATE F1 CANNOT ANSWER THIS
  The previous model was trained only on footage from the 21:00 start of
  each recording - it had never seen daylight. The new one is trained on a
  dataset that now includes ~1,065 morning frames (07:00 / 08:30).

  Compare the two on one blended validation set and a single number comes
  out that conflates two different effects:
      - daytime accuracy going UP   (the point of the exercise)
      - night accuracy going DOWN   (catastrophic forgetting, a real risk
        when you add a large new domain and retrain)
  Either could dominate. An overall gain could still hide a night
  regression - and night is when loitering and watchlist alerts matter most.

  So score the SAME model on two disjoint slices and report both.

HOW FRAMES ARE SLICED
  Filename encodes provenance:
      *_seek<ts>_0700_* / *_0830_*  -> daylight (seek_harvest.py)
      everything else               -> night/evening (sequential harvests,
                                       all taken from the 21:00 file start)
  This is provenance, not a brightness guess, so it does not inherit the
  known mislabelling where daylight (~143 luma) and lit-night (~100 luma)
  both fall in the 'mid' bucket.

USAGE
  python -m backend.scripts.eval_by_daypart --baseline models_gujarat_yolov8s.pt \
      --tuned runs/detect/output/detector_train/<name>/weights/best.pt
"""
from __future__ import annotations

import argparse
import re
import sys
from pathlib import Path

# Reuse the honest scorer: common class space + IoU matching against the
# teacher labels. Importing it keeps the two scripts from drifting apart.
from backend.scripts.eval_detector import COCO_TO_IDX, _iou, _load_labels

DAYLIGHT_RE = re.compile(r"_seek\d+_(\d{4})_")


def is_daylight(p: Path) -> bool:
    """True for frames pulled by seek_harvest.py at a daylight target."""
    m = DAYLIGHT_RE.search(p.name)
    if not m:
        return False
    hhmm = int(m.group(1))
    return 500 <= hhmm <= 1800          # 05:00-18:00


def score(weights: str, imgs: list[Path], lbl_dir: Path, imgsz: int,
          from_coco: bool, quantize: int | None) -> dict:
    from ultralytics import YOLO

    model = YOLO(weights)
    kw = {"quantize": quantize} if quantize else {}
    tp = fp = fn = 0
    for img in imgs:
        r = model(str(img), imgsz=imgsz, conf=0.45, verbose=False, **kw)[0]
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
            if best >= 0.5:
                tp += 1
                used.add(best_i)
            else:
                fp += 1
        fn += len(gt) - len(used)

    prec = tp / (tp + fp) if (tp + fp) else 0.0
    rec = tp / (tp + fn) if (tp + fn) else 0.0
    f1 = 2 * prec * rec / (prec + rec) if (prec + rec) else 0.0
    return {"n": len(imgs), "tp": tp, "fp": fp, "fn": fn,
            "precision": prec, "recall": rec, "f1": f1}


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--data", type=Path,
                    default=Path("data/detection_train/gujarat.yaml"))
    ap.add_argument("--baseline", default="models_gujarat_yolov8s.pt",
                    help="Currently deployed model (already remapped, so it "
                         "is NOT scored as COCO).")
    ap.add_argument("--tuned", required=True)
    ap.add_argument("--imgsz", type=int, default=1280)
    ap.add_argument("--quantize", type=int, default=16,
                    help="16 = FP16, matching the deployed config.")
    ap.add_argument("--baseline-is-coco", action="store_true",
                    help="Set only when comparing against STOCK yolov8s.")
    args = ap.parse_args()

    import yaml as _yaml
    root = Path(_yaml.safe_load(args.data.read_text())["path"])
    val = sorted((root / "images" / "val").glob("*.jpg"))
    lbl = root / "labels" / "val"
    if not val:
        print(f"No val images under {root}", file=sys.stderr)
        sys.exit(1)

    day = [p for p in val if is_daylight(p)]
    night = [p for p in val if not is_daylight(p)]
    print(f"val split: {len(day)} daylight / {len(night)} night-evening "
          f"(total {len(val)})")
    if not day:
        print("\nNo daylight frames in the val split. Re-run autolabel_frames\n"
              "after the seek harvest so the new frames are split in.",
              file=sys.stderr)
        sys.exit(1)

    rows = []
    for label, weights, from_coco in (
        ("baseline", args.baseline, args.baseline_is_coco),
        ("tuned", args.tuned, False),
    ):
        for slice_name, imgs in (("daylight", day), ("night", night)):
            if not imgs:
                continue
            print(f"  scoring {label:<8} on {slice_name} ({len(imgs)})...",
                  flush=True)
            r = score(weights, imgs, lbl, args.imgsz, from_coco, args.quantize)
            rows.append((label, slice_name, r))

    print()
    print(f"{'model':<10} {'slice':<10} {'n':>5} {'TP':>6} {'FP':>6} {'FN':>6} "
          f"{'prec':>7} {'recall':>7} {'F1':>7}")
    print("-" * 74)
    for label, slice_name, r in rows:
        print(f"{label:<10} {slice_name:<10} {r['n']:>5} {r['tp']:>6} {r['fp']:>6} "
              f"{r['fn']:>6} {r['precision']:>7.3f} {r['recall']:>7.3f} "
              f"{r['f1']:>7.3f}")

    def get(lbl_, sl):
        for a, b, r in rows:
            if a == lbl_ and b == sl:
                return r
        return None

    print("\n--- verdict ---")
    for sl in ("daylight", "night"):
        b, t = get("baseline", sl), get("tuned", sl)
        if not b or not t:
            continue
        d = t["f1"] - b["f1"]
        pct = d / max(b["f1"], 1e-9) * 100
        print(f"{sl:<9} F1 {b['f1']:.3f} -> {t['f1']:.3f}  ({pct:+.1f}%)")

    bn, tn = get("baseline", "night"), get("tuned", "night")
    if bn and tn:
        drop = bn["f1"] - tn["f1"]
        print()
        if drop > 0.02:
            print("WARNING: NIGHT PERFORMANCE REGRESSED. Adding a large new")
            print("domain can cause catastrophic forgetting. Do NOT deploy on")
            print("the strength of a daylight gain alone - night is when")
            print("loitering and watchlist alerts matter most.")
        else:
            print("Night held steady - the daylight gain is not being paid")
            print("for out of night accuracy.")

    print("\nREMINDER: labels are teacher-generated. This measures agreement")
    print("with the SAHI teacher, not ground-truth accuracy.")


if __name__ == "__main__":
    main()
