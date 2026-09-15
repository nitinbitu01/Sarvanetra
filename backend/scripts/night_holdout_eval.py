"""backend/scripts/night_holdout_eval.py — settle v3 vs v4 on NIGHT data
neither model has seen.

WHY THIS EXISTS
  Comparing v3 and v4 on v4's val split was unfair: 713 of those 843 frames
  predate v3's training, so v3 was largely being scored on images it had
  fitted. Correcting for that on the 130 frames neither model saw flipped
  the result from "v4 is 4-11% worse" to "v4 is 2.6% better".

  But those 130 frames were ALL daylight - the deep harvest had targeted
  07:00 and 08:30. So night performance was completely unmeasured on fair
  data, and night is when loitering and watchlist alerts matter most.
  Deploying on a daylight-only result would repeat exactly the mistake the
  day/night split evaluation exists to prevent.

  The 23:30 / 02:00 harvest fixed that: those frames were collected after
  BOTH models finished training, so neither has ever seen them.

METHOD
  1. Label the night frames with the same SAHI teacher used for training,
     into a scratch directory (the real dataset is left untouched).
  2. Score v3 and v4 against those labels in the common class space.

  Both models are graded against the teacher, not human ground truth, so
  this measures agreement with the big model on unseen night footage - the
  same yardstick used throughout, and stated rather than implied.

USAGE
  python -m backend.scripts.night_holdout_eval
"""
from __future__ import annotations

import argparse
import shutil
import sys
from pathlib import Path

IMAGES_ROOT = Path("data/detection_train/images")
SCRATCH = Path("data/detection_train/_night_holdout")

COCO_TO_IDX = {0: 0, 2: 1, 3: 2, 5: 3, 7: 4}


def find_night_frames() -> list[Path]:
    """Deep-harvested frames from the night targets."""
    out = []
    for p in IMAGES_ROOT.rglob("*.jpg"):
        if "train" in p.parts or "val" in p.parts:
            continue
        if "_deep" not in p.name:
            continue
        # filenames carry the clock target: ..._deep<ts>_2330_0007.jpg
        for tag in ("_2330_", "_0200_"):
            if tag in p.name:
                out.append(p)
                break
    return sorted(out)


def label_with_teacher(frames: list[Path], teacher: str, conf: float,
                       slice_px: int) -> Path:
    from sahi import AutoDetectionModel
    from sahi.predict import get_sliced_prediction
    import cv2

    img_dir = SCRATCH / "images"
    lbl_dir = SCRATCH / "labels"
    for d in (img_dir, lbl_dir):
        if d.exists():
            shutil.rmtree(d)
        d.mkdir(parents=True, exist_ok=True)

    model = AutoDetectionModel.from_pretrained(
        model_type="ultralytics", model_path=teacher,
        confidence_threshold=conf, device="cuda:0",
    )
    print(f"labelling {len(frames)} night frame(s) with {teacher} "
          f"({slice_px}px slices)...", flush=True)

    for i, src in enumerate(frames, 1):
        img = cv2.imread(str(src))
        if img is None:
            continue
        h, w = img.shape[:2]
        res = get_sliced_prediction(
            img, model, slice_height=slice_px, slice_width=slice_px,
            overlap_height_ratio=0.2, overlap_width_ratio=0.2, verbose=0,
        )
        lines = []
        for o in res.object_prediction_list:
            cid = o.category.id
            if cid not in COCO_TO_IDX:
                continue
            bb = o.bbox
            lines.append(
                f"{COCO_TO_IDX[cid]} "
                f"{((bb.minx + bb.maxx) / 2) / w:.6f} "
                f"{((bb.miny + bb.maxy) / 2) / h:.6f} "
                f"{(bb.maxx - bb.minx) / w:.6f} "
                f"{(bb.maxy - bb.miny) / h:.6f}")
        shutil.copy2(src, img_dir / src.name)
        (lbl_dir / f"{src.stem}.txt").write_text("\n".join(lines))
        if i % 50 == 0:
            print(f"  {i}/{len(frames)}", flush=True)

    print(f"labelled -> {SCRATCH}")
    return SCRATCH


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--teacher", default="yolov8x.pt")
    ap.add_argument("--conf", type=float, default=0.5)
    ap.add_argument("--slice", type=int, default=640)
    ap.add_argument("--imgsz", type=int, default=1280)
    ap.add_argument("--v3", default="models_gujarat_yolov8s.pt")
    ap.add_argument("--v4",
                    default="runs/detect/output/detector_train/gujarat_v4/weights/best.pt")
    ap.add_argument("--skip-label", action="store_true",
                    help="Reuse labels from a previous run.")
    args = ap.parse_args()

    frames = find_night_frames()
    if len(frames) < 50:
        print(f"Only {len(frames)} night frame(s) found - run "
              f"deep_harvest --times 23:30,02:00 first.", file=sys.stderr)
        sys.exit(1)
    print(f"night hold-out frames: {len(frames)}\n")

    if not args.skip_label:
        label_with_teacher(frames, args.teacher, args.conf, args.slice)

    sys.path.insert(0, str(Path.cwd()))
    from backend.scripts.eval_by_daypart import score

    imgs = sorted((SCRATCH / "images").glob("*.jpg"))
    lbl_dir = SCRATCH / "labels"
    print(f"\nscoring {len(imgs)} unseen NIGHT frames\n")

    rows = []
    for name, w in (("v3 (deployed)", args.v3), ("v4 (new)", args.v4)):
        print(f"  scoring {name}...", flush=True)
        rows.append((name, score(w, imgs, lbl_dir, args.imgsz, False, 16)))

    print()
    print(f"{'model':<15} {'n':>5} {'TP':>6} {'FP':>6} {'FN':>6} "
          f"{'prec':>7} {'recall':>7} {'F1':>7}")
    print("-" * 64)
    for n, r in rows:
        print(f"{n:<15} {r['n']:>5} {r['tp']:>6} {r['fp']:>6} {r['fn']:>6} "
              f"{r['precision']:>7.3f} {r['recall']:>7.3f} {r['f1']:>7.3f}")

    b, t = rows[0][1]["f1"], rows[1][1]["f1"]
    delta = (t - b) / max(b, 1e-9) * 100
    print(f"\nNIGHT, unseen by both models:")
    print(f"  F1 {b:.3f} (v3) -> {t:.3f} (v4)   ({delta:+.1f}%)   n={rows[0][1]['n']}")

    print("\n--- verdict ---")
    if t >= b - 0.01:
        print("v4 holds or improves at night. Combined with the daylight")
        print("hold-out result (+2.6%), deploying v4 is justified.")
    else:
        print("v4 REGRESSED at night. Keep v3 deployed - night is when")
        print("loitering and watchlist alerts matter most, and a daylight")
        print("gain does not pay for a night loss.")
    print("\nREMINDER: teacher-generated labels; this is agreement with "
          "yolov8x, not ground truth.")


if __name__ == "__main__":
    main()
