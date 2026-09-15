"""backend/scripts/build_pairs_from_crops.py — turn two picked crops into a
labelled validation pair, in the exact folder contract reid_validate.py (and
its face-mode counterpart) already expect.

USAGE
  # Same person, seen on two different cameras:
  python -m backend.scripts.build_pairs_from_crops \
      --dataset-dir data/real_gujarat_pairs \
      --same data/labelling_crops/person/CAM-1/track_0007.jpg \
             data/labelling_crops/person/CAM-3/track_0021.jpg \
      --lighting day --quality high --angle frontal

  # Confidently two different people:
  python -m backend.scripts.build_pairs_from_crops \
      --dataset-dir data/real_gujarat_pairs \
      --different data/labelling_crops/person/CAM-1/track_0003.jpg \
                  data/labelling_crops/person/CAM-5/track_0044.jpg

WHY --different PAIRS MATTER AS MUCH AS --same ONES
  The false-positive rate — the number that actually gates auto-dispatch —
  is computed from the different/ set. A handful of same/ pairs and no
  different/ pairs produces a report with an undefined FPR, same as the
  synthetic-proxy failure mode this whole validation pipeline exists to
  avoid. Label both sides, and label difficult different/ pairs (similar
  build, similar clothing) if you have them — those are what a real
  false-positive rate is measured against, not two obviously unrelated
  people.

WHY THIS DOESN'T DEDUPLICATE OR VALIDATE IMAGE CONTENT
  It copies whatever two files you point it at. It cannot tell you whether
  they're actually the same/different person — that judgment is the one
  part of this pipeline that has to stay human. What it guards against is
  the mechanical mistakes: wrong folder, colliding pair numbers, a
  metadata.json that doesn't parse.
"""
from __future__ import annotations

import argparse
import json
import shutil
import sys
from pathlib import Path


def next_pair_number(subdir: Path) -> int:
    if not subdir.is_dir():
        return 1
    existing = sorted(subdir.glob("pair_*_a.*"))
    if not existing:
        return 1
    nums = []
    for p in existing:
        # pair_0007_a.jpg -> 7
        stem = p.stem  # pair_0007_a
        parts = stem.split("_")
        if len(parts) >= 3 and parts[1].isdigit():
            nums.append(int(parts[1]))
    return (max(nums) + 1) if nums else len(existing) + 1


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--dataset-dir", required=True, type=Path,
                        help="e.g. data/real_gujarat_pairs or "
                             "data/real_gujarat_face_pairs — this is the "
                             "exact directory you'll later pass to "
                             "reid_validate.py's --dataset-dir.")
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument("--same", nargs=2, metavar=("CROP_A", "CROP_B"))
    group.add_argument("--different", nargs=2, metavar=("CROP_A", "CROP_B"))
    parser.add_argument("--lighting", choices=["day", "night"], default=None)
    parser.add_argument("--quality", choices=["high", "low"], default=None)
    parser.add_argument("--angle", choices=["frontal", "off_axis"], default=None)
    args = parser.parse_args()

    crop_a, crop_b = (Path(p) for p in (args.same or args.different))
    for p in (crop_a, crop_b):
        if not p.is_file():
            print(f"Crop not found: {p}", file=sys.stderr)
            sys.exit(1)

    is_same = args.same is not None
    subdir_name = "same" if is_same else "different"
    subdir = args.dataset_dir / subdir_name
    subdir.mkdir(parents=True, exist_ok=True)

    n = next_pair_number(subdir)
    pair_id = f"pair_{n:04d}"
    dest_a = subdir / f"{pair_id}_a{crop_a.suffix}"
    dest_b = subdir / f"{pair_id}_b{crop_b.suffix}"
    shutil.copy2(crop_a, dest_a)
    shutil.copy2(crop_b, dest_b)

    meta_path = args.dataset_dir / "metadata.json"
    metadata = {}
    if meta_path.is_file():
        try:
            metadata = json.loads(meta_path.read_text())
        except json.JSONDecodeError:
            print(f"WARNING: {meta_path} did not parse as JSON — starting a "
                  f"fresh one rather than risk overwriting it. Check it by "
                  f"hand.", file=sys.stderr)
            metadata = {}

    entry = {}
    if args.lighting: entry["lighting"] = args.lighting
    if args.quality: entry["quality"] = args.quality
    if args.angle: entry["angle"] = args.angle
    if entry:
        metadata[pair_id] = entry
        meta_path.write_text(json.dumps(metadata, indent=2))

    print(f"Wrote {subdir_name}/{pair_id}: {dest_a.name}, {dest_b.name}"
          f"  (source: {crop_a} , {crop_b})")
    if not entry:
        print("No --lighting/--quality/--angle given — this pair won't "
              "appear in the fairness/stratification breakdown, but it "
              "still counts toward the headline FPR/accuracy numbers.")


if __name__ == "__main__":
    main()
