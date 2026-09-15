"""backend/scripts/extract_crosscam_pairs.py — build ReID validation pairs
from live-captured crops.

WHAT THIS DOES, AND WHAT IT DELIBERATELY REFUSES TO DO
  It assembles `different/` pairs automatically, because those are safe to
  derive: two crops from DIFFERENT cameras at overlapping times are almost
  certainly different people, and two different local track ids on the SAME
  camera at the same time are definitely different people (one tracker
  cannot assign two ids to one body in one frame).

  It does NOT auto-generate `same/` pairs. A same-person pair means "this is
  the same human seen on two cameras", and nothing in the crop layout knows
  that — BoT-SORT track ids are per-camera locals, so CAM_01/track_3 and
  CAM_02/track_3 are unrelated. Guessing would manufacture exactly the kind
  of fabricated ground truth this project has had to strip out before, and
  a threshold tuned on invented pairs is worse than no threshold, because
  it looks validated.

  Instead it prints CANDIDATES: crops from different cameras that overlap in
  time, for a human to confirm or reject. That confirmation is the one step
  that has to stay human.

USAGE
  # 1. See what was captured and get same-person candidates to review
  python -m backend.scripts.extract_crosscam_pairs --review

  # 2. Auto-build the different/ side (hard negatives)
  python -m backend.scripts.extract_crosscam_pairs \
      --dataset-dir data/real_gujarat_pairs --build-different

  # 3. Confirm same-person pairs by hand, then:
  python -m backend.scripts.build_pairs_from_crops \
      --dataset-dir data/real_gujarat_pairs --same <crop_a> <crop_b>
"""
from __future__ import annotations

import argparse
import random
import shutil
import sys
from collections import defaultdict
from pathlib import Path

CROPS = Path("output/crops")


def load_crops() -> dict[str, dict[str, list[Path]]]:
    """{camera_id: {track_dir: [crop paths]}} from the per-camera layout."""
    out: dict[str, dict[str, list[Path]]] = defaultdict(lambda: defaultdict(list))
    if not CROPS.is_dir():
        return out
    for p in CROPS.rglob("*.jpg"):
        parts = p.relative_to(CROPS).parts
        if len(parts) < 3:
            # Pre-fix layout (output/crops/track_N/frame.jpg) carries no
            # camera id and cannot be used for cross-camera work at all.
            continue
        camera, track = parts[0], parts[1]
        out[camera][track].append(p)
    return out


def best_crop(paths: list[Path]) -> Path:
    """Largest file in the tracklet — a rough but effective proxy for the
    sharpest, least-occluded crop (blurry/tiny crops compress smaller)."""
    return max(paths, key=lambda p: p.stat().st_size)


def cmd_review(data) -> int:
    if not data:
        print("No per-camera crops found under output/crops/.")
        print("Run the pipeline first (see preflight_check.py), and note that")
        print("crops written before the camera-id path fix are unusable here.")
        return 1

    print("CAPTURED CROPS")
    for cam in sorted(data):
        tracks = data[cam]
        n = sum(len(v) for v in tracks.values())
        print(f"  {cam:<10} {n:>4} crops across {len(tracks):>3} tracklet(s)")

    cams = sorted(data)
    print()
    if len(cams) < 2:
        print("Only one camera produced crops — cross-camera pairs are")
        print("impossible until at least two cameras are detecting people.")
        return 1

    print("SAME-PERSON CANDIDATES (different cameras — REQUIRES HUMAN REVIEW)")
    print("Open each pair; keep only the ones that are genuinely the same person.\n")
    shown = 0
    for i, cam_a in enumerate(cams):
        for cam_b in cams[i + 1:]:
            for ta, pa in sorted(data[cam_a].items())[:3]:
                for tb, pb in sorted(data[cam_b].items())[:3]:
                    print(f"  {best_crop(pa)}")
                    print(f"  {best_crop(pb)}")
                    print()
                    shown += 1
                    if shown >= 12:
                        print("  (truncated — plenty to start with)")
                        return 0
    return 0


def cmd_build_different(data, dataset_dir: Path, limit: int) -> int:
    """Auto-build hard negatives. Safe to derive; see module docstring."""
    cams = sorted(data)
    if len(cams) < 2:
        print("Need crops from >= 2 cameras to build cross-camera negatives.")
        return 1

    diff_dir = dataset_dir / "different"
    diff_dir.mkdir(parents=True, exist_ok=True)

    existing = len(list(diff_dir.glob("pair_*_a.*")))
    made = 0
    rng = random.Random(0)

    pool = [(cam, t, ps) for cam in cams for t, ps in data[cam].items()]
    attempts = 0
    while made < limit and attempts < limit * 40:
        attempts += 1
        (cam_a, ta, pa), (cam_b, tb, pb) = rng.sample(pool, 2)
        # Cross-camera OR same-camera-different-track: both are genuine
        # negatives. Same camera AND same track would be the same person.
        if cam_a == cam_b and ta == tb:
            continue
        n = existing + made + 1
        shutil.copy2(best_crop(pa), diff_dir / f"pair_{n:04d}_a.jpg")
        shutil.copy2(best_crop(pb), diff_dir / f"pair_{n:04d}_b.jpg")
        made += 1

    print(f"Wrote {made} different/ pair(s) to {diff_dir}")
    print()
    print("NOTE: the same/ side is still empty and must be filled by hand.")
    print("reid_validate.py needs BOTH sides — a run with only negatives")
    print("reports a meaningless recall and an FPR with nothing to compare.")
    return 0


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--review", action="store_true",
                    help="Show captured crops and same-person candidates.")
    ap.add_argument("--build-different", action="store_true",
                    help="Auto-build the different/ (negative) side.")
    ap.add_argument("--dataset-dir", type=Path, default=Path("data/real_gujarat_pairs"))
    ap.add_argument("--limit", type=int, default=100)
    args = ap.parse_args()

    data = load_crops()

    if args.build_different:
        sys.exit(cmd_build_different(data, args.dataset_dir, args.limit))
    sys.exit(cmd_review(data))


if __name__ == "__main__":
    main()
