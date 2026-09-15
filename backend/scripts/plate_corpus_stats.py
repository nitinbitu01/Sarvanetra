"""backend/scripts/plate_corpus_stats.py — corpus summary, split by detector.

WHY SPLIT BY MODEL
  Clips 1-27 were mined with the YOLOv8s Gujarat model. If a different
  detector mines the rest, the corpus holds crops from two sources - and any
  later comparison ("CAM_11 yielded less than CAM_09") cannot separate a
  camera difference from a model change unless the split is visible.

  Worse, `plate_px_est` is derived from bounding-box WIDTH. Two detectors that
  both find the same vehicle but draw slightly tighter or looser boxes will
  produce different estimates for identical footage, which shifts which tracks
  clear the eligibility threshold. So the per-model width distribution is
  printed too: if the medians differ materially, the threshold needs adjusting
  for the newer half rather than being silently inconsistent.

  Rows written before the `model` field existed are reported as "untagged"
  rather than being guessed at.

USAGE
  python -m backend.scripts.plate_corpus_stats
"""
from __future__ import annotations

import json
from collections import defaultdict
from pathlib import Path

import numpy as np

MANIFEST = Path("output/plate_corpus/manifest.jsonl")


def main() -> None:
    if not MANIFEST.is_file():
        raise SystemExit(f"no manifest at {MANIFEST}")

    rows = [json.loads(l) for l in MANIFEST.open(encoding="utf-8")]
    by_model: dict[str, list] = defaultdict(list)
    for r in rows:
        by_model[r.get("model", "untagged (mined before model tagging)")].append(r)

    print(f"crops in manifest : {len(rows)}")
    print(f"distinct detectors: {len(by_model)}\n")

    medians = {}
    for model, rs in sorted(by_model.items(), key=lambda kv: -len(kv[1])):
        vw = np.array([r["veh_w"] for r in rs], dtype=float)
        pe = np.array([r["plate_px_est"] for r in rs], dtype=float)
        cams = sorted({r["camera"] for r in rs})
        tracks = {(r["camera"], r["clip"], r["track"]) for r in rs}
        medians[model] = float(np.median(vw))
        print(f"--- {model} ---")
        print(f"  crops        : {len(rs):,}")
        print(f"  tracks       : {len(tracks):,}")
        print(f"  cameras ({len(cams)}) : {', '.join(cams)}")
        print(f"  veh_w        : median {np.median(vw):.0f}px  "
              f"p10 {np.percentile(vw,10):.0f}  p90 {np.percentile(vw,90):.0f}")
        print(f"  plate_px_est : median {np.median(pe):.0f}px  "
              f"p90 {np.percentile(pe,90):.0f}")
        print()

    if len(medians) > 1:
        vals = list(medians.items())
        base_name, base = vals[0]
        print("--- cross-model comparison ---")
        print("NOTE: this compares whatever each detector happened to mine, "
              "which is\nnot the same footage. For a true A/B, mine one "
              "already-done clip with the\nnew detector and compare that clip "
              "against its original rows.\n")
        for name, med in vals[1:]:
            shift = (med - base) / max(base, 1e-9) * 100
            print(f"  {name} vs {base_name}: median veh_w {shift:+.1f}%")
            if abs(shift) > 10:
                print(f"    -> over 10%: eligibility thresholds are NOT "
                      f"comparable across the two halves. Adjust "
                      f"--min-plate-px for the newer detector.")
            else:
                print("    -> under 10%: thresholds remain broadly comparable.")

    print("\n--- per camera (all models) ---")
    per_cam: dict[str, set] = defaultdict(set)
    for r in rows:
        per_cam[r["camera"]].add((r["clip"], r["track"]))
    for cam in sorted(per_cam, key=lambda c: -len(per_cam[c])):
        models = {r.get("model", "untagged") for r in rows
                  if r["camera"] == cam}
        tag = "" if len(models) == 1 else "   <- MIXED DETECTORS"
        print(f"  {cam:<9} {len(per_cam[cam]):>6} tracks{tag}")


if __name__ == "__main__":
    main()
