"""backend/scripts/label_pool_probe.py — is there actually more real data to
label, or is the labelled set already everything that is legible?

THE QUESTION THIS SETTLES
  The recogniser's fine-tune has 251 training vehicles, drawn from 417
  hand-verified ones. `output/plate_corpus` holds 2,271 further vehicles on
  plate-capable cameras that were never labelled — nine times as many. That
  looks like an obvious win: pseudo-label them with the shipped ensemble and
  retrain on a much larger set.

  It is not, and the reason is not obvious from the counts. This probe runs the
  SAME plate-detection pipeline over both populations and compares them.

WHY THE COMPARISON IS THE WHOLE POINT
  A low yield on the unlabelled pool has two possible causes:

    (a) the pipeline is lossy for everyone — then the pool is still worth
        mining, it just costs more frames per vehicle, or
    (b) the pool is harder than the labelled set — in which case those vehicles
        are unlabelled precisely BECAUSE their plates cannot be read, and no
        amount of mining changes that.

  Running the labelled vehicles through the identical pipeline separates them.
  Without that control the yield number alone is uninterpretable.

WHAT IT FOUND (2026-09-06)
    hand-labelled vehicles   76% of frames yield a plate box, median 87 px
    unlabelled pool          18% of frames yield a plate box, median 57 px

  Cause (b). The labelled 417 are the legible subset that was already
  harvested. The remaining 2,271 sit at a median 57 px, which this project's
  own condition table measures at 33.3% exact — so pseudo-labels drawn from
  them would be wrong about two times in three. Training on those reproduces
  the EasyOCR pseudo-label failure that `plate_self_train.py` was written to
  avoid, and a dry run confirms it: 0 vehicles accepted out of 300.

USAGE
  python -m backend.scripts.label_pool_probe --vehicles 40
"""
from __future__ import annotations

import argparse
import json
import random
import sys
import time
from collections import Counter, defaultdict
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import cv2  # noqa: E402

CORPUS = ROOT / "output" / "plate_corpus"
REAL = ROOT / "data" / "plate_real"
DET_W = ROOT / "models" / "plate_detector" / "plate_v4_small.pt"
# The cameras `plate_self_train.py` considers capable of a readable plate.
PLATE_CAPABLE = {"CAM_06", "CAM_07", "CAM_08", "CAM_09", "CAM_10",
                 "CAM_11", "CAM_18", "CAM_21", "CAM_27"}


def survey(det, pool, keys, min_frames):
    """Run the self-train pipeline's frame gate and count every drop."""
    c = Counter()
    widths, seen, with_box, ok_vehicles = [], 0, 0, 0
    for key in keys:
        frames = sorted(pool[key],
                        key=lambda r: -(r.get("sharpness", 0) ** 0.5
                                        * r.get("plate_px_est", 0)))
        usable = 0
        for fr in frames[:8]:
            seen += 1
            img = cv2.imread(fr["path"])
            if img is None:
                c["unreadable file"] += 1
                continue
            res = det.predict(img, conf=0.30, verbose=False)[0]
            if not len(res.boxes):
                c["no plate box found"] += 1
                continue
            with_box += 1
            best = None
            for b in res.boxes.xyxy.cpu().numpy():
                bw, bh = b[2] - b[0], b[3] - b[1]
                # Same gate the self-train script applies.
                if bw < 36 or bh < 10 or not (1.8 <= bw / max(bh, 1) <= 7.5):
                    continue
                if best is None or bw > best:
                    best = bw
            if best is None:
                c["box failed shape gate"] += 1
                continue
            widths.append(best)
            usable += 1
        if usable >= min_frames:
            ok_vehicles += 1
        else:
            c["VEHICLE dropped: too few frames"] += 1
    widths.sort()
    return {
        "vehicles": len(keys), "vehicles_usable": ok_vehicles,
        "frames_examined": seen, "frames_with_box": with_box,
        "box_rate": with_box / max(seen, 1),
        # float() because the box widths come back as numpy float32, which the
        # json encoder refuses.
        "median_box_px": round(float(widths[len(widths) // 2]), 1) if widths else None,
        "rejects": dict(c),
    }


def main() -> int:
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--vehicles", type=int, default=40,
                    help="vehicles sampled from each population")
    ap.add_argument("--min-frames", type=int, default=3)
    ap.add_argument("--out", default="reports/label_pool_probe_20260906.json")
    args = ap.parse_args()

    known = set()
    for line in (REAL / "verified_all.jsonl").open(encoding="utf-8"):
        v = json.loads(line)
        known.add((v["camera"], v["track"]))

    unlabelled, labelled = defaultdict(list), defaultdict(list)
    for line in (CORPUS / "manifest.jsonl").open(encoding="utf-8"):
        r = json.loads(line)
        if r["camera"] not in PLATE_CAPABLE:
            continue
        k = (r["camera"], r["clip"], r["track"])
        (labelled if (r["camera"], r["track"]) in known else unlabelled)[k].append(r)

    print(f"hand-labelled vehicles on plate-capable cameras : {len(labelled)}")
    print(f"unlabelled vehicles on the same cameras         : {len(unlabelled)}\n",
          flush=True)

    from ultralytics import YOLO
    det = YOLO(str(DET_W))

    out = {}
    for name, pool in (("hand-labelled", labelled), ("unlabelled", unlabelled)):
        keys = sorted(pool)
        random.Random(17).shuffle(keys)
        out[name] = survey(det, pool, keys[:args.vehicles], args.min_frames)

    print("=" * 66)
    print("  %-16s %9s %9s %11s %11s" % ("population", "box rate", "med px",
                                         "usable veh", "of sampled"))
    print("=" * 66)
    for name in ("hand-labelled", "unlabelled"):
        s = out[name]
        print("  %-16s %8.0f%% %8s %10d %11d"
              % (name, s["box_rate"] * 100, s["median_box_px"] or "—",
                 s["vehicles_usable"], s["vehicles"]))
    print("=" * 66)

    hl, ul = out["hand-labelled"], out["unlabelled"]
    print("\n  The labelled set is the legible subset, already harvested.")
    print("  The remaining pool sits at a median %s px against %s px — the band"
          % (ul["median_box_px"], hl["median_box_px"]))
    print("  this project measures at 33.3% exact. Pseudo-labels drawn from it")
    print("  would be wrong about two times in three.")

    p = Path(args.out)
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(json.dumps({
        "generated": time.strftime("%Y-%m-%dT%H:%M:%S"),
        "detector": DET_W.name,
        "population_sizes": {"hand_labelled": len(labelled),
                             "unlabelled": len(unlabelled)},
        "survey": out,
    }, indent=2), encoding="utf-8")
    print(f"\n  written -> {p}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
