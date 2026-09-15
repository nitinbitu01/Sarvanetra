"""backend/scripts/plate_detector_audit.py — is the existing plate detector's
0.995 mAP real, or is it reading its own training data?

THE SUSPICION
  data/plate_detect holds frames named CAM_04_..._t19116_f5708, _f5711, _f5715 -
  consecutive frames of ONE vehicle. If the train/val split was made per image,
  neighbouring frames of the same plate sat on both sides, and the val score
  measures memorisation rather than detection. That is the exact flaw that
  invalidated an earlier recogniser run, so it gets checked before this
  detector is trusted with anything.

WHAT THIS REPORTS
  1. Track overlap between train and val. Any shared (camera, track) means the
     published mAP is inflated; the size of the overlap says by how much.
  2. Camera coverage - a detector trained on four cameras may or may not hold
     on the other eighteen, and the harvest plan depends on it holding.
  3. A fresh-data check: run the detector over corpus frames from vehicles that
     appear in NEITHER split and report the hit rate. There are no boxes to
     score against there, so this is a DETECTION RATE, not precision - it
     answers "does it find plates on unseen cameras", which is what the
     harvest needs, and it is reported as such rather than dressed up as mAP.

USAGE
  python -m backend.scripts.plate_detector_audit
"""
from __future__ import annotations

import argparse
import json
import random
import re
from collections import defaultdict
from pathlib import Path

DET = Path("data/plate_detect")
CORPUS = Path("output/plate_corpus")
WEIGHTS = Path("runs/detect/runs/plate/plate_v2/weights/best.pt")

# Two naming schemes are in circulation and both must parse, because a
# filename this audit silently skips is a vehicle it cannot check for leakage.
#   CAM_04_CAM_04_0730_t19116_f5708   (data/plate_detect, original harvest)
#   CAM_09_t38526_f010944             (data/plate_detect_v2, box recovery)
NAMES = [
    re.compile(r"^(CAM[_-][\w-]+?)_CAM[_-][\w-]+?_(\d+)_t(\d+)_f\d+"),
    re.compile(r"^(CAM[_-][\w-]+?)_t(\d+)_f\d+$"),
]


def parse(stem: str):
    m = NAMES[0].match(stem)
    if m:
        return m.group(1), m.group(2), int(m.group(3))
    m = NAMES[1].match(stem)
    if m:
        return m.group(1), "", int(m.group(2))
    return None


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--probe", type=int, default=300,
                    help="Unseen corpus frames to run the detector over.")
    ap.add_argument("--conf", type=float, default=0.25)
    ap.add_argument("--weights", default=str(WEIGHTS))
    ap.add_argument("--data", default=str(DET),
                    help="Dataset whose split is audited and whose vehicles "
                         "are excluded from the fresh-data probe.")
    args = ap.parse_args()
    globals()["DET"] = Path(args.data)
    globals()["WEIGHTS"] = Path(args.weights)

    split = {}
    for s in ("train", "val"):
        names = [p.stem for p in (DET / "images" / s).glob("*.jpg")]
        split[s] = names
        print(f"{s:<6} images {len(names)}")

    parsed = {s: [parse(n) for n in v] for s, v in split.items()}
    for s, v in parsed.items():
        bad = sum(1 for x in v if x is None)
        if bad:
            print(f"  {s}: {bad} filenames did not parse - excluded from audit")

    tracks = {s: {(x[0], x[2]) for x in v if x} for s, v in parsed.items()}
    cams = {s: {x[0] for x in v if x} for s, v in parsed.items()}
    shared = tracks["train"] & tracks["val"]

    print("\n" + "=" * 66)
    print("SPLIT INTEGRITY")
    print("=" * 66)
    print(f"train vehicles      : {len(tracks['train'])}")
    print(f"val vehicles        : {len(tracks['val'])}")
    print(f"vehicles in BOTH    : {len(shared)}")
    if shared:
        frac = len(shared) / max(len(tracks["val"]), 1)
        print(f"  -> {frac*100:.0f}% of val vehicles were also trained on.")
        print("  The published mAP is inflated by memorisation and must not be")
        print("  quoted. It says nothing about unseen plates.")
    else:
        print("  -> clean: no vehicle appears on both sides.")
    print(f"\ntrain cameras       : {sorted(cams['train'])}")
    print(f"val cameras         : {sorted(cams['val'])}")

    # Fresh data: vehicles the detector has never seen, from every camera in
    # the corpus - including cameras absent from the training split, which is
    # where a detector trained on four cameras is most likely to fail.
    seen = tracks["train"] | tracks["val"]
    by_cam = defaultdict(list)
    for line in (CORPUS / "manifest.jsonl").open(encoding="utf-8"):
        r = json.loads(line)
        if (r["camera"], r["track"]) in seen:
            continue
        by_cam[r["camera"]].append(r)

    rng = random.Random(7)
    probe = []
    per_cam = max(1, args.probe // max(len(by_cam), 1))
    for cam, rows in sorted(by_cam.items()):
        probe.extend(rng.sample(rows, min(per_cam, len(rows))))
    print(f"\nunseen probe frames : {len(probe)} across {len(by_cam)} cameras",
          flush=True)

    if not WEIGHTS.is_file():
        print(f"\nweights not found at {WEIGHTS} - skipping the fresh-data run")
        return

    from ultralytics import YOLO
    model = YOLO(str(WEIGHTS))

    hits = defaultdict(int)
    total = defaultdict(int)
    widths = []
    for r in probe:
        total[r["camera"]] += 1
        res = model.predict(r["path"], conf=args.conf, verbose=False)[0]
        if len(res.boxes):
            hits[r["camera"]] += 1
            b = res.boxes.xyxy.cpu().numpy()
            widths.extend((b[:, 2] - b[:, 0]).tolist())

    print("\n" + "=" * 66)
    print(f"DETECTION RATE ON UNSEEN VEHICLES  (conf >= {args.conf})")
    print("=" * 66)
    print("Fraction of vehicle crops where a plate box was found at all.")
    print("No ground-truth boxes exist here, so this is RECALL-LIKE only -")
    print("a false box on a headlight counts as a hit and would show up as")
    print("garbage at labelling time, not here.")
    print("-" * 66)
    print(f"{'camera':<14} {'frames':>8} {'found':>8} {'rate':>8}")
    print("-" * 66)
    for cam in sorted(total):
        t, h = total[cam], hits[cam]
        star = " *" if cam not in cams["train"] else ""
        print(f"{cam:<14} {t:>8} {h:>8} {h/max(t,1)*100:>7.0f}%{star}")
    print("-" * 66)
    T, H = sum(total.values()), sum(hits.values())
    print(f"{'ALL':<14} {T:>8} {H:>8} {H/max(T,1)*100:>7.0f}%")
    if widths:
        widths.sort()
        med = widths[len(widths) // 2]
        print(f"\ndetected plate width: median {med:.0f}px  "
              f"p10 {widths[int(len(widths)*0.1)]:.0f}px  "
              f"p90 {widths[int(len(widths)*0.9)]:.0f}px")
    print("\n* = camera absent from the detector's training split")
    print("\nCompare against the EasyOCR band heuristic currently used for")
    print("harvesting, which yielded 151 crops from 1,944 vehicles (7.8%).")


if __name__ == "__main__":
    main()
