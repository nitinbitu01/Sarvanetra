"""Re-split the plate detector dataset onto the recogniser's holdout.

recover_plate_boxes splits by vehicle with seed 4242. The recogniser splits by
vehicle with seed 1000. Both splits are individually clean, and that is not
enough: a vehicle the detector trained on can sit in the recogniser's test
set, so an end-to-end measurement over the recogniser's holdout would score
capture rate on plates the detector was trained to find.

That is the same contamination that produced a 0.995 mAP for the previous
detector and an 80% exact-match reading earlier in this work — both measured
on data the model had seen, both wrong, neither obviously so.

The fix is to make the detector's val set a superset of the recogniser's test
set, by defining it as exactly those vehicles. Then any vehicle used for an
end-to-end number was trained on by neither model, and one measurement is
honest for both stages.

This moves files between train/ and val/ rather than re-running the recovery,
since the boxes themselves do not change.

Run:  python -m backend.scripts.align_detector_split
"""
from __future__ import annotations

import json
import random
import shutil
import sys
from collections import defaultdict
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

REAL = ROOT / "data" / "plate_real"
DATA = ROOT / "data" / "plate_detect_v2"


def main() -> int:
    # The recogniser's test split, reproduced exactly as plate_final_model and
    # plate_strict_baseline define it.
    truth = set()
    for line in (REAL / "verified_all.jsonl").open(encoding="utf-8"):
        v = json.loads(line)
        if v.get("text"):
            truth.add((v["camera"], v["track"]))
    by_track = defaultdict(list)
    for line in (REAL / "labels.jsonl").open(encoding="utf-8"):
        r = json.loads(line)
        k = (r["camera"], r["track"])
        if k in truth:
            by_track[k].append(r["file"])
    kk = sorted(by_track)
    random.Random(1000).shuffle(kk)
    rec_test = set(kk[:max(20, int(len(kk) * 0.20))])
    print(f"recogniser test vehicles : {len(rec_test)}")

    rows = [json.loads(l) for l in (DATA / "boxes.jsonl").open(encoding="utf-8")]
    det_vehicles = {(r["camera"], r["track"]) for r in rows}
    print(f"detector dataset vehicles: {len(det_vehicles)}")

    val_vehicles = det_vehicles & rec_test
    print(f"new val (their overlap)  : {len(val_vehicles)}")
    print(f"new train                : {len(det_vehicles - val_vehicles)}\n")

    if not val_vehicles:
        print("No overlap — nothing to align.", file=sys.stderr)
        return 1

    # Where each image currently is, and where it belongs.
    moved = {"to_val": 0, "to_train": 0, "stayed": 0}
    for r in rows:
        key = (r["camera"], r["track"])
        want = "val" if key in val_vehicles else "train"
        stem = Path(r["frame"]).stem
        for sub, ext in (("images", ".jpg"), ("labels", ".txt")):
            here = None
            for split in ("train", "val"):
                p = DATA / sub / split / f"{stem}{ext}"
                if p.is_file():
                    here = (split, p)
                    break
            if here is None:
                continue
            split, src = here
            if split == want:
                if sub == "images":
                    moved["stayed"] += 1
                continue
            dst = DATA / sub / want / f"{stem}{ext}"
            dst.parent.mkdir(parents=True, exist_ok=True)
            shutil.move(str(src), str(dst))
            if sub == "images":
                moved["to_val" if want == "val" else "to_train"] += 1

    n_train = len(list((DATA / "images" / "train").glob("*.jpg")))
    n_val = len(list((DATA / "images" / "val").glob("*.jpg")))
    print(f"moved to val   : {moved['to_val']} images")
    print(f"moved to train : {moved['to_train']} images")
    print(f"already correct: {moved['stayed']} images")
    print(f"\ntrain: {n_train} images   val: {n_val} images")

    # boxes.jsonl records the split too, so downstream checks stay consistent.
    with (DATA / "boxes.jsonl").open("w", encoding="utf-8") as fh:
        for r in rows:
            r["split"] = ("val" if (r["camera"], r["track"]) in val_vehicles
                          else "train")
            fh.write(json.dumps(r) + "\n")

    (DATA / "val_vehicles.json").write_text(
        json.dumps(sorted(f"{c}|{t}" for c, t in val_vehicles), indent=1),
        encoding="utf-8")
    print(f"wrote {DATA / 'val_vehicles.json'} — the vehicles any end-to-end "
          f"number must be restricted to.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
