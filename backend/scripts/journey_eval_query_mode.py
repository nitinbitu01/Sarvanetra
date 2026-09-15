"""backend/scripts/journey_eval_query_mode.py — measure the feature the brief
actually asks for, which is not the one that was built.

THE DISTINCTION THAT CHANGES THE PROBLEM
  What was built and measured: take every sighting, find its best partner on
  another camera. 840 sightings, perhaps 10-20 with a real partner, so a base
  rate around 2%. Reaching 90% precision from a 2% prior needs a likelihood
  ratio of roughly 440:1, which plate readings that are 59% exact cannot
  supply. It measured 10%.

  What the brief asks for: "type in any license plate number, and the system
  draws that car's journey". An officer types a plate they have a reason to
  believe exists - from an FIR, a witness, a hotlist. The question then becomes
  "did THIS vehicle pass THIS camera", asked 16 times. If the vehicle passed
  two of sixteen cameras, the base rate is 12.5%, and 90% precision needs about
  63:1 - roughly seven times less evidence.

  Same index, same scores, a different question and a very different prior.
  The earlier number answered the wrong one.

WHAT IS MEASURED HERE
  For each labelled vehicle, its plate is used as a query and every camera is
  asked whether that vehicle appears there. Ground truth comes from the human
  labels: the camera where it was read is a positive, every other camera is a
  negative. Precision and recall are then computed over camera-level decisions,
  which is what the officer actually sees.

  Vehicles the recogniser trained on are reported separately, as always.

USAGE
  python -m backend.scripts.journey_eval_query_mode
"""
from __future__ import annotations

import argparse
import json
from collections import defaultdict
from pathlib import Path

import numpy as np

from backend.scripts.plate_final_model import load_data, split
from backend.services.journey_search import JourneySearch

REAL = Path("data/plate_real")
INDEX = Path("output/journey_index")


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--index", default=str(INDEX))
    args = ap.parse_args()

    js = JourneySearch(Path(args.index))
    recs = js.records
    cameras = sorted({r["camera"] for r in recs})
    by_cam = defaultdict(list)
    for i, r in enumerate(recs):
        by_cam[r["camera"]].append(i)
    print(f"index: {len(recs)} sightings across {len(cameras)} cameras")

    truth = {}
    for line in (REAL / "verified_all.jsonl").open(encoding="utf-8"):
        v = json.loads(line)
        if v.get("text"):
            truth[(v["camera"], v["track"])] = v["text"]
    where = {}
    for i, r in enumerate(recs):
        where.setdefault((r["camera"], r["track"]), i)

    # camera(s) each plate is genuinely on, from the human labels
    plate_cams = defaultdict(set)
    for (cam, trk), p in truth.items():
        if (cam, trk) in where:
            plate_cams[p].add(cam)

    _, by_track = load_data()
    test_k, val_k, train_k = split(by_track, 1000)
    trained = {truth[k] for k in train_k if k in truth}

    queries = sorted(plate_cams)
    print(f"queries (labelled plates present in the index): {len(queries)}\n",
          flush=True)

    # For every query and every camera, the best score on that camera.
    rows = []
    for n, plate in enumerate(queries, 1):
        s = js.ctc_scores(plate)
        for cam in cameras:
            idxs = by_cam[cam]
            if not idxs:
                continue
            best = float(max(s[i] for i in idxs))
            rows.append({"plate": plate, "camera": cam, "score": best,
                         "present": cam in plate_cams[plate],
                         "trained": plate in trained})
        if n % 50 == 0:
            print(f"  {n}/{len(queries)}", flush=True)

    unseen = [r for r in rows if not r["trained"]]
    pos = [r["score"] for r in unseen if r["present"]]
    neg = [r["score"] for r in unseen if not r["present"]]
    print(f"\ncamera-level decisions (unseen vehicles): {len(unseen)}")
    print(f"  vehicle present : {len(pos)}")
    print(f"  vehicle absent  : {len(neg)}")
    base = len(pos) / max(len(unseen), 1)
    print(f"  base rate       : {base*100:.1f}%")

    ps, ns = np.sort(pos), np.sort(neg)
    print("\n" + "=" * 66)
    print('"DID THIS VEHICLE PASS THIS CAMERA?"  - score distribution')
    print("=" * 66)
    print(f"{'':<12} {'p10':>9} {'median':>9} {'p90':>9}")
    print(f"{'present':<12} {ps[len(ps)//10]:>9.2f} {ps[len(ps)//2]:>9.2f} "
          f"{ps[len(ps)*9//10]:>9.2f}")
    print(f"{'absent':<12} {ns[len(ns)//10]:>9.2f} {ns[len(ns)//2]:>9.2f} "
          f"{ns[len(ns)*9//10]:>9.2f}")

    print("\n" + "=" * 66)
    print("OPERATING POINTS")
    print("=" * 66)
    print(f"{'threshold':>10} {'reported':>10} {'correct':>9} "
          f"{'PRECISION':>11} {'RECALL':>9}")
    print("-" * 66)
    best90 = None
    for th in np.arange(-30, 2, 1.5):
        rep = [r for r in unseen if r["score"] >= th]
        if len(rep) < 5:
            continue
        ok = sum(1 for r in rep if r["present"])
        prec = ok / len(rep)
        rec = ok / max(len(pos), 1)
        mark = ""
        if prec >= 0.90 and best90 is None:
            best90 = (th, prec, rec, len(rep))
            mark = "  <-- 90%"
        print(f"{th:>10.1f} {len(rep):>10} {ok:>9} {prec*100:>10.1f}% "
              f"{rec*100:>8.1f}%{mark}")
    print("=" * 66)

    if best90:
        th, prec, rec, n = best90
        print(f"\nAt score >= {th:.1f}: {prec*100:.1f}% of the camera hits the "
              f"system\nreports are real, and it finds {rec*100:.0f}% of the "
              f"cameras a vehicle\nactually passed. Base rate is "
              f"{base*100:.1f}%, so this is a "
              f"{prec/max(base,1e-9):.0f}x lift over guessing.")
    else:
        print(f"\nNo threshold reaches 90% precision at a {base*100:.1f}% base "
              f"rate.\nPlate likelihood alone is not enough evidence; an "
              f"independent signal\n(vehicle appearance) is required.")


if __name__ == "__main__":
    main()
