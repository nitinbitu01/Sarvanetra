"""backend/scripts/journey_eval_precision.py — what does the system do when the
plate was never there?

THE HALF OF THE MEASUREMENT THAT DECIDES WHETHER THIS IS USABLE
  Recall answers "if the vehicle is in the index, is it ranked first?" - 93.8%
  on unseen vehicles. That number says nothing about the case an investigator
  meets constantly: a plate typed from a witness statement, an FIR, or a
  half-remembered number, for a vehicle that never passed these cameras.

  Left alone, a ranking system always returns something. It will put SOME track
  at position one for any query, and an officer reading a confident-looking top
  hit has no way to tell it apart from a real find. A false link sends an
  investigation after the wrong vehicle, which is worse than returning nothing.

  So the system needs a threshold below which it says "not found", and the
  threshold has to be measured rather than assumed.

HOW THE NEGATIVES ARE MADE
  Plates that are grammatically valid, plausible for Gujarat traffic, and
  absent from the index. Two kinds, because they test different things:

    random    an unrelated valid plate. Easy to reject, and a system that
              cannot even do this is broken.
    near-miss one character changed from a plate that IS indexed. This is the
              adversarial case - it is what a typo produces, and what a
              partially-remembered plate looks like. If the score cannot
              separate these, the threshold is worthless in practice.

WHAT IS REPORTED
  The score distributions of present and absent queries, and the operating
  point that keeps a stated precision. Precision is the number that matters
  here: of the searches the system answers, how many are the right vehicle.

USAGE
  python -m backend.scripts.journey_eval_precision --negatives 200
"""
from __future__ import annotations

import argparse
import json
import random
import string
from collections import defaultdict
from pathlib import Path

import numpy as np

from backend.scripts.plate_final_model import load_data, split
from backend.services.journey_search import JourneySearch

REAL = Path("data/plate_real")
INDEX = Path("output/journey_index")
LETTERS = string.ascii_uppercase
DIGITS = string.digits


def near_miss(plate: str, rng: random.Random) -> str:
    """One character changed, respecting the position's character class.

    A near-miss that violates the plate grammar would be rejected by the
    format check before scoring ever happened, which would make the test
    easier than reality. Keeping it legal is the point.
    """
    if len(plate) < 6:
        return plate
    for _ in range(20):
        i = rng.randrange(len(plate))
        cur = plate[i]
        pool = LETTERS if cur.isalpha() else DIGITS
        new = rng.choice([c for c in pool if c != cur])
        cand = plate[:i] + new + plate[i + 1:]
        if cand != plate:
            return cand
    return plate


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--index", default=str(INDEX))
    ap.add_argument("--positives", type=int, default=120)
    ap.add_argument("--negatives", type=int, default=200)
    args = ap.parse_args()

    js = JourneySearch(Path(args.index))
    where = {}
    for i, r in enumerate(js.records):
        where.setdefault((r["camera"], r["track"]), i)

    truth = {}
    for line in (REAL / "verified_all.jsonl").open(encoding="utf-8"):
        v = json.loads(line)
        if v.get("text"):
            truth[(v["camera"], v["track"])] = v["text"]
    same_vehicle = defaultdict(set)
    for k, p in truth.items():
        if k in where:
            same_vehicle[p].add(where[k])

    _, by_track = load_data()
    test_k, val_k, train_k = split(by_track, 1000)
    trained = set(train_k)

    rng = random.Random(7)
    # Positives from UNSEEN vehicles only - the trained ones score too well
    # and would move the threshold somewhere it does not belong.
    pos = [(k, p) for k, p in truth.items()
           if k in where and k not in trained]
    rng.shuffle(pos)
    pos = pos[:args.positives]

    indexed_plates = {r["plate"] for r in js.records} | set(same_vehicle)
    negs = []
    while len(negs) < args.negatives // 2:
        p = (rng.choice(LETTERS) + rng.choice(LETTERS)
             + f"{rng.randrange(100):02d}"
             + rng.choice(LETTERS) + rng.choice(LETTERS)
             + f"{rng.randrange(10000):04d}")
        if p not in indexed_plates:
            negs.append(("random", p))
    src = [p for _, p in pos]
    while len(negs) < args.negatives:
        p = near_miss(rng.choice(src), rng)
        if p not in indexed_plates:
            negs.append(("near-miss", p))

    print(f"index      : {len(js.records)} vehicles")
    print(f"positives  : {len(pos)} (unseen vehicles that ARE indexed)")
    print(f"negatives  : {len(negs)} "
          f"({sum(1 for k,_ in negs if k=='random')} random, "
          f"{sum(1 for k,_ in negs if k=='near-miss')} near-miss)\n", flush=True)

    def top_score(plate):
        s = js.ctc_scores(plate)
        j = int(np.argmax(s))
        return float(s[j]), j

    pos_scores, neg_scores = [], []
    for n, (key, plate) in enumerate(pos, 1):
        s, j = top_score(plate)
        pos_scores.append((s, j in same_vehicle[plate]))
        if n % 25 == 0:
            print(f"  positives {n}/{len(pos)}", flush=True)
    for n, (kind, plate) in enumerate(negs, 1):
        s, _ = top_score(plate)
        neg_scores.append((s, kind))
        if n % 50 == 0:
            print(f"  negatives {n}/{len(negs)}", flush=True)

    ps = sorted(s for s, _ in pos_scores)
    ns = sorted(s for s, _ in neg_scores)
    print("\n" + "=" * 66)
    print("TOP-HIT SCORE DISTRIBUTION")
    print("=" * 66)
    print(f"{'':<12} {'p10':>9} {'median':>9} {'p90':>9}")
    print(f"{'present':<12} {ps[len(ps)//10]:>9.2f} {ps[len(ps)//2]:>9.2f} "
          f"{ps[len(ps)*9//10]:>9.2f}")
    print(f"{'absent':<12} {ns[len(ns)//10]:>9.2f} {ns[len(ns)//2]:>9.2f} "
          f"{ns[len(ns)*9//10]:>9.2f}")

    print("\n" + "=" * 66)
    print("OPERATING POINTS  (answer only above the threshold)")
    print("=" * 66)
    print(f"{'threshold':>10} {'answered':>10} {'correct':>9} "
          f"{'PRECISION':>11} {'RECALL':>9} {'false alarms':>13}")
    print("-" * 66)
    best = None
    lo, hi = min(ps[0], ns[0]), max(ps[-1], ns[-1])
    for th in np.linspace(lo, hi, 40):
        answered = [(s, ok) for s, ok in pos_scores if s >= th]
        fa = [1 for s, _ in neg_scores if s >= th]
        n_ans = len(answered) + len(fa)
        if n_ans < 5:
            continue
        correct = sum(ok for _, ok in answered)
        prec = correct / n_ans
        rec = correct / max(len(pos_scores), 1)
        if prec >= 0.90 and best is None:
            best = (th, prec, rec, len(fa))
        if int((th - lo) / max(hi - lo, 1e-9) * 40) % 5 == 0:
            print(f"{th:>10.2f} {n_ans:>10} {correct:>9} "
                  f"{prec*100:>10.1f}% {rec*100:>8.1f}% {len(fa):>13}")
    print("=" * 66)

    if best:
        th, prec, rec, fa = best
        print(f"\nAt score >= {th:.2f} the system answers with "
              f"{prec*100:.1f}% precision\nand finds {rec*100:.1f}% of the "
              f"vehicles that are actually there.\n"
              f"{fa} of {len(negs)} plates that were never seen produced a "
              f"false answer;\nthe rest correctly returned nothing.")
    else:
        print("\nNo threshold reaches 90% precision. The score does not "
              "separate\npresent from absent plates well enough to answer "
              "safely, and the\nsystem should return ranked CANDIDATES for a "
              "human to confirm\nrather than a single answer.")

    nm = sorted(s for s, k in neg_scores if k == "near-miss")
    rd = sorted(s for s, k in neg_scores if k == "random")
    if nm and rd:
        print(f"\nnear-miss negatives score {nm[len(nm)//2]:.2f} at the "
              f"median against\n{rd[len(rd)//2]:.2f} for random ones - the "
              f"gap between those is how much\nharder a typo is to reject "
              f"than an unrelated plate.")


if __name__ == "__main__":
    main()
