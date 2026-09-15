"""backend/scripts/journey_eval_recall.py — can the system find a vehicle when
you type its plate?

THE MEASUREMENT, AND WHY IT NEEDS NO NEW LABELLING
  417 vehicles have a plate a human read, and each one's track is known. So
  every one of them is a ready-made query with a known right answer: type the
  plate, and check whether the correct track comes back first. That is exactly
  the feature the brief asks for - "type in any license plate number, and the
  system draws that car's journey" - and it can be scored today without anyone
  labelling anything further.

  It is also the honest place to look for 90%. Reading a plate correctly is
  capped near 59% by the pixels. Ranking the right track above 5,593 others is
  a different and much easier problem, because the query supplies the answer
  and the model only has to prefer the right track.

THE COMPARISON THAT MATTERS
  Two scorers are run on identical queries:

    exact string    the decoded reading must equal the query. This is what
                    most such systems do, and it inherits the recogniser's
                    59% directly.
    CTC likelihood  the query is scored against the stored log-probabilities.
                    A track read as GJ01A81234 still ranks first for
                    GJ01AB1234 when B was a close second at that timestep.

  The gap between them is what storing logits instead of strings buys, and it
  is the single design decision this feature rests on.

TRAINED AND UNSEEN ARE REPORTED SEPARATELY
  The recogniser was fitted on some of these vehicles. Their logits are
  sharper for their own plate and their recall would flatter the system. The
  headline figure is the unseen split; the trained split is shown beside it so
  the size of that effect is visible rather than hidden.

USAGE
  python -m backend.scripts.journey_eval_recall
"""
from __future__ import annotations

import argparse
import json
import random
from collections import defaultdict
from pathlib import Path

import numpy as np

from backend.scripts.plate_final_model import load_data, split
from backend.services.journey_search import JourneySearch, weighted_edit

REAL = Path("data/plate_real")
INDEX = Path("output/journey_index")


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--index", default=str(INDEX))
    ap.add_argument("--limit", type=int, default=0)
    args = ap.parse_args()

    js = JourneySearch(Path(args.index))
    print(f"indexed vehicles : {len(js.records)}")

    # Which (camera, track) each indexed record is.
    where = {}
    for i, r in enumerate(js.records):
        where.setdefault((r["camera"], r["track"]), i)

    truth = {}
    for line in (REAL / "verified_all.jsonl").open(encoding="utf-8"):
        v = json.loads(line)
        if v.get("text"):
            truth[(v["camera"], v["track"])] = v["text"]

    # The recogniser's own split, so trained and unseen can be separated.
    _, by_track = load_data()
    test_k, val_k, train_k = split(by_track, 1000)
    trained = set(train_k)

    # A plate can be indexed under more than one track - the tracker splits a
    # vehicle, or the vehicle genuinely passes twice. The operator typed a
    # PLATE and wants that vehicle, so any track of it is a correct answer.
    # Scoring only the one arbitrarily-chosen "gold" track marks a system that
    # returned the right car as having failed, which is a bug in the metric
    # rather than in the system.
    same_vehicle = defaultdict(set)
    for k, p in truth.items():
        if k in where:
            same_vehicle[p].add(where[k])
    queries = [(k, p) for k, p in truth.items() if k in where]
    missing = len(truth) - len(queries)
    dupes = sum(1 for v in same_vehicle.values() if len(v) > 1)
    if args.limit:
        random.Random(3).shuffle(queries)
        queries = queries[:args.limit]
    print(f"labelled vehicles: {len(truth)}")
    print(f"  present in index: {len(queries)}")
    print(f"  absent (no plate found by detector): {missing}")
    print(f"  plates indexed under >1 track: {dupes}\n", flush=True)

    stats = {"unseen": {"n": 0, "r1": 0, "r5": 0, "mrr": 0.0,
                        "s1": 0, "s5": 0},
             "trained": {"n": 0, "r1": 0, "r5": 0, "mrr": 0.0,
                         "s1": 0, "s5": 0}}
    fails = []

    def first_hit(order, accept) -> int:
        """Rank of the first returned track that IS the queried vehicle."""
        for pos, idx in enumerate(order, 1):
            if idx in accept:
                return pos
        return len(order)

    for n, (key, plate) in enumerate(queries, 1):
        accept = same_vehicle[plate]           # every track of this vehicle
        gold = where[key]
        bucket = "trained" if key in trained else "unseen"
        b = stats[bucket]
        b["n"] += 1

        # --- CTC likelihood ranking
        ctc = js.ctc_scores(plate)
        order = np.argsort(-ctc)
        rank = first_hit(order, accept)
        b["r1"] += rank == 1
        b["r5"] += rank <= 5
        b["mrr"] += 1.0 / rank

        # --- string baseline, ranked by weighted edit distance so it is given
        # the fairest possible reading of "string matching"
        eds = np.array([weighted_edit(plate, r["plate"]) if r["plate"] else 99.0
                        for r in js.records])
        srank = first_hit(np.argsort(eds), accept)
        b["s1"] += srank == 1
        b["s5"] += srank <= 5

        if rank > 1 and len(fails) < 12:
            fails.append((plate, js.records[gold]["plate"], rank,
                          js.records[order[0]]["plate"]))
        if n % 25 == 0:
            print(f"  {n}/{len(queries)}", flush=True)

    print("\n" + "=" * 70)
    print("CAN THE SYSTEM FIND A VEHICLE FROM ITS PLATE?")
    print(f"index: {len(js.records)} vehicles across "
          f"{len({r['camera'] for r in js.records})} cameras")
    print("=" * 70)
    print(f"{'':<10} {'queries':>8} {'CTC R@1':>9} {'CTC R@5':>9} "
          f"{'MRR':>7} {'string R@1':>11} {'string R@5':>11}")
    print("-" * 70)
    for name in ("unseen", "trained"):
        b = stats[name]
        if not b["n"]:
            continue
        n_ = b["n"]
        print(f"{name:<10} {n_:>8} {b['r1']/n_*100:>8.1f}% "
              f"{b['r5']/n_*100:>8.1f}% {b['mrr']/n_:>7.3f} "
              f"{b['s1']/n_*100:>10.1f}% {b['s5']/n_*100:>10.1f}%")
    tot = {k: stats["unseen"][k] + stats["trained"][k]
           for k in ("n", "r1", "r5", "s1", "s5")}
    tot["mrr"] = stats["unseen"]["mrr"] + stats["trained"]["mrr"]
    if tot["n"]:
        print("-" * 70)
        print(f"{'ALL':<10} {tot['n']:>8} {tot['r1']/tot['n']*100:>8.1f}% "
              f"{tot['r5']/tot['n']*100:>8.1f}% {tot['mrr']/tot['n']:>7.3f} "
              f"{tot['s1']/tot['n']*100:>10.1f}% "
              f"{tot['s5']/tot['n']*100:>10.1f}%")
    print("=" * 70)

    u = stats["unseen"]
    if u["n"]:
        gain = (u["r1"] - u["s1"]) / u["n"] * 100
        print(f"\nOn vehicles the recogniser never saw, CTC scoring ranks the "
              f"right\ntrack first {u['r1']/u['n']*100:.1f}% of the time "
              f"against {u['s1']/u['n']*100:.1f}% for string matching "
              f"({gain:+.1f} points).")
        print("That gap is what storing the model's uncertainty buys over "
              "storing\nits final answer.")

    if fails:
        print(f"\nqueries where the right track was not first "
              f"({len(fails)} shown):")
        print(f"  {'queried':<12} {'index read':<12} {'rank':>5}  "
              f"{'what ranked first'}")
        for q, got, rank, first in fails:
            print(f"  {q:<12} {got:<12} {rank:>5}  {first}")


if __name__ == "__main__":
    main()
