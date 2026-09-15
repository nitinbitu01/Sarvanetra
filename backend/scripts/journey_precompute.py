"""backend/scripts/journey_precompute.py — run every known plate through the
journey search and cache the answers.

WHY PRECOMPUTE
  The search itself is fast - one batched CTC pass over the index, about 50 ms -
  but it needs PyTorch and the stored log-probabilities, neither of which a
  self-contained demo page can carry. Precomputing the answers lets the page be
  a single file that works with no server and no network, which is what a
  hackathon venue actually provides.

  The answers are the live system's, unaltered. Nothing is filtered, reordered
  or improved on the way out; a query that returns nothing is cached as
  returning nothing.

WHAT IS QUERIED
  Every plate a human read, and every distinct well-formed reading in the
  index. The first set has ground truth attached, so the page can show which
  answers are independently confirmed. The second set is what an officer would
  actually type for a vehicle nobody labelled, and including it stops the demo
  being a tour of the answers that happen to be known.

USAGE
  python -m backend.scripts.journey_precompute
"""
from __future__ import annotations

import argparse
import json
from collections import defaultdict
from pathlib import Path

from backend.scripts.plate_beam_decode import complete
from backend.services.journey_search import JourneySearch

REAL = Path("data/plate_real")
INDEX = Path("output/journey_index")
OUT = Path("output/journey_answers.json")


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--index", default=str(INDEX))
    ap.add_argument("--out", default=str(OUT))
    args = ap.parse_args()

    js = JourneySearch(Path(args.index))
    print(f"index: {len(js.records)} sightings")

    truth = {}
    for line in (REAL / "verified_all.jsonl").open(encoding="utf-8"):
        v = json.loads(line)
        if v.get("text"):
            truth[(v["camera"], v["track"])] = v["text"]
    # The camera(s) a plate was actually read on by a human, so the page can
    # mark which of its own answers are independently confirmed.
    gt_cams = defaultdict(set)
    for (cam, trk), p in truth.items():
        gt_cams[p].add(cam)

    queries = set(gt_cams)
    for r in js.records:
        p = r.get("plate") or ""
        if len(p) >= 9 and complete(p) and not (p[:2].isalpha()
                                                and p[2:4] == "00"):
            queries.add(p)
    queries = sorted(queries)
    print(f"queries: {len(queries)} "
          f"({len(gt_cams)} with human ground truth)\n", flush=True)

    answers = {}
    multi = 0
    for n, q in enumerate(queries, 1):
        j = js.journey(q)
        for s in j["sightings"]:
            s["ground_truth"] = s["camera"] in gt_cams.get(q, ())
        j["gt_cameras"] = sorted(gt_cams.get(q, ()))
        answers[q] = j
        if j["cameras"] > 1:
            multi += 1
        if n % 100 == 0:
            print(f"  {n}/{len(queries)}  ({multi} multi-camera)", flush=True)

    Path(args.out).write_text(json.dumps(answers), encoding="utf-8")

    found = [a for a in answers.values() if a["sightings"]]
    print("\n" + "=" * 58)
    print(f"queries answered      : {len(answers)}")
    print(f"  returned a sighting : {len(found)}")
    print(f"  returned nothing    : {len(answers) - len(found)}")
    print(f"  multi-camera        : {multi}")
    conf = sum(a["confirmed"] for a in answers.values())
    print(f"  confirmed sightings : {conf}")
    print(f"saved -> {args.out}")
    print("=" * 58)

    if multi:
        print("\nmulti-camera results:")
        for q, a in answers.items():
            if a["cameras"] > 1:
                cams = " -> ".join(f"{s['camera']}@{(s['timestamp'] or '')[11:16]}"
                                   for s in a["sightings"])
                gt = "".join("*" if s["ground_truth"] else "."
                             for s in a["sightings"])
                print(f"  {q:<12} {cams}   gt[{gt}]")


if __name__ == "__main__":
    main()
