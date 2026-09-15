"""backend/scripts/journey_build_verify_set.py — the candidate links a human
needs to judge before any journey number can be quoted.

WHY THIS IS THE BOTTLENECK
  Journey linking has been built and cannot be measured. Four cross-camera
  pairs were judged by eye - two right, two wrong - and four points support no
  threshold and no accuracy claim. A colour comparison was tried and inverted
  the ordering; an ImageNet embedding got the ordering right with a 0.005
  margin, which is a line fitted through four dots.

  Everything downstream needs the same thing: cross-camera pairs a human has
  ruled on. That is the only input this project does not already have.

WHAT IS OFFERED FOR JUDGEMENT
  Pairs the system would actually propose, across a RANGE of scores, not just
  its best ones. Only showing high-confidence pairs measures the easy end and
  reports precision that will not hold when the threshold moves. The spread is
  what lets precision be plotted against threshold afterwards.

  Control pairs are mixed in: two sightings the system scores as clearly
  unrelated. They cost the labeller a few seconds each and catch the failure
  mode where someone clicks through agreeing with everything - if the controls
  come back marked SAME, the whole batch is unreliable and should be redone.

WHAT IS DELIBERATELY NOT SHOWN
  The plate readings. A labeller told the system read GJ03CR1031 on both sides
  will see two matching strings and agree. The question being asked is whether
  the VEHICLES match, and the answer has to come from the vehicles.

USAGE
  python -m backend.scripts.journey_build_verify_set --pairs 120
"""
from __future__ import annotations

import argparse
import base64
import json
import random
from pathlib import Path

import cv2
import numpy as np

from backend.scripts.plate_beam_decode import complete
from backend.services.journey_search import JourneySearch

INDEX = Path("output/journey_index")
OUT = Path("data/journey_verify")


def linkable(r) -> bool:
    p = r.get("plate") or ""
    return (len(p) >= 9 and complete(p)
            and not (p[:2].isalpha() and p[2:4] == "00")
            and r.get("n_frames", 0) >= 2)


def thumb(path, h=150):
    img = cv2.imread(str(path))
    if img is None:
        return ""
    s = h / img.shape[0]
    img = cv2.resize(img, (max(1, int(img.shape[1] * s)), h),
                     interpolation=cv2.INTER_CUBIC)
    ok, buf = cv2.imencode(".jpg", img, [cv2.IMWRITE_JPEG_QUALITY, 86])
    return base64.b64encode(buf).decode() if ok else ""


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--pairs", type=int, default=120)
    ap.add_argument("--controls", type=int, default=10)
    ap.add_argument("--min-score", type=float, default=-22.0,
                    help="Loose on purpose. A batch drawn only from the "
                         "system's best links measures the easy end and "
                         "produces a precision figure that collapses as soon "
                         "as the threshold is lowered.")
    args = ap.parse_args()

    js = JourneySearch(INDEX)
    recs = js.records
    ok = [i for i, r in enumerate(recs) if linkable(r)]
    print(f"indexed {len(recs)}, linkable {len(ok)}")

    seen = set()
    cands = []
    for n, i in enumerate(ok, 1):
        scores = js.ctc_scores(recs[i]["plate"])
        for j in np.argsort(-scores)[:6]:
            j = int(j)
            if j == i or recs[j]["camera"] == recs[i]["camera"]:
                continue
            if scores[j] < args.min_score:
                break
            key = tuple(sorted((i, j)))
            if key in seen:
                continue
            seen.add(key)
            cands.append({"a": i, "b": j, "score": float(scores[j])})
        if n % 200 == 0:
            print(f"  {n}/{len(ok)} scanned, {len(cands)} candidates",
                  flush=True)

    cands.sort(key=lambda c: -c["score"])
    print(f"cross-camera candidates : {len(cands)}")

    # Spread the sample across the score range rather than taking the top N,
    # so precision can be plotted against threshold afterwards.
    if len(cands) > args.pairs:
        idx = np.linspace(0, len(cands) - 1, args.pairs).astype(int)
        chosen = [cands[k] for k in dict.fromkeys(idx.tolist())]
    else:
        chosen = cands

    rng = random.Random(9)
    for _ in range(args.controls):
        a, b = rng.sample(ok, 2)
        if recs[a]["camera"] == recs[b]["camera"]:
            continue
        chosen.append({"a": a, "b": b, "score": None, "control": True})
    rng.shuffle(chosen)

    OUT.mkdir(parents=True, exist_ok=True)
    items = []
    for k, c in enumerate(chosen):
        ra, rb = recs[c["a"]], recs[c["b"]]
        ta = next((thumb(p) for p in ra.get("crops", [])
                   if Path(p).is_file()), "")
        tb = next((thumb(p) for p in rb.get("crops", [])
                   if Path(p).is_file()), "")
        if not ta or not tb:
            continue
        items.append({
            "id": k, "a_img": ta, "b_img": tb,
            "a_cam": ra["camera"], "b_cam": rb["camera"],
            "a_name": ra.get("camera_name"), "b_name": rb.get("camera_name"),
            "a_ts": (ra.get("timestamp") or "")[11:19],
            "b_ts": (rb.get("timestamp") or "")[11:19],
            "score": c.get("score"), "control": c.get("control", False),
            "a_key": f'{ra["camera"]}|{ra["track"]}',
            "b_key": f'{rb["camera"]}|{rb["track"]}',
        })

    (OUT / "pairs.json").write_text(json.dumps(items), encoding="utf-8")
    real = [i for i in items if not i["control"]]
    print(f"\npairs for judgement : {len(items)} "
          f"({len(real)} candidates + "
          f"{len(items)-len(real)} controls)")
    if real:
        ss = sorted(i["score"] for i in real)
        print(f"score range         : {ss[0]:.1f} to {ss[-1]:.1f}")
    print(f"saved -> {OUT}/pairs.json")


if __name__ == "__main__":
    main()
