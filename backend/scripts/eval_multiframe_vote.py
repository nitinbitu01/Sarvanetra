"""backend/scripts/eval_multiframe_vote.py — read every frame of a vehicle, then vote.

WHY VOTING WORKS NOW AND FAILED BEFORE
  An earlier attempt voted across EasyOCR reads and produced ~15% correct
  labels. It failed because EasyOCR's errors are SYSTEMATIC: it misread the
  same glyph the same way on every frame, so twenty-five frames agreed on one
  wrong answer. 'GJ1CO9007' carried perfect agreement and was wrong.

  The fine-tuned CRNN fails differently. Its held-out mistakes land on
  DIFFERENT positions each time:

      GJ11BH9381 -> GJ11BH9384    last digit
      GJ05CX3227 -> GJ05CK3227    position 6
      GJ04EE9493 -> GJ04EE9483    position 8
      GJ18BM8262 -> GJ18BM8267    last digit

  Errors scattered across positions are the condition under which voting
  actually helps - a character wrong in one frame is usually right in the
  next, so the majority at each position recovers the plate even when no
  single frame reads it correctly end to end.

WHAT THIS COSTS
  Nothing. Every labelled vehicle already has 4-7 crops on disk; the previous
  evaluation read exactly one and discarded the rest. No new labels, no new
  training, no new data collection.

HONEST COMPARISON
  Scored on the SAME held-out vehicles, with the SAME model and the SAME
  train/test split seed as the single-frame run, so the only variable is
  whether frames are combined. Anything else would not be a measurement.

USAGE
  python -m backend.scripts.eval_multiframe_vote
"""
from __future__ import annotations

import argparse
import json
import random
from collections import Counter, defaultdict
from pathlib import Path

import cv2
import numpy as np
import torch

from backend.scripts.indian_plate_grammar import decode_plate
from backend.scripts.train_plate_recognizer import (CHARS, CRNN, IMG_H, IMG_W,
                                                    STOI, ctc_decode)

REAL = Path("data/plate_real")
VERIFIED = REAL / "verified_all.jsonl"
CKPT = Path("models/plate_recognizer/finetuned.pt")


def lev(a: str, b: str) -> int:
    dp = list(range(len(b) + 1))
    for i, ca in enumerate(a, 1):
        prev, dp[0] = dp[0], i
        for j, cb in enumerate(b, 1):
            cur = dp[j]
            dp[j] = min(dp[j] + 1, dp[j - 1] + 1, prev + (ca != cb))
            prev = cur
    return dp[len(b)]


def vote(reads: list[tuple[str, float]]) -> tuple[str, float]:
    """Per-position weighted vote among reads sharing the winning length.

    Reads of different lengths cannot be compared position-by-position, so the
    length carrying the most confidence wins first, then each position is
    decided independently within that group.
    """
    if not reads:
        return "", 0.0
    by_len: dict[int, list] = defaultdict(list)
    for t, w in reads:
        by_len[len(t)].append((t, w))
    best_len = max(by_len, key=lambda L: sum(w for _, w in by_len[L]))
    grp = by_len[best_len]
    out, agree = [], 0.0
    for i in range(best_len):
        tally: Counter = Counter()
        for t, w in grp:
            tally[t[i]] += w
        ch, top = tally.most_common(1)[0]
        out.append(ch)
        agree += top / max(sum(tally.values()), 1e-9)
    return "".join(out), agree / max(best_len, 1)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--holdout", type=int, default=40)
    ap.add_argument("--seed", type=int, default=1337)
    args = ap.parse_args()

    verified = [json.loads(l) for l in VERIFIED.open(encoding="utf-8")]
    verified = [v for v in verified
                if v["text"] and all(c in STOI for c in v["text"])]
    truth = {(v["camera"], v["track"]): v["text"] for v in verified}

    rows = [json.loads(l) for l in (REAL / "labels.jsonl").open(encoding="utf-8")]
    by_track: dict[tuple, list[str]] = defaultdict(list)
    for r in rows:
        k = (r["camera"], r["track"])
        if k in truth:
            by_track[k].append(r["file"])

    # Reproduce the exact split used by the single-frame run.
    keys = sorted(by_track)
    random.seed(args.seed)
    random.shuffle(keys)
    test_keys = keys[:args.holdout]

    dev = "cuda" if torch.cuda.is_available() else "cpu"
    ck = torch.load(CKPT, map_location=dev, weights_only=False)
    model = CRNN(len(CHARS) + 1).to(dev)
    model.load_state_dict(ck["model"])
    model.eval()
    print(f"model    : {CKPT}")
    print(f"held out : {len(test_keys)} vehicles "
          f"(same split, seed {args.seed})\n", flush=True)

    def read_one(f: str) -> tuple[str, float] | None:
        img = cv2.imread(str(REAL / "images" / f), cv2.IMREAD_GRAYSCALE)
        if img is None:
            return None
        img = cv2.resize(img, (IMG_W, IMG_H), interpolation=cv2.INTER_AREA)
        x = torch.from_numpy(img).float().div(127.5).sub(1.0)[None, None].to(dev)
        with torch.no_grad():
            logits = model(x)
            # Mean max-softmax across timesteps as a crude per-read confidence,
            # so a hesitant frame carries less weight than a decisive one.
            conf = float(logits.softmax(2).max(dim=2).values.mean())
            return ctc_decode(logits)[0], conf

    single_ok = vote_ok = 0
    s_num = v_num = den = 0
    changed = []

    for k in test_keys:
        gt = truth[k]
        files = by_track[k]
        reads = []
        for f in files:
            r = read_one(f)
            if r and r[0]:
                d = decode_plate(r[0])
                reads.append((d["plate"] or r[0], r[1]))
        if not reads:
            continue
        # Single-frame baseline: the first crop, as the previous eval used.
        s_pred = reads[0][0]
        v_raw, agree = vote(reads)
        d = decode_plate(v_raw)
        v_pred = d["plate"] or v_raw

        single_ok += (s_pred == gt)
        vote_ok += (v_pred == gt)
        s_num += lev(s_pred, gt)
        v_num += lev(v_pred, gt)
        den += len(gt)
        if s_pred != v_pred:
            changed.append((gt, s_pred, v_pred, len(reads), agree))

    n = len(test_keys)
    print("--- vehicles where voting changed the answer ---")
    print(f"{'truth':<13} {'single':<13} {'voted':<13} {'frames':>6} {'agree':>6}")
    print("-" * 58)
    for gt, s, v, nf, ag in changed:
        mark = ("  FIXED" if v == gt and s != gt else
                "  BROKE" if s == gt and v != gt else "")
        print(f"{gt:<13} {s:<13} {v:<13} {nf:>6} {ag:>6.2f}{mark}")

    print("\n" + "=" * 58)
    print(f"single-frame exact : {single_ok}/{n} = {single_ok/max(n,1)*100:.1f}%"
          f"   CER {s_num/max(den,1)*100:.1f}%")
    print(f"multi-frame  exact : {vote_ok}/{n} = {vote_ok/max(n,1)*100:.1f}%"
          f"   CER {v_num/max(den,1)*100:.1f}%")
    print("=" * 58)
    fixed = sum(1 for g, s, v, _, _ in changed if v == g and s != g)
    broke = sum(1 for g, s, v, _, _ in changed if s == g and v != g)
    print(f"\nfixed by voting  : {fixed}")
    print(f"broken by voting : {broke}")
    print(f"crops per vehicle: {sum(len(by_track[k]) for k in test_keys)/max(n,1):.1f}")


if __name__ == "__main__":
    main()
