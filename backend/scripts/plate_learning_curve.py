"""Would more labels reach 80% strict, or has the curve flattened?

Strict exact-match sits at 49.0% (47.0-51.8 across three splits). Getting to
80% means +31 points, and every cheap lever has now been measured and spent:

    post-processing / grammar / LLM   ceiling 49.1%   (perfect-oracle test)
    multi-frame pixel fusion          34.5%           (twice, independently)
    detector input size               -7%             (crops already upscaled)
    64x256 ensemble over 32x128       +7 pts, CER halved
    best-view selection               +3.6 pts

What has never been measured here is the shape of the data curve. The project
has 417 labelled vehicles, of which 251 train the recogniser. That is very
little for a 36-class, 10-position sequence task, and one prior note in
plate_harvest_for_labelling records "23.1% to 76.9% ... the only limit left is
label count" — a claim that has never been checked against a curve.

The distinction matters more than anything else in the plan:

    still climbing steeply    labelling is the whole answer, and the number of
                              labels needed can be read off the curve
    flattening                labels are spent; the limit is resolution or
                              architecture, and more annotation buys nothing

This trains the same architecture on growing fractions of the training pool
and scores each on a fixed, untouched test set — so the only thing varying is
how much data the model saw. Subsets are nested and split by vehicle, never by
crop, because a plate with eleven crops on both sides of a split measures
memorisation (that protocol reports 89.3% against a true 49.0%).

Run:  python -m backend.scripts.plate_learning_curve --epochs 12
"""
from __future__ import annotations

import argparse
import json
import random
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from backend.scripts.plate_eval_clean import (                      # noqa: E402
    DS, _read_one, collate, lev,
)
from backend.scripts.train_plate_recognizer import (                # noqa: E402
    CHARS, CRNN, IMG_H, IMG_W, STOI,
)
from backend.scripts.indian_plate_grammar import decode_plate       # noqa: E402
from torch.utils.data import DataLoader                             # noqa: E402

REAL = ROOT / "data" / "plate_real"
FRACTIONS = (0.2, 0.4, 0.6, 0.8, 1.0)


def build_pool():
    truth = {}
    for l in (REAL / "verified_all.jsonl").open(encoding="utf-8"):
        v = json.loads(l)
        if v.get("text") and all(c in STOI for c in v["text"]):
            truth[(v["camera"], v["track"])] = v["text"]
    by_track = defaultdict(list)
    for l in (REAL / "labels.jsonl").open(encoding="utf-8"):
        r = json.loads(l)
        k = (r["camera"], r["track"])
        if k in truth:
            by_track[k].append(r["file"])
    return truth, by_track


def items_for(keys, by_track, truth):
    # DS in plate_eval_clean expects dicts with "file" and "text", and reads
    # images relative to its own REAL path.
    return [{"file": f, "text": truth[k]}
            for k in keys for f in by_track[k]]


def train(items, dev, epochs, seed):
    torch.manual_seed(seed)
    random.seed(seed)
    model = CRNN(len(CHARS) + 1).to(dev)
    opt = torch.optim.AdamW(model.parameters(), lr=3e-4)
    lossf = torch.nn.CTCLoss(blank=0, zero_infinity=True)
    dl = DataLoader(DS(items), batch_size=64, shuffle=True,
                    collate_fn=collate, num_workers=0,
                    drop_last=len(items) > 64)
    model.train()
    for _ in range(epochs):
        for imgs, targets, tlens in dl:
            imgs = imgs.to(dev)
            logits = model(imgs)
            lp = torch.log_softmax(logits, dim=2).permute(1, 0, 2)
            ilens = torch.full((imgs.size(0),), lp.size(0), dtype=torch.long)
            loss = lossf(lp, targets, ilens, tlens)
            opt.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 5.0)
            opt.step()
    model.eval()
    return model


def score(model, keys, by_track, truth, dev):
    exact = 0
    cer = 0.0
    n = 0
    for k in keys:
        gt = truth[k]
        # Best-view protocol: the read the deployed engine would take.
        best, best_conf = "", -1e9
        for f in by_track[k]:
            txt, conf = _read_one(model, f, dev, want_conf=True)
            if txt is not None and conf > best_conf:
                best_conf, best = conf, txt
        d = decode_plate(best)
        got = d["plate"] or best
        n += 1
        exact += (got == gt)
        cer += lev(got, gt) / max(1, len(gt))
    return (100 * exact / max(n, 1)), (100 * cer / max(n, 1)), n


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--epochs", type=int, default=12)
    ap.add_argument("--repeats", type=int, default=2)
    args = ap.parse_args()

    truth, by_track = build_pool()
    keys = sorted(by_track)
    dev = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"{len(keys)} labelled vehicles, {sum(len(v) for v in by_track.values())} crops\n")

    curves = defaultdict(list)
    for rep in range(args.repeats):
        kk = list(keys)
        random.Random(1000 + rep).shuffle(kk)
        n = len(kk)
        n_test = max(20, int(n * 0.20))
        test_k = kk[:n_test]
        pool = kk[n_test:]

        print(f"split {rep+1}: {len(pool)} train pool, {len(test_k)} test",
              flush=True)
        for frac in FRACTIONS:
            take = max(10, int(len(pool) * frac))
            # Nested subsets: the 40% run contains every vehicle the 20% run
            # had, so the curve reflects added data rather than a different
            # sample of it.
            sub = pool[:take]
            model = train(items_for(sub, by_track, truth), dev, args.epochs,
                          seed=1000 + rep)
            ex, cer, cnt = score(model, test_k, by_track, truth, dev)
            curves[frac].append((ex, cer))
            print(f"   {take:>4} vehicles ({frac:>4.0%})  "
                  f"exact {ex:>5.1f}%   CER {cer:>5.1f}%", flush=True)
            del model
            torch.cuda.empty_cache()

    print(f"\n{'train vehicles':<18}{'exact':>9}{'CER':>9}")
    print("-" * 38)
    pool_size = len(keys) - max(20, int(len(keys) * 0.20))
    xs, ys = [], []
    for frac in FRACTIONS:
        vals = curves[frac]
        ex = float(np.mean([v[0] for v in vals]))
        cer = float(np.mean([v[1] for v in vals]))
        n_v = max(10, int(pool_size * frac))
        xs.append(n_v)
        ys.append(ex)
        print(f"{n_v:<18}{ex:>8.1f}%{cer:>8.1f}%")

    # Slope over the last half of the curve says whether labelling still pays.
    if len(xs) >= 3:
        dx = xs[-1] - xs[len(xs) // 2]
        dy = ys[-1] - ys[len(ys) // 2]
        per_100 = 100 * dy / max(dx, 1)
        print(f"\nslope over the upper half: {per_100:+.1f} points per 100 "
              f"more labelled vehicles")
        if per_100 > 2.0:
            need = (80.0 - ys[-1]) / max(per_100, 1e-6) * 100
            print(f"extrapolating that slope, 80% strict would need roughly "
                  f"{need:,.0f} more labelled vehicles")
            print("(extrapolation assumes the slope holds, which it will not "
                  "forever — treat it as an order of magnitude)")
        else:
            print("the curve has flattened: more labels of this kind are not "
                  "the route to 80%, and the limit lies in resolution or "
                  "architecture instead")
    return 0


if __name__ == "__main__":
    sys.exit(main())
