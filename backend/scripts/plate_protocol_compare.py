"""backend/scripts/plate_protocol_compare.py — the same model, scored the five
ways a plate recogniser is commonly reported.

WHY THIS EXISTS
  A comparable accuracy figure needs a protocol attached to it. The strict
  protocol used elsewhere in this project reports 46.0%; the same weights on
  the same footage report far higher under looser but entirely common choices.
  None of the looser ones are dishonest by intent - they are the defaults of
  most training scripts - but they measure different things, and two numbers
  produced under different protocols cannot be subtracted.

  So this runs one model and reports all of them side by side. If someone else
  reports 81% on this footage, this table says which protocol would produce
  that figure from this model, and therefore what to ask them.

THE FIVE

  strict          Split BY VEHICLE, checkpoint chosen on a separate val set,
                  test read once. Exact match on all characters. This is the
                  number that predicts behaviour on tomorrow's traffic.

  test-selected   Same split, but the checkpoint is the best-scoring epoch ON
                  TEST - which is what a training loop does by default when
                  its "validation" set is the only held-out data. Reports the
                  maximum of ~18 noisy evaluations as if it were one.

  crop split      Frames split randomly instead of by vehicle. A plate with
                  eleven crops lands on both sides, so the model is scored on
                  plates it memorised. This is the single largest inflator and
                  the hardest to notice.

  char accuracy   Fraction of CHARACTERS right rather than whole plates. A
                  perfectly reasonable metric that answers a different
                  question - and on a ten-character string it is always far
                  higher than exact match.

  within-1        Plates wrong by at most one character. Sometimes quoted as
                  "accuracy" because a human operator can often resolve the
                  last character from context.

USAGE
  python -m backend.scripts.plate_protocol_compare --repeats 3
"""
from __future__ import annotations

import argparse
import json
import random
import statistics
from collections import defaultdict
from pathlib import Path

import cv2
import torch
import torch.nn as nn
from torch.utils.data import DataLoader

from backend.scripts.indian_plate_grammar import decode_plate
from backend.scripts.plate_eval_clean import DS, _read_one, _vote, collate, lev
from backend.scripts.train_plate_recognizer import (BLANK, CHARS, CRNN, STOI)

REAL = Path("data/plate_real")
CKPT_IN = Path("models/plate_recognizer/best.pt")


def evaluate(model, items, dev, multiframe=True):
    """Returns exact, char-accuracy, within-1, on the given items."""
    ex = w1 = 0
    num = den = 0
    with torch.no_grad():
        for it in items:
            if multiframe and it.get("files"):
                reads = [r for r in (_read_one(model, f, dev)
                                     for f in it["files"]) if r]
                pred = _vote(reads)
            else:
                pred = _read_one(model, it["file"], dev)
            if not pred:
                continue
            d = decode_plate(pred)
            fin = d["plate"] or pred
            gt = it["text"]
            dist = lev(fin, gt)
            ex += fin == gt
            w1 += dist <= 1
            num += dist
            den += len(gt)
    n = max(len(items), 1)
    return ex / n, 1.0 - num / max(den, 1), w1 / n


def train(train_items, sel_items, dev, epochs, lr, batch, seed=0):
    """Trains and returns the checkpoint that scored best on sel_items.

    sel_items is the ONLY thing that differs between the strict and the
    test-selected protocols: pass a separate val set for the first, pass the
    test set itself for the second.
    """
    torch.manual_seed(seed)
    ck = torch.load(CKPT_IN, map_location=dev, weights_only=False)
    model = CRNN(len(CHARS) + 1).to(dev)
    model.load_state_dict(ck["model"])
    tl = DataLoader(DS(train_items), batch_size=batch, shuffle=True,
                    collate_fn=collate, num_workers=0,
                    drop_last=len(train_items) > batch)
    crit = nn.CTCLoss(blank=BLANK, zero_infinity=True)
    opt = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-4)
    best, best_state = -1.0, None
    for ep in range(1, epochs + 1):
        model.train()
        for x, y, ylen in tl:
            x, y, ylen = x.to(dev), y.to(dev), ylen.to(dev)
            opt.zero_grad(set_to_none=True)
            logits = model(x)
            lp = logits.log_softmax(2).permute(1, 0, 2)
            xlen = torch.full((x.size(0),), logits.size(1), dtype=torch.long,
                              device=dev)
            crit(lp, y, xlen, ylen).backward()
            nn.utils.clip_grad_norm_(model.parameters(), 5.0)
            opt.step()
        if ep % 5 == 0 or ep == epochs:
            model.eval()
            a, _, _ = evaluate(model, sel_items, dev)
            if a > best:
                best = a
                best_state = {k: t.detach().clone()
                              for k, t in model.state_dict().items()}
    model.load_state_dict(best_state)
    model.eval()
    return model


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--repeats", type=int, default=3)
    ap.add_argument("--epochs", type=int, default=90)
    ap.add_argument("--lr", type=float, default=2e-4)
    ap.add_argument("--batch", type=int, default=32)
    args = ap.parse_args()

    verified = [json.loads(l) for l in (REAL / "verified_all.jsonl").open(encoding="utf-8")]
    truth = {(v["camera"], v["track"]): v["text"] for v in verified
             if v["text"] and all(c in STOI for c in v["text"])}
    by_track = defaultdict(list)
    for line in (REAL / "labels.jsonl").open(encoding="utf-8"):
        r = json.loads(line)
        k = (r["camera"], r["track"])
        if k in truth:
            by_track[k].append({"file": r["file"], "text": truth[k]})

    dev = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"vehicles {len(truth)}   crops "
          f"{sum(len(v) for v in by_track.values())}\n", flush=True)

    res = defaultdict(list)
    for i in range(args.repeats):
        keys = sorted(by_track)
        rng = random.Random(1000 + i)
        rng.shuffle(keys)
        n = len(keys)
        n_test = max(20, int(n * 0.20))
        n_val = max(20, int(n * 0.20))
        test_k, val_k = keys[:n_test], keys[n_test:n_test + n_val]
        train_k = keys[n_test + n_val:]
        mk = lambda ks: [{"file": by_track[k][0]["file"], "text": truth[k],
                          "files": [c["file"] for c in by_track[k]]} for k in ks]
        val_items, test_items = mk(val_k), mk(test_k)
        tr_items = [it for k in train_k for it in by_track[k]]

        # 1-3 and 5: split by vehicle, checkpoint on val.
        m = train(tr_items, val_items, dev, args.epochs, args.lr, args.batch,
                  seed=i)
        ex, ca, w1 = evaluate(m, test_items, dev)
        res["strict (exact)"].append(ex)
        res["char accuracy"].append(ca)
        res["within-1 char"].append(w1)

        # 2: identical run, but the checkpoint is chosen on TEST.
        m2 = train(tr_items, test_items, dev, args.epochs, args.lr, args.batch,
                   seed=i)
        ex2, _, _ = evaluate(m2, test_items, dev)
        res["test-selected"].append(ex2)

        # 3: random CROP split - a plate's frames land on both sides.
        all_crops = [dict(it, camera=k[0], track=k[1])
                     for k in by_track for it in by_track[k]]
        rng2 = random.Random(5000 + i)
        rng2.shuffle(all_crops)
        cut = int(len(all_crops) * 0.20)
        c_test = [{"file": c["file"], "text": c["text"]} for c in all_crops[:cut]]
        c_train = all_crops[cut:]
        m3 = train(c_train, c_test, dev, args.epochs, args.lr, args.batch,
                   seed=i)
        ex3, _, _ = evaluate(m3, c_test, dev, multiframe=False)
        res["crop split"].append(ex3)

        print(f"split {i+1}  strict {ex*100:.1f}%  test-sel {ex2*100:.1f}%  "
              f"crop-split {ex3*100:.1f}%  char {ca*100:.1f}%  "
              f"within-1 {w1*100:.1f}%", flush=True)
        del m, m2, m3
        torch.cuda.empty_cache()

    print("\n" + "=" * 64)
    print("ONE MODEL, ONE FOOTAGE SET, FIVE WAYS OF REPORTING IT")
    print("=" * 64)
    order = ["strict (exact)", "test-selected", "crop split",
             "within-1 char", "char accuracy"]
    for k in order:
        v = res[k]
        print(f"{k:<18} {statistics.mean(v)*100:>6.1f}%   "
              f"({min(v)*100:.1f}-{max(v)*100:.1f})")
    print("=" * 64)
    print("\nAll five describe the same weights on the same plates. Only the")
    print("first predicts behaviour on footage the model has not seen, which")
    print("is the only thing an operator cares about. When comparing against")
    print("someone else's figure, establish which of these rows it is before")
    print("concluding anything - the gap between the top and bottom row here")
    print("is larger than any modelling change measured in this project.")


if __name__ == "__main__":
    main()
