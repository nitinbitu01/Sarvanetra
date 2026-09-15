"""backend/scripts/plate_ablate_batch.py — did the new labels help? Measured on
ONE fixed test set, which is the only way the question can be answered.

THE TRAP THIS AVOIDS
  Adding batch 7 moved the headline number from 55.6% to 47.9%, which looks
  like the new labels made things worse. It is not evidence of that. Growing
  the label pool also grew and reshuffled the held-out set - 54 plates became
  71 different plates - so the two figures are scores on two different exams.

  Worse, the exams differ in a direction that predicts exactly this result.
  Batch 7 plates are the ones a person could only read after being shown four
  frames; they are, by construction, the harder plates. Mixing them into the
  pool makes any held-out sample harder, so the measured accuracy falls even
  if the model improved.

  The same confound already produced one wrong conclusion in this project
  ("later batches are hurting"), which was withdrawn once the varying holdout
  was noticed. This script exists so it cannot happen a third time.

THE DESIGN
  Test and validation plates are drawn ONLY from the vehicles that existed
  before the new batch, and are frozen. Two models are then trained:

      without : the old vehicles, minus test and val
      with    : the same, plus every new-batch vehicle

  Both are scored on the identical frozen test set. The only difference
  between the arms is whether the new labels were in training, so the gap
  between them is the effect of those labels and nothing else.

  Several seeds are run because a single 55-plate test set has a wide error
  bar. The PAIRED difference per seed is the number to read - both arms share
  a test set within a seed, so most of the split noise cancels between them.

USAGE
  python -m backend.scripts.plate_ablate_batch --new verified_batch7.jsonl
  python -m backend.scripts.plate_ablate_batch --new verified_batch7.jsonl --repeats 4
"""
from __future__ import annotations

import argparse
import json
import random
import statistics
from collections import defaultdict
from pathlib import Path

import torch
import torch.nn as nn
from torch.utils.data import DataLoader

from backend.scripts.plate_eval_clean import DS, collate, score
from backend.scripts.train_plate_recognizer import BLANK, CHARS, CRNN, STOI

REAL = Path("data/plate_real")
CKPT_IN = Path("models/plate_recognizer/best.pt")


def train_one(train_items, val_items, dev, epochs, lr, batch):
    ck = torch.load(CKPT_IN, map_location=dev, weights_only=False)
    model = CRNN(len(CHARS) + 1).to(dev)
    model.load_state_dict(ck["model"])
    tl = DataLoader(DS(train_items), batch_size=batch, shuffle=True,
                    collate_fn=collate, num_workers=0,
                    drop_last=len(train_items) > batch)
    crit = nn.CTCLoss(blank=BLANK, zero_infinity=True)
    opt = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-4)
    best_v, best_state = -1.0, None
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
            v, _, _ = score(model, val_items, dev)
            if v > best_v:
                best_v = v
                best_state = {k: t.detach().clone()
                              for k, t in model.state_dict().items()}
    model.load_state_dict(best_state)
    return model


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--new", default="verified_batch7.jsonl",
                    help="The batch whose contribution is being measured.")
    ap.add_argument("--repeats", type=int, default=3)
    ap.add_argument("--epochs", type=int, default=90)
    ap.add_argument("--lr", type=float, default=2e-4)
    ap.add_argument("--batch", type=int, default=32)
    args = ap.parse_args()

    verified = [json.loads(l) for l in (REAL / "verified_all.jsonl").open(encoding="utf-8")]
    truth = {(v["camera"], v["track"]): v["text"] for v in verified
             if v["text"] and all(c in STOI for c in v["text"])}

    new_keys = set()
    p = REAL / args.new
    if p.is_file():
        for line in p.open(encoding="utf-8"):
            v = json.loads(line)
            new_keys.add((v["camera"], v["track"]))
    new_keys &= set(truth)
    old_keys = [k for k in truth if k not in new_keys]
    print(f"vehicles total : {len(truth)}")
    print(f"  pre-existing : {len(old_keys)}")
    print(f"  new batch    : {len(new_keys)}")

    by_track = defaultdict(list)
    for line in (REAL / "labels.jsonl").open(encoding="utf-8"):
        r = json.loads(line)
        k = (r["camera"], r["track"])
        if k in truth:
            by_track[k].append({"file": r["file"], "text": truth[k]})

    old_keys = sorted(k for k in old_keys if by_track[k])
    new_avail = sorted(k for k in new_keys if by_track[k])
    print(f"  with crops   : {len(old_keys)} old / {len(new_avail)} new\n",
          flush=True)

    dev = "cuda" if torch.cuda.is_available() else "cpu"
    deltas, wo_all, wi_all = [], [], []

    print(f"{'seed':<6} {'without':>9} {'with':>9} {'delta':>8}   (voted reads)")
    print("-" * 50)
    for i in range(args.repeats):
        rng = random.Random(2000 + i)
        keys = list(old_keys)
        rng.shuffle(keys)
        n_test = max(20, int(len(keys) * 0.22))
        n_val = max(15, int(len(keys) * 0.18))
        test_k = keys[:n_test]
        val_k = keys[n_test:n_test + n_val]
        train_old = keys[n_test + n_val:]

        def items(ks):
            return [{"file": by_track[k][0]["file"], "text": truth[k],
                     "files": [c["file"] for c in by_track[k]]} for k in ks]

        val_items, test_items = items(val_k), items(test_k)
        tr_wo = [it for k in train_old for it in by_track[k]]
        tr_wi = tr_wo + [it for k in new_avail for it in by_track[k]]

        m_wo = train_one(tr_wo, val_items, dev, args.epochs, args.lr, args.batch)
        a_wo, _, _ = score(m_wo, test_items, dev, multiframe=True)
        m_wi = train_one(tr_wi, val_items, dev, args.epochs, args.lr, args.batch)
        a_wi, _, _ = score(m_wi, test_items, dev, multiframe=True)

        deltas.append(a_wi - a_wo)
        wo_all.append(a_wo)
        wi_all.append(a_wi)
        print(f"{2000+i:<6} {a_wo*100:>8.1f}% {a_wi*100:>8.1f}% "
              f"{(a_wi-a_wo)*100:>+7.1f}", flush=True)

    md = statistics.mean(deltas)
    print("-" * 50)
    print(f"{'mean':<6} {statistics.mean(wo_all)*100:>8.1f}% "
          f"{statistics.mean(wi_all)*100:>8.1f}% {md*100:>+7.1f}")
    print("\n" + "=" * 62)
    print(f"EFFECT OF {args.new}")
    print("=" * 62)
    print(f"paired difference : {md*100:+.1f} points")
    if len(deltas) > 1:
        sd = statistics.stdev(deltas)
        print(f"spread across seeds: {min(deltas)*100:+.1f} to "
              f"{max(deltas)*100:+.1f}  (sd {sd*100:.1f})")
        if abs(md) < sd:
            print("\nThe difference is smaller than its own spread - this does")
            print("not establish that the new labels helped OR hurt. More")
            print("seeds would narrow it; more labels would settle it.")
        elif md > 0:
            print("\nThe new labels help, consistently across seeds.")
        else:
            print("\nThe new labels HURT on a fixed test set, which is a real")
            print("result rather than a harder-exam artefact. Check them for")
            print("misreadings before adding more of the same kind.")
    print("=" * 62)
    print("\nEvery test plate here predates the new batch, so both arms sat the")
    print("same exam. This figure is comparable; the headline accuracies from")
    print("differently-sized pools are not.")


if __name__ == "__main__":
    main()
