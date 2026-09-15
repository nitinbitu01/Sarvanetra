"""backend/scripts/plate_ensemble.py — several small recognisers instead of one
large one.

THE REASONING
  A bigger model was the obvious way to buy accuracy and it is the wrong trade
  here: TrOCR-small is roughly ten times the parameters and several times the
  latency for no measured gain, because 417 labelled plates cannot support a
  62M-parameter model. Capacity is not what is missing.

  What IS available is variance reduction. Two CRNNs trained from the same
  pretrained weights on the same data with different seeds and different batch
  orders converge to different minima, and they get different plates wrong.
  Averaging their per-timestep distributions before CTC decoding cancels part
  of that disagreement. It is the oldest trick in the book precisely because it
  works when data is scarce.

  Crucially it also respects the constraint the bigger model violated. Three
  CRNNs cost 3 x 0.76 ms batched - about 2.3 ms per plate, still an order of
  magnitude under the 19.6 ms TrOCR needs, and they batch and parallelise the
  way one model does.

AVERAGE THE PROBABILITIES, NOT THE STRINGS
  Voting on decoded strings throws away everything the models were unsure
  about, and CTC strings of different lengths do not align for voting anyway.
  Averaging the softmax at each timestep keeps the uncertainty, so a timestep
  where one model is torn between 8 and 6 and the other is confident about 8
  resolves to 8 - which string voting could not do.

MEASURED AGAINST A SINGLE MODEL ON THE SAME SPLITS
  Each ensemble member and its single-model baseline are trained on identical
  data and evaluated on identical plates, so the difference is the ensembling
  and nothing else. An ensemble that does not clear the split noise is not
  worth three times the compute, and this reports whether it does.

USAGE
  python -m backend.scripts.plate_ensemble --members 3 --repeats 3
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
from backend.scripts.plate_eval_clean import DS, _vote, collate, lev, score
from backend.scripts.train_plate_recognizer import (BLANK, CHARS, CRNN, IMG_H,
                                                    IMG_W, ITOS, STOI)

REAL = Path("data/plate_real")
CKPT_IN = Path("models/plate_recognizer/best.pt")


@torch.no_grad()
def ens_decode(models, g, dev):
    """Mean softmax across members, then greedy CTC collapse.

    no_grad is not an optimisation here, it is required: without it every crop
    builds and retains an autograd graph across three models, and the run dies
    of memory exhaustion partway through the second split.
    """
    g = cv2.resize(g, (IMG_W, IMG_H), interpolation=cv2.INTER_AREA)
    x = torch.from_numpy(g).float().div(127.5).sub(1.0)[None, None].to(dev)
    probs = None
    for m in models:
        p = m(x).softmax(2)
        probs = p if probs is None else probs + p
    probs = probs / len(models)
    ids = probs.argmax(2)[0].tolist()
    out, prev = [], -1
    for i in ids:
        if i != prev and i != BLANK:
            out.append(ITOS.get(int(i), ""))
        prev = i
    conf = float(probs.max(2).values[0].mean())
    return "".join(out), conf


def train_member(train_items, val_items, dev, seed, epochs, lr, batch):
    torch.manual_seed(seed)
    random.seed(seed)
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
    model.eval()
    return model


def eval_ensemble(models, test_items, dev):
    ex = num = den = 0
    for it in test_items:
        reads = []
        for f in it["files"]:
            g = cv2.imread(str(REAL / "images" / f), cv2.IMREAD_GRAYSCALE)
            if g is None:
                continue
            t, _ = ens_decode(models, g, dev)
            if t:
                reads.append(t)
        if not reads:
            continue
        voted = _vote(reads)
        d = decode_plate(voted)
        fin = d["plate"] or voted
        ex += fin == it["text"]
        num += lev(fin, it["text"])
        den += len(it["text"])
    return ex / max(len(test_items), 1), num / max(den, 1)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--members", type=int, default=3)
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
    print(f"vehicles : {len(truth)}   crops : "
          f"{sum(len(v) for v in by_track.values())}")
    print(f"members  : {args.members}\n", flush=True)

    singles, ensembles = [], []
    print(f"{'split':<8} {'single':>9} {'ensemble':>10} {'delta':>8}")
    print("-" * 40)
    for i in range(args.repeats):
        keys = sorted(by_track)
        rng = random.Random(1000 + i)
        rng.shuffle(keys)
        n = len(keys)
        n_test = max(20, int(n * 0.20))
        n_val = max(20, int(n * 0.20))
        test_k, val_k = keys[:n_test], keys[n_test:n_test + n_val]
        train_k = keys[n_test + n_val:]
        train_items = [it for k in train_k for it in by_track[k]]
        mk = lambda ks: [{"file": by_track[k][0]["file"], "text": truth[k],
                          "files": [c["file"] for c in by_track[k]]} for k in ks]
        val_items, test_items = mk(val_k), mk(test_k)

        members = [train_member(train_items, val_items, dev, 100 + i * 10 + j,
                                args.epochs, args.lr, args.batch)
                   for j in range(args.members)]

        # The first member IS the single-model baseline - same data, same
        # seed, same checkpoint selection - so the comparison isolates the
        # ensembling rather than confounding it with a different training run.
        s_acc, _, _ = score(members[0], test_items, dev, multiframe=True)
        e_acc, _ = eval_ensemble(members, test_items, dev)
        singles.append(s_acc)
        ensembles.append(e_acc)
        print(f"{i+1:<8} {s_acc*100:>8.1f}% {e_acc*100:>9.1f}% "
              f"{(e_acc-s_acc)*100:>+7.1f}", flush=True)
        del members
        torch.cuda.empty_cache()

    ms, me = statistics.mean(singles), statistics.mean(ensembles)
    d = [e - s for e, s in zip(ensembles, singles)]
    print("-" * 40)
    print(f"{'mean':<8} {ms*100:>8.1f}% {me*100:>9.1f}% "
          f"{statistics.mean(d)*100:>+7.1f}")
    print("\n" + "=" * 60)
    print(f"ensemble of {args.members}  :  {statistics.mean(d)*100:+.1f} points")
    if len(d) > 1:
        sd = statistics.stdev(d)
        print(f"spread            :  {min(d)*100:+.1f} to {max(d)*100:+.1f} "
              f"(sd {sd*100:.1f})")
        if statistics.mean(d) <= sd:
            print("\nThe gain does not clear its own spread. Three times the")
            print("compute for a difference this size is not worth taking.")
        else:
            print(f"\nConsistent gain. Cost is {args.members}x inference: about "
                  f"{0.76*args.members:.1f} ms per plate batched,")
            print("against 19.6 ms for the transformer that gained nothing.")
    print("=" * 60)


if __name__ == "__main__":
    main()
