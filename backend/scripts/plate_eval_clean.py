"""backend/scripts/plate_eval_clean.py — an accuracy figure that is not inflated
by checkpoint selection.

THE BIAS THIS REMOVES
  finetune_plate_recognizer.py evaluates on the held-out set every 5 epochs and
  saves whenever the score improves. With ~18 evaluations on 55 plates, the
  saved "best" is the maximum of 18 noisy draws from the same small set - which
  is optimistic by construction, not by accident. The checkpoint on disk scores
  81.8% that way, while the final epoch of the same run scored 52-62%. Neither
  is the model's real accuracy: the first is a maximum over the test set, the
  second is one arbitrary draw.

  So the honest figure needs three disjoint sets, split BY TRACK:
      train  - gradient updates
      val    - decides which epoch to keep; may be peeked at freely
      test   - scored exactly once, after the checkpoint is frozen

  The test number is then an estimate of performance on plates the model has
  neither trained on nor been selected against. It will be lower than 81.8%.
  That is the point - a number that survives a judge asking how it was
  obtained is worth more than a higher one that does not.

WHY THIS DECIDES THE CCPD QUESTION
  The brief now demands >90%. Pretraining on a large real-plate corpus might
  add 5-15 points. Whether that closes the gap depends entirely on where the
  honest baseline sits, and that is what this measures. Deciding before
  measuring would be guessing.

REPEATS
  --repeats runs the whole thing on several different splits. One split of 42
  test plates has a wide error bar; the spread across splits shows how wide,
  which is the number that was missing from every comparison run today.

USAGE
  python -m backend.scripts.plate_eval_clean
  python -m backend.scripts.plate_eval_clean --repeats 3 --epochs 90
"""
from __future__ import annotations

import argparse
import json
import random
import statistics
from collections import defaultdict
from pathlib import Path

import cv2
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset

from backend.scripts.indian_plate_grammar import decode_plate
from backend.scripts.plate_beam_decode import beam_decode
from backend.scripts.train_plate_recognizer import (BLANK, CHARS, CRNN, IMG_H,
                                                    IMG_W, ITOS, STOI,
                                                    ctc_decode)

REAL = Path("data/plate_real")
CKPT_IN = Path("models/plate_recognizer/best.pt")


def lev(a: str, b: str) -> int:
    dp = list(range(len(b) + 1))
    for i, ca in enumerate(a, 1):
        prev, dp[0] = dp[0], i
        for j, cb in enumerate(b, 1):
            cur = dp[j]
            dp[j] = min(dp[j] + 1, dp[j - 1] + 1, prev + (ca != cb))
            prev = cur
    return dp[len(b)]


class DS(Dataset):
    def __init__(self, items):
        self.items = items

    def __len__(self):
        return len(self.items)

    def __getitem__(self, i):
        it = self.items[i]
        img = cv2.imread(str(REAL / "images" / it["file"]), cv2.IMREAD_GRAYSCALE)
        if img is None:
            img = np.zeros((IMG_H, IMG_W), np.uint8)
        if random.random() < 0.4:
            a, b = random.uniform(0.85, 1.15), random.uniform(-15, 15)
            img = np.clip(img.astype(np.float32) * a + b, 0, 255).astype(np.uint8)
        if random.random() < 0.25:
            img = cv2.GaussianBlur(img, (3, 3), random.uniform(0.3, 0.7))
        if random.random() < 0.3:
            h, w = img.shape
            M = np.float32([[1, 0, random.randint(-2, 2)],
                            [0, 1, random.randint(-1, 1)]])
            img = cv2.warpAffine(img, M, (w, h), borderMode=cv2.BORDER_REPLICATE)
        img = cv2.resize(img, (IMG_W, IMG_H), interpolation=cv2.INTER_AREA)
        x = torch.from_numpy(img).float().div(127.5).sub(1.0).unsqueeze(0)
        lab = torch.tensor([STOI[c] for c in it["text"] if c in STOI],
                           dtype=torch.long)
        return x, lab, len(lab)


def collate(b):
    xs, ls, ns = zip(*b)
    return torch.stack(xs), torch.cat(ls), torch.tensor(ns, dtype=torch.long)


def _read_one(model, fname, dev, want_conf: bool = False,
              decoder: str = "greedy"):
    img = cv2.imread(str(REAL / "images" / fname), cv2.IMREAD_GRAYSCALE)
    if img is None:
        return (None, 0.0) if want_conf else None
    img = cv2.resize(img, (IMG_W, IMG_H), interpolation=cv2.INTER_AREA)
    x = torch.from_numpy(img).float().div(127.5).sub(1.0)
    logits = model(x[None, None].to(dev))
    if decoder == "beam":
        cands = beam_decode(logits, ITOS, blank=BLANK)
        text = cands[0][0] if cands else ""
    else:
        text = ctc_decode(logits)[0]
    if not want_conf:
        return text
    # Confidence = mean top-class probability over the timesteps that actually
    # emitted a character. Blank frames are excluded: a plate is mostly blank
    # timesteps, and including them measures how sure the model is that nothing
    # is there, which says nothing about the glyphs it did emit.
    p = logits.softmax(2)[0]
    top, idx = p.max(1)
    keep = idx != BLANK
    conf = float(top[keep].mean()) if bool(keep.any()) else 0.0
    return text, conf


def _vote(reads: list[str]) -> str:
    """Character-position majority across a vehicle's frames.

    A deployment never sees one crop of a vehicle - it sees the whole track.
    Different frames blur different glyphs, so an error in one frame is often
    outvoted by the others. This only fixes RANDOM disagreement; a confusion
    the model makes identically on every frame (6 for G, 0 for O) survives
    voting untouched, which is why grammar decoding still runs afterwards.

    Only reads of the modal length vote, so a dropped or doubled character in
    one frame cannot shift every position after it.
    """
    reads = [r for r in reads if r]
    if not reads:
        return ""
    lens = defaultdict(int)
    for r in reads:
        lens[len(r)] += 1
    best_len = max(lens, key=lambda L: (lens[L], L))
    same = [r for r in reads if len(r) == best_len]
    out = []
    for i in range(best_len):
        c = defaultdict(int)
        for r in same:
            c[r[i]] += 1
        out.append(max(c, key=c.get))
    return "".join(out)


def score(model, items, dev, multiframe: bool = False,
          decoder: str = "greedy"):
    """Returns (exact, CER, records) where each record is (correct, confidence).

    The records feed the precision/coverage curve: a system that emits every
    read at 51% is not the same product as one that emits only the reads it is
    sure of. Which of those two the brief's ">90% correct" refers to is decided
    by that curve, not by the headline average.
    """
    model.eval()
    ex = num = den = 0
    rec = []
    with torch.no_grad():
        for it in items:
            if multiframe and it.get("files"):
                got = [_read_one(model, f, dev, True, decoder)
                       for f in it["files"]]
                got = [(t, c) for t, c in got if t]
                if not got:
                    continue
                pred = _vote([t for t, _ in got])
                # Two independent signals of trust, multiplied: how sure the
                # network was per frame, and how often the frames agreed with
                # the voted result. A confident read that every frame disputes
                # deserves neither score on its own.
                conf = sum(c for _, c in got) / len(got)
                agree = sum(t == pred for t, _ in got) / len(got)
                conf *= agree
            else:
                pred, conf = _read_one(model, it["file"], dev, True, decoder)
            if pred is None:
                continue
            d = decode_plate(pred)
            final = d["plate"] or pred
            ok = final == it["text"]
            ex += ok
            num += lev(final, it["text"])
            den += len(it["text"])
            rec.append((ok, conf))
    return ex / max(len(items), 1), num / max(den, 1), rec


def pick_threshold(val_rec, test_rec, target: float):
    """Choose the operating threshold on VAL, then report it on TEST.

    An operator does not want every reading; they want the readings the system
    stands behind, and a flag on the rest. That needs a confidence cut-off, and
    where the cut-off is set decides both how often the system speaks and how
    often it is right.

    The cut-off must be chosen on data the reported number does not come from.
    Reading it off the test curve - picking whichever threshold happens to
    reach the target there - reports the maximum of a dozen thresholds as if it
    were the performance of one, and the number will not survive contact with
    new footage. So: the lowest threshold whose val precision clears the
    target, which is the one that keeps the most coverage; then whatever that
    same threshold does on test, reported as it falls.

    Returns (threshold, coverage on test, precision on test, val precision).
    """
    grid = [i / 100 for i in range(0, 100, 2)]
    chosen = None
    for th in grid:
        kept = [r for r in val_rec if r[1] >= th]
        if len(kept) < 8:
            break
        p = sum(ok for ok, _ in kept) / len(kept)
        if p >= target:
            chosen = th
            break
    if chosen is None:
        chosen = max(grid)
    vkept = [r for r in val_rec if r[1] >= chosen]
    vprec = (sum(ok for ok, _ in vkept) / len(vkept)) if vkept else 0.0
    tkept = [r for r in test_rec if r[1] >= chosen]
    if not tkept:
        return chosen, 0.0, 0.0, vprec
    return (chosen, len(tkept) / max(len(test_rec), 1),
            sum(ok for ok, _ in tkept) / len(tkept), vprec)


def one_run(seed, epochs, lr, batch, dev, by_track, truth):
    keys = sorted(by_track)
    rng = random.Random(seed)
    rng.shuffle(keys)
    n = len(keys)
    n_test = max(20, int(n * 0.20))
    n_val = max(20, int(n * 0.20))
    test_k = keys[:n_test]
    val_k = keys[n_test:n_test + n_val]
    train_k = keys[n_test + n_val:]

    train_items = [it for k in train_k for it in by_track[k]]
    # Carry every frame of a test vehicle, not just the first, so the same
    # split can be scored single-frame and multi-frame with no other change.
    val_items = [{"file": by_track[k][0]["file"], "text": truth[k],
                  "files": [c["file"] for c in by_track[k]]} for k in val_k]
    test_items = [{"file": by_track[k][0]["file"], "text": truth[k],
                   "files": [c["file"] for c in by_track[k]]} for k in test_k]

    ck = torch.load(CKPT_IN, map_location=dev, weights_only=False)
    model = CRNN(len(CHARS) + 1).to(dev)
    model.load_state_dict(ck["model"])

    tl = DataLoader(DS(train_items), batch_size=batch, shuffle=True,
                    collate_fn=collate, num_workers=0,
                    drop_last=len(train_items) > batch)
    crit = nn.CTCLoss(blank=BLANK, zero_infinity=True)
    opt = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-4)

    best_val = -1.0
    best_state = None
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
            if v > best_val:
                best_val = v
                best_state = {k: t.detach().clone()
                              for k, t in model.state_dict().items()}

    # Freeze the selected checkpoint, THEN look at test for the first time.
    model.load_state_dict(best_state)
    # Four readings of the SAME frozen model on the SAME plates. Because only
    # the decoder and the frame count change, the split-to-split noise that
    # swamped every earlier comparison cancels between these columns.
    t_acc, _, _ = score(model, test_items, dev)
    b_acc, _, _ = score(model, test_items, dev, decoder="beam")
    m_acc, _, m_rec = score(model, test_items, dev, multiframe=True)
    mb_acc, _, mb_rec = score(model, test_items, dev, multiframe=True,
                              decoder="beam")
    # Confidence records on VAL as well as test. The operating threshold has to
    # be chosen somewhere, and choosing it on test - which is what an earlier
    # run of this script effectively did by reading the threshold off the test
    # curve - reports the best of several thresholds as though it were the
    # performance of one. Selecting on val and applying to test costs a little
    # accuracy on paper and is the only figure that would survive a judge
    # asking how the threshold was picked.
    _, _, v_rec = score(model, val_items, dev, multiframe=True)
    frames = sum(len(it["files"]) for it in test_items) / max(len(test_items), 1)
    return (len(train_k), len(val_k), len(test_k), best_val,
            t_acc, b_acc, m_acc, mb_acc, frames, m_rec, v_rec)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--epochs", type=int, default=90)
    ap.add_argument("--batch", type=int, default=32)
    ap.add_argument("--lr", type=float, default=2e-4)
    ap.add_argument("--repeats", type=int, default=3,
                    help="Different splits. The SPREAD across these is the "
                         "error bar that every earlier comparison lacked.")
    ap.add_argument("--target-precision", type=float, default=0.80,
                    help="Precision the operating threshold aims for. The "
                         "threshold is chosen on val and reported on test.")
    args = ap.parse_args()

    verified = [json.loads(l) for l in (REAL / "verified_all.jsonl").open(encoding="utf-8")]
    verified = [v for v in verified
                if v["text"] and all(c in STOI for c in v["text"])]
    truth = {(v["camera"], v["track"]): v["text"] for v in verified}

    rows = [json.loads(l) for l in (REAL / "labels.jsonl").open(encoding="utf-8")]
    by_track = defaultdict(list)
    for r in rows:
        k = (r["camera"], r["track"])
        if k in truth:
            by_track[k].append({"file": r["file"], "text": truth[k]})

    dev = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"labelled vehicles : {len(truth)}")
    print(f"crops             : {sum(len(v) for v in by_track.values())}")
    print(f"device            : {dev}\n", flush=True)

    cols = {"1f greedy": [], "1f beam": [], "voted greedy": [], "voted beam": []}
    all_rec = []
    gated = []          # (threshold from val, coverage on test, precision on test)
    print(f"{'split':<8} {'1f greedy':>10} {'1f beam':>9} "
          f"{'voted greedy':>13} {'voted beam':>11}")
    print("-" * 56)
    for i in range(args.repeats):
        seed = 1000 + i
        (ntr, nva, nte, val, acc, bacc,
         macc, mbacc, frames, rec, vrec) = one_run(seed, args.epochs, args.lr,
                                                   args.batch, dev, by_track,
                                                   truth)
        for k, v in zip(cols, (acc, bacc, macc, mbacc)):
            cols[k].append(v)
        all_rec.extend(rec)
        gated.append(pick_threshold(vrec, rec, args.target_precision))
        print(f"{i+1:<8} {acc*100:>9.1f}% {bacc*100:>8.1f}% "
              f"{macc*100:>12.1f}% {mbacc*100:>10.1f}%", flush=True)

    base = statistics.mean(cols["1f greedy"])
    print("\n" + "=" * 68)
    print("HONEST HELD-OUT ACCURACY  (checkpoint chosen on val, test seen once)")
    print(f"{args.repeats} splits, {nte} plates each, train {ntr} / val {nva}")
    print("=" * 68)
    print(f"{'configuration':<20} {'mean':>8} {'range':>15} {'vs greedy':>11}")
    print("-" * 68)
    for k, v in cols.items():
        mean = statistics.mean(v)
        print(f"{k:<20} {mean*100:>7.1f}% "
              f"{min(v)*100:>7.1f}-{max(v)*100:<7.1f}"
              f"{(mean-base)*100:>+10.1f}")
    print("=" * 68)
    accs = cols["1f greedy"]
    m = base
    mm = statistics.mean(cols["voted beam"])
    sd = statistics.stdev(accs) if len(accs) > 1 else 0.0
    print("\nThe four columns share a model and a test set per split, so the")
    print("differences BETWEEN them are paired and far more reliable than the")
    print("split-to-split spread within any one column suggests.")
    print(f"\nAny future change smaller than about {max(sd*2*100, 5):.0f} points is")
    print("indistinguishable from split noise with this much labelled data.")
    print(f"Gap to the 90% the brief demands: {(0.90 - mm)*100:.0f} points.\n")

    # Precision vs coverage. Reads pooled across all splits - each is a plate
    # the model that read it had never trained on, so pooling adds sample size
    # without leaking anything.
    print("=" * 68)
    print("PRECISION AT A CONFIDENCE THRESHOLD  (voted reads, all splits)")
    print("=" * 68)
    print("Of the plates the system CHOOSES to report, how many are right?")
    print(f"{'threshold':<12} {'reported':>10} {'coverage':>10} {'precision':>11}")
    print("-" * 68)
    hit90 = None
    for th in (0.0, 0.50, 0.60, 0.70, 0.80, 0.85, 0.90, 0.95):
        kept = [r for r in all_rec if r[1] >= th]
        if len(kept) < 10:
            continue
        prec = sum(ok for ok, _ in kept) / len(kept)
        cov = len(kept) / max(len(all_rec), 1)
        mark = ""
        if prec >= 0.90 and hit90 is None:
            hit90 = (th, cov, prec)
            mark = "  <-- 90% precision"
        print(f">= {th:<9.2f} {len(kept):>10} {cov*100:>9.0f}% "
              f"{prec*100:>10.1f}%{mark}")
    print("=" * 68)
    print("\nThat table is diagnostic only - reading a threshold off it would be")
    print("choosing an operating point on the test set. The honest version is")
    print("below: the threshold is picked on val, then applied to test unseen.")

    print("\n" + "=" * 68)
    print(f"OPERATING POINT  (threshold chosen on VAL for "
          f">={args.target_precision*100:.0f}% precision)")
    print("=" * 68)
    print(f"{'split':<8} {'threshold':>10} {'val prec':>10} "
          f"{'TEST cov':>10} {'TEST prec':>11}")
    print("-" * 68)
    for i, (th, cov, prec, vp) in enumerate(gated, 1):
        print(f"{i:<8} {th:>10.2f} {vp*100:>9.1f}% "
              f"{cov*100:>9.1f}% {prec*100:>10.1f}%")
    print("-" * 68)
    mc = statistics.mean(c for _, c, _, _ in gated)
    mp = statistics.mean(p for _, _, p, _ in gated)
    print(f"{'mean':<8} {'':>10} {'':>10} {mc*100:>9.1f}% {mp*100:>10.1f}%")
    print("=" * 68)
    print(f"\nOn plates the system commits to - {mc*100:.0f}% of vehicles - it is")
    print(f"{mp*100:.0f}% correct. The rest are flagged for review rather than")
    print("reported wrongly. Both figures come from plates that were never")
    print("trained on and never used to pick the threshold.")


if __name__ == "__main__":
    main()
