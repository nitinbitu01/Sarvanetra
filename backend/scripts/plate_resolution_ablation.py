"""backend/scripts/plate_resolution_ablation.py — does multi-frame fusion help
once the recogniser is actually able to see the extra pixels?

THE FLAW THIS CORRECTS
  Fusion was measured and scored 28.8% against string voting's 36.4%, which
  looked like a refutation. It was not a fair test. The recogniser takes a
  32x128 input, so a fused image three times larger is resampled straight back
  down before it is read - every pixel fusion recovered was discarded on the
  way in. At 128px across ten characters there are 12.8 pixels per character,
  which is roughly the resolution at which strokes stop being separable.

  So fusion and input size are entangled: neither can help alone. Fusion
  without a larger input has nothing to deliver its gain through, and a larger
  input without fusion is upsampling with no new information in it.

FOUR ARMS, BECAUSE ONE COMPARISON WOULD NOT SAY WHICH CHANGE DID THE WORK
      32x128  unfused    the current pipeline, as a baseline
      32x128  fused      fusion with no room to pay off
      64x256  unfused    more input, no new information
      64x256  fused      both together
  Confounding these was already a mistake once in this project - a label batch
  looked harmful because the test set had changed at the same time - and the
  fix each time is to vary one thing at a time and pay for the extra runs.

COST IS PART OF THE RESULT
  A 64x256 input is four times the pixels and roughly four times the inference
  cost: about 3 ms per plate against 0.76 ms, still six times faster than the
  transformer that gained nothing. If the accuracy gain does not clear the
  split noise, the larger input is not worth taking regardless.

USAGE
  python -m backend.scripts.plate_resolution_ablation --build
  python -m backend.scripts.plate_resolution_ablation --repeats 2
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
from backend.scripts.plate_eval_clean import _vote, lev
from backend.scripts.plate_multiframe_fusion import fuse, load_groups
from backend.scripts.train_plate_recognizer import BLANK, CHARS, ITOS, STOI

REAL = Path("data/plate_real")
FUSED = REAL / "fused"
CKPT_IN = Path("models/plate_recognizer/best.pt")


class CRNN2(nn.Module):
    """The project CRNN, made height-agnostic.

    The original ends with a (2,1) convolution that assumes the feature map is
    exactly two rows tall, which holds for a 32px input and breaks for 64px.
    Collapsing height with an adaptive pool instead makes the same trunk work
    at any input height, so the two arms of this ablation differ ONLY in the
    resolution they are fed - not in architecture, which would confound them.
    """

    def __init__(self, n_classes: int) -> None:
        super().__init__()
        self.cnn = nn.Sequential(
            nn.Conv2d(1, 64, 3, 1, 1), nn.BatchNorm2d(64), nn.ReLU(True),
            nn.MaxPool2d(2, 2),
            nn.Conv2d(64, 128, 3, 1, 1), nn.BatchNorm2d(128), nn.ReLU(True),
            nn.MaxPool2d(2, 2),
            nn.Conv2d(128, 256, 3, 1, 1), nn.BatchNorm2d(256), nn.ReLU(True),
            nn.Conv2d(256, 256, 3, 1, 1), nn.BatchNorm2d(256), nn.ReLU(True),
            nn.MaxPool2d((2, 1), (2, 1)),
            nn.Conv2d(256, 512, 3, 1, 1), nn.BatchNorm2d(512), nn.ReLU(True),
            nn.MaxPool2d((2, 1), (2, 1)),
            nn.Conv2d(512, 512, 3, 1, 1), nn.BatchNorm2d(512), nn.ReLU(True),
        )
        self.pool = nn.AdaptiveAvgPool2d((1, None))      # collapse height only
        self.rnn = nn.LSTM(512, 256, num_layers=2, bidirectional=True,
                           batch_first=True, dropout=0.15)
        self.fc = nn.Linear(512, n_classes)

    def forward(self, x):
        f = self.pool(self.cnn(x)).squeeze(2).permute(0, 2, 1)
        r, _ = self.rnn(f)
        return self.fc(r)


def decode(logits):
    out = []
    for seq in logits.argmax(dim=2).cpu().numpy():
        s, prev = [], -1
        for k in seq:
            if k != prev and k != BLANK:
                s.append(ITOS.get(int(k), ""))
            prev = k
        out.append("".join(s))
    return out


class DS(Dataset):
    def __init__(self, items, h, w, train):
        self.items, self.h, self.w, self.train = items, h, w, train

    def __len__(self):
        return len(self.items)

    def __getitem__(self, i):
        it = self.items[i]
        g = cv2.imread(it["path"], cv2.IMREAD_GRAYSCALE)
        if g is None:
            g = np.zeros((self.h, self.w), np.uint8)
        if self.train:
            if random.random() < 0.4:
                a, b = random.uniform(0.85, 1.15), random.uniform(-15, 15)
                g = np.clip(g.astype(np.float32) * a + b, 0, 255).astype(np.uint8)
            if random.random() < 0.25:
                g = cv2.GaussianBlur(g, (3, 3), random.uniform(0.3, 0.7))
        g = cv2.resize(g, (self.w, self.h), interpolation=cv2.INTER_AREA)
        x = torch.from_numpy(g).float().div(127.5).sub(1.0).unsqueeze(0)
        lab = torch.tensor([STOI[c] for c in it["text"] if c in STOI],
                           dtype=torch.long)
        return x, lab, len(lab)


def collate(b):
    xs, ls, ns = zip(*b)
    return torch.stack(xs), torch.cat(ls), torch.tensor(ns, dtype=torch.long)


def build_fused(args) -> None:
    """Cache one fused image per labelled vehicle."""
    FUSED.mkdir(parents=True, exist_ok=True)
    groups, truth = load_groups(min_frames=2)
    print(f"vehicles with >=2 frames : {len(groups)}", flush=True)
    n = 0
    index = {}
    for i, (k, files) in enumerate(sorted(groups.items()), 1):
        crops = []
        for f in files:
            g = cv2.imread(str(REAL / "images" / f), cv2.IMREAD_GRAYSCALE)
            if g is not None:
                crops.append(g)
        if len(crops) < 2:
            continue
        img, used = fuse(crops, args.upscale)
        if img is None:
            continue
        name = f"{k[0]}_t{k[1]}.png"          # PNG: fusion output is the one
        cv2.imwrite(str(FUSED / name), img)   # place JPEG artefacts would hurt
        index[f"{k[0]}|{k[1]}"] = {"file": name, "text": truth[k],
                                   "frames": used}
        n += 1
        if i % 50 == 0:
            print(f"  {i}/{len(groups)}  ({n} written)", flush=True)
    (FUSED / "index.json").write_text(json.dumps(index), encoding="utf-8")
    print(f"\nfused images: {n} -> {FUSED}")


def train_eval(arm, train_items, val_items, test_items, dev, args):
    h, w = arm["h"], arm["w"]
    torch.manual_seed(arm["seed"])
    model = CRNN2(len(CHARS) + 1).to(dev)
    # The pretrained weights are for the original trunk; only the layers whose
    # shapes still match are carried over, and the rest start fresh. Loading
    # nothing would put this arm at a disadvantage the ablation is not testing.
    ck = torch.load(CKPT_IN, map_location=dev, weights_only=False)
    sd = model.state_dict()
    keep = {k: v for k, v in ck["model"].items()
            if k in sd and sd[k].shape == v.shape}
    model.load_state_dict({**sd, **keep})

    tl = DataLoader(DS(train_items, h, w, True), batch_size=args.batch,
                    shuffle=True, collate_fn=collate, num_workers=0,
                    drop_last=len(train_items) > args.batch)
    crit = nn.CTCLoss(blank=BLANK, zero_infinity=True)
    opt = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=1e-4)

    def run(items):
        model.eval()
        ex = num = den = 0
        with torch.no_grad():
            for it in items:
                reads = []
                for p in it["paths"]:
                    g = cv2.imread(p, cv2.IMREAD_GRAYSCALE)
                    if g is None:
                        continue
                    g = cv2.resize(g, (w, h), interpolation=cv2.INTER_AREA)
                    x = torch.from_numpy(g).float().div(127.5).sub(1.0)
                    reads.append(decode(model(x[None, None].to(dev)))[0])
                if not reads:
                    continue
                voted = _vote(reads)
                d = decode_plate(voted)
                fin = d["plate"] or voted
                ex += fin == it["text"]
                num += lev(fin, it["text"])
                den += len(it["text"])
        return ex / max(len(items), 1), num / max(den, 1)

    best_v, best_state = -1.0, None
    for ep in range(1, args.epochs + 1):
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
        if ep % 5 == 0 or ep == args.epochs:
            v, _ = run(val_items)
            if v > best_v:
                best_v = v
                best_state = {k: t.detach().clone()
                              for k, t in model.state_dict().items()}
    model.load_state_dict(best_state)
    acc, cer = run(test_items)
    del model
    torch.cuda.empty_cache()
    return acc, cer


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--build", action="store_true")
    ap.add_argument("--upscale", type=int, default=3)
    ap.add_argument("--repeats", type=int, default=2)
    ap.add_argument("--epochs", type=int, default=70)
    ap.add_argument("--batch", type=int, default=32)
    ap.add_argument("--lr", type=float, default=2e-4)
    args = ap.parse_args()

    if args.build:
        build_fused(args)
        return

    index = json.loads((FUSED / "index.json").read_text(encoding="utf-8"))
    groups, truth = load_groups(min_frames=2)
    keys = [k for k in sorted(groups) if f"{k[0]}|{k[1]}" in index]
    print(f"vehicles usable : {len(keys)}")

    unfused = {k: [str(REAL / "images" / f) for f in groups[k]] for k in keys}
    fused = {k: [str(FUSED / index[f"{k[0]}|{k[1]}"]["file"])] for k in keys}

    dev = "cuda" if torch.cuda.is_available() else "cpu"
    ARMS = [("32x128 unfused", 32, 128, False), ("32x128 fused", 32, 128, True),
            ("64x256 unfused", 64, 256, False), ("64x256 fused", 64, 256, True)]
    res = defaultdict(list)

    for i in range(args.repeats):
        rng = random.Random(1000 + i)
        ks = list(keys)
        rng.shuffle(ks)
        n = len(ks)
        n_test = max(20, int(n * 0.20))
        n_val = max(20, int(n * 0.20))
        test_k, val_k = ks[:n_test], ks[n_test:n_test + n_val]
        train_k = ks[n_test + n_val:]

        for name, h, w, use_fused in ARMS:
            src = fused if use_fused else unfused
            tr = [{"path": p, "text": truth[k]} for k in train_k for p in src[k]]
            mk = lambda kk: [{"paths": src[k], "text": truth[k]} for k in kk]
            acc, cer = train_eval({"h": h, "w": w, "seed": 100 + i},
                                  tr, mk(val_k), mk(test_k), dev, args)
            res[name].append(acc)
            print(f"  split {i+1}  {name:<16} {acc*100:>5.1f}%  "
                  f"CER {cer*100:.1f}%", flush=True)
        print()

    print("=" * 60)
    print(f"RESOLUTION x FUSION   ({args.repeats} splits, {n_test} test each)")
    print("=" * 60)
    base = statistics.mean(res["32x128 unfused"])
    for name, _, _, _ in ARMS:
        v = res[name]
        print(f"{name:<18} {statistics.mean(v)*100:>6.1f}%   "
              f"({min(v)*100:.1f}-{max(v)*100:.1f})   "
              f"{(statistics.mean(v)-base)*100:>+6.1f}")
    print("=" * 60)
    print("\nRead the two 'unfused' rows against each other to see what input")
    print("size alone bought, and each 'fused' row against the 'unfused' row")
    print("at the same size to see what fusion bought there. A gain that only")
    print("appears in the bottom row means the two are doing the work jointly,")
    print("which is the hypothesis under test.")


if __name__ == "__main__":
    main()
