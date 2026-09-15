"""backend/scripts/plate_final_model.py - settle the input-resolution question
fairly, then train the model that ships.

THE FLAW IN THE FIRST ATTEMPT
  A previous ablation reported +13.8 points for a 64x256 input and the figure
  cannot be trusted. It compared two runs of CRNN2, a variant whose final layer
  is a 3x3 convolution where the original uses a (2,1) - so the pretrained
  weights for that layer never loaded. CRNN2 scored 28.7% at 32x128 where the
  original CRNN scores 43.0% on the same footage, meaning the variant carries a
  14-point handicap of its own. Its 42.5% at 64x256 is therefore roughly the
  original's baseline, and the "+13.8" may be nothing more than the larger
  input compensating for weights that were thrown away.

  Shipping on that number would have bought nothing and looked like a win.

THE FAIR TEST
  Keep the architecture EXACTLY as it is and add one height-only pooling stage
  for the doubled input. Pooling has no parameters, so every convolution and
  batch-norm weight loads unchanged and the two arms differ in one thing only:
  how many pixels reach the network.

      H=32:  32 -> 16 -> 8 -> 4 -> 2 -> conv(2,1) -> 1
      H=64:  64 -> 32 -> 16 -> 8 -> 4 -> [extra pool] -> 2 -> conv(2,1) -> 1

  Width follows the same trunk: 128px gives 32 timesteps, 256px gives 64. Both
  are comfortably above the 11 characters CTC needs.

WHY MORE PIXELS SHOULD MATTER AT ALL
  Ten characters across a 128px input is 12.8 pixels each, which is close to
  where the strokes that separate 8 from B stop being separable. At 256px it is
  25.6. The plate crops themselves are 45-130px wide, so the larger input is not
  inventing detail - it is declining to throw away the detail the crop already
  has by downsampling it less aggressively.

COST IS REPORTED, NOT ASSUMED
  Four times the pixels is roughly four times the inference cost. The figure is
  measured here rather than estimated, because a model that reads better and
  cannot keep up with the cameras is not an improvement.

USAGE
  python -m backend.scripts.plate_final_model --compare --repeats 3
  python -m backend.scripts.plate_final_model --train-final --members 3
"""
from __future__ import annotations

import argparse
import json
import random
import statistics
import time
from collections import defaultdict
from pathlib import Path

import cv2
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset

from backend.scripts.indian_plate_grammar import decode_plate
from backend.scripts.plate_eval_clean import _vote, lev
from backend.scripts.train_plate_recognizer import BLANK, CHARS, ITOS, STOI

REAL = Path("data/plate_real")
CKPT_IN = Path("models/plate_recognizer/best.pt")
OUT_DIR = Path("models/plate_recognizer")


class PlateCRNN(nn.Module):
    """The project CRNN, with an optional extra height-pool for tall inputs.

    Every convolution and batch-norm is identical to the original, so the
    pretrained checkpoint loads completely in both configurations. The only
    difference is one parameterless pooling stage, which is what makes the
    resolution comparison honest.
    """

    def __init__(self, n_classes: int, img_h: int = 32) -> None:
        super().__init__()
        layers = [
            nn.Conv2d(1, 64, 3, 1, 1), nn.BatchNorm2d(64), nn.ReLU(True),
            nn.MaxPool2d(2, 2),
            nn.Conv2d(64, 128, 3, 1, 1), nn.BatchNorm2d(128), nn.ReLU(True),
            nn.MaxPool2d(2, 2),
            nn.Conv2d(128, 256, 3, 1, 1), nn.BatchNorm2d(256), nn.ReLU(True),
            nn.Conv2d(256, 256, 3, 1, 1), nn.BatchNorm2d(256), nn.ReLU(True),
            nn.MaxPool2d((2, 1), (2, 1)),
            nn.Conv2d(256, 512, 3, 1, 1), nn.BatchNorm2d(512), nn.ReLU(True),
            nn.MaxPool2d((2, 1), (2, 1)),
        ]
        # One extra halving per doubling of input height, so the final (2,1)
        # convolution always sees exactly two rows and collapses to one.
        extra = 0
        h = img_h
        while h // (16 * (2 ** extra)) > 2:
            extra += 1
        layers += [nn.MaxPool2d((2, 1), (2, 1))] * extra
        layers += [nn.Conv2d(512, 512, (2, 1), 1, 0), nn.BatchNorm2d(512),
                   nn.ReLU(True)]
        self.cnn = nn.Sequential(*layers)
        self.rnn = nn.LSTM(512, 256, num_layers=2, bidirectional=True,
                           batch_first=True, dropout=0.15)
        self.fc = nn.Linear(512, n_classes)

    def forward(self, x):
        f = self.cnn(x).squeeze(2).permute(0, 2, 1)
        r, _ = self.rnn(f)
        return self.fc(r)


def load_pretrained(model, dev):
    """Load the checkpoint and report how much of it actually landed.

    Silence here is how the previous ablation went wrong: weights failed to
    load, the run trained anyway, and the resulting handicap looked like a
    property of the input size. The count is printed so that cannot recur.
    """
    ck = torch.load(CKPT_IN, map_location=dev, weights_only=False)
    sd = model.state_dict()
    src = ck["model"]
    # Layer names shift when pooling stages are inserted, so match by ORDER of
    # parameterised layers rather than by name.
    tgt_keys = [k for k in sd if sd[k].dim() > 0]
    src_keys = [k for k in src if src[k].dim() > 0]
    loaded = 0
    new = dict(sd)
    si = 0
    for tk in tgt_keys:
        while si < len(src_keys) and src[src_keys[si]].shape != sd[tk].shape:
            si += 1
        if si < len(src_keys):
            new[tk] = src[src_keys[si]]
            si += 1
            loaded += 1
    model.load_state_dict(new)
    return loaded, len(tgt_keys)


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
        g = cv2.imread(str(REAL / "images" / it["file"]), cv2.IMREAD_GRAYSCALE)
        if g is None:
            g = np.zeros((self.h, self.w), np.uint8)
        if self.train:
            if random.random() < 0.4:
                a, b = random.uniform(0.85, 1.15), random.uniform(-15, 15)
                g = np.clip(g.astype(np.float32) * a + b, 0, 255).astype(np.uint8)
            if random.random() < 0.25:
                g = cv2.GaussianBlur(g, (3, 3), random.uniform(0.3, 0.7))
            if random.random() < 0.3:
                hh, ww = g.shape
                M = np.float32([[1, 0, random.randint(-2, 2)],
                                [0, 1, random.randint(-1, 1)]])
                g = cv2.warpAffine(g, M, (ww, hh), borderMode=cv2.BORDER_REPLICATE)
        g = cv2.resize(g, (self.w, self.h), interpolation=cv2.INTER_AREA)
        x = torch.from_numpy(g).float().div(127.5).sub(1.0).unsqueeze(0)
        lab = torch.tensor([STOI[c] for c in it["text"] if c in STOI],
                           dtype=torch.long)
        return x, lab, len(lab)


def collate(b):
    xs, ls, ns = zip(*b)
    return torch.stack(xs), torch.cat(ls), torch.tensor(ns, dtype=torch.long)


def read_vehicle(models, files, h, w, dev):
    """Voted read of one vehicle, averaging member probabilities per frame."""
    reads = []
    with torch.no_grad():
        for f in files:
            g = cv2.imread(str(REAL / "images" / f), cv2.IMREAD_GRAYSCALE)
            if g is None:
                continue
            g = cv2.resize(g, (w, h), interpolation=cv2.INTER_AREA)
            x = torch.from_numpy(g).float().div(127.5).sub(1.0)[None, None].to(dev)
            probs = None
            for m in models:
                p = m(x).softmax(2)
                probs = p if probs is None else probs + p
            ids = (probs / len(models)).argmax(2)[0].tolist()
            s, prev = [], -1
            for k in ids:
                if k != prev and k != BLANK:
                    s.append(ITOS.get(int(k), ""))
                prev = k
            reads.append("".join(s))
    return _vote(reads) if reads else ""


def score(models, items, h, w, dev):
    for m in models:
        m.eval()
    ex = num = den = 0
    for it in items:
        raw = read_vehicle(models, it["files"], h, w, dev)
        if not raw:
            continue
        d = decode_plate(raw)
        fin = d["plate"] or raw
        ex += fin == it["text"]
        num += lev(fin, it["text"])
        den += len(it["text"])
    return ex / max(len(items), 1), num / max(den, 1)


def train_one(train_items, val_items, h, w, dev, args, seed):
    torch.manual_seed(seed)
    random.seed(seed)
    model = PlateCRNN(len(CHARS) + 1, img_h=h).to(dev)
    loaded, total = load_pretrained(model, dev)
    tl = DataLoader(DS(train_items, h, w, True), batch_size=args.batch,
                    shuffle=True, collate_fn=collate, num_workers=0,
                    drop_last=len(train_items) > args.batch)
    crit = nn.CTCLoss(blank=BLANK, zero_infinity=True)
    opt = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=1e-4)
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
            v, _ = score([model], val_items, h, w, dev)
            if v > best_v:
                best_v = v
                best_state = {k: t.detach().clone()
                              for k, t in model.state_dict().items()}
    model.load_state_dict(best_state)
    model.eval()
    return model, loaded, total


def load_data():
    verified = [json.loads(l) for l in (REAL / "verified_all.jsonl").open(encoding="utf-8")]
    truth = {(v["camera"], v["track"]): v["text"] for v in verified
             if v["text"] and all(c in STOI for c in v["text"])}
    by_track = defaultdict(list)
    for line in (REAL / "labels.jsonl").open(encoding="utf-8"):
        r = json.loads(line)
        k = (r["camera"], r["track"])
        if k in truth:
            by_track[k].append({"file": r["file"], "text": truth[k]})
    return truth, by_track


def split(by_track, seed):
    keys = sorted(by_track)
    random.Random(seed).shuffle(keys)
    n = len(keys)
    n_test = max(20, int(n * 0.20))
    n_val = max(20, int(n * 0.20))
    return keys[:n_test], keys[n_test:n_test + n_val], keys[n_test + n_val:]


def compare(args) -> None:
    truth, by_track = load_data()
    dev = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"vehicles {len(truth)}   crops "
          f"{sum(len(v) for v in by_track.values())}\n", flush=True)

    ARMS = [("32x128", 32, 128), ("64x256", 64, 256)]
    res = defaultdict(list)
    for i in range(args.seed_offset, args.seed_offset + args.repeats):
        test_k, val_k, train_k = split(by_track, 1000 + i)
        mk = lambda ks: [{"files": [c["file"] for c in by_track[k]],
                          "text": truth[k]} for k in ks]
        val_items, test_items = mk(val_k), mk(test_k)
        tr = [it for k in train_k for it in by_track[k]]
        for name, h, w in ARMS:
            m, loaded, total = train_one(tr, val_items, h, w, dev, args,
                                         100 + i)
            acc, cer = score([m], test_items, h, w, dev)
            res[name].append(acc)
            print(f"  split {i+1}  {name:<8} {acc*100:>5.1f}%  CER {cer*100:.1f}%"
                  f"   (pretrained {loaded}/{total} tensors)", flush=True)
            del m
            torch.cuda.empty_cache()
        print()

    print("=" * 58)
    print(f"INPUT RESOLUTION, SAME ARCHITECTURE  ({args.repeats} splits)")
    print("=" * 58)
    base = statistics.mean(res["32x128"])
    for name, _, _ in ARMS:
        v = res[name]
        print(f"{name:<10} {statistics.mean(v)*100:>6.1f}%  "
              f"({min(v)*100:.1f}-{max(v)*100:.1f})  "
              f"{(statistics.mean(v)-base)*100:>+6.1f}")
    print("=" * 58)
    d = [a - b for a, b in zip(res["64x256"], res["32x128"])]
    if len(d) > 1:
        sd = statistics.stdev(d)
        print(f"\npaired difference {statistics.mean(d)*100:+.1f} "
              f"(sd {sd*100:.1f}) across {len(d)} splits")
        if statistics.mean(d) <= sd:
            print("Does not clear its own spread - keep 32x128 and the speed.")
        else:
            print("Consistent. The larger input earns its four-times cost.")


def train_final(args) -> None:
    truth, by_track = load_data()
    dev = "cuda" if torch.cuda.is_available() else "cpu"
    h, w = args.height, args.width
    test_k, val_k, train_k = split(by_track, 1000)
    mk = lambda ks: [{"files": [c["file"] for c in by_track[k]],
                      "text": truth[k]} for k in ks]
    val_items, test_items = mk(val_k), mk(test_k)
    tr = [it for k in train_k for it in by_track[k]]
    print(f"train {len(train_k)} / val {len(val_k)} / test {len(test_k)} "
          f"vehicles   {len(tr)} crops")
    print(f"input {h}x{w}   members {args.members}\n", flush=True)

    members = []
    for j in range(args.members):
        m, loaded, total = train_one(tr, val_items, h, w, dev, args, 200 + j)
        members.append(m)
        acc, _ = score([m], test_items, h, w, dev)
        print(f"  member {j+1}  test {acc*100:.1f}%  "
              f"(pretrained {loaded}/{total})", flush=True)

    acc, cer = score(members, test_items, h, w, dev)
    n_params = sum(p.numel() for p in members[0].parameters())

    # Latency on the same crops the pipeline will see.
    files = [c["file"] for k in test_k[:40] for c in by_track[k]][:200]
    for _ in range(2):
        read_vehicle(members, files[:20], h, w, dev)
    if dev == "cuda":
        torch.cuda.synchronize()
    t0 = time.perf_counter()
    read_vehicle(members, files, h, w, dev)
    if dev == "cuda":
        torch.cuda.synchronize()
    ms = (time.perf_counter() - t0) / max(len(files), 1) * 1000

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    for j, m in enumerate(members):
        torch.save({"model": m.state_dict(), "chars": CHARS, "img_h": h,
                    "img_w": w, "member": j}, OUT_DIR / f"final_m{j}.pt")

    print("\n" + "=" * 58)
    print("FINAL MODEL")
    print("=" * 58)
    print(f"exact match (voted, ensemble) : {acc*100:.1f}%")
    print(f"CER                           : {cer*100:.1f}%")
    print(f"character accuracy            : {(1-cer)*100:.1f}%")
    print(f"parameters per member         : {n_params/1e6:.2f} M")
    print(f"latency ({args.members} members)          : {ms:.2f} ms/plate "
          f"({1000/max(ms,1e-6):.0f} plates/sec)")
    print(f"saved                         : {OUT_DIR}/final_m*.pt")
    print("=" * 58)
    print("\nTest vehicles were never trained on and never used to select a")
    print("checkpoint. One split only - treat the figure as +/- the spread")
    print("the comparison runs showed, not as exact.")


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--compare", action="store_true")
    ap.add_argument("--train-final", action="store_true")
    ap.add_argument("--repeats", type=int, default=3)
    ap.add_argument("--seed-offset", type=int, default=0,
                    help="Start from a later split seed, to add fresh splits "
                         "to a comparison instead of repeating ones already "
                         "run.")
    ap.add_argument("--members", type=int, default=3)
    ap.add_argument("--height", type=int, default=64)
    ap.add_argument("--width", type=int, default=256)
    ap.add_argument("--epochs", type=int, default=80)
    ap.add_argument("--batch", type=int, default=32)
    ap.add_argument("--lr", type=float, default=2e-4)
    args = ap.parse_args()
    if args.compare:
        compare(args)
    elif args.train_final:
        train_final(args)
    else:
        ap.print_help()


if __name__ == "__main__":
    main()
