"""backend/scripts/train_plate_recognizer.py — CRNN+CTC plate recogniser.

WHY THIS IS THE HIGHEST-VALUE COMPONENT LEFT
  The pipeline already locates plates and decodes them into valid all-India
  grammar. What it cannot do is READ them correctly: measured exact-match is
  ~30%. A real example from CAM_08:

      actual   DL8CAQ7196
      read     DL40T7190     <- 8->4, C->0, A->T, Q dropped

  Those are not pixel failures. That plate is legible in the crop. They are
  EasyOCR failures - a general scene-text model being asked to read plate
  glyphs it was never specialised for. A plate located and misread is worth
  nothing operationally, so correctness outranks coverage.

WHY CRNN+CTC RATHER THAN A HEAVIER MODEL
  A plate is a single line of characters on a fixed-aspect strip. CTC handles
  the variable-length sequence without needing per-character boxes, trains
  quickly on one GPU, and exports cleanly to ONNX for deployment. Transformer
  recognisers (PARSeq, TrOCR) score higher on general scene text but need far
  more data and compute for a gain that does not show up on a constrained
  36-symbol alphabet with known grammar. The grammar decoder downstream
  already supplies the structural knowledge a transformer would have to learn.

TRAINING DATA
  98,586 synthetic plates, degraded to match the measured statistics of this
  footage - all 37 state codes, single and two-row layouts, all four colour
  schemes, and the blur/noise/JPEG profile of the real clips. Synthetic
  teaches glyph shapes across a distribution no harvest could cover; the
  corpus contains six non-Gujarat plates in total.

  Real crops fine-tune afterwards. That second stage is what adapts the model
  to this fleet's specific degradation, and skipping it is what sank the
  synthetic-composite detector earlier.

USAGE
  python -m backend.scripts.train_plate_recognizer --epochs 25
  python -m backend.scripts.train_plate_recognizer --epochs 25 --batch 256
"""
from __future__ import annotations

import argparse
import json
import random
import time
from pathlib import Path

import cv2
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset

SYNTH = Path("data/synth_plates")
OUT = Path("models/plate_recognizer")
CHARS = "0123456789ABCDEFGHIJKLMNOPQRSTUVWXYZ"
BLANK = 0                                   # CTC blank occupies index 0
STOI = {c: i + 1 for i, c in enumerate(CHARS)}
ITOS = {i + 1: c for i, c in enumerate(CHARS)}
IMG_H, IMG_W = 32, 128


class PlateDS(Dataset):
    """Plate crops resized to a fixed strip, greyscale, normalised."""

    def __init__(self, rows: list[dict], root: Path, train: bool) -> None:
        self.rows = rows
        self.root = root
        self.train = train

    def __len__(self) -> int:
        return len(self.rows)

    def __getitem__(self, i: int):
        r = self.rows[i]
        img = cv2.imread(str(self.root / r["file"]), cv2.IMREAD_GRAYSCALE)
        if img is None:
            img = np.zeros((IMG_H, IMG_W), np.uint8)
        if self.train:
            # Light train-time jitter only. The heavy degradation already
            # happened at generation time and matches the real distribution;
            # piling more on would push the data off it.
            if random.random() < 0.3:
                a = random.uniform(0.8, 1.2)
                b = random.uniform(-18, 18)
                img = np.clip(img.astype(np.float32) * a + b, 0, 255).astype(np.uint8)
            if random.random() < 0.2:
                img = cv2.GaussianBlur(img, (3, 3), random.uniform(0.3, 0.8))
        img = cv2.resize(img, (IMG_W, IMG_H), interpolation=cv2.INTER_AREA)
        x = torch.from_numpy(img).float().div(127.5).sub(1.0).unsqueeze(0)
        label = torch.tensor([STOI[c] for c in r["text"] if c in STOI],
                             dtype=torch.long)
        return x, label, len(label)


def collate(batch):
    xs, labels, lens = zip(*batch)
    return (torch.stack(xs), torch.cat(labels),
            torch.tensor(lens, dtype=torch.long))


class CRNN(nn.Module):
    """CNN feature extractor -> BiLSTM sequence model -> per-timestep logits.

    Height is collapsed to 1 while width is preserved as the time axis, so a
    128px-wide strip yields 32 timesteps - comfortably more than the 11
    characters an Indian plate can carry, which CTC requires.
    """

    def __init__(self, n_classes: int) -> None:
        super().__init__()
        self.cnn = nn.Sequential(
            nn.Conv2d(1, 64, 3, 1, 1), nn.BatchNorm2d(64), nn.ReLU(True),
            nn.MaxPool2d(2, 2),                          # 16 x 64
            nn.Conv2d(64, 128, 3, 1, 1), nn.BatchNorm2d(128), nn.ReLU(True),
            nn.MaxPool2d(2, 2),                          # 8 x 32
            nn.Conv2d(128, 256, 3, 1, 1), nn.BatchNorm2d(256), nn.ReLU(True),
            nn.Conv2d(256, 256, 3, 1, 1), nn.BatchNorm2d(256), nn.ReLU(True),
            nn.MaxPool2d((2, 1), (2, 1)),                # 4 x 32  (keep width)
            nn.Conv2d(256, 512, 3, 1, 1), nn.BatchNorm2d(512), nn.ReLU(True),
            nn.MaxPool2d((2, 1), (2, 1)),                # 2 x 32
            nn.Conv2d(512, 512, (2, 1), 1, 0), nn.BatchNorm2d(512), nn.ReLU(True),
        )                                                # 1 x 32
        self.rnn = nn.LSTM(512, 256, num_layers=2, bidirectional=True,
                           batch_first=True, dropout=0.15)
        self.fc = nn.Linear(512, n_classes)

    def forward(self, x):
        f = self.cnn(x).squeeze(2).permute(0, 2, 1)      # B, T, C
        r, _ = self.rnn(f)
        return self.fc(r)


def ctc_decode(logits: torch.Tensor) -> list[str]:
    """Greedy CTC: argmax, collapse repeats, drop blanks."""
    out = []
    for seq in logits.argmax(dim=2).cpu().numpy():
        s, prev = [], -1
        for k in seq:
            if k != prev and k != BLANK:
                s.append(ITOS.get(int(k), ""))
            prev = k
        out.append("".join(s))
    return out


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--epochs", type=int, default=25)
    ap.add_argument("--batch", type=int, default=192)
    ap.add_argument("--lr", type=float, default=3e-4)
    ap.add_argument("--val-frac", type=float, default=0.03)
    ap.add_argument("--workers", type=int, default=4)
    args = ap.parse_args()

    rows = [json.loads(l) for l in (SYNTH / "labels.jsonl").open(encoding="utf-8")]
    rows = [r for r in rows if all(c in STOI for c in r["text"])]
    random.seed(1337)
    random.shuffle(rows)
    n_val = max(500, int(len(rows) * args.val_frac))
    val_rows, train_rows = rows[:n_val], rows[n_val:]

    dev = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"device       : {dev}")
    print(f"train / val  : {len(train_rows)} / {len(val_rows)}")
    print(f"alphabet     : {len(CHARS)} symbols + blank\n", flush=True)

    root = SYNTH / "images"
    tl = DataLoader(PlateDS(train_rows, root, True), batch_size=args.batch,
                    shuffle=True, num_workers=args.workers, collate_fn=collate,
                    pin_memory=True, drop_last=True, persistent_workers=True)
    vl = DataLoader(PlateDS(val_rows, root, False), batch_size=args.batch,
                    shuffle=False, num_workers=2, collate_fn=collate,
                    persistent_workers=True)

    model = CRNN(len(CHARS) + 1).to(dev)
    crit = nn.CTCLoss(blank=BLANK, zero_infinity=True)
    opt = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=1e-4)
    sched = torch.optim.lr_scheduler.OneCycleLR(
        opt, max_lr=args.lr * 3, total_steps=args.epochs * len(tl), pct_start=0.15)
    scaler = torch.amp.GradScaler(dev, enabled=(dev == "cuda"))

    OUT.mkdir(parents=True, exist_ok=True)
    best = 0.0
    for ep in range(1, args.epochs + 1):
        model.train()
        tot, nb, t0 = 0.0, 0, time.time()
        for x, y, ylen in tl:
            x, y, ylen = x.to(dev, non_blocking=True), y.to(dev), ylen.to(dev)
            opt.zero_grad(set_to_none=True)
            with torch.amp.autocast(dev, enabled=(dev == "cuda")):
                logits = model(x)
                lp = logits.log_softmax(2).permute(1, 0, 2)
                xlen = torch.full((x.size(0),), logits.size(1),
                                  dtype=torch.long, device=dev)
                loss = crit(lp.float(), y, xlen, ylen)
            scaler.scale(loss).backward()
            scaler.unscale_(opt)
            nn.utils.clip_grad_norm_(model.parameters(), 5.0)
            scaler.step(opt)
            scaler.update()
            sched.step()
            tot += loss.item()
            nb += 1

        model.eval()
        exact = seen = 0
        cer_num = cer_den = 0
        with torch.no_grad():
            for x, y, ylen in vl:
                # `off` MUST reset per batch: collate concatenates that
                # batch's labels into a fresh tensor, so an offset carried
                # over from the previous batch indexes past the end and every
                # ground-truth string comes back empty. That produced a CER of
                # 1440% - impossible by construction, and the signal that the
                # metric rather than the model was broken.
                off = 0
                pred = ctc_decode(model(x.to(dev)))
                for p, L in zip(pred, ylen.tolist()):
                    gt = "".join(ITOS[int(k)] for k in y[off:off + L])
                    off += L
                    seen += 1
                    exact += (p == gt)
                    # Levenshtein for CER
                    dp = list(range(len(gt) + 1))
                    for i, pc in enumerate(p, 1):
                        prev, dp[0] = dp[0], i
                        for j, gc in enumerate(gt, 1):
                            cur = dp[j]
                            dp[j] = min(dp[j] + 1, dp[j - 1] + 1,
                                        prev + (pc != gc))
                            prev = cur
                    cer_num += dp[len(gt)]
                    cer_den += len(gt)
        acc = exact / max(seen, 1)
        cer = cer_num / max(cer_den, 1)
        print(f"ep {ep:>2}/{args.epochs}  loss {tot/max(nb,1):.4f}  "
              f"exact {acc*100:>5.1f}%  CER {cer*100:>5.2f}%  "
              f"({time.time()-t0:.0f}s)", flush=True)
        if acc > best:
            best = acc
            torch.save({"model": model.state_dict(), "chars": CHARS,
                        "img_h": IMG_H, "img_w": IMG_W, "exact": acc,
                        "cer": cer}, OUT / "best.pt")

    print(f"\nbest exact-match on synthetic val: {best*100:.1f}%")
    print(f"saved -> {OUT/'best.pt'}")
    print("\nThis is accuracy on SYNTHETIC data. It is an upper bound, not a")
    print("result: the model has not yet seen a real plate. The fine-tune on")
    print("real crops, and measurement against hand-labelled ground truth,")
    print("are what turn this into a number worth quoting.")


if __name__ == "__main__":
    main()
