"""backend/scripts/finetune_plate_recognizer.py — PHASE A pilot: does fine-tuning
on real, human-verified labels beat EasyOCR?

THE MEASURED BASELINE (65 hand-labelled plates, real ground truth)
    EasyOCR exact-match   20.0%
    within one character  64.6%
    CER                   13.4%
    top errors            2->3 x13, O->B x6, 2->7 x4, 4->1 x4

  Two thirds of plates are wrong by a SINGLE character. The plates are legible
  and the errors are systematic glyph confusions - the failure mode supervised
  training removes. That is why this pilot is worth running rather than
  assuming.

WHY THIS SHOULD SUCCEED WHERE FOUR ATTEMPTS FAILED
    synthetic recogniser   97.2% on synthetic, gibberish on real (domain gap)
    synthetic composites   detector learned paste artifacts
    OCR bootstrap          detector learned the burned-in caption
    pseudo-labels          ~15% correct, because voting only fixes RANDOM
                           errors and EasyOCR's are systematic
  Every one lacked real ground truth. This has it, for the first time.

SPLIT BY TRACK, NOT BY CROP - the thing that would silently invalidate this
  Each labelled vehicle contributes ~7 crops of the SAME plate. Splitting
  crop-wise would put frames of one plate in both train and test, and the
  model would be scored on plates it had memorised. Grouping by track keeps
  the held-out plates genuinely unseen.

USAGE
  python -m backend.scripts.finetune_plate_recognizer
  python -m backend.scripts.finetune_plate_recognizer --epochs 60 --holdout 13
"""
from __future__ import annotations

import argparse
import json
import random
from collections import defaultdict
from pathlib import Path

import cv2
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset

from backend.scripts.indian_plate_grammar import decode_plate
from backend.scripts.train_plate_recognizer import (BLANK, CHARS, CRNN, IMG_H,
                                                    IMG_W, ITOS, STOI,
                                                    ctc_decode)

def _enhance(g, mode: str):
    """Optional preprocessing, applied identically to train AND test.

    Inference-only enhancement measured WORSE than none (unsharp 49.1% vs
    52.7% baseline; bilateral 20.0%), but that test was unfair: the model had
    only ever seen untouched crops, so enhancement handed it a distribution it
    was never trained on. The honest version applies the same transform on
    both sides, which is what this flag is for.
    """
    if mode == "none":
        return g
    import cv2 as _cv
    if mode == "unsharp":
        return _cv.addWeighted(g, 1.7, _cv.GaussianBlur(g, (0, 0), 1.6), -0.7, 0)
    if mode == "clahe":
        return _cv.createCLAHE(clipLimit=2.5, tileGridSize=(8, 4)).apply(g)
    if mode == "upscale":
        return _cv.resize(g, None, fx=2, fy=2, interpolation=_cv.INTER_LANCZOS4)
    if mode == "clahe+unsharp":
        g = _cv.createCLAHE(clipLimit=2.5, tileGridSize=(8, 4)).apply(g)
        return _cv.addWeighted(g, 1.7, _cv.GaussianBlur(g, (0, 0), 1.6), -0.7, 0)
    return g


ENH = "none"
REAL = Path("data/plate_real")
VERIFIED = REAL / "verified_all.jsonl"
CKPT_IN = Path("models/plate_recognizer/best.pt")
CKPT_OUT = Path("models/plate_recognizer/finetuned.pt")


def lev(a: str, b: str) -> int:
    dp = list(range(len(b) + 1))
    for i, ca in enumerate(a, 1):
        prev, dp[0] = dp[0], i
        for j, cb in enumerate(b, 1):
            cur = dp[j]
            dp[j] = min(dp[j] + 1, dp[j - 1] + 1, prev + (ca != cb))
            prev = cur
    return dp[len(b)]


class RealDS(Dataset):
    def __init__(self, items: list[dict], train: bool) -> None:
        self.items = items
        self.train = train

    def __len__(self) -> int:
        return len(self.items)

    def __getitem__(self, i: int):
        it = self.items[i]
        img = cv2.imread(str(REAL / "images" / it["file"]), cv2.IMREAD_GRAYSCALE)
        if img is None:
            img = np.zeros((IMG_H, IMG_W), np.uint8)
        if self.train:
            # Mild jitter only - with ~450 crops of 52 plates, aggressive
            # augmentation distorts more than it regularises.
            if random.random() < 0.4:
                a, b = random.uniform(0.85, 1.15), random.uniform(-15, 15)
                img = np.clip(img.astype(np.float32) * a + b, 0, 255).astype(np.uint8)
            if random.random() < 0.25:
                img = cv2.GaussianBlur(img, (3, 3), random.uniform(0.3, 0.7))
            if random.random() < 0.3:
                h, w = img.shape
                dx, dy = random.randint(-2, 2), random.randint(-1, 1)
                M = np.float32([[1, 0, dx], [0, 1, dy]])
                img = cv2.warpAffine(img, M, (w, h), borderMode=cv2.BORDER_REPLICATE)
        img = _enhance(img, ENH)
        img = cv2.resize(img, (IMG_W, IMG_H), interpolation=cv2.INTER_AREA)
        x = torch.from_numpy(img).float().div(127.5).sub(1.0).unsqueeze(0)
        lab = torch.tensor([STOI[c] for c in it["text"] if c in STOI],
                           dtype=torch.long)
        return x, lab, len(lab)


def collate(b):
    xs, ls, ns = zip(*b)
    return torch.stack(xs), torch.cat(ls), torch.tensor(ns, dtype=torch.long)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--epochs", type=int, default=60)
    ap.add_argument("--batch", type=int, default=32)
    ap.add_argument("--lr", type=float, default=2e-4,
                    help="Low: this ADJUSTS the synthetic-pretrained model. "
                         "Too high and it forgets the rare states it only ever "
                         "saw synthetically.")
    ap.add_argument("--enhance", default="none",
                    choices=["none","unsharp","clahe","upscale","clahe+unsharp"],
                    help="Preprocessing applied to BOTH train and test.")
    ap.add_argument("--holdout", type=int, default=13,
                    help="Vehicles (not crops) held out entirely.")
    args = ap.parse_args()
    global ENH
    ENH = args.enhance

    verified = [json.loads(l) for l in VERIFIED.open(encoding="utf-8")]
    verified = [v for v in verified
                if v["text"] and all(c in STOI for c in v["text"])]
    truth = {(v["camera"], v["track"]): v["text"] for v in verified}
    ocr_of = {(v["camera"], v["track"]): v["ocr"] for v in verified}
    print(f"hand-verified vehicles : {len(truth)}")

    # Expand each labelled vehicle to ALL of its crops.
    all_rows = [json.loads(l) for l in (REAL / "labels.jsonl").open(encoding="utf-8")]
    by_track: dict[tuple, list[dict]] = defaultdict(list)
    for r in all_rows:
        k = (r["camera"], r["track"])
        if k in truth:
            by_track[k].append({"file": r["file"], "text": truth[k]})
    n_crops = sum(len(v) for v in by_track.values())
    print(f"crops available        : {n_crops} "
          f"({n_crops/max(len(by_track),1):.1f} per vehicle)")

    # Split BY TRACK so a held-out plate never appears in training.
    keys = sorted(by_track)
    random.seed(1337)
    random.shuffle(keys)
    test_keys, train_keys = keys[:args.holdout], keys[args.holdout:]
    train_items = [it for k in train_keys for it in by_track[k]]
    test_items = [{"key": k, "file": by_track[k][0]["file"], "text": truth[k]}
                  for k in test_keys]
    print(f"train / test vehicles  : {len(train_keys)} / {len(test_keys)}")
    print(f"train crops            : {len(train_items)}\n", flush=True)

    dev = "cuda" if torch.cuda.is_available() else "cpu"
    ck = torch.load(CKPT_IN, map_location=dev, weights_only=False)
    model = CRNN(len(CHARS) + 1).to(dev)
    model.load_state_dict(ck["model"])
    print(f"loaded synthetic-pretrained model "
          f"(synthetic val exact {ck['exact']*100:.1f}%)")

    tl = DataLoader(RealDS(train_items, True), batch_size=args.batch,
                    shuffle=True, collate_fn=collate, num_workers=0,
                    drop_last=len(train_items) > args.batch)
    crit = nn.CTCLoss(blank=BLANK, zero_infinity=True)
    opt = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=1e-4)

    def evaluate() -> tuple[float, float, list]:
        model.eval()
        rec = []
        ex = num = den = 0
        with torch.no_grad():
            for it in test_items:
                img = cv2.imread(str(REAL / "images" / it["file"]),
                                 cv2.IMREAD_GRAYSCALE)
                if img is None:
                    continue
                img = _enhance(img, ENH)
                img = cv2.resize(img, (IMG_W, IMG_H), interpolation=cv2.INTER_AREA)
                x = torch.from_numpy(img).float().div(127.5).sub(1.0)
                pred = ctc_decode(model(x[None, None].to(dev)))[0]
                d = decode_plate(pred)
                final = d["plate"] or pred
                gt = it["text"]
                ex += (final == gt)
                num += lev(final, gt)
                den += len(gt)
                rec.append((gt, ocr_of.get(it["key"], ""), pred, final))
        return ex / max(len(test_items), 1), num / max(den, 1), rec

    acc0, cer0, _ = evaluate()
    print(f"BEFORE fine-tune : exact {acc0*100:.1f}%  CER {cer0*100:.1f}%\n",
          flush=True)

    best = acc0
    for ep in range(1, args.epochs + 1):
        model.train()
        tot = nb = 0
        for x, y, ylen in tl:
            x, y, ylen = x.to(dev), y.to(dev), ylen.to(dev)
            opt.zero_grad(set_to_none=True)
            logits = model(x)
            lp = logits.log_softmax(2).permute(1, 0, 2)
            xlen = torch.full((x.size(0),), logits.size(1), dtype=torch.long,
                              device=dev)
            loss = crit(lp, y, xlen, ylen)
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), 5.0)
            opt.step()
            tot += loss.item()
            nb += 1
        if ep % 5 == 0 or ep == args.epochs:
            acc, cer, rec = evaluate()
            flag = ""
            if acc > best:
                best = acc
                torch.save({"model": model.state_dict(), "chars": CHARS,
                            "img_h": IMG_H, "img_w": IMG_W, "exact": acc,
                            "cer": cer}, CKPT_OUT)
                flag = "  <- saved"
            print(f"ep {ep:>3}  loss {tot/max(nb,1):.4f}  "
                  f"exact {acc*100:>5.1f}%  CER {cer*100:>5.1f}%{flag}",
                  flush=True)

    acc, cer, rec = evaluate()
    print("\n" + "=" * 68)
    print("PHASE A RESULT  (held-out vehicles, never seen in training)")
    print("=" * 68)
    print(f"{'ground truth':<14} {'EasyOCR':<14} {'CRNN raw':<14} {'CRNN final':<14}")
    print("-" * 68)
    for gt, ocr, raw, fin in rec:
        mark = "OK" if fin == gt else ""
        print(f"{gt:<14} {ocr:<14} {raw[:13]:<14} {fin:<14} {mark}")
    ocr_ex = sum(g == o for g, o, _, _ in rec) / max(len(rec), 1)
    print("-" * 68)
    print(f"EasyOCR    exact {ocr_ex*100:>5.1f}%")
    print(f"fine-tuned exact {best*100:>5.1f}%   (best checkpoint)")
    print(f"final      exact {acc*100:>5.1f}%   CER {cer*100:.1f}%")
    print("=" * 68)
    print(f"\nheld-out set is {len(test_items)} plates - small, so treat this as")
    print("a DIRECTION, not a precise figure. If it clearly beats EasyOCR,")
    print("labelling several hundred more is worth the time.")


if __name__ == "__main__":
    main()
