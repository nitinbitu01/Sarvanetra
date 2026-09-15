"""backend/scripts/plate_trocr_finetune.py — is a recogniser pretrained on real
text better than a small CRNN pretrained on synthetic plates?

THE ARGUMENT FOR TRYING IT
  The current CRNN was pretrained on plates rendered by a script in this repo -
  clean glyphs, synthetic blur, invented degradation. That pretraining was
  measured at 97.2% on its own synthetic validation set and produced gibberish
  on real crops, which is the classic sign that it taught the model about the
  renderer rather than about plates.

  TrOCR was pretrained on millions of real and photorealistic text images, so it
  arrives already knowing what a degraded glyph looks like. With only 417
  labelled plates the quality of the starting point dominates - there is not
  enough real data to teach a model glyph appearance from scratch, so it has to
  come from pretraining.

THE CONSTRAINT THAT DECIDES WHETHER IT CAN BE ADOPTED
  The pipeline must not get slower. The CRNN is 5.84M parameters and reads a
  plate in 5.51 ms alone or 0.76 ms batched. TrOCR-small is roughly ten times
  the parameters and, being autoregressive, spends a forward pass per character
  instead of one for the whole string.

  So accuracy alone cannot justify a switch. This script reports latency beside
  accuracy, on the same crops, and if TrOCR wins on reading but loses on speed
  the honest conclusion is not "adopt it" but "distil it" - use it to teach the
  small model, and keep the small model in the pipeline.

COMPARABILITY IS THE POINT
  The split protocol is copied from plate_eval_clean deliberately: same seeds,
  same by-vehicle grouping, same three-way train/val/test, same multi-frame
  voting. Only the model changes, so the difference is attributable to it.
  Evaluating a new model under a new protocol would produce a number that
  cannot be compared with the 46.0% the CRNN scored.

USAGE
  python -m backend.scripts.plate_trocr_finetune --repeats 2 --epochs 8
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
from torch.utils.data import DataLoader, Dataset

from backend.scripts.indian_plate_grammar import decode_plate
from backend.scripts.plate_eval_clean import _vote, lev
from backend.scripts.train_plate_recognizer import STOI

REAL = Path("data/plate_real")
MODEL_ID = "microsoft/trocr-small-printed"


class PlateDS(Dataset):
    """Crops as RGB, upscaled to what the vision encoder expects.

    Plates are grayscale and roughly 4:1, while the encoder wants a square RGB
    image. Stretching a 60x15 crop to 384x384 is a large resample, but it is
    the same resample the pretrained encoder saw during its own training on
    text lines, so it is the transform the weights expect rather than a
    distortion introduced here.
    """

    def __init__(self, items, processor, train: bool):
        self.items = items
        self.proc = processor
        self.train = train

    def __len__(self):
        return len(self.items)

    def __getitem__(self, i):
        it = self.items[i]
        g = cv2.imread(str(REAL / "images" / it["file"]), cv2.IMREAD_GRAYSCALE)
        if g is None:
            g = np.zeros((16, 64), np.uint8)
        if self.train:
            # Same mild jitter the CRNN gets, for the same reason: with a few
            # thousand crops of 417 plates, heavy augmentation distorts more
            # than it regularises.
            if random.random() < 0.4:
                a, b = random.uniform(0.85, 1.15), random.uniform(-15, 15)
                g = np.clip(g.astype(np.float32) * a + b, 0, 255).astype(np.uint8)
            if random.random() < 0.25:
                g = cv2.GaussianBlur(g, (3, 3), random.uniform(0.3, 0.7))
        rgb = cv2.cvtColor(g, cv2.COLOR_GRAY2RGB)
        px = self.proc(images=rgb, return_tensors="pt").pixel_values[0]
        ids = self.proc.tokenizer(it["text"], return_tensors="pt",
                                  padding="max_length", max_length=16,
                                  truncation=True).input_ids[0]
        # -100 masks padding out of the loss; without it the model is rewarded
        # for predicting padding, which it will happily learn to do.
        ids = torch.where(ids == self.proc.tokenizer.pad_token_id,
                          torch.tensor(-100), ids)
        return px, ids


def read_batch(model, proc, files, dev, max_len=14):
    imgs = []
    for f in files:
        g = cv2.imread(str(REAL / "images" / f), cv2.IMREAD_GRAYSCALE)
        if g is None:
            continue
        imgs.append(cv2.cvtColor(g, cv2.COLOR_GRAY2RGB))
    if not imgs:
        return []
    px = proc(images=imgs, return_tensors="pt").pixel_values.to(dev)
    with torch.no_grad():
        out = model.generate(px, max_new_tokens=max_len, num_beams=1)
    txt = proc.batch_decode(out, skip_special_tokens=True)
    return ["".join(c for c in t.upper() if c in STOI) for t in txt]


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--model", default=MODEL_ID)
    ap.add_argument("--repeats", type=int, default=2)
    ap.add_argument("--epochs", type=int, default=8)
    ap.add_argument("--batch", type=int, default=8)
    ap.add_argument("--lr", type=float, default=5e-5)
    args = ap.parse_args()

    from transformers import TrOCRProcessor, VisionEncoderDecoderModel

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
    proc = TrOCRProcessor.from_pretrained(args.model)
    print(f"labelled vehicles : {len(truth)}")
    print(f"crops             : {sum(len(v) for v in by_track.values())}")
    print(f"model             : {args.model}\n", flush=True)

    accs_1f, accs_voted, lat = [], [], []
    for i in range(args.repeats):
        # Identical seed and proportions to plate_eval_clean, so the resulting
        # numbers sit next to that script's without adjustment.
        keys = sorted(by_track)
        rng = random.Random(1000 + i)
        rng.shuffle(keys)
        n = len(keys)
        n_test = max(20, int(n * 0.20))
        n_val = max(20, int(n * 0.20))
        test_k, val_k = keys[:n_test], keys[n_test:n_test + n_val]
        train_k = keys[n_test + n_val:]
        train_items = [it for k in train_k for it in by_track[k]]

        model = VisionEncoderDecoderModel.from_pretrained(args.model).to(dev)
        model.config.decoder_start_token_id = proc.tokenizer.cls_token_id \
            or proc.tokenizer.bos_token_id
        model.config.pad_token_id = proc.tokenizer.pad_token_id
        n_params = sum(p.numel() for p in model.parameters())
        if i == 0:
            print(f"parameters        : {n_params/1e6:.1f} M "
                  f"(CRNN is 5.84 M)\n", flush=True)

        dl = DataLoader(PlateDS(train_items, proc, True), batch_size=args.batch,
                        shuffle=True, num_workers=0)
        opt = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=0.01)

        best_val, best_state = -1.0, None
        for ep in range(1, args.epochs + 1):
            model.train()
            tot = nb = 0
            for px, ids in dl:
                px, ids = px.to(dev), ids.to(dev)
                opt.zero_grad(set_to_none=True)
                loss = model(pixel_values=px, labels=ids).loss
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                opt.step()
                tot += loss.item()
                nb += 1
            model.eval()
            ok = 0
            for k in val_k:
                pred = read_batch(model, proc, [by_track[k][0]["file"]], dev)
                if pred:
                    d = decode_plate(pred[0])
                    ok += (d["plate"] or pred[0]) == truth[k]
            v = ok / max(len(val_k), 1)
            print(f"  split {i+1} ep {ep:>2}  loss {tot/max(nb,1):.4f}  "
                  f"val {v*100:.1f}%", flush=True)
            if v > best_val:
                best_val = v
                best_state = {kk: t.detach().cpu().clone()
                              for kk, t in model.state_dict().items()}

        model.load_state_dict(best_state)
        model.eval()

        ex1 = exv = num = den = 0
        t0 = time.perf_counter()
        n_reads = 0
        for k in test_k:
            files = [c["file"] for c in by_track[k]]
            reads = read_batch(model, proc, files, dev)
            n_reads += len(reads)
            if not reads:
                continue
            one = decode_plate(reads[0])
            one = one["plate"] or reads[0]
            ex1 += one == truth[k]
            voted = _vote(reads)
            d = decode_plate(voted)
            fin = d["plate"] or voted
            exv += fin == truth[k]
            num += lev(fin, truth[k])
            den += len(truth[k])
        if dev == "cuda":
            torch.cuda.synchronize()
        ms = (time.perf_counter() - t0) / max(n_reads, 1) * 1000

        a1 = ex1 / max(len(test_k), 1)
        av = exv / max(len(test_k), 1)
        accs_1f.append(a1)
        accs_voted.append(av)
        lat.append(ms)
        print(f"  split {i+1}  TEST 1-frame {a1*100:.1f}%  voted {av*100:.1f}%"
              f"  CER {num/max(den,1)*100:.1f}%  {ms:.1f} ms/plate\n", flush=True)
        del model, best_state
        torch.cuda.empty_cache()

    print("=" * 66)
    print("TrOCR vs CRNN  (same splits, same protocol, same test plates)")
    print("=" * 66)
    print(f"{'':<16} {'TrOCR':>12} {'CRNN':>12}")
    print("-" * 66)
    print(f"{'1-frame':<16} {statistics.mean(accs_1f)*100:>11.1f}% "
          f"{38.3:>11.1f}%")
    print(f"{'voted':<16} {statistics.mean(accs_voted)*100:>11.1f}% "
          f"{46.0:>11.1f}%")
    print(f"{'ms per plate':<16} {statistics.mean(lat):>12.1f} {5.51:>12.2f}")
    print("=" * 66)
    print("\nCRNN figures are from plate_eval_clean over 5 splits; TrOCR here")
    print(f"is over {args.repeats}. Treat a gap under ~10 points as unresolved -")
    print("that is the split noise this much labelled data supports.")


if __name__ == "__main__":
    main()
