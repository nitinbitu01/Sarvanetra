"""backend/scripts/plate_recognizer_infer.py — run the trained CRNN on REAL plates.

THE ONLY TEST THAT COUNTS
  The recogniser scores 97.2% exact-match with 0.43% CER on synthetic
  validation. That number is worth nothing on its own - it has never seen a
  real plate. Three plate DETECTORS scored 0.99+ on their own validation
  splits earlier in this work and every one of them was unusable in the field:
  v1 fired on the burned-in station caption, v2 learned to find pasted
  composites rather than plates.

  So this compares the CRNN against EasyOCR on the SAME real crops, where
  EasyOCR's answers are already known and several have been verified by eye
  against the image:

      GJ32K4588   verified correct   (EasyOCR read '6J32K4588')
      GJ03LB0535  actual             (EasyOCR read 'GJ03LO0535' - B->O)
      DL8CAQ7196  actual             (EasyOCR read 'DL40T7190'  - mostly wrong)

  Those last two are the bar. If the CRNN reads them correctly it has earned
  its place; if it does not, synthetic pretraining alone was insufficient and
  the fine-tune on real crops is mandatory rather than optional.

USAGE
  python -m backend.scripts.plate_recognizer_infer
  python -m backend.scripts.plate_recognizer_infer --limit 40
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import cv2
import numpy as np
import torch

from backend.scripts.indian_plate_grammar import apply_state_prior, decode_plate
from backend.scripts.train_plate_recognizer import (BLANK, CRNN, IMG_H, IMG_W,
                                                    ITOS, ctc_decode)

CKPT = Path("models/plate_recognizer/best.pt")
READS = Path("output/plate_multiframe.jsonl")


def load_model(device: str):
    ck = torch.load(CKPT, map_location=device, weights_only=False)
    model = CRNN(len(ck["chars"]) + 1).to(device)
    model.load_state_dict(ck["model"])
    model.eval()
    print(f"recogniser: {CKPT}  (synthetic val: exact {ck['exact']*100:.1f}%, "
          f"CER {ck['cer']*100:.2f}%)")
    return model


def read_strip(model, device: str, strip: np.ndarray) -> str:
    """Read one plate strip. Input is cropped to the plate, not the vehicle."""
    g = cv2.cvtColor(strip, cv2.COLOR_BGR2GRAY) if strip.ndim == 3 else strip
    g = cv2.resize(g, (IMG_W, IMG_H), interpolation=cv2.INTER_AREA)
    x = torch.from_numpy(g).float().div(127.5).sub(1.0)[None, None].to(device)
    with torch.no_grad():
        return ctc_decode(model(x))[0]


def band_of(img):
    h, w = img.shape[:2]
    b = img[int(h * 0.42):int(h * 0.97), int(w * 0.08):int(w * 0.92)]
    if b.size == 0 or b.shape[1] < 20:
        return None
    if b.shape[1] < 360:
        f = 360 / b.shape[1]
        b = cv2.resize(b, None, fx=f, fy=f, interpolation=cv2.INTER_CUBIC)
    return b


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--limit", type=int, default=30)
    args = ap.parse_args()

    dev = "cuda" if torch.cuda.is_available() else "cpu"
    model = load_model(dev)

    rows = [json.loads(l) for l in READS.open(encoding="utf-8")]
    rows.sort(key=lambda r: -r["agreement"])
    rows = rows[:args.limit]
    print(f"\ncomparing on {len(rows)} real crops "
          f"(EasyOCR result already known for each)\n")
    print(f"{'EasyOCR':<14} {'CRNN raw':<14} {'CRNN decoded':<14} {'agree':>6}")
    print("-" * 56)

    agree_n = 0
    decoded_n = 0
    for r in rows:
        img = cv2.imread(r["path"])
        if img is None:
            continue
        b = band_of(img)
        if b is None:
            continue
        raw = read_strip(model, dev, b)
        d = decode_plate(raw)
        plate = d["plate"] or "-"
        if d["plate"]:
            decoded_n += 1
            plate, _st, _n = apply_state_prior(d["plate"], d["state"],
                                               r["agreement"], r["votes"])
        same = "YES" if plate == r["plate"] else ""
        agree_n += (plate == r["plate"])
        print(f"{r['plate']:<14} {raw[:13]:<14} {plate:<14} {same:>6}")

    n = len(rows)
    print("\n" + "=" * 56)
    print(f"CRNN produced a valid plate : {decoded_n}/{n}")
    print(f"agreed with EasyOCR         : {agree_n}/{n}")
    print("=" * 56)
    print("\nAgreement is NOT accuracy - EasyOCR is wrong on several of these")
    print("(GJ03LB0535 was read 'GJ03LO0535', DL8CAQ7196 read 'DL40T7190').")
    print("Disagreement may mean the CRNN is RIGHT. Only the images settle it,")
    print("which is what the hand-labelled ground-truth set is for.")


if __name__ == "__main__":
    main()
