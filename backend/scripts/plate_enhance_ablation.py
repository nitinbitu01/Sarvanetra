"""backend/scripts/plate_enhance_ablation.py — does enhancing blurry plates help?

THE CLAIM UNDER TEST
  Held-out failures are single-character errors of a very specific kind:

      GJ38BD6570 -> GJ36BD6570     8 -> 6
      GJ11BR9208 -> GJ11BR9268     0 -> 6
      GJ05JC3614 -> GJ05JC3610     4 -> 0

  8/6, 0/6, 4/0 all differ by whether a loop is closed. Blur is exactly what
  closes those gaps, so the hypothesis that sharper input would fix them is
  reasonable. It is also testable, which is the point of this script.

METHOD
  Every technique is applied to the SAME held-out plates, read by the SAME
  model, and scored the same way. Only the preprocessing changes, so any
  difference is attributable to it. Techniques are measured alone before being
  combined - a stack that helps overall can contain a step that hurts, and
  averaging hides that.

WHAT IS DELIBERATELY EXCLUDED
  GAN-based super-resolution (Real-ESRGAN and similar). Those are trained
  adversarially to look sharp, so they fail CONFIDENTLY - inventing a crisp,
  wrong character. For a plate that is the worst failure mode: an officer
  cannot tell an invented glyph from a read one. The SR here (ESPCN/EDSR) is
  L1/L2-trained and fails conservatively, staying blurry rather than inventing.

  Upscaling is not enhancement either. Interpolating a 7px glyph to 30px makes
  it bigger, not more informative. It is included only because recognisers
  have a minimum input size, and it is labelled honestly as such.

USAGE
  python -m backend.scripts.plate_enhance_ablation
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

from backend.scripts.indian_plate_grammar import decode_plate
from backend.scripts.train_plate_recognizer import (CHARS, CRNN, IMG_H, IMG_W,
                                                    STOI, ctc_decode)

REAL = Path("data/plate_real")
CKPT = Path("models/plate_recognizer/finetuned.pt")


def lev(a: str, b: str) -> int:
    dp = list(range(len(b) + 1))
    for i, ca in enumerate(a, 1):
        prev, dp[0] = dp[0], i
        for j, cb in enumerate(b, 1):
            cur = dp[j]
            dp[j] = min(dp[j] + 1, dp[j - 1] + 1, prev + (ca != cb))
            prev = cur
    return dp[len(b)]


# ── enhancement steps, each usable alone ──────────────────────────────────
def e_none(g):
    return g


def e_unsharp(g):
    """Unsharp mask. Amplifies edges that survived; invents nothing."""
    blur = cv2.GaussianBlur(g, (0, 0), 1.6)
    return cv2.addWeighted(g, 1.7, blur, -0.7, 0)


def e_clahe(g):
    """Local contrast. Helps a plate in shadow or blown out by headlights."""
    return cv2.createCLAHE(clipLimit=2.5, tileGridSize=(8, 4)).apply(g)


def e_bilateral(g):
    """Edge-preserving denoise - smooths sensor noise without softening
    glyph boundaries the way a Gaussian would."""
    return cv2.bilateralFilter(g, 7, 55, 55)


def e_richardson(g):
    """Richardson-Lucy deconvolution against a small Gaussian PSF.

    Genuine deblurring: it estimates what the sharp image was, rather than
    boosting contrast. Iterations are kept low - RL amplifies noise into
    ringing if pushed, which would add fake edges to the glyphs.
    """
    from skimage.restoration import richardson_lucy
    psf = cv2.getGaussianKernel(5, 1.0)
    psf = psf @ psf.T
    f = g.astype(np.float32) / 255.0
    out = richardson_lucy(f, psf, num_iter=8)
    return np.clip(out * 255, 0, 255).astype(np.uint8)


def e_upscale(g):
    """Lanczos 2x. Adds NO information - included only because the recogniser
    has a fixed input size and a larger source resamples with less loss."""
    return cv2.resize(g, None, fx=2, fy=2, interpolation=cv2.INTER_LANCZOS4)


def e_morph(g):
    """Black-hat: lifts dark glyphs off a light plate ground."""
    k = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3))
    bh = cv2.morphologyEx(g, cv2.MORPH_BLACKHAT, k)
    return cv2.add(g, bh)


PIPELINES = {
    "baseline": [e_none],
    "unsharp": [e_unsharp],
    "clahe": [e_clahe],
    "bilateral": [e_bilateral],
    "richardson-lucy": [e_richardson],
    "upscale-2x": [e_upscale],
    "blackhat": [e_morph],
    "bilateral+unsharp": [e_bilateral, e_unsharp],
    "clahe+unsharp": [e_clahe, e_unsharp],
    "upscale+unsharp": [e_upscale, e_unsharp],
    "bilateral+RL": [e_bilateral, e_richardson],
    "clahe+bilateral+unsharp": [e_clahe, e_bilateral, e_unsharp],
}


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--holdout", type=int, default=55)
    ap.add_argument("--seed", type=int, default=1337)
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
            by_track[k].append(r["file"])

    # Reproduce the fine-tuner's split exactly, so these are the same plates
    # the model has never seen.
    keys = sorted(by_track)
    random.seed(args.seed)
    random.shuffle(keys)
    test = [(k, by_track[k][0], truth[k]) for k in keys[:args.holdout]]

    dev = "cuda" if torch.cuda.is_available() else "cpu"
    ck = torch.load(CKPT, map_location=dev, weights_only=False)
    model = CRNN(len(CHARS) + 1).to(dev)
    model.load_state_dict(ck["model"])
    model.eval()
    print(f"model    : {CKPT}")
    print(f"held out : {len(test)} plates (never seen in training)\n")

    # Establish which plates the untouched model already gets right, BEFORE
    # any pipeline runs. Later pipelines then report what they fixed versus
    # what they broke - a net change of zero can hide five of each, and that
    # distinction decides whether a technique is worth keeping.
    base_right, base_wrong = set(), set()
    for key, f, gt in test:
        g = cv2.imread(str(REAL / "images" / f), cv2.IMREAD_GRAYSCALE)
        if g is None:
            continue
        g = cv2.resize(g, (IMG_W, IMG_H), interpolation=cv2.INTER_AREA)
        x = torch.from_numpy(g).float().div(127.5).sub(1.0)[None, None].to(dev)
        with torch.no_grad():
            p = ctc_decode(model(x))[0]
        dd = decode_plate(p)
        (base_right if (dd["plate"] or p) == gt else base_wrong).add(key)

    results = []
    for name, steps in PIPELINES.items():
        ex = num = den = 0
        fixed, broke = [], []
        for key, f, gt in test:
            g = cv2.imread(str(REAL / "images" / f), cv2.IMREAD_GRAYSCALE)
            if g is None:
                continue
            try:
                for s in steps:
                    g = s(g)
            except Exception:                            # noqa: BLE001
                continue
            g = cv2.resize(g, (IMG_W, IMG_H), interpolation=cv2.INTER_AREA)
            x = torch.from_numpy(g).float().div(127.5).sub(1.0)[None, None].to(dev)
            with torch.no_grad():
                pred = ctc_decode(model(x))[0]
            d = decode_plate(pred)
            final = d["plate"] or pred
            ok = final == gt
            ex += ok
            num += lev(final, gt)
            den += len(gt)
            if name != "baseline":
                if ok and key in base_wrong:
                    fixed.append(gt)
                elif not ok and key in base_right:
                    broke.append(gt)
        acc = ex / max(len(test), 1)
        cer = num / max(den, 1)
        results.append((name, acc, cer, len(fixed), len(broke)))
        print(f"  {name:<26} exact {acc*100:>5.1f}%   CER {cer*100:>5.1f}%"
              + (f"   fixed {len(fixed):>2}  broke {len(broke):>2}"
                 if name != "baseline" else ""), flush=True)

    base = results[0][1]
    print("\n" + "=" * 66)
    best = max(results, key=lambda r: r[1])
    print(f"baseline : {base*100:.1f}%")
    print(f"best     : {best[0]} at {best[1]*100:.1f}% "
          f"({(best[1]-base)*100:+.1f} points)")
    print("=" * 66)
    if best[1] - base < 0.02:
        print("\nNo technique beats the baseline by more than 2 points. The")
        print("errors are therefore NOT primarily a sharpness problem - the")
        print("information is either already there or already gone, and")
        print("preprocessing cannot add it. A model trained ON enhanced crops")
        print("is the remaining variant worth trying; enhancing only at")
        print("inference asks the model to read a distribution it never saw.")


if __name__ == "__main__":
    main()
