"""backend/scripts/plate_multiframe_fusion.py — combine a vehicle's frames at the
PIXEL level, before recognition, instead of voting on strings after it.

THE DISTINCTION THAT MAKES THIS LEGITIMATE
  Single-image super-resolution invents detail. A GAN trained to make plates
  look sharp will produce a crisp, confident, wrong character, and an officer
  cannot tell an invented glyph from a read one. That failure mode is worse
  than blur, because blur admits it does not know.

  Multi-frame fusion invents nothing. A moving vehicle lands on the sensor grid
  at a different sub-pixel offset in every frame, so each frame is a genuinely
  different SAMPLE of the same plate. Combining them recovers detail that is
  present in the measurements and absent from any one of them. It is the same
  principle as drizzle in astronomy - real information, not plausible fiction.

WHY THIS IS THE MOST PROMISING REMAINING LEVER
  On the 2025 real-world low-resolution benchmark, super-resolving a single
  image took recognition from 1.7% to 31.1%, and using MULTIPLE super-resolved
  images took it to 44.7%. The multi-frame step is the largest single gain in
  that literature, and this project has the raw material already: up to 25
  corpus frames per labelled vehicle, currently used only to vote on decoded
  strings - which is to say, used after the recogniser has already discarded
  the pixels.

REGISTRATION IS THE WHOLE PROBLEM
  Fusing misaligned frames blurs rather than sharpens, so alignment must be
  sub-pixel or the operation is worse than doing nothing. Two choices matter:

  ECC rather than feature matching. On a 60px plate there are no reliable
  keypoints - SIFT and ORB find noise. ECC optimises the correlation between
  whole images directly, which is what works at this scale.

  Frames are aligned to EACH OTHER, not to an absolute geometry. Single-frame
  rectification was measured on this footage and hurt (0 plates fixed, 5
  broken) because a 50px plate's border is not reliable enough to define a
  rectangle. Relative alignment asks a much easier question and does not
  depend on finding the plate's edges at all.

  Every frame that fails to converge is DROPPED rather than included with a
  bad warp, and fusion is by median rather than mean, so one misregistered
  frame that slips through cannot smear the result.

USAGE
  python -m backend.scripts.plate_multiframe_fusion --preview 12
  python -m backend.scripts.plate_multiframe_fusion --evaluate
"""
from __future__ import annotations

import argparse
import json
import random
from collections import defaultdict
from pathlib import Path

import cv2
import numpy as np

REAL = Path("data/plate_real")

CANON_H = 48          # canonical height frames are registered at
UPSCALE = 3           # fusion grid factor


def _prep(g, out_w, out_h):
    """Bring a crop onto the common canonical grid."""
    return cv2.resize(g, (out_w, out_h), interpolation=cv2.INTER_CUBIC)


def _to_gray(img: np.ndarray) -> np.ndarray:
    if img.ndim == 3 and img.shape[2] >= 3:
        return cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    return img


def fuse(crops: list[np.ndarray], upscale: int = UPSCALE,
         motion=cv2.MOTION_AFFINE, iters: int = 80, eps: float = 1e-5,
         max_translation_px: float = 12.0):
    """Sub-pixel register and median-fuse a vehicle's crops.

    Handles both grayscale and BGR crops cleanly.
    Returns (fused image, frames actually used). Falls back to the sharpest
    single crop when fewer than two frames register, so callers always get a
    usable image and never a silently degraded one.
    """
    if not crops:
        return None, 0

    valid_crops = [c for c in crops if c is not None and c.size > 0]
    if not valid_crops:
        return None, 0

    is_bgr = (valid_crops[0].ndim == 3 and valid_crops[0].shape[2] >= 3)

    # Sharpest first: reference frame must be the sharpest crop
    def _sharpness(img):
        g = _to_gray(img)
        return cv2.Laplacian(g, cv2.CV_64F).var()

    order = sorted(valid_crops, key=lambda img: -_sharpness(img))

    ar = np.median([g.shape[1] / max(g.shape[0], 1) for g in order])
    h = CANON_H * upscale
    w = int(round(h * ar))
    if w < 24 or w > 2000:
        return _prep(order[0], max(24, min(w, 2000)), h), 1

    ref = _prep(order[0], w, h)
    ref_gray = _to_gray(ref).astype(np.float32)
    ref_n = cv2.normalize(ref_gray, None, 0, 1, cv2.NORM_MINMAX)

    stack = [ref.astype(np.float32)]
    crit = (cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, iters, eps)

    for g in order[1:]:
        cur = _prep(g, w, h)
        cur_gray = _to_gray(cur).astype(np.float32)
        cur_n = cv2.normalize(cur_gray, None, 0, 1, cv2.NORM_MINMAX)
        warp = np.eye(2, 3, dtype=np.float32)
        try:
            # ECC returns the correlation it achieved; a poor score means the
            # frames did not actually align and including it would smear the
            # fusion, so it is dropped.
            cc, warp = cv2.findTransformECC(ref_n, cur_n, warp, motion, crit,
                                            None, 5)
        except cv2.error:
            continue

        if not np.isfinite(cc) or cc < 0.55:
            continue

        # Translation check: ensure alignment didn't latch onto adjacent vehicle or jitter
        tx, ty = warp[0, 2], warp[1, 2]
        if np.sqrt(tx**2 + ty**2) > max_translation_px:
            continue

        aligned = cv2.warpAffine(cur.astype(np.float32), warp, (w, h),
                                 flags=cv2.INTER_CUBIC + cv2.WARP_INVERSE_MAP,
                                 borderMode=cv2.BORDER_REPLICATE)
        stack.append(aligned)

    if len(stack) < 2:
        return ref.astype(np.uint8), 1

    # Median, not mean: one frame that registered badly enough to pass the
    # correlation gate still cannot pull the result, whereas a mean would let it.
    fused = np.median(np.stack(stack), axis=0)
    return np.clip(fused, 0, 255).astype(np.uint8), len(stack)


def load_groups(min_frames: int = 3):
    from backend.scripts.train_plate_recognizer import STOI
    truth = {}
    for line in (REAL / "verified_all.jsonl").open(encoding="utf-8"):
        v = json.loads(line)
        if v.get("text") and all(c in STOI for c in v["text"]):
            truth[(v["camera"], v["track"])] = v["text"]
    groups = defaultdict(list)
    for line in (REAL / "labels.jsonl").open(encoding="utf-8"):
        r = json.loads(line)
        k = (r["camera"], r["track"])
        if k in truth:
            groups[k].append(r["file"])
    return {k: v for k, v in groups.items() if len(v) >= min_frames}, truth


def _preview(args) -> None:
    groups, truth = load_groups()
    keys = sorted(groups)
    random.Random(args.seed).shuffle(keys)
    keys = keys[:args.preview]
    print(f"vehicles with >=3 frames : {len(groups)}")
    print(f"previewing               : {len(keys)}\n", flush=True)

    TW, TH = 360, 78
    rows = []
    for k in keys:
        crops = []
        for f in groups[k]:
            g = cv2.imread(str(REAL / "images" / f), cv2.IMREAD_GRAYSCALE)
            if g is not None:
                crops.append(g)
        if len(crops) < 2:
            continue
        fused, used = fuse(crops, args.upscale)
        sharp = max(crops, key=lambda g: cv2.Laplacian(g, cv2.CV_64F).var())

        def fit(x):
            t = np.full((TH, TW), 20, np.uint8)
            s = min(TW / x.shape[1], TH / x.shape[0])
            v = cv2.resize(x, (max(1, int(x.shape[1] * s)),
                               max(1, int(x.shape[0] * s))),
                           interpolation=cv2.INTER_CUBIC)
            t[(TH - v.shape[0]) // 2:(TH - v.shape[0]) // 2 + v.shape[0],
              (TW - v.shape[1]) // 2:(TW - v.shape[1]) // 2 + v.shape[1]] = v
            return t
        rows.append((fit(sharp), fit(fused), truth[k], len(crops), used))

    sheet = np.full((len(rows) * (TH + 16), 2 * TW + 6, 3), 18, np.uint8)
    for i, (a, b, txt, n, used) in enumerate(rows):
        y = i * (TH + 16) + 16
        sheet[y:y + TH, 0:TW] = cv2.cvtColor(a, cv2.COLOR_GRAY2BGR)
        sheet[y:y + TH, TW + 6:2 * TW + 6] = cv2.cvtColor(b, cv2.COLOR_GRAY2BGR)
        cv2.putText(sheet, f"{txt}   sharpest single | fused {used}/{n} frames",
                    (2, y - 4), cv2.FONT_HERSHEY_SIMPLEX, 0.42,
                    (0, 220, 255), 1, cv2.LINE_AA)
    out = Path(args.out)
    cv2.imwrite(str(out), sheet, [cv2.IMWRITE_JPEG_QUALITY, 95])
    print(f"{len(rows)} pairs -> {out}")
    print("\nLeft is the sharpest single frame, right is the fusion. If the")
    print("right column is not visibly cleaner, this idea is finished and no")
    print("pipeline should be built on it.")


def _evaluate(args) -> None:
    import torch

    from backend.scripts.indian_plate_grammar import decode_plate
    from backend.scripts.plate_eval_clean import _vote, lev
    from backend.scripts.train_plate_recognizer import (CHARS, CRNN, IMG_H,
                                                        IMG_W, ctc_decode)

    groups, truth = load_groups()
    keys = sorted(groups)
    random.Random(args.seed).shuffle(keys)
    keys = keys[:args.n]

    dev = "cuda" if torch.cuda.is_available() else "cpu"
    ck = torch.load(Path("models/plate_recognizer/finetuned.pt"),
                    map_location=dev, weights_only=False)
    model = CRNN(len(CHARS) + 1).to(dev)
    model.load_state_dict(ck["model"])
    model.eval()

    def read(g):
        g = cv2.resize(g, (IMG_W, IMG_H), interpolation=cv2.INTER_AREA)
        x = torch.from_numpy(g).float().div(127.5).sub(1.0)[None, None].to(dev)
        with torch.no_grad():
            return ctc_decode(model(x))[0]

    res = {"sharpest": [0, 0, 0], "voted": [0, 0, 0], "fused": [0, 0, 0]}
    for k in keys:
        crops = []
        for f in groups[k]:
            g = cv2.imread(str(REAL / "images" / f), cv2.IMREAD_GRAYSCALE)
            if g is not None:
                crops.append(g)
        if len(crops) < 2:
            continue
        gt = truth[k]
        sharp = max(crops, key=lambda g: cv2.Laplacian(g, cv2.CV_64F).var())
        fused, _ = fuse(crops, args.upscale)
        cand = {
            "sharpest": read(sharp),
            "voted": _vote([read(c) for c in crops]),
            "fused": read(fused),
        }
        for name, raw in cand.items():
            d = decode_plate(raw)
            fin = d["plate"] or raw
            res[name][0] += fin == gt
            res[name][1] += lev(fin, gt)
            res[name][2] += len(gt)

    n = len(keys)
    print("\n" + "=" * 58)
    print(f"PIXEL FUSION vs STRING VOTING   ({n} vehicles)")
    print("=" * 58)
    print(f"{'method':<12} {'exact':>8} {'CER':>8}")
    print("-" * 58)
    for name in ("sharpest", "voted", "fused"):
        ex, num, den = res[name]
        print(f"{name:<12} {ex/max(n,1)*100:>7.1f}% "
              f"{num/max(den,1)*100:>7.1f}%")
    print("=" * 58)
    print("\nThe recogniser was trained on unfused crops, so 'fused' is being")
    print("asked to read a distribution it has never seen. A gain here despite")
    print("that is strong evidence; a small loss is not yet a refutation - the")
    print("fair test is retraining on fused images, which is the next step if")
    print("the preview looks right.")


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--preview", type=int, default=0)
    ap.add_argument("--evaluate", action="store_true")
    ap.add_argument("--n", type=int, default=150)
    ap.add_argument("--upscale", type=int, default=UPSCALE)
    ap.add_argument("--seed", type=int, default=3)
    ap.add_argument("--out", default="fusion_preview.jpg")
    args = ap.parse_args()
    if args.preview:
        _preview(args)
    elif args.evaluate:
        _evaluate(args)
    else:
        ap.print_help()


if __name__ == "__main__":
    main()
