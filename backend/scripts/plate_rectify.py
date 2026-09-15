"""backend/scripts/plate_rectify.py — straighten a plate crop before reading it.

THE PROBLEM IT ADDRESSES
  A CCTV camera almost never faces a plate square-on. It looks down from a
  pole and off to one side, so the plate arrives rotated and keystoned: the
  near edge longer than the far edge, the baseline tilted a few degrees. A CRNN
  reads a horizontal strip of fixed height, so a tilted plate puts parts of
  two character rows into the same column of the feature map, and the network
  spends capacity undoing geometry instead of recognising glyphs.

  Every commercial ALPR pipeline rectifies before recognition. This one has
  not, which makes it one of the few remaining levers that costs no labels.

DESIGNED TO FAIL SAFE, BECAUSE THE ALTERNATIVE IS SILENT DAMAGE
  On a 50px plate the border is often the only strong structure, and it can be
  missing on one side, broken by a bumper shadow, or confused with the number
  plate frame. A confident-but-wrong quadrilateral would warp a readable plate
  into an unreadable one, and nothing downstream could tell that had happened.

  So each stage is gated: the angle correction applies only when the estimate
  is within a plausible range, and the perspective correction only when four
  corners are found that actually form a plate-shaped convex quadrilateral.
  When a gate fails the input is returned untouched. A missed correction costs
  a little accuracy; a wrong one costs a plate.

MEASURED, NOT ASSUMED
  Twelve enhancement techniques were tried on these crops and none beat the
  baseline by more than two points. Rectification is a different kind of
  operation - it changes geometry rather than pixel statistics - but that is
  an argument for testing it, not for believing it. Use --compare to score it
  against the untouched crops on held-out plates before adopting it.

USAGE
  python -m backend.scripts.plate_rectify --compare
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

MAX_ROT_DEG = 18.0          # beyond this the estimate is noise, not a plate
MIN_QUAD_AREA_FRAC = 0.45   # a real plate fills most of its own crop


def _deskew(g: np.ndarray) -> np.ndarray:
    """Rotate so the character baseline is horizontal.

    The angle comes from the minimum-area rectangle around the dark glyph mass
    rather than from Hough lines: on a small plate the border lines are often
    broken, while the glyph mass is always present - it is the thing being
    read.
    """
    h, w = g.shape[:2]
    if h < 8 or w < 20:
        return g
    blur = cv2.GaussianBlur(g, (3, 3), 0)
    # Otsu on the inverted image: glyphs are usually dark on light, and this
    # keeps them as foreground either way after the polarity check below.
    _, bw = cv2.threshold(blur, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
    if bw.mean() > 127:                       # foreground took over: invert
        bw = 255 - bw
    pts = cv2.findNonZero(bw)
    if pts is None or len(pts) < 20:
        return g
    (_, _), (rw, rh), ang = cv2.minAreaRect(pts)
    # OpenCV reports the angle of the shorter side under some builds; fold it
    # into [-45, 45] so a wide plate reads as a small tilt, not an 89 degree one.
    if rw < rh:
        ang += 90.0
    if ang > 45:
        ang -= 90
    if ang < -45:
        ang += 90
    if abs(ang) < 0.6 or abs(ang) > MAX_ROT_DEG:
        return g                               # nothing to do, or implausible
    M = cv2.getRotationMatrix2D((w / 2, h / 2), ang, 1.0)
    return cv2.warpAffine(g, M, (w, h), flags=cv2.INTER_CUBIC,
                          borderMode=cv2.BORDER_REPLICATE)


def _unwarp(g: np.ndarray) -> np.ndarray:
    """Remove keystone by mapping the plate's own quadrilateral to a rectangle.

    Only applied when a four-cornered, convex, plate-shaped contour is found
    that fills most of the crop. Anything else returns the input unchanged -
    see the module docstring on failing safe.
    """
    h, w = g.shape[:2]
    if h < 12 or w < 30:
        return g
    e = cv2.Canny(cv2.GaussianBlur(g, (3, 3), 0), 40, 130)
    e = cv2.dilate(e, np.ones((2, 2), np.uint8), iterations=1)
    cnts, _ = cv2.findContours(e, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not cnts:
        return g
    c = max(cnts, key=cv2.contourArea)
    if cv2.contourArea(c) < MIN_QUAD_AREA_FRAC * h * w:
        return g
    peri = cv2.arcLength(c, True)
    quad = cv2.approxPolyDP(c, 0.03 * peri, True)
    if len(quad) != 4 or not cv2.isContourConvex(quad):
        return g
    p = quad.reshape(4, 2).astype(np.float32)
    s, d = p.sum(1), np.diff(p, axis=1).ravel()
    src = np.array([p[np.argmin(s)], p[np.argmin(d)],
                    p[np.argmax(s)], p[np.argmax(d)]], np.float32)
    wa = max(np.linalg.norm(src[0] - src[1]), np.linalg.norm(src[3] - src[2]))
    ha = max(np.linalg.norm(src[0] - src[3]), np.linalg.norm(src[1] - src[2]))
    if wa < 24 or ha < 8 or not (1.8 <= wa / ha <= 7.5):
        return g                               # not plate-shaped after all
    dst = np.array([[0, 0], [wa - 1, 0], [wa - 1, ha - 1], [0, ha - 1]],
                   np.float32)
    M = cv2.getPerspectiveTransform(src, dst)
    return cv2.warpPerspective(g, M, (int(wa), int(ha)), flags=cv2.INTER_CUBIC,
                               borderMode=cv2.BORDER_REPLICATE)


def rectify(g: np.ndarray, mode: str = "full") -> np.ndarray:
    if mode == "none":
        return g
    if mode in ("deskew", "full"):
        g = _deskew(g)
    if mode == "full":
        g = _unwarp(g)
    return g


def _compare(args) -> None:
    """Score rectified against untouched crops on the SAME held-out plates."""
    import torch

    from backend.scripts.indian_plate_grammar import decode_plate
    from backend.scripts.train_plate_recognizer import (CHARS, CRNN, IMG_H,
                                                        IMG_W, STOI,
                                                        ctc_decode)

    verified = [json.loads(l) for l in (REAL / "verified_all.jsonl").open(encoding="utf-8")]
    truth = {(v["camera"], v["track"]): v["text"] for v in verified
             if v["text"] and all(c in STOI for c in v["text"])}
    by_track = defaultdict(list)
    for line in (REAL / "labels.jsonl").open(encoding="utf-8"):
        r = json.loads(line)
        if (r["camera"], r["track"]) in truth:
            by_track[(r["camera"], r["track"])].append(r["file"])

    keys = sorted(by_track)
    random.Random(args.seed).shuffle(keys)
    test = [(by_track[k][0], truth[k]) for k in keys[:args.holdout]]

    dev = "cuda" if torch.cuda.is_available() else "cpu"
    ck = torch.load(Path("models/plate_recognizer/finetuned.pt"),
                    map_location=dev, weights_only=False)
    model = CRNN(len(CHARS) + 1).to(dev)
    model.load_state_dict(ck["model"])
    model.eval()
    print(f"held out : {len(test)} plates\n")

    results = {}
    per_plate = {}
    for mode in ("none", "deskew", "full"):
        ok = 0
        got = []
        with torch.no_grad():
            for f, gt in test:
                g = cv2.imread(str(REAL / "images" / f), cv2.IMREAD_GRAYSCALE)
                if g is None:
                    continue
                g = rectify(g, mode)
                g = cv2.resize(g, (IMG_W, IMG_H), interpolation=cv2.INTER_AREA)
                x = torch.from_numpy(g).float().div(127.5).sub(1.0)
                pred = ctc_decode(model(x[None, None].to(dev)))[0]
                d = decode_plate(pred)
                hit = (d["plate"] or pred) == gt
                ok += hit
                got.append(hit)
        results[mode] = ok / max(len(test), 1)
        per_plate[mode] = got
        print(f"  {mode:<8} exact {results[mode]*100:>5.1f}%")

    base = results["none"]
    print("\n" + "=" * 58)
    for mode in ("deskew", "full"):
        fixed = sum(1 for a, b in zip(per_plate["none"], per_plate[mode])
                    if not a and b)
        broke = sum(1 for a, b in zip(per_plate["none"], per_plate[mode])
                    if a and not b)
        print(f"{mode:<8} {(results[mode]-base)*100:>+6.1f} points   "
              f"fixed {fixed:>2}  broke {broke:>2}")
    print("=" * 58)
    print("\nfixed/broke matters more than the net change: a net zero can hide")
    print("five of each, and a technique that breaks working plates is not")
    print("worth keeping even when the average survives.")


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--compare", action="store_true")
    ap.add_argument("--holdout", type=int, default=55)
    ap.add_argument("--seed", type=int, default=1337)
    args = ap.parse_args()
    if args.compare:
        _compare(args)
    else:
        ap.print_help()


if __name__ == "__main__":
    main()
