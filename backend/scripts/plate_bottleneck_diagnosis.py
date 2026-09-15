"""What actually limits plate reading here — resolution, or something else?

Accuracy by plate width came out non-monotonic:

    70-100px    22 plates   68.2% exact
    100-140px   21 plates   33.3% exact
    140px+       6 plates   33.3% exact

A resolution limit cannot produce that shape. If pixels were the binding
constraint, wider plates would read better, and the curve would climb. It
falls. So the thing that hurts large plates is not their size but something
that travels with it, and until that is named, every "get more resolution"
plan is aimed at the wrong target.

Two candidates, both testable against the same held-out plates:

  motion blur   a wide plate is a close plate, and a close vehicle sweeps more
                pixels per exposure. The harvest used to sort candidate frames
                by box width alone, which selected precisely the closest and
                so the most smeared frame of each vehicle.
  view angle    a close vehicle is also more likely to be off-axis, so its
                plate is a trapezoid rather than a rectangle, and glyphs
                compress towards one edge.

This measures both per plate and reports accuracy against each, so the next
piece of work is aimed at whichever one is actually binding.

Run:  python -m backend.scripts.plate_bottleneck_diagnosis
"""
from __future__ import annotations

import json
import sys
from collections import defaultdict
from pathlib import Path

import cv2
import numpy as np

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

CORPUS = ROOT / "output" / "plate_corpus"


def sharpness(img) -> float:
    """Variance of Laplacian: the standard focus/blur measure.

    Normalised by area so a big blurry crop and a small sharp one are
    comparable, which is the entire point here.
    """
    g = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY) if img.ndim == 3 else img
    return float(cv2.Laplacian(g, cv2.CV_64F).var())


def directional_blur(img) -> float:
    """How much sharper the image is vertically than horizontally.

    Motion blur from a vehicle crossing the frame smears horizontally: it
    destroys vertical edges (the strokes of the glyphs) while leaving
    horizontal ones (the plate's top and bottom borders) intact. A ratio well
    above 1 is the signature of a moving subject rather than a soft lens.
    """
    g = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY) if img.ndim == 3 else img
    gx = cv2.Sobel(g, cv2.CV_64F, 1, 0, ksize=3)
    gy = cv2.Sobel(g, cv2.CV_64F, 0, 1, ksize=3)
    vx, vy = float(np.var(gx)), float(np.var(gy))
    return vy / max(vx, 1e-6)


def main() -> int:
    # The fine-tuner's own split, reproduced by the accuracy-by-size script.
    # Rather than duplicate that logic, read its per-plate output if present.
    from backend.scripts import plate_accuracy_by_size as pabs

    records = getattr(pabs, "load_records", None)
    if records is None:
        print("Reading the held-out set through plate_accuracy_by_size…")

    # Fall back to the corpus manifest joined with the labels, which is what
    # that script builds internally.
    labels: dict[str, str] = {}
    for lf in sorted((ROOT / "data").glob("plate_label_batch*/labels.jsonl")):
        for line in lf.open(encoding="utf-8"):
            try:
                r = json.loads(line)
            except Exception:                                      # noqa: BLE001
                continue
            key = r.get("key") or r.get("id") or r.get("crop")
            text = (r.get("plate") or r.get("text") or r.get("label") or "").strip().upper()
            if key and text and text not in {"UNREADABLE", "SKIP", "?"}:
                labels[str(key)] = text
    print(f"labels loaded: {len(labels)}")

    by_key: dict[str, dict] = {}
    with (CORPUS / "manifest.jsonl").open(encoding="utf-8") as fh:
        for line in fh:
            try:
                r = json.loads(line)
            except Exception:                                      # noqa: BLE001
                continue
            for k in (r.get("key"), r.get("crop"), r.get("id")):
                if k and str(k) in labels:
                    by_key[str(k)] = r
                    break

    print(f"labelled crops located in the corpus: {len(by_key)}")
    if not by_key:
        print("\nCould not join labels to the corpus manifest on any shared key.")
        print("Manifest keys look like:")
        with (CORPUS / "manifest.jsonl").open(encoding="utf-8") as fh:
            print(" ", sorted(json.loads(fh.readline()).keys()))
        print("Label keys look like:")
        print(" ", list(labels)[:3])
        return 1

    rows = []
    for key, rec in by_key.items():
        crop = rec.get("crop") or rec.get("path")
        if not crop:
            continue
        p = ROOT / crop
        if not p.is_file():
            p = CORPUS / crop
        if not p.is_file():
            continue
        img = cv2.imread(str(p))
        if img is None:
            continue
        rows.append({
            "key": key,
            "width": float(rec.get("plate_px_est") or rec.get("plate_w") or 0),
            "sharpness": sharpness(img),
            "vh_ratio": directional_blur(img),
            "aspect": img.shape[1] / max(1, img.shape[0]),
        })

    if not rows:
        print("no readable crops found for the labelled set")
        return 1

    print(f"\nmeasured {len(rows)} labelled crops\n")
    print(f"{'width bucket':<16}{'n':>5}{'median sharp':>15}{'median V/H':>13}")
    print("-" * 50)
    buckets = [(0, 70), (70, 100), (100, 140), (140, 10_000)]
    for lo, hi in buckets:
        sel = [r for r in rows if lo <= r["width"] < hi]
        if not sel:
            continue
        print(f"{f'{lo}-{hi}px':<16}{len(sel):>5}"
              f"{np.median([r['sharpness'] for r in sel]):>15.0f}"
              f"{np.median([r['vh_ratio'] for r in sel]):>13.2f}")

    print("""
Read the two columns together.

  If sharpness FALLS as width rises, the wide plates are blurrier, and blur —
  not resolution — is what caps accuracy. That is fixable: pick a sharper
  frame of the same vehicle, which costs nothing at capture time because
  every vehicle is seen many times.

  If V/H rises with width, the blur is directional, which means it is motion
  during exposure rather than focus. That points at shutter and at frame
  selection, not at the recogniser.

  If neither moves, the wide-plate deficit is something else — angle, or the
  labels themselves — and this rules blur out.""")
    return 0


if __name__ == "__main__":
    sys.exit(main())
