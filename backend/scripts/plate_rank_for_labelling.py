"""backend/scripts/plate_rank_for_labelling.py — order a harvest so the most
readable crops are labelled first.

WHY ORDERING MATTERS MORE THAN FILTERING
  A random sample of the 1,047 detector crops was inspected: roughly a sixth
  are not plates at all (a shop sign reading CHITRA, Gujarati lettering on a
  van, a stretch of road), and roughly a third are plates too degraded for a
  person to read. The detector was trained on 656 boxes and has learned
  "rectangular text-like region", which is most of what 656 boxes can teach.

  Deleting the doubtful ones would be the wrong fix. The threshold that
  removes a shop sign also removes the hardest real plates, and those are
  precisely the examples that teach the recogniser something new. Ordering
  costs nothing and throws nothing away: the labeller works down the list, the
  good crops arrive first, and they stop when the effort stops paying. A crop
  that is obviously not a plate is dismissed in about a second, so the cost of
  leaving it in the list is small - while the cost of filtering out a hard
  real plate is a training example that cannot be recovered.

WHAT THE SCORE USES, AND WHAT IT DELIBERATELY DOES NOT
  Sharpness (variance of Laplacian) and contrast, both measured on the crop
  after normalising for size, plus detected plate width. These correlate with
  whether a human can read the glyphs.

  It does NOT use OCR. Ranking by what EasyOCR can read would reproduce the
  bias that limited the previous harvest to 7.8% yield: it would put the
  plates the machine already handles at the top and bury the ones worth
  labelling. The point of this batch is to reach the plates OCR cannot read.

USAGE
  python -m backend.scripts.plate_rank_for_labelling
  python -m backend.scripts.plate_rank_for_labelling --src data/plate_label_batch6
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import cv2
import numpy as np


def legibility(img) -> dict:
    """Size-normalised readability proxies for one crop."""
    g = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY) if img.ndim == 3 else img
    # Resample to a fixed height first: variance of Laplacian scales with
    # resolution, so comparing a 30px crop against a 90px one unnormalised
    # ranks by size rather than by clarity.
    h, w = g.shape[:2]
    if h < 4 or w < 8:
        return {"sharp": 0.0, "contrast": 0.0, "ink": 0.0}
    s = 40.0 / h
    gg = cv2.resize(g, (max(8, int(w * s)), 40), interpolation=cv2.INTER_AREA)
    sharp = float(cv2.Laplacian(gg, cv2.CV_64F).var())
    contrast = float(gg.std())
    # Fraction of pixels that are strongly dark or strongly light after local
    # normalisation - glyphs on a plate are bimodal, a blurred smear is not.
    norm = cv2.normalize(gg, None, 0, 255, cv2.NORM_MINMAX)
    ink = float(((norm < 70) | (norm > 185)).mean())
    return {"sharp": sharp, "contrast": contrast, "ink": ink}


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--src", default="data/plate_label_batch6")
    args = ap.parse_args()

    src = Path(args.src)
    rows = [json.loads(l) for l in (src / "labels.jsonl").open(encoding="utf-8")]
    print(f"crops: {len(rows)}")

    scored = []
    for r in rows:
        img = cv2.imread(str(src / "images" / r["file"]))
        if img is None:
            continue
        m = legibility(img)
        r.update({k: round(v, 3) for k, v in m.items()})
        scored.append(r)

    # Rank each factor by percentile rather than raw value, so one factor with
    # a long tail cannot dominate the ordering.
    def pct(key):
        vals = sorted(r[key] for r in scored)
        return {v: i / max(len(vals) - 1, 1) for i, v in enumerate(vals)}

    p_sharp, p_con, p_w = pct("sharp"), pct("contrast"), pct("plate_w")
    for r in scored:
        r["rank_score"] = round(
            0.45 * p_sharp[r["sharp"]] +
            0.30 * p_con[r["contrast"]] +
            0.25 * p_w[r["plate_w"]], 4)
    scored.sort(key=lambda r: -r["rank_score"])

    out = src / "labels_ranked.jsonl"
    with out.open("w", encoding="utf-8") as f:
        for r in scored:
            f.write(json.dumps(r) + "\n")

    print(f"\nwrote {out}")
    print("=" * 58)
    print(f"{'decile':<10} {'plate_w':>9} {'sharp':>9} {'contrast':>9}")
    print("-" * 58)
    n = len(scored)
    for d in range(10):
        chunk = scored[d * n // 10:(d + 1) * n // 10]
        if not chunk:
            continue
        print(f"top {(d+1)*10:>3}%  {np.mean([c['plate_w'] for c in chunk]):>9.0f} "
              f"{np.mean([c['sharp'] for c in chunk]):>9.0f} "
              f"{np.mean([c['contrast'] for c in chunk]):>9.1f}")
    print("=" * 58)
    print("\nLabel from the top. The list degrades gradually rather than at a")
    print("cliff, so stop when the crops stop being worth the time - that")
    print("judgement is better made by the person labelling than by a"
          " threshold here.")


if __name__ == "__main__":
    main()
