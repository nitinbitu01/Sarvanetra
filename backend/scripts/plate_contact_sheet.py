"""backend/scripts/plate_contact_sheet.py — one image showing every plate read.

WHY A CONTACT SHEET
  Fifty separate files cannot be scanned quickly, and the whole point of the
  evidence images is that a human checks the decoded string against the actual
  pixels. Putting them in a grid with the decode printed underneath makes that
  a one-glance job: any row where the printed text disagrees with the plate in
  the picture is a defect you can see immediately.

  Verification of a plate read is not optional and cannot be done from the
  numbers alone - a confident wrong read looks exactly like a confident right
  one in a JSON file.

USAGE
  python -m backend.scripts.plate_contact_sheet
  python -m backend.scripts.plate_contact_sheet --cols 4 --cell-w 460
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import cv2
import numpy as np

READS = Path("output/plate_multiframe.jsonl")
OUT = Path("output/plate_contact_sheet.jpg")


def band_of(img):
    h, w = img.shape[:2]
    b = img[int(h * 0.42):int(h * 0.97), int(w * 0.08):int(w * 0.92)]
    return None if b.size == 0 or b.shape[1] < 20 else b


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--cols", type=int, default=5)
    ap.add_argument("--cell-w", type=int, default=380)
    ap.add_argument("--cell-h", type=int, default=190)
    ap.add_argument("--label-h", type=int, default=46)
    args = ap.parse_args()

    rows = [json.loads(l) for l in READS.open(encoding="utf-8")]
    # Highest agreement first: the reads most likely correct lead, so a
    # reviewer sees the quality gradient rather than a random order.
    rows.sort(key=lambda r: (-r["agreement"], -r["votes"]))
    print(f"reads: {len(rows)}")

    cw, ch, lh = args.cell_w, args.cell_h, args.label_h
    cols = args.cols
    nrows = (len(rows) + cols - 1) // cols
    sheet = np.full((nrows * (ch + lh), cols * cw, 3), 26, np.uint8)

    used = 0
    for i, r in enumerate(rows):
        img = cv2.imread(r["path"])
        if img is None:
            continue
        b = band_of(img)
        if b is None:
            continue
        # Letterbox into the cell so aspect ratio is preserved - stretching a
        # plate would make character shapes unjudgeable.
        s = min(cw / b.shape[1], ch / b.shape[0])
        rs = cv2.resize(b, (max(1, int(b.shape[1] * s)),
                            max(1, int(b.shape[0] * s))))
        rr, cc = divmod(used, cols)
        y0, x0 = rr * (ch + lh), cc * cw
        oy = y0 + (ch - rs.shape[0]) // 2
        ox = x0 + (cw - rs.shape[1]) // 2
        sheet[oy:oy + rs.shape[0], ox:ox + rs.shape[1]] = rs

        # Colour by evidence strength so weak reads are visually obvious.
        agree, votes = r["agreement"], r["votes"]
        col = ((80, 220, 80) if agree >= 0.95 and votes >= 2 else
               (60, 200, 240) if agree >= 0.85 else (80, 140, 240))
        cv2.putText(sheet, r["plate"], (x0 + 8, y0 + ch + 24),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.62, col, 2, cv2.LINE_AA)
        note = f"{r['camera']}  {r['plate_px_est']:.0f}px  v{votes} a{agree:.2f}"
        if r.get("prior_note"):
            note += "  [state-corrected]"
        cv2.putText(sheet, note, (x0 + 8, y0 + ch + 41),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.40, (165, 165, 165), 1, cv2.LINE_AA)
        cv2.rectangle(sheet, (x0 + 2, y0 + 2), (x0 + cw - 3, y0 + ch + lh - 3),
                      (60, 60, 60), 1)
        used += 1

    sheet = sheet[:((used + cols - 1) // cols) * (ch + lh)]
    OUT.parent.mkdir(parents=True, exist_ok=True)
    cv2.imwrite(str(OUT), sheet, [cv2.IMWRITE_JPEG_QUALITY, 94])
    print(f"placed {used} plates -> {OUT.resolve()}")
    print(f"size: {sheet.shape[1]}x{sheet.shape[0]}")
    print("\ngreen = strong evidence (agree>=0.95, 2+ frames)")
    print("cyan  = moderate      red = weak, check these first")
    print("\nCompare each printed string against the plate in its picture.")


if __name__ == "__main__":
    main()
