"""backend/scripts/plate_reject_nonplates.py — drop the detector's non-plates
before a human spends hours on them.

WHAT WENT WRONG UPSTREAM
  The detector was trained on 656 boxes, which is enough to learn "horizontal
  rectangle containing text" and not enough to learn "vehicle registration
  plate". On Gujarat traffic footage those are very different sets: buses
  carry the operator's name and website across the panel, shopfronts sit in
  the background, and one camera has a station caption burned into the frame.
  A sample of the harvest put roughly a sixth in that category.

  Ranking the harvest by sharpness and contrast then made it worse, not
  better. Signage is large, high contrast and well lit; real plates are small
  and grey. Sorting on image quality put every false positive at the top - the
  ordering was actively worse than random, which is why it is replaced here
  rather than tuned.

TWO REJECTIONS, BOTH GROUNDED IN SOMETHING ALREADY MEASURED

  1. CAMERA. An earlier survey established which cameras can see a plate at
     all, from their mounting geometry. On the others the detector cannot be
     finding plates, because there are none to find - so every detection there
     is a false positive by construction. This removes them wholesale.

  2. PLATE-LIKENESS, judged by the recogniser rather than by image quality.
     The recogniser is only ~45% accurate at reading a plate exactly, but that
     is not what it is asked here. It is asked whether the glyph sequence
     could be a plate at all: an Indian plate is 9-11 characters that fit
     LL-DD-L{1,3}-DDDD, while 'www.ramanitravels.com' and 'SLEEPER COACH' are
     not close to that under any character substitution.

     This is deliberately a WEAK test. It rejects on structure, never on
     confidence or sharpness, so a barely-legible real plate that the
     recogniser reads wrongly still passes as long as it reads roughly the
     right SHAPE. Rejecting on confidence would repeat the earlier mistake of
     discarding exactly the hard examples the training set needs.

USAGE
  python -m backend.scripts.plate_reject_nonplates --src data/plate_label_batch6
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import cv2
import torch

from backend.scripts.plate_beam_decode import LETTERS, valid_prefix
from backend.scripts.train_plate_recognizer import (CHARS, CRNN, IMG_H, IMG_W,
                                                    ctc_decode)

# Cameras whose geometry allows a readable plate, from the earlier survey and
# recorded in config.yaml as anpr.cameras. CAM_11 is included because the
# detector audit found a 35% hit rate there.
PLATE_CAPABLE = {"CAM_06", "CAM_07", "CAM_08", "CAM_09", "CAM_10",
                 "CAM_11", "CAM_18", "CAM_21", "CAM_27"}

CKPT = Path("models/plate_recognizer/finetuned.pt")


def plate_like(text: str) -> tuple[bool, str]:
    """Could this reading be a plate, allowing for glyph confusions?"""
    t = "".join(c for c in text.upper() if c.isalnum())
    if not (8 <= len(t) <= 11):
        return False, f"length {len(t)}"
    # The first two characters carry the state code. Digits that are commonly
    # confused with letters are allowed through, since the point is to reject
    # signage rather than to grade the reading.
    head = t[:2].replace("0", "O").replace("1", "I").replace("6", "G") \
                .replace("5", "S").replace("8", "B")
    if not all(c in LETTERS for c in head):
        return False, "no letter head"
    if not valid_prefix(head):
        return False, f"bad state {head}"
    return True, ""


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--src", default="data/plate_label_batch6")
    ap.add_argument("--out", default="")
    ap.add_argument("--keep-all-cameras", action="store_true",
                    help="Skip the camera rejection (diagnostic only).")
    args = ap.parse_args()

    src = Path(args.src)
    rows = [json.loads(l) for l in (src / "labels.jsonl").open(encoding="utf-8")]
    print(f"harvested crops : {len(rows)}")

    if not args.keep_all_cameras:
        before = len(rows)
        rows = [r for r in rows if r["camera"] in PLATE_CAPABLE]
        print(f"after camera    : {len(rows)}  ({before - len(rows)} dropped "
              f"from cameras that cannot see a plate)")

    dev = "cuda" if torch.cuda.is_available() else "cpu"
    ck = torch.load(CKPT, map_location=dev, weights_only=False)
    model = CRNN(len(CHARS) + 1).to(dev)
    model.load_state_dict(ck["model"])
    model.eval()

    kept, dropped = [], []
    with torch.no_grad():
        for r in rows:
            g = cv2.imread(str(src / "images" / r["file"]), cv2.IMREAD_GRAYSCALE)
            if g is None:
                continue
            g = cv2.resize(g, (IMG_W, IMG_H), interpolation=cv2.INTER_AREA)
            x = torch.from_numpy(g).float().div(127.5).sub(1.0)[None, None].to(dev)
            raw = ctc_decode(model(x))[0]
            ok, why = plate_like(raw)
            r["raw_read"] = raw
            (kept if ok else dropped).append((r, why))

    print(f"after shape     : {len(kept)}  ({len(dropped)} rejected as not "
          f"plate-shaped text)")

    out = Path(args.out) if args.out else src / "labels_filtered.jsonl"
    with out.open("w", encoding="utf-8") as f:
        for r, _ in kept:
            f.write(json.dumps(r) + "\n")

    from collections import Counter
    print("\nrejection reasons:")
    for why, n in Counter(w for _, w in dropped).most_common(8):
        print(f"  {why:<20} {n:>5}")
    print("\nkept per camera:")
    for c, n in Counter(r["camera"] for r, _ in kept).most_common():
        print(f"  {c:<12} {n:>5}")
    print(f"\nwrote {out}")
    print("\nEvery rejection is on STRUCTURE, never on sharpness or confidence,")
    print("so hard-but-real plates survive. Check a sample by eye before")
    print("labelling - the previous ordering attempt looked sound in the")
    print("numbers and was wrong in the images.")


if __name__ == "__main__":
    main()
