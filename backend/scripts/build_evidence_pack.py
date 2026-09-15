"""backend/scripts/build_evidence_pack.py — the data behind an accuracy claim,
in a form a judge can check rather than take on trust.

WHY A NUMBER ALONE PERSUADES NOBODY
  Every team at the table claims 95%. A judge who has heard that five times
  discounts the sixth, and rightly - the figure carries no information without
  the protocol that produced it. What does carry information is evidence the
  judge can inspect: this crop, this ground truth, this is what the standard
  tool read, this is what our model read.

  The failures matter as much as the successes and are included deliberately.
  A judge shown only wins assumes the losses are being hidden. A judge shown a
  plate they cannot read either, next to the model's wrong answer, understands
  that the limit is the camera rather than the modelling - which is the actual
  finding of this project and cannot be conveyed by a percentage.

WHAT THIS EMITS
  A JSON file of held-out vehicles the model never trained on, each with its
  crop, the human ground truth, the EasyOCR baseline read, and the model's
  read. Sorted so the page can offer easy, hard and impossible cases as
  separate sections.

EVERY PLATE HERE IS HELD OUT
  The split is the one the final model was trained under, so nothing in this
  pack was seen during training or used to pick a checkpoint. Building an
  evidence pack from training data would be the most embarrassing possible
  thing to be caught doing, and it is easy to do by accident.

USAGE
  python -m backend.scripts.build_evidence_pack --out evidence_pack.json
"""
from __future__ import annotations

import argparse
import base64
import json
from pathlib import Path

import cv2
import torch

from backend.scripts.indian_plate_grammar import decode_plate
from backend.scripts.plate_eval_clean import _vote, lev
from backend.scripts.plate_final_model import (PlateCRNN, load_data,
                                               read_vehicle, split)
from backend.scripts.train_plate_recognizer import CHARS

REAL = Path("data/plate_real")
OUT_DIR = Path("models/plate_recognizer")


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--out", default="evidence_pack.json")
    ap.add_argument("--height", type=int, default=64)
    ap.add_argument("--width", type=int, default=256)
    ap.add_argument("--members", type=int, default=3)
    ap.add_argument("--easyocr", action="store_true",
                    help="Also run EasyOCR for the baseline column. Slower, "
                         "but the comparison is the point of the pack.")
    args = ap.parse_args()

    truth, by_track = load_data()
    # The SAME split the final model was trained under - seed 1000.
    test_k, val_k, train_k = split(by_track, 1000)
    print(f"held-out vehicles : {len(test_k)}")

    dev = "cuda" if torch.cuda.is_available() else "cpu"
    members = []
    for j in range(args.members):
        p = OUT_DIR / f"final_m{j}.pt"
        if not p.is_file():
            continue
        ck = torch.load(p, map_location=dev, weights_only=False)
        m = PlateCRNN(len(CHARS) + 1, img_h=args.height).to(dev)
        m.load_state_dict(ck["model"])
        m.eval()
        members.append(m)
    print(f"ensemble members  : {len(members)}")

    reader = None
    if args.easyocr:
        import easyocr
        print("loading EasyOCR for the baseline column...", flush=True)
        reader = easyocr.Reader(["en"], gpu=True, verbose=False)

    rows = []
    for i, k in enumerate(test_k, 1):
        files = [c["file"] for c in by_track[k]]
        gt = truth[k]
        raw = read_vehicle(members, files, args.height, args.width, dev)
        d = decode_plate(raw)
        ours = d["plate"] or raw

        # Show the sharpest crop - it is the one a person would be given if
        # asked to read the plate, so the comparison is fair to the human.
        best, best_s = None, -1.0
        for f in files:
            g = cv2.imread(str(REAL / "images" / f), cv2.IMREAD_GRAYSCALE)
            if g is None:
                continue
            s = float(cv2.Laplacian(g, cv2.CV_64F).var())
            if s > best_s:
                best_s, best = s, f
        if best is None:
            continue

        base = ""
        if reader:
            # The baseline gets EVERY advantage our model gets: all the same
            # frames to vote across, and the same Indian-plate grammar
            # correction applied to its output. A baseline denied the
            # post-processing would make the comparison flattering and
            # indefensible the moment a judge asked about it.
            cand = []
            for f in files:
                img = cv2.imread(str(REAL / "images" / f))
                if img is None:
                    continue
                try:
                    res = reader.readtext(
                        img, allowlist="ABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789")
                except Exception:                              # noqa: BLE001
                    continue
                t = "".join(c for c in "".join(x[1] for x in res).upper()
                            if c.isalnum())
                if t:
                    cand.append(t)
            if cand:
                voted = _vote(cand)
                bd = decode_plate(voted)
                base = bd["plate"] or voted

        rows.append({
            "camera": k[0], "track": k[1],
            "truth": gt, "ours": ours, "easyocr": base,
            "ok": ours == gt,
            "dist": lev(ours, gt),
            "sharp": round(best_s, 1),
            "frames": len(files),
            "img": base64.b64encode(
                (REAL / "images" / best).read_bytes()).decode(),
        })
        if i % 20 == 0:
            print(f"  {i}/{len(test_k)}", flush=True)

    ok = sum(r["ok"] for r in rows)
    ocr_ok = sum(r["easyocr"] == r["truth"] for r in rows) if reader else 0
    chars = sum(len(r["truth"]) for r in rows)
    dist = sum(r["dist"] for r in rows)

    pack = {
        "summary": {
            "held_out_vehicles": len(rows),
            "exact_match": round(ok / max(len(rows), 1) * 100, 1),
            "char_accuracy": round((1 - dist / max(chars, 1)) * 100, 1),
            "easyocr_exact": round(ocr_ok / max(len(rows), 1) * 100, 1)
            if reader else None,
            "within_one_char": round(
                sum(r["dist"] <= 1 for r in rows) / max(len(rows), 1) * 100, 1),
        },
        "rows": rows,
    }
    Path(args.out).write_text(json.dumps(pack), encoding="utf-8")

    s = pack["summary"]
    print("\n" + "=" * 56)
    print(f"held-out vehicles : {s['held_out_vehicles']}")
    print(f"exact match       : {s['exact_match']}%")
    print(f"character accuracy: {s['char_accuracy']}%")
    print(f"within one char   : {s['within_one_char']}%")
    if s["easyocr_exact"] is not None:
        print(f"EasyOCR exact     : {s['easyocr_exact']}%")
    print("=" * 56)
    print(f"\nwrote {args.out}")


if __name__ == "__main__":
    main()
