"""Are the recovered plate boxes actually on the plates?

recover_plate_boxes locates each labelled crop inside its source frame by
correlation and writes the result as a YOLO box. A high correlation says the
crop was found; it does not by itself say the box is on the plate. If the
harvest band geometry were slightly wrong, every box would be shifted by the
same amount, every correlation would still score near 1.0, and the detector
would learn to fire on bumpers — a failure invisible in the recovery output.

The check that catches that: cut each recovered box out of its frame and read
it. Every box sits on a plate a human already transcribed, so the text is
known. If the boxes are right, reads should land near the recogniser's
measured holdout rate. If they are shifted, reads collapse toward noise.

Two controls make the number interpretable:

    recovered box       what training will actually see
    box shifted 15%     the same boxes moved down by 15% of their height,
                        which is what a mis-set band would produce

The shifted control matters because "50% exact" means nothing on its own — it
only means something against what a deliberately wrong box scores on the same
plates with the same recogniser.

Run:  python -m backend.scripts.verify_recovered_boxes
      python -m backend.scripts.verify_recovered_boxes --data data/plate_detect_v2
"""
from __future__ import annotations

import argparse
import random
import sys
from pathlib import Path

import cv2
import numpy as np
import torch
import torch.nn.functional as F

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from backend.scripts.indian_plate_grammar import decode_plate      # noqa: E402
from backend.scripts.plate_width_gate_eval import greedy, lev      # noqa: E402
from backend.scripts.train_plate_recognizer import CHARS           # noqa: E402

MODELS = ROOT / "models" / "plate_recognizer"


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", type=Path, default=ROOT / "data" / "plate_detect_v2")
    ap.add_argument("--sample", type=int, default=150)
    args = ap.parse_args()

    # The recovery writes the plate text into a sidecar so this check can find
    # ground truth without re-deriving the filename mapping.
    truth_file = args.data / "boxes.jsonl"
    if not truth_file.is_file():
        print(f"{truth_file} not found — re-run recover_plate_boxes so it "
              f"records the text alongside each box.", file=sys.stderr)
        return 1

    import json
    rows = [json.loads(l) for l in truth_file.open(encoding="utf-8")]
    random.Random(7).shuffle(rows)
    rows = rows[:args.sample]
    print(f"checking {len(rows)} recovered boxes\n", flush=True)

    dev = "cuda" if torch.cuda.is_available() else "cpu"
    members, hw = [], None
    for i in range(3):
        p = MODELS / f"final_m{i}.pt"
        if not p.is_file():
            continue
        ck = torch.load(p, map_location=dev, weights_only=False)
        from backend.scripts.plate_final_model import PlateCRNN
        m = PlateCRNN(len(ck.get("chars", CHARS)) + 1, img_h=int(ck["img_h"]))
        m.load_state_dict(ck["model"])
        members.append(m.to(dev).eval())
        hw = (int(ck["img_h"]), int(ck["img_w"]))
    h, w = hw

    def read(bgr: np.ndarray) -> str:
        if bgr is None or bgr.size == 0:
            return ""
        g = (cv2.cvtColor(bgr, cv2.COLOR_BGR2GRAY)
             if bgr.ndim == 3 and bgr.shape[2] >= 3 else bgr.reshape(bgr.shape[:2]))
        x = cv2.resize(g, (w, h), interpolation=cv2.INTER_AREA)
        x = torch.from_numpy(x).float().div(127.5).sub(1.0)[None, None].to(dev)
        with torch.no_grad():
            lp = torch.stack([F.log_softmax(m(x), dim=2) for m in members]
                             ).mean(dim=0)[0].cpu().numpy()
        raw = greedy(lp)
        d = decode_plate(raw)
        return d["plate"] or raw

    variants = ("recovered box", "box shifted 15%")
    stats = {v: {"n": 0, "exact": 0, "cer": 0.0} for v in variants}
    widths = []

    for r in rows:
        frame = cv2.imread(r["frame"])
        if frame is None:
            continue
        fh, fw = frame.shape[:2]
        x, y, bw, bh = r["box"]
        gt = r["text"]
        widths.append(bw)

        for name in variants:
            dy = int(0.15 * bh) if "shifted" in name else 0
            y1, y2 = max(0, y + dy), min(fh, y + bh + dy)
            x1, x2 = max(0, x), min(fw, x + bw)
            if y2 <= y1 or x2 <= x1:
                continue
            got = read(frame[y1:y2, x1:x2])
            s = stats[name]
            s["n"] += 1
            s["exact"] += (got == gt)
            s["cer"] += lev(got, gt) / max(1, len(gt))

    print(f"{'box':<20}{'n':>5}{'exact':>9}{'CER':>9}")
    print("-" * 44)
    for name in variants:
        s = stats[name]
        if not s["n"]:
            continue
        print(f"{name:<20}{s['n']:>5}{100*s['exact']/s['n']:>8.1f}%"
              f"{100*s['cer']/s['n']:>8.1f}%")

    if widths:
        print(f"\nrecovered box width: median {np.median(widths):.0f}px, "
              f"range {min(widths)}-{max(widths)}px")

    a, b = stats["recovered box"], stats["box shifted 15%"]
    if a["n"] and b["n"]:
        gap = 100 * (a["exact"] / a["n"] - b["exact"] / b["n"])
        print(f"""
The recovered boxes read {gap:+.1f} points better than the same boxes shifted
down by 15% of their height. A clear positive gap means the boxes are on the
plates and are safe to train on. A gap near zero would mean the recogniser is
reading something that survives being moved — which would mean the boxes are
not tightly on the plate, and training on them would teach the wrong target.""")
    return 0


if __name__ == "__main__":
    sys.exit(main())
