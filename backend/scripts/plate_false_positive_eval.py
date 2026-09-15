"""Does anything currently separate a real plate from a grille or a badge?

Two live-feed reads on cam21 were confidently wrong in a way that matters more
than a character error: the crop was not a plate at all. One was a car grille
read as "GJ01T0670" at 0.93 confidence; one was a dashboard badge read as
"GJ04V0626" at 0.95. Both were graded "reliable". Neither the plate detector's
own aspect-ratio filter (1.8-6.0, meant to reject "square logos") nor the
recogniser's confidence caught them — a grille slat and a rectangular badge
are exactly the right shape to pass an aspect-ratio gate, and the CRNN turns
out to emit a confident, grammar-valid string on textured non-text input
rather than a low-confidence one. GJ01T0670 and GJ04V0626 both parse as
syntactically valid Indian plates, so decode_plate() would not catch them
either.

This measures whether the plate DETECTOR's own confidence — the thing that
found the box in the first place — separates real plates from exactly this
failure mode, using real image content rather than synthetic negatives.

NEGATIVES ARE REAL, NOT SYNTHETIC
  For each of the 1,965 recovered plate boxes (frame + exact box + ground
  truth text), a negative is the SAME box shape shifted vertically up by 1-2x
  its own height — off the plate and onto whatever sits directly above it on
  a real vehicle: grille, bumper trim, a badge, a headlight edge. That is the
  same region class that produced both live false positives, cut from the
  same domain of images the positives come from, so it is a fair test of the
  detector rather than an easy one.

WHAT THIS DECIDES
  If detector confidence cleanly separates the two classes, raising
  PLATE_DET_CONF is the fix and this reports the value. If it does not, the
  detector cannot tell a plate from a grille by confidence alone and a
  different signal is needed (this also measures whether the recogniser's own
  confidence does any better, having already seen it fail twice).

Run:  python -m backend.scripts.plate_false_positive_eval
"""
from __future__ import annotations

import json
import random
import sys
from pathlib import Path

import cv2
import numpy as np

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
if str(ROOT / "backend") not in sys.path:
    sys.path.insert(0, str(ROOT / "backend"))

BOXES = ROOT / "data" / "plate_detect_v2" / "boxes.jsonl"
PLATE_DET_IMGSZ = 320


def negative_box(x: int, y: int, w: int, h: int, frame_h: int,
                 rng: random.Random) -> tuple[int, int, int, int] | None:
    """Same shape, shifted up onto the grille/badge/bumper region above it."""
    shift = rng.choice([1.2, 1.6, 2.0]) * h
    ny = int(y - shift)
    if ny < 0:
        return None
    return (x, ny, w, h)


def main() -> int:
    from ultralytics import YOLO
    from services.anpr_engine import get_anpr_engine

    engine = get_anpr_engine()
    pdet = engine.plate_detector
    if pdet is None:
        print("primary plate detector not loaded")
        return 1

    rows = [json.loads(l) for l in BOXES.open(encoding="utf-8")]
    rng = random.Random(11)
    rng.shuffle(rows)
    rows = rows[:250]

    pos_scores, neg_scores = [], []
    pos_ocr, neg_ocr = [], []
    neg_examples = []

    for r in rows:
        frame = cv2.imread(r["frame"])
        if frame is None:
            continue
        fh, fw = frame.shape[:2]
        x, y, w, h = r["box"]

        # ── Positive: the real plate, re-detected exactly as production does ──
        pad_w, pad_h = int(0.3 * w), int(0.5 * h)
        vx1, vy1 = max(0, x - pad_w), max(0, y - pad_h)
        vx2, vy2 = min(fw, x + w + pad_w), min(fh, y + h + pad_h)
        veh = frame[vy1:vy2, vx1:vx2]
        if veh.size == 0:
            continue
        res = pdet.predict(veh, imgsz=PLATE_DET_IMGSZ, conf=0.05,
                           verbose=False, device=engine.device)
        boxes = res[0].boxes
        if boxes is not None and len(boxes):
            pos_scores.append(float(boxes.conf.max()))
            i = int(boxes.conf.argmax())
            px1, py1, px2, py2 = (int(v) for v in boxes.xyxy[i].cpu().numpy())
            crop = veh[max(0, py1):py2, max(0, px1):px2]
            if crop.size:
                _, conf = engine._crnn_infer(cv2.resize(
                    cv2.cvtColor(crop, cv2.COLOR_BGR2GRAY)
                    if crop.ndim == 3 else crop,
                    (engine._rec_hw[1], engine._rec_hw[0])))
                pos_ocr.append(conf)

        # ── Negative: same shape, shifted onto grille/badge/bumper territory ──
        nb = negative_box(x, y, w, h, fh, rng)
        if nb is None:
            continue
        nx, ny, nw, nh = nb
        nvx1, nvy1 = max(0, nx - pad_w), max(0, ny - pad_h)
        nvx2, nvy2 = min(fw, nx + nw + pad_w), min(fh, ny + nh + pad_h)
        nveh = frame[nvy1:nvy2, nvx1:nvx2]
        if nveh.size == 0:
            continue
        nres = pdet.predict(nveh, imgsz=PLATE_DET_IMGSZ, conf=0.05,
                            verbose=False, device=engine.device)
        nboxes = nres[0].boxes
        if nboxes is not None and len(nboxes):
            sc = float(nboxes.conf.max())
            neg_scores.append(sc)
            i = int(nboxes.conf.argmax())
            px1, py1, px2, py2 = (int(v) for v in nboxes.xyxy[i].cpu().numpy())
            ncrop = nveh[max(0, py1):py2, max(0, px1):px2]
            if ncrop.size:
                text, conf = engine._crnn_infer(cv2.resize(
                    cv2.cvtColor(ncrop, cv2.COLOR_BGR2GRAY)
                    if ncrop.ndim == 3 else ncrop,
                    (engine._rec_hw[1], engine._rec_hw[0])))
                neg_ocr.append(conf)
                if sc > 0.25 and len(neg_examples) < 6:
                    neg_examples.append((r["camera"], sc, conf, text))

    print(f"{len(pos_scores)} positive (real plate) detections")
    print(f"{len(neg_scores)} negative (grille/badge/bumper) detections\n")

    def summarize(name, pos, neg):
        pos = np.array(pos) if pos else np.array([0.0])
        neg = np.array(neg) if neg else np.array([0.0])
        print(f"{name}")
        print(f"  positives: median={np.median(pos):.3f} "
              f"p10={np.percentile(pos,10):.3f}")
        print(f"  negatives: median={np.median(neg):.3f} "
              f"p90={np.percentile(neg,90):.3f}")
        for thresh in (0.25, 0.5, 0.75, 0.9):
            tp = float((pos >= thresh).mean()) if len(pos) else 0.0
            fp = float((neg >= thresh).mean()) if len(neg) else 0.0
            print(f"  threshold {thresh:.2f}: keeps {100*tp:.1f}% of real "
                  f"plates, lets through {100*fp:.1f}% of grille/badge crops")
        print()

    summarize("DETECTOR CONFIDENCE (the box-finding score)", pos_scores, neg_scores)
    summarize("OCR CONFIDENCE (what the recogniser reports)", pos_ocr, neg_ocr)

    if neg_examples:
        print("negatives the detector was confident about (score > 0.25):")
        for cam, sc, oc, txt in neg_examples:
            print(f"  {cam}  det_conf={sc:.2f}  ocr_conf={oc:.2f}  read={txt!r}")

    print("""
If detector confidence separates the classes cleanly (positives bunched high,
negatives bunched low), raising PLATE_DET_CONF is the fix — read off the
threshold above that keeps most real plates while rejecting most grille/badge
crops. If OCR confidence does not separate them (as the two live cases
suggest), that confirms the recogniser cannot be used to catch this failure
mode after the fact — it has to be stopped at detection.""")
    return 0


if __name__ == "__main__":
    sys.exit(main())
