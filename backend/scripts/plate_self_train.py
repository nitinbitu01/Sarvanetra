"""backend/scripts/plate_self_train.py — put the 89,000 unlabelled crops to work,
using the model's own high-confidence reads as extra training data.

WHY THIS IS NOT THE PSEUDO-LABEL ATTEMPT THAT ALREADY FAILED
  An earlier attempt pseudo-labelled with EasyOCR and produced labels that were
  about 15% correct. It failed for a specific reason worth restating: EasyOCR's
  errors are SYSTEMATIC. It misreads the same glyph the same way on every frame,
  so twenty-five frames of one plate agree on the same wrong string and
  multi-frame voting confirms it. 'GJ1CO9007' carried agreement 1.00 and was
  wrong.

  Two things are different now. The source is the fine-tuned recogniser rather
  than off-the-shelf OCR, and that recogniser's confidence is calibrated - at
  the 0.95 cut it is 82.5% precise, where a few hours ago the same curve was
  flat at 56% and confidence carried no information at all. A label source that
  is 82% correct on the subset it commits to is a different proposition from
  one that is 15% correct everywhere.

  Systematic error has not gone away, though, so it is guarded against rather
  than assumed absent - see the gates below.

WHAT THE EXTRA DATA CAN AND CANNOT TEACH
  A pseudo-label is only assigned where the model already reads confidently, so
  it does not teach the model strings it gets wrong. What it does add is
  VARIETY: thousands of new vehicles, plates, angles, lighting and blur
  conditions the labelled set never covered, attached to strings that are
  usually right. That is worth having, and it is the honest reason to expect a
  modest gain rather than a large one.

FOUR GATES, EACH CLOSING A DIFFERENT FAILURE
  confidence    the read must clear a threshold measured on held-out data.
  agreement     the frames of one vehicle must agree with the voted string;
                a plate read three different ways across four frames is not
                confidently anything.
  grammar       the string must be a legal Indian plate. This is weak evidence
                on its own but it costs nothing and removes obvious garbage.
  novelty       the plate must not duplicate a string already in the training
                set. A pseudo-label that repeats a known plate adds a copy of
                something already learned and inflates its weight.

  The pseudo-labelled crops are also weighted BELOW the human labels in
  training, because they are not the same quality of evidence and should not
  be allowed to outvote the labels that are known to be right.

USAGE
  python -m backend.scripts.plate_self_train --conf 0.95 --max 6000
  python -m backend.scripts.plate_self_train --dry-run
"""
from __future__ import annotations

import argparse
import json
import random
from collections import Counter, defaultdict
from pathlib import Path

import cv2
import torch

from backend.scripts.indian_plate_grammar import decode_plate
from backend.scripts.plate_beam_decode import complete
from backend.scripts.plate_eval_clean import _vote
from backend.scripts.train_plate_recognizer import (BLANK, CHARS, CRNN, IMG_H,
                                                    IMG_W, ITOS, STOI,
                                                    ctc_decode)

CORPUS = Path("output/plate_corpus")
REAL = Path("data/plate_real")
OUT = REAL / "pseudo_labels.jsonl"
# The production detector, not the v3 checkpoint this script was written
# against. v4 was retrained 2026-09-02 and finds 2.6x more plates on live
# footage; with v3 a dry run rejected 226 of 300 vehicles for "too few frames",
# which was the detector missing plates rather than the plates being absent.
DET_W = Path("models/plate_detector/plate_v4_small.pt")
DET_W_LEGACY = Path("runs/detect/runs/plate_v3/recovered/weights/best.pt")
CKPT = Path("models/plate_recognizer/finetuned.pt")

PLATE_CAPABLE = {"CAM_06", "CAM_07", "CAM_08", "CAM_09", "CAM_10",
                 "CAM_11", "CAM_18", "CAM_21", "CAM_27"}


def read_with_conf(models, g, dev, img_h=IMG_H, img_w=IMG_W):
    """Read one crop with the ensemble, averaging member probabilities.

    WHY THIS TAKES A LIST NOW
      The original read with `finetuned.pt` alone. Measured on 300 real crops
      (reports/scale_sensitivity_20260906.txt), that checkpoint scores **25.7%**
      where the shipped 3-member ensemble scores **80.0%**. A pseudo-label
      source that is wrong three times in four does not produce training data,
      it produces noise with a confidence attached — which is precisely how the
      earlier EasyOCR attempt failed. Labels are generated with the best reader
      available, not the one that happened to be wired here first.
    """
    g = cv2.resize(g, (img_w, img_h), interpolation=cv2.INTER_AREA)
    x = torch.from_numpy(g).float().div(127.5).sub(1.0)[None, None].to(dev)
    probs = None
    for m in models:
        p = m(x).softmax(2)
        probs = p if probs is None else probs + p
    probs = probs / len(models)

    ids = probs.argmax(2)[0].tolist()
    out, prev = [], -1
    for k in ids:
        if k != prev and k != BLANK:
            out.append(ITOS.get(int(k), ""))
        prev = k
    text = "".join(out)

    top, idx = probs[0].max(1)
    keep = idx != BLANK
    conf = float(top[keep].mean()) if bool(keep.any()) else 0.0
    return text, conf


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--conf", type=float, default=0.95,
                    help="Confidence floor. 0.95 was 82.5% precise on held-out "
                         "plates; lowering it trades label quality for volume.")
    ap.add_argument("--agree", type=float, default=0.75,
                    help="Fraction of a vehicle's frames that must match the "
                         "voted string.")
    ap.add_argument("--min-frames", type=int, default=3,
                    help="Vehicles with fewer frames cannot demonstrate "
                         "agreement, so they are skipped.")
    ap.add_argument("--max", type=int, default=6000, help="Crops to emit.")
    ap.add_argument("--max-vehicles", type=int, default=4000,
                    help="Vehicles to examine.")
    ap.add_argument("--dry-run", action="store_true")
    ap.add_argument("--legacy-reader", action="store_true",
                    help="Generate labels with finetuned.pt instead of the "
                         "shipped ensemble. Only for reproducing the original "
                         "run — it reads real crops at 25.7%% against 80.0%%.")
    args = ap.parse_args()

    known = {}
    for line in (REAL / "verified_all.jsonl").open(encoding="utf-8"):
        v = json.loads(line)
        known[(v["camera"], v["track"])] = v["text"]
    known_strings = set(known.values())
    print(f"human-labelled vehicles : {len(known)}")
    print(f"distinct plate strings  : {len(known_strings)}")

    tracks = defaultdict(list)
    for line in (CORPUS / "manifest.jsonl").open(encoding="utf-8"):
        r = json.loads(line)
        if (r["camera"], r["track"]) in known:
            continue
        if r["camera"] not in PLATE_CAPABLE:
            continue
        tracks[(r["camera"], r["clip"], r["track"])].append(r)
    keys = sorted(tracks)
    random.Random(17).shuffle(keys)
    keys = keys[:args.max_vehicles]
    print(f"unlabelled vehicles     : {len(keys)}\n", flush=True)

    dev = "cuda" if torch.cuda.is_available() else "cpu"
    from ultralytics import YOLO
    det_w = DET_W_LEGACY if args.legacy_reader else DET_W
    det = YOLO(str(det_w))
    print(f"detector                : {det_w.name}")

    if args.legacy_reader:
        ck = torch.load(CKPT, map_location=dev, weights_only=False)
        model = CRNN(len(CHARS) + 1).to(dev)
        model.load_state_dict(ck["model"])
        model.eval()
        models, r_h, r_w = [model], IMG_H, IMG_W
        print(f"reader                  : {CKPT.name} "
              f"({r_h}x{r_w}) — 25.7% on real crops, kept only for "
              f"reproducing the original run")
    else:
        from backend.scripts.eval_shipped_recognizer import load_ensemble
        models, r_h, r_w, names = load_ensemble(dev)
        print(f"reader                  : shipped ensemble "
              f"{', '.join(names)} ({r_h}x{r_w}) — 80.0% on real crops")

    img_dir = REAL / "images"
    img_dir.mkdir(parents=True, exist_ok=True)
    out_rows = []
    reject = Counter()
    seen_strings = set()
    scanned = 0

    with torch.no_grad():
        for key in keys:
            if len(out_rows) >= args.max:
                break
            cam, clip, track = key
            frames = tracks[key]
            frames.sort(key=lambda r: -(r.get("sharpness", 0) ** 0.5
                                        * r.get("plate_px_est", 0)))
            scanned += 1
            crops, reads, confs = [], [], []
            for fr in frames[:8]:
                img = cv2.imread(fr["path"])
                if img is None:
                    continue
                res = det.predict(img, conf=0.30, verbose=False)[0]
                if not len(res.boxes):
                    continue
                h, w = img.shape[:2]
                best = None
                for b in res.boxes.xyxy.cpu().numpy():
                    x1, y1, x2, y2 = b
                    bw, bh = x2 - x1, y2 - y1
                    if bw < 36 or bh < 10 or not (1.8 <= bw / max(bh, 1) <= 7.5):
                        continue
                    if best is None or bw > best[0]:
                        best = (bw, (x1, y1, x2, y2))
                if best is None:
                    continue
                x1, y1, x2, y2 = best[1]
                mx, my = (x2 - x1) * 0.08, (y2 - y1) * 0.20
                c = img[int(max(0, y1 - my)):int(min(h, y2 + my)),
                        int(max(0, x1 - mx)):int(min(w, x2 + mx))]
                if c.size == 0 or c.shape[1] < 30:
                    continue
                g = cv2.cvtColor(c, cv2.COLOR_BGR2GRAY)
                t, cf = read_with_conf(models, g, dev, r_h, r_w)
                crops.append((c, Path(fr["path"]).stem))
                reads.append(t)
                confs.append(cf)

            if len(reads) < args.min_frames:
                reject["too few frames"] += 1
                continue
            voted = _vote(reads)
            d = decode_plate(voted)
            final = d["plate"] or voted
            mean_conf = sum(confs) / len(confs)
            agree = sum(r == voted for r in reads) / len(reads)

            if mean_conf < args.conf:
                reject["low confidence"] += 1
                continue
            if agree < args.agree:
                reject["frames disagree"] += 1
                continue
            if not complete(final) or not all(c in STOI for c in final):
                reject["fails grammar"] += 1
                continue
            if final in known_strings or final in seen_strings:
                reject["duplicate plate"] += 1
                continue
            seen_strings.add(final)
            for c, stem in crops:
                name = f"ps_{cam}_t{track}_{stem}.jpg"
                out_rows.append({"file": name, "text": final, "camera": cam,
                                 "track": track, "pseudo": True,
                                 "conf": round(mean_conf, 3),
                                 "agree": round(agree, 2), "_img": c})
            if len(out_rows) % 500 < len(crops):
                print(f"  {len(out_rows)} crops from {len(seen_strings)} "
                      f"vehicles ({scanned} scanned)", flush=True)

    print("\n" + "=" * 60)
    print(f"vehicles scanned  : {scanned}")
    print(f"accepted vehicles : {len(seen_strings)}")
    print(f"pseudo crops      : {len(out_rows)}")
    print(f"acceptance rate   : {len(seen_strings)/max(scanned,1)*100:.1f}%")
    print("\nrejections:")
    for k, n in reject.most_common():
        print(f"  {k:<18} {n:>6}")
    print("=" * 60)

    if args.dry_run:
        print("\n--dry-run: nothing written.")
        return

    with OUT.open("w", encoding="utf-8") as f:
        for r in out_rows:
            img = r.pop("_img")
            cv2.imwrite(str(img_dir / r["file"]), img,
                        [cv2.IMWRITE_JPEG_QUALITY, 96])
            f.write(json.dumps(r) + "\n")
    print(f"\nwrote {OUT}")
    print("\nThese are NOT ground truth. They are kept in a separate file so")
    print("they can be weighted below the human labels during training, and")
    print("removed entirely if the ablation shows they do not help.")


if __name__ == "__main__":
    main()
