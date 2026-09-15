"""backend/scripts/eval_shipped_recognizer.py — what does the model that
actually ships read, on real Gujarat footage it has never seen?

WHY THIS SCRIPT EXISTS
  `reports/anpr_validation/anpr_validation_report.md` is withdrawn. Its
  per-tier recognition figures (84.50% Tier 1, 2.37% CER) were measured on a
  set whose Tier 1 contains *no real footage at all* — 96.3% of that benchmark
  is synthetic — so they are not this system's accuracy on Gujarat CCTV and
  must not be quoted as such.

  Every other route to the number retrains first: `plate_eval_clean.py` and
  `plate_final_model.py --train-final` both run gradient updates and then score
  a model they just built. That answers "how well can this architecture do",
  which is a different question from "what does the checkpoint on disk do".

  This script trains nothing. It loads `models/plate_recognizer/final_m*.pt` —
  the exact ensemble the live pipeline uses — and scores it once.

WHY THE NUMBER IS HONEST
  The split is `plate_final_model.split(by_track, 1000)`, the same call and the
  same seed `train_final` used when those checkpoints were produced. So the
  test vehicles here are precisely the ones the shipped ensemble
    * never took a gradient step on, and
    * was never selected against (that was the val split's job).

  Splitting BY TRACK, not by crop, matters: one vehicle contributes several
  frames, and splitting by crop would put the same car on both sides of the
  line and inflate the result.

  Scoring is per VEHICLE, not per crop — multi-frame voting across a track is
  what the live pipeline does, so scoring single crops would measure something
  the product never does.

CONDITION STRATIFICATION
  The brief this system is built against asks for accuracy "across diverse
  real-world conditions". A single headline number cannot answer that, so every
  scored vehicle is also tagged by the four conditions that are measurable from
  its own crops — resolution, blur, lighting and tilt — and accuracy is reported
  per slice.

  Tags are computed from the pixels, not from a label file, so they cannot
  drift out of sync with the images and need no annotation pass. Each vehicle
  contributes several crops; the median across them is used, because one
  blurred frame in eleven does not make the vehicle a blurred case.

USAGE
  python -m backend.scripts.eval_shipped_recognizer
  python -m backend.scripts.eval_shipped_recognizer --out reports/xyz.json
"""
from __future__ import annotations

import argparse
import json
import statistics
import sys
import time
from collections import defaultdict
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import cv2  # noqa: E402
import numpy as np  # noqa: E402
import torch  # noqa: E402

from backend.scripts.indian_plate_grammar import decode_plate  # noqa: E402
from backend.scripts.plate_beam_decode import beam_decode  # noqa: E402
from backend.scripts.plate_eval_clean import _vote, lev  # noqa: E402
from backend.scripts.plate_final_model import (  # noqa: E402
    OUT_DIR, REAL, PlateCRNN, load_data, read_vehicle, split)
from backend.scripts.train_plate_recognizer import BLANK, CHARS, ITOS  # noqa: E402


def read_vehicle_ex(models, files, h, w, dev, decoder="greedy"):
    """Voted read of one vehicle, returning the confidence as well.

    `plate_final_model.read_vehicle` is the production path and returns only
    the string. This mirrors it exactly for `decoder="greedy"` — same
    member-probability averaging, same collapse, same `_vote` — and adds two
    things the shipped path does not expose: a confidence, needed for the
    selective-prediction curve, and a beam-decode arm for A/B testing.

    BEAM ARM
      `beam_decode` applies its own log-softmax. Feeding it log(mean_probs) is
      therefore exact rather than approximate: log_softmax(log p) = log p when
      p already sums to one, so the ensemble average is preserved instead of
      being re-normalised into something else.
    """
    reads, confs = [], []
    with torch.no_grad():
        for f in files:
            g = cv2.imread(str(REAL / "images" / f), cv2.IMREAD_GRAYSCALE)
            if g is None:
                continue
            g = cv2.resize(g, (w, h), interpolation=cv2.INTER_AREA)
            x = torch.from_numpy(g).float().div(127.5).sub(1.0)[None, None].to(dev)
            probs = None
            for m in models:
                p = m(x).softmax(2)
                probs = p if probs is None else probs + p
            probs = (probs / len(models))[0]           # (T, C)

            top, idx = probs.max(1)
            keep = idx != BLANK
            confs.append(float(top[keep].mean()) if bool(keep.any()) else 0.0)

            if decoder == "beam":
                lp = torch.log(probs.clamp_min(1e-12)).cpu().numpy()
                cands = beam_decode(lp, ITOS, blank=BLANK, beam_width=16, topk=6)
                reads.append(cands[0][0] if cands else "")
            else:
                ids = probs.argmax(1).tolist()
                s, prev = [], -1
                for k in ids:
                    if k != prev and k != BLANK:
                        s.append(ITOS.get(int(k), ""))
                    prev = k
                reads.append("".join(s))

    if not reads:
        return "", 0.0
    return _vote(reads), (sum(confs) / len(confs) if confs else 0.0)


def _band(value, edges, labels):
    """Map a value into a named band. `edges` are the upper bounds."""
    for edge, label in zip(edges, labels):
        if value < edge:
            return label
    return labels[-1]


def _tilt_deg(gray: np.ndarray) -> float:
    """Tilt of the glyph mass, via minAreaRect on high-contrast pixels.

    Deliberately not the engine's `_plate_quad`: importing ANPREngine pulls in
    two YOLO models for a measurement that needs none. This is the same
    minAreaRect technique on the same input, and it only has to rank crops into
    bands, not drive a warp.
    """
    try:
        g = cv2.GaussianBlur(gray, (3, 3), 0)
        _, th = cv2.threshold(g, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
        pts = cv2.findNonZero(255 - th)
        if pts is None or len(pts) < 20:
            return 0.0
        angle = cv2.minAreaRect(pts)[-1]
        # OpenCV reports (-90, 0]; fold onto a signed deviation from horizontal.
        if angle < -45:
            angle += 90
        return abs(float(angle))
    except Exception:
        return 0.0


def condition_tags(files) -> dict:
    """Resolution, blur, lighting and tilt for one vehicle, from its crops."""
    widths, blurs, brights, tilts = [], [], [], []
    for f in files:
        g = cv2.imread(str(REAL / "images" / f), cv2.IMREAD_GRAYSCALE)
        if g is None:
            continue
        widths.append(g.shape[1])
        blurs.append(float(cv2.Laplacian(g, cv2.CV_64F).var()))
        brights.append(float(g.mean()))
        tilts.append(_tilt_deg(g))
    if not widths:
        return {}

    med = statistics.median
    w, b, br, t = med(widths), med(blurs), med(brights), med(tilts)
    return {
        "native_w": round(w, 1),
        "blur_var": round(b, 1),
        "brightness": round(br, 1),
        "tilt_deg": round(t, 2),
        # Width bands follow this project's own measured accuracy curve:
        # ~0% below 40px, 17.1% at 40-70px, 44.8% at 70-90px.
        "width_band": _band(w, [40, 70, 90],
                            ["w_under40", "w_40_70", "w_70_90", "w_90plus"]),
        "blur_band": _band(b, [100, 500],
                           ["blurry", "moderate", "sharp"]),
        # Brightness-derived, and named that way rather than "night": this is
        # the crop's own exposure, not a clock reading.
        "lighting_band": _band(br, [60, 110, 170],
                               ["dark", "dim", "normal", "bright"]),
        "tilt_band": _band(t, [2, 5, 15],
                           ["tilt_0_2", "tilt_2_5", "tilt_5_15", "tilt_15plus"]),
    }


def load_ensemble(dev):
    """Load every shipped member. Fails loudly rather than scoring a subset.

    Silently scoring 2 of 3 members would report a number for a model that is
    not the one running in production, which is the exact failure mode this
    script was written to correct.
    """
    paths = sorted(OUT_DIR.glob("final_m*.pt"))
    if not paths:
        raise SystemExit(f"no final_m*.pt in {OUT_DIR} — nothing ships yet")

    members, h, w = [], None, None
    for p in paths:
        ck = torch.load(p, map_location=dev, weights_only=False)
        ih, iw = int(ck.get("img_h", 32)), int(ck.get("img_w", 128))
        if h is None:
            h, w = ih, iw
        elif (ih, iw) != (h, w):
            raise SystemExit(
                f"{p.name} expects {ih}x{iw} but the first member wants {h}x{w}"
                " — these checkpoints are not one ensemble")
        m = PlateCRNN(len(CHARS) + 1, img_h=ih).to(dev)
        missing, unexpected = m.load_state_dict(ck["model"], strict=False)
        if missing or unexpected:
            raise SystemExit(
                f"{p.name} did not load cleanly "
                f"({len(missing)} missing, {len(unexpected)} unexpected "
                "tensors). A partially loaded model would score as a handicap "
                "of the checkpoint rather than of the design.")
        m.eval()
        members.append(m)
    return members, h, w, [p.name for p in paths]


def main() -> int:
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--out", default="reports/shipped_recognizer_eval.json")
    ap.add_argument("--seed", type=int, default=1000,
                    help="split seed; must match the one train_final used (1000)")
    ap.add_argument("--decoder", choices=("greedy", "beam"), default="greedy",
                    help="greedy reproduces the shipped path; beam runs "
                         "grammar-constrained CTC prefix search for A/B")
    args = ap.parse_args()

    dev = "cuda" if torch.cuda.is_available() else "cpu"
    members, h, w, names = load_ensemble(dev)

    truth, by_track = load_data()
    test_k, val_k, train_k = split(by_track, args.seed)
    items = [{"key": k,
              "files": [c["file"] for c in by_track[k]],
              "text": truth[k]} for k in test_k]

    print(f"device        : {dev}")
    print(f"ensemble      : {len(members)} members ({', '.join(names)}) @ {h}x{w}")
    print(f"vehicles      : train {len(train_k)} / val {len(val_k)} / "
          f"test {len(test_k)}")
    print(f"test crops    : {sum(len(i['files']) for i in items)}\n", flush=True)

    exact = 0
    lev_sum = char_total = 0
    rows = []
    t0 = time.perf_counter()
    for n, it in enumerate(items, 1):
        raw, read_conf = read_vehicle_ex(members, it["files"], h, w, dev,
                                         args.decoder)
        # The grammar pass is part of the shipping pipeline, so it is part of
        # the measurement. `decode_plate` returning nothing means it rejected
        # the read, and the raw string is what the pipeline would fall back to.
        fin = decode_plate(raw)["plate"] or raw if raw else ""
        ok = fin == it["text"]
        exact += ok
        lev_sum += lev(fin, it["text"])
        char_total += len(it["text"])
        tags = condition_tags(it["files"])
        rows.append({"camera": it["key"][0], "track": it["key"][1],
                     "truth": it["text"], "raw": raw, "final": fin,
                     "correct": bool(ok), "frames": len(it["files"]),
                     "confidence": round(read_conf, 4),
                     "conditions": tags})
        if n % 10 == 0:
            print(f"  {n}/{len(items)} vehicles", flush=True)
    elapsed = time.perf_counter() - t0

    # ── Per-condition slices ────────────────────────────────────────────────
    # Each vehicle lands in exactly one band per axis, so the slices within an
    # axis partition the set and their counts must sum back to the total.
    slices: dict = defaultdict(lambda: {"n": 0, "exact": 0, "lev": 0, "chars": 0})
    for r in rows:
        c = r.get("conditions") or {}
        d = lev(r["final"], r["truth"])
        for axis in ("width_band", "blur_band", "lighting_band", "tilt_band"):
            key = c.get(axis)
            if not key:
                continue
            s = slices[(axis, key)]
            s["n"] += 1
            s["exact"] += 1 if r["correct"] else 0
            s["lev"] += d
            s["chars"] += len(r["truth"])

    n = max(len(items), 1)
    acc = exact / n
    cer = lev_sum / max(char_total, 1)
    # A read that is wrong by one character is still recoverable by the
    # watchlist matcher, so it is worth separating from a total miss.
    within1 = sum(1 for r in rows if lev(r["final"], r["truth"]) <= 1) / n

    print("\n" + "=" * 62)
    print("  SHIPPED RECOGNISER — HELD-OUT REAL GUJARAT FOOTAGE")
    print("=" * 62)
    print(f"  Vehicles scored          : {len(items)}")
    print(f"  Exact match (voted)      : {acc*100:.1f}%  ({exact}/{len(items)})")
    print(f"  Within 1 character       : {within1*100:.1f}%")
    print(f"  Character error rate     : {cer*100:.1f}%")
    print(f"  Character accuracy       : {(1-cer)*100:.1f}%")
    print(f"  Wall clock               : {elapsed:.1f}s "
          f"({elapsed/n*1000:.0f} ms/vehicle, {dev})")
    print("=" * 62)
    print("  Real footage only. No synthetic image is scored here, and no")
    print("  test vehicle was trained on or used to select a checkpoint.")

    # ── Condition-stratified report ─────────────────────────────────────────
    AXES = [("width_band", "RESOLUTION (native plate width)",
             ["w_under40", "w_40_70", "w_70_90", "w_90plus"]),
            ("blur_band", "BLUR (Laplacian variance)",
             ["blurry", "moderate", "sharp"]),
            ("lighting_band", "LIGHTING (crop brightness)",
             ["dark", "dim", "normal", "bright"]),
            ("tilt_band", "ANGLE (glyph-mass tilt)",
             ["tilt_0_2", "tilt_2_5", "tilt_5_15", "tilt_15plus"])]

    by_condition: dict = {}
    for axis, title, order in AXES:
        print("\n" + "-" * 62)
        print(f"  {title}")
        print("-" * 62)
        print(f"  {'slice':<14}{'n':>5}{'exact':>10}{'CER':>9}   note")
        for key in order:
            s = slices.get((axis, key))
            if not s or not s["n"]:
                print(f"  {key:<14}{0:>5}{'—':>10}{'—':>9}   no vehicles in band")
                by_condition[key] = {"n": 0}
                continue
            ex = s["exact"] / s["n"]
            cer = s["lev"] / max(s["chars"], 1)
            # A slice this small cannot separate itself from the overall rate;
            # say so on the line rather than letting it be quoted as a finding.
            note = "too few to compare" if s["n"] < 10 else ""
            print(f"  {key:<14}{s['n']:>5}{ex*100:>9.1f}%{cer*100:>8.1f}%   {note}")
            by_condition[key] = {"n": s["n"], "exact": ex, "cer": cer,
                                 "reliable": s["n"] >= 10}

    print("\n" + "-" * 62)
    print("  Bands are computed from the crops themselves (median across a")
    print("  vehicle's frames), so they cannot drift from the images. Slices")
    print("  under 10 vehicles are reported but not comparable.")

    # ── Selective prediction ────────────────────────────────────────────────
    # An ANPR system that answers "not read" is more useful to an investigator
    # than one that guesses wrong, because a wrong plate sends a patrol after
    # the wrong vehicle. Every vendor quotes accuracy against a read rate for
    # this reason. Sweeping the confidence floor gives the honest pair.
    print("\n" + "-" * 62)
    print("  SELECTIVE PREDICTION — accuracy against coverage")
    print("-" * 62)
    print("  %-11s %8s %10s %11s" % ("min conf", "answered", "coverage", "accuracy"))
    curve = []
    for thr in [0.0, 0.50, 0.60, 0.70, 0.80, 0.85, 0.90, 0.93, 0.95, 0.97]:
        kept = [r for r in rows if r.get("confidence", 0.0) >= thr]
        if len(kept) < 5:
            continue
        a = sum(1 for r in kept if r["correct"]) / len(kept)
        cov = len(kept) / n
        curve.append({"min_conf": thr, "answered": len(kept),
                      "coverage": cov, "accuracy": a})
        print("  %-11.2f %8d %9.1f%% %10.1f%%" % (thr, len(kept), cov * 100, a * 100))
    print("-" * 62)
    hit80 = next((c for c in curve if c["accuracy"] >= 0.80), None)
    if hit80:
        print("  Reaches 80%% accuracy at confidence >= %.2f, answering %.0f%%"
              % (hit80["min_conf"], hit80["coverage"] * 100))
        print("  of vehicles. State BOTH numbers — an accuracy without its")
        print("  coverage is not a claim, it is a selection.")
    else:
        print("  No confidence floor reaches 80%% accuracy on this set.")

    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps({
        "generated": time.strftime("%Y-%m-%dT%H:%M:%S"),
        "device": dev, "ensemble": names, "input": [h, w],
        "split_seed": args.seed,
        "vehicles": {"train": len(train_k), "val": len(val_k),
                     "test": len(test_k)},
        "test_crops": sum(len(i["files"]) for i in items),
        "exact_match": acc, "within_1_char": within1,
        "cer": cer, "char_accuracy": 1 - cer,
        "ms_per_vehicle": elapsed / n * 1000,
        "decoder": args.decoder,
        "by_condition": by_condition,
        "selective_prediction": curve,
        "results": rows,
    }, indent=2), encoding="utf-8")
    print(f"\n  written -> {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
