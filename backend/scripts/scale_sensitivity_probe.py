"""backend/scripts/scale_sensitivity_probe.py — is the low-resolution loss a
TRAINING problem or an INFORMATION problem?

WHY THIS EXISTS
  The condition-stratified evaluation measured a steep accuracy curve against
  native plate width on real footage:

        40-70 px   33.3%
        70-90 px   57.5%
        >= 90 px   75.0%

  Two very different explanations fit that curve, and they imply opposite
  decisions:

    (a) INFORMATION — at 60 px the characters are simply not in the image any
        more. No retraining recovers what was never captured. The only lever is
        camera optics.
    (b) TRAINING — the information is there, but the recogniser was pre-trained
        on a corpus whose plates are much wider than the fleet's, so it never
        learned to read at that scale. Retraining on a corrected width
        distribution would recover real points.

  Choosing wrongly costs days of GPU time on a retrain that cannot help, or
  leaves an easy win on the table. This probe separates them.

METHOD
  Take synthetic plates, where the ground truth is exact and the image is
  clean — no mud, no motion blur, no compression beyond the generator's own.
  Render each one down to a series of native widths, then feed it through the
  same path a real crop takes (resize to the model's 64x256 input) and score
  the shipped ensemble.

  That isolates scale. Any drop is caused by resolution alone, because nothing
  else about the image changed.

WHAT IT ACTUALLY FOUND (2026-09-06) — READ THIS BEFORE THE TABLE
  The shipped ensemble scores near ZERO on clean synthetic plates at every
  width. That is not a scale result and must not be read as one.

  The control run settles it. Scored on the same 300 synthetic images and the
  same 300 real crops, with raw CTC and no grammar pass:

      best.pt      (synthetic-pretrained, 32x128)   synthetic 79.7%   real 25.7%
      final_m0.pt  (shipped, 64x256)                synthetic  0.7%   real 80.0%

  So the images are readable — the pretrained checkpoint reads them at 79.7%.
  The shipped ensemble simply is not the same model any more. Its real-crop
  fine-tune has moved it so far that nothing of the synthetic domain survives.

  THE CONSEQUENCE, which is the point of running this at all: correcting the
  synthetic corpus's width distribution changes the PRE-TRAINING stage, and the
  fine-tune then overwrites that stage's behaviour completely. Expect close to
  zero gain, for the same structural reason grammar beam search gained exactly
  zero — it improves a stage whose output is discarded downstream.

  The lever that acts on the stage that decides the outcome is more LABELLED
  REAL CROPS. The fine-tune currently has 251 training vehicles.

  This probe therefore cannot separate the information-vs-training question on
  the shipped model, because the shipped model does not read this domain. It is
  kept because the control run above is the evidence for the paragraph above.

USAGE
  python -m backend.scripts.scale_sensitivity_probe
  python -m backend.scripts.scale_sensitivity_probe --n 600
"""
from __future__ import annotations

import argparse
import json
import random
import statistics
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import cv2  # noqa: E402
import numpy as np  # noqa: E402
import torch  # noqa: E402

from backend.scripts.eval_shipped_recognizer import load_ensemble  # noqa: E402
from backend.scripts.indian_plate_grammar import decode_plate  # noqa: E402
from backend.scripts.plate_eval_clean import lev  # noqa: E402
from backend.scripts.plate_final_model import PlateCRNN  # noqa: E402
from backend.scripts.train_plate_recognizer import BLANK, ITOS  # noqa: E402

SYNTH = ROOT / "data" / "synth_plates"
# The bands the real evaluation reports, so the two tables can be read side by
# side, plus the corpus's own typical width as an upper reference point.
WIDTHS = [55, 80, 100, 130, 200]


def read_at_width(models, gray: np.ndarray, target_w: int, h: int, w: int, dev):
    """Downscale to `target_w` native, then take the normal inference path."""
    if target_w is not None:
        scale = target_w / max(gray.shape[1], 1)
        if scale < 1.0:
            small = cv2.resize(gray, (target_w, max(4, int(gray.shape[0] * scale))),
                               interpolation=cv2.INTER_AREA)
        else:
            small = gray
    else:
        small = gray
    x = cv2.resize(small, (w, h), interpolation=cv2.INTER_AREA)
    t = torch.from_numpy(x).float().div(127.5).sub(1.0)[None, None].to(dev)
    with torch.no_grad():
        probs = None
        for m in models:
            p = m(t).softmax(2)
            probs = p if probs is None else probs + p
        ids = (probs / len(models)).argmax(2)[0].tolist()
    out, prev = [], -1
    for k in ids:
        if k != prev and k != BLANK:
            out.append(ITOS.get(int(k), ""))
        prev = k
    raw = "".join(out)
    return decode_plate(raw)["plate"] or raw


def control_run(dev, n=300, seed=5) -> dict:
    """Score the pretrained checkpoint and the shipped one on BOTH domains.

    Without this, a near-zero row in the width table below reads as "the model
    cannot handle small plates" when the real cause is that it no longer reads
    this domain at all. The control is what makes the table interpretable.
    """
    from backend.scripts.train_plate_recognizer import CHARS as _CH

    rng = random.Random(seed)
    syn = [json.loads(l) for l in (SYNTH / "labels.jsonl").open(encoding="utf-8")]
    syn_s = rng.sample(syn, min(n, len(syn)))

    verified = [json.loads(l) for l
                in (ROOT / "data/plate_real/verified_all.jsonl").open(encoding="utf-8")]
    truth = {(v["camera"], v["track"]): v["text"] for v in verified if v.get("text")}
    real_items = []
    for line in (ROOT / "data/plate_real/labels.jsonl").open(encoding="utf-8"):
        r = json.loads(line)
        if (r["camera"], r["track"]) in truth:
            real_items.append({"file": r["file"],
                               "text": truth[(r["camera"], r["track"])]})
    real_s = rng.sample(real_items, min(n, len(real_items)))

    def _score(model, items, h, w, root):
        ex = 0
        for it in items:
            g = cv2.imread(str(root / it["file"]), cv2.IMREAD_GRAYSCALE)
            if g is None:
                continue
            x = cv2.resize(g, (w, h), interpolation=cv2.INTER_AREA)
            t = torch.from_numpy(x).float().div(127.5).sub(1.0)[None, None].to(dev)
            with torch.no_grad():
                ids = model(t).softmax(2).argmax(2)[0].tolist()
            o, prev = [], -1
            for k in ids:
                if k != prev and k != BLANK:
                    o.append(ITOS.get(int(k), ""))
                prev = k
            ex += "".join(o) == it["text"]
        return ex / max(len(items), 1)

    print("CONTROL — same images, two checkpoints, raw CTC (no grammar)")
    print("  %-30s %11s %10s" % ("checkpoint", "synthetic", "real"))
    out = {}
    for name in ("best.pt", "final_m0.pt"):
        p = ROOT / "models" / "plate_recognizer" / name
        if not p.is_file():
            continue
        ck = torch.load(p, map_location=dev, weights_only=False)
        h, w = int(ck.get("img_h", 32)), int(ck.get("img_w", 128))
        m = PlateCRNN(len(_CH) + 1, img_h=h).to(dev)
        m.load_state_dict(ck["model"], strict=False)
        m.eval()
        s_syn = _score(m, syn_s, h, w, SYNTH / "images")
        s_real = _score(m, real_s, h, w, ROOT / "data/plate_real/images")
        print("  %-30s %10.1f%% %9.1f%%"
              % ("%s (%dx%d)" % (name, h, w), s_syn * 100, s_real * 100))
        out[name] = {"synthetic": s_syn, "real": s_real, "input": [h, w]}
        del m
        if dev == "cuda":
            torch.cuda.empty_cache()
    print("  n = %d synthetic, %d real (real sample includes trained vehicles,\n"
          "  so its figure is optimistic — the held-out number is 59.0%%)\n"
          % (len(syn_s), len(real_s)))
    return out


def main() -> int:
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--n", type=int, default=500)
    ap.add_argument("--seed", type=int, default=11)
    ap.add_argument("--out", default="reports/scale_sensitivity_20260906.json")
    args = ap.parse_args()

    rows = [json.loads(l) for l in (SYNTH / "labels.jsonl").open(encoding="utf-8")]

    # First question, answerable without the model: how wide IS the training
    # corpus, really? The claim of a train/serve mismatch rests on this.
    rw = sorted(r["render_w"] for r in rows if r.get("render_w"))
    print("TRAINING CORPUS WIDTH DISTRIBUTION")
    print("  n = %d   min %d   p10 %d   median %d   p90 %d   max %d"
          % (len(rw), rw[0], rw[len(rw) // 10], rw[len(rw) // 2],
             rw[len(rw) * 9 // 10], rw[-1]))
    for lo, hi in ((0, 40), (40, 70), (70, 90), (90, 10 ** 9)):
        k = sum(1 for x in rw if lo <= x < hi)
        print("    %-9s %6d  %5.1f%%"
              % ("%d-%dpx" % (lo, hi) if hi < 10 ** 9 else "90px+", k,
                 100.0 * k / len(rw)))
    print()

    dev = "cuda" if torch.cuda.is_available() else "cpu"
    control = control_run(dev)

    rng = random.Random(args.seed)
    sample = rng.sample(rows, min(args.n, len(rows)))
    models, h, w, names = load_ensemble(dev)
    print("ensemble : %d members @ %dx%d on %s" % (len(models), h, w, dev))
    print("sample   : %d clean synthetic plates\n" % len(sample), flush=True)

    stats = {tw: {"n": 0, "exact": 0, "lev": 0, "chars": 0} for tw in WIDTHS}
    stats["native"] = {"n": 0, "exact": 0, "lev": 0, "chars": 0}
    t0 = time.perf_counter()

    for i, r in enumerate(sample, 1):
        g = cv2.imread(str(SYNTH / "images" / r["file"]), cv2.IMREAD_GRAYSCALE)
        if g is None:
            continue
        truth = r["text"]
        for key in ["native"] + WIDTHS:
            tw = None if key == "native" else key
            # Only downscale. Asking what a 200px render looks like at 200px
            # when it was rendered at 134 would measure upsampling, not scale.
            if tw is not None and tw > g.shape[1]:
                continue
            pred = read_at_width(models, g, tw, h, w, dev)
            s = stats[key]
            s["n"] += 1
            s["exact"] += 1 if pred == truth else 0
            s["lev"] += lev(pred, truth)
            s["chars"] += len(truth)
        if i % 100 == 0:
            print("  %d/%d" % (i, len(sample)), flush=True)

    elapsed = time.perf_counter() - t0
    print("\n" + "=" * 62)
    print("  CLEAN SYNTHETIC PLATES, SCORED AT CONTROLLED NATIVE WIDTH")
    print("=" * 62)
    print("  %-12s %6s %10s %9s" % ("native width", "n", "exact", "CER"))
    out = {}
    for key in ["native"] + WIDTHS:
        s = stats[key]
        if not s["n"]:
            continue
        ex = s["exact"] / s["n"]
        cer = s["lev"] / max(s["chars"], 1)
        label = "as rendered" if key == "native" else "%d px" % key
        print("  %-12s %6d %9.1f%% %8.1f%%" % (label, s["n"], ex * 100, cer * 100))
        out[str(key)] = {"n": s["n"], "exact": ex, "cer": cer}
    print("=" * 62)
    print("  Clean images: exact ground truth, no mud, motion or road grime.")
    print("  Any loss here is caused by resolution alone.")
    print("\n  Real footage, same model, for comparison:")
    print("    40-70px  33.3%   70-90px  57.5%   90px+  75.0%")
    print("\n  %.0fs" % elapsed)

    p = Path(args.out)
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(json.dumps({
        "generated": time.strftime("%Y-%m-%dT%H:%M:%S"),
        "ensemble": names, "device": dev, "sample": len(sample),
        "corpus_width": {"n": len(rw), "median": rw[len(rw) // 2],
                         "p10": rw[len(rw) // 10], "p90": rw[len(rw) * 9 // 10]},
        "control": control,
        "by_width": out,
        "real_footage_reference": {"40_70": 0.333, "70_90": 0.575, "90plus": 0.750},
    }, indent=2), encoding="utf-8")
    print("  written -> %s" % p)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
