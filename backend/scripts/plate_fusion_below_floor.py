"""Can multi-frame fusion recover plates BELOW the readable floor?

Multi-frame fusion was measured on this footage before and lost: 34.5% exact
against 48.2% for simply picking the best view. That test ran on human-framed
crops, nearly all of them already wider than the 70px floor — plates a single
frame already reads. Fusion had no information to add there, and the median
stack cost sharpness.

The band where it could matter was never tested. Below 70px exact match is
0.0%, and that band is roughly 96% of live traffic. One frame there is
genuinely unreadable; eight frames with real sub-pixel offsets between them
carry more total information than any one of them, and that is the only
mechanism available that can add detail rather than interpolate it. If fusion
works anywhere, it works here.

Two experiments:

  A  CONTROLLED   take vehicles whose views are comfortably readable, shrink
                  every view to a chosen sub-floor width, and compare best
                  single view against fusion. Ground truth is certain and the
                  degradation is known exactly, so this isolates the mechanism.

  B  REAL         vehicles whose plates are natively below the floor, fused
                  from their own corpus frames. Harder and noisier, but it is
                  the actual production case.

A is run first because if fusion cannot help when the degradation is clean and
controlled, it will not help on real footage either, and B's noise would make
that ambiguous rather than obvious.

Run:  python -m backend.scripts.plate_fusion_below_floor
      python -m backend.scripts.plate_fusion_below_floor --width 55
"""
from __future__ import annotations

import argparse
import json
import random
import sys
from collections import defaultdict
from pathlib import Path

import cv2
import numpy as np
import torch
import torch.nn.functional as F

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from backend.scripts.indian_plate_grammar import decode_plate       # noqa: E402
from backend.scripts.plate_multiframe_fusion import fuse            # noqa: E402
from backend.scripts.plate_width_gate_eval import greedy, lev       # noqa: E402
from backend.scripts.train_plate_recognizer import CHARS, STOI      # noqa: E402

REAL = ROOT / "data" / "plate_real"
MODELS = ROOT / "models" / "plate_recognizer"


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--width", type=int, default=50,
                    help="Simulated sub-floor native width for experiment A.")
    ap.add_argument("--min-views", type=int, default=4)
    args = ap.parse_args()

    truth = {}
    for line in (REAL / "verified_all.jsonl").open(encoding="utf-8"):
        v = json.loads(line)
        if v.get("text") and all(c in STOI for c in v["text"]):
            truth[(v["camera"], v["track"])] = v["text"]
    by_track = defaultdict(list)
    for line in (REAL / "labels.jsonl").open(encoding="utf-8"):
        r = json.loads(line)
        k = (r["camera"], r["track"])
        if k in truth:
            by_track[k].append(r["file"])

    kk = sorted(by_track)
    random.Random(1000).shuffle(kk)
    clean = [k for k in kk[:max(20, int(len(kk) * 0.20))]
             if len(by_track[k]) >= args.min_views]
    print(f"{len(clean)} holdout vehicles with >= {args.min_views} views\n",
          flush=True)

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

    def read(gray):
        x = cv2.resize(gray, (w, h), interpolation=cv2.INTER_AREA)
        x = torch.from_numpy(x).float().div(127.5).sub(1.0)[None, None].to(dev)
        with torch.no_grad():
            lp = torch.stack([F.log_softmax(m(x), dim=2) for m in members]
                             ).mean(dim=0)[0].cpu().numpy()
        return greedy(lp), float(np.exp(np.mean(np.max(lp, axis=1))))

    def finalise(raw):
        d = decode_plate(raw)
        return d["plate"] or raw

    def shrink(g, target_w):
        """Simulate a more distant vehicle, then restore display size.

        Downscale destroys the detail; the upscale afterwards only puts the
        crop back on a comparable canvas, exactly as the engine's resize to
        the model input does. No information returns.
        """
        th = max(4, int(round(g.shape[0] * target_w / max(g.shape[1], 1))))
        small = cv2.resize(g, (target_w, th), interpolation=cv2.INTER_AREA)
        return small

    variants = ("best single view", "fused (all views)")
    stats = {v: {"n": 0, "exact": 0, "cer": 0.0} for v in variants}
    n_fused_used = []

    for k in clean:
        gt = truth[k]
        views = []
        for f in by_track[k]:
            g = cv2.imread(str(REAL / "images" / f), cv2.IMREAD_GRAYSCALE)
            if g is not None and g.shape[1] >= args.width:
                views.append(shrink(g, args.width))
        if len(views) < args.min_views:
            continue

        best, bc = "", -1e9
        for v in views:
            t, c = read(v)
            if c > bc:
                bc, best = c, t
        got_single = finalise(best)

        try:
            fused, used = fuse(views)
        except Exception:
            continue
        n_fused_used.append(used)
        got_fused = finalise(read(fused)[0])

        for name, got in (("best single view", got_single),
                          ("fused (all views)", got_fused)):
            s = stats[name]
            s["n"] += 1
            s["exact"] += (got == gt)
            s["cer"] += lev(got, gt) / max(1, len(gt))

    print(f"A  CONTROLLED — every view shrunk to {args.width}px native "
          f"(floor is 70px)")
    print("=" * 62)
    print(f"{'method':<24}{'n':>5}{'exact':>9}{'CER':>9}")
    print("-" * 48)
    for name in variants:
        s = stats[name]
        if not s["n"]:
            continue
        print(f"{name:<24}{s['n']:>5}{100*s['exact']/s['n']:>8.1f}%"
              f"{100*s['cer']/s['n']:>8.1f}%")

    a, b = stats["best single view"], stats["fused (all views)"]
    if a["n"]:
        d_ex = 100 * (b["exact"] - a["exact"]) / a["n"]
        d_cer = 100 * (a["cer"] - b["cer"]) / a["n"]
        n = a["n"]
        p = a["exact"] / max(n, 1)
        se = 100 * (p * (1 - p) / max(n, 1)) ** 0.5
        print(f"\nfusion vs best single view: {d_ex:+.1f} points exact, "
              f"{d_cer:+.1f} points CER")
        print(f"median frames actually fused: "
              f"{np.median(n_fused_used) if n_fused_used else 0:.0f} "
              f"(rest failed the ECC correlation gate)")
        print(f"standard error about {se:.1f} points, so a gap under "
              f"{2*se:.0f} points is not established.")
        print("""
A clear positive gap is the only result that justifies building Layer 5 into
the pipeline. Fusion costs an ECC registration per frame per vehicle, which is
real latency in a 27-camera 24/7 loop, and it was already measured to LOSE
above the floor — so it would have to be applied only below it.""")
    return 0


if __name__ == "__main__":
    sys.exit(main())
