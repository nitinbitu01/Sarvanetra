"""What does recognition do as plates get BIGGER? The optics ROI question.

An optical upgrade — telephoto lens, one lane per camera — raises plate width
at the capture zone. Its business case assumes recognition improves with that
width. Whether it does is an empirical question about this recogniser, and it
decides whether the spend buys 82% or 60%.

An earlier pass here reported a correlation of +0.044 between width and error
count and called it "essentially zero". That was measured on one best view per
vehicle (74 samples) and was dragged around by a 4-sample bucket above 140px.
The per-bucket exact-match rates over the same data pointed the other way —
30.8% at 70-90px rising to 61.5% at 110-140px — so the two summaries
disagreed and the correlation was the less trustworthy of them.

This resolves it with every view of every holdout vehicle rather than one,
which multiplies the sample roughly tenfold and puts real confidence
intervals on each bucket. Two questions:

  1  does exact match keep climbing with width, or flatten after the floor
  2  what should be expected at the 150-250px an optical upgrade would deliver

Extrapolation beyond the widest bucket present is stated as extrapolation.
The honest limit of this data is the widest bucket that has enough samples to
carry a confidence interval.

Run:  python -m backend.scripts.plate_width_response_curve
"""
from __future__ import annotations

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
from backend.scripts.plate_width_gate_eval import greedy, lev       # noqa: E402
from backend.scripts.train_plate_recognizer import CHARS, STOI      # noqa: E402

REAL = ROOT / "data" / "plate_real"
MODELS = ROOT / "models" / "plate_recognizer"

BANDS = [(40, 70), (70, 90), (90, 110), (110, 130), (130, 160), (160, 10_000)]


def main() -> int:
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
    clean = kk[:max(20, int(len(kk) * 0.20))]

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

    def read(g):
        x = cv2.resize(g, (w, h), interpolation=cv2.INTER_AREA)
        x = torch.from_numpy(x).float().div(127.5).sub(1.0)[None, None].to(dev)
        with torch.no_grad():
            lp = torch.stack([F.log_softmax(m(x), dim=2) for m in members]
                             ).mean(dim=0)[0].cpu().numpy()
        return greedy(lp)

    widths, oks, nerrs, lens = [], [], [], []
    for k in clean:
        gt = truth[k]
        for f in by_track[k]:
            g = cv2.imread(str(REAL / "images" / f), cv2.IMREAD_GRAYSCALE)
            if g is None:
                continue
            raw = read(g)
            d = decode_plate(raw)
            got = d["plate"] or raw
            ne = (sum(1 for a, b in zip(got, gt) if a != b)
                  if len(got) == len(gt) else lev(got, gt))
            widths.append(g.shape[1])
            oks.append(got == gt)
            nerrs.append(ne)
            lens.append(len(gt))

    W = np.array(widths)
    OK = np.array(oks, dtype=float)
    NE = np.array(nerrs, dtype=float)
    LN = np.array(lens, dtype=float)
    print(f"{len(W)} single-view reads from {len(clean)} holdout vehicles\n")

    print(f"{'plate width':<14}{'n':>6}{'exact':>9}{'95% CI':>15}{'char acc':>11}")
    print("-" * 58)
    for lo, hi in BANDS:
        s = (W >= lo) & (W < hi)
        n = int(s.sum())
        if n < 3:
            continue
        p = float(OK[s].mean())
        se = (p * (1 - p) / n) ** 0.5
        ca = 1.0 - NE[s].sum() / max(LN[s].sum(), 1)
        lab = f"{lo}-{hi}px" if hi < 10_000 else f"{lo}px+"
        ci = f"{100*max(0,p-1.96*se):.0f}-{100*min(1,p+1.96*se):.0f}%"
        print(f"{lab:<14}{n:>6}{100*p:>8.1f}%{ci:>15}{100*ca:>10.1f}%")

    big = W >= 70
    print(f"\ncorrelation, width vs exact (all widths) : "
          f"{np.corrcoef(W, OK)[0, 1]:+.3f}")
    print(f"correlation, width vs exact (>=70px only): "
          f"{np.corrcoef(W[big], OK[big])[0, 1]:+.3f}")

    # Logistic fit on the samples above the floor, used only to say whether
    # the slope is distinguishable from flat — not to predict far beyond the
    # data, which this sample cannot support.
    x = W[big].astype(float)
    y = OK[big]
    if len(x) > 20 and 0 < y.mean() < 1:
        xs = (x - x.mean()) / max(x.std(), 1e-6)
        b0, b1 = 0.0, 0.0
        for _ in range(200):
            z = b0 + b1 * xs
            pr = 1.0 / (1.0 + np.exp(-z))
            g0 = np.sum(y - pr)
            g1 = np.sum((y - pr) * xs)
            b0 += 0.01 * g0
            b1 += 0.01 * g1
        se_b1 = 1.0 / max(np.sqrt(np.sum(pr * (1 - pr) * xs ** 2)), 1e-6)
        print(f"logistic slope on width (>=70px): {b1:+.3f} "
              f"+/- {1.96*se_b1:.3f} (95%)")
        verdict = ("width still helps above the floor"
                   if b1 - 1.96 * se_b1 > 0 else
                   "slope not distinguishable from flat")
        print(f"  -> {verdict}")
        for target in (150, 200, 250):
            zt = b0 + b1 * ((target - x.mean()) / max(x.std(), 1e-6))
            print(f"  extrapolated exact @ {target}px: "
                  f"{100/(1+np.exp(-zt)):.0f}%  (EXTRAPOLATION - "
                  f"widest bucket measured is {int(x.max())}px)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
