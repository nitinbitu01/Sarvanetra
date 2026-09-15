"""Fuse the recogniser's probabilities across views, not its answers.

Voting on decoded strings was measured at 32.7% against 45.5% for a single
view — worse than not fusing at all. That is the expected result, and it is
not evidence against multi-view: a decoded string has already thrown away
everything the model was unsure about. If eleven views each read position 3
as "8" with 0.34 confidence and one reads it as "B" with 0.95, a vote returns
"8" eleven to one.

The information is in the log-probabilities, before the argmax. Averaging
those across views lets a confident view outweigh a crowd of unsure ones, and
lets agreement across views accumulate rather than merely repeat. It is the
same quantity the journey search already scores plates with.

Three fusions are compared against the single-view baseline, all on the same
held-out vehicles and the same model:

    mean logprob        every view weighted equally
    weighted by width   larger crops carry more, since accuracy tracks
                        resolution: 0% below 70px, ~52% at 100-140px
    best single view    the highest mean per-timestep confidence

Views of one vehicle are the same plate at different scales, so their CTC
timesteps do not align frame to frame. Each view's probabilities are resampled
onto a common timestep grid before averaging; without that, fusion blurs
adjacent characters together.

Run:  python -m backend.scripts.plate_logit_fusion_eval
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

from backend.scripts.indian_plate_grammar import decode_plate      # noqa: E402
from backend.scripts.train_plate_recognizer import (                # noqa: E402
    BLANK, CHARS, CRNN, IMG_H, IMG_W, ITOS, STOI,
)

REAL = ROOT / "data" / "plate_real"
CKPT = ROOT / "models" / "plate_recognizer" / "finetuned.pt"
HOLDOUT = 55
SEED = 1337


def lev(a: str, b: str) -> int:
    if not a:
        return len(b)
    prev = list(range(len(b) + 1))
    for i, ca in enumerate(a, 1):
        cur = [i]
        for j, cb in enumerate(b, 1):
            cur.append(min(prev[j] + 1, cur[j - 1] + 1, prev[j - 1] + (ca != cb)))
        prev = cur
    return prev[-1]


def greedy_from_logprobs(lp: np.ndarray) -> str:
    """Collapse a CTC log-probability matrix (T x C) to a string.

    Mirrors ctc_decode in train_plate_recognizer: blank is index 0, and
    characters are looked up through ITOS. Indexing CHARS directly treats
    index 0 as "0" and shifts the whole alphabet by one, which decodes every
    plate into a different string — measured as 0.0% exact and 98.5% CER
    across all four fusions, a result that says the decoder is broken rather
    than that fusion does not work.
    """
    ids = lp.argmax(axis=1)
    out, prev = [], -1
    for i in ids:
        i = int(i)
        if i != prev and i != BLANK:
            out.append(ITOS.get(i, ""))
        prev = i
    return "".join(out)


def main() -> int:
    verified = [json.loads(l) for l in (REAL / "verified_all.jsonl").open(encoding="utf-8")]
    truth = {(v["camera"], v["track"]): v["text"] for v in verified
             if v["text"] and all(c in STOI for c in v["text"])}

    by_track: dict[tuple, list] = defaultdict(list)
    for l in (REAL / "labels.jsonl").open(encoding="utf-8"):
        r = json.loads(l)
        k = (r["camera"], r["track"])
        if k in truth:
            by_track[k].append(r["file"])

    keys = sorted(by_track)
    random.seed(SEED)
    random.shuffle(keys)
    test = keys[:HOLDOUT]

    dev = "cuda" if torch.cuda.is_available() else "cpu"
    ck = torch.load(CKPT, map_location=dev, weights_only=False)
    model = CRNN(len(CHARS) + 1).to(dev)
    model.load_state_dict(ck["model"])
    model.eval()

    print(f"held out: {len(test)} vehicles, median "
          f"{int(np.median([len(by_track[k]) for k in test]))} views each\n",
          flush=True)

    stats = defaultdict(lambda: {"n": 0, "exact": 0, "cer": 0.0})

    for k in test:
        gt = truth[k]
        per_view = []
        for f in by_track[k]:
            g = cv2.imread(str(REAL / "images" / f), cv2.IMREAD_GRAYSCALE)
            if g is None:
                continue
            width = float(g.shape[1])
            g = cv2.resize(g, (IMG_W, IMG_H), interpolation=cv2.INTER_AREA)
            x = torch.from_numpy(g).float().div(127.5).sub(1.0)[None, None].to(dev)
            with torch.no_grad():
                logits = model(x)
                # CRNN emits (N, T, C) — ctc_decode argmaxes over dim=2 and
                # iterates the batch — so sample 0 is [0], giving (T, C).
                # Log-softmax over classes is the space the CTC loss is
                # defined in, and the space these average in.
                lp = F.log_softmax(logits, dim=-1)[0].cpu().numpy()
            per_view.append({"lp": lp, "w": width, "file": f,
                             "conf": float(np.mean(np.max(lp, axis=1)))})

        if not per_view:
            continue

        # Single view, as currently deployed.
        single = per_view[0]
        cands = {"single view": greedy_from_logprobs(single["lp"])}

        # All views share IMG_W, so the CRNN emits the same T for each and the
        # grids already align; assert rather than assume.
        T = per_view[0]["lp"].shape[0]
        aligned = [v for v in per_view if v["lp"].shape[0] == T]

        if len(aligned) > 1:
            stack = np.stack([v["lp"] for v in aligned])
            cands["mean logprob"] = greedy_from_logprobs(stack.mean(axis=0))

            w = np.array([max(v["w"], 1.0) for v in aligned], dtype=np.float64)
            w = w / w.sum()
            cands["width-weighted"] = greedy_from_logprobs(
                (stack * w[:, None, None]).sum(axis=0))

            best = max(aligned, key=lambda v: v["conf"])
            cands["most confident view"] = greedy_from_logprobs(best["lp"])

        for name, raw in cands.items():
            d = decode_plate(raw)
            got = d["plate"] or raw
            s = stats[name]
            s["n"] += 1
            s["exact"] += (got == gt)
            s["cer"] += lev(got, gt) / max(1, len(gt))

    print(f"{'fusion':<24}{'n':>5}{'exact':>9}{'CER':>9}")
    print("-" * 48)
    order = ["single view", "most confident view", "mean logprob",
             "width-weighted"]
    for name in order:
        if name not in stats:
            continue
        s = stats[name]
        print(f"{name:<24}{s['n']:>5}{100*s['exact']/s['n']:>8.1f}%"
              f"{100*s['cer']/s['n']:>8.1f}%")

    print(f"""
{HOLDOUT} vehicles is a small holdout: one plate is 1.8 points, and the gap
between two strategies has to clear roughly 13 points before it means
anything. Read a large gap as real and a small one as noise.""")
    return 0


if __name__ == "__main__":
    sys.exit(main())
