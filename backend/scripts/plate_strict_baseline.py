"""The strict number, on plates the models never saw, with a real sample.

Two earlier figures in this work rest on 14 plates, where one read is 7 points
and nothing separates 50% from 57%. This uses the ensemble's own test split —
83 vehicles it never trained on — which is the largest genuinely clean holdout
available.

Four configurations, so each change can be attributed:

    32x128, first view        what production read with before today
    32x128, best view         view selection alone
    64x256 ensemble, first    the model change alone
    64x256 ensemble, best     both

"Best view" means the frame the model is most confident about, out of the
dozen the tracker already collects per vehicle. The pipeline processes all of
them either way, so choosing between them costs nothing.

All configurations read raw crops resized with INTER_AREA, which is what both
models were trained on — the engine's enhancement chain was measured to cost
accuracy on the 64x256 members and is no longer applied to them.

Run:  python -m backend.scripts.plate_strict_baseline
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
    BLANK, CHARS, CRNN, ITOS, STOI,
)

REAL = ROOT / "data" / "plate_real"
MODELS = ROOT / "models" / "plate_recognizer"


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


def greedy(lp: np.ndarray) -> str:
    out, prev = [], -1
    for i in lp.argmax(axis=1):
        i = int(i)
        if i != prev and i != BLANK:
            out.append(ITOS.get(i, ""))
        prev = i
    return "".join(out)


def main() -> int:
    truth = {}
    for l in (REAL / "verified_all.jsonl").open(encoding="utf-8"):
        v = json.loads(l)
        if v.get("text") and all(c in STOI for c in v["text"]):
            truth[(v["camera"], v["track"])] = v["text"]
    by_track = defaultdict(list)
    for l in (REAL / "labels.jsonl").open(encoding="utf-8"):
        r = json.loads(l)
        k = (r["camera"], r["track"])
        if k in truth:
            by_track[k].append(r["file"])

    keys = sorted(by_track)
    kk = list(keys)
    random.Random(1000).shuffle(kk)      # plate_final_model's own split
    clean = kk[:max(20, int(len(kk) * 0.20))]
    print(f"{len(clean)} vehicles, none of them in any model's training set\n",
          flush=True)

    dev = "cuda" if torch.cuda.is_available() else "cpu"

    ck = torch.load(MODELS / "best.pt", map_location=dev, weights_only=False)
    old = CRNN(len(ck.get("chars", CHARS)) + 1)
    old.load_state_dict(ck["model"])
    old = old.to(dev).eval()
    old_hw = (int(ck.get("img_h", 32)), int(ck.get("img_w", 128)))

    members, hw = [], None
    for i in range(3):
        p = MODELS / f"final_m{i}.pt"
        if not p.is_file():
            continue
        c = torch.load(p, map_location=dev, weights_only=False)
        from backend.scripts.plate_final_model import PlateCRNN
        m = PlateCRNN(len(c.get("chars", CHARS)) + 1, img_h=int(c["img_h"]))
        m.load_state_dict(c["model"])
        members.append(m.to(dev).eval())
        hw = (int(c["img_h"]), int(c["img_w"]))

    def lp_single(model, g, h, w):
        x = cv2.resize(g, (w, h), interpolation=cv2.INTER_AREA)
        x = torch.from_numpy(x).float().div(127.5).sub(1.0)[None, None].to(dev)
        with torch.no_grad():
            return F.log_softmax(model(x), dim=2)[0].cpu().numpy()

    def lp_ens(g, h, w):
        x = cv2.resize(g, (w, h), interpolation=cv2.INTER_AREA)
        x = torch.from_numpy(x).float().div(127.5).sub(1.0)[None, None].to(dev)
        with torch.no_grad():
            return torch.stack([F.log_softmax(m(x), dim=2) for m in members]
                               ).mean(dim=0)[0].cpu().numpy()

    configs = ["32x128 first", "32x128 best", "ens 64x256 first",
               "ens 64x256 best"]
    stats = {c: {"n": 0, "exact": 0, "cer": 0.0} for c in configs}

    for k in clean:
        gt = truth[k]
        imgs = []
        for f in by_track[k]:
            g = cv2.imread(str(REAL / "images" / f), cv2.IMREAD_GRAYSCALE)
            if g is not None:
                imgs.append(g)
        if not imgs:
            continue

        def finalise(raw):
            d = decode_plate(raw)
            return d["plate"] or raw

        reads = {}
        reads["32x128 first"] = finalise(greedy(lp_single(old, imgs[0], *old_hw)))

        best, bc = "", -1e9
        for im in imgs:
            lp = lp_single(old, im, *old_hw)
            c = float(np.mean(np.max(lp, axis=1)))
            if c > bc:
                bc, best = c, greedy(lp)
        reads["32x128 best"] = finalise(best)

        if members:
            reads["ens 64x256 first"] = finalise(greedy(lp_ens(imgs[0], *hw)))
            best, bc = "", -1e9
            for im in imgs:
                lp = lp_ens(im, *hw)
                c = float(np.mean(np.max(lp, axis=1)))
                if c > bc:
                    bc, best = c, greedy(lp)
            reads["ens 64x256 best"] = finalise(best)

        for name, got in reads.items():
            s = stats[name]
            s["n"] += 1
            s["exact"] += (got == gt)
            s["cer"] += lev(got, gt) / max(1, len(gt))

    print(f"{'configuration':<22}{'exact':>9}{'CER':>9}{'vs baseline':>14}")
    print("-" * 56)
    base = None
    for name in configs:
        s = stats[name]
        if not s["n"]:
            continue
        ex = 100 * s["exact"] / s["n"]
        cer = 100 * s["cer"] / s["n"]
        if base is None:
            base = ex
            delta = ""
        else:
            delta = f"{ex - base:+.1f} pts"
        print(f"{name:<22}{ex:>8.1f}%{cer:>8.1f}%{delta:>14}")

    n = stats[configs[0]]["n"]
    # Standard error of a proportion, not 2/sqrt(n): at p≈0.45 and n=83 that
    # is about 5.5 points, so two configurations need to differ by roughly 11
    # before the gap means anything on exact match.
    p = (base or 45.0) / 100.0
    se = 100 * (p * (1 - p) / max(n, 1)) ** 0.5
    print(f"""
{n} vehicles, so one read is {100/max(n,1):.1f} points. The standard error on
exact match here is about {se:.1f} points, so a gap under roughly {2*se:.0f}
points is not established by this sample. CER settles sooner because it
averages over about {n*10} characters rather than {n} all-or-nothing outcomes.

Strict exact match on 10 characters is the target. To reach 80% the CER has
to fall to roughly 2.5%, since character errors become close to independent
as the model improves.""")
    return 0


if __name__ == "__main__":
    sys.exit(main())
