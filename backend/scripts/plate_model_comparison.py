"""What the production engine reads with, against what is on disk.

anpr_engine loads models/plate_recognizer/best.pt, which is a 32x128 CRNN.
Sitting beside it, unused, are final_m0/m1/m2.pt — three ensemble members
trained at 64x256.

That difference matters for a specific, measured reason. The recogniser's
errors are substitutions in the four-digit block (88.9% of correct-length
errors), between glyph pairs that differ only in fine strokes: 6 read as 5,
9 as 3, 8 as 0. Those strokes need pixels. A 32-pixel-tall input has each
character about 20 px high; at 64 it has 40, and the stroke that separates a
6 from a 5 stops being sub-pixel.

It also decides whether a wider crop is worth anything. At a 128 px model
input, every plate crop above 128 px wide is downscaled before the model sees
it — so the 100-140 px crops that measured 51.7% were being thrown away down
to 128 regardless.

Compared on the same 55 held-out vehicles, same split, same seed as every
other plate figure here:

    best.pt @ 32x128        what production reads with today
    final_m0 @ 64x256       one higher-resolution member
    m0+m1+m2 @ 64x256       the ensemble, averaged in log-probability space

Run:  python -m backend.scripts.plate_model_comparison
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
    BLANK, CHARS, CRNN, ITOS, STOI, ctc_decode,
)

REAL = ROOT / "data" / "plate_real"
MODELS = ROOT / "models" / "plate_recognizer"
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


def greedy(lp: np.ndarray) -> str:
    out, prev = [], -1
    for i in lp.argmax(axis=1):
        i = int(i)
        if i != prev and i != BLANK:
            out.append(ITOS.get(i, ""))
        prev = i
    return "".join(out)


def load(path: Path, dev):
    """Load a checkpoint and build the matching architecture."""
    ck = torch.load(path, map_location=dev, weights_only=False)
    h, w = int(ck.get("img_h", 32)), int(ck.get("img_w", 128))
    if h == 32:
        model = CRNN(len(CHARS) + 1)
    else:
        from backend.scripts.plate_final_model import PlateCRNN
        model = PlateCRNN(len(CHARS) + 1, img_h=h)
    model.load_state_dict(ck["model"])
    return model.to(dev).eval(), h, w


def logprobs(model, dev, gray: np.ndarray, h: int, w: int) -> np.ndarray:
    g = cv2.resize(gray, (w, h), interpolation=cv2.INTER_AREA)
    x = torch.from_numpy(g).float().div(127.5).sub(1.0)[None, None].to(dev)
    with torch.no_grad():
        return F.log_softmax(model(x), dim=-1)[0].cpu().numpy()


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

    prod, ph, pw = load(MODELS / "best.pt", dev)
    print(f"production model : best.pt  {ph}x{pw}")

    members = []
    for name in ("final_m0.pt", "final_m1.pt", "final_m2.pt"):
        p = MODELS / name
        if p.is_file():
            m, h, w = load(p, dev)
            members.append((name, m, h, w))
    if members:
        print(f"ensemble members : {len(members)} at "
              f"{members[0][2]}x{members[0][3]}\n", flush=True)

    stats = {k: {"n": 0, "exact": 0, "cer": 0.0}
             for k in ("prod", "single_hi", "ensemble", "prod_best_view",
                       "ensemble_best_view")}

    for k in test:
        gt = truth[k]
        imgs = []
        for f in by_track[k]:
            g = cv2.imread(str(REAL / "images" / f), cv2.IMREAD_GRAYSCALE)
            if g is not None:
                imgs.append(g)
        if not imgs:
            continue

        def finalise(raw: str) -> str:
            d = decode_plate(raw)
            return d["plate"] or raw

        reads = {}
        reads["prod"] = finalise(greedy(logprobs(prod, dev, imgs[0], ph, pw)))

        # Same first-view protocol, higher-resolution model.
        if members:
            _, m0, h0, w0 = members[0]
            reads["single_hi"] = finalise(greedy(logprobs(m0, dev, imgs[0], h0, w0)))

            # Ensemble: average log-probabilities of the members on the SAME
            # input. Unlike averaging across views, these share a timestep
            # grid by construction, so the average is well defined.
            stack = np.stack([logprobs(m, dev, imgs[0], h, w)
                              for _, m, h, w in members])
            reads["ensemble"] = finalise(greedy(stack.mean(axis=0)))

        # Best-view selection, which measured 49.1% on the production model,
        # applied to both so the two gains can be seen separately.
        def most_confident(model, h, w) -> str:
            best, best_conf = "", -1e9
            for im in imgs:
                lp = logprobs(model, dev, im, h, w)
                c = float(np.mean(np.max(lp, axis=1)))
                if c > best_conf:
                    best_conf, best = c, greedy(lp)
            return finalise(best)

        reads["prod_best_view"] = most_confident(prod, ph, pw)
        if members:
            confs = []
            for im in imgs:
                stack = np.stack([logprobs(m, dev, im, h, w)
                                  for _, m, h, w in members])
                mean_lp = stack.mean(axis=0)
                confs.append((float(np.mean(np.max(mean_lp, axis=1))), mean_lp))
            reads["ensemble_best_view"] = finalise(
                greedy(max(confs, key=lambda c: c[0])[1]))

        for name, got in reads.items():
            s = stats[name]
            s["n"] += 1
            s["exact"] += (got == gt)
            s["cer"] += lev(got, gt) / max(1, len(gt))

    print(f"{'configuration':<34}{'exact':>9}{'CER':>9}")
    print("-" * 54)
    labels = {
        "prod": "best.pt 32x128, first view",
        "single_hi": "final_m0 64x256, first view",
        "ensemble": "ensemble x3 64x256, first view",
        "prod_best_view": "best.pt 32x128, best view",
        "ensemble_best_view": "ensemble x3 64x256, best view",
    }
    for name in ("prod", "single_hi", "ensemble", "prod_best_view",
                 "ensemble_best_view"):
        s = stats[name]
        if not s["n"]:
            continue
        print(f"{labels[name]:<34}{100*s['exact']/s['n']:>8.1f}%"
              f"{100*s['cer']/s['n']:>8.1f}%")

    base = stats["prod"]
    best_name = max(stats, key=lambda n: stats[n]["exact"] if stats[n]["n"] else -1)
    if stats[best_name]["n"] and base["n"]:
        gain = 100 * (stats[best_name]["exact"] - base["exact"]) / base["n"]
        print(f"\nbest configuration is {labels[best_name]}: "
              f"{gain:+.1f} points against what production reads with today")

    print(f"""
{base['n']} plates, so one read is {100/max(base['n'],1):.1f} points. Treat
CER as the more sensitive column: it averages over every character rather
than over {base['n']} all-or-nothing outcomes.""")
    return 0


if __name__ == "__main__":
    sys.exit(main())
