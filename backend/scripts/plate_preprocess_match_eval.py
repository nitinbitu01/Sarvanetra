"""Does the live preprocessing match what the recogniser was trained on?

The 64x256 ensemble members were trained on raw grayscale crops: read the
file, optionally jitter brightness/blur/shift as augmentation, resize with
INTER_AREA, normalise. Nothing else — plate_final_model's dataset applies no
CLAHE and no denoising.

The engine feeds them something different. PlatePreprocessorV2 runs a
lighting-dependent chain first — CLAHE at clipLimit 2.0 plus
fastNlMeansDenoising by day, gamma lift plus stronger CLAHE plus adaptive
threshold plus morphology by night — and only then resizes.

A model reads what it was trained to read. Enhancement that was never in the
training distribution changes the pixel statistics it learned, and the loss
from that is silent: no error, no warning, just worse reads. The 57.1% figure
measured for this ensemble used raw crops, matching training; production does
not, so that number may not be what production gets.

This scores the same plates four ways to separate the two effects:

    raw resize            exactly what training did
    day pipeline          CLAHE + denoise, what the engine applies by day
    night pipeline        the stronger night chain
    engine auto           whatever the engine's own lighting classifier picks

If raw wins clearly, the engine should hand the recogniser raw crops and keep
enhancement for the detector and for display only.

Run:  python -m backend.scripts.plate_preprocess_match_eval
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
from backend.services.plate_utils_v2 import PlatePreprocessorV2     # noqa: E402

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

    # Only vehicles the ensemble never trained on. plate_final_model splits
    # with seed 1000 and keeps keys[:n_test] as its test set.
    keys = sorted(by_track)
    kk = list(keys)
    random.Random(1000).shuffle(kk)
    n_test = max(20, int(len(kk) * 0.20))
    clean = kk[:n_test]
    print(f"{len(clean)} vehicles the ensemble never saw\n", flush=True)

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
    if not members:
        print("no ensemble members found")
        return 1
    h, w = hw
    print(f"{len(members)} members at {h}x{w}\n", flush=True)

    pre = PlatePreprocessorV2(target_h=h, target_w=w)

    def read(gray_ready: np.ndarray) -> str:
        x = torch.from_numpy(gray_ready).float().div(127.5).sub(1.0)[None, None].to(dev)
        with torch.no_grad():
            lp = torch.stack([F.log_softmax(m(x), dim=2) for m in members]
                             ).mean(dim=0)[0].cpu().numpy()
        raw = greedy(lp)
        d = decode_plate(raw)
        return d["plate"] or raw

    variants = ("raw resize", "day pipeline", "night pipeline", "engine auto")
    stats = {v: {"n": 0, "exact": 0, "cer": 0.0} for v in variants}

    for k in clean:
        gt = truth[k]
        for f in by_track[k][:1]:      # first view, one protocol throughout
            g = cv2.imread(str(REAL / "images" / f), cv2.IMREAD_GRAYSCALE)
            if g is None:
                continue

            prepared = {
                "raw resize": cv2.resize(g, (w, h), interpolation=cv2.INTER_AREA),
                "day pipeline": pre.preprocess(g, lighting="day"),
                "night pipeline": pre.preprocess(g, lighting="night"),
                "engine auto": pre.preprocess(g, lighting="auto"),
            }
            for name, img in prepared.items():
                if img is None:
                    continue
                got = read(img)
                s = stats[name]
                s["n"] += 1
                s["exact"] += (got == gt)
                s["cer"] += lev(got, gt) / max(1, len(gt))

    print(f"{'preprocessing':<22}{'exact':>9}{'CER':>9}")
    print("-" * 42)
    for name in variants:
        s = stats[name]
        if not s["n"]:
            continue
        print(f"{name:<22}{100*s['exact']/s['n']:>8.1f}%{100*s['cer']/s['n']:>8.1f}%")

    raw = stats["raw resize"]
    auto = stats["engine auto"]
    if raw["n"] and auto["n"]:
        d_ex = 100 * (raw["exact"] - auto["exact"]) / raw["n"]
        d_cer = 100 * (auto["cer"] - raw["cer"]) / raw["n"]
        print(f"\nraw against what the engine currently applies: "
              f"{d_ex:+.1f} points exact, {d_cer:+.1f} points CER")
        if d_ex > 3 or d_cer > 2:
            print("\nThe enhancement is costing accuracy. It was tuned for the "
                  "32x128 model and is outside the 64x256 members' training "
                  "distribution; the recogniser should be given raw crops.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
