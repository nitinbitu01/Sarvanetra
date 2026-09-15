"""Does rectification help the CURRENT pipeline? (the old test does not say)

plate_rectify.py ships a --compare mode, and it reports full rectification
losing 9.1 points. That number cannot be applied to the pipeline as it stands
now, because --compare loads the old 32x128 single CRNN and splits with its
own seed. Since it was written, the recogniser became a 3-member 64x256
ensemble reading raw crops, the grammar decoder stopped deleting 'O', and
per-character cross-frame voting was added.

Two of those changes plausibly move the answer in opposite directions:

  64x256 doubles the vertical resolution, so a tilted baseline now smears
  across twice as many rows — geometry should matter MORE, favouring
  rectification.

  Per-character voting across views already absorbs errors that appear in
  only some frames, and a tilt varies frame to frame as the vehicle moves —
  so voting may already be recovering what rectification would fix, leaving
  it nothing to add.

Which dominates is not predictable, so it is measured, on the same seed-1000
holdout every other current number uses, at both single-view and after
voting. A technique is only adopted if it wins at the stage the engine
actually runs.

fixed/broke is reported alongside the net, because a net of zero can hide
five of each, and breaking a plate that already read correctly is worse than
failing to fix one that did not.

Run:  python -m backend.scripts.plate_rectify_current_eval
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
from backend.scripts.plate_rectify import rectify                   # noqa: E402
from backend.scripts.plate_width_gate_eval import greedy, lev       # noqa: E402
from backend.scripts.train_plate_recognizer import CHARS, STOI      # noqa: E402

REAL = ROOT / "data" / "plate_real"
MODELS = ROOT / "models" / "plate_recognizer"
MODES = ("none", "deskew", "full")


def char_vote(reads: list[dict]) -> str:
    if not reads:
        return ""
    lens = [len(r["t"]) for r in reads]
    target = max(set(lens), key=lens.count)
    valid = [r for r in reads if len(r["t"]) == target]
    out = []
    for pos in range(target):
        acc: dict[str, float] = defaultdict(float)
        for r in valid:
            acc[r["t"][pos]] += r["w"]
        out.append(max(acc.items(), key=lambda kv: kv[1])[0])
    return "".join(out)


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
    print(f"{len(clean)} holdout vehicles | {len(members)} members at {h}x{w}\n",
          flush=True)

    def read(g):
        x = cv2.resize(g, (w, h), interpolation=cv2.INTER_AREA)
        x = torch.from_numpy(x).float().div(127.5).sub(1.0)[None, None].to(dev)
        with torch.no_grad():
            lp = torch.stack([F.log_softmax(m(x), dim=2) for m in members]
                             ).mean(dim=0)[0].cpu().numpy()
        return greedy(lp), float(np.exp(np.mean(np.max(lp, axis=1))))

    single = {m: {"n": 0, "ok": 0} for m in MODES}
    voted = {m: {"n": 0, "ok": 0} for m in MODES}
    per_plate = {m: [] for m in MODES}
    changed = {m: 0 for m in MODES}

    for k in clean:
        gt = truth[k]
        views = []
        for f in by_track[k]:
            g = cv2.imread(str(REAL / "images" / f), cv2.IMREAD_GRAYSCALE)
            if g is not None and g.shape[1] >= 70:
                views.append(g)
        if not views:
            continue

        for mode in MODES:
            reads = []
            best, bc = "", -1e9
            for g in views:
                gg = g if mode == "none" else rectify(g, mode=mode)
                if mode != "none" and gg is not None and gg.shape != g.shape:
                    changed[mode] += 1
                t, c = read(gg if gg is not None else g)
                reads.append({"t": t, "w": c})
                if c > bc:
                    bc, best = c, t
            d = decode_plate(best)
            got_s = d["plate"] or best
            single[mode]["n"] += 1
            single[mode]["ok"] += (got_s == gt)

            v = char_vote(reads)
            dv = decode_plate(v)
            got_v = dv["plate"] or v
            voted[mode]["n"] += 1
            voted[mode]["ok"] += (got_v == gt)
            per_plate[mode].append(got_v == gt)

    print(f"{'mode':<10}{'single view':>14}{'+ char voting':>16}"
          f"{'crops altered':>16}")
    print("-" * 58)
    for mode in MODES:
        s, v = single[mode], voted[mode]
        alt = "-" if mode == "none" else str(changed[mode])
        print(f"{mode:<10}{100*s['ok']/max(s['n'],1):>13.1f}%"
              f"{100*v['ok']/max(v['n'],1):>15.1f}%{alt:>16}")

    base = per_plate["none"]
    print(f"\n{'mode':<10}{'net':>9}{'fixed':>8}{'broke':>8}   verdict")
    print("-" * 58)
    for mode in ("deskew", "full"):
        cur = per_plate[mode]
        fixed = sum(1 for b, c in zip(base, cur) if c and not b)
        broke = sum(1 for b, c in zip(base, cur) if b and not c)
        net = 100 * (sum(cur) - sum(base)) / max(len(base), 1)
        verdict = ("ADOPT" if fixed > broke and net > 0
                   else "REJECT — breaks working plates" if broke > fixed
                   else "no effect")
        print(f"{mode:<10}{net:>+8.1f}%{fixed:>8}{broke:>8}   {verdict}")

    print("""
Measured against the pipeline as it actually runs — 64x256 ensemble, raw
crops, fixed grammar, per-character voting — rather than the 32x128 model
plate_rectify's own --compare still loads.""")
    return 0


if __name__ == "__main__":
    sys.exit(main())
