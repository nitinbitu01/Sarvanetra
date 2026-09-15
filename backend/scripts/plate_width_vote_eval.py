"""Should a natively larger view outvote a smaller one?

Native crop width was measured to be the wrong thing to multiply confidence
by — see plate_width_gate_eval, where every setting below a 0.95 floor made
confidence worse at separating correct reads from incorrect ones. That result
is about a single read in isolation.

Voting is a different question. The aggregator weights each of a vehicle's
reads by `confidence * quality/100`, where quality is PlateFrameSelector's
sharpness and contrast score. Sharpness does not capture size: a small plate
in crisp focus scores well, and then outvotes a larger, slightly softer view
that carried four times the evidence. Here width is not being asked to predict
whether one read is right — only to rank two views of the same vehicle, which
is a much easier question and the one it is actually informative about.

Four weightings on the ensemble's own holdout, all with the same reads:

    conf only                 confidence, ignoring the views' quality
    conf x quality            what the aggregator does today
    conf x quality x width    width as an additional linear term
    widest view only          take the largest view, discard the rest

The last is the honest control. If simply picking the widest view matches
weighted voting, the weighting is decoration and the pipeline should just
select a view.

Run:  python -m backend.scripts.plate_width_vote_eval
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
from backend.scripts.plate_width_gate_eval import greedy, lev      # noqa: E402
from backend.scripts.train_plate_recognizer import CHARS, STOI     # noqa: E402
from backend.services.frame_selector import PlateFrameSelector     # noqa: E402

REAL = ROOT / "data" / "plate_real"
MODELS = ROOT / "models" / "plate_recognizer"
MIN_PLATE_NATIVE_PX = 70
RELIABLE_PLATE_NATIVE_PX = 100


def char_vote(reads: list[dict]) -> str:
    """The aggregator's per-position weighted vote, reproduced."""
    if not reads:
        return ""
    lengths = [len(r["text"]) for r in reads]
    target = max(set(lengths), key=lengths.count)
    valid = [r for r in reads if len(r["text"]) == target]
    out = []
    for pos in range(target):
        w: dict[str, float] = defaultdict(float)
        for r in valid:
            w[r["text"][pos]] += r["weight"]
        out.append(max(w.items(), key=lambda kv: kv[1])[0])
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
    selector = PlateFrameSelector(min_sharpness=12.0, min_size=(50, 15))

    def read(gray: np.ndarray) -> tuple[str, float]:
        x = cv2.resize(gray, (w, h), interpolation=cv2.INTER_AREA)
        x = torch.from_numpy(x).float().div(127.5).sub(1.0)[None, None].to(dev)
        with torch.no_grad():
            lp = torch.stack([F.log_softmax(m(x), dim=2) for m in members]
                             ).mean(dim=0)[0].cpu().numpy()
        raw = greedy(lp)
        return raw, float(np.exp(np.mean(np.max(lp, axis=1))))

    schemes = ["conf only", "conf x quality", "conf x quality x width",
               "widest view only"]
    stats = {s: {"n": 0, "exact": 0, "cer": 0.0} for s in schemes}
    multi = 0

    for k in clean:
        gt = truth[k]
        views = []
        for f in by_track[k]:
            g = cv2.imread(str(REAL / "images" / f), cv2.IMREAD_GRAYSCALE)
            if g is None or g.shape[1] < MIN_PLATE_NATIVE_PX:
                continue
            raw, conf = read(g)
            if not raw:
                continue
            q = float(selector.score(g))
            nw = g.shape[1]
            # Linear from the gate to the reliable width, then flat — past
            # 100px the resize is no longer what limits the view.
            wf = 0.5 + 0.5 * max(0.0, min(1.0,
                 (nw - MIN_PLATE_NATIVE_PX)
                 / (RELIABLE_PLATE_NATIVE_PX - MIN_PLATE_NATIVE_PX)))
            views.append({"text": raw, "conf": conf, "q": q, "w": nw,
                          "wf": wf})
        if not views:
            continue
        if len(views) > 1:
            multi += 1

        def finalise(raw: str) -> str:
            d = decode_plate(raw)
            return d["plate"] or raw

        got = {}
        for v in views:
            v["weight"] = v["conf"]
        got["conf only"] = finalise(char_vote(views))
        for v in views:
            v["weight"] = v["conf"] * v["q"] / 100.0
        got["conf x quality"] = finalise(char_vote(views))
        for v in views:
            v["weight"] = v["conf"] * v["q"] / 100.0 * v["wf"]
        got["conf x quality x width"] = finalise(char_vote(views))
        widest = max(views, key=lambda v: v["w"])
        got["widest view only"] = finalise(widest["text"])

        for name, g_ in got.items():
            s = stats[name]
            s["n"] += 1
            s["exact"] += (g_ == gt)
            s["cer"] += lev(g_, gt) / max(1, len(gt))

    n = stats["conf only"]["n"]
    print(f"{n} vehicles with at least one view above the {MIN_PLATE_NATIVE_PX}px "
          f"gate; {multi} of them\nhave more than one view, so only those can "
          f"differ between schemes.\n")
    print(f"{'vote weighting':<26}{'exact':>9}{'CER':>9}{'vs current':>13}")
    print("-" * 58)
    base = 100 * stats["conf x quality"]["exact"] / max(stats["conf x quality"]["n"], 1)
    for name in schemes:
        s = stats[name]
        if not s["n"]:
            continue
        ex = 100 * s["exact"] / s["n"]
        d = "" if name == "conf x quality" else f"{ex - base:+.1f} pts"
        print(f"{name:<26}{ex:>8.1f}%{100*s['cer']/s['n']:>8.1f}%{d:>13}")

    p = base / 100.0
    se = 100 * (p * (1 - p) / max(n, 1)) ** 0.5
    print(f"""
Standard error is about {se:.1f} points on this sample, so a gap under roughly
{2*se:.0f} points is not established. Only {multi} vehicles have more than one
view above the gate, which further limits how much any weighting can move:
the rest are single-view and score identically under all four schemes.""")
    return 0


if __name__ == "__main__":
    sys.exit(main())
