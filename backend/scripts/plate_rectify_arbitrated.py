"""Rectification that cannot break a plate: let the recogniser choose.

Applying rectification unconditionally was measured on 785 crops through the
current pipeline and it loses: 48.8% against 54.8%, breaking 61 plates to fix
14, McNemar p<0.001. The transform is not useless — it fixed 14 — but there is
no way to know in advance which case you are in, so blind application pays 61
to win 14.

The idea here is to stop deciding in advance. Rectification becomes a
CANDIDATE GENERATOR rather than a transformation: the crop is read several
ways and the recogniser's own confidence picks the winner. That is sound
because confidence is not noise — measured AUC 0.732 at separating correct
reads from incorrect ones on this holdout — so the arbiter carries real
signal even though it is imperfect.

The property that matters: a candidate can only win by being MORE confident
than the original, so a warp that destroys the plate produces a low-confidence
garbage read and simply loses. Damage is bounded by how often confidence is
wrong, not by how often the warp is wrong.

Strategies measured, all against the same baseline:

  A  none                 read the crop as-is
  B  full always          what blind rectification does today
  C  argmax confidence    read none/deskew/full, keep the most confident
  D  single interpolation warp straight to the model's 64x256 input instead of
                          to the quad's own size and then resizing, which is
                          two lots of interpolation blur on an 80px plate
  E  candidates as views  hand every variant to the character voter as an
                          extra view of the same plate, so a good rectification
                          adds a vote and a bad one is outvoted

E is the one that fits the existing architecture: the voter already weights
per-position by confidence, so rectified variants need no special handling.

Significance by McNemar on the discordant pairs, because a net percentage can
hide a technique that fixes and breaks in equal numbers.

Run:  python -m backend.scripts.plate_rectify_arbitrated
"""
from __future__ import annotations

import json
import random
import sys
from collections import defaultdict
from math import comb
from pathlib import Path

import cv2
import numpy as np
import torch
import torch.nn.functional as F

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from backend.scripts.indian_plate_grammar import decode_plate       # noqa: E402
from backend.scripts.plate_rectify import _deskew, rectify          # noqa: E402
from backend.scripts.plate_width_gate_eval import greedy            # noqa: E402
from backend.scripts.train_plate_recognizer import CHARS, STOI      # noqa: E402

REAL = ROOT / "data" / "plate_real"
MODELS = ROOT / "models" / "plate_recognizer"

MIN_QUAD_AREA_FRAC = 0.25
TARGET_W, TARGET_H = 256, 64


def unwarp_direct(g: np.ndarray, out_w: int = TARGET_W,
                  out_h: int = TARGET_H) -> np.ndarray | None:
    """Perspective-correct straight onto the model's input grid.

    The shipped _unwarp maps the quad to its own pixel size, after which the
    caller resizes that to 64x256 — two interpolations, and on an 80px plate
    the second one is resampling something already softened by the first.
    Composing the warp with the scale does it in one pass.

    Returns None rather than the input when no plate-shaped quad is found, so
    the caller can tell "no rectification available" from "rectified".
    """
    h, w = g.shape[:2]
    if h < 12 or w < 30:
        return None
    e = cv2.Canny(cv2.GaussianBlur(g, (3, 3), 0), 40, 130)
    e = cv2.dilate(e, np.ones((2, 2), np.uint8), iterations=1)
    cnts, _ = cv2.findContours(e, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not cnts:
        return None
    c = max(cnts, key=cv2.contourArea)
    if cv2.contourArea(c) < MIN_QUAD_AREA_FRAC * h * w:
        return None
    peri = cv2.arcLength(c, True)
    quad = cv2.approxPolyDP(c, 0.03 * peri, True)
    if len(quad) != 4 or not cv2.isContourConvex(quad):
        return None
    p = quad.reshape(4, 2).astype(np.float32)
    s, d = p.sum(1), np.diff(p, axis=1).ravel()
    src = np.array([p[np.argmin(s)], p[np.argmin(d)],
                    p[np.argmax(s)], p[np.argmax(d)]], np.float32)
    wa = max(np.linalg.norm(src[0] - src[1]), np.linalg.norm(src[3] - src[2]))
    ha = max(np.linalg.norm(src[0] - src[3]), np.linalg.norm(src[1] - src[2]))
    if wa < 24 or ha < 8 or not (1.8 <= wa / ha <= 7.5):
        return None
    dst = np.array([[0, 0], [out_w - 1, 0],
                    [out_w - 1, out_h - 1], [0, out_h - 1]], np.float32)
    M = cv2.getPerspectiveTransform(src, dst)
    return cv2.warpPerspective(g, M, (out_w, out_h), flags=cv2.INTER_CUBIC,
                               borderMode=cv2.BORDER_REPLICATE)


def mcnemar(fixed: int, broke: int) -> float:
    n = fixed + broke
    if n == 0:
        return 1.0
    p = sum(comb(n, i) for i in range(min(fixed, broke) + 1)) / (2 ** n) * 2
    return min(p, 1.0)


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
    print(f"{len(clean)} holdout vehicles | ensemble {h}x{w}\n", flush=True)

    def read(g):
        """(text, confidence). Already-sized inputs skip the resize."""
        x = g if (g.shape[0] == h and g.shape[1] == w) else \
            cv2.resize(g, (w, h), interpolation=cv2.INTER_AREA)
        t = torch.from_numpy(x).float().div(127.5).sub(1.0)[None, None].to(dev)
        with torch.no_grad():
            lp = torch.stack([F.log_softmax(m(t), dim=2) for m in members]
                             ).mean(dim=0)[0].cpu().numpy()
        return greedy(lp), float(np.exp(np.mean(np.max(lp, axis=1))))

    def finalise(raw):
        d = decode_plate(raw)
        return d["plate"] or raw

    def char_vote(reads):
        if not reads:
            return ""
        lens = [len(r[0]) for r in reads]
        target = max(set(lens), key=lens.count)
        valid = [r for r in reads if len(r[0]) == target]
        out = []
        for pos in range(target):
            acc = defaultdict(float)
            for t, c in valid:
                acc[t[pos]] += c
            out.append(max(acc.items(), key=lambda kv: kv[1])[0])
        return "".join(out)

    STRATS = ("A none", "B full always", "C argmax conf",
              "D single-interp", "E candidates as views")
    crop_ok = {s: [] for s in STRATS}          # per-crop outcomes
    veh_ok = {s: [] for s in STRATS}           # per-vehicle, after voting

    for k in clean:
        gt = truth[k]
        per_strat_reads = {s: [] for s in STRATS}
        for f in by_track[k]:
            g = cv2.imread(str(REAL / "images" / f), cv2.IMREAD_GRAYSCALE)
            if g is None or g.shape[1] < 70:
                continue

            r_none = read(g)
            g_desk = _deskew(g)
            r_desk = read(g_desk)
            g_full = rectify(g, mode="full")
            r_full = read(g_full if g_full is not None else g)
            g_dir = unwarp_direct(_deskew(g))
            r_dir = read(g_dir) if g_dir is not None else r_none

            cands = [r_none, r_desk, r_full]
            best = max(cands, key=lambda r: r[1])
            cands_dir = [r_none, r_desk, r_dir]
            best_dir = max(cands_dir, key=lambda r: r[1])

            picks = {
                "A none": r_none,
                "B full always": r_full,
                "C argmax conf": best,
                "D single-interp": best_dir,
                "E candidates as views": r_none,   # voting handled below
            }
            for s in STRATS:
                per_strat_reads[s].append(picks[s])
                crop_ok[s].append(finalise(picks[s][0]) == gt)
            # E contributes every variant as its own view of this frame.
            per_strat_reads["E candidates as views"].extend(
                [r_desk, r_dir if g_dir is not None else r_none])

        for s in STRATS:
            rs = per_strat_reads[s]
            if not rs:
                continue
            veh_ok[s].append(finalise(char_vote(rs)) == gt)

    n_crop = len(crop_ok["A none"])
    n_veh = len(veh_ok["A none"])
    print(f"{n_crop} crops, {n_veh} vehicles\n")
    print(f"{'strategy':<24}{'per crop':>10}{'per vehicle':>13}"
          f"{'fixed':>7}{'broke':>7}{'p':>8}")
    print("-" * 72)
    base_c = crop_ok["A none"]
    base_v = veh_ok["A none"]
    for s in STRATS:
        c, v = crop_ok[s], veh_ok[s]
        pc = 100 * sum(c) / max(len(c), 1)
        pv = 100 * sum(v) / max(len(v), 1)
        if s == "A none":
            print(f"{s:<24}{pc:>9.1f}%{pv:>12.1f}%{'-':>7}{'-':>7}{'-':>8}")
            continue
        fixed = sum(1 for b, x in zip(base_v, v) if x and not b)
        broke = sum(1 for b, x in zip(base_v, v) if b and not x)
        print(f"{s:<24}{pc:>9.1f}%{pv:>12.1f}%{fixed:>7}{broke:>7}"
              f"{mcnemar(fixed, broke):>8.3f}")

    print("""
"fixed/broke/p" compare per-VEHICLE outcomes after voting, which is what the
engine emits. A strategy is only worth adopting if it fixes clearly more than
it breaks — a p above about 0.05 means the sample cannot tell the difference
from chance, whatever the headline percentage looks like.""")
    return 0


if __name__ == "__main__":
    sys.exit(main())
