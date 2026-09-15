"""What the native-width gate costs, and what it buys.

The recogniser resizes every crop to 256px wide, so a 58px crop stretched 4x
and a 240px native crop arrive looking identical and are read with the same
apparent certainty. They do not deserve the same trust, and the engine now
says so in two places:

    extract_plate_candidate   drops crops narrower than MIN_PLATE_NATIVE_PX
    recognize_plate           scales confidence between the floor and
                              RELIABLE_PLATE_NATIVE_PX

Both are trade-offs against coverage, and neither is worth having unless the
numbers support it. This measures three things on the ensemble's own holdout —
vehicles no model trained on:

  1  exact match and CER banded by native crop width, which is what the two
     thresholds were set from
  2  what the 70px gate actually removes: how many reads are lost, and how
     accuracy on the surviving reads moves
  3  whether tempered confidence separates right reads from wrong ones better
     than raw confidence does, scored as AUC — because the point of the
     tempering is that downstream consumers (watchlist hits, track voting)
     can threshold on confidence and get what they expect

A gate that drops 30% of reads to gain 3 points is a bad trade for a search
system, where a missing vehicle is worse than an uncertain one. A gate that
drops reads which were 0% correct anyway costs nothing. Point 2 decides which
of those this is.

Run:  python -m backend.scripts.plate_width_gate_eval
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
    BLANK, CHARS, ITOS, STOI,
)

REAL = ROOT / "data" / "plate_real"
MODELS = ROOT / "models" / "plate_recognizer"

MIN_PLATE_NATIVE_PX = 70
RELIABLE_PLATE_NATIVE_PX = 100


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


def temper(conf: float, native_w: int) -> float:
    """The engine's scaling, reproduced exactly."""
    if native_w >= RELIABLE_PLATE_NATIVE_PX:
        return conf
    span = RELIABLE_PLATE_NATIVE_PX - MIN_PLATE_NATIVE_PX
    frac = (native_w - MIN_PLATE_NATIVE_PX) / max(span, 1)
    return conf * (0.55 + 0.45 * max(0.0, min(1.0, frac)))


def auc(scores: list[float], labels: list[int]) -> float:
    """Probability a correct read scores above an incorrect one.

    Rank-based, so ties contribute half — which matters here because tempering
    maps a band of widths onto the same multiplier.
    """
    pos = [s for s, y in zip(scores, labels) if y]
    neg = [s for s, y in zip(scores, labels) if not y]
    if not pos or not neg:
        return float("nan")
    wins = sum((p > n) + 0.5 * (p == n) for p in pos for n in neg)
    return wins / (len(pos) * len(neg))


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
    random.Random(1000).shuffle(kk)          # plate_final_model's own split
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
    if not members:
        print("no ensemble members found")
        return 1
    h, w = hw
    print(f"{len(clean)} vehicles no model trained on, {len(members)} members "
          f"at {h}x{w}\n", flush=True)

    def read(gray: np.ndarray) -> tuple[str, float]:
        x = cv2.resize(gray, (w, h), interpolation=cv2.INTER_AREA)
        x = torch.from_numpy(x).float().div(127.5).sub(1.0)[None, None].to(dev)
        with torch.no_grad():
            lp = torch.stack([F.log_softmax(m(x), dim=2) for m in members]
                             ).mean(dim=0)[0].cpu().numpy()
        raw = greedy(lp)
        conf = float(np.exp(np.mean(np.max(lp, axis=1))))
        d = decode_plate(raw)
        return (d["plate"] or raw), conf

    # Best view per vehicle, which is what the pipeline selects, along with the
    # native width of the crop that view came from.
    rows = []
    for k in clean:
        gt = truth[k]
        best = None
        for f in by_track[k]:
            g = cv2.imread(str(REAL / "images" / f), cv2.IMREAD_GRAYSCALE)
            if g is None:
                continue
            got, conf = read(g)
            if best is None or conf > best[1]:
                best = (got, conf, g.shape[1])
        if best is None:
            continue
        got, conf, native_w = best
        rows.append({
            "gt": gt, "got": got, "conf": conf, "w": native_w,
            "exact": got == gt, "cer": lev(got, gt) / max(1, len(gt)),
        })

    print("=" * 66)
    print("1  ACCURACY BY NATIVE CROP WIDTH")
    print("=" * 66)
    print(f"{'native width':<18}{'plates':>8}{'share':>9}{'exact':>9}{'CER':>9}"
          f"{'upscale':>10}")
    print("-" * 66)
    bands = [(0, 70), (70, 100), (100, 140), (140, 200), (200, 10_000)]
    for lo, hi in bands:
        sel = [r for r in rows if lo <= r["w"] < hi]
        if not sel:
            continue
        label = f"{lo}-{hi}px" if hi < 10_000 else f"{lo}px+"
        up = 256 / np.median([r["w"] for r in sel])
        print(f"{label:<18}{len(sel):>8}{100*len(sel)/len(rows):>8.1f}%"
              f"{100*sum(r['exact'] for r in sel)/len(sel):>8.1f}%"
              f"{100*np.mean([r['cer'] for r in sel]):>8.1f}%{up:>9.1f}x")

    print("\n" + "=" * 66)
    print("2  WHAT THE GATE COSTS AND BUYS")
    print("=" * 66)
    kept = [r for r in rows if r["w"] >= MIN_PLATE_NATIVE_PX]
    lost = [r for r in rows if r["w"] < MIN_PLATE_NATIVE_PX]
    print(f"{'':<22}{'plates':>8}{'exact':>9}{'CER':>9}")
    print("-" * 50)
    print(f"{'no gate (all reads)':<22}{len(rows):>8}"
          f"{100*sum(r['exact'] for r in rows)/max(len(rows),1):>8.1f}%"
          f"{100*np.mean([r['cer'] for r in rows]):>8.1f}%")
    if kept:
        print(f"{'gate at 70px (kept)':<22}{len(kept):>8}"
              f"{100*sum(r['exact'] for r in kept)/len(kept):>8.1f}%"
              f"{100*np.mean([r['cer'] for r in kept]):>8.1f}%")
    if lost:
        print(f"{'  dropped by the gate':<22}{len(lost):>8}"
              f"{100*sum(r['exact'] for r in lost)/len(lost):>8.1f}%"
              f"{100*np.mean([r['cer'] for r in lost]):>8.1f}%")
        print(f"\ncoverage lost: {100*len(lost)/len(rows):.1f}% of plates, "
              f"of which {sum(r['exact'] for r in lost)} were being read "
              f"correctly.")
    else:
        print("\nno crop in this holdout falls below the gate — the holdout is "
              "human-framed\nlabelled crops, which are wider than what the "
              "detector cuts in production.")

    print("\n" + "=" * 66)
    print("3  DOES TEMPERING MAKE CONFIDENCE MEAN MORE?")
    print("=" * 66)
    labels = [int(r["exact"]) for r in kept]
    raw_auc = auc([r["conf"] for r in kept], labels)
    tmp_auc = auc([temper(r["conf"], r["w"]) for r in kept], labels)
    print(f"{'confidence':<22}{'AUC':>9}")
    print("-" * 32)
    print(f"{'raw':<22}{raw_auc:>9.3f}")
    print(f"{'width-tempered':<22}{tmp_auc:>9.3f}")
    print(f"""
AUC is the chance a correct read outranks an incorrect one. 0.5 is a coin
flip. A gain here means a downstream threshold — a watchlist hit, a track
vote — separates good reads from bad ones better than it did, which is the
whole point of carrying native width forward rather than discarding it at the
resize.""")
    return 0


if __name__ == "__main__":
    sys.exit(main())
