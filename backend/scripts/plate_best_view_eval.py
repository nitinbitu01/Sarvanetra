"""How much of the accuracy gap is the evaluation, not the recogniser?

The headline figure — 45.5% exact — comes from plate_accuracy_by_size, which
scores `by_track[k][0]`: the first harvested view of each vehicle, chosen by
nothing. But it buckets that score by the vehicle's *widest* view. So a
vehicle whose best frame is 120px is filed under 120px while being read from
whichever frame happened to sort first, and the reported curve came out
backwards — 68% at 70-100px, 33% at 100-140px. A resolution limit cannot
produce that shape.

Meanwhile the corpus holds 12 views of most vehicles. No deployed ANPR reads
one arbitrary frame and discards eleven; it reads the best one it gets, or
combines them. The measurement was answering a question nobody asks.

This scores the same held-out vehicles four ways, on the same model and the
same split, so the comparison is like for like:

    first view      what is currently reported
    widest view     the best-resolved frame
    sharpest view   the least motion-blurred frame
    all views       per-character vote across every frame of that vehicle

The gap between the first and the last is accuracy already present in the
data and thrown away by how it is read.

Run:  python -m backend.scripts.plate_best_view_eval
"""
from __future__ import annotations

import json
import random
import sys
from collections import Counter, defaultdict
from pathlib import Path

import cv2
import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from backend.scripts.indian_plate_grammar import decode_plate      # noqa: E402
from backend.scripts.train_plate_recognizer import (                # noqa: E402
    CHARS, CRNN, IMG_H, IMG_W, STOI, ctc_decode,
)

REAL = ROOT / "data" / "plate_real"
CORPUS = ROOT / "output" / "plate_corpus"
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


def read_one(model, dev, path: Path) -> tuple[str, np.ndarray | None]:
    """Decoded plate and the per-timestep log-probabilities behind it."""
    g = cv2.imread(str(path), cv2.IMREAD_GRAYSCALE)
    if g is None:
        return "", None
    g = cv2.resize(g, (IMG_W, IMG_H), interpolation=cv2.INTER_AREA)
    x = torch.from_numpy(g).float().div(127.5).sub(1.0)[None, None].to(dev)
    with torch.no_grad():
        logits = model(x)
        pred = ctc_decode(logits)[0]
    d = decode_plate(pred)
    return (d["plate"] or pred), None


def vote(reads: list[str]) -> str:
    """Per-position character vote across the reads of one vehicle.

    Only reads of the modal length take part. Mixing a 9-character read with a
    10-character one shifts every position after the disagreement, so voting
    across lengths corrupts the positions that were right.
    """
    reads = [r for r in reads if r]
    if not reads:
        return ""
    modal_len = Counter(len(r) for r in reads).most_common(1)[0][0]
    same = [r for r in reads if len(r) == modal_len]
    return "".join(Counter(c).most_common(1)[0][0] for c in zip(*same))


def main() -> int:
    verified = [json.loads(l) for l in (REAL / "verified_all.jsonl").open(encoding="utf-8")]
    truth = {(v["camera"], v["track"]): v["text"] for v in verified
             if v["text"] and all(c in STOI for c in v["text"])}

    rows = [json.loads(l) for l in (REAL / "labels.jsonl").open(encoding="utf-8")]
    by_track: dict[tuple, list] = defaultdict(list)
    for r in rows:
        k = (r["camera"], r["track"])
        if k in truth:
            by_track[k].append(r["file"])

    # Per-view width and sharpness, measured from the crop the model reads.
    #
    # Not joined from the manifest: label filenames come in two shapes
    # (`_00.jpg` indexed and `_xf011800.jpg` frame-stamped) and a track has
    # more manifest rows than labelled views — 25 against 12 on the first
    # vehicle checked — so no index correspondence holds. An earlier version
    # of this script joined on filename, silently matched nothing, and gave
    # every view width 0 and sharpness 0. All three selection strategies then
    # returned max() of an all-equal list, which is the first element, and
    # produced three identical rows that looked like a finding.
    #
    # Measuring the crop directly is also the more honest quantity: it is the
    # resolution and focus of the pixels actually fed to the recogniser, not
    # of the detector box they were cut from.
    def crop_meta(name: str) -> dict:
        img = cv2.imread(str(REAL / "images" / name), cv2.IMREAD_GRAYSCALE)
        if img is None:
            return {"w": 0.0, "s": 0.0}
        return {"w": float(img.shape[1]),
                "s": float(cv2.Laplacian(img, cv2.CV_64F).var())}

    view_meta: dict[str, dict] = {}
    px_of: dict[tuple, float] = {}
    for k in by_track:
        for f in by_track[k]:
            if f not in view_meta:
                view_meta[f] = crop_meta(f)
        px_of[k] = max((view_meta[f]["w"] for f in by_track[k]), default=0.0)

    keys = sorted(by_track)
    random.seed(SEED)
    random.shuffle(keys)
    test = [k for k in keys[:HOLDOUT]]
    print(f"labelled vehicles : {len(truth)}")
    print(f"held out          : {len(test)}")
    print(f"views available   : median "
          f"{int(np.median([len(by_track[k]) for k in test]))} per vehicle\n",
          flush=True)

    dev = "cuda" if torch.cuda.is_available() else "cpu"
    ck = torch.load(CKPT, map_location=dev, weights_only=False)
    model = CRNN(len(CHARS) + 1).to(dev)
    model.load_state_dict(ck["model"])
    model.eval()

    results = {k: {} for k in ("first", "widest", "sharpest", "vote")}
    per_vehicle = []

    for k in test:
        files = by_track[k]
        gt = truth[k]
        metas = [(f, view_meta.get(f, {"w": 0.0, "s": 0.0})) for f in files]

        picks = {
            "first": files[0],
            "widest": max(metas, key=lambda m: m[1]["w"])[0],
            # Sharpness first, size second — the frame-selection rule the
            # harvest now uses, because the widest view of a vehicle is the
            # closest and so the most motion-blurred.
            "sharpest": max(metas, key=lambda m: m[1]["s"] ** 0.5 * m[1]["w"])[0],
        }

        reads = {}
        for name, f in picks.items():
            reads[name], _ = read_one(model, dev, REAL / "images" / f)

        all_reads = [read_one(model, dev, REAL / "images" / f)[0] for f in files]
        reads["vote"] = vote(all_reads)

        for name, got in reads.items():
            results[name].setdefault("n", 0)
            results[name]["n"] += 1
            results[name]["exact"] = results[name].get("exact", 0) + (got == gt)
            results[name]["cer"] = results[name].get("cer", 0.0) + (
                lev(got, gt) / max(1, len(gt)))

        per_vehicle.append({
            "key": k, "gt": gt, "width": px_of.get(k, 0.0),
            "n_views": len(files),
            # Width of the crop each strategy actually read, so accuracy can
            # be reported against the frame it was measured on.
            "read_width_first": view_meta[picks["first"]]["w"],
            "read_width_widest": view_meta[picks["widest"]]["w"],
            **{f"read_{n}": v for n, v in reads.items()},
        })

    print(f"{'strategy':<26}{'exact':>9}{'CER':>9}")
    print("-" * 44)
    for name in ("first", "widest", "sharpest", "vote"):
        r = results[name]
        n = r["n"]
        label = {
            "first": "first view (as reported)",
            "widest": "widest view",
            "sharpest": "sharpest x size view",
            "vote": "vote across all views",
        }[name]
        print(f"{label:<26}{100*r['exact']/n:>8.1f}%{100*r['cer']/n:>8.1f}%")

    # Accuracy against the width of the crop that was actually read.
    #
    # Bucketing by the vehicle's widest view while scoring a different view
    # answers a question with no operational meaning. A capture envelope is a
    # promise about the frame the system reads, so the width on the x-axis has
    # to be that frame's width.
    print(f"\n{'read-crop width':<18}{'n':>5}{'exact':>9}   (widest view, read from it)")
    print("-" * 56)
    for lo, hi in ((0, 70), (70, 100), (100, 140), (140, 200), (200, 10_000)):
        sel = [p for p in per_vehicle if lo <= p["read_width_widest"] < hi]
        if not sel:
            continue
        ok = sum(1 for p in sel if p["read_widest"] == p["gt"])
        print(f"{f'{lo}-{hi}px':<18}{len(sel):>5}{100*ok/len(sel):>8.1f}%")

    # The capture envelope: accuracy against coverage, as a vendor states it.
    print(f"\n{'accept crops >=':<18}{'vehicles':>10}{'coverage':>11}{'exact':>9}")
    print("-" * 50)
    total = len(per_vehicle)
    for floor in (0, 70, 80, 90, 100, 110, 120, 140, 160):
        sel = [p for p in per_vehicle if p["read_width_widest"] >= floor]
        if len(sel) < 5:
            continue
        ok = sum(1 for p in sel if p["read_widest"] == p["gt"])
        print(f"{f'{floor}px':<18}{len(sel):>10}{100*len(sel)/total:>10.0f}%"
              f"{100*ok/len(sel):>8.1f}%")

    out = ROOT / "output" / "plate_best_view_eval.json"
    out.write_text(json.dumps(
        {"holdout": len(test), "results": results, "per_vehicle": per_vehicle},
        indent=1, default=str), encoding="utf-8")
    print(f"\nwrote {out}")

    print("""
The first row is the number currently quoted. The others use the same model
and the same vehicles — every point of difference is accuracy that was
already in the footage and discarded by reading one arbitrary frame.""")
    return 0


if __name__ == "__main__":
    sys.exit(main())
