"""Is the recogniser misreading characters, or dropping them?

"It read a plate" and "it read the plate correctly" are different claims, and
a decode rate of 100% establishes only the first. This separates the ways a
read can be wrong, because each has a different cause and a different fix:

    dropped     prediction is shorter than the truth. CTC collapsed two
                identical adjacent characters, or the crop was cut short.
                A grammar rule cannot recover a character that was never
                emitted.
    inserted    prediction is longer. Usually plate-border pixels or a
                bolt read as a glyph — a cropping problem.
    substituted right length, wrong characters. This is the classic OCR
                confusion (8/B, 0/O, 1/I) and the one a format grammar and
                a confusion-aware matcher can actually fix.

It also reports which position in the plate fails. An Indian registration is
state(2) + district(1-2) + series(1-3) + number(4), and those fields are not
equally hard: the state code has 30-odd valid values and can be corrected
against a list, while the four-digit number is unconstrained and an error in
it is unrecoverable.

Run:  python -m backend.scripts.plate_error_modes
"""
from __future__ import annotations

import json
import random
import sys
from collections import Counter, defaultdict
from pathlib import Path

import cv2
import torch

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from backend.scripts.indian_plate_grammar import decode_plate      # noqa: E402
from backend.scripts.train_plate_recognizer import (                # noqa: E402
    CHARS, CRNN, IMG_H, IMG_W, STOI, ctc_decode,
)

REAL = ROOT / "data" / "plate_real"
CKPT = ROOT / "models" / "plate_recognizer" / "finetuned.pt"
HOLDOUT = 55
SEED = 1337


def align(pred: str, gt: str):
    """Levenshtein backtrace: the edit operations turning pred into gt."""
    n, m = len(pred), len(gt)
    d = [[0] * (m + 1) for _ in range(n + 1)]
    for i in range(n + 1):
        d[i][0] = i
    for j in range(m + 1):
        d[0][j] = j
    for i in range(1, n + 1):
        for j in range(1, m + 1):
            d[i][j] = min(d[i - 1][j] + 1, d[i][j - 1] + 1,
                          d[i - 1][j - 1] + (pred[i - 1] != gt[j - 1]))
    ops = []
    i, j = n, m
    while i > 0 or j > 0:
        if i > 0 and j > 0 and d[i][j] == d[i - 1][j - 1] + (pred[i - 1] != gt[j - 1]):
            ops.append(("match" if pred[i - 1] == gt[j - 1] else "sub",
                        pred[i - 1], gt[j - 1]))
            i, j = i - 1, j - 1
        elif i > 0 and d[i][j] == d[i - 1][j] + 1:
            ops.append(("insert", pred[i - 1], ""))   # pred has an extra char
            i -= 1
        else:
            ops.append(("delete", "", gt[j - 1]))     # pred dropped a char
            j -= 1
    return list(reversed(ops))


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

    exact = 0
    n = 0
    op_counts = Counter()
    confusions = Counter()
    len_delta = Counter()
    field_errors = Counter()
    field_totals = Counter()
    examples = []

    for k in test:
        gt = truth[k]
        f = by_track[k][0]
        g = cv2.imread(str(REAL / "images" / f), cv2.IMREAD_GRAYSCALE)
        if g is None:
            continue
        g = cv2.resize(g, (IMG_W, IMG_H), interpolation=cv2.INTER_AREA)
        x = torch.from_numpy(g).float().div(127.5).sub(1.0)[None, None].to(dev)
        with torch.no_grad():
            raw = ctc_decode(model(x))[0]
        d = decode_plate(raw)
        pred = d["plate"] or raw

        n += 1
        if pred == gt:
            exact += 1
            continue

        ops = align(pred, gt)
        kinds = Counter(o[0] for o in ops)
        # Classify the whole read by its dominant failure.
        if kinds["delete"] and not kinds["insert"] and not kinds["sub"]:
            op_counts["dropped characters only"] += 1
        elif kinds["insert"] and not kinds["delete"] and not kinds["sub"]:
            op_counts["extra characters only"] += 1
        elif kinds["sub"] and not kinds["delete"] and not kinds["insert"]:
            op_counts["substitutions only"] += 1
        else:
            op_counts["mixed"] += 1

        len_delta[len(pred) - len(gt)] += 1
        for kind, p, t in ops:
            if kind == "sub":
                confusions[f"{t}->{p}"] += 1

        # Which field of the registration failed, when lengths agree.
        if len(pred) == len(gt) and len(gt) >= 9:
            spans = {"state (2)": (0, 2), "district (2)": (2, 4),
                     "series (2)": (4, len(gt) - 4),
                     "number (4)": (len(gt) - 4, len(gt))}
            for name, (a, b) in spans.items():
                field_totals[name] += 1
                if pred[a:b] != gt[a:b]:
                    field_errors[name] += 1

        if len(examples) < 12:
            examples.append((gt, pred))

    wrong = n - exact
    print(f"\nheld out: {n} plates")
    print(f"exact match: {exact} ({100*exact/max(n,1):.1f}%)")
    print(f"wrong      : {wrong} ({100*wrong/max(n,1):.1f}%)\n")

    print("HOW THE WRONG READS ARE WRONG")
    print("-" * 52)
    for kind, cnt in op_counts.most_common():
        print(f"  {kind:<28}{cnt:>4}   {100*cnt/max(wrong,1):>5.1f}% of errors")

    print("\nLENGTH: predicted minus truth")
    print("-" * 52)
    for delta in sorted(len_delta):
        label = ("correct length" if delta == 0 else
                 f"{abs(delta)} char {'short' if delta < 0 else 'long'}")
        print(f"  {label:<28}{len_delta[delta]:>4}")

    if field_totals:
        print("\nWHICH PART OF THE REGISTRATION FAILS")
        print("  (only reads of the right length, so positions line up)")
        print("-" * 52)
        for name in ("state (2)", "district (2)", "series (2)", "number (4)"):
            if field_totals.get(name):
                err = field_errors.get(name, 0)
                print(f"  {name:<28}{err:>4} / {field_totals[name]:<4}"
                      f"{100*err/field_totals[name]:>7.1f}% wrong")

    if confusions:
        print("\nMOST COMMON CHARACTER SUBSTITUTIONS  (truth -> predicted)")
        print("-" * 52)
        for pair, cnt in confusions.most_common(12):
            print(f"  {pair:<12}{cnt:>4}")

    print("\nEXAMPLES")
    print("-" * 52)
    for gt, pred in examples:
        mark = "".join("^" if i >= len(pred) or i >= len(gt) or pred[i] != gt[i]
                       else " " for i in range(max(len(gt), len(pred))))
        print(f"  truth {gt}")
        print(f"  read  {pred}")
        print(f"        {mark}")
    print("""
Substitutions are recoverable: a format grammar and a confusion-aware
matcher can fix them. Dropped characters are not — nothing downstream can
restore a glyph the recogniser never emitted, so those reads need better
pixels, not better post-processing.""")
    return 0


if __name__ == "__main__":
    sys.exit(main())
