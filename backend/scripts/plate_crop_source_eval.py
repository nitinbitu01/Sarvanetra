"""Detector box or guessed band — which crop does the recogniser read better?

extract_plate_candidate has two tiers. Tier 1 runs the plate detector on the
vehicle crop and uses its box. Tier 2 fires when that finds nothing, and takes
a fixed slice of the vehicle instead: for a car, 72-95% of its height and
20-80% of its width, on the assumption the plate is somewhere in there.

Tier 2 is not the fallback it looks like. The plate detector was measured
finding a box in only about 5% of vehicle crops — 2.3% here, 7.3% on a
different camera set — which matches what this repository already recorded
for it: "8% detection rate, 0% on most cameras... 89% of the val set was also
in train. It memorised." So nearly every plate the system reads in production
is being read from the guessed band.

That matters because a guessed band is not centred on the plate. It includes
bumper, shadow and whatever else sits in the lower fifth of the vehicle, and
the plate inside it is off-centre and surrounded by clutter. Whether that
costs accuracy, and how much, decides whether retraining the detector is
worth doing — there are 804 labelled plate boxes on disk to retrain it with.

The corpus stores vehicle crops, so both tiers can be reproduced exactly on
the same vehicles, with ground truth available for each.

Run:  python -m backend.scripts.plate_crop_source_eval
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

from ultralytics import YOLO                                       # noqa: E402
from backend.scripts.indian_plate_grammar import decode_plate      # noqa: E402
from backend.scripts.train_plate_recognizer import (                # noqa: E402
    BLANK, CHARS, ITOS, STOI,
)

REAL = ROOT / "data" / "plate_real"
CORPUS = ROOT / "output" / "plate_corpus"
MODELS = ROOT / "models" / "plate_recognizer"
PLATE_MODEL = ROOT / "runs/detect/runs/plate/plate_v3_ft/weights/best.pt"
PLATE_CONF = 0.15


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


def heuristic_band(veh: np.ndarray) -> np.ndarray | None:
    """Tier 2 as the engine computes it, for a car."""
    h, w = veh.shape[:2]
    y1, y2 = int(0.72 * h), int(0.95 * h)
    x1, x2 = int(0.20 * w), int(0.80 * w)
    if y2 <= y1 or x2 <= x1:
        return None
    return veh[y1:y2, x1:x2]


def main() -> int:
    truth = {}
    for l in (REAL / "verified_all.jsonl").open(encoding="utf-8"):
        v = json.loads(l)
        if v.get("text") and all(c in STOI for c in v["text"]):
            truth[(v["camera"], v["track"])] = v["text"]

    # Vehicle crops from the corpus, grouped by the vehicle they belong to.
    veh_crops = defaultdict(list)
    for l in (CORPUS / "manifest.jsonl").open(encoding="utf-8"):
        r = json.loads(l)
        k = (r["camera"], r["track"])
        if k in truth:
            veh_crops[k].append(r["path"])

    keys = sorted(veh_crops)
    random.Random(1000).shuffle(keys)
    clean = keys[:max(20, int(len(keys) * 0.20))]
    print(f"{len(clean)} labelled vehicles with corpus crops\n", flush=True)

    dev = "cuda" if torch.cuda.is_available() else "cpu"
    pdet = YOLO(str(PLATE_MODEL))

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
    h, w = hw

    def read(crop) -> str:
        if crop is None or crop.size == 0:
            return ""
        # ndim == 3 is not the same as "colour": a crop can be (h, w, 1),
        # which cvtColor rejects rather than passing through.
        if crop.ndim == 3 and crop.shape[2] >= 3:
            g = cv2.cvtColor(crop, cv2.COLOR_BGR2GRAY)
        else:
            g = crop.reshape(crop.shape[0], crop.shape[1])
        g = cv2.resize(g, (w, h), interpolation=cv2.INTER_AREA)
        x = torch.from_numpy(g).float().div(127.5).sub(1.0)[None, None].to(dev)
        with torch.no_grad():
            lp = torch.stack([F.log_softmax(m(x), dim=2) for m in members]
                             ).mean(dim=0)[0].cpu().numpy()
        raw = greedy(lp)
        d = decode_plate(raw)
        return d["plate"] or raw

    stats = {k: {"n": 0, "exact": 0, "cer": 0.0}
             for k in ("detector box", "guessed band", "labelled crop")}
    det_hits = 0
    det_tries = 0

    # The labelled crop for the same vehicle, as the upper bound: it is what a
    # human framed, and the model was trained on crops like it.
    lab_files = defaultdict(list)
    for l in (REAL / "labels.jsonl").open(encoding="utf-8"):
        r = json.loads(l)
        k = (r["camera"], r["track"])
        if k in truth:
            lab_files[k].append(r["file"])

    for k in clean:
        gt = truth[k]
        # One representative vehicle crop: the widest, which is the best view
        # the pipeline had of it.
        paths = veh_crops[k]
        best_path, best_w = None, -1
        for p in paths:
            fp = ROOT / p
            if not fp.is_file():
                continue
            im = cv2.imread(str(fp))
            if im is not None and im.shape[1] > best_w:
                best_w, best_path = im.shape[1], im
        if best_path is None:
            continue
        veh = best_path

        det_tries += 1
        r = pdet.predict(veh, imgsz=320, conf=PLATE_CONF, verbose=False,
                         device="cuda:0")
        b = r[0].boxes
        det_crop = None
        if b is not None and len(b):
            det_hits += 1
            i = int(b.conf.argmax())
            x1, y1, x2, y2 = (int(v) for v in b.xyxy[i].cpu().numpy())
            pad_w, pad_h = int(0.08 * (x2 - x1)), int(0.10 * (y2 - y1))
            x1, y1 = max(0, x1 - pad_w), max(0, y1 - pad_h)
            x2 = min(veh.shape[1], x2 + pad_w)
            y2 = min(veh.shape[0], y2 + pad_h)
            det_crop = veh[y1:y2, x1:x2]

        candidates = {
            "detector box": det_crop,
            "guessed band": heuristic_band(veh),
        }
        if lab_files.get(k):
            lc = cv2.imread(str(REAL / "images" / lab_files[k][0]),
                            cv2.IMREAD_GRAYSCALE)
            candidates["labelled crop"] = lc

        for name, crop in candidates.items():
            if crop is None:
                continue
            got = read(crop)
            s = stats[name]
            s["n"] += 1
            s["exact"] += (got == gt)
            s["cer"] += lev(got, gt) / max(1, len(gt))

    print(f"plate detector found a box in {det_hits}/{det_tries} vehicle crops "
          f"({100*det_hits/max(det_tries,1):.1f}%)\n")

    print(f"{'crop source':<18}{'n':>5}{'exact':>9}{'CER':>9}")
    print("-" * 42)
    for name in ("labelled crop", "detector box", "guessed band"):
        s = stats[name]
        if not s["n"]:
            continue
        print(f"{name:<18}{s['n']:>5}{100*s['exact']/s['n']:>8.1f}%"
              f"{100*s['cer']/s['n']:>8.1f}%")

    print("""
"labelled crop" is what a human framed and what the model was trained on, so
it is the ceiling. "detector box" is tier 1 and "guessed band" is tier 2, the
one that actually runs for most vehicles. The gap between the last two is
what retraining the plate detector would be worth; the gap to the first is
what perfect framing would be worth.""")
    return 0


if __name__ == "__main__":
    sys.exit(main())
