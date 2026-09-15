"""The only plate number that matters: reads per vehicle, end to end.

Recognition accuracy on plates the system already found is a partial number.
A system that finds 1.4% of plates and reads 52% of those correctly delivers
0.7% of vehicles identified, and quoting the 52% alone would be misleading.
This measures both halves on the same vehicles and multiplies them out:

    capture rate    of vehicles with a human-verified plate, how many does
                    the detector find a plate box on at all
    read accuracy   of those, how many are transcribed exactly right
    end to end      the product, which is what an operator experiences

Two detectors are compared on identical vehicles: the incumbent, trained on
392 frames with 89% of its val set also in train, and whatever new weights are
passed with --new.

The evaluation vehicles come from the recogniser's own holdout split (seed
1000), and the detector's val split is by vehicle too, so a vehicle used here
was seen by neither model. That is the only way this number means anything;
the previous detector reported 0.995 mAP precisely because its split leaked.

Run:  python -m backend.scripts.plate_end_to_end_eval --new output/detector_train/plate_v4/weights/best.pt
"""
from __future__ import annotations

import argparse
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
from backend.scripts.plate_width_gate_eval import greedy           # noqa: E402
from backend.scripts.train_plate_recognizer import CHARS, STOI     # noqa: E402

REAL = ROOT / "data" / "plate_real"
CORPUS = ROOT / "output" / "plate_corpus"
MODELS = ROOT / "models" / "plate_recognizer"
OLD_DET = ROOT / "runs/detect/runs/plate/plate_v3_ft/weights/best.pt"
MIN_PLATE_NATIVE_PX = 70


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--new", type=Path, required=True)
    ap.add_argument("--imgsz", type=int, default=320)
    ap.add_argument("--conf", type=float, default=0.15)
    ap.add_argument("--gate", type=int, default=MIN_PLATE_NATIVE_PX,
                    help="Native width gate, as the engine applies it.")
    args = ap.parse_args()

    from ultralytics import YOLO

    truth = {}
    for line in (REAL / "verified_all.jsonl").open(encoding="utf-8"):
        v = json.loads(line)
        if v.get("text") and all(c in STOI for c in v["text"]):
            truth[(v["camera"], v["track"])] = v["text"]

    # Vehicle crops, which is what the plate detector sees in production.
    veh = defaultdict(list)
    for line in (CORPUS / "manifest.jsonl").open(encoding="utf-8"):
        r = json.loads(line)
        k = (r["camera"], r["track"])
        if k in truth:
            veh[k].append(r["path"])

    keys = sorted(veh)
    random.Random(1000).shuffle(keys)
    clean = keys[:max(20, int(len(keys) * 0.20))]

    # Restrict to vehicles the DETECTOR also held out. The recogniser's split
    # (seed 1000) and the detector's original split (seed 4242) are each clean
    # on their own, but they disagree about which vehicles are test — so a
    # vehicle the detector trained on can sit in the recogniser's holdout, and
    # capture rate measured over it would be scored on plates the detector was
    # trained to find. align_detector_split writes the intersection.
    vfile = ROOT / "data" / "plate_detect_v2" / "val_vehicles.json"
    if vfile.is_file():
        allowed = {tuple(s.split("|", 1)) for s in json.loads(vfile.read_text())}
        allowed = {(c, int(t)) for c, t in allowed}
        before = len(clean)
        clean = [k for k in clean if k in allowed]
        print(f"{before} recogniser-holdout vehicles, {len(clean)} of them also "
              f"held out by the detector.\nMeasuring on those {len(clean)} — "
              f"trained on by neither model.\n", flush=True)
    else:
        print(f"WARNING: {vfile} missing. Run align_detector_split first, or "
              f"this number is\ncontaminated by detector training vehicles.\n",
              file=sys.stderr)
        print(f"{len(clean)} vehicles with verified plates and corpus crops\n",
              flush=True)

    # The incumbent has its own training set, and it overlaps this holdout —
    # the repository already records that 89% of its val split was also in its
    # train split. Comparing a retrained model against an incumbent scoring on
    # its own training data would understate the retrain by exactly that
    # overlap, so the vehicles it memorised are identified and excluded.
    old_ds = ROOT / "data" / "plate_detect" / "images"
    incumbent_seen = set()
    if old_ds.is_dir():
        import re
        pat = re.compile(r"(CAM[_-][\w-]+?)_.*?t(\d+)")
        for sub in ("train", "val"):
            d = old_ds / sub
            if not d.is_dir():
                continue
            for f in list(d.glob("*.jpg")) + list(d.glob("*.png")):
                m = pat.search(f.stem)
                if m:
                    incumbent_seen.add((m.group(1), int(m.group(2))))
    fair = [k for k in clean if k not in incumbent_seen]
    n_seen = len(clean) - len(fair)
    if n_seen:
        print(f"The incumbent trained on {n_seen} of these {len(clean)} "
              f"vehicles.\nA second table reports the {len(fair)} neither "
              f"detector has ever seen.\n", flush=True)

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

    def read_lp(bgr):
        g = (cv2.cvtColor(bgr, cv2.COLOR_BGR2GRAY)
             if bgr.ndim == 3 and bgr.shape[2] >= 3 else bgr.reshape(bgr.shape[:2]))
        x = cv2.resize(g, (w, h), interpolation=cv2.INTER_AREA)
        x = torch.from_numpy(x).float().div(127.5).sub(1.0)[None, None].to(dev)
        with torch.no_grad():
            return torch.stack([F.log_softmax(m(x), dim=2) for m in members]
                               ).mean(dim=0)[0].cpu().numpy()

    def char_vote(reads):
        if not reads:
            return ""
        lens = [len(r["t"]) for r in reads]
        target = max(set(lens), key=lens.count)
        valid = [r for r in reads if len(r["t"]) == target]
        out = []
        for pos in range(target):
            acc = defaultdict(float)
            for r in valid:
                acc[r["t"][pos]] += r["w"]
            out.append(max(acc.items(), key=lambda kv: kv[1])[0])
        return "".join(out)

    detectors = {"incumbent": OLD_DET, "retrained": args.new}
    results = {}
    per_vehicle = {}

    for label, path in detectors.items():
        if not Path(path).is_file():
            print(f"  {label}: {path} not found, skipping")
            continue
        det = YOLO(str(path))
        captured = exact = 0
        widths = []
        hits = {}

        for k in clean:
            gt = truth[k]
            reads = []
            for p in veh[k]:
                fp = ROOT / p if not Path(p).is_absolute() else Path(p)
                img = cv2.imread(str(fp))
                if img is None:
                    continue
                r = det.predict(img, imgsz=args.imgsz, conf=args.conf,
                                verbose=False, device=0 if dev == "cuda" else "cpu")
                b = r[0].boxes
                if b is None or not len(b):
                    continue
                i = int(b.conf.argmax())
                x1, y1, x2, y2 = (int(v) for v in b.xyxy[i].cpu().numpy())
                pw, ph = int(0.08 * (x2 - x1)), int(0.10 * (y2 - y1))
                x1, y1 = max(0, x1 - pw), max(0, y1 - ph)
                x2 = min(img.shape[1], x2 + pw)
                y2 = min(img.shape[0], y2 + ph)
                crop = img[y1:y2, x1:x2]
                if crop.size == 0 or crop.shape[1] < args.gate:
                    continue
                lp = read_lp(crop)
                conf = float(np.exp(np.mean(np.max(lp, axis=1))))
                reads.append({"t": greedy(lp), "w": conf})
                widths.append(crop.shape[1])
            if not reads:
                hits[k] = (False, False)
                continue
            captured += 1
            voted = char_vote(reads)
            d = decode_plate(voted)
            ok = (d["plate"] or voted) == gt
            if ok:
                exact += 1
            hits[k] = (True, ok)

        n = len(clean)
        results[label] = {
            "capture": 100 * captured / max(n, 1),
            "read": 100 * exact / max(captured, 1),
            "e2e": 100 * exact / max(n, 1),
            "captured": captured, "exact": exact, "n": n,
            "median_w": float(np.median(widths)) if widths else 0.0,
        }
        per_vehicle[label] = hits

    print(f"{'detector':<14}{'capture':>10}{'read acc':>11}{'end to end':>13}"
          f"{'median px':>12}")
    print("-" * 60)
    for label in ("incumbent", "retrained"):
        r = results.get(label)
        if not r:
            continue
        print(f"{label:<14}{r['capture']:>9.1f}%{r['read']:>10.1f}%"
              f"{r['e2e']:>12.1f}%{r['median_w']:>11.0f}px")

    if n_seen and len(per_vehicle) == 2:
        print(f"\nOn the {len(fair)} vehicles NEITHER detector was trained on:")
        print(f"{'detector':<14}{'capture':>10}{'read acc':>11}{'end to end':>13}")
        print("-" * 48)
        for label in ("incumbent", "retrained"):
            hv = per_vehicle.get(label)
            if not hv:
                continue
            sub = [hv[k] for k in fair if k in hv]
            if not sub:
                continue
            cap = sum(1 for c, _ in sub if c)
            ex = sum(1 for _, o in sub if o)
            print(f"{label:<14}{100*cap/len(sub):>9.1f}%"
                  f"{100*ex/max(cap,1):>10.1f}%{100*ex/len(sub):>12.1f}%")
        print("\nThis is the comparison that decides whether the retrain "
              "helped. The table\nabove it credits the incumbent for "
              f"{n_seen} vehicles it had already memorised.")

    if len(results) == 2:
        a, b = results["incumbent"], results["retrained"]
        print(f"""
capture   {a['capture']:.1f}% -> {b['capture']:.1f}%   ({b['captured']-a['captured']:+d} vehicles found)
read acc  {a['read']:.1f}% -> {b['read']:.1f}%   (on the plates each one found)
end to end {a['e2e']:.1f}% -> {b['e2e']:.1f}%   ({b['exact']-a['exact']:+d} vehicles correctly identified)

Read accuracy is measured on a DIFFERENT set of plates for each detector -
whichever ones it found - so the two read columns are not directly comparable.
A better detector can lower read accuracy while raising end to end, by finding
harder plates the old one missed entirely. End to end is the honest column.""")
    return 0


if __name__ == "__main__":
    sys.exit(main())
