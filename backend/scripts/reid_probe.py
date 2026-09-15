"""backend/scripts/reid_probe.py — can an off-the-shelf embedding tell one
vehicle from another on this footage, before anything is trained?

WHY THIS RUNS FIRST
  A hand-built colour-histogram comparison was written for this job, measured,
  and found to rank the WRONG journeys above the right ones - GJ02ED8004, a
  black car matched to a silver one, scored higher than two genuinely correct
  links. The background dominated: a vehicle crop is largely road, and two
  stretches of road look alike whatever is parked on them.

  Training a re-identification network is the obvious next step and is several
  hours of work with real domain risk. An ImageNet-pretrained backbone costs
  nothing, already encodes colour, shape and texture, and cannot overfit to
  these cameras because it never saw them. If it separates the classes, the
  training is unnecessary; if it does not, that is worth knowing before
  spending the afternoon on a model that may not either.

THE TEST SET, AND ITS HONEST LIMIT
  Cross-camera ground truth barely exists here - two journeys were confirmed by
  eye and two refuted. Four pairs cannot measure anything, so the probe uses a
  proxy:

    positives  the FIRST and LAST usable frame of one track. Same vehicle, and
               a real change of scale and angle as it approaches the camera.
    negatives  frames from two different tracks on the same camera. Same road,
               same lighting, different vehicle - which is the confusion that
               actually matters, and much harder than a random pair from a
               different camera would be.

  This is a proxy and not the real task. Front-versus-rear across two cameras
  is harder than near-versus-far on one. A backbone that fails here will
  certainly fail there; one that succeeds here still has to be checked against
  the four real pairs, which this also does.

USAGE
  python -m backend.scripts.reid_probe --pairs 400
"""
from __future__ import annotations

import argparse
import json
import random
from collections import defaultdict
from pathlib import Path

import cv2
import numpy as np
import torch

CORPUS = Path("output/plate_corpus")
REAL = Path("data/plate_real")


def load_backbone(name: str, dev: str):
    """An ImageNet-pretrained trunk with its classifier removed."""
    import torchvision.models as tv
    if name == "resnet18":
        m = tv.resnet18(weights=tv.ResNet18_Weights.IMAGENET1K_V1)
        m.fc = torch.nn.Identity()
        size = 224
    elif name == "mobilenet_v3_small":
        m = tv.mobilenet_v3_small(
            weights=tv.MobileNet_V3_Small_Weights.IMAGENET1K_V1)
        m.classifier = torch.nn.Identity()
        size = 224
    else:
        raise SystemExit(f"unknown backbone {name}")
    return m.eval().to(dev), size


MEAN = np.array([0.485, 0.456, 0.406], np.float32)
STD = np.array([0.229, 0.224, 0.225], np.float32)


@torch.no_grad()
def embed(model, paths, size, dev, batch=32):
    out = []
    buf = []
    for p in paths:
        img = cv2.imread(str(p))
        if img is None:
            buf.append(np.zeros((size, size, 3), np.float32))
            continue
        # Centre crop before resizing: the outer band of a vehicle box is road
        # and sky, and it is what defeated the histogram comparison.
        h, w = img.shape[:2]
        img = img[int(h * .10):int(h * .92), int(w * .08):int(w * .92)]
        img = cv2.cvtColor(cv2.resize(img, (size, size)), cv2.COLOR_BGR2RGB)
        buf.append((img.astype(np.float32) / 255.0 - MEAN) / STD)
    for i in range(0, len(buf), batch):
        x = torch.from_numpy(np.stack(buf[i:i + batch])).permute(0, 3, 1, 2)
        f = model(x.to(dev))
        f = torch.nn.functional.normalize(f.flatten(1), dim=1)
        out.append(f.cpu().numpy())
    return np.concatenate(out) if out else np.zeros((0, 1), np.float32)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--backbone", default="resnet18",
                    choices=["resnet18", "mobilenet_v3_small"])
    ap.add_argument("--pairs", type=int, default=400)
    args = ap.parse_args()

    truth = {}
    for line in (REAL / "verified_all.jsonl").open(encoding="utf-8"):
        v = json.loads(line)
        if v.get("text"):
            truth[(v["camera"], v["track"])] = v["text"]

    frames = defaultdict(list)
    for line in (CORPUS / "manifest.jsonl").open(encoding="utf-8"):
        r = json.loads(line)
        k = (r["camera"], r["track"])
        if k in truth:
            frames[k].append(r)
    usable = {k: sorted(v, key=lambda r: r["frame"])
              for k, v in frames.items() if len(v) >= 4}
    print(f"labelled vehicles with >=4 frames : {len(usable)}")

    rng = random.Random(5)
    keys = sorted(usable)
    rng.shuffle(keys)
    keys = keys[:args.pairs]

    # Positives: first vs last frame of one track.
    pos = [(usable[k][0]["path"], usable[k][-1]["path"]) for k in keys]

    # Negatives: two different vehicles on the SAME camera - same road, same
    # light, different car. A negative drawn from another camera would be
    # separable on background alone and would flatter the result.
    by_cam = defaultdict(list)
    for k in keys:
        by_cam[k[0]].append(k)
    neg = []
    for k in keys:
        pool = [o for o in by_cam[k[0]] if o != k]
        if not pool:
            continue
        o = rng.choice(pool)
        neg.append((usable[k][0]["path"], usable[o][-1]["path"]))
    print(f"positive pairs : {len(pos)}   negative pairs : {len(neg)}\n",
          flush=True)

    dev = "cuda" if torch.cuda.is_available() else "cpu"
    model, size = load_backbone(args.backbone, dev)
    print(f"backbone : {args.backbone} ({size}px), device {dev}", flush=True)

    def score_pairs(pairs):
        a = embed(model, [p[0] for p in pairs], size, dev)
        b = embed(model, [p[1] for p in pairs], size, dev)
        return (a * b).sum(1)

    ps = np.sort(score_pairs(pos))
    ns = np.sort(score_pairs(neg))

    print("\n" + "=" * 64)
    print("COSINE SIMILARITY BETWEEN VEHICLE CROPS")
    print("=" * 64)
    print(f"{'':<24} {'p10':>8} {'median':>8} {'p90':>8}")
    print(f"{'same vehicle':<24} {ps[len(ps)//10]:>8.3f} "
          f"{ps[len(ps)//2]:>8.3f} {ps[len(ps)*9//10]:>8.3f}")
    print(f"{'different vehicle':<24} {ns[len(ns)//10]:>8.3f} "
          f"{ns[len(ns)//2]:>8.3f} {ns[len(ns)*9//10]:>8.3f}")

    # Separability, as the area under the ROC curve.
    allv = np.concatenate([ps, ns])
    lab = np.concatenate([np.ones_like(ps), np.zeros_like(ns)])
    order = np.argsort(-allv)
    lab = lab[order]
    tp = np.cumsum(lab)
    fp = np.cumsum(1 - lab)
    auc = float(np.trapezoid(tp / max(tp[-1], 1), fp / max(fp[-1], 1)))
    print(f"\nAUC : {auc:.3f}   (0.5 is a coin flip, 1.0 is perfect)")

    best = (0.0, None)
    for th in np.linspace(allv.min(), allv.max(), 200):
        tpr = float((ps >= th).mean())
        fpr = float((ns >= th).mean())
        if tpr - fpr > best[0]:
            best = (tpr - fpr, (th, tpr, fpr))
    if best[1]:
        th, tpr, fpr = best[1]
        print(f"best cut {th:.3f}: keeps {tpr*100:.0f}% of true pairs, "
              f"admits {fpr*100:.0f}% of false ones")
    print("=" * 64)
    if auc < 0.75:
        print("\nToo weak to gate a journey link. Training a re-identification")
        print("model is the remaining option - and this result is a warning")
        print("that the crops themselves may not carry enough vehicle signal.")
    else:
        print("\nStrong enough to test against the real cross-camera pairs.")


if __name__ == "__main__":
    main()
