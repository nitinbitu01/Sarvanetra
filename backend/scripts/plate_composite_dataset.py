"""backend/scripts/plate_composite_dataset.py — paste synthetic plates onto REAL
vehicle crops to build a balanced detector training set.

THE PROBLEM THIS SOLVES
  Harvested labels came from a weak OCR pipeline, so they exist only where that
  pipeline could already read:

      CAM_09 276 boxes | CAM_07 37 | CAM_10 37 | CAM_08 34 | CAM_06 7 | CAM_04 4
      CAM_01 0 | CAM_02 0 | CAM_03 0 | CAM_05 0

  70% from one camera, four cameras with nothing at all. A detector trained on
  that learns CAM_09's viewing geometry, which is exactly the model that fails
  at a junction it has not seen. Detector v1 confirmed it - on CAM_01 it fired
  on the burned-in station caption 87% of the time and found no plates.

THE FIX
  Compositing decouples WHERE the training data comes from (any real crop, from
  any camera) from WHETHER it can be labelled (always - we drew the plate, so
  the box is exact). Sample vehicle crops evenly across all cameras, paste a
  synthetic plate where a plate belongs, and every camera contributes equally
  with pixel-perfect ground truth.

  This also removes the caption failure entirely: the model now sees thousands
  of examples of "plate here, caption there, only one is the target".

REALISM IS WHAT MAKES IT TRANSFER
  A flat paste teaches the model to find rectangles that look pasted. Each
  composite therefore gets:
    - scale from real geometry (a plate is 0.20-0.36 of vehicle width)
    - placement in the lower-centre band, where plates actually sit
    - perspective warp, since these cameras look down at traffic
    - photometric match - the plate is relit to the local brightness of the
      region it lands on, so a plate on a shadowed car is shadowed
    - edge feathering and a light blur pass, so it shares the crop's focus
  Pasting over the plate region also covers any real plate already there,
  which prevents teaching the detector that a genuine plate is background.

USAGE
  python -m backend.scripts.plate_composite_dataset --n 6000
  python -m backend.scripts.plate_composite_dataset --n 12000 --keep-real
"""
from __future__ import annotations

import argparse
import json
import random
from collections import defaultdict
from pathlib import Path

import cv2
import numpy as np

CORPUS = Path("output/plate_corpus")
MANIFEST = CORPUS / "manifest.jsonl"
SYNTH = Path("data/synth_plates")
OUT = Path("data/plate_detect_synth")
REAL_LABELS = Path("data/plate_detect")


def photometric_match(plate: np.ndarray, region: np.ndarray) -> np.ndarray:
    """Relight the plate to the brightness/contrast of where it lands.

    A pasted plate that keeps its own exposure is trivially separable from the
    scene, and the detector learns that shortcut instead of learning plates.
    """
    pg = cv2.cvtColor(plate, cv2.COLOR_BGR2GRAY).astype(np.float32)
    rg = cv2.cvtColor(region, cv2.COLOR_BGR2GRAY).astype(np.float32)
    ps, pm = pg.std() + 1e-6, pg.mean()
    rs, rm = rg.std() + 1e-6, rg.mean()
    # Move partway toward the region's statistics: full matching would wash
    # out the plate's intentional high contrast, which is a real property.
    gain = np.clip((rs / ps) ** 0.35, 0.55, 1.5)
    bias = (rm - pm) * 0.45
    return np.clip(plate.astype(np.float32) * gain + bias, 0, 255).astype(np.uint8)


def paste(veh: np.ndarray, plate: np.ndarray) -> tuple[np.ndarray, tuple] | None:
    H, W = veh.shape[:2]
    frac = random.uniform(0.20, 0.36)          # plate width / vehicle width
    tw = int(W * frac)
    if tw < 24 or tw >= W - 8:
        return None
    th = max(8, int(tw * plate.shape[0] / plate.shape[1]))
    p = cv2.resize(plate, (tw, th), interpolation=cv2.INTER_AREA)

    # Perspective: these cameras look down, so plates are skewed and
    # foreshortened rather than fronto-parallel.
    m = random.uniform(0.0, 0.14)
    src = np.float32([[0, 0], [tw, 0], [tw, th], [0, th]])
    dst = np.float32([
        [tw * random.uniform(0, m), th * random.uniform(0, m)],
        [tw * (1 - random.uniform(0, m)), th * random.uniform(0, m)],
        [tw * (1 - random.uniform(0, m)), th * (1 - random.uniform(0, m))],
        [tw * random.uniform(0, m), th * (1 - random.uniform(0, m))]])
    M = cv2.getPerspectiveTransform(src, dst)
    p = cv2.warpPerspective(p, M, (tw, th), borderMode=cv2.BORDER_REPLICATE)

    # Lower-centre band - where plates actually are on a vehicle.
    x0 = int(random.uniform(0.5 - frac / 2 - 0.10, 0.5 - frac / 2 + 0.10) * W)
    y0 = int(random.uniform(0.52, 0.80) * H)
    x0 = max(0, min(W - tw - 1, x0))
    y0 = max(0, min(H - th - 1, y0))
    region = veh[y0:y0 + th, x0:x0 + tw]
    if region.shape[:2] != p.shape[:2]:
        return None

    p = photometric_match(p, region)
    # Share the crop's focus: a razor-sharp plate on a soft vehicle is a
    # giveaway the detector would latch onto.
    if random.random() < 0.65:
        p = cv2.GaussianBlur(p, (3, 3), random.uniform(0.3, 1.1))

    # Feathered alpha so edges do not read as a hard rectangle.
    a = np.ones((th, tw), np.float32)
    b = max(1, int(min(th, tw) * 0.12))
    a[:b, :] *= np.linspace(0, 1, b)[:, None]
    a[-b:, :] *= np.linspace(1, 0, b)[:, None]
    a[:, :b] *= np.linspace(0, 1, b)[None, :]
    a[:, -b:] *= np.linspace(1, 0, b)[None, :]
    a = a[..., None]
    out = veh.copy()
    out[y0:y0 + th, x0:x0 + tw] = (p * a + region * (1 - a)).astype(np.uint8)
    return out, (x0, y0, x0 + tw, y0 + th)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--n", type=int, default=6000)
    ap.add_argument("--val-frac", type=float, default=0.12)
    ap.add_argument("--min-veh-w", type=int, default=140,
                    help="Vehicle crops narrower than this cannot carry a "
                         "plate wide enough to be worth learning.")
    ap.add_argument("--keep-real", action="store_true", default=True,
                    help="Also copy in the real harvested labels. Synthetic "
                         "teaches placement and shape; real teaches this "
                         "fleet's actual plates.")
    ap.add_argument("--seed", type=int, default=1337)
    args = ap.parse_args()

    random.seed(args.seed)
    np.random.seed(args.seed)

    synth = sorted((SYNTH / "images").glob("*.jpg"))
    if not synth:
        raise SystemExit(f"no synthetic plates in {SYNTH} - run synth_plates first")
    rows = [json.loads(l) for l in MANIFEST.open(encoding="utf-8")]

    # One crop per track, grouped by camera, so the sample can be balanced.
    best: dict[tuple, dict] = {}
    for r in rows:
        k = (r["camera"], r["clip"], r["track"])
        s = r["sharpness"] * (r["plate_px_est"] ** 0.5)
        if k not in best or s > best[k]["_s"]:
            best[k] = {**r, "_s": s}
    by_cam: dict[str, list] = defaultdict(list)
    for r in best.values():
        by_cam[r["camera"]].append(r)

    cams = sorted(by_cam)
    print(f"synthetic plates : {len(synth)}")
    print(f"vehicle crops    : {sum(len(v) for v in by_cam.values())} "
          f"across {len(cams)} cameras")
    for c in cams:
        print(f"    {c}: {len(by_cam[c])}")

    for split in ("train", "val"):
        for kind in ("images", "labels"):
            (OUT / kind / split).mkdir(parents=True, exist_ok=True)

    made = 0
    per_cam_made: dict[str, int] = defaultdict(int)
    attempts = 0
    # Round-robin across cameras: every camera contributes equally regardless
    # of how many plates OCR happened to find there. This IS the fix.
    while made < args.n and attempts < args.n * 8:
        attempts += 1
        cam = cams[made % len(cams)]
        pool = by_cam[cam]
        if not pool:
            continue
        r = random.choice(pool)
        veh = cv2.imread(r["path"])
        if veh is None or veh.shape[1] < args.min_veh_w:
            continue
        plate = cv2.imread(str(random.choice(synth)))
        if plate is None:
            continue
        res = paste(veh, plate)
        if res is None:
            continue
        img, (x1, y1, x2, y2) = res
        split = "val" if random.random() < args.val_frac else "train"
        stem = f"syn_{cam}_{made:06d}"
        cv2.imwrite(str(OUT / "images" / split / f"{stem}.jpg"), img,
                    [cv2.IMWRITE_JPEG_QUALITY, 93])
        H, W = img.shape[:2]
        (OUT / "labels" / split / f"{stem}.txt").write_text(
            f"0 {(x1+x2)/2/W:.6f} {(y1+y2)/2/H:.6f} "
            f"{(x2-x1)/W:.6f} {(y2-y1)/H:.6f}\n")
        made += 1
        per_cam_made[cam] += 1
        if made % 1000 == 0:
            print(f"  {made}/{args.n}", flush=True)

    n_real = 0
    if args.keep_real:
        for split in ("train", "val"):
            src_i = REAL_LABELS / "images" / split
            if not src_i.is_dir():
                continue
            for p in src_i.glob("*.jpg"):
                lab = REAL_LABELS / "labels" / split / f"{p.stem}.txt"
                if not lab.is_file():
                    continue
                img = cv2.imread(str(p))
                if img is None:
                    continue
                cv2.imwrite(str(OUT / "images" / split / f"real_{p.name}"), img,
                            [cv2.IMWRITE_JPEG_QUALITY, 93])
                (OUT / "labels" / split / f"real_{p.stem}.txt").write_text(
                    lab.read_text())
                n_real += 1

    (OUT / "plate.yaml").write_text(
        f"path: {OUT.resolve().as_posix()}\n"
        "train: images/train\nval: images/val\n\nnc: 1\nnames: [plate]\n")

    print("\n" + "=" * 58)
    print(f"composited : {made}")
    print(f"real added : {n_real}")
    print(f"total      : {made + n_real}")
    print(f"per camera : {dict(sorted(per_cam_made.items()))}")
    print(f"\ndataset -> {OUT / 'plate.yaml'}")
    print("\nEvery camera now contributes equally, with exact boxes. Check a")
    print("few composites by eye: if the pasted plates look obviously stuck")
    print("on, the detector will learn 'pasted' rather than 'plate'.")


if __name__ == "__main__":
    main()
