"""backend/scripts/plate_phase2_bootstrap.py — ANPR Phase 2 gate.

THE QUESTION THIS ANSWERS
  Phase 1 mined 71,507 crops across 3,688 vehicle tracks. Every yield figure
  quoted so far came from `plate_px_est`, which is derived from VEHICLE WIDTH
  via a fixed ratio - and that proxy is known to be wrong here:

    * the cameras are mounted high and angled down, so plates are foreshortened
      or hidden by the vehicle roof entirely;
    * 52% of the corpus is buses and trucks, which are ~2500mm wide, not the
      1800mm the 0.28 ratio assumes.

  So the honest question is not "how wide do we think the plate is" but
  "how many of these crops actually contain a plate something can find". This
  script measures that, and every downstream decision depends on the answer.

METHOD
  EasyOCR's detection stage (CRAFT) is a text-region detector, and a number
  plate is a text region. Run it over each crop and keep detections that look
  like a plate rather than like signage or bodywork:
    - positioned in the lower part of the vehicle
    - aspect ratio in the range a plate occupies
    - wide enough to be worth reading at all
  This is a BOOTSTRAP, not the final detector. Phase 2 proper fine-tunes
  YOLOv8n on the boxes this proposes, after human verification.

ONE CROP PER TRACK
  Sampling every crop would count the same vehicle up to 25 times and inflate
  the result. Each track contributes its single best frame, scored on
  sharpness and estimated plate size - the frame Phase 3 fusion would build
  around. The output is therefore a per-VEHICLE rate, which is the number that
  matters.

USAGE
  python -m backend.scripts.plate_phase2_bootstrap
  python -m backend.scripts.plate_phase2_bootstrap --max-tracks 400 --save-examples 40
"""
from __future__ import annotations

import argparse
import json
from collections import defaultdict
from pathlib import Path

import cv2
import numpy as np

CORPUS = Path("output/plate_corpus")
MANIFEST = CORPUS / "manifest.jsonl"
OUT_JSON = Path("output/plate_phase2_bootstrap.json")
EXAMPLES = Path("output/plate_phase2_examples")
PROPOSALS = Path("output/plate_proposals.jsonl")

CLS_NAME = {1: "car", 2: "motorcycle", 3: "bus", 4: "truck"}


def plate_like(box, crop_w: int, crop_h: int, min_w: int) -> bool:
    """Does this text region sit and look like a number plate?"""
    xs = [p[0] for p in box]
    ys = [p[1] for p in box]
    w, h = max(xs) - min(xs), max(ys) - min(ys)
    if w < min_w or h <= 0:
        return False
    ar = w / h
    if not (1.5 <= ar <= 7.0):          # plates are wide; signage often isn't
        return False
    cy = (min(ys) + max(ys)) / 2
    if cy < crop_h * 0.35:              # upper third is roof/windscreen
        return False
    if w > crop_w * 0.85:               # spans the whole vehicle - not a plate
        return False
    return True


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--max-tracks", type=int, default=800,
                    help="Tracks to sample. Each contributes ONE best frame.")
    ap.add_argument("--min-plate-w", type=int, default=18,
                    help="Ignore text regions narrower than this - below it "
                         "there is nothing to read even in principle.")
    ap.add_argument("--conf", type=float, default=0.20,
                    help="EasyOCR confidence floor. Low on purpose: this is "
                         "measuring whether a plate is FINDABLE, not whether "
                         "it is readable yet.")
    ap.add_argument("--save-examples", type=int, default=30)
    args = ap.parse_args()

    if not MANIFEST.is_file():
        raise SystemExit(f"no manifest at {MANIFEST} - run Phase 1 first")

    rows = [json.loads(l) for l in MANIFEST.open(encoding="utf-8")]
    print(f"manifest rows : {len(rows)}")

    # Best frame per track: sharp and large. Both matter - a big blurry crop
    # and a small sharp one are both unreadable.
    best: dict[tuple, dict] = {}
    for r in rows:
        k = (r["camera"], r["clip"], r["track"])
        score = r["sharpness"] * (r["plate_px_est"] ** 0.5)
        if k not in best or score > best[k]["_score"]:
            best[k] = {**r, "_score": score}
    tracks = list(best.values())
    print(f"distinct tracks: {len(tracks)}")

    tracks.sort(key=lambda r: -r["plate_px_est"])
    if args.max_tracks and len(tracks) > args.max_tracks:
        # Spread the sample across the size range rather than taking the top N,
        # so the result reflects the corpus and not just its easiest members.
        idx = np.linspace(0, len(tracks) - 1, args.max_tracks).astype(int)
        tracks = [tracks[i] for i in idx]
    print(f"sampling       : {len(tracks)} tracks\n")

    import easyocr
    print("loading EasyOCR...", flush=True)
    reader = easyocr.Reader(["en"], gpu=True, verbose=False)

    EXAMPLES.mkdir(parents=True, exist_ok=True)
    prop_f = PROPOSALS.open("w", encoding="utf-8")

    by_cls = defaultdict(lambda: {"n": 0, "hit": 0})
    by_cam = defaultdict(lambda: {"n": 0, "hit": 0})
    by_bin = defaultdict(lambda: {"n": 0, "hit": 0})
    saved = 0
    hits = 0

    for i, r in enumerate(tracks, 1):
        img = cv2.imread(r["path"])
        if img is None:
            continue
        h, w = img.shape[:2]
        cname = CLS_NAME.get(r["cls"], str(r["cls"]))
        pbin = ("<50" if r["plate_px_est"] < 50 else
                "50-70" if r["plate_px_est"] < 70 else
                "70-100" if r["plate_px_est"] < 100 else ">=100")

        found = reader.readtext(img)
        cands = [(b, t, float(c)) for b, t, c in found
                 if c >= args.conf and plate_like(b, w, h, args.min_plate_w)]

        for d in (by_cls[cname], by_cam[r["camera"]], by_bin[pbin]):
            d["n"] += 1
        if cands:
            hits += 1
            for d in (by_cls[cname], by_cam[r["camera"]], by_bin[pbin]):
                d["hit"] += 1
            b, t, c = max(cands, key=lambda x: x[2])
            xs = [p[0] for p in b]
            ys = [p[1] for p in b]
            prop_f.write(json.dumps({
                "path": r["path"], "camera": r["camera"], "cls": cname,
                "plate_px_est": r["plate_px_est"],
                "box": [int(min(xs)), int(min(ys)), int(max(xs)), int(max(ys))],
                "text": t, "conf": round(c, 3),
                "box_w": int(max(xs) - min(xs)),
            }) + "\n")
            if saved < args.save_examples:
                vis = img.copy()
                cv2.rectangle(vis, (int(min(xs)), int(min(ys))),
                              (int(max(xs)), int(max(ys))), (0, 255, 0), 2)
                cv2.imwrite(str(EXAMPLES /
                            f"{saved:03d}_{cname}_{int(max(xs)-min(xs))}px.jpg"), vis)
                saved += 1

        if i % 100 == 0:
            print(f"  {i}/{len(tracks)}  hit rate {hits/i*100:.1f}%", flush=True)

    prop_f.close()
    n = len(tracks)

    def table(title, d, key="") -> None:
        print(f"\n--- {title} ---")
        for k in sorted(d, key=lambda k: -d[k]["n"]):
            v = d[k]
            print(f"  {str(k):<12} {v['hit']:>4}/{v['n']:<5} "
                  f"({v['hit']/max(v['n'],1)*100:>5.1f}%)")

    print("\n" + "=" * 62)
    print(f"PHASE 2 GATE: {hits} of {n} vehicles have a findable plate "
          f"({hits/max(n,1)*100:.1f}%)")
    print("=" * 62)
    table("by vehicle class", by_cls)
    table("by estimated plate width", by_bin)
    table("by camera", by_cam)

    OUT_JSON.write_text(json.dumps({
        "sampled": n, "hits": hits, "rate": hits / max(n, 1),
        "by_class": {k: dict(v) for k, v in by_cls.items()},
        "by_bin": {k: dict(v) for k, v in by_bin.items()},
        "by_camera": {k: dict(v) for k, v in by_cam.items()},
    }, indent=2))

    print(f"\nproposals -> {PROPOSALS}  (boxes to verify + fine-tune on)")
    print(f"examples  -> {EXAMPLES}   (open these - trust your eyes over the %)")
    print(f"summary   -> {OUT_JSON}")
    print("\nREAD THIS AS: what fraction of vehicles have a plate a detector")
    print("can LOCATE. Whether those plates can be READ is Phase 4, and will")
    print("be lower. A findable plate is necessary, not sufficient.")


if __name__ == "__main__":
    main()
