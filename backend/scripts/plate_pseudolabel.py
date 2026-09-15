"""backend/scripts/plate_pseudolabel.py — STEP 1: build a real labelled plate set.

WHAT THIS PRODUCES
  Tight plate-region crops paired with a voted text label, harvested from the
  whole corpus. These are the training pairs the CRNN needs and has never had.

THE ASYMMETRY THAT MAKES MACHINE LABELS GOOD ENOUGH
  A single EasyOCR read is ~30% exact. But the same vehicle appears in up to
  25 frames, and each frame errs at a DIFFERENT position:

      frame 1  GJ32K4588
      frame 2  GJ32K45B8    <- wrong at 8
      frame 3  6J32K4588    <- wrong at 1
      frame 4  GJ32K4588
      frame 5  GJ32K4S88    <- wrong at 7

  Voting per character position recovers GJ32K4588 even though four of the
  five reads were individually wrong. The teacher is not the same reader - it
  is the reader plus twenty-five chances plus grammar.

WHY TIGHT CROPS, NOT BANDS
  The CRNN was trained on plate images that FILL the frame. Feeding it the
  whole lower-vehicle band put the plate at ~20% of the width, and after the
  resize to 128x32 there was nothing left to read: it returned 'G33ZK458' for
  a plate verified as GJ32K4588. Every crop saved here is cut to the text box
  OCR actually found, which is the input distribution the model expects.

PRECISION OVER QUANTITY - non-negotiable
  Only tracks whose vote agreement clears the threshold are written. A wrong
  label does not merely waste a sample; it teaches the model to be confidently
  wrong, which for police work is worse than reporting nothing. Fewer clean
  labels beat more dirty ones, every time.

USAGE
  python -m backend.scripts.plate_pseudolabel
  python -m backend.scripts.plate_pseudolabel --min-agreement 0.9 --max-frames 15
"""
from __future__ import annotations

import argparse
import json
from collections import Counter, defaultdict
from pathlib import Path

import cv2
import numpy as np

from backend.scripts.indian_plate_grammar import apply_state_prior, decode_plate

CORPUS = Path("output/plate_corpus")
MANIFEST = CORPUS / "manifest.jsonl"
OUT = Path("data/plate_real")
ALLOW = "ABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789"
OSD = {"BRIDGE", "BHAI", "CHIMAN", "CSITMS", "CSI", "BRIDC", "GSRTC",
       "CARRIAGE", "EXPERIENCE", "SOMNATH", "CHITRA", "JANPATH", "PTZ"}


def band_of(img):
    h, w = img.shape[:2]
    b = img[int(h * 0.42):int(h * 0.97), int(w * 0.08):int(w * 0.92)]
    if b.size == 0 or b.shape[1] < 20:
        return None
    if b.shape[1] < 360:
        f = 360 / b.shape[1]
        b = cv2.resize(b, None, fx=f, fy=f, interpolation=cv2.INTER_CUBIC)
    return b


def vote(reads: list[tuple[str, float]]) -> tuple[str | None, float, int]:
    """Per-position weighted vote among reads of equal length."""
    if not reads:
        return None, 0.0, 0
    by_len: dict[int, list] = defaultdict(list)
    for t, w in reads:
        by_len[len(t)].append((t, w))
    best_len = max(by_len, key=lambda L: sum(w for _, w in by_len[L]))
    grp = by_len[best_len]
    out, agree = [], 0.0
    for i in range(best_len):
        tally: Counter = Counter()
        for t, w in grp:
            tally[t[i]] += w
        ch, top = tally.most_common(1)[0]
        out.append(ch)
        agree += top / max(sum(tally.values()), 1e-9)
    return "".join(out), agree / best_len, len(grp)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--min-plate-px", type=float, default=45.0)
    ap.add_argument("--max-frames", type=int, default=15,
                    help="Frames per vehicle. 15 gives voting plenty of "
                         "material without doubling runtime.")
    ap.add_argument("--min-agreement", type=float, default=0.85,
                    help="Vote agreement floor for writing a label. Raise this "
                         "if Step 2 finds the labels are noisy.")
    ap.add_argument("--min-votes", type=int, default=2,
                    help="Require corroboration: a single read has no vote to "
                         "check it against.")
    ap.add_argument("--conf", type=float, default=0.10)
    ap.add_argument("--include-heavy", action="store_true", default=False,
                    help="Buses and trucks. OFF by default: their plates are "
                         "rarely visible from these downward angles, yet their "
                         "width inflates plate_px_est so they sort to the TOP "
                         "of the queue. Including them dropped a 60-track "
                         "sample from ~12% labelled to 1.7%.")
    ap.add_argument("--max-tracks", type=int, default=0, help="0 = all")
    args = ap.parse_args()

    rows = [json.loads(l) for l in MANIFEST.open(encoding="utf-8")]
    tracks: dict[tuple, list[dict]] = defaultdict(list)
    for r in rows:
        tracks[(r["camera"], r["clip"], r["track"])].append(r)

    keep_cls = {1} | ({3, 4} if args.include_heavy else set())
    elig = []
    for k, frames in tracks.items():
        best = max(frames, key=lambda r: r["plate_px_est"])
        if best["cls"] not in keep_cls or best["plate_px_est"] < args.min_plate_px:
            continue
        frames.sort(key=lambda r: -(r["sharpness"] * r["plate_px_est"]))
        elig.append((k, frames[:args.max_frames]))
    elig.sort(key=lambda t: -max(f["plate_px_est"] for f in t[1]))
    if args.max_tracks:
        elig = elig[:args.max_tracks]

    total = sum(len(f) for _, f in elig)
    print(f"tracks in corpus : {len(tracks)}")
    print(f"eligible         : {len(elig)}")
    print(f"frames to read   : {total}")
    print(f"agreement floor  : {args.min_agreement}  (min {args.min_votes} votes)")
    print("saving TIGHT plate crops, not vehicle bands\n", flush=True)

    import easyocr
    print("loading EasyOCR...", flush=True)
    reader = easyocr.Reader(["en"], gpu=True, verbose=False)

    (OUT / "images").mkdir(parents=True, exist_ok=True)
    lab_f = (OUT / "labels.jsonl").open("w", encoding="utf-8")

    n_tracks_ok = n_crops = 0
    states: Counter = Counter()
    agrees = []

    for i, (key, frames) in enumerate(elig, 1):
        reads: list[tuple[str, float]] = []
        # Keep every frame's crop and its own read, so once the vote settles
        # the winning label can be attached to ALL of that vehicle's crops.
        per_frame: list[tuple[np.ndarray, str]] = []
        for fr in frames:
            img = cv2.imread(fr["path"])
            if img is None:
                continue
            b = band_of(img)
            if b is None:
                continue
            for box, txt, cf in reader.readtext(b, allowlist=ALLOW, batch_size=8):
                if cf < args.conf:
                    continue
                cleaned = "".join(c for c in txt.upper() if c.isalnum())
                if len(cleaned) < 8 or any(t in cleaned for t in OSD):
                    continue
                d = decode_plate(cleaned)
                if not (d["plate"] and d["score"] >= 0.85):
                    continue
                reads.append((d["plate"], float(cf) * (fr["plate_px_est"] ** 0.5)))
                xs = [p[0] for p in box]
                ys = [p[1] for p in box]
                x1, x2 = int(min(xs)), int(max(xs))
                y1, y2 = int(min(ys)), int(max(ys))
                px, py = int((x2 - x1) * 0.05), int((y2 - y1) * 0.22)
                cx1, cy1 = max(0, x1 - px), max(0, y1 - py)
                cx2 = min(b.shape[1], x2 + px)
                cy2 = min(b.shape[0], y2 + py)
                tight = b[cy1:cy2, cx1:cx2]
                if tight.size and tight.shape[1] >= 32 and tight.shape[0] >= 10:
                    per_frame.append((tight.copy(), d["plate"]))
                break

        if len(reads) < args.min_votes:
            continue
        plate, agree, nv = vote(reads)
        d = decode_plate(plate or "")
        if not d["plate"] or agree < args.min_agreement:
            continue
        final, state, _note = apply_state_prior(d["plate"], d["state"], agree, nv)

        # Attach the VOTED label to every crop of this vehicle. Individual
        # frames were often wrong; the consensus is what we trust.
        for j, (crop, _own) in enumerate(per_frame):
            name = f"{key[0]}_{key[1]}_t{key[2]}_{j:02d}.jpg"
            cv2.imwrite(str(OUT / "images" / name), crop,
                        [cv2.IMWRITE_JPEG_QUALITY, 96])
            lab_f.write(json.dumps({
                "file": name, "text": final, "state": state,
                "agreement": round(agree, 3), "votes": nv,
                "camera": key[0], "track": key[2],
            }) + "\n")
            n_crops += 1
        n_tracks_ok += 1
        states[state] += 1
        agrees.append(agree)

        if i % 200 == 0:
            print(f"  {i}/{len(elig)} tracks | {n_tracks_ok} labelled "
                  f"| {n_crops} crops", flush=True)

    lab_f.close()
    print("\n" + "=" * 60)
    print(f"vehicles labelled : {n_tracks_ok} / {len(elig)} "
          f"({n_tracks_ok/max(len(elig),1)*100:.1f}%)")
    print(f"training crops    : {n_crops}")
    if agrees:
        a = np.array(agrees)
        print(f"agreement         : median {np.median(a):.3f}  "
              f"min {a.min():.3f}")
    print(f"states            : {dict(states.most_common(8))}")
    print(f"\nsaved -> {OUT}")
    print("\nSTEP 2 NEXT: hand-check 50 of these against their images. If under")
    print("90% are correct, raise --min-agreement and rerun. Training on wrong")
    print("labels produces a confidently wrong model, which is worse than none.")


if __name__ == "__main__":
    main()
