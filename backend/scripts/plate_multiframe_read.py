"""backend/scripts/plate_multiframe_read.py — read every frame of a track, then vote.

THE WASTE THIS FIXES
  Phase 2 picked ONE "best" frame per vehicle and read it. If that frame
  failed, the vehicle was recorded as unread - even though up to 24 other
  frames of the same vehicle sat unused on disk. Yield was 14 plates from 727
  vehicles (1.9%).

  A vehicle crossing the frame is photographed 25 times at different distances,
  angles and motion states. The frame that is sharpest is not necessarily the
  one where the plate is least foreshortened, or least occluded, or best lit.
  Reading all of them and combining the results costs only compute, and for a
  surveillance system a missed plate is unrecoverable while compute is cheap.

CONSENSUS, NOT FIRST-HIT
  Per-frame OCR is noisy in a specific way: it confuses individual glyphs.
  Voting per CHARACTER POSITION across all reads of the same vehicle recovers
  the plate even when NO single frame read it correctly end to end - each frame
  can be wrong somewhere different, and the majority is right at each position.
  That is the mechanism that makes this more than "try harder".

  Votes are weighted by OCR confidence and by plate size, so a large sharp
  frame outweighs a small blurry one rather than counting equally.

USAGE
  python -m backend.scripts.plate_multiframe_read
  python -m backend.scripts.plate_multiframe_read --max-tracks 300 --min-plate-px 50
"""
from __future__ import annotations

import argparse
import json
from collections import Counter, defaultdict
from pathlib import Path

import cv2
import numpy as np

from backend.scripts.indian_plate_grammar import (apply_state_prior,
                                                  decode_plate, is_plausible)

CORPUS = Path("output/plate_corpus")
MANIFEST = CORPUS / "manifest.jsonl"
OUT = Path("output/plate_multiframe.jsonl")
EXAMPLES = Path("output/plate_multiframe_examples")

ALLOW = "ABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789"
OSD = {"BRIDGE", "BHAI", "CHIMAN", "CSITMS", "CSI", "BRIDC", "GSRTC",
       "CARRIAGE", "EXPERIENCE", "SOMNATH", "CHITRA"}


def band_of(img):
    """Plate search region: lower-middle of the vehicle, upscaled for OCR."""
    h, w = img.shape[:2]
    b = img[int(h * 0.42):int(h * 0.97), int(w * 0.08):int(w * 0.92)]
    if b.size == 0 or b.shape[1] < 20:
        return None
    if b.shape[1] < 360:
        f = 360 / b.shape[1]
        b = cv2.resize(b, None, fx=f, fy=f, interpolation=cv2.INTER_CUBIC)
    return b


def vote(reads: list[tuple[str, float]]) -> tuple[str | None, float, int]:
    """Character-position vote across every decoded read of one vehicle.

    Reads are grouped by length first - averaging a 9- and a 10-character read
    position-by-position would be meaningless. The most-supported length wins,
    then each position is decided independently.
    """
    if not reads:
        return None, 0.0, 0
    by_len: dict[int, list[tuple[str, float]]] = defaultdict(list)
    for txt, w in reads:
        by_len[len(txt)].append((txt, w))
    # Prefer the length carrying the most total weight, not the most reads.
    best_len = max(by_len, key=lambda L: sum(w for _, w in by_len[L]))
    group = by_len[best_len]

    out = []
    agree = 0.0
    for i in range(best_len):
        tally: Counter = Counter()
        for txt, w in group:
            tally[txt[i]] += w
        ch, top = tally.most_common(1)[0]
        out.append(ch)
        agree += top / max(sum(tally.values()), 1e-9)
    return "".join(out), agree / best_len, len(group)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--min-plate-px", type=float, default=50.0,
                    help="Lower than Phase 2's 70px: with 25 frames and voting, "
                         "marginal vehicles become worth attempting.")
    ap.add_argument("--max-tracks", type=int, default=400)
    ap.add_argument("--max-frames", type=int, default=25)
    ap.add_argument("--conf", type=float, default=0.10,
                    help="Deliberately low. A weak read still casts a weighted "
                         "vote; it is not accepted on its own.")
    ap.add_argument("--include-heavy", action="store_true")
    ap.add_argument("--save-examples", type=int, default=40)
    args = ap.parse_args()

    rows = [json.loads(l) for l in MANIFEST.open(encoding="utf-8")]
    tracks: dict[tuple, list[dict]] = defaultdict(list)
    for r in rows:
        tracks[(r["camera"], r["clip"], r["track"])].append(r)

    keep_cls = {1} | ({3, 4} if args.include_heavy else set())
    elig = []
    for k, frames in tracks.items():
        best = max(frames, key=lambda r: r["plate_px_est"])
        if best["cls"] in keep_cls and best["plate_px_est"] >= args.min_plate_px:
            frames.sort(key=lambda r: -(r["sharpness"] * r["plate_px_est"]))
            elig.append((k, frames[:args.max_frames], best))
    elig.sort(key=lambda t: -t[2]["plate_px_est"])
    if args.max_tracks:
        elig = elig[:args.max_tracks]

    total_frames = sum(len(f) for _, f, _ in elig)
    print(f"tracks total    : {len(tracks)}")
    print(f"eligible        : {len(elig)}")
    print(f"frames to read  : {total_frames} "
          f"({total_frames/max(len(elig),1):.1f} per vehicle)")
    print("Phase 2 read ONE frame per vehicle and got 14/727 (1.9%).\n",
          flush=True)

    import easyocr
    print("loading EasyOCR...", flush=True)
    reader = easyocr.Reader(["en"], gpu=True, verbose=False)

    EXAMPLES.mkdir(parents=True, exist_ok=True)
    out_f = OUT.open("w", encoding="utf-8")
    hits = saved = 0
    single_hits = 0
    corrected = 0
    states: Counter = Counter()

    for i, (key, frames, best) in enumerate(elig, 1):
        reads: list[tuple[str, float]] = []
        # Remember WHICH frame produced each read, and where the text sat, so
        # the saved example shows the evidence rather than an unrelated frame.
        # (An earlier version saved the sharpest frame regardless, which made
        # visual verification meaningless - a roof-only crop was presented as a
        # confident read.)
        origin: dict[str, tuple[str, tuple]] = {}
        for fr in frames:
            img = cv2.imread(fr["path"])
            if img is None:
                continue
            b = band_of(img)
            if b is None:
                continue
            for box, txt, c in reader.readtext(b, allowlist=ALLOW, batch_size=8):
                if c < args.conf:
                    continue
                cleaned = "".join(ch for ch in txt.upper() if ch.isalnum())
                if len(cleaned) < 8 or any(t in cleaned for t in OSD):
                    continue
                d = decode_plate(cleaned)
                if d["plate"] and d["score"] >= 0.85:
                    # Weight: OCR confidence x plate size. A large sharp frame
                    # should outvote a small blurry one.
                    reads.append((d["plate"], float(c) * (fr["plate_px_est"] ** 0.5)))
                    xs = [p[0] for p in box]
                    ys = [p[1] for p in box]
                    origin.setdefault(d["plate"], (fr["path"],
                                                   (int(min(xs)), int(min(ys)),
                                                    int(max(xs)), int(max(ys)))))

        if not reads:
            continue
        plate, agree, n = vote(reads)
        d = decode_plate(plate or "")
        if not (d["plate"] and is_plausible(d, 0.85)):
            continue

        # Geographic tie-break. Only fires on weak evidence; a repeatedly and
        # confidently read out-of-state plate passes through untouched.
        final, state, note = apply_state_prior(d["plate"], d["state"], agree, n)
        if note:
            corrected += 1

        hits += 1
        if n == 1:
            single_hits += 1
        states[state] += 1
        src_path, src_box = origin.get(d["plate"], (best["path"], None))
        out_f.write(json.dumps({
            "camera": key[0], "clip": key[1], "track": key[2],
            "plate": final, "state": state, "raw_plate": d["plate"],
            "prior_note": note, "score": d["score"],
            "agreement": round(agree, 3), "votes": n,
            "plate_px_est": best["plate_px_est"], "path": src_path,
        }) + "\n")

        if saved < args.save_examples:
            img = cv2.imread(src_path)
            b = band_of(img) if img is not None else None
            if b is not None:
                if src_box:
                    cv2.rectangle(b, src_box[:2], src_box[2:], (0, 255, 0), 2)
                tag = "CORR" if note else "ok"
                cv2.imwrite(str(EXAMPLES /
                            f"{saved:03d}_{final}_{state}_{tag}_"
                            f"v{n}_a{agree:.2f}.jpg"), b)
                saved += 1

        if i % 25 == 0:
            print(f"  {i}/{len(elig)}  plates {hits} "
                  f"({hits/i*100:.1f}%)", flush=True)

    out_f.close()
    n = len(elig)
    print("\n" + "=" * 62)
    print(f"MULTI-FRAME RESULT")
    print(f"  vehicles attempted : {n}")
    print(f"  plates decoded     : {hits} ({hits/max(n,1)*100:.1f}%)")
    print(f"  from a single vote : {single_hits}  "
          f"(rest needed multiple frames)")
    print(f"  state-prior fixes  : {corrected}  "
          f"(weak-evidence rare states rewritten)")
    print(f"  Phase 2 baseline   : 1.9% (one frame per vehicle)")
    print("=" * 62)
    print(f"\nby state : {dict(states.most_common())}")
    print(f"saved    -> {OUT}")
    print(f"examples -> {EXAMPLES}")


if __name__ == "__main__":
    main()
