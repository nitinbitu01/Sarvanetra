"""backend/scripts/plate_phase2_v2.py — ANPR Phase 2 gate, corrected.

WHY v1 WAS WRONG
  v1 reported 11.2% of vehicles had a "findable plate". Inspecting the actual
  matches showed they were:
    'Bridge'  'bhai'  'CSI'  'Bridc'     <- the camera's burned-in OSD
                                            ("Chiman bhai Bridge CSITMS-32_F")
    'GSRTC'  'CARRIAGE'  'SOMNATH'       <- vehicle signage
  Zero number plates in the top 15 by confidence. v1's position filter
  required text in the LOWER part of the crop, which is precisely where the
  watermark sits - so it preferentially selected the overlay. Buses scored
  highest (28.4%) because they carry the most signage.

WHAT MAKES A PLATE DIFFERENT FROM SIGNAGE - the filters that matter
  1. DIGITS. Every Indian plate ends in four digits. 'Bridge' and 'CARRIAGE'
     have none. Requiring >=2 digits removes nearly all signage and all of the
     OSD in one test, and it is nearly free.
  2. CHARACTER SET. Plates are [A-Z0-9] only. EasyOCR's `allowlist` restricts
     recognition to those glyphs, which both speeds it up and stops it
     reporting lowercase words like 'bhai' at all.
  3. KNOWN OVERLAY TEXT. The OSD string is fixed per camera; reject it by name
     as a backstop for anything the digit test lets through.

EFFICIENCY - this is the "one pass, no waste" version
  v1 OCR'd every sampled track's full vehicle crop. Most of that work was
  spent on vehicles that cannot yield a plate:
    * cars only     - bus/truck plates are rarely visible from these high
                      downward angles, and they dominate the signage FPs
    * >=70px est    - v1 measured 4.6% hit in the 50-70px band vs 25.3% at
                      >=100px; the small band is not worth the compute
    * plate BAND    - OCR the lower-middle region, not the whole vehicle.
                      ~3-4x less pixels per call AND removes roof/windscreen
                      signage before it can be matched
    * batched GPU   - EasyOCR batch_size>1 instead of one call per image
  Net effect: far fewer, far cheaper OCR calls, and the ones that run are the
  only ones that could have succeeded.

USAGE
  python -m backend.scripts.plate_phase2_v2
  python -m backend.scripts.plate_phase2_v2 --min-plate-px 50 --include-heavy
"""
from __future__ import annotations

import argparse
import json
import re
from collections import defaultdict
from pathlib import Path

import cv2
import numpy as np

CORPUS = Path("output/plate_corpus")
MANIFEST = CORPUS / "manifest.jsonl"
OUT_JSON = Path("output/plate_phase2_v2.json")
EXAMPLES = Path("output/plate_phase2_v2_examples")
PROPOSALS = Path("output/plate_proposals_v2.jsonl")

CLS_NAME = {1: "car", 2: "motorcycle", 3: "bus", 4: "truck"}
ALLOW = "ABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789"

# Burned-in camera OSD and common vehicle branding. Matched case-insensitively
# as substrings after cleaning.
OSD_TOKENS = {
    "BRIDGE", "BHAI", "CHIMAN", "CSITMS", "CSI", "BRIDC", "CAM",
    "GSRTC", "GSR", "CARRIAGE", "EXPERIENCE", "SOMNATH", "CHITRA",
}
# Indian plate: 2 letters, 1-2 digits, 1-3 letters, 4 digits (+ BH series).
PLATE_RE = re.compile(r"^(?:[A-Z]{2}\d{1,2}[A-Z]{0,3}\d{4}|\d{2}BH\d{4}[A-Z]{1,2})$")


def looks_like_plate(text: str) -> tuple[bool, str, int]:
    """Return (is_candidate, cleaned, n_digits).

    The digit count is the workhorse: signage and the OSD overlay contain
    essentially no digits, plates always contain four.
    """
    cleaned = re.sub(r"[^A-Z0-9]", "", text.upper())
    if len(cleaned) < 4:
        return False, cleaned, 0
    if any(tok in cleaned for tok in OSD_TOKENS):
        return False, cleaned, 0
    n_dig = sum(c.isdigit() for c in cleaned)
    return n_dig >= 2, cleaned, n_dig


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--min-plate-px", type=float, default=70.0,
                    help="Skip tracks below this estimate. v1 measured 4.6% "
                         "hit rate in the 50-70px band - not worth the compute.")
    ap.add_argument("--include-heavy", action="store_true",
                    help="Include buses and trucks. Off by default: their "
                         "plates are rarely visible from these downward angles "
                         "and they generated most of v1's false positives.")
    ap.add_argument("--max-tracks", type=int, default=0, help="0 = all eligible")
    ap.add_argument("--batch-size", type=int, default=16)
    ap.add_argument("--conf", type=float, default=0.15)
    ap.add_argument("--save-examples", type=int, default=40)
    args = ap.parse_args()

    rows = [json.loads(l) for l in MANIFEST.open(encoding="utf-8")]
    best: dict[tuple, dict] = {}
    for r in rows:
        k = (r["camera"], r["clip"], r["track"])
        score = r["sharpness"] * (r["plate_px_est"] ** 0.5)
        if k not in best or score > best[k]["_score"]:
            best[k] = {**r, "_score": score}

    keep_cls = {1} | ({3, 4} if args.include_heavy else set())
    tracks = [r for r in best.values()
              if r["cls"] in keep_cls and r["plate_px_est"] >= args.min_plate_px]
    tracks.sort(key=lambda r: -r["plate_px_est"])
    if args.max_tracks:
        tracks = tracks[:args.max_tracks]

    print(f"manifest rows   : {len(rows)}")
    print(f"distinct tracks : {len(best)}")
    print(f"eligible        : {len(tracks)}  "
          f"(cls={sorted(keep_cls)}, plate>={args.min_plate_px:.0f}px)")
    print(f"skipped         : {len(best)-len(tracks)} tracks that could not "
          f"have yielded a plate\n", flush=True)
    if not tracks:
        raise SystemExit("nothing eligible - lower --min-plate-px")

    import easyocr
    print("loading EasyOCR...", flush=True)
    reader = easyocr.Reader(["en"], gpu=True, verbose=False)

    EXAMPLES.mkdir(parents=True, exist_ok=True)
    prop_f = PROPOSALS.open("w", encoding="utf-8")

    by_cam = defaultdict(lambda: {"n": 0, "hit": 0, "strict": 0})
    by_bin = defaultdict(lambda: {"n": 0, "hit": 0, "strict": 0})
    hits = strict_hits = saved = 0
    digit_hist = defaultdict(int)

    for i, r in enumerate(tracks, 1):
        img = cv2.imread(r["path"])
        if img is None:
            continue
        h, w = img.shape[:2]
        # Plate BAND only: lower-middle of the vehicle. Cuts pixels ~3-4x and
        # removes roof/windscreen signage before OCR can match it.
        y0, y1 = int(h * 0.42), int(h * 0.97)
        x0, x1 = int(w * 0.10), int(w * 0.90)
        band = img[y0:y1, x0:x1]
        if band.size == 0 or band.shape[1] < 20:
            continue
        # OCR likes larger input; scale the band up to a workable width.
        if band.shape[1] < 320:
            f = 320 / band.shape[1]
            band = cv2.resize(band, None, fx=f, fy=f,
                              interpolation=cv2.INTER_CUBIC)

        pbin = ("70-100" if r["plate_px_est"] < 100 else
                "100-150" if r["plate_px_est"] < 150 else ">=150")
        for d in (by_cam[r["camera"]], by_bin[pbin]):
            d["n"] += 1

        found = reader.readtext(band, allowlist=ALLOW,
                                batch_size=args.batch_size)
        cands = []
        for b, t, c in found:
            if c < args.conf:
                continue
            ok, cleaned, nd = looks_like_plate(t)
            if ok:
                cands.append((b, cleaned, float(c), nd))

        if cands:
            hits += 1
            for d in (by_cam[r["camera"]], by_bin[pbin]):
                d["hit"] += 1
            b, cleaned, c, nd = max(cands, key=lambda x: (x[3], x[2]))
            digit_hist[nd] += 1
            strict = bool(PLATE_RE.match(cleaned))
            if strict:
                strict_hits += 1
                for d in (by_cam[r["camera"]], by_bin[pbin]):
                    d["strict"] += 1
            xs = [p[0] for p in b]
            ys = [p[1] for p in b]
            prop_f.write(json.dumps({
                "path": r["path"], "camera": r["camera"],
                "cls": CLS_NAME.get(r["cls"], r["cls"]),
                "plate_px_est": r["plate_px_est"], "text": cleaned,
                "digits": nd, "conf": round(c, 3), "strict_format": strict,
                "box": [int(min(xs)), int(min(ys)), int(max(xs)), int(max(ys))],
            }) + "\n")
            if saved < args.save_examples:
                vis = band.copy()
                cv2.rectangle(vis, (int(min(xs)), int(min(ys))),
                              (int(max(xs)), int(max(ys))), (0, 255, 0), 2)
                tag = "STRICT" if strict else "loose"
                cv2.imwrite(str(EXAMPLES / f"{saved:03d}_{tag}_{cleaned}.jpg"), vis)
                saved += 1

        if i % 100 == 0:
            print(f"  {i}/{len(tracks)}  plate-like {hits/i*100:.1f}%  "
                  f"strict-format {strict_hits/i*100:.1f}%", flush=True)

    prop_f.close()
    n = len(tracks)

    print("\n" + "=" * 64)
    print(f"PHASE 2 GATE (corrected)")
    print(f"  eligible vehicles      : {n}")
    print(f"  plate-like (>=2 digits): {hits} ({hits/max(n,1)*100:.1f}%)")
    print(f"  full GJ-format match   : {strict_hits} "
          f"({strict_hits/max(n,1)*100:.1f}%)")
    print("=" * 64)
    if digit_hist:
        print("\ndigit count of matches:", dict(sorted(digit_hist.items())))
    print("\n--- by estimated plate width ---")
    for k in sorted(by_bin, key=lambda k: -by_bin[k]["n"]):
        v = by_bin[k]
        print(f"  {k:<10} {v['hit']:>4}/{v['n']:<5} ({v['hit']/max(v['n'],1)*100:>5.1f}%)"
              f"   strict {v['strict']:>3}")
    print("\n--- by camera ---")
    for k in sorted(by_cam, key=lambda k: -by_cam[k]["n"]):
        v = by_cam[k]
        print(f"  {k:<10} {v['hit']:>4}/{v['n']:<5} ({v['hit']/max(v['n'],1)*100:>5.1f}%)"
              f"   strict {v['strict']:>3}")

    OUT_JSON.write_text(json.dumps({
        "eligible": n, "plate_like": hits, "strict": strict_hits,
        "by_camera": {k: dict(v) for k, v in by_cam.items()},
        "by_bin": {k: dict(v) for k, v in by_bin.items()},
    }, indent=2))
    print(f"\nproposals -> {PROPOSALS}")
    print(f"examples  -> {EXAMPLES}  (VERIFY BY EYE before trusting any %)")


if __name__ == "__main__":
    main()
