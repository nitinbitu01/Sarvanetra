"""Does the wired-in rectifier help, harm, or cost — measured through the engine.

The rectifier is now inside ANPREngine.recognize_plate rather than in a
research script, so this measures the path production actually takes, with
SENTINEL_RECTIFY toggled on and off over identical crops.

Three questions, because a change can pass one and fail another:

  accuracy   does exact match move, per crop and per vehicle after voting
  safety     how many plates it fixes against how many it breaks, with a
             McNemar p — a net of zero can hide five of each
  cost       milliseconds added per crop, and how often the expensive path is
             even entered, since the tilt gate skips it for most crops

The design intent is that this cannot lose: variants only replace the
original when the recogniser is strictly more confident about them. This
verifies that intent against real crops rather than trusting it.

Run:  python -m backend.scripts.verify_rectify_wired
"""
from __future__ import annotations

import json
import random
import sys
import time
from collections import defaultdict
from math import comb
from pathlib import Path

import cv2
import numpy as np

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
if str(ROOT / "backend") not in sys.path:
    sys.path.insert(0, str(ROOT / "backend"))

from backend.scripts.train_plate_recognizer import STOI              # noqa: E402

REAL = ROOT / "data" / "plate_real"


def mcnemar(fixed: int, broke: int) -> float:
    n = fixed + broke
    if n == 0:
        return 1.0
    p = sum(comb(n, i) for i in range(min(fixed, broke) + 1)) / (2 ** n) * 2
    return min(p, 1.0)


def main() -> int:
    import services.anpr_engine as AE
    from services.anpr_engine import get_anpr_engine

    engine = get_anpr_engine()

    truth = {}
    for line in (REAL / "verified_all.jsonl").open(encoding="utf-8"):
        v = json.loads(line)
        if v.get("text") and all(c in STOI for c in v["text"]):
            truth[(v["camera"], v["track"])] = v["text"]
    by_track = defaultdict(list)
    for line in (REAL / "labels.jsonl").open(encoding="utf-8"):
        r = json.loads(line)
        k = (r["camera"], r["track"])
        if k in truth:
            by_track[k].append(r["file"])

    kk = sorted(by_track)
    random.Random(1000).shuffle(kk)
    clean = kk[:max(20, int(len(kk) * 0.20))]

    want_h, want_w = getattr(engine, "_rec_hw", (64, 256))
    results = {}
    per_crop = {}
    timing = {}

    for flag in (False, True):
        AE.RECTIFY_ENABLED = flag
        ok_crop, ok_veh, n_crop = 0, 0, 0
        outcomes = []
        t_total = 0.0
        for k in clean:
            gt = truth[k]
            texts = []
            for f in by_track[k]:
                g = cv2.imread(str(REAL / "images" / f), cv2.IMREAD_GRAYSCALE)
                if g is None or g.shape[1] < 70:
                    continue
                strip = cv2.resize(g, (want_w, want_h),
                                   interpolation=cv2.INTER_AREA)
                raw_bgr = cv2.cvtColor(g, cv2.COLOR_GRAY2BGR)
                t0 = time.perf_counter()
                plate, conf, _ = engine.recognize_plate(
                    strip, raw_bgr_crop=raw_bgr, agreement=0.5, votes=1)
                t_total += time.perf_counter() - t0
                n_crop += 1
                hit = (plate == gt)
                ok_crop += hit
                if plate:
                    texts.append((plate, conf))
            # Per vehicle: the most confident read, which is what a single
            # pass of the engine settles on before temporal voting adds more.
            if texts:
                best = max(texts, key=lambda t: t[1])[0]
                v_ok = (best == gt)
                ok_veh += v_ok
                outcomes.append(v_ok)
            else:
                outcomes.append(False)
        results[flag] = (100 * ok_crop / max(n_crop, 1),
                         100 * ok_veh / max(len(outcomes), 1))
        per_crop[flag] = outcomes
        timing[flag] = 1000.0 * t_total / max(n_crop, 1)
        print(f"  rectify={'ON ' if flag else 'OFF'}  "
              f"{n_crop} crops read", flush=True)

    off_c, off_v = results[False]
    on_c, on_v = results[True]
    fixed = sum(1 for a, b in zip(per_crop[False], per_crop[True]) if b and not a)
    broke = sum(1 for a, b in zip(per_crop[False], per_crop[True]) if a and not b)

    print(f"\n{'':<14}{'per crop':>11}{'per vehicle':>14}{'ms/crop':>10}")
    print("-" * 50)
    print(f"{'rectify OFF':<14}{off_c:>10.1f}%{off_v:>13.1f}%{timing[False]:>10.2f}")
    print(f"{'rectify ON':<14}{on_c:>10.1f}%{on_v:>13.1f}%{timing[True]:>10.2f}")
    print(f"{'delta':<14}{on_c-off_c:>+10.1f}%{on_v-off_v:>+13.1f}%"
          f"{timing[True]-timing[False]:>+10.2f}")
    print(f"\nfixed {fixed}   broke {broke}   McNemar p={mcnemar(fixed, broke):.3f}")

    # How often the expensive path is entered at all.
    tilts = []
    for k in clean:
        for f in by_track[k]:
            g = cv2.imread(str(REAL / "images" / f), cv2.IMREAD_GRAYSCALE)
            if g is not None and g.shape[1] >= 70:
                tilts.append(abs(engine._plate_tilt_deg(g)))
    tilts = np.array(tilts)
    gated = ((tilts >= AE.RECTIFY_MIN_TILT_DEG)
             & (tilts <= AE.RECTIFY_MAX_TILT_DEG))
    print(f"\ntilt gate: {gated.sum()}/{len(tilts)} crops "
          f"({100*gated.mean():.1f}%) are tilted enough to rectify.")
    print(f"the other {100*(1-gated.mean()):.1f}% cost one Otsu threshold and "
          f"nothing else.")
    print(f"median tilt on this fleet: {np.median(tilts):.2f} degrees")

    if broke > fixed:
        print("\nVERDICT: it breaks more than it fixes — leave SENTINEL_RECTIFY=0.")
    elif fixed > broke:
        print("\nVERDICT: net positive on this holdout.")
    else:
        print("\nVERDICT: no plate changed outcome. Safe, and idle on this "
              "fleet's geometry — it earns its place on tilted footage, "
              "not here.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
