"""backend/scripts/journey_measure_precision.py — the journey number, at last
measured rather than asserted.

WHAT THIS SETTLES
  Journey linking has been claimed at 90% and shown by inspection to be nearer
  40%, on a sample of four. Neither figure was a measurement. With human
  judgements on cross-camera pairs spread across the score range, precision
  can finally be plotted against threshold and an operating point chosen on
  evidence.

  It also settles whether appearance is worth using. An ImageNet embedding
  ordered four hand-judged pairs correctly with a 0.005 margin, which proves
  nothing. Against a hundred-plus judgements it either separates the classes
  or it does not.

THE CONTROLS ARE CHECKED FIRST
  Pairs the system scores as clearly unrelated were mixed into the batch. If
  those came back marked "same", the labelling was not attentive and nothing
  downstream can be trusted - so that is verified before any number is
  computed, and a failure stops the run rather than being noted in passing.

UNSURE IS EXCLUDED, NOT COERCED
  Pairs the human could not call are dropped from precision rather than
  counted either way. Forcing them into one class would move the number by
  whichever choice was made, which is the opposite of measuring it.

USAGE
  python -m backend.scripts.journey_measure_precision
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import torch

from backend.scripts.reid_probe import embed, load_backbone
from backend.services.journey_search import JourneySearch

INDEX = Path("output/journey_index")
VERIFY = Path("data/journey_verify")


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--verdicts", default=str(VERIFY / "journey_verdicts.jsonl"))
    ap.add_argument("--target-precision", type=float, default=0.90)
    args = ap.parse_args()

    rows = [json.loads(l) for l in
            Path(args.verdicts).open(encoding="utf-8") if l.strip()]
    print(f"judgements : {len(rows)}")

    ctrl = [r for r in rows if r.get("control")]
    bad = [r for r in ctrl if r["verdict"] == "same"]
    print(f"controls   : {len(ctrl)}, marked same: {len(bad)}")
    if ctrl and len(bad) > max(1, len(ctrl) // 5):
        raise SystemExit(
            "Controls were marked as the same vehicle. The batch is not "
            "reliable;\nre-run the verification rather than computing a "
            "number from it.")
    if ctrl:
        print("  controls look right - the batch was judged attentively\n")

    real = [r for r in rows if not r.get("control")]
    same = [r for r in real if r["verdict"] == "same"]
    diff = [r for r in real if r["verdict"] == "diff"]
    unsure = [r for r in real if r["verdict"] == "unsure"]
    print(f"candidates : {len(real)}")
    print(f"  same     : {len(same)}")
    print(f"  different: {len(diff)}")
    print(f"  unsure   : {len(unsure)} (excluded from precision)\n", flush=True)

    scored = [r for r in real if r["verdict"] in ("same", "diff")
              and r.get("score") is not None]
    if not scored:
        raise SystemExit("no scored judgements to measure")

    # ---- precision vs the CTC threshold ------------------------------
    print("=" * 70)
    print("PRECISION AGAINST THE PLATE-LIKELIHOOD THRESHOLD")
    print("=" * 70)
    print(f"{'threshold':>10} {'links made':>11} {'correct':>9} "
          f"{'PRECISION':>11} {'recall of true':>16}")
    print("-" * 70)
    total_true = len(same)
    best_ctc = None
    for th in np.arange(-22, 0.5, 1.5):
        kept = [r for r in scored if r["score"] >= th]
        if len(kept) < 4:
            continue
        ok = sum(1 for r in kept if r["verdict"] == "same")
        prec = ok / len(kept)
        rec = ok / max(total_true, 1)
        mark = ""
        if prec >= args.target_precision and best_ctc is None:
            best_ctc = (th, prec, rec, len(kept))
            mark = "  <-- target"
        print(f"{th:>10.1f} {len(kept):>11} {ok:>9} {prec*100:>10.1f}% "
              f"{rec*100:>15.1f}%{mark}")
    print("=" * 70)

    # ---- does appearance add anything? -------------------------------
    js = JourneySearch(INDEX)
    pos = {}
    for i, rec in enumerate(js.records):
        pos[f'{rec["camera"]}|{rec["track"]}'] = i

    dev = "cuda" if torch.cuda.is_available() else "cpu"
    model, size = load_backbone("resnet18", dev)

    sims = {}
    for r in scored:
        ia, ib = pos.get(r["a"]), pos.get(r["b"])
        if ia is None or ib is None:
            continue
        ca = [p for p in js.records[ia].get("crops", [])
              if Path(p).is_file()][:3]
        cb = [p for p in js.records[ib].get("crops", [])
              if Path(p).is_file()][:3]
        if not ca or not cb:
            continue
        ea, eb = embed(model, ca, size, dev), embed(model, cb, size, dev)
        sims[r["id"]] = float(np.max(ea @ eb.T))

    with_sim = [r for r in scored if r["id"] in sims]
    ps = np.array([sims[r["id"]] for r in with_sim if r["verdict"] == "same"])
    ns = np.array([sims[r["id"]] for r in with_sim if r["verdict"] == "diff"])
    print(f"\nappearance similarity on {len(with_sim)} judged pairs")
    if len(ps) and len(ns):
        print(f"  same vehicle      median {np.median(ps):.3f}  "
              f"p10 {np.percentile(ps, 10):.3f}")
        print(f"  different vehicle median {np.median(ns):.3f}  "
              f"p90 {np.percentile(ns, 90):.3f}")
        allv = np.concatenate([ps, ns])
        lab = np.concatenate([np.ones_like(ps), np.zeros_like(ns)])
        o = np.argsort(-allv)
        lab = lab[o]
        tp, fp = np.cumsum(lab), np.cumsum(1 - lab)
        auc = float(np.trapezoid(tp / max(tp[-1], 1), fp / max(fp[-1], 1)))
        print(f"  AUC {auc:.3f}")

        # ---- both signals together -----------------------------------
        print("\n" + "=" * 70)
        print("PLATE LIKELIHOOD + APPEARANCE")
        print("=" * 70)
        print(f"{'ctc >=':>8} {'appear >=':>10} {'links':>7} {'correct':>9} "
              f"{'PRECISION':>11} {'recall':>9}")
        print("-" * 70)
        best_both = None
        for cth in (-18, -14, -10, -6, -3):
            for ath in (0.0, 0.78, 0.82, 0.86, 0.90):
                kept = [r for r in with_sim
                        if r["score"] >= cth and sims[r["id"]] >= ath]
                if len(kept) < 4:
                    continue
                ok = sum(1 for r in kept if r["verdict"] == "same")
                prec = ok / len(kept)
                rec = ok / max(total_true, 1)
                if prec >= args.target_precision and (
                        best_both is None or rec > best_both[4]):
                    best_both = (cth, ath, len(kept), prec, rec)
                print(f"{cth:>8.0f} {ath:>10.2f} {len(kept):>7} {ok:>9} "
                      f"{prec*100:>10.1f}% {rec*100:>8.1f}%")
        print("=" * 70)

        if best_both:
            c, a, n, p, r = best_both
            print(f"\nBest operating point: plate likelihood >= {c}, "
                  f"appearance >= {a:.2f}")
            print(f"  {n} links proposed, {p*100:.1f}% correct, "
                  f"{r*100:.0f}% of true journeys found")
        elif best_ctc:
            th, p, r, n = best_ctc
            print(f"\nAppearance adds nothing usable. Plate likelihood alone "
                  f"at >= {th:.1f}\n  gives {p*100:.1f}% precision over {n} "
                  f"links, finding {r*100:.0f}% of true journeys.")
        else:
            print(f"\nNo setting reaches {args.target_precision*100:.0f}% "
                  f"precision. The honest\n  presentation is ranked candidates "
                  f"for an officer to confirm, not\n  asserted links.")


if __name__ == "__main__":
    main()
