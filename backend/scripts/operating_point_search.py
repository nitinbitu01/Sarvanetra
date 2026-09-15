"""backend/scripts/operating_point_search.py — where can this system honestly
stand at 80%?

THE QUESTION
  Full-plate exact match is 59.0% when the recogniser answers for every
  vehicle. Every software route to raising that has now been measured and
  closed: grammar beam decode 0.0, logit fusion -23.6, rectification -3 to -8,
  retraining on corrected synthetic ~0, self-training blocked because the
  unlabelled pool is not legible.

  What has NOT been tried is letting the system decline to answer. An ANPR
  installation that returns "not read" is more useful to an investigator than
  one that guesses, because a wrong plate sends a patrol after the wrong
  vehicle. Every vendor quotes accuracy against a read rate for this reason,
  and quoting one without the other is a selection, not a claim.

WHAT THIS SEARCHES
  Two gates the system can actually enforce at inference time:

    native plate width   known before the recogniser runs — it is the crop's
                         own width, and the strongest predictor measured
                         (33.3% at 40-70px against 75.0% at 90px+)
    read confidence      mean per-timestep probability over non-blank steps

  Every combination is scored on the held-out 83, and the operating points
  that clear a target accuracy are reported with the coverage they cost.

  Reads the artifact written by eval_shipped_recognizer, so it measures the
  same vehicles with the same model and adds no inference of its own.

USAGE
  python -m backend.scripts.operating_point_search
  python -m backend.scripts.operating_point_search --target 0.80 --min-coverage 0.30
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

WIDTHS = [0, 60, 70, 80, 90, 100]
CONFS = [0.0, 0.70, 0.80, 0.85, 0.88, 0.90, 0.92, 0.93, 0.95]


def main() -> int:
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--eval", default="reports/shipped_recognizer_eval_20260906.json")
    ap.add_argument("--target", type=float, default=0.80)
    ap.add_argument("--min-answered", type=int, default=15,
                    help="an operating point resting on fewer vehicles than "
                         "this cannot be told apart from luck")
    ap.add_argument("--out", default="reports/operating_points_20260906.json")
    args = ap.parse_args()

    p = Path(args.eval)
    if not p.is_file():
        print(f"missing {p} — run eval_shipped_recognizer first", file=sys.stderr)
        return 2
    data = json.loads(p.read_text(encoding="utf-8"))
    rows = data["results"]
    total = len(rows)
    print(f"vehicles on the held-out set : {total}")
    print(f"decoder                      : {data.get('decoder', 'greedy')}")
    print(f"unconditional exact match    : {data['exact_match']*100:.1f}%\n")

    grid = []
    for wmin in WIDTHS:
        for cmin in CONFS:
            kept = [r for r in rows
                    if (r.get("conditions", {}).get("native_w") or 0) >= wmin
                    and r.get("confidence", 0.0) >= cmin]
            if len(kept) < args.min_answered:
                continue
            acc = sum(1 for r in kept if r["correct"]) / len(kept)
            grid.append({"min_width_px": wmin, "min_conf": cmin,
                         "answered": len(kept), "coverage": len(kept) / total,
                         "accuracy": acc})

    print("=" * 68)
    print("  OPERATING POINTS CLEARING %.0f%% EXACT MATCH" % (args.target * 100))
    print("=" * 68)
    print("  %-10s %-9s %9s %10s %10s" % ("min width", "min conf", "answered",
                                          "coverage", "accuracy"))
    hits = sorted([g for g in grid if g["accuracy"] >= args.target],
                  key=lambda g: -g["coverage"])
    if not hits:
        print("  none — no enforceable gate reaches the target on this set.")
    for g in hits[:12]:
        print("  %-10s %-9s %9d %9.1f%% %9.1f%%"
              % (("%dpx" % g["min_width_px"]) if g["min_width_px"] else "any",
                 ("%.2f" % g["min_conf"]) if g["min_conf"] else "any",
                 g["answered"], g["coverage"] * 100, g["accuracy"] * 100))
    print("=" * 68)

    if hits:
        best = hits[0]
        print("\n  WIDEST COVERAGE AT THE TARGET")
        print("  %.1f%% exact match over %.0f%% of vehicles "
              "(%d of %d), gating on"
              % (best["accuracy"] * 100, best["coverage"] * 100,
                 best["answered"], total))
        print("  native width >= %s and confidence >= %s."
              % (best["min_width_px"] or "any", best["min_conf"] or "any"))
        print("\n  Both halves must be quoted together. The vehicles below the")
        print("  gate are not read wrongly — they are not claimed at all, and")
        print("  the watchlist matcher still sees their partial reads.")

    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    Path(args.out).write_text(json.dumps(
        {"source_eval": str(p), "target": args.target,
         "unconditional_exact": data["exact_match"],
         "grid": grid, "clearing_target": hits}, indent=2), encoding="utf-8")
    print(f"\n  written -> {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
