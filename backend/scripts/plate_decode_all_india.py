"""backend/scripts/plate_decode_all_india.py — decode reads against ALL-INDIA
plate grammar and report what the Gujarat-only decoder was discarding.

Gujarat's roads carry vehicles registered across every Indian state. The first
decoder validated GJ01-GJ38 and dropped everything else, which for a
surveillance system is a silent, unrecoverable failure: if the vehicle in an
incident carries an MH or RJ plate, the read is thrown away at the moment it
matters most.

This re-decodes the same raw OCR strings with the full grammar
(backend/scripts/indian_plate_grammar.py) and prints the state breakdown, so
the cost of the old assumption is measured rather than asserted.

USAGE
  python -m backend.scripts.plate_decode_all_india
  python -m backend.scripts.plate_decode_all_india --min-score 0.85
"""
from __future__ import annotations

import argparse
import json
from collections import Counter
from pathlib import Path

from backend.scripts.indian_plate_grammar import (STATE_CODES, decode_plate,
                                                  is_plausible)

PROPOSALS = Path("output/plate_proposals_v2.jsonl")
OUT = Path("output/plate_decoded_all_india.jsonl")


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--min-score", type=float, default=0.85,
                    help="Structural confidence floor. 0.85 requires a real "
                         "state code; lower it to trade precision for recall.")
    args = ap.parse_args()

    rows = [json.loads(l) for l in PROPOSALS.open(encoding="utf-8")]
    print(f"raw OCR reads : {len(rows)}\n")

    accepted, states, reasons = [], Counter(), Counter()
    for r in rows:
        d = decode_plate(r["text"])
        if is_plausible(d, args.min_score):
            accepted.append({**r, **{k: d[k] for k in
                                     ("plate", "state", "format", "score")}})
            states[d["state"]] += 1
        else:
            reasons[d["reason"].split()[0] or "low"] += 1

    print("--- decoded plates ---")
    for r in sorted(accepted, key=lambda r: -r["conf"]):
        flag = "" if r["state"] == "GJ" else "   <- OUT OF STATE"
        print(f"  {r['camera']:<8} {r['plate_px_est']:>5.0f}px  "
              f"'{r['text']}' -> {r['plate']:<12} [{r['state']}] "
              f"conf {r['conf']:.2f}{flag}")

    gj = states.get("GJ", 0)
    other = sum(states.values()) - gj
    print(f"\naccepted        : {len(accepted)} of {len(rows)}")
    print(f"  Gujarat       : {gj}")
    print(f"  other states  : {other}"
          + (f"  <- these were DROPPED by the GJ-only decoder" if other else ""))
    print(f"\nby state        : {dict(states.most_common())}")
    print(f"rejections      : {dict(reasons)}")

    OUT.write_text("\n".join(json.dumps(r) for r in accepted))
    print(f"\nsaved -> {OUT}")
    print(f"\nGrammar covers {len(STATE_CODES)} state/UT codes plus Bharat and "
          "military series.")
    print("NOTE: 'decoded' still is not 'verified'. Structural validity means "
          "the string could be a real plate, not that it is this vehicle's.")


if __name__ == "__main__":
    main()
