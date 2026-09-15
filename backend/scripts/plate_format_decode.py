"""backend/scripts/plate_format_decode.py — how many raw OCR reads are actually
valid plates once positional constraints are applied?

THE OBSERVATION THIS IS BUILT ON
  Phase 2 v2 reported only 0.3% of reads matching the Gujarat plate format.
  But the raw strings tell a different story:

      6J03085274   6J01K02977   6J32K4588   6J198R4294

  '6J' is 'GJ' with a G->6 misread. These ARE Gujarat plates. Raw OCR makes a
  small, PREDICTABLE set of glyph confusions, and an unconstrained matcher
  throws away every read containing one.

WHY POSITION RESOLVES IT
  An Indian plate is  LL DD L{1,3} DDDD  - e.g. GJ 01 KP 7667. So the character
  class at each position is known in advance, and the ambiguity collapses:

      position 0-1 must be LETTERS -> a '6' there can only be 'G'
      position 2-3 must be DIGITS  -> an 'O' there can only be '0'

  The same glyph maps in opposite directions depending on where it sits, which
  is why this cannot be done with a global find-and-replace and why it is worth
  a real decoder. No model, no training - pure logic, and close to free.

  RTO validation is the second gate: Gujarat districts run GJ-01..GJ-38, so a
  decoded 'GJ99' is rejected outright.

USAGE
  python -m backend.scripts.plate_format_decode
"""
from __future__ import annotations

import json
import re
from collections import Counter
from pathlib import Path

PROPOSALS = Path("output/plate_proposals_v2.jsonl")
OUT = Path("output/plate_decoded.jsonl")

# Glyph confusions, resolved by the character class the position demands.
TO_LETTER = {"6": "G", "0": "O", "1": "I", "5": "S", "8": "B", "2": "Z", "4": "A"}
TO_DIGIT = {"O": "0", "Q": "0", "D": "0", "I": "1", "L": "1", "S": "5",
            "B": "8", "Z": "2", "G": "6", "A": "4", "T": "7"}

VALID_RTO = {f"GJ{i:02d}" for i in range(1, 39)}          # GJ-01 .. GJ-38
PLATE_RE = re.compile(r"^[A-Z]{2}\d{2}[A-Z]{1,3}\d{4}$")


def decode(raw: str) -> tuple[str | None, str]:
    """Force `raw` into plate grammar. Returns (plate or None, reason)."""
    s = re.sub(r"[^A-Z0-9]", "", raw.upper())
    # The trailing 4 digits and leading 2 letters are the anchors; the middle
    # letter group is 1-3 chars, so total length is 9-11.
    if not (9 <= len(s) <= 11):
        return None, f"length {len(s)}"

    state = s[:2]
    nums = s[2:4]
    series = s[4:-4]
    last4 = s[-4:]

    state = "".join(TO_LETTER.get(c, c) if not c.isalpha() else c for c in state)
    nums = "".join(TO_DIGIT.get(c, c) if not c.isdigit() else c for c in nums)
    series = "".join(TO_LETTER.get(c, c) if not c.isalpha() else c for c in series)
    last4 = "".join(TO_DIGIT.get(c, c) if not c.isdigit() else c for c in last4)

    cand = f"{state}{nums}{series}{last4}"
    if not PLATE_RE.match(cand):
        return None, f"grammar {cand}"
    if state + nums not in VALID_RTO:
        return None, f"bad RTO {state}{nums}"
    return cand, "ok"


def main() -> None:
    rows = [json.loads(l) for l in PROPOSALS.open(encoding="utf-8")]
    print(f"raw proposals : {len(rows)}")
    raw_strict = sum(r["strict_format"] for r in rows)
    print(f"strict BEFORE : {raw_strict} ({raw_strict/max(len(rows),1)*100:.1f}%)\n")

    ok, reasons = [], Counter()
    for r in rows:
        plate, why = decode(r["text"])
        if plate:
            ok.append({**r, "decoded": plate})
        else:
            reasons[why.split()[0]] += 1

    print("--- DECODED as valid Gujarat plates ---")
    for r in sorted(ok, key=lambda r: -r["conf"]):
        print(f"  {r['camera']:<8} {r['plate_px_est']:>5.0f}px  "
              f"'{r['text']}'  ->  {r['decoded']}   (conf {r['conf']:.2f})")

    print(f"\nvalid after decode : {len(ok)} of {len(rows)} "
          f"({len(ok)/max(len(rows),1)*100:.1f}%)")
    print(f"strict before      : {raw_strict}")
    print(f"gain               : {len(ok) - raw_strict} additional plates "
          f"recovered by positional constraints alone")
    print(f"\nrejection reasons  : {dict(reasons)}")

    OUT.write_text("\n".join(json.dumps(r) for r in ok))
    print(f"\nsaved -> {OUT}")
    print("\nCAVEAT: 'valid format' is NOT 'correct'. These are plausible "
          "plates, not verified ones. Confirming they match the vehicle needs "
          "human labelling - that is Phase 6, and it is not optional.")


if __name__ == "__main__":
    main()
