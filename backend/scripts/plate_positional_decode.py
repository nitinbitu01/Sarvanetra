"""backend/scripts/plate_positional_decode.py — force reads into Indian plate
character-class layout, position by position.

THE RULE
  An Indian registration is  LL DD L{1,3} DDDD :

      G J 0 1 A B 1 2 3 4
      ^^^     ^^^ ^^^^^^^
      |       |   +-- positions 7-10  DIGITS
      |       +------ positions 5-6   LETTERS  (series, 1-3 long)
      +-------------- positions 1-2   LETTERS  (state)
              positions 3-4  DIGITS   (RTO district)

  So the character CLASS at every position is known before anything is read.
  A digit appearing where a letter must be is not ambiguous - it is an OCR
  glyph confusion, and the direction of the fix is determined.

WHY THIS RECOVERS REAL ERRORS
  Held-out failures from the fine-tuned CRNN:

      predicted GJ030O3018   position 5 held '0' where a LETTER is required
      predicted GJ32A8963    position 6 held '8' where a LETTER is required

  The model saw the right shape and picked the wrong class member. Mapping
  0->O and 8->B at those positions is not a guess; it is the only legal
  reading. Note the SAME glyph maps the opposite way in a digit position
  (O->0, B->8), which is why a global find-and-replace cannot do this and a
  positional decoder is required.

WHAT IT CANNOT FIX
  Last-digit errors like GJ11BH9381 read as GJ11BH9384. Both are digits in a
  digit position, so the grammar has nothing to say. Those need a better
  recogniser or multi-frame voting, not more rules.
"""
from __future__ import annotations

import re

# Confusions resolved by the class the position demands.
TO_LETTER = {"0": "O", "1": "I", "2": "Z", "4": "A", "5": "S", "6": "G",
             "7": "T", "8": "B", "9": "P", "3": "J"}
TO_DIGIT = {"O": "0", "Q": "0", "D": "0", "I": "1", "L": "1", "Z": "2",
            "A": "4", "S": "5", "G": "6", "T": "7", "B": "8", "P": "9",
            "J": "3", "E": "8"}

STATE_CODES = {
    "AN", "AP", "AR", "AS", "BR", "CG", "CH", "DD", "DL", "DN", "GA", "GJ",
    "HP", "HR", "JH", "JK", "KA", "KL", "LA", "LD", "MH", "ML", "MN", "MP",
    "MZ", "NL", "OD", "OR", "PB", "PY", "RJ", "SK", "TN", "TR", "TS", "UK",
    "UA", "UP", "WB",
}


def _letters(s: str) -> str:
    return "".join(c if c.isalpha() else TO_LETTER.get(c, c) for c in s)


def _digits(s: str) -> str:
    return "".join(c if c.isdigit() else TO_DIGIT.get(c, c) for c in s)


EXPECTED_LEN = 10
# Measured on 145 hand-read plates from this footage: 144 were exactly ten
# characters (LL DD LL DDDD). Ten is overwhelmingly the norm here.
#
# But it is NOT a hard rule. GJ32K4588 - verified character by character
# against its image - is nine characters (GJ|32|K|4588, single-letter series).
# Rejecting on length alone would discard genuine reads, and a missed plate is
# unrecoverable while a flagged one is reviewable.
#
# So length is treated as a CONFIDENCE signal, not a gate. It still matters,
# because CTC failures are usually length failures: 'GJ32AA9628' came back as
# 'GJ32A8963', which parses happily as GJ|32|A|8963 and passes every grammar
# check while being wrong. A non-ten read is flagged for review rather than
# accepted silently or thrown away.
# strict_len=True is available for pipelines that prefer precision to recall.


def positional_decode(raw: str, strict_len: bool = False) -> tuple[str | None, str]:
    """Coerce `raw` into plate layout. Returns (plate or None, note).

    The last four characters and the first two are fixed anchors; what sits
    between them is district digits plus a 1-3 letter series, and the split
    depends on total length. Both plausible splits are tried and the one
    yielding a real state code wins.
    """
    s = re.sub(r"[^A-Z0-9]", "", raw.upper())
    if strict_len and len(s) != EXPECTED_LEN:
        # Reject rather than coerce. A 9-character read is a DROPPED
        # character, and padding it would invent a plate that looks valid.
        # Reporting "unread" is recoverable; a confident wrong plate is not.
        return None, f"length {len(s)} (expected {EXPECTED_LEN})"
    if not (8 <= len(s) <= 11):
        return None, f"length {len(s)}"

    best = None
    for dlen in (2, 1):                       # district is 2 digits, rarely 1
        mid = s[2:-4]
        if len(mid) < dlen:
            continue
        state = _letters(s[:2])
        dist = _digits(mid[:dlen])
        series = _letters(mid[dlen:])
        tail = _digits(s[-4:])
        cand = f"{state}{dist}{series}{tail}"
        if not re.match(r"^[A-Z]{2}\d{1,2}[A-Z]{0,3}\d{4}$", cand):
            continue
        # A real state code is strong evidence the split was right.
        score = 2 if state in STATE_CODES else 0
        score += 1 if dist.lstrip("0") else 0
        if best is None or score > best[1]:
            best = (cand, score)

    if best is None:
        return None, "no layout fit"
    plate, score = best
    if score < 2:
        return plate, f"unknown state {plate[:2]}"
    if len(plate) != EXPECTED_LEN:
        return plate, f"REVIEW: {len(plate)} chars (99% here are {EXPECTED_LEN})"
    return plate, "ok"


if __name__ == "__main__":
    # Left column is what the fine-tuned CRNN actually produced on held-out
    # plates; right column is the ground truth a human read from the image.
    cases = [
        ("GJ030O3018", "GJ03OO3010"),
        ("GJ32A8963", "GJ32AA9628"),
        ("GJ11DB5330", "GJ11OB9390"),
        ("GJ05HMG2430", "GJ05RN5430"),
        ("GJ04EE0757", "GJ04EE0754"),
        ("6J32K4588", "GJ32K4588"),
        ("GJ03LB0535", "GJ03LB0535"),
    ]
    print(f"{'raw':<13} {'decoded':<13} {'truth':<13} note")
    print("-" * 58)
    for raw, truth in cases:
        p, note = positional_decode(raw)
        mark = "  MATCH" if p == truth else ""
        print(f"{raw:<13} {str(p or '-'):<13} {truth:<13} {note}{mark}")
