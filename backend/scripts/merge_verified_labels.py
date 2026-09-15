"""backend/scripts/merge_verified_labels.py — combine the labelling batches into
one training set, rejecting entries that would teach the wrong thing.

WHAT GETS REJECTED, AND WHY EACH IS WORSE THAN A MISSING LABEL

  not a plate    'GSRTC' and '18003131212' are a bus operator's name and its
                 phone number. The labeller transcribed what was on screen,
                 correctly - the detector should not have offered those crops.
                 Training on them teaches the recogniser to read signage.

  partial read   'GJ03HH38' is the first eight characters of a plate whose
                 tail was illegible. It is not a shorter plate; it is a
                 truncated one. A CTC model trained on truncated targets
                 learns to stop early, and it will then truncate plates it
                 could otherwise have read in full. That damage appears
                 nowhere in the accuracy figure except as a lower one.

  conflict       the same vehicle labelled twice with different readings. One
                 of them is wrong and there is no way to tell which, so both
                 are set aside for a human rather than guessed between.

  A rejected entry costs one training example. A wrong one costs that plus
  whatever it teaches, which is why the bar here is deliberately strict.

USAGE
  python -m backend.scripts.merge_verified_labels
  python -m backend.scripts.merge_verified_labels --dry-run
"""
from __future__ import annotations

import argparse
import json
import shutil
from collections import Counter
from pathlib import Path

from backend.scripts.indian_plate_grammar import TO_DIGIT, TO_LETTER
from backend.scripts.plate_beam_decode import PATTERNS, _fits, complete

REAL = Path("data/plate_real")
OUT = REAL / "verified_all.jsonl"

LETTERS = set("ABCDEFGHIJKLMNOPQRSTUVWXYZ")
DIGITS = set("0123456789")


def repair(t: str) -> str | None:
    """Coerce a reading onto a plate pattern using glyph confusions only.

    A labeller types what the glyph LOOKS like, and on a plate the same shape
    is a letter in one position and a digit in another - O and 0, I and 1, G
    and 6. 'GJO1RO0313' is not a bad label; it is GJ01RO0313 written with the
    letter O where the format requires the digit. Rejecting it discards a
    correct human reading over a keystroke.

    Two limits keep this from fabricating plates, and the first version of it
    did fabricate four out of five before they were added:

      LENGTH. Only 10- and 11-character readings are repaired. A partial read
      is 8 or 9 characters, and the shorter plate patterns will happily absorb
      one - 'GJ11DB203' became 'GJ11D8203', which looks like a plate and is
      not the plate on the vehicle. Ten characters is the standard modern
      format, so a full-length reading is a complete one.

      EDIT BUDGET. At most two substitutions. One or two glyph confusions is
      what a careful reader makes; needing five means the reading and the
      pattern disagree about what the plate says, and the pattern should not
      win that argument.

    Beyond those, only substitutions from the known confusion tables are
    allowed, only where the pattern demands the other class, and a character
    already of the right class is never touched. If any character cannot be
    coerced the repair fails and the entry is rejected rather than forced.
    """
    if len(t) < 10:
        return None
    for pat in PATTERNS:
        if len(t) != len(pat):
            continue
        out, edits = [], 0
        for ch, want in zip(t, pat):
            if want == "A":
                if ch not in LETTERS:
                    ch, edits = TO_LETTER.get(ch, ""), edits + 1
            else:
                if ch not in DIGITS:
                    ch, edits = TO_DIGIT.get(ch, ""), edits + 1
            if not ch or edits > 2:
                break
            out.append(ch)
        else:
            cand = "".join(out)
            if _fits(cand, pat) and complete(cand):
                return cand
    return None

# Collection order. Later batches win a straight duplicate, since they were
# read with better tooling; a CONFLICT is reported rather than resolved.
BATCHES = [
    "plate_labels_verified.jsonl",
    "verified_batch2.jsonl",
    "verified_batch3.jsonl",
    "verified_batch4.jsonl",
    "verified_batch5.jsonl",
    "verified_batch6.jsonl",
    "verified_batch7.jsonl",
]


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--dry-run", action="store_true")
    ap.add_argument("--keep-partial", action="store_true",
                    help="Keep readings that do not fit any plate pattern. "
                         "Off by default - see the module docstring.")
    args = ap.parse_args()

    kept: dict[tuple, dict] = {}
    stats = Counter()
    rejected, conflicts, repairs = [], [], []

    for name in BATCHES:
        p = REAL / name
        if not p.is_file():
            continue
        n_in = n_ok = 0
        for line in p.open(encoding="utf-8"):
            v = json.loads(line)
            n_in += 1
            t = (v.get("text") or "").strip().upper()
            t = "".join(c for c in t if c.isalnum())
            if not t:
                stats["blank"] += 1
                continue
            if not complete(t):
                fixed = repair(t)
                if fixed:
                    repairs.append((t, fixed))
                    stats["repaired glyph confusion"] += 1
                    t = fixed
                elif not args.keep_partial:
                    stats["not a plate / partial"] += 1
                    rejected.append((name, t, v.get("file", "")))
                    continue
            key = (v["camera"], v["track"])
            if key in kept and kept[key]["text"] != t:
                conflicts.append((key, kept[key]["text"], t, name))
                stats["conflict"] += 1
                continue
            kept[key] = {"file": v.get("file", ""), "camera": v["camera"],
                         "track": v["track"], "ocr": v.get("ocr", ""),
                         "text": t}
            n_ok += 1
        print(f"{name:<32} {n_in:>5} in  {n_ok:>5} accepted")

    print("\n" + "=" * 62)
    print(f"vehicles in training set : {len(kept)}")
    for k, n in stats.most_common():
        print(f"  rejected, {k:<24} {n:>5}")
    print("=" * 62)

    lens = Counter(len(v["text"]) for v in kept.values())
    print("\nlength distribution:")
    for k in sorted(lens):
        print(f"  {k:>2} chars : {lens[k]:>5}")
    states = Counter(v["text"][:2] for v in kept.values())
    print(f"\nstates: {len(states)} distinct")
    for s, n in states.most_common(10):
        print(f"  {s} : {n}")

    if repairs:
        print(f"\nrepaired glyph confusions ({len(repairs)}), first 15:")
        for a, b in repairs[:15]:
            print(f"  {a:<14} -> {b}")

    if rejected:
        print(f"\nrejected readings ({len(rejected)}), first 15:")
        for b, t, f in rejected[:15]:
            print(f"  {t:<14} {b:<28} {f}")
    if conflicts:
        print(f"\nCONFLICTS needing a human ({len(conflicts)}):")
        for k, a, b, src in conflicts[:10]:
            print(f"  {k}: {a!r} vs {b!r} (from {src})")

    if args.dry_run:
        print("\n--dry-run: nothing written.")
        return

    if OUT.is_file():
        shutil.copy(OUT, OUT.with_suffix(".jsonl.bak"))
        print(f"\nbacked up existing -> {OUT.name}.bak")
    with OUT.open("w", encoding="utf-8") as f:
        for v in kept.values():
            f.write(json.dumps(v) + "\n")
    print(f"wrote {OUT}  ({len(kept)} vehicles)")


if __name__ == "__main__":
    main()
