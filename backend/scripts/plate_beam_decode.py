"""backend/scripts/plate_beam_decode.py — CTC decoding constrained to the Indian
plate format, applied DURING the search rather than after it.

WHAT IS WRONG WITH THE CURRENT DECODER
  Today the recogniser takes the single most likely character at each timestep
  (greedy), collapses repeats, and only then asks indian_plate_grammar whether
  the result looks like a plate. By that point the decision is made. If the
  greedy path spelled GJ0IAB1234, the grammar can substitute the I for a 1
  because position 3 must be a digit - but it is repairing a string, not
  reconsidering the reading. Whenever the correct character was the model's
  SECOND choice at that timestep, greedy has already thrown it away and no
  amount of post-hoc repair can bring it back.

  Prefix beam search keeps several readings alive at once. Constraining it to
  the plate format means a partial reading that cannot become a legal plate is
  dropped immediately, and the probability mass it held goes to readings that
  can. The correct character being second-choice is then survivable: the beam
  that took it stays in contention and often wins on the following timesteps.

  This costs no labelled data, which is what makes it worth doing now - every
  other lever needs the labelling round to finish first.

THE FORMATS, AND WHY MORE THAN ONE
  The dominant modern format is LL DD LL DDDD - two letters of state, two
  digits of RTO, a one-to-three letter series, four digits. Older and shorter
  series exist and still drive on these roads, and the BH (Bharat) series
  inverts the order entirely. Encoding only the dominant format would make the
  decoder confidently wrong on every plate that is not in it, which is worse
  than the unconstrained decoder it replaces.

STATE CODES ARE CHECKED AS PREFIXES, NOT AS COMPLETE CODES
  After one character the decoder cannot know whether G is going to be GA or
  GJ, so it must keep both. Only once two letters are read is the pair checked
  against the real list. Rejecting on the first character would delete every
  plate whose state shares an initial with another.
"""
from __future__ import annotations

from collections import defaultdict
from math import log

import numpy as np

from backend.scripts.indian_plate_grammar import STATE_CODES

LETTERS = set("ABCDEFGHIJKLMNOPQRSTUVWXYZ")
DIGITS = set("0123456789")

# 'A' = letter, 'D' = digit. Ordered roughly by how often they occur, which
# only affects tie-breaking between equally probable complete readings.
PATTERNS = (
    "AADDAADDDD",     # GJ 01 AB 1234  - dominant modern format
    "AADDADDDD",      # GJ 01 A 1234
    "AADDAAADDDD",    # GJ 01 ABC 1234
    "AADDDDDD",       # GJ 01 1234     - older short series
    "DDAADDDDAA",     # 22 BH 1234 AB  - Bharat series, inverted
)

_STATE_PREFIXES = {s[:1] for s in STATE_CODES} | STATE_CODES


def _fits(prefix: str, pat: str) -> bool:
    """Could this prefix still become a plate of this pattern?"""
    if len(prefix) > len(pat):
        return False
    for i, ch in enumerate(prefix):
        if pat[i] == "A" and ch not in LETTERS:
            return False
        if pat[i] == "D" and ch not in DIGITS:
            return False
    # State code, checked only once both of its characters are present.
    if pat.startswith("AA"):
        if len(prefix) == 1 and prefix[0] not in _STATE_PREFIXES:
            return False
        if len(prefix) >= 2 and prefix[:2] not in STATE_CODES:
            return False
    else:                                   # BH series: positions 2-3 are 'BH'
        if len(prefix) >= 3 and prefix[2] != "B":
            return False
        if len(prefix) >= 4 and prefix[2:4] != "BH":
            return False
    return True


def valid_prefix(prefix: str) -> bool:
    return not prefix or any(_fits(prefix, p) for p in PATTERNS)


def complete(s: str) -> bool:
    return any(len(s) == len(p) and _fits(s, p) for p in PATTERNS)


def beam_decode(logits, itos: dict, blank: int = 0, beam_width: int = 24,
                topk: int = 6, constrain: bool = True) -> list[tuple[str, float]]:
    """CTC prefix beam search over one sample's logits (T, C).

    Returns candidates best-first as (text, log-probability).

    topk caps how many characters are considered per timestep. The tail of a
    36-way softmax is noise, and extending the beam into it costs time while
    adding readings that never win.
    """
    if hasattr(logits, "detach"):
        logits = logits.detach().float().cpu().numpy()
    logits = np.asarray(logits, dtype=np.float64)
    if logits.ndim == 3:
        logits = logits[0]
    # log-softmax, computed stably.
    m = logits.max(axis=1, keepdims=True)
    logp = logits - m - np.log(np.exp(logits - m).sum(axis=1, keepdims=True))

    NEG = -1e30
    # prefix -> [log p(ending in blank), log p(ending in non-blank)]
    beams: dict[str, list[float]] = {"": [0.0, NEG]}

    def lse(a: float, b: float) -> float:
        if a <= NEG and b <= NEG:
            return NEG
        hi, lo = (a, b) if a > b else (b, a)
        return hi + log(1.0 + np.exp(lo - hi)) if hi - lo < 60 else hi

    T = logp.shape[0]
    for t in range(T):
        cand = np.argpartition(logp[t], -topk)[-topk:]
        nxt: dict[str, list[float]] = defaultdict(lambda: [NEG, NEG])
        for prefix, (pb, pnb) in beams.items():
            total = lse(pb, pnb)
            for c in cand:
                p = logp[t, c]
                if c == blank:
                    e = nxt[prefix]
                    e[0] = lse(e[0], total + p)
                    continue
                ch = itos.get(int(c))
                if ch is None:
                    continue
                if prefix and ch == prefix[-1]:
                    # Repeat of the last character: extends the same prefix
                    # unless separated by a blank, in which case it doubles it.
                    e = nxt[prefix]
                    e[1] = lse(e[1], pnb + p)
                    new = prefix + ch
                    if not constrain or valid_prefix(new):
                        e2 = nxt[new]
                        e2[1] = lse(e2[1], pb + p)
                else:
                    new = prefix + ch
                    if not constrain or valid_prefix(new):
                        e2 = nxt[new]
                        e2[1] = lse(e2[1], total + p)
        if not nxt:
            break
        beams = dict(sorted(nxt.items(),
                            key=lambda kv: -lse(kv[1][0], kv[1][1]))[:beam_width])

    scored = sorted(((k, lse(v[0], v[1])) for k, v in beams.items()),
                    key=lambda kv: -kv[1])
    if constrain:
        # Prefer complete plates; keep the rest as fallbacks rather than
        # returning nothing when no beam closed a full pattern.
        done = [s for s in scored if complete(s[0])]
        if done:
            return done + [s for s in scored if not complete(s[0])]
    return scored


def best(logits, itos: dict, blank: int = 0, **kw) -> str:
    r = beam_decode(logits, itos, blank=blank, **kw)
    return r[0][0] if r else ""
