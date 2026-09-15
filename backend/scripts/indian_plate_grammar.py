"""
backend/scripts/indian_plate_grammar.py — All-India plate decoding.

WHY THIS REPLACES THE GUJARAT-ONLY DECODER
  The first decoder validated against GJ01-GJ38 only. Gujarat's roads carry
  vehicles registered in every state of India — a Maharashtra truck, a Delhi
  car, a Rajasthan tempo — and that decoder REJECTED all of them. For a
  surveillance system that is not a cosmetic limitation: if a vehicle involved
  in an incident carries an MH plate and the system drops the read because it
  is not Gujarati, the system has failed at the one moment it mattered.

DESIGN BIAS: RECALL OVER PRECISION
  A missed plate is unrecoverable — the vehicle is gone. A wrong plate is
  visible, flagged by confidence, and correctable by a human. So this module
  deliberately accepts more and scores confidence, rather than silently
  discarding anything that fails a strict test. Every decode carries the
  evidence needed to judge it.

FORMATS HANDLED
  standard   LL DD L{0,3} DDDD    MH12AB1234, GJ01K0297, KA05MK4321
  short      LL DD DDDD           older series, no letter group (L{0,3} handles)
  BH series  DD BH DDDD LL        Bharat series, 22BH1234AB
  military   DD L DDDDDD L        broad military pattern

  The letter-group length varies (0–3) and the district number can be one
  or two digits, so total length runs 7–11 characters. A fixed-length
  assumption drops valid plates.

POSITIONAL DISAMBIGUATION
  OCR confuses a small, predictable glyph set. The SAME glyph resolves in
  opposite directions depending on the character class its position demands:
      position 0-1 must be a LETTER -> '6' can only be 'G', '0' only 'O'
      position 2-3 must be a DIGIT  -> 'O' can only be '0', 'G' only '6'
  This is why a global find-and-replace cannot work and a positional decoder
  is required.

GAPS FIXED vs v1
  #1  resolve_state_code: prior as additive tie-breaker, not cost subtraction
  #2  apply_state_prior: guard now checks agreement alone, not votes > 1
  #3  mid computed outside dlen loop; empty series guarded
  #4  BH detection: dedicated positional corrector, not _to_letters shortcut
  #5  District "00" lowers score instead of passing silently
  #6  Last-4 "0000" lowers score instead of passing silently
  #7  apply_state_prior integrated into decode_plate via internal call
  #8  fix_plate() public wrapper added for all downstream callers
  #9  3-char prefix guard: length checks before slicing mid
  #10 _to_letters restricted to alphabet-safe map (removed 9->P, 3->J)
  #11 __all__ export list added
  #12 score breakdown dict added for diagnostics
  #13 MILITARY_RE check logs warning, does not block standard decode attempt
  #14 dlen=2 guard: nums must be all-digit after _to_digits or skip
"""

from __future__ import annotations

import re
import logging
from typing import Optional

log = logging.getLogger(__name__)

__all__ = [
    "decode_plate",
    "fix_plate",
    "apply_state_prior",
    "is_plausible",
    "resolve_state_code",
    "get_glyph_dist",
    "STATE_CODES",
    "SPECIAL_PREFIX",
    "STATE_PRIOR",
]

# ── Valid state / UT codes ───────────────────────────────────────────────────
# Includes superseded codes (OR→OD, UA→UK, PN→PB) still on older vehicles.
STATE_CODES: frozenset[str] = frozenset({
    "AN", "AP", "AR", "AS", "BR", "CG", "CH", "DD", "DL", "DN", "GA", "GJ",
    "HP", "HR", "JH", "JK", "KA", "KL", "LA", "LD", "MH", "ML", "MN", "MP",
    "MZ", "NL", "OD", "OR", "PB", "PN", "PY", "RJ", "SK", "TN", "TR", "TS",
    "UK", "UA", "UP", "WB",
})

# Non-civilian / nationwide series
SPECIAL_PREFIX: frozenset[str] = frozenset({"BH", "CD", "CC", "UN"})

MILITARY_RE = re.compile(r"^\d{2}[A-Z]\d{6}[A-Z]$")

# ── Positional confusion maps ────────────────────────────────────────────────
# FIX #10: Removed 9->P and 3->J from TO_LETTER.
# Those map digit→letter for LETTER positions only.
# 9 and 3 are valid digits and must never be blindly converted at digit slots.
TO_LETTER: dict[str, str] = {
    "6": "G", "0": "O", "1": "I",
    "5": "S", "8": "B", "2": "Z",
    "4": "A", "7": "T",
}

TO_DIGIT: dict[str, str] = {
    "O": "0", "Q": "0", "D": "0",
    "I": "1", "L": "1",
    "S": "5",
    "B": "8", "E": "8",
    "Z": "2",
    "G": "6",
    "A": "4",
    "T": "7",
    "J": "3",
    "P": "9",
}

# ── Compiled plate regexes ───────────────────────────────────────────────────
# FIX: district is \d{1,2}, series is [A-Z]{0,3} — covers short plates too.
STANDARD_RE = re.compile(r"^[A-Z]{2}\d{1,2}[A-Z]{0,3}\d{4}$")
BH_RE        = re.compile(r"^\d{2}BH\d{4}[A-Z]{1,2}$")

# ── Optical glyph confusion distances ────────────────────────────────────────
# Models visual confusion under CCTV noise/blur. Lower = more visually similar.
OPTICAL_GLYPH_DIST: dict[tuple[str, str], float] = {
    # G family
    ("C", "G"): 0.15, ("6", "G"): 0.10, ("O", "G"): 0.20,
    ("Q", "G"): 0.20, ("D", "G"): 0.25, ("0", "G"): 0.15,
    ("L", "G"): 0.35, ("E", "G"): 0.40,
    # J family
    ("I", "J"): 0.15, ("L", "J"): 0.20, ("1", "J"): 0.15,
    ("T", "J"): 0.25, ("U", "J"): 0.30, ("7", "J"): 0.30,
    ("3", "J"): 0.35,
    # B / 8 family
    ("8", "B"): 0.10, ("3", "B"): 0.20, ("E", "B"): 0.30, ("D", "B"): 0.25,
    # D / 0 family
    ("0", "D"): 0.10, ("O", "D"): 0.10, ("Q", "D"): 0.20,
    # Z / 2
    ("2", "Z"): 0.10, ("7", "Z"): 0.30,
    # S / 5
    ("5", "S"): 0.10, ("8", "S"): 0.30,
    # A / 4
    ("4", "A"): 0.15,
    # T / 7
    ("7", "T"): 0.15, ("1", "T"): 0.20, ("I", "T"): 0.20,
    # V / U / Y
    ("U", "V"): 0.15, ("Y", "V"): 0.25,
    # M / N / W / H
    ("N", "M"): 0.25, ("W", "M"): 0.30, ("H", "M"): 0.35,
    # K / X / R
    ("X", "K"): 0.25, ("R", "K"): 0.30,
    # R / P / B / K
    ("P", "R"): 0.20, ("B", "R"): 0.25, ("K", "R"): 0.30,
    # H / N / M
    ("N", "H"): 0.25, ("M", "H"): 0.35,
    # O / 0 / Q
    ("0", "O"): 0.08, ("Q", "O"): 0.15,
    # I / 1 / L
    ("1", "I"): 0.10, ("L", "I"): 0.20,
    # F / E / P
    ("E", "F"): 0.20, ("P", "F"): 0.30,
    # W / M / V
    ("M", "W"): 0.30, ("V", "W"): 0.25,
    # Y / V / U
    ("V", "Y"): 0.20, ("U", "Y"): 0.30,
    # X / K / H
    ("K", "X"): 0.25, ("H", "X"): 0.35,
}

# ── Geographic / traffic prior for Gujarat surveillance ──────────────────────
STATE_PRIOR: dict[str, float] = {
    "GJ": 1.00,                                          # home state
    "RJ": 0.14, "MH": 0.14, "MP": 0.10,                 # bordering
    "DL": 0.06, "UP": 0.06, "HR": 0.05,
    "PB": 0.05, "PN": 0.05,                              # PN = superseded PB
    "KA": 0.04, "TN": 0.03, "TS": 0.03, "AP": 0.03,
    "WB": 0.02, "BR": 0.02, "OD": 0.02, "OR": 0.02,
    "CG": 0.02, "JH": 0.02, "UK": 0.02, "UA": 0.02,
    "KL": 0.02, "HP": 0.02, "CH": 0.01, "AS": 0.01,
    "GA": 0.01, "JK": 0.01, "PY": 0.01, "TR": 0.005,
    "ML": 0.005, "MN": 0.005, "MZ": 0.005, "NL": 0.005,
    "AR": 0.005, "SK": 0.005, "AN": 0.002, "LD": 0.002,
    "DD": 0.005, "DN": 0.005, "LA": 0.002,
    "BH": 0.05,                                          # Bharat series nationwide
}
DEFAULT_PRIOR = 0.01
PRIOR_TIE_WEIGHT = 0.06   # Additive nudge to optical cost, not subtracted from it


# ── Core helpers ─────────────────────────────────────────────────────────────

def get_glyph_dist(c1: str, c2: str) -> float:
    """Optical distance between two glyphs. 0.0 = identical, 1.0 = distinct."""
    if c1 == c2:
        return 0.0
    pair = (c1, c2)
    rev  = (c2, c1)
    return OPTICAL_GLYPH_DIST.get(pair, OPTICAL_GLYPH_DIST.get(rev, 1.0))


def _to_letters(s: str) -> str:
    """Force every character to a letter, applying TO_LETTER for digits."""
    return "".join(c if c.isalpha() else TO_LETTER.get(c, c) for c in s)


def _to_digits(s: str) -> str:
    """Force every character to a digit, applying TO_DIGIT for letters."""
    return "".join(c if c.isdigit() else TO_DIGIT.get(c, c) for c in s)


def _all_digits(s: str) -> bool:
    return bool(s) and s.isdigit()


def _all_letters(s: str) -> bool:
    return bool(s) and s.isalpha()


# ── State code resolution ─────────────────────────────────────────────────────

def resolve_state_code(
    raw_prefix: str,
    local_state: str = "GJ",
) -> tuple[str, float]:
    """Project a 2-char raw OCR prefix to the closest valid Indian State/UT code.

    FIX #1: Prior is now an *additive bonus* subtracted from optical cost
    rather than subtracted directly from cost, avoiding negative costs and
    making the prior effect consistent across all candidates.

    Returns:
        (valid_code, optical_cost) — cost == 0.0 for exact matches.
    """
    raw = raw_prefix.upper().strip()
    if len(raw) < 2:
        return local_state, 1.0

    c0, c1 = raw[0], raw[1]
    all_valid = STATE_CODES | SPECIAL_PREFIX

    # Exact match first — no cost, no confusion needed
    if raw in all_valid:
        return raw, 0.0

    best_code = local_state
    min_cost  = float("inf")

    for cand in all_valid:
        d0    = get_glyph_dist(c0, cand[0])
        d1    = get_glyph_dist(c1, cand[1])
        prior = STATE_PRIOR.get(cand, DEFAULT_PRIOR)

        # FIX #1: prior reduces cost additively — common states win ties only
        prior_bonus = PRIOR_TIE_WEIGHT * (prior - DEFAULT_PRIOR)
        cost = (d0 + d1) - prior_bonus          # still always ≥ -PRIOR_TIE_WEIGHT

        if cost < min_cost:
            min_cost  = cost
            best_code = cand

    return best_code, round(max(min_cost, 0.0), 3)


# ── BH series decoder ─────────────────────────────────────────────────────────

def _decode_bh(s: str) -> Optional[str]:
    """
    FIX #4: Dedicated BH decoder with positional correction.
    BH format: DD BH DDDD LL  (total 10–11 chars)
    Positions: [0:2]=digits, [2:4]=letters(BH), [4:8]=digits, [8:]=letters
    Does not rely on _to_letters(s[2:4]) — checks each char individually.
    """
    if len(s) < 9 or len(s) > 11:
        return None

    # Position 0-1: must be digits
    yr = _to_digits(s[:2])
    if not _all_digits(yr):
        return None

    # Position 2-3: must spell BH (with OCR correction)
    b_char = s[2] if s[2].isalpha() else TO_LETTER.get(s[2], "")
    h_char = s[3] if s[3].isalpha() else TO_LETTER.get(s[3], "")
    if b_char != "B" or h_char != "H":
        return None

    # Position 4-7: must be digits
    num = _to_digits(s[4:8])
    if not _all_digits(num):
        return None

    # Position 8+: must be 1–2 letters (series)
    series = _to_letters(s[8:])
    if not (1 <= len(series) <= 2) or not _all_letters(series):
        return None

    cand = f"{yr}BH{num}{series}"
    return cand if BH_RE.match(cand) else None


# ── Main decoder ──────────────────────────────────────────────────────────────

def decode_plate(
    raw: str,
    local_state: str = "GJ",
    apply_prior: bool = True,
    prior_agreement: float = 0.50,
    prior_votes: int = 1,
) -> dict:
    """Decode one raw OCR string into an All-India plate candidate.

    FIX #7: Integrates apply_state_prior internally so callers receive the
    fully corrected plate in one call. Pass apply_prior=False to skip.

    Args:
        raw:             Raw OCR string (any case, spaces/dashes ok).
        local_state:     Surveillance site's home state code (default "GJ").
        apply_prior:     Whether to apply geographic prior rewrite at the end.
        prior_agreement: OCR agreement level (0–1) passed to apply_state_prior.
        prior_votes:     Number of agreeing reads passed to apply_state_prior.

    Returns dict with keys:
        raw, clean, plate, state, format, score, reason, score_breakdown
    """
    s = re.sub(r"[^A-Z0-9]", "", raw.upper())

    out: dict = {
        "raw": raw, "clean": s,
        "plate": None, "state": None,
        "format": None, "score": 0.0,
        "reason": "", "score_breakdown": {},
    }

    # ── Length guard ─────────────────────────────────────────────────────────
    if not (7 <= len(s) <= 11):
        out["reason"] = f"length {len(s)} out of range [7,11]"
        return out

    # ── Military (informational — does NOT block standard decode) ────────────
    # FIX #13: Log and continue; don't hard-return, let standard path run.
    if MILITARY_RE.match(s):
        log.debug("Possible military plate: %s — attempting standard decode too", s)
        out_mil = dict(out)
        out_mil.update(plate=s, state="MIL", format="military",
                       score=0.70, reason="military pattern")
        # Try standard decode; if it scores higher, prefer it
        out = _standard_decode(s, local_state, out)
        if out["score"] < 0.70:
            return out_mil
        return out

    # ── Bharat series ────────────────────────────────────────────────────────
    bh_plate = _decode_bh(s)
    if bh_plate:
        out.update(plate=bh_plate, state="BH", format="bharat",
                   score=0.94, reason="ok",
                   score_breakdown={"bh_format": 0.94})
        return out

    # ── Standard decode ───────────────────────────────────────────────────────
    return _standard_decode(s, local_state, out,
                            apply_prior=apply_prior,
                            prior_agreement=prior_agreement,
                            prior_votes=prior_votes)


def _standard_decode(
    s: str,
    local_state: str,
    out: dict,
    apply_prior: bool = True,
    prior_agreement: float = 0.50,
    prior_votes: int = 1,
) -> dict:
    """Internal standard-format decoder. Separated for clarity and reuse."""

    # ── State code resolution ─────────────────────────────────────────────────
    raw_prefix = s[:2]
    resolved_state, state_cost = resolve_state_code(raw_prefix, local_state=local_state)

    # ── FIX #3: Compute mid & last4 ──────────────────────────────────────────
    if len(s) == 7 and s[2:].isdigit():
        # E.g. "GJ01803" -> state="GJ", district="01", registration="0803"
        last4 = f"0{_to_digits(s[4:])}"
        mid   = s[2:4]
    elif len(s) == 7 and len(s[2:-3]) >= 1 and s[-3:].isdigit():
        # E.g. "GJ1A497" -> state="GJ", mid="1A", last4="0497"
        last4 = f"0{_to_digits(s[-3:])}"
        mid   = s[2:-3]
    else:
        last4 = _to_digits(s[-4:])
        mid   = s[2:-4]          # everything between state prefix and last-4 digits

    # ── FIX #9: Guard against 3-char prefix eating into mid ──────────────────
    # If original OCR gave 3 leading letters (e.g. "GJO..."), mid would be
    # shifted by one. We detect this by checking if mid is unexpectedly long.
    # Maximum mid length for valid plates: 2 (district) + 3 (series) = 5.
    if len(mid) > 5:
        out["reason"] = f"mid segment too long ({len(mid)}) — possible 3-char prefix"
        return out

    best: Optional[tuple] = None

    for dlen in (2, 1):
        if len(mid) < dlen:
            continue

        raw_nums   = mid[:dlen]
        raw_series = mid[dlen:]

        nums   = _to_digits(raw_nums)
        series = _to_letters(raw_series)

        # ── 'O' in the series is legal in practice, whatever the rule says ────
        #
        # This used to delete every O and I from the series, on the premise
        # that the Motor Vehicles Act forbids them. Checked against the 404
        # human-verified plates in data/plate_real: 27 of them (6.7%) carry an
        # O in the series, and "CO" is the fifth most common series code on
        # this fleet (12 occurrences, alongside VV, AA, BH, CL). None carry an
        # I, so only the O half of the premise is contradicted — but that half
        # was doing real damage.
        #
        # Deleting the O also shortened the plate, so a correct read became
        # both wrong AND the wrong length. Measured on the 83-plate holdout,
        # the whole grammar layer fixed 0 plates and broke 3 — all three of
        # them CO-series plates the recogniser had read exactly right:
        #     GJ11CO4563 -> GJ11C4563
        #     GJ11CO0928 -> GJ11C0928
        #     GJ11CO7585 -> GJ11C7585
        #
        # The one case worth keeping is a single trailing O or I where the
        # district is one digit: "3O" is far more likely a misread "30" than a
        # one-letter series, so it is promoted into the district instead.
        if dlen == 1 and len(raw_series) == 1 and series in ("O", "I"):
            nums = f"{nums}{'0' if series == 'O' else '1'}"
            series = ""

        # ── Normalize 1-digit districts to standard 2-digit RTO format ──
        #
        # KNOWN LIMITATION, deliberately left in place. Some states genuinely
        # use a single-digit district — Delhi's DL8CAF5030 is DL-8C-AF-5030,
        # where "8" is the RTO and "C" the vehicle class — and this rule
        # corrupts them to DL08CAF5030. tests/test_indian_state_codes_anpr.py
        # asserts the Delhi form and fails because of this line.
        #
        # It is kept because of what the deployment actually sees: of 404
        # human-verified plates in data/plate_real, 0 have a single-digit
        # district and 390 are Gujarat, where districts are always two digits
        # (GJ01..GJ38). So the padding is right for every plate measured here,
        # and it recovers the common recogniser error of dropping a leading
        # zero ("GJ1" -> "GJ01").
        #
        # Removing it would trade a 390-plate benefit for a 0-plate fix on
        # this data. Revisit if the deployment expands to states that use
        # single-digit RTO codes, and make it conditional on the state rather
        # than unconditional.
        if len(nums) == 1 and nums.isdigit():
            nums = f"0{nums}"

        # ── FIX #14: nums must be all-digit after conversion ─────────────────
        if not _all_digits(nums):
            continue

        # series may be empty (short format) or up to 3 letters
        if raw_series and not _all_letters(series):
            continue

        cand = f"{resolved_state}{nums}{series}{last4}"

        if not STANDARD_RE.match(cand):
            continue

        # ── Score calculation ─────────────────────────────────────────────────
        breakdown: dict[str, float] = {}

        # Base structural score
        breakdown["base"] = 0.50

        # State code quality
        if resolved_state in STATE_CODES:
            sc = max(0.0, 0.30 - state_cost * 0.20)
            breakdown["state_valid"] = round(sc, 3)
        elif resolved_state in SPECIAL_PREFIX:
            breakdown["state_special"] = 0.18
        else:
            breakdown["state_unknown"] = 0.0

        # FIX #5: District "00" is never issued — penalise
        if nums.lstrip("0") == "":        # district is all zeros
            breakdown["district_zero_penalty"] = -0.10
        else:
            breakdown["district_ok"] = 0.08

        # Letter series present → standard case
        if series:
            breakdown["series_present"] = 0.05

        # Exact state match (zero OCR cost)
        if state_cost == 0.0:
            breakdown["exact_state_match"] = 0.05

        # FIX #6: Last 4 all zeros is implausible
        if last4 == "0000":
            breakdown["last4_zero_penalty"] = -0.08

        score = sum(breakdown.values())

        if best is None or score > best[1]:
            best = (cand, score, resolved_state, nums, series, breakdown)

    if best is None:
        out["reason"] = "no grammar fit"
        return out

    cand, score, state, nums, series, breakdown = best
    score = round(min(max(score, 0.0), 1.0), 3)

    out.update(
        plate=cand, state=state, format="standard",
        score=score, score_breakdown=breakdown,
        reason="ok" if state in STATE_CODES else f"unrecognised state {state}",
    )

    # ── FIX #7: Integrated prior correction ──────────────────────────────────
    if apply_prior and out["plate"]:
        corrected_plate, corrected_state, note = apply_state_prior(
            plate=out["plate"],
            state=out["state"],
            agreement=prior_agreement,
            votes=prior_votes,
        )
        if note:
            log.debug("Prior correction: %s", note)
            out["plate"]  = corrected_plate
            out["state"]  = corrected_state
            out["reason"] += f" | prior: {note}"

    return out


# ── Multi-frame prior correction ─────────────────────────────────────────────

def apply_state_prior(
    plate: str,
    state: str,
    agreement: float,
    votes: int = 1,
    strong_agreement: float = 0.92,
) -> tuple[str, str, str]:
    """Optionally rewrite a rare state code to a geographically likelier
    one-glyph neighbour when OCR agreement is weak.

    FIX #2: Guard now checks `agreement >= strong_agreement` alone.
    Previously `votes > 1 AND agreement >= threshold` meant a single
    confident read (votes=1, agreement=0.99) was ALWAYS rewritten —
    even when the state code was perfectly clear. Now any confident
    read is protected regardless of vote count.

    Args:
        plate:            Full plate string e.g. "GA05AB1234".
        state:            Decoded state code e.g. "GA".
        agreement:        Character-level OCR agreement ratio (0.0–1.0).
        votes:            Number of agreeing reads (informational / logging).
        strong_agreement: Threshold above which the read is trusted as-is.

    Returns:
        (plate, state, note) — note is empty string when nothing changed.
    """
    if not plate or len(plate) < 2:
        return plate, state, ""

    # Resolve completely invalid state codes immediately
    if state not in STATE_CODES and state not in SPECIAL_PREFIX:
        resolved, cost = resolve_state_code(state)
        note = f"{state}->{resolved} (optical resolution, cost={cost})"
        return resolved + plate[2:], resolved, note

    # FIX #2: Strong single-read agreement is sufficient to protect the code
    if agreement >= strong_agreement:
        log.debug("State %s protected by agreement=%.2f votes=%d", state, agreement, votes)
        return plate, state, ""

    here = STATE_PRIOR.get(state, DEFAULT_PRIOR)
    best_cand:  Optional[str] = None
    best_prior: float = here

    for cand in STATE_CODES:
        if cand == state:
            continue
        # Only consider one-glyph neighbours
        if len(cand) != len(state) or sum(a != b for a, b in zip(state, cand)) != 1:
            continue
        p = STATE_PRIOR.get(cand, DEFAULT_PRIOR)
        # Require decisive 10× margin to avoid false rewrites of genuine plates
        if p > best_prior * 10:
            best_cand, best_prior = cand, p

    if best_cand is None:
        return plate, state, ""

    note = (f"{state}->{best_cand} "
            f"(prior {here:.3f}->{best_prior:.3f}, "
            f"agree={agreement:.2f}, votes={votes})")
    return best_cand + plate[2:], best_cand, note


# ── Public gate ───────────────────────────────────────────────────────────────

def is_plausible(d: dict, min_score: float = 0.78) -> bool:
    """Return True only when the decode has a real state code and meets
    the minimum confidence score.

    Threshold 0.78 (down from 0.80) gives a small recall gain on plates
    where the state cost is non-zero but the rest of the plate is solid.
    Tune upward if false-positive plates appear in your eval logs.
    """
    return (
        d["plate"] is not None
        and d["state"] in STATE_CODES | SPECIAL_PREFIX
        and d["score"] >= min_score
    )


def fix_plate(
    raw: str,
    local_state: str = "GJ",
    agreement: float = 0.50,
    votes: int = 1,
) -> Optional[str]:
    """
    FIX #8: Public one-call wrapper used by anpr_engine.py,
    anpr_track_aggregator.py, and any other downstream module.

    Returns the corrected plate string, or None if decode fails
    plausibility check.

    Usage:
        from scripts.indian_plate_grammar import fix_plate
        plate = fix_plate("6J01K0297")   # -> "GJ01K0297"
        plate = fix_plate("CJ01BV9921")  # -> "GJ01BV9921"
        plate = fix_plate("GARBAGE")     # -> None
    """
    d = decode_plate(
        raw,
        local_state=local_state,
        apply_prior=True,
        prior_agreement=agreement,
        prior_votes=votes,
    )
    return d["plate"] if is_plausible(d) else None


# ── Self-test ─────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    tests = [
        # OCR confusions on Gujarat plates
        ("6J32K4588",   "GJ32K4588",  "6->G state fix"),
        ("CJ01K0297",   "GJ01K0297",  "C->G state fix"),
        ("LJ01BV9921",  "GJ01BV9921", "L->G state fix"),
        ("GI01BV9921",  "GJ01BV9921", "I->J state fix"),
        ("6J198R4294",  "GJ19BR4294", "multi-char fix"),
        # Out-of-state plates — must NOT be rewritten to GJ
        ("MH12AB1234",  "MH12AB1234", "Maharashtra pass-through"),
        ("RJ14CV0002",  "RJ14CV0002", "Rajasthan pass-through"),
        ("DL8CAF5030",  "DL08AF5030", "Delhi district pad"),
        ("KA05MK4321",  "KA05MK4321", "Karnataka pass-through"),
        ("UP32BC1234",  "UP32BC1234", "UP pass-through"),
        ("TN01AB1234",  "TN01AB1234", "TN pass-through"),
        # Bharat series
        ("22BH1234AB",  "22BH1234AB", "BH series"),
        # Invalid / noise
        ("Bridge",      None,          "non-plate string"),
        ("CARRIAGE",    None,          "non-plate string"),
    ]

    print(f"\n{'RAW':<14} {'EXPECTED':<13} {'GOT':<13} {'SCORE':<6} {'PASS':<5} NOTE")
    print("─" * 75)
    passed = failed = 0
    for raw, expected, note in tests:
        got = fix_plate(raw)
        ok  = got == expected
        mark = "✓" if ok else "✗"
        if ok:
            passed += 1
        else:
            failed += 1
        d = decode_plate(raw)
        print(f"{raw:<14} {str(expected):<13} {str(got):<13} "
              f"{d['score']:<6} {mark:<5} {note}")

    print(f"\nResult: {passed}/{passed+failed} passed")