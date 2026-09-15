"""
plate_utils.py — Gujarat license plate text normalization, validation, and
confusable-character correction for Sentinel Gujarat ANPR pipeline.

All normalization logic lives here. Every place that creates OR reads a plate
string (seed_data.py, anpr.py, anpr_track_aggregator.py) must import and use
normalize_plate_text() so that seed-time and read-time strings are always
byte-identical for the same physical plate.

Gujarat plate format: GJ<DD><L|LL><DDDD>
  Examples: GJ05AB1234   GJ01X9999   GJ27AZ0001

The regex is loaded from config at call time so it can be tuned without code
changes, but the confusable-correction logic is inherently positional and
lives here because it depends on knowing the expected character class at each
index.
"""

from __future__ import annotations

import logging
import re

logger = logging.getLogger(__name__)

# ── Known confusable characters — positional context determines correction ──
# In DIGIT positions: convert letter-forms that look like digits.
# In LETTER positions: convert digit-forms that look like letters.
_DIGIT_TO_LETTER: dict[str, str] = {
    "0": "O",
    "1": "I",
    "5": "S",
    "8": "B",
}
_LETTER_TO_DIGIT: dict[str, str] = {
    "O": "0",
    "I": "1",
    "S": "5",
    "B": "8",
}


def normalize_plate_text(raw: str | None) -> str | None:
    """Normalize raw OCR output into canonical Gujarat plate format.

    Steps:
      1. Strip all whitespace and convert to uppercase.
         e.g. "gj 05 ab 1234" → "GJ05AB1234"
      2. Remove any non-alphanumeric characters that OCR may hallucinate
         (dashes, dots, slashes).
      3. Apply confusable-character correction using the expected Gujarat
         plate structure (GJ + 2 digit + 1-2 letter + 4 digit).

    This function is intentionally LENIENT about input and STRICT about output.
    It does NOT validate — call is_valid_plate() separately.

    Args:
        raw: Raw string from OCR, or None.

    Returns:
        Normalized uppercase string with no spaces/special chars, or None
        if input is None or empty after stripping.
    """
    if not raw:
        return None

    # Step 1: uppercase + collapse all whitespace
    text = raw.upper().replace(" ", "")

    # Step 2: remove non-alphanumeric (dashes, dots, slashes, brackets)
    text = re.sub(r"[^A-Z0-9]", "", text)

    if not text:
        return None

    # Step 3: apply positional confusable correction if text looks like a plate
    text = correct_confusable_chars(text)

    logger.debug("normalize_plate_text: %r → %r", raw, text)
    return text


def correct_confusable_chars(text: str) -> str:
    """Apply positional confusable-character correction for Gujarat plates.

    The correction is based on the expected GJ<DD><L|LL><DDDD> structure.
    It handles both 8-char (GJ + 2D + 1L + 4D = 9 total incl GJ prefix)
    and 9-char (GJ + 2D + 2L + 4D = 10 total) variants.

    For each character, if it is in the digit-expected zone, convert
    letter-forms (O, I, S, B) to their digit equivalents (0, 1, 5, 8), and
    vice versa in the letter-expected zone.

    If the text does not match either expected length or the GJ prefix,
    correction is skipped (the text is likely garbage from a non-plate region).

    Args:
        text: Uppercase alphanumeric string.

    Returns:
        Corrected string, or original string if correction is not applicable.
    """
    if not text.startswith("GJ"):
        # Not a Gujarat plate — don't corrupt whatever it is
        return text

    body = text[2:]  # Strip the "GJ" prefix

    # Determine variant: 7 chars (2D+1L+4D) or 8 chars (2D+2L+4D)
    if len(body) == 7:
        # Expected: D D L D D D D
        digit_positions = {0, 1, 3, 4, 5, 6}
        letter_positions = {2}
    elif len(body) == 8:
        # Expected: D D L L D D D D
        digit_positions = {0, 1, 4, 5, 6, 7}
        letter_positions = {2, 3}
    else:
        # Unexpected length — skip correction, return as-is
        logger.debug(
            "correct_confusable_chars: body length %d unexpected for GJ plate — skipping",
            len(body),
        )
        return text

    corrected = list(body)
    for i, ch in enumerate(corrected):
        if i in digit_positions and ch in _LETTER_TO_DIGIT:
            corrected[i] = _LETTER_TO_DIGIT[ch]
            logger.debug(
                "Confusable correction at pos %d: %r → %r (digit zone)", i, ch, corrected[i]
            )
        elif i in letter_positions and ch in _DIGIT_TO_LETTER:
            corrected[i] = _DIGIT_TO_LETTER[ch]
            logger.debug(
                "Confusable correction at pos %d: %r → %r (letter zone)", i, ch, corrected[i]
            )

    return "GJ" + "".join(corrected)


def is_valid_plate(text: str | None, plate_regex: str) -> bool:
    """Return True if text matches the Gujarat plate regex pattern.

    Args:
        text: Normalized plate string (from normalize_plate_text).
        plate_regex: Regex pattern from config (e.g. '^GJ\\d{2}[A-Z]{1,2}\\d{4}$').

    Returns:
        True if the pattern matches, False otherwise (including None input).
    """
    if not text:
        return False
    matched = bool(re.match(plate_regex, text))
    if not matched:
        logger.debug("is_valid_plate: %r does not match pattern %r", text, plate_regex)
    return matched


def plates_are_equal(plate_a: str | None, plate_b: str | None) -> bool:
    """Compare two normalized plate strings for equality.

    Both sides are re-normalized before comparison to prevent silent mismatches
    due to one side not having gone through normalize_plate_text().

    IMPORTANT: use this function for ALL watchlist comparisons, not raw ==.

    Args:
        plate_a: First plate string (may be raw or already normalized).
        plate_b: Second plate string.

    Returns:
        True only if both sides normalize to the same non-None string.
    """
    na = normalize_plate_text(plate_a)
    nb = normalize_plate_text(plate_b)
    return na is not None and nb is not None and na == nb
