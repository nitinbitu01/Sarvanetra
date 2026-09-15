"""
backend/services/threshold_validation.py — is a matcher threshold validated?

WHY THIS EXISTS
  WATCHLIST_FACE_MATCH_THRESHOLD (0.80) has never been validated against real
  labelled face pairs. The only report in reports/ is synthetic, and its
  headline numbers are a property of the random number generator: independent
  512-d unit vectors are near-orthogonal, so no "different person" pair can
  approach 0.80 and the false-positive rate comes out at a meaningless 0%.

  Until Day 14 that was a documentation problem. It is now an operational one:
  a WATCHLIST_FACE_MATCH is classified CRITICAL, which auto-dispatches a
  named officer to a physical location. An unvalidated threshold deciding
  where police physically go is a different category of risk from an
  unvalidated threshold labelling a row in a table.

  This module makes "has this been validated?" a queryable fact rather than a
  paragraph in a README, and lets the dispatch path act on the answer.

WHAT COUNTS AS VALIDATED
  A report in reports/ that is BOTH:
    - for the right mode (face), and
    - NOT generated from synthetic data.
  scripts/reid_validate.py stamps every synthetic run with an explicit
  "SYNTHETIC" marker, so the check is a substring test against the report the
  tool itself wrote — not a separate registry someone has to remember to
  update, which would drift from reality the first time it was skipped.

  Deliberately NOT auto-satisfiable: there is no config flag that marks the
  threshold validated. Producing a real report is the only way, because the
  only thing that actually reduces the risk is measuring against real data.
"""
from __future__ import annotations

import logging
import re
from dataclasses import dataclass
from pathlib import Path

logger = logging.getLogger(__name__)

_PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
_REPORTS_DIR = _PROJECT_ROOT / "reports"

# scripts/reid_validate.py writes these markers into every report it
# generates. Matching on the tool's own output keeps this honest: it cannot
# report "validated" for a run that did not happen.
_SYNTHETIC_MARKERS = (
    "SYNTHETIC PROXY",
    "PROXY VALIDATION",
    "THIS NUMBER IS NOT A SAFETY RESULT",
    "DEGENERATE RUN",
)


@dataclass(frozen=True)
class ValidationState:
    validated: bool
    report_path: str | None
    reason: str
    fpr_at_threshold: float | None = None

    def as_dict(self) -> dict:
        return {
            "validated": self.validated,
            "report": self.report_path,
            "reason": self.reason,
            "false_positive_rate": self.fpr_at_threshold,
        }


def _extract_fpr(text: str) -> float | None:
    """Pull the reported false-positive rate out of a report, if present."""
    m = re.search(r"False Positive Rate.*?\*\*([\d.]+)%\*\*", text, re.IGNORECASE)
    if m:
        try:
            return float(m.group(1)) / 100.0
        except ValueError:
            return None
    return None


def face_threshold_validation_state() -> ValidationState:
    """Has the face-watchlist threshold been validated on real data?"""
    if not _REPORTS_DIR.exists():
        return ValidationState(
            False, None,
            "No reports/ directory — face-watchlist validation has never been run.",
        )

    reports = sorted(
        _REPORTS_DIR.glob("face_watchlist_validation_*.md"),
        key=lambda p: p.stat().st_mtime,
        reverse=True,
    )
    if not reports:
        return ValidationState(
            False, None,
            "No face_watchlist_validation_*.md report exists. Run "
            "`python -m backend.scripts.reid_validate --mode face "
            "--dataset-dir data/test_face_pairs/` against REAL labelled pairs.",
        )

    latest = reports[0]
    try:
        text = latest.read_text(encoding="utf-8", errors="replace")
    except OSError as exc:
        return ValidationState(False, str(latest), f"Report unreadable: {exc}")

    rel = str(latest.relative_to(_PROJECT_ROOT)).replace("\\", "/")
    for marker in _SYNTHETIC_MARKERS:
        if marker in text:
            return ValidationState(
                False, rel,
                f"Latest report is synthetic/degenerate (contains '{marker}'). "
                "Its numbers measure the random number generator, not face "
                "recognition accuracy. Real labelled Gujarat CCTV face pairs "
                "are required — see data/test_face_pairs/README.md.",
            )

    return ValidationState(
        True, rel, "Validated against a non-synthetic labelled dataset.",
        _extract_fpr(text),
    )


def face_matching_is_validated() -> bool:
    return face_threshold_validation_state().validated
