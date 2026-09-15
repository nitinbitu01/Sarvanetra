"""backend/scripts/verify_measured_evidence.py — catches the exact mistake
that got made twice: a number typed into docs/MEASURED_EVIDENCE.md by hand,
then the system moves on without the doc.

WHY A CHECKER RATHER THAN AUTO-INJECTION
  Auto-injecting live numbers into the doc's prose was considered and rejected.
  The doc's value is the reasoning around each number — "why 2 rows, not 0"
  — and a template that only knows how to substitute a digit would either
  freeze that prose forever or need to regenerate paragraphs it has no
  business writing. A checker that fails loudly when reality has moved is
  cheaper and cannot corrupt the prose.

  This is the same pattern as `test_analytics_honesty.py` — assert against the
  live system, not against what the doc claims about it.

WHAT IT CHECKS
  Only the handful of numbers that have already drifted once:
    - promoted calibration count and which cameras
    - fleet census headline (30 deployed / footage / dark set)
    - field campaign size (25, post the needs_legs correction)
  Not exhaustive by design — it grows the next time a stale number is caught,
  rather than trying to parse arbitrary prose today.

Run:
  python -m backend.scripts.verify_measured_evidence
Exit code is nonzero if anything has drifted, so it can gate a demo dry run.
"""
from __future__ import annotations

import json
import re
import sqlite3
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

DOC = ROOT / "docs" / "MEASURED_EVIDENCE.md"
DB = ROOT / "output" / "sentinel.db"
CAMPAIGN = ROOT / "reports" / "field_campaign.json"


def check(label: str, condition: bool, detail: str) -> bool:
    print("  [%s] %s" % ("OK" if condition else "STALE", label))
    if not condition:
        print("         %s" % detail)
    return condition


def main() -> int:
    from backend.services.fleet_census import census
    from backend.db.session import SessionLocal

    doc = DOC.read_text(encoding="utf-8") if DOC.is_file() else ""
    ok = True

    con = sqlite3.connect(str(DB))
    promoted = sorted(r[0] for r in con.execute(
        "SELECT camera_id FROM camera_calibrations WHERE is_active = 1"
    ).fetchall())
    con.close()

    ok &= check(
        "promoted calibration count matches doc's '2 active rows'",
        len(promoted) == 2 and "**2 active rows**" in doc,
        "live=%d %s vs doc claims '2 active rows' (found: %s)"
        % (len(promoted), promoted, "yes" if "**2 active rows**" in doc else "no"))

    ok &= check(
        "promoted cameras are CAM_08 and CAM_11, and the doc names both",
        promoted == ["CAM_08", "CAM_11"]
        and "CAM_08" in doc and "CAM_11" in doc,
        "live=%s" % promoted)

    db = SessionLocal()
    try:
        cen = census(db)
    finally:
        db.close()
    headline = cen.headline()
    m = re.search(r"(\d+) of (\d+) deployed cameras have footage", headline)
    live_footage, live_fleet = (int(m.group(1)), int(m.group(2))) if m else (None, None)
    doc_has_current = ("27 of 30 deployed cameras have footage" in doc)
    ok &= check(
        "fleet census headline matches what the doc quotes",
        live_footage == 27 and live_fleet == 30 and doc_has_current,
        "live='%s' vs doc expects '27 of 30 deployed cameras have footage'"
        % headline)

    if CAMPAIGN.is_file():
        n_campaign = len(json.loads(CAMPAIGN.read_text(encoding="utf-8"))["cameras"])
        doc_has_25 = bool(re.search(r"\b25\b.*camera", doc))
        ok &= check(
            "field campaign size (25) is what the doc would need to say "
            "if it quotes a campaign count",
            n_campaign == 25,
            "live campaign has %d cameras tracked" % n_campaign)

    print("\n" + ("ALL CHECKS PASS — safe to quote docs/MEASURED_EVIDENCE.md as-is"
                  if ok else
                  "STALE NUMBERS FOUND — update docs/MEASURED_EVIDENCE.md before "
                  "quoting it, using the live values printed above"))
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
