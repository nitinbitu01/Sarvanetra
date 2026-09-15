"""
backend/scripts/reid_calibration_report.py — Threshold recalibration feedback loop.

PURPOSE:
  Reads ReIDCalibrationLog from the DB (populated by reid_matcher.py for every
  resolved identity decision) and computes:
  - False-merge rate estimate from AUTO_MERGE decisions that were later confirmed wrong
  - Review queue outcome rates (approval vs rejection ratio)
  - Suggested threshold adjustments based on real operational data

OUTPUT:
  Console summary + reports/reid_calibration_<date>.md

CRITICAL: This script does NOT auto-apply any threshold changes.
  It produces a SUGGESTION that an authorized admin must review before
  manually updating REID_AUTO_MERGE_THRESHOLD and REID_REVIEW_LOWER_THRESHOLD
  in .env. The 0.85/0.65 defaults are not "set forever" — they should be
  revisited periodically as this report accumulates more ground-truth data.

HOW TO RUN:
  python -m backend.scripts.reid_calibration_report

HOW OFTEN:
  Run monthly (or after any significant volume of human review decisions
  has accumulated in reid_calibration_log). More data = better suggestion.
  With <50 labeled decisions, treat the suggestion as directional only.
"""
from __future__ import annotations

import argparse
import sys
from datetime import datetime
from pathlib import Path

_PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))


def _load_calibration_data(db):
    """Load all ReIDCalibrationLog rows as dicts."""
    from backend.db.models import ReIDCalibrationLog
    rows = db.query(ReIDCalibrationLog).order_by(ReIDCalibrationLog.created_at.asc()).all()
    return [
        {
            "id": r.id,
            "similarity_score": r.similarity_score,
            "decision": r.decision,
            "was_correct": r.was_correct,
            "time_gap_seconds": r.time_gap_seconds,
            "created_at": r.created_at,
        }
        for r in rows
    ]


def analyze(rows: list[dict], auto_t: float, review_t: float) -> dict:
    """Compute calibration statistics from log rows."""
    total = len(rows)
    if total == 0:
        return {"error": "No calibration data found."}

    auto_merges = [r for r in rows if r["decision"] == "REID_AUTO_MERGE"]
    new_identities = [r for r in rows if r["decision"] == "REID_NEW_IDENTITY"]
    reviews_approved = [r for r in rows if r["decision"] == "REID_REVIEW_APPROVED"]
    reviews_rejected = [r for r in rows if r["decision"] == "REID_REVIEW_REJECTED"]

    # False-merge rate: AUTO_MERGE decisions where was_correct=False
    reviewed_auto = [r for r in auto_merges if r["was_correct"] is not None]
    false_merges = [r for r in reviewed_auto if r["was_correct"] is False]
    false_merge_rate = len(false_merges) / len(reviewed_auto) if reviewed_auto else None

    # Review queue outcome rate
    total_reviews = len(reviews_approved) + len(reviews_rejected)
    approval_rate = len(reviews_approved) / total_reviews if total_reviews > 0 else None
    rejection_rate = len(reviews_rejected) / total_reviews if total_reviews > 0 else None

    # Score statistics for each decision type
    def score_stats(subset):
        scores = [r["similarity_score"] for r in subset if r["similarity_score"] is not None]
        if not scores:
            return {"n": 0, "mean": None, "min": None, "max": None}
        import numpy as np
        return {
            "n": len(scores),
            "mean": round(float(__import__("numpy").mean(scores)), 4),
            "min": round(min(scores), 4),
            "max": round(max(scores), 4),
        }

    # Threshold suggestions
    suggestions = []
    if false_merge_rate is not None and false_merge_rate > 0.02 and len(reviewed_auto) >= 20:
        suggestions.append(
            f"False-merge rate is {false_merge_rate:.1%} (>2% threshold). "
            f"Consider RAISING auto-merge threshold above {auto_t}."
        )
    elif false_merge_rate is not None and false_merge_rate < 0.005 and approval_rate is not None and approval_rate > 0.9:
        suggestions.append(
            f"False-merge rate is {false_merge_rate:.1%} and review approval rate is {approval_rate:.1%}. "
            f"Consider LOWERING auto-merge threshold toward {round(auto_t - 0.02, 2)} "
            f"(fewer items needing human review)."
        )
    elif total_reviews < 20:
        suggestions.append(
            f"Only {total_reviews} reviewed decisions available. "
            f"Suggestion is directional only — collect more data before adjusting thresholds."
        )

    if rejection_rate is not None and rejection_rate > 0.5 and total_reviews >= 20:
        suggestions.append(
            f"Review rejection rate is {rejection_rate:.1%}. "
            f"Consider RAISING the review-queue lower threshold above {review_t} "
            f"(many suggestions are wrong — fewer borderlines reaching officers)."
        )
    elif approval_rate is not None and approval_rate > 0.9 and total_reviews >= 20:
        suggestions.append(
            f"Review approval rate is {approval_rate:.1%}. "
            f"Consider LOWERING the auto-merge threshold toward {round(auto_t - 0.03, 2)} "
            f"(officers agree most of the time — let the engine decide more)."
        )

    return {
        "total": total,
        "auto_merges": score_stats(auto_merges),
        "new_identities": score_stats(new_identities),
        "reviews_approved": score_stats(reviews_approved),
        "reviews_rejected": score_stats(reviews_rejected),
        "false_merge_rate": false_merge_rate,
        "false_merges_from_reviewed_auto": len(false_merges),
        "reviewed_auto_count": len(reviewed_auto),
        "approval_rate": approval_rate,
        "rejection_rate": rejection_rate,
        "total_reviews": total_reviews,
        "suggestions": suggestions,
    }


def generate_report(stats: dict, output_dir: Path, auto_t: float, review_t: float) -> Path:
    date_str = datetime.utcnow().strftime("%Y-%m-%d %H:%M UTC")

    if "error" in stats:
        report = f"# ReID Calibration Report\n\nGenerated: {date_str}\n\n⚠️ {stats['error']}\n"
    else:
        false_merge_display = (
            f"{stats['false_merge_rate']:.1%}" if stats["false_merge_rate"] is not None
            else "N/A (no reviewed auto-merges)"
        )
        approval_display = (
            f"{stats['approval_rate']:.1%}" if stats["approval_rate"] is not None
            else "N/A"
        )
        rejection_display = (
            f"{stats['rejection_rate']:.1%}" if stats["rejection_rate"] is not None
            else "N/A"
        )

        suggestions_md = "\n".join(f"- {s}" for s in stats["suggestions"]) or "- No changes suggested at this time."

        report = f"""# Sentinel Gujarat — ReID Calibration Report

Generated: {date_str}
Current thresholds: auto-merge={auto_t}, review-lower={review_t}

> ⚠️ This report is a SUGGESTION. Do NOT auto-apply threshold changes.
> An authorized admin must review and manually update `.env` if they agree.

---

## Decision Volume

| Decision Type | Count | Mean Score | Min | Max |
|---------------|-------|------------|-----|-----|
| AUTO_MERGE | {stats['auto_merges']['n']} | {stats['auto_merges']['mean'] or 'N/A'} | {stats['auto_merges']['min'] or 'N/A'} | {stats['auto_merges']['max'] or 'N/A'} |
| NEW_IDENTITY | {stats['new_identities']['n']} | {stats['new_identities']['mean'] or 'N/A'} | {stats['new_identities']['min'] or 'N/A'} | {stats['new_identities']['max'] or 'N/A'} |
| REVIEW_APPROVED | {stats['reviews_approved']['n']} | {stats['reviews_approved']['mean'] or 'N/A'} | {stats['reviews_approved']['min'] or 'N/A'} | {stats['reviews_approved']['max'] or 'N/A'} |
| REVIEW_REJECTED | {stats['reviews_rejected']['n']} | {stats['reviews_rejected']['mean'] or 'N/A'} | {stats['reviews_rejected']['min'] or 'N/A'} | {stats['reviews_rejected']['max'] or 'N/A'} |
| **Total** | **{stats['total']}** | | | |

---

## Key Metrics

| Metric | Value | Interpretation |
|--------|-------|----------------|
| False-merge rate | {false_merge_display} | % of auto-merges later confirmed wrong |
| Reviewed auto-merges | {stats['reviewed_auto_count']} / {stats['auto_merges']['n']} | Ground truth coverage |
| Review approval rate | {approval_display} | How often officers agree with suggestions |
| Review rejection rate | {rejection_display} | How often suggestions are wrong |
| Total human reviews | {stats['total_reviews']} | Sample size for suggestions |

---

## Threshold Suggestions

{suggestions_md}

> **To apply a change:** Update `.env`:
> ```
> REID_AUTO_MERGE_THRESHOLD=<new_value>
> REID_REVIEW_LOWER_THRESHOLD=<new_value>
> ```
> Then restart the backend and re-run `scripts/reid_validate.py` to confirm improvement.

---

## Data Quality Note

{"⚠️ Fewer than 50 reviewed decisions — treat suggestions as directional only." if stats['total_reviews'] < 50 else "✅ Sufficient reviewed decisions for reliable suggestions."}

Re-run this report monthly or after significant new review data accumulates.
"""

    output_dir.mkdir(parents=True, exist_ok=True)
    filename = f"reid_calibration_{datetime.utcnow().strftime('%Y%m%d_%H%M%S')}.md"
    report_path = output_dir / filename
    report_path.write_text(report, encoding="utf-8")
    return report_path


def main() -> None:
    parser = argparse.ArgumentParser(description="ReID threshold calibration report")
    parser.add_argument(
        "--output-dir", type=Path,
        default=_PROJECT_ROOT / "reports",
        help="Directory for the output report. Default: reports/"
    )
    args = parser.parse_args()

    from backend.db.session import SessionLocal
    from backend.core.config import settings

    db = SessionLocal()
    try:
        rows = _load_calibration_data(db)
        print(f"Loaded {len(rows)} calibration log entries.")
    finally:
        db.close()

    stats = analyze(rows, settings.REID_AUTO_MERGE_THRESHOLD,
                    settings.REID_REVIEW_LOWER_THRESHOLD)

    # Print summary to console
    if "error" in stats:
        print(f"⚠️ {stats['error']}")
    else:
        print(f"\n=== Calibration Summary ===")
        print(f"Total decisions:       {stats['total']}")
        print(f"False-merge rate:      {stats['false_merge_rate']:.1%}" if stats['false_merge_rate'] is not None else "False-merge rate: N/A")
        print(f"Review approval rate:  {stats['approval_rate']:.1%}" if stats['approval_rate'] is not None else "Review approval rate: N/A")
        print(f"\nSuggestions:")
        for s in stats["suggestions"]:
            print(f"  → {s}")

    report_path = generate_report(
        stats, args.output_dir,
        settings.REID_AUTO_MERGE_THRESHOLD,
        settings.REID_REVIEW_LOWER_THRESHOLD,
    )
    print(f"\n[OK] Report written to: {report_path}")
    print("\n[NOTE] No changes have been applied. Review the report and update .env manually if you agree.")


if __name__ == "__main__":
    main()
