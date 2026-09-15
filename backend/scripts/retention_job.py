"""
backend/scripts/retention_job.py — Journey data + audit log retention purge (Day 9 §3).

PURPOSE:
  Enforces the two separate retention windows required by Day 9:
    1. Journey rows (journeys table): hard-delete after JOURNEY_RETENTION_DAYS.
       Both legacy rows (entry_timestamp column) and Day-6+ rows (seen_at) are
       handled, using whichever timestamp is non-null.
    2. Journey query log rows (journey_query_log): hard-delete after
       AUDIT_RETENTION_DAYS. The audit trail intentionally outlives the data
       it audits so access-pattern reviews remain possible even after the
       underlying journey data is gone.

  In both cases a summary count of deleted rows is logged — no silent purges.

WHAT IS DELETED:
  - journeys rows where the effective timestamp < now - JOURNEY_RETENTION_DAYS.
    Hard-delete (not soft-delete) because journey rows don't have an
    is_deleted column and the data minimisation goal IS the deletion itself.
  - journey_query_log rows where query_time < now - AUDIT_RETENTION_DAYS.

SEPARATE FROM reid_retention_job.py:
  - That job soft-deletes GlobalPerson records + rebuilds FAISS.
  - This job hard-deletes the movement-path rows themselves and their access log.
  - Both are wired into APScheduler in main.py on separate cron slots.

HOW TO RUN:
  Standalone:
    python -m backend.scripts.retention_job

  Via APScheduler (automatic, wired in main.py lifespan):
    Runs daily at 03:00 UTC.

  Dry run (no changes, just report what would be deleted):
    python -m backend.scripts.retention_job --dry-run

  Override retention window for testing (e.g. to test with 0 days):
    JOURNEY_RETENTION_DAYS=0 python -m backend.scripts.retention_job
"""
from __future__ import annotations

import argparse
import logging
import sys
from datetime import datetime, timedelta
from pathlib import Path

_PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))

logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
logger = logging.getLogger(__name__)


def run_journey_retention(dry_run: bool = False) -> dict:
    """Hard-delete journey rows older than JOURNEY_RETENTION_DAYS.

    Uses seen_at (Day-6+ rows) with fallback to entry_timestamp (legacy rows).
    Logs a count of deleted rows — never silent.

    Returns:
        Dict with deleted_count, cutoff, dry_run, run_at.
    """
    from backend.core.config import settings
    from backend.db.models import Journey, AuditLog
    from backend.db.session import SessionLocal

    db = SessionLocal()
    deleted_count = 0

    try:
        now = datetime.utcnow()
        cutoff = now - timedelta(days=settings.JOURNEY_RETENTION_DAYS)

        # Count candidates first (same query, no delete) for logging.
        # SQLAlchemy OR across nullable columns: a row matches if EITHER
        # seen_at < cutoff OR (seen_at is NULL AND entry_timestamp < cutoff_epoch).
        from sqlalchemy import or_, and_, cast, Float
        from backend.db.models import Journey

        cutoff_epoch = cutoff.timestamp()

        candidates_q = db.query(Journey).filter(
            or_(
                # Day-6+ rows: seen_at is the authoritative timestamp
                and_(Journey.seen_at.isnot(None), Journey.seen_at < cutoff),
                # Legacy Day-1/3 rows: entry_timestamp is a Unix float
                and_(
                    Journey.seen_at.is_(None),
                    Journey.entry_timestamp.isnot(None),
                    Journey.entry_timestamp < cutoff_epoch,
                ),
            )
        )

        candidates = candidates_q.all()
        count = len(candidates)
        logger.info(
            "Journey retention: found %d row(s) older than %s (JOURNEY_RETENTION_DAYS=%d, dry_run=%s).",
            count, cutoff.date().isoformat(), settings.JOURNEY_RETENTION_DAYS, dry_run,
        )

        if not dry_run and count > 0:
            for row in candidates:
                db.delete(row)
            deleted_count = count
            # Audit the deletion itself
            audit = AuditLog(
                user_id=None,
                action="JOURNEY_RETENTION_PURGE",
                resource_type="journey",
                resource_id=None,
                details=__import__("json").dumps({
                    "deleted_count": deleted_count,
                    "cutoff": cutoff.isoformat(),
                    "journey_retention_days": settings.JOURNEY_RETENTION_DAYS,
                    "run_at": now.isoformat(),
                }),
            )
            db.add(audit)
            db.commit()
            logger.info(
                "Journey retention: deleted %d row(s), cutoff=%s.",
                deleted_count, cutoff.date().isoformat(),
            )
        elif dry_run:
            deleted_count = count  # report what would have been deleted

        return {
            "deleted_count": deleted_count,
            "cutoff": cutoff.isoformat(),
            "journey_retention_days": settings.JOURNEY_RETENTION_DAYS,
            "dry_run": dry_run,
            "run_at": now.isoformat(),
        }

    except Exception as exc:
        db.rollback()
        logger.exception("Journey retention job failed: %s", exc)
        return {"deleted_count": 0, "error": str(exc)}
    finally:
        db.close()


def run_audit_retention(dry_run: bool = False) -> dict:
    """Hard-delete journey_query_log rows older than AUDIT_RETENTION_DAYS.

    The audit trail outlives the journey data it describes — default 365 days
    vs 30 days for journey rows. Both periods are config values.

    Returns:
        Dict with deleted_count, cutoff, dry_run, run_at.
    """
    from backend.core.config import settings
    from backend.db.models import AuditLog, JourneyQueryLog
    from backend.db.session import SessionLocal

    db = SessionLocal()
    deleted_count = 0

    try:
        now = datetime.utcnow()
        cutoff = now - timedelta(days=settings.AUDIT_RETENTION_DAYS)

        candidates = (
            db.query(JourneyQueryLog)
            .filter(JourneyQueryLog.query_time < cutoff)
            .all()
        )
        count = len(candidates)
        logger.info(
            "Audit retention: found %d journey_query_log row(s) older than %s "
            "(AUDIT_RETENTION_DAYS=%d, dry_run=%s).",
            count, cutoff.date().isoformat(), settings.AUDIT_RETENTION_DAYS, dry_run,
        )

        if not dry_run and count > 0:
            for row in candidates:
                db.delete(row)
            deleted_count = count
            # Log the audit-log purge in the general audit_log so there's
            # a permanent record that a purge happened (the journey_query_log
            # rows are gone, but the fact of the purge is recorded).
            purge_audit = AuditLog(
                user_id=None,
                action="AUDIT_LOG_RETENTION_PURGE",
                resource_type="journey_query_log",
                resource_id=None,
                details=__import__("json").dumps({
                    "deleted_count": deleted_count,
                    "cutoff": cutoff.isoformat(),
                    "audit_retention_days": settings.AUDIT_RETENTION_DAYS,
                    "run_at": now.isoformat(),
                }),
            )
            db.add(purge_audit)
            db.commit()
            logger.info(
                "Audit retention: deleted %d journey_query_log row(s), cutoff=%s.",
                deleted_count, cutoff.date().isoformat(),
            )
        elif dry_run:
            deleted_count = count

        return {
            "deleted_count": deleted_count,
            "cutoff": cutoff.isoformat(),
            "audit_retention_days": settings.AUDIT_RETENTION_DAYS,
            "dry_run": dry_run,
            "run_at": now.isoformat(),
        }

    except Exception as exc:
        db.rollback()
        logger.exception("Audit retention job failed: %s", exc)
        return {"deleted_count": 0, "error": str(exc)}
    finally:
        db.close()


async def run_full_retention(dry_run: bool = False) -> dict:
    """Run both retention passes sequentially. Called by APScheduler."""
    journey_result = run_journey_retention(dry_run=dry_run)
    audit_result = run_audit_retention(dry_run=dry_run)
    return {"journey": journey_result, "audit_log": audit_result}


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Day 9 journey + audit-log retention purge job"
    )
    parser.add_argument("--dry-run", action="store_true",
                        help="Report what would be deleted without making changes.")
    args = parser.parse_args()

    import asyncio
    result = asyncio.run(run_full_retention(dry_run=args.dry_run))

    print(f"\n{'=== DRY RUN ===' if args.dry_run else '=== EXECUTED ==='}")
    j = result.get("journey", {})
    a = result.get("audit_log", {})
    print(f"Journey rows {'would delete' if args.dry_run else 'deleted'}: "
          f"{j.get('deleted_count', 0)}  (cutoff: {j.get('cutoff', '?')})")
    print(f"Audit log rows {'would delete' if args.dry_run else 'deleted'}: "
          f"{a.get('deleted_count', 0)}  (cutoff: {a.get('cutoff', '?')})")
    if j.get("error"):
        print(f"Journey error: {j['error']}")
    if a.get("error"):
        print(f"Audit error: {a['error']}")
    if args.dry_run:
        print("\nNo changes were made. Re-run without --dry-run to execute.")


if __name__ == "__main__":
    main()
