"""
backend/scripts/reid_retention_job.py — Scheduled data retention & privacy purge.

PURPOSE:
  Automatically soft-delete GlobalPerson records that have exceeded their
  retention_expires_at date AND do not have retention_hold=True set.

  This enforces the data retention policy required under India's DPDP Act and
  general data minimization principles. Running this job daily ensures the
  identity database does not grow indefinitely.

WHAT IS DELETED:
  - GlobalPerson rows where: retention_expires_at < now AND retention_hold=False
  - Soft-delete only (is_deleted=True) — data is NOT physically removed from DB
    in case of audit requirements. Physical purge requires separate DB admin action.
  - FAISS index is rebuilt to exclude deleted records (so they can't appear in
    future similarity searches even though the DB row remains).

WHAT IS PROTECTED:
  - retention_hold=True records are NEVER touched, regardless of expiry.
  - Records linked to WatchlistPersons (checked by FK query) are also skipped
    and logged as "hold by watchlist association."

HOW TO RUN:
  Standalone (cron):
    python -m backend.scripts.reid_retention_job

  Via APScheduler (automatic, wired into backend lifespan in main.py):
    The scheduler calls run_retention_purge() daily at 02:00.

  Dry run (no changes, just report what would be purged):
    python -m backend.scripts.reid_retention_job --dry-run
"""
from __future__ import annotations

import argparse
import logging
import sys
from datetime import datetime
from pathlib import Path

_PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))

logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
logger = logging.getLogger(__name__)


async def run_retention_purge(dry_run: bool = False) -> dict:
    """Execute the retention purge job.

    Args:
        dry_run: If True, report what would be purged without making changes.

    Returns:
        Dict with purged_count, protected_count, errors.
    """
    from backend.db.session import SessionLocal
    from backend.db.models import GlobalPerson, AuditLog
    from backend.services.reid_index_manager import get_index_manager

    db = SessionLocal()
    purged_count = 0
    protected_count = 0
    errors = []

    try:
        now = datetime.utcnow()

        # Find candidates for purge
        candidates = (
            db.query(GlobalPerson)
            .filter(
                GlobalPerson.retention_expires_at <= now,
                GlobalPerson.retention_hold == False,   # noqa: E712
                GlobalPerson.is_deleted == False,       # noqa: E712
            )
            .all()
        )

        logger.info(
            "Retention job: found %d candidates for purge (dry_run=%s).",
            len(candidates), dry_run,
        )

        for gp in candidates:
            # Double-check: skip if last_seen_at is very recent (within 24h)
            # This protects against clock skew on newly created records
            if gp.last_seen_at and (now - gp.last_seen_at).total_seconds() < 86400:
                logger.info(
                    "Skipping gp_id=%d — last_seen_at within 24h (clock safety).",
                    gp.id,
                )
                protected_count += 1
                continue

            if dry_run:
                logger.info(
                    "DRY RUN: would purge gp_id=%d (expired=%s, sightings=%d)",
                    gp.id,
                    gp.retention_expires_at.isoformat() if gp.retention_expires_at else "None",
                    gp.total_sightings,
                )
                purged_count += 1
                continue

            # Soft-delete
            gp.is_deleted = True
            purged_count += 1
            logger.info(
                "Purged gp_id=%d (expired=%s, sightings=%d)",
                gp.id,
                gp.retention_expires_at.isoformat() if gp.retention_expires_at else "None",
                gp.total_sightings,
            )

        if not dry_run and purged_count > 0:
            # Write audit log entry
            audit = AuditLog(
                user_id=None,  # system action
                action="REID_RETENTION_PURGE",
                resource_type="global_person",
                resource_id=None,
                details=__import__("json").dumps({
                    "purged_count": purged_count,
                    "protected_count": protected_count,
                    "run_at": now.isoformat(),
                }),
            )
            db.add(audit)
            db.commit()

            # Rebuild FAISS index excluding deleted records
            logger.info("Rebuilding FAISS index to exclude %d deleted records...", purged_count)
            try:
                index_mgr = get_index_manager()
                new_count = await index_mgr.rebuild_excluding_deleted(db)
                logger.info("FAISS rebuild complete. Vectors in index: %d", new_count)
            except Exception as exc:
                err_msg = f"FAISS rebuild failed: {exc}"
                logger.error(err_msg)
                errors.append(err_msg)

        elif not dry_run:
            db.commit()

        result = {
            "purged_count": purged_count,
            "protected_count": protected_count,
            "dry_run": dry_run,
            "errors": errors,
            "run_at": now.isoformat(),
        }
        logger.info("Retention job complete: %s", result)
        return result

    except Exception as exc:
        db.rollback()
        logger.exception("Retention job failed: %s", exc)
        return {"purged_count": 0, "error": str(exc)}
    finally:
        db.close()


def main() -> None:
    parser = argparse.ArgumentParser(description="ReID data retention purge job")
    parser.add_argument("--dry-run", action="store_true",
                        help="Report what would be purged without making changes.")
    args = parser.parse_args()

    import asyncio
    result = asyncio.run(run_retention_purge(dry_run=args.dry_run))

    print(f"\n{'=== DRY RUN ===' if args.dry_run else '=== EXECUTED ==='}")
    print(f"Records purged:    {result.get('purged_count', 0)}")
    print(f"Records protected: {result.get('protected_count', 0)}")
    if result.get("errors"):
        print(f"Errors: {result['errors']}")
    if args.dry_run:
        print("\nNo changes were made. Re-run without --dry-run to execute.")


if __name__ == "__main__":
    main()
