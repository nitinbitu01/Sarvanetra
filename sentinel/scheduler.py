# sentinel/scheduler.py
"""
sentinel/scheduler.py — APScheduler jobs for hourly/daily snapshots, data retention, and audit verification.
"""

import structlog
from apscheduler.events import EVENT_JOB_ERROR, EVENT_JOB_MISSED
from apscheduler.schedulers.asyncio import AsyncIOScheduler

from sentinel.db import AsyncSessionLocal

log = structlog.get_logger(__name__)

scheduler = AsyncIOScheduler(timezone="UTC")


def _on_job_error(event):
    log.error("scheduler.job_failed",
              job_id=event.job_id,
              exception=str(event.exception),
              traceback=str(getattr(event, "traceback", "")))


def _on_job_missed(event):
    log.warning("scheduler.job_missed",
                job_id=event.job_id,
                scheduled_time=str(event.scheduled_run_time))


scheduler.add_listener(_on_job_error, EVENT_JOB_ERROR)
scheduler.add_listener(_on_job_missed, EVENT_JOB_MISSED)


async def _run_with_db(func):
    """Wrap a DB-dependent job with session lifecycle."""
    async with AsyncSessionLocal() as db:
        try:
            await func(db)
            await db.commit()
        except Exception:
            await db.rollback()
            raise
        finally:
            await db.close()


# ── Hourly analytics snapshot (:05 past every hour) ───────────────────────────
async def _hourly_snapshot_job():
    from sentinel.analytics.service import take_hourly_snapshot
    await _run_with_db(take_hourly_snapshot)

scheduler.add_job(
    _hourly_snapshot_job,
    trigger="cron",
    minute=5,
    id="hourly_analytics_snapshot",
    replace_existing=True,
    max_instances=1,
    misfire_grace_time=1800,
    coalesce=True,
)


# ── Daily analytics snapshot (00:05 UTC) ──────────────────────────────────────
async def _daily_snapshot_job():
    from sentinel.analytics.service import take_daily_snapshot
    await _run_with_db(take_daily_snapshot)

scheduler.add_job(
    _daily_snapshot_job,
    trigger="cron",
    hour=0,
    minute=5,
    id="daily_analytics_snapshot",
    replace_existing=True,
    max_instances=1,
    misfire_grace_time=3600,
    coalesce=True,
)


# ── Data retention enforcement (03:00 UTC) ────────────────────────────────────
async def _retention_job():
    from sentinel.retention.service import enforce_retention_policies
    await _run_with_db(enforce_retention_policies)

scheduler.add_job(
    _retention_job,
    trigger="cron",
    hour=3,
    minute=0,
    id="data_retention",
    replace_existing=True,
    max_instances=1,
    misfire_grace_time=7200,
    coalesce=True,
)


# ── Audit chain verification (Sunday 04:00 UTC) ───────────────────────────────
async def _audit_verify_job():
    from sentinel.audit.service import verify_audit_chain
    async with AsyncSessionLocal() as db:
        result = await verify_audit_chain(db)
        if not result["valid"]:
            log.critical("audit.chain_integrity_failed",
                         first_broken_at=result.get("first_broken_at"),
                         action="IMMEDIATE_INVESTIGATION_REQUIRED")
        else:
            log.info("audit.chain_verified", rows_checked=result["rows_checked"])

scheduler.add_job(
    _audit_verify_job,
    trigger="cron",
    day_of_week="sun",
    hour=4,
    minute=0,
    id="audit_chain_verification",
    replace_existing=True,
    max_instances=1,
)
