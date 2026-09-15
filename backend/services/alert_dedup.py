"""
backend/services/alert_dedup.py — Alert deduplication (Day 11).

Called as a shared hook immediately after db.add(alert)/db.flush() in each of
the four alert-firing modules (face_watchlist_matcher, loitering_detector,
crowd_detector, abandoned_object_detector).

WHAT IT DOES
  Looks for an existing Alert row that:
    - shares the same global_id as the new alert
    - shares the same alert_type
    - was created within the last DEDUP_WINDOW_SEC seconds
    - is itself not already merged into something else (only consider primaries)

  If found, compares the two and sets merged_into_alert_id on the loser.
  Returns the winner so the caller can broadcast/log the primary alert.

WHAT IT DOES NOT DO
  - Delete any Alert row. Evidence integrity (Day 5 design rule) means every
    fired alert remains queryable forever, even a redundant one.
  - Change lifecycle_status on any row. Merging is about display grouping, not
    a judgment on validity.
  - Dedup across different alert_types. A WATCHLIST_FACE_MATCH and a LOITERING
    for the same person within 60s are two different, both-valid signals.
  - Dedup alerts where global_id is NULL. The guard returns immediately on a
    null global_id, so this function is currently inert on every real alert
    in the live DB (see the Step 0 note in the Day 11 prompt: all tracks have
    global_id=NULL until the Track-persistence fix from Part A lands). That is
    the correct behavior, NOT a bug to work around.

TIE-BREAK (documented as a judgment call, not an obvious default)
  If both alerts have non-null confidence → higher confidence wins.
  Otherwise (loitering / crowd / abandoned-object have no natural confidence)
  → earliest created_at wins (first detection, not most recent). This is a
  deliberate choice: the first camera to fire is the one that caught the event
  earliest and is therefore the most informative primary record.

USAGE
  Called internally by each firing module. Do not call this from API routes.

    from backend.services.alert_dedup import apply_dedup

    db.add(alert)
    db.flush()                            # populate alert.id
    primary = apply_dedup(alert, db)      # may set merged_into_alert_id
    await check_zone_incident(primary, db)
    db.commit()                           # caller owns the commit
"""
from __future__ import annotations

import json
import logging
from datetime import datetime, timedelta
from typing import Any

from sqlalchemy.orm import Session

logger = logging.getLogger(__name__)

ACTION_ALERT_DEDUPED = "ALERT_DEDUPED"


def apply_dedup(new_alert: Any, db: Session) -> Any:
    """Dedup new_alert against recent primaries with the same global_id+type.

    Args:
        new_alert: The freshly flushed Alert ORM instance (id is populated).
        db:        The active session (same one the caller will commit).

    Returns:
        The primary Alert instance — either new_alert (if it won or no dupe
        found) or an existing alert (if new_alert was merged away).

    Side-effects:
        Sets merged_into_alert_id on the loser and flushes (but does not
        commit — the caller owns the transaction).
        Schedules an async broadcast and synchronous audit log entry.
    """
    from backend.core.config import settings
    from backend.db.models import Alert

    # ── Guard: null global_id → never dedup ──────────────────────────────────
    # See module docstring. Until Part A's Track-persistence fix lands,
    # every alert in this system has global_id=NULL, so this returns immediately
    # for every real alert. That is the correct, inert behavior (checkpoint 4).
    if not new_alert.global_id:
        return new_alert

    # ── Find existing primaries with same global_id + alert_type in window ───
    window_start = datetime.utcnow() - timedelta(seconds=settings.DEDUP_WINDOW_SEC)
    candidates = (
        db.query(Alert)
        .filter(
            Alert.global_id == new_alert.global_id,
            Alert.alert_type == new_alert.alert_type,
            Alert.created_at >= window_start,
            Alert.merged_into_alert_id.is_(None),   # only consider current primaries
            Alert.id != new_alert.id,               # exclude self
            Alert.is_deleted == False,
        )
        .all()
    )

    if not candidates:
        return new_alert  # no dupe — new_alert is primary

    # ── Pick the best existing candidate (earliest among candidates) ─────────
    # We compare new_alert against the single best existing primary. If there
    # are multiple existing candidates (edge case: three cameras firing within
    # the same window), we take the one that would win the tie-break against
    # new_alert — i.e. the one that wins by confidence first, then by
    # earliest created_at.
    def _score(a: Any) -> tuple:
        """Lower tuple → higher priority. Sorts by: high confidence first
        (negated so lower tuple = better), then earliest created_at."""
        conf = a.confidence if a.confidence is not None else -1.0
        return (-conf, a.created_at)

    best_existing = min(candidates, key=_score)

    # ── Tie-break between new_alert and best_existing ────────────────────────
    new_conf = new_alert.confidence
    best_conf = best_existing.confidence

    if new_conf is not None and best_conf is not None:
        # Both have confidence scores (WATCHLIST_FACE_MATCH) → higher wins.
        if new_conf >= best_conf:
            # new_alert wins: merge best_existing into it.
            loser, winner = best_existing, new_alert
        else:
            # existing wins: merge new_alert into it.
            loser, winner = new_alert, best_existing
    else:
        # Neither or only one has confidence (behavior alerts: loitering /
        # crowd / abandoned-object). Tie-break: earliest created_at wins.
        # Judgment call: first detection is the most informative primary
        # record — it caught the event before the others did.
        if new_alert.created_at <= best_existing.created_at:
            loser, winner = best_existing, new_alert
        else:
            loser, winner = new_alert, best_existing

    # ── Set merged_into_alert_id on the loser ────────────────────────────────
    loser.merged_into_alert_id = winner.id
    db.flush()   # persist the FK; caller commits

    logger.info(
        "ALERT_DEDUP: primary=%d loser=%d type=%s global_id=%s "
        "(tie-break: %s)",
        winner.id, loser.id, new_alert.alert_type, new_alert.global_id,
        "confidence" if (new_conf is not None and best_conf is not None)
        else "earliest_timestamp",
    )

    # ── Audit log (sync, within the same session/tx) ─────────────────────────
    try:
        from backend.services.audit_logger import log_audit
        log_audit(
            db, user=None, action=ACTION_ALERT_DEDUPED,
            resource_type="alert", resource_id=winner.id,
            details={
                "primary_alert_id": winner.id,
                "merged_alert_id": loser.id,
                "alert_type": new_alert.alert_type,
                "global_id": new_alert.global_id,
                "tie_break": (
                    "confidence"
                    if (new_conf is not None and best_conf is not None)
                    else "earliest_timestamp"
                ),
            },
        )
    except Exception:
        logger.exception("Failed to write ALERT_DEDUPED audit log")

    # ── Schedule async broadcast (fire-and-forget, non-blocking) ─────────────
    # We can't await here (apply_dedup is sync so it can be called from both
    # sync and async contexts). Schedule via asyncio.ensure_future instead.
    _schedule_dedup_broadcast(winner.id, loser.id)

    return winner


def _schedule_dedup_broadcast(primary_id: int, merged_id: int) -> None:
    """Schedule a WebSocket broadcast from a synchronous context."""
    import asyncio

    async def _do_broadcast() -> None:
        try:
            from backend.ws.dashboard_ws import broadcast
            await broadcast({
                "type": "alert_merged",
                "primary_alert_id": primary_id,
                "merged_alert_id": merged_id,
            })
        except Exception as exc:
            logger.debug("Failed to broadcast alert_merged: %s", exc)

    try:
        loop = asyncio.get_event_loop()
        if loop.is_running():
            asyncio.ensure_future(_do_broadcast())
        # If no running loop (test context), skip — broadcast is best-effort
    except RuntimeError:
        pass
