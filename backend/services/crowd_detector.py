"""
backend/services/crowd_detector.py — Crowd anomaly detection (Day 8, §1 + §3 + §4).

Core detection math (unchanged in shape from the pre-hardening version): a
camera's current on-screen person count is compared against a rolling
average of recent counts (the "baseline"). A candidate surge is a count that
is both above CROWD_MIN_ABSOLUTE_COUNT and at least CROWD_SURGE_MULTIPLIER
times the baseline. What's new in this pass is durability (Redis, not an
in-process ring buffer), the sustained-duration + cooldown gating (§3-style
lifecycle discipline applied to the detector itself, not just the alert
afterward), and the drift warning (§4).

Two negative cases the sustained-duration gate exists specifically for:
  - "Brief group photo, 10s": a real but short-lived count spike that
    shouldn't fire. CROWD_MIN_DURATION_SEC requires the surge condition to
    hold continuously before an alert fires, so a spike shorter than that
    window never fires.
  - "Gradual organic growth": because the baseline is a genuinely rolling
    average over the same window the current count is compared against, slow
    organic growth raises baseline_avg roughly in step with the live count —
    the ratio against CROWD_SURGE_MULTIPLIER never crosses threshold. This
    falls out of the rolling-baseline design itself, not a special case.

Per-camera state, keyed on the camera's string id (not per-track) — crowd
anomaly is inherently a camera-level signal.
"""
from __future__ import annotations

import json
import logging
import time
from dataclasses import dataclass
from datetime import datetime
from typing import Any

logger = logging.getLogger(__name__)

ALERT_TYPE = "CROWD_ANOMALY"
ACTION_CROWD_ANOMALY_FIRED = "CROWD_ANOMALY_ALERT_FIRED"


@dataclass(frozen=True)
class CrowdResult:
    decision: str
    camera_id: str
    count: int | None = None
    baseline_avg: float | None = None
    alert_id: int | None = None


class CrowdAnomalyDetector:
    def __init__(self) -> None:
        # Process-local dedupe/drift trackers, keyed by camera_id. Not
        # Redis-backed deliberately: losing these on restart just means one
        # extra baseline sample gets processed and one drift comparison is
        # skipped after a restart — much cheaper to accept than to persist,
        # and unlike the loitering/baseline windows themselves, nothing here
        # affects whether an alert fires or not.
        self._last_video_time: dict[str, float] = {}
        self._last_baseline_avg: dict[str, float] = {}

    async def on_frame_tick(
        self, camera_str_id: str, camera_db_id: int | None, video_time: float, current_count: int,
    ) -> CrowdResult:
        """Feed one (video_time, current_count) sample for a camera. Called
        on every detection event for that camera — see connection_manager.py's
        Day 8 hook, which passes len(current_state) as current_count."""
        from backend.core.config import settings
        from backend.services.state import get_behavior_state

        t0 = time.monotonic()
        state = get_behavior_state()

        # Dedupe: multiple track events can share the same frame/video_time
        # (one event per visible track). Only sample the baseline once per
        # distinct video_time so a crowded frame doesn't inflate its own
        # baseline weight relative to a sparse one.
        last_seen = self._last_video_time.get(camera_str_id)
        if last_seen is not None and video_time <= last_seen:
            return CrowdResult(decision="DUPLICATE_TICK", camera_id=camera_str_id)
        self._last_video_time[camera_str_id] = video_time

        # §4: frames_processed_total{camera_id} — counted here since a
        # distinct video_time is exactly "one new frame reached the behavior
        # engine" (this dedupe is the frame boundary; detector.py/tracker.py
        # are not touched to add frame-level instrumentation there).
        from backend.services.metrics import counters
        counters.incr("frames_processed_total", {"camera_id": camera_str_id})

        baseline_key = f"crowd:baseline:{camera_str_id}"
        member = f"{video_time}:{current_count}"
        ttl = int(settings.CROWD_BASELINE_WINDOW_SEC * 2)

        ok = await state.zadd_point(baseline_key, video_time, member, ttl)
        if not ok:
            self._log_tick(camera_str_id, video_time, "SKIPPED_NO_REDIS", t0)
            return CrowdResult(decision="SKIPPED_NO_REDIS", camera_id=camera_str_id)

        # Exclusive upper bound — see loitering_detector.py's identical fix
        # for why: an inclusive bound here would evict the very sample that
        # makes the window boundary correct at the threshold instant.
        window_start = video_time - settings.CROWD_BASELINE_WINDOW_SEC
        await state.zremrangebyscore(baseline_key, "-inf", f"({window_start}")

        points_raw = await state.zrange_withscores(baseline_key)
        if points_raw is None:
            self._log_tick(camera_str_id, video_time, "SKIPPED_NO_REDIS", t0)
            return CrowdResult(decision="SKIPPED_NO_REDIS", camera_id=camera_str_id)

        try:
            counts = [int(member.rsplit(":", 1)[1]) for member, _ in points_raw]
        except (ValueError, IndexError):
            logger.error("Malformed crowd baseline point in key=%s — skipping tick.", baseline_key)
            return CrowdResult(decision="ERROR", camera_id=camera_str_id)

        # Baseline excludes the just-added current point so a sudden crowd
        # can't inflate the very baseline it's being compared against.
        prior_counts = counts[:-1] if len(counts) > 1 else counts
        baseline_avg = sum(prior_counts) / len(prior_counts) if prior_counts else 0.0

        self._check_drift(camera_str_id, baseline_avg)

        cooldown_key = f"crowd:cooldown:{camera_str_id}"
        surge_start_key = f"crowd:surge_start:{camera_str_id}"

        is_candidate = (
            current_count >= settings.CROWD_MIN_ABSOLUTE_COUNT
            and baseline_avg > 0
            and current_count >= baseline_avg * settings.CROWD_SURGE_MULTIPLIER
        )

        if not is_candidate:
            await state.delete(surge_start_key)
            self._log_tick(camera_str_id, video_time, "NO_SURGE", t0)
            return CrowdResult(decision="NO_SURGE", camera_id=camera_str_id,
                                count=current_count, baseline_avg=baseline_avg)

        if await state.exists(cooldown_key):
            self._log_tick(camera_str_id, video_time, "COOLDOWN_ACTIVE", t0)
            return CrowdResult(decision="COOLDOWN_ACTIVE", camera_id=camera_str_id,
                                count=current_count, baseline_avg=baseline_avg)

        surge_start_raw = await state.get_simple(surge_start_key)
        if surge_start_raw is None:
            await state.set_simple(surge_start_key, str(video_time),
                                    ttl_seconds=int(settings.CROWD_MIN_DURATION_SEC * 3))
            self._log_tick(camera_str_id, video_time, "SURGE_STARTED", t0)
            return CrowdResult(decision="SURGE_STARTED", camera_id=camera_str_id,
                                count=current_count, baseline_avg=baseline_avg)

        sustained = video_time - float(surge_start_raw)
        if sustained < settings.CROWD_MIN_DURATION_SEC:
            self._log_tick(camera_str_id, video_time, "SURGE_NOT_SUSTAINED", t0)
            return CrowdResult(decision="SURGE_NOT_SUSTAINED", camera_id=camera_str_id,
                                count=current_count, baseline_avg=baseline_avg)

        alert_id = await self._fire_alert(
            camera_str_id=camera_str_id, camera_db_id=camera_db_id, current_count=current_count,
            baseline_avg=baseline_avg, sustained_sec=sustained, video_time=video_time,
        )
        await state.set_simple(cooldown_key, "1", ttl_seconds=int(settings.CROWD_COOLDOWN_SEC))
        await state.delete(surge_start_key)
        self._log_tick(camera_str_id, video_time, "CROWD_ANOMALY_FIRED", t0)
        return CrowdResult(decision="CROWD_ANOMALY_FIRED", camera_id=camera_str_id,
                            count=current_count, baseline_avg=baseline_avg, alert_id=alert_id)

    def _check_drift(self, camera_str_id: str, baseline_avg: float) -> None:
        """§4: warn if the rolling baseline itself jumps sharply between
        consecutive computations — often a sign of a miscounted/degraded
        detector upstream, not an actual crowd."""
        from backend.core.config import settings

        prev = self._last_baseline_avg.get(camera_str_id)
        if prev and prev > 0 and baseline_avg > 0:
            ratio = (baseline_avg / prev) if baseline_avg >= prev else (prev / baseline_avg)
            if ratio >= settings.CROWD_DRIFT_WARN_MULTIPLIER:
                logger.warning(
                    "Crowd baseline drift on %s: %.2f -> %.2f (%.1fx) — check upstream "
                    "detector health, this may not be a real crowd change.",
                    camera_str_id, prev, baseline_avg, ratio,
                )
        self._last_baseline_avg[camera_str_id] = baseline_avg

    @staticmethod
    async def _fire_alert(
        camera_str_id: str, camera_db_id: int | None, current_count: int, baseline_avg: float,
        sustained_sec: float, video_time: float,
    ) -> int | None:
        from backend.db.models import Alert
        from backend.db.session import SessionLocal
        from backend.services.audit_logger import log_audit
        from backend.services.metrics import counters
        from backend.services.sentinel_iq import IQContext, compute_alert_iq

        db = SessionLocal()
        try:
            # Day 10: crowd anomaly is camera-level, not person-level — there
            # is no track to trace, so the cross-zone multiplier will record
            # `not_applicable_no_track` and no-op. That is the correct and
            # permanent answer here, distinct from an unresolved global id.
            iq = compute_alert_iq(ALERT_TYPE, IQContext(
                db=db, camera_db_id=camera_db_id, camera_str_id=camera_str_id,
            ))

            alert = Alert(
                alert_type=ALERT_TYPE,
                camera_id=camera_str_id or str(camera_db_id or "CAM_01"),
                subject_label=f"Crowd anomaly — {current_count} people "
                               f"(baseline ~{baseline_avg:.1f}) at {camera_str_id}",
                confidence=None,
                danger_score=None,
                iq_contribution=iq.total if iq else None,
                iq_breakdown_json=iq.to_json() if iq else None,
                status="new",
                lifecycle_status="OPEN",
                meta_json=json.dumps({
                    "camera_id": camera_str_id, "current_count": current_count,
                    "baseline_avg": round(baseline_avg, 2), "sustained_sec": round(sustained_sec, 1),
                    "video_time": video_time,
                }),
            )
            db.add(alert)
            db.flush()

            from backend.services.alert_dedup import apply_dedup
            from backend.services.evidence_capture import on_alert_fired
            from backend.services.zone_incident_manager import check_zone_incident

            primary_alert = apply_dedup(alert, db)
            await check_zone_incident(primary_alert, db)
            # Day 13: severity + evidence capture on the PRIMARY alert.
            await on_alert_fired(primary_alert, db, camera_str_id)

            log_audit(
                db, user=None, action=ACTION_CROWD_ANOMALY_FIRED,
                resource_type="alert", resource_id=alert.id,
                details={
                    "camera": camera_str_id, "current_count": current_count,
                    "baseline_avg": round(baseline_avg, 2), "sustained_sec": round(sustained_sec, 1),
                    "merged_into_alert_id": alert.merged_into_alert_id,
                },
            )
            db.commit()

            counters.incr("alerts_fired_total", {"type": ALERT_TYPE, "camera_id": camera_str_id})

            alert_dict = {
                "id": alert.id, "alert_type": alert.alert_type,
                "merged_into_alert_id": alert.merged_into_alert_id,
                "subject_label": alert.subject_label, "confidence": alert.confidence,
                "danger_score": alert.danger_score, "status": alert.status,
                "iq_contribution": alert.iq_contribution,
                "iq_breakdown": iq.to_dict() if iq else None,
                "lifecycle_status": alert.lifecycle_status, "snapshot_path": None,
                "evidence_hash": None, "false_positive_reason": None,
                "camera_name": camera_str_id, "camera_zone": None,
                "created_at": alert.created_at.isoformat() if alert.created_at else "",
                "metadata": json.loads(alert.meta_json),
            }
            await CrowdAnomalyDetector._broadcast(alert_dict)

            logger.info("CROWD_ANOMALY fired: %s count=%d baseline=%.1f sustained=%.1fs",
                        camera_str_id, current_count, baseline_avg, sustained_sec)
            return alert.id
        except Exception:
            db.rollback()
            logger.exception("Failed to fire CROWD_ANOMALY alert for %s", camera_str_id)
            return None
        finally:
            db.close()

    @staticmethod
    async def _broadcast(alert_dict: dict[str, Any]) -> None:
        try:
            from backend.ws.dashboard_ws import broadcast
            await broadcast({"type": "new_alert", "alert": alert_dict})
        except Exception as exc:
            logger.debug("Failed to broadcast CROWD_ANOMALY alert: %s", exc)

    @staticmethod
    def _log_tick(camera_id: str, video_time: float, decision: str, t0: float) -> None:
        from backend.services.metrics import counters
        latency_ms = (time.monotonic() - t0) * 1000
        counters.observe_latency("crowd", latency_ms)
        logger.debug(json.dumps({
            "camera_id": camera_id, "global_id": None, "detector": "crowd",
            "video_time": video_time, "decision": decision, "latency_ms": round(latency_ms, 2),
        }))


_detector_instance: CrowdAnomalyDetector | None = None


def get_crowd_detector() -> CrowdAnomalyDetector:
    global _detector_instance
    if _detector_instance is None:
        _detector_instance = CrowdAnomalyDetector()
    return _detector_instance
