"""
backend/services/abandoned_object_detector.py — Abandoned-object detection (Day 9 §4 + §6).

WHAT THIS DETECTS:
  An object is considered "abandoned" when:
    1. A static object appears (its centroid drifts <= STATIC_MOVEMENT_PX_THRESHOLD
       over consecutive ticks).
    2. Its candidate owner (the last person within OWNERSHIP_RADIUS_METERS when it
       first became static) has left the radius.
    3. The object remains static AND unattended for >= ABANDON_DURATION_SEC of
       continuous video_time.
    4. No other person has been within OWNERSHIP_RADIUS_PX during the static window
       (the "any person nearby" suppression rule — a bystander standing near an
       object should not trigger an alert).
    5. The object was NOT present during the warm-up window (baseline objects —
       things that were already there when recording started — never fire).

FALSE-POSITIVE RISK PROFILE:
  This detector has the highest FP risk in the system. Three inputs in particular
  drive the most risk:
    STATIC_MOVEMENT_PX_THRESHOLD — too high: moving objects fire; too low: jitter FPs
    ABANDON_DURATION_SEC         — too low: brief deposits fire; too high: misses fast drop-and-run
    OWNERSHIP_RADIUS_METERS      — too large: owner 10m away still "claims" the object

  Re-run tests/abandoned_object_eval/eval_runner.py whenever any of these three is
  changed. They are all settings values, not constants, so a tuning change is a
  .env edit, not a code edit — but the regression check still applies regardless.

STATE PERSISTENCE (Day 9 §6):
  All per-object state lives in Redis (via state.py) using:
    object_track:{object_track_id}     — hash: camera_id, class, first_seen_time,
                                         last_seen_time, position_history (capped list),
                                         candidate_owner_global_id, is_baseline, status
    object_proximity:{object_track_id} — simple TTL key; set while any person is
                                         within radius, auto-expires when they leave

  If Redis is unreachable, skip detection and log a warning (same pattern as
  loitering_detector.py and crowd_detector.py). Never buffer in memory, never crash.

CALIBRATION (Day 9 §4):
  OWNERSHIP_RADIUS_PX = OWNERSHIP_RADIUS_METERS * px_per_meter, looked up via
  camera_calibration (same CalibrationCache as loitering_detector.py). If a camera
  has no calibration row, falls back to DEFAULT_PX_PER_METER and tags the resulting
  alert with calibration_method='uncalibrated' in details_json.
  STATIC_MOVEMENT_PX_THRESHOLD stays in pixel space (it guards against camera/object
  jitter, not a physical distance).

WIRED INTO THE LIVE PIPELINE (Day 10):
  config.yaml's detection.class_filter includes backpack (24), handbag (26),
  and suitcase (28) — previously it did not, so no object tracks existed for
  this detector to consume regardless of whether anything called it.
  main.py's pipeline loop runs a third TrackStateManager for these classes
  and emits confirmed object tracks via event_emitter.emit_objects() as
  event_type="object_track" (a separate JSONL/queue stream from person
  events — see event_emitter.py's module docstring). pipeline_bridge's drain
  loop routes those events to connection_manager.handle_object_event(),
  which computes person-to-object proximity from its own in-memory person
  positions and calls the two entry points below, in the order that
  matters — proximity refreshed before the object tick that reads it.
  Gated behind connection_manager._abandoned_ready, flipped by
  set_abandoned_ready() from main.py's lifespan.

  is_baseline is decided once per object track, in main.py, from whether the
  track's first confirmation fell within config.yaml's
  abandoned_object.warmup_sec of pipeline start — see that key's comment for
  what the heuristic does and does not claim to know; it is a product
  decision that shipped with a documented default, not a solved problem.

INTERFACE:
  AbandonedObjectDetector.on_object_update(camera_str_id, camera_db_id, object_track_id,
      object_class, cx, cy, video_time, is_baseline)
    → Called once per confirmed object track per processed frame, from
      connection_manager.handle_object_event().

  AbandonedObjectDetector.on_person_positions(camera_str_id, object_track_ids_nearby,
      video_time)
    → Called once per object event for which connection_manager found at
      least one currently-tracked person within OWNERSHIP_RADIUS_PX (it
      already holds every confirmed person's latest bbox in
      self.current_state) — spatial proximity computed by the caller, kept
      out of this class so its logic stays pure and testable.
"""
from __future__ import annotations

import json
import logging
import math
import time
from dataclasses import dataclass
from datetime import datetime
from typing import Any

logger = logging.getLogger(__name__)

ALERT_TYPE = "ABANDONED_OBJECT"
ACTION_ABANDONED_FIRED = "ABANDONED_OBJECT_ALERT_FIRED"

# Maximum entries kept in position_history per object (capped to limit Redis payload).
_MAX_POSITION_HISTORY = 20


@dataclass(frozen=True)
class AbandonResult:
    decision: str
    object_track_id: str
    span_sec: float | None = None
    alert_id: int | None = None


class AbandonedObjectDetector:
    """Per-frame state machine for abandoned-object detection.

    Call on_object_update() for each detected static object per frame.
    Call on_person_positions() once per frame with all objects that have a
    person within radius — the caller owns the spatial query.
    """

    async def on_object_update(
        self,
        camera_str_id: str,
        camera_db_id: int | None,
        object_track_id: str,
        object_class: str,
        cx: float,
        cy: float,
        video_time: float,
        is_baseline: bool = False,
    ) -> AbandonResult:
        """Process one tick for a detected static object.

        Args:
            camera_str_id: Camera string ID (e.g. 'CAM-01').
            camera_db_id:  Integer FK to cameras table (None = not yet resolved).
            object_track_id: Unique track ID for this object (e.g. 'CAM-01:obj:42').
            object_class:  Object class label (e.g. 'suitcase', 'bag').
            cx, cy:        Centroid of the object's bounding box (pixels).
            video_time:    Current video timestamp (seconds, monotonic within clip).
            is_baseline:   True if this object was already present at recording start
                           (warm-up window). Baseline objects never fire.

        Returns:
            AbandonResult with decision and, on alert fire, alert_id.
        """
        from backend.core.config import settings
        from backend.services.state import get_behavior_state

        t0 = time.monotonic()
        state = get_behavior_state()

        # ── Baseline guard: objects that were already there never fire ───────
        if is_baseline:
            self._log_tick(camera_str_id, object_track_id, video_time, "BASELINE_OBJECT", t0)
            return AbandonResult(decision="BASELINE_OBJECT", object_track_id=object_track_id)

        # ── Load existing track state from Redis ─────────────────────────────
        existing = await state.hget_object_track(object_track_id)
        if existing is None:
            # Redis unreachable — skip this tick, log, continue next frame.
            self._log_tick(camera_str_id, object_track_id, video_time, "SKIPPED_NO_REDIS", t0)
            return AbandonResult(decision="SKIPPED_NO_REDIS", object_track_id=object_track_id)

        now_iso = datetime.utcnow().isoformat()
        ttl = int(settings.ABANDON_DURATION_SEC * 3)  # generous TTL — object may persist long

        if not existing:
            # First time we see this object — initialize its state.
            pos_history = [{"cx": cx, "cy": cy, "t": video_time}]
            ok = await state.hset_object_track(object_track_id, {
                "camera_id": camera_str_id,
                "class": object_class,
                "first_seen_time": video_time,
                "last_seen_time": video_time,
                "position_history": pos_history,
                "candidate_owner_global_id": "",
                "is_baseline": "false",
                "status": "tracking",
            }, ttl_seconds=ttl)
            if not ok:
                self._log_tick(camera_str_id, object_track_id, video_time, "SKIPPED_NO_REDIS", t0)
                return AbandonResult(decision="SKIPPED_NO_REDIS", object_track_id=object_track_id)
            self._log_tick(camera_str_id, object_track_id, video_time, "FIRST_SEEN", t0)
            return AbandonResult(decision="FIRST_SEEN", object_track_id=object_track_id)

        # ── Check if already fired (debounce) ────────────────────────────────
        if existing.get("status") == "alert_fired":
            await state.expire_object_track(object_track_id, ttl)
            self._log_tick(camera_str_id, object_track_id, video_time, "ALREADY_FIRED", t0)
            return AbandonResult(decision="ALREADY_FIRED", object_track_id=object_track_id)

        # ── Update position history (capped) ─────────────────────────────────
        pos_history: list[dict] = existing.get("position_history") or []
        pos_history.append({"cx": cx, "cy": cy, "t": video_time})
        if len(pos_history) > _MAX_POSITION_HISTORY:
            pos_history = pos_history[-_MAX_POSITION_HISTORY:]

        # ── Static check: is the object actually still? ───────────────────────
        # Compare latest position vs position from ABANDON_DURATION_SEC ago.
        # If the centroid has drifted more than STATIC_MOVEMENT_PX_THRESHOLD it
        # isn't static — reset the first_seen_time to now (rolling window).
        first_seen_time = float(existing.get("first_seen_time", video_time))
        span_sec = video_time - first_seen_time

        if len(pos_history) >= 2:
            oldest_in_window = pos_history[0]
            drift = math.hypot(cx - oldest_in_window["cx"], cy - oldest_in_window["cy"])
            if drift > settings.STATIC_MOVEMENT_PX_THRESHOLD:
                # Object moved — reset window.
                await state.hset_object_track(object_track_id, {
                    "first_seen_time": video_time,
                    "last_seen_time": video_time,
                    "position_history": [{"cx": cx, "cy": cy, "t": video_time}],
                    "status": "tracking",
                }, ttl_seconds=ttl)
                self._log_tick(camera_str_id, object_track_id, video_time, "OBJECT_MOVED", t0)
                return AbandonResult(decision="OBJECT_MOVED", object_track_id=object_track_id)

        # ── Proximity suppression: any person nearby → suppress alert ─────────
        #
        # "Has the owner left?" is answered on the SAME clock as "how long has
        # this been unattended?" — video_time. It previously was not: the
        # proximity marker was a Redis key whose only means of clearing was a
        # wall-clock TTL, while first_seen_time/span_sec/ABANDON_DURATION_SEC
        # are all video_time. Two clocks answering one question meant:
        #   - On any feed faster than real-time (config.yaml's
        #     `loop_video: true`, or catch-up after a backlog), minutes of
        #     video_time passed inside a few seconds of wall-clock, so the TTL
        #     had not expired and genuine abandonment was suppressed outright.
        #   - On a stalled or slow feed, the reverse: the marker expired while
        #     the owner was still standing next to the object.
        #   - tests/abandoned_object_eval's only true-positive clip could never
        #     fire at all (120s of video_time in ~10ms of wall-clock), so this
        #     detector's positive path was never actually verified.
        # The marker now carries the video_time of the sighting; its TTL is
        # garbage collection only. See BehaviorState.set_proximity_flag.
        last_nearby_vt = await state.get_proximity_video_time(object_track_id)
        proximity_window = settings.ABANDON_PROXIMITY_WINDOW_SEC

        if last_nearby_vt is None:
            # No record, or a legacy time-less "1" sentinel. Fall back to bare
            # existence: if something marked this object as attended but we
            # cannot tell when, suppressing the alert is the conservative
            # reading — a missed abandonment alert is recoverable, a false
            # CRITICAL-adjacent alert on an attended bag is not.
            someone_nearby = await state.get_proximity_flag(object_track_id)
        else:
            someone_nearby = (video_time - last_nearby_vt) <= proximity_window

        if someone_nearby:
            # Restart the unattended clock, don't merely pause it.
            #
            # span_sec is measured from first_seen_time, so leaving that alone
            # while a person stands beside the object means the object keeps
            # accruing "unattended" time WHILE IT IS BEING ATTENDED. The
            # question this detector answers is "has this been unattended for
            # ABANDON_DURATION_SEC", and the clock for that can only start
            # when the last person left.
            #
            # This was previously unreachable and therefore unnoticed: the
            # wall-clock proximity TTL (see above) meant the flag never
            # cleared during a run, so nothing ever came out of this branch to
            # then fire incorrectly. Fixing the clock exposed it — the
            # edge_bystander_near_object clip fired a false positive at t=60
            # having counted the 30 seconds a bystander spent standing next to
            # the suitcase as unattended time. That clip's own description
            # already specified the intended rule: "After the bystander leaves
            # at t=50, the abandonment window restarts from t=50."
            #
            # OBJECT_MOVED above already resets first_seen_time on the same
            # reasoning, so this keeps the two reset paths consistent.
            await state.hset_object_track(object_track_id, {
                "first_seen_time": video_time,
                "last_seen_time": video_time,
                "position_history": pos_history,
                "status": "tracking",
            }, ttl_seconds=ttl)
            self._log_tick(camera_str_id, object_track_id, video_time, "PERSON_NEARBY", t0)
            return AbandonResult(decision="PERSON_NEARBY", object_track_id=object_track_id,
                                  span_sec=span_sec)

        # ── Duration check: has the object been unattended long enough? ───────
        await state.hset_object_track(object_track_id, {
            "last_seen_time": video_time,
            "position_history": pos_history,
        }, ttl_seconds=ttl)

        if span_sec < settings.ABANDON_DURATION_SEC:
            self._log_tick(camera_str_id, object_track_id, video_time, "WINDOW_NOT_FULL", t0)
            return AbandonResult(decision="WINDOW_NOT_FULL", object_track_id=object_track_id,
                                  span_sec=span_sec)

        # ── All conditions met: fire alert ────────────────────────────────────
        px_per_meter, calibration_method = self._get_calibration(camera_str_id)
        alert_id = await self._fire_alert(
            camera_str_id=camera_str_id,
            camera_db_id=camera_db_id,
            object_track_id=object_track_id,
            object_class=object_class,
            cx=cx, cy=cy,
            span_sec=span_sec,
            calibration_method=calibration_method,
            video_time=video_time,
        )
        # Mark as fired to debounce further ticks for this object.
        await state.hset_object_track(object_track_id, {"status": "alert_fired"}, ttl_seconds=ttl)
        self._log_tick(camera_str_id, object_track_id, video_time, "ABANDONED_FIRED", t0)
        return AbandonResult(decision="ABANDONED_FIRED", object_track_id=object_track_id,
                              span_sec=span_sec, alert_id=alert_id)

    async def on_person_positions(
        self,
        camera_str_id: str,
        object_track_ids_with_nearby_person: list[str],
        video_time: float,
        proximity_ttl_sec: int = 10,
    ) -> None:
        """Update proximity flags for objects that have a person within radius.

        Called once per frame by connection_manager. The caller computes
        spatial proximity (it already holds all bbox centroids); this method
        just sets/refreshes the Redis proximity flags.

        Args:
            camera_str_id: Camera string ID (for logging).
            object_track_ids_with_nearby_person: Object track IDs for which at
                least one person is currently within OWNERSHIP_RADIUS_PX.
            video_time: Current video timestamp. This is what gets STORED and
                is what on_object_update() compares against — the marker is a
                timestamp, not a boolean with an expiry.
            proximity_ttl_sec: Garbage-collection TTL for the marker key only.
                It is NOT the "how long does the object stay attended" window
                any more — that is settings.ABANDON_PROXIMITY_WINDOW_SEC, in
                video_time, read on the on_object_update() side. Keep this
                generous; it exists so the key cannot outlive the object track
                indefinitely. See the comment in on_object_update() for why
                the two were separated.
        """
        from backend.core.config import settings
        from backend.services.state import get_behavior_state
        state = get_behavior_state()

        # Floor the GC TTL well above the semantic window so the key cannot
        # vanish while it is still meaningful. Wall-clock and video_time are
        # not comparable in general, but this is a lower bound, not a rule.
        gc_ttl = max(int(proximity_ttl_sec), int(settings.ABANDON_DURATION_SEC * 3))

        for oid in object_track_ids_with_nearby_person:
            await state.set_proximity_flag(oid, ttl_seconds=gc_ttl, video_time=video_time)

    @staticmethod
    def _get_calibration(camera_str_id: str) -> tuple[float, str]:
        from backend.services.camera_calibration import get_px_per_meter
        return get_px_per_meter(camera_str_id)

    @staticmethod
    async def _fire_alert(
        camera_str_id: str,
        camera_db_id: int | None,
        object_track_id: str,
        object_class: str,
        cx: float,
        cy: float,
        span_sec: float,
        calibration_method: str,
        video_time: float,
    ) -> int | None:
        from backend.db.models import Alert
        from backend.db.session import SessionLocal
        from backend.services.audit_logger import log_audit
        from backend.services.metrics import counters
        from backend.services.sentinel_iq import IQContext, compute_alert_iq

        db = SessionLocal()
        try:
            # Day 10: the subject here is an OBJECT track, not a person track,
            # so there is no `tracks` row and no journey to trace — the
            # cross-zone multiplier records `not_applicable_no_track` and
            # no-ops. Correct and permanent for this detector.
            iq = compute_alert_iq(ALERT_TYPE, IQContext(
                db=db, camera_db_id=camera_db_id, camera_str_id=camera_str_id,
            ))

            alert = Alert(
                alert_type=ALERT_TYPE,
                camera_id=camera_str_id or str(camera_db_id or "CAM_01"),
                subject_label=(
                    f"Abandoned {object_class} — {round(span_sec)}s unattended ({camera_str_id})"
                ),
                confidence=None,
                danger_score=None,
                iq_contribution=iq.total if iq else None,
                iq_breakdown_json=iq.to_json() if iq else None,
                status="new",
                lifecycle_status="OPEN",
                meta_json=json.dumps({
                    "object_track_id": object_track_id,
                    "object_class": object_class,
                    "centroid": {"cx": round(cx, 1), "cy": round(cy, 1)},
                    "span_sec": round(span_sec, 1),
                    "calibration_method": calibration_method,
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
                db, user=None, action=ACTION_ABANDONED_FIRED,
                resource_type="alert", resource_id=alert.id,
                details={
                    "object_track_id": object_track_id,
                    "camera": camera_str_id,
                    "object_class": object_class,
                    "span_sec": round(span_sec, 1),
                    "calibration_method": calibration_method,
                    "merged_into_alert_id": alert.merged_into_alert_id,
                },
            )
            db.commit()

            counters.incr("alerts_fired_total",
                          {"type": ALERT_TYPE, "camera_id": camera_str_id})

            alert_dict = {
                "id": alert.id, "alert_type": alert.alert_type,
                "merged_into_alert_id": alert.merged_into_alert_id,
                "subject_label": alert.subject_label,
                "confidence": alert.confidence,
                "danger_score": alert.danger_score,
                "status": alert.status,
                "iq_contribution": alert.iq_contribution,
                "iq_breakdown": iq.to_dict() if iq else None,
                "lifecycle_status": alert.lifecycle_status,
                "snapshot_path": None, "evidence_hash": None,
                "false_positive_reason": None,
                "camera_name": camera_str_id, "camera_zone": None,
                "created_at": alert.created_at.isoformat() if alert.created_at else "",
                "metadata": json.loads(alert.meta_json),
            }
            await AbandonedObjectDetector._broadcast(alert_dict)

            logger.info(
                "ABANDONED_OBJECT fired: object_track_id=%s class=%s span=%.1fs (%s)",
                object_track_id, object_class, span_sec, calibration_method,
            )
            return alert.id
        except Exception:
            db.rollback()
            logger.exception("Failed to fire ABANDONED_OBJECT alert for %s", object_track_id)
            return None
        finally:
            db.close()

    @staticmethod
    async def _broadcast(alert_dict: dict[str, Any]) -> None:
        try:
            from backend.ws.dashboard_ws import broadcast
            await broadcast({"type": "new_alert", "alert": alert_dict})
        except Exception as exc:
            logger.debug("Failed to broadcast ABANDONED_OBJECT alert: %s", exc)

    @staticmethod
    def _log_tick(camera_id: str, object_track_id: str, video_time: float,
                   decision: str, t0: float) -> None:
        from backend.services.metrics import counters
        latency_ms = (time.monotonic() - t0) * 1000
        counters.observe_latency("abandoned_object", latency_ms)
        logger.debug(json.dumps({
            "camera_id": camera_id,
            "object_track_id": object_track_id,
            "detector": "abandoned_object",
            "video_time": video_time,
            "decision": decision,
            "latency_ms": round(latency_ms, 2),
        }))


_detector_instance: AbandonedObjectDetector | None = None


def get_abandoned_object_detector() -> AbandonedObjectDetector:
    """Return the singleton AbandonedObjectDetector (constructed once)."""
    global _detector_instance
    if _detector_instance is None:
        _detector_instance = AbandonedObjectDetector()
    return _detector_instance
