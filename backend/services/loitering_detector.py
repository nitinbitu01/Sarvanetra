"""
backend/services/loitering_detector.py — Loitering detection (Day 8, §1 + §2 + §3).

Core detection math (UNCHANGED from the pre-hardening version, per the Day 8
prompt's explicit instruction): a track is loitering if every position it has
reported over the last LOITER_DURATION_SEC of video_time stays within
LOITER_RADIUS_METERS (converted to pixels per-camera) of the centroid of
those same points. What changed in this pass is WHERE that rolling window
lives (Redis, not an in-process deque — survives restarts) and HOW the
radius is derived (per-camera calibration, not a flat pixel constant).

Identity key: keyed on f"{camera_id}:{track_id}" (the pipeline's own local
track id), NOT global_id. global_id is only populated once Day 6's ReID
resolves asynchronously and isn't guaranteed present on every event in this
build (see connection_manager.py's dispatch-once bookkeeping for the
`track_confirmed`/`crop_bgr`/`db_track_id` fields, which are read defensively
but not populated by the current event_emitter.py schema/main.py pipeline —
a pre-existing Day 6/7 gap, not something this pass touches). Practically,
this is also the semantically correct granularity for THIS detector anyway:
loitering asks "has this person been continuously dwelling in one spot,"
which is about continuous presence, not cross-camera re-identification. If a
track genuinely breaks (person leaves frame, tracker assigns a new ID on
return), a fresh loitering window is exactly the right behavior — the old
window's active-flag key naturally expires via TTL once no more updates
arrive, so a broken-and-resumed track does not silently continue accumulating
time from before the break, and does not double-fire.

Alert firing is debounced via a Redis-backed active flag (behavior:active:
LOITERING:{identity_key}) — while a track continues to satisfy the loitering
condition, its flag TTL is refreshed on every tick instead of firing a new
alert every frame; the flag expires naturally (allowing a fresh alert) only
once updates stop arriving for that identity_key.
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

ALERT_TYPE = "LOITERING"
ACTION_LOITERING_FIRED = "LOITERING_ALERT_FIRED"


@dataclass(frozen=True)
class LoiterResult:
    decision: str
    identity_key: str
    span_sec: float | None = None
    max_dist_px: float | None = None
    radius_px: float | None = None
    alert_id: int | None = None


def _is_night() -> bool:
    """Night window, matching the IQ scorer's definition.

    Reads IQ_NIGHT_START_HOUR / IQ_NIGHT_END_HOUR so the zone thresholds and
    the danger-score night multiplier can never disagree about what "night"
    means - two different definitions in one alert would be indefensible if a
    reviewer asked why a 21:00 event was scored as day.
    """
    from datetime import datetime, timedelta

    from backend.core.config import settings

    offset = float(getattr(settings, "IQ_TIMEZONE_OFFSET_HOURS", 5.5))
    hour = (datetime.utcnow() + timedelta(hours=offset)).hour
    start = int(getattr(settings, "IQ_NIGHT_START_HOUR", 20))
    end = int(getattr(settings, "IQ_NIGHT_END_HOUR", 6))
    return hour >= start or hour < end if start > end else start <= hour < end


def _centroid_and_max_dist(points: list[tuple[float, float]]) -> tuple[float, float, float]:
    """Return (centroid_x, centroid_y, max_distance_from_centroid) for a list
    of (x, y) points. Pure function — the actual "detection math" the Day 8
    prompt says not to change."""
    cx = sum(p[0] for p in points) / len(points)
    cy = sum(p[1] for p in points) / len(points)
    max_dist = max(math.hypot(px - cx, py - cy) for px, py in points)
    return cx, cy, max_dist


class LoiteringDetector:
    async def on_track_position(
        self,
        camera_str_id: str,
        camera_db_id: int | None,
        track_id: int,
        cx: float,
        cy: float,
        video_time: float,
    ) -> LoiterResult:
        """Feed one position sample for a track. Called on every detection
        event for that track (see connection_manager.py's Day 8 hook) — this
        is the continuous per-frame update the loitering window needs, unlike
        ReID/face-watchlist which fire once at confirmation.
        """
        from backend.core.config import settings
        from backend.services.state import get_behavior_state

        t0 = time.monotonic()
        identity_key = f"{camera_str_id}:{track_id}"
        state = get_behavior_state()

        key = f"loiter:{identity_key}"
        # Member MUST be unique per sample, not just "cx,cy" — a stationary
        # (loitering) person reports the same or near-identical pixel
        # position repeatedly, which is exactly the case this detector needs
        # to accumulate. A bare "cx,cy" member (as a first pass at this
        # literally read) collides on every repeat: ZADD with an existing
        # member overwrites its score instead of adding a new point, so the
        # window's earliest score never gets older than the last few ticks
        # and the span never reaches LOITER_DURATION_SEC — verified this
        # actually happens (not just a theoretical risk) before fixing it.
        # Prefixing with video_time guarantees a unique member per call.
        member = f"{video_time}:{cx:.1f},{cy:.1f}"
        ttl = int(settings.LOITER_DURATION_SEC * 2)

        ok = await state.zadd_point(key, video_time, member, ttl)
        if not ok:
            self._log_tick(camera_str_id, identity_key, video_time, "SKIPPED_NO_REDIS", t0)
            return LoiterResult(decision="SKIPPED_NO_REDIS", identity_key=identity_key)

        # Exclusive upper bound ("(" prefix, Redis ZREMRANGEBYSCORE syntax) —
        # drop points strictly OLDER than the window, keep a point exactly AT
        # window_start. An inclusive bound here would evict the very point
        # that makes the window "full" right at the threshold instant.
        window_start = video_time - settings.LOITER_DURATION_SEC
        await state.zremrangebyscore(key, "-inf", f"({window_start}")

        points_raw = await state.zrange_withscores(key)
        if not points_raw:
            self._log_tick(camera_str_id, identity_key, video_time, "NO_POINTS", t0)
            return LoiterResult(decision="NO_POINTS", identity_key=identity_key)

        earliest_score = points_raw[0][1]
        span = video_time - earliest_score
        if span < settings.LOITER_DURATION_SEC:
            self._log_tick(camera_str_id, identity_key, video_time, "WINDOW_NOT_FULL", t0)
            return LoiterResult(decision="WINDOW_NOT_FULL", identity_key=identity_key, span_sec=span)

        try:
            points = [
                tuple(map(float, member.split(":", 1)[1].split(",")))
                for member, _ in points_raw
            ]
        except (ValueError, IndexError):
            logger.error("Malformed loiter point in key=%s — skipping tick.", key)
            return LoiterResult(decision="ERROR", identity_key=identity_key)

        _, _, max_dist = _centroid_and_max_dist(points)

        px_per_meter, calibration_method = self._get_calibration(camera_str_id)
        radius_px = settings.LOITER_RADIUS_METERS * px_per_meter

        if max_dist > radius_px:
            self._log_tick(camera_str_id, identity_key, video_time, "MOVING_TOO_MUCH", t0)
            return LoiterResult(decision="MOVING_TOO_MUCH", identity_key=identity_key,
                                 max_dist_px=max_dist, radius_px=radius_px)

        # ── Zone gate ───────────────────────────────────────────────────────
        # Staying still is not suspicious by itself. WHERE it happens, and for
        # how long relative to that place, is what decides. A 60s dwell fired
        # on a woman waiting at a busy junction at 7am - correct by the rule,
        # useless to an operator, and a stream of those is how a control room
        # learns to ignore the alert feed entirely.
        #
        # Zones are per camera (config/zones/<CAM>.json, drawn on a real frame)
        # and carry their own day/night thresholds:
        #   exempt     bus stop, auto stand, shop front — never alerts
        #   normal     footpath, open road — 10 min day / 5 min night
        #   sensitive  parked vehicles, ATM — 3 min day / 2 min night
        #
        # A camera with no zone file falls back to the `normal` defaults and
        # logs a warning: failing open keeps detection alive, whereas silently
        # skipping it would hide a missing config as if it were quiet footage.
        try:
            from backend.services.zone_engine import get_zone_engine
            verdict = get_zone_engine().evaluate(
                camera_str_id, cx, cy, dwell_sec=span, is_night=_is_night())
            if not verdict.should_alert:
                self._log_tick(camera_str_id, identity_key, video_time,
                               f"ZONE_SUPPRESSED:{verdict.zone_type}", t0)
                return LoiterResult(decision="ZONE_SUPPRESSED",
                                    identity_key=identity_key,
                                    max_dist_px=max_dist, radius_px=radius_px,
                                    span_sec=span)
        except Exception as exc:                              # noqa: BLE001
            # A broken zone config must not disable loitering detection - fall
            # through to the un-gated behaviour rather than dropping the alert.
            logger.warning("Zone engine unavailable (%s) — alerting without "
                           "zone gating.", exc)

        # ── Path shape ──────────────────────────────────────────────────────
        # The window's own points describe HOW the person stayed, not just how
        # long. Someone waiting for a lift stands still; someone casing a spot
        # paces, circles, and returns. Both hold position for the same
        # duration, so duration alone cannot separate them - but the path can.
        traj = None
        traj_bonus, traj_reasons = 0.0, []
        try:
            from backend.services import trajectory_analyzer as _traj
            traj = _traj.analyse(points)
            traj_bonus, traj_reasons = _traj.suspicion_bonus(traj)
        except Exception as exc:                              # noqa: BLE001
            logger.debug("Trajectory analysis unavailable: %s", exc)

        # ── Per-camera alert budget ─────────────────────────────────────────
        # The per-person debounce above stops ONE person re-alerting. It cannot
        # stop fifteen different people crossing the threshold on a busy
        # junction inside ten minutes - fifteen individually valid alerts that
        # together destroy the operator's attention. Suppressions are counted,
        # so a noisy camera shows up in the stats rather than disappearing.
        try:
            from backend.services.alert_cooldown import get_cooldown
            allowed, cd_reason = get_cooldown().allow(
                camera_str_id, ALERT_TYPE, force=traj_bonus >= 4.0)
            if not allowed:
                self._log_tick(camera_str_id, identity_key, video_time,
                               "COOLDOWN_SUPPRESSED", t0)
                logger.info("[loiter] %s suppressed — %s",
                            identity_key, cd_reason)
                return LoiterResult(decision="COOLDOWN_SUPPRESSED",
                                    identity_key=identity_key,
                                    max_dist_px=max_dist, radius_px=radius_px,
                                    span_sec=span)
        except Exception as exc:                              # noqa: BLE001
            logger.debug("Cooldown unavailable: %s", exc)

        # ── Condition met: within radius for the full window ────────────────
        active_key = f"behavior:active:LOITERING:{identity_key}"
        active = await state.get_flag(active_key)
        now_iso = datetime.utcnow().isoformat()

        if active:
            # Already flagged — refresh TTL, don't re-fire. This is the
            # debounce: one alert per continuous loitering episode.
            await state.set_flag(active_key, {**active, "last_seen_at": now_iso}, ttl_seconds=ttl)
            self._log_tick(camera_str_id, identity_key, video_time, "ALREADY_ACTIVE", t0)
            return LoiterResult(decision="ALREADY_ACTIVE", identity_key=identity_key,
                                 max_dist_px=max_dist, radius_px=radius_px)

        await state.set_flag(active_key, {"started_at": now_iso, "last_fired_at": now_iso}, ttl_seconds=ttl)

        alert_id = await self._fire_alert(
            camera_str_id=camera_str_id, camera_db_id=camera_db_id, track_id=track_id,
            identity_key=identity_key, span_sec=span, max_dist_px=max_dist, radius_px=radius_px,
            calibration_method=calibration_method, video_time=video_time,
            cx=cx, cy=cy,
            traj=traj.to_dict() if traj is not None else None,
            traj_bonus=traj_bonus, traj_reasons=traj_reasons,
        )
        self._log_tick(camera_str_id, identity_key, video_time, "LOITERING_FIRED", t0)
        return LoiterResult(decision="LOITERING_FIRED", identity_key=identity_key,
                             max_dist_px=max_dist, radius_px=radius_px, alert_id=alert_id)

    @staticmethod
    def _get_calibration(camera_str_id: str) -> tuple[float, str]:
        from backend.services.camera_calibration import get_px_per_meter
        return get_px_per_meter(camera_str_id)

    @staticmethod
    async def _fire_alert(
        camera_str_id: str, camera_db_id: int | None, track_id: int, identity_key: str,
        span_sec: float, max_dist_px: float, radius_px: float, calibration_method: str,
        video_time: float, cx: float | None = None, cy: float | None = None,
        traj: dict | None = None, traj_bonus: float = 0.0,
        traj_reasons: list[str] | None = None,
    ) -> int | None:
        from backend.db.models import Alert
        from backend.db.session import SessionLocal
        from backend.services.audit_logger import log_audit
        from backend.services.metrics import counters
        from backend.services.sentinel_iq import IQContext, compute_alert_iq

        db = SessionLocal()
        try:
            # Day 10: score once, here, at fire time — in this session, so the
            # cross-zone journey lookup sees the same transaction. `track_id`
            # here is the PIPELINE track id, not the tracks PK, so it is
            # passed as pipeline_track_id + camera_str_id for the engine to
            # resolve (see IQContext's docstring).
            iq = compute_alert_iq(ALERT_TYPE, IQContext(
                db=db, camera_db_id=camera_db_id, camera_str_id=camera_str_id,
                pipeline_track_id=track_id,
            ))

            alert = Alert(
                alert_type=ALERT_TYPE,
                camera_id=camera_str_id or str(camera_db_id or "CAM_01"),
                track_id=track_id,
                subject_label=f"Loitering — {round(span_sec)}s in place ({camera_str_id})",
                confidence=None,
                danger_score=None,  # danger scoring system not built yet — same placeholder
                                     # posture as Day 7's WATCHLIST_MATCH_DANGER_SCORE comment
                iq_contribution=iq.total if iq else None,
                iq_breakdown_json=iq.to_json() if iq else None,
                status="new",
                lifecycle_status="OPEN",
                meta_json=json.dumps({
                    "identity_key": identity_key,
                    "span_sec": round(span_sec, 1),
                    "max_dist_px": round(max_dist_px, 1),
                    "radius_px": round(radius_px, 1),
                    "calibration_method": calibration_method,
                    "video_time": video_time,
                    # WHERE the subject was, in frame pixels. Without this an
                    # alert can only say "someone loitered on this camera" -
                    # the evidence frame has to box every person present and
                    # leave a reviewer guessing which one it meant. An officer
                    # acting on the alert needs to be pointed at a person, and
                    # a judge shown the system needs the same.
                    "cx": round(cx, 1) if cx is not None else None,
                    "cy": round(cy, 1) if cy is not None else None,
                    # Path-shape measurements and the score they contributed.
                    # Kept raw so a reviewer can judge whether the signal was
                    # real - an alert whose reasoning cannot be inspected is
                    # one an officer learns to distrust.
                    "trajectory": traj,
                    "trajectory_bonus": traj_bonus,
                    "trajectory_reasons": traj_reasons or [],
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
                db, user=None, action=ACTION_LOITERING_FIRED,
                resource_type="alert", resource_id=alert.id,
                details={
                    "identity_key": identity_key, "camera": camera_str_id,
                    "span_sec": round(span_sec, 1), "calibration_method": calibration_method,
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
            await LoiteringDetector._broadcast(alert_dict)

            logger.info("LOITERING fired: %s span=%.1fs max_dist=%.1fpx radius=%.1fpx (%s)",
                        identity_key, span_sec, max_dist_px, radius_px, calibration_method)
            return alert.id
        except Exception:
            db.rollback()
            logger.exception("Failed to fire LOITERING alert for %s", identity_key)
            return None
        finally:
            db.close()

    @staticmethod
    async def _broadcast(alert_dict: dict[str, Any]) -> None:
        try:
            from backend.ws.dashboard_ws import broadcast
            await broadcast({"type": "new_alert", "alert": alert_dict})
        except Exception as exc:
            logger.debug("Failed to broadcast LOITERING alert: %s", exc)

    @staticmethod
    def _log_tick(camera_id: str, identity_key: str, video_time: float, decision: str, t0: float) -> None:
        from backend.services.metrics import counters
        latency_ms = (time.monotonic() - t0) * 1000
        counters.observe_latency("loitering", latency_ms)
        logger.debug(json.dumps({
            "camera_id": camera_id, "global_id": identity_key, "detector": "loitering",
            "video_time": video_time, "decision": decision, "latency_ms": round(latency_ms, 2),
        }))


_detector_instance: LoiteringDetector | None = None


def get_loitering_detector() -> LoiteringDetector:
    global _detector_instance
    if _detector_instance is None:
        _detector_instance = LoiteringDetector()
    return _detector_instance
