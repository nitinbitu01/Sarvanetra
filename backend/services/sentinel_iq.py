"""
backend/services/sentinel_iq.py — SENTINEL IQ threat scoring engine (Day 10).

WHAT THIS IS
────────────
One number per alert, and one rolling number per camera, where every digit
traces back to a stored breakdown. No client-side recomputation, no second
stored aggregate that can drift from the alerts backing it.

    per-alert    compute_alert_iq()      → written to Alert.iq_contribution
                                           ONCE, at fire time. Source of truth.
    per-camera   get_camera_iq_score()   → SUMS those stored contributions
                                           live, on read. Never recomputes a
                                           contribution, never stores a total.

The asymmetry is deliberate. A contribution depends on conditions AT THE
MOMENT THE ALERT FIRED (was it night then? had that person crossed zones by
then?) and would silently change if recomputed later — so it is frozen. An
aggregate depends on which alerts are currently active, which changes every
time an officer dismisses one — so it is never frozen.

WHAT IS DELIBERATELY NOT HERE
─────────────────────────────
`PERIMETER_BREACH` has no entry in IQ_BASE_SCORES, and that is not an
oversight to be tidied up by adding a 0.0. No detector in this system fires
it. compute_alert_iq() returns None for it, callers skip writing a
contribution, and the API surfaces it as a static roadmap label with no
number attached — because a 0.0 on a score card reads as "we checked the
perimeter and it was fine", which would be a lie. Weapon detection is not
here at all, in any form.

Three multipliers from the original design are also absent, for the same
reason: high-crime-zone weighting, school/hospital proximity, and
previously-flagged-individual history. There is no GIS layer and no incident
history in this system to compute any of them from. A hardcoded plausible
constant would be indistinguishable from a real signal on the score card,
which is exactly the failure mode this engine exists to avoid.

Usage — inside an alert-firing module, in the same DB session:

    from backend.services.sentinel_iq import IQContext, compute_alert_iq

    breakdown = compute_alert_iq(ALERT_TYPE, IQContext(
        db=db, camera_db_id=camera_db_id, camera_str_id=camera_str_id,
        pipeline_track_id=track_id,
    ))
    alert = Alert(
        ...,
        iq_contribution=breakdown.total if breakdown else None,
        iq_breakdown_json=breakdown.to_json() if breakdown else None,
    )
"""
from __future__ import annotations

import json
import logging
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from typing import Any

logger = logging.getLogger(__name__)


# ── Base scores ───────────────────────────────────────────────────────────────
# Config-driven in the sense that this is the ONE table; no firing module
# carries its own number. Values are relative severity weights, not
# probabilities and not percentages.
IQ_BASE_SCORES: dict[str, float] = {
    "WATCHLIST_FACE_MATCH": 7.0,
    "LOITERING": 3.0,
    "CROWD_ANOMALY": 4.0,
    "ABANDONED_OBJECT": 5.0,
}

# Alert types that appear on the score card as a static label and never as a
# number. Nothing in this system fires these — see the module docstring.
IQ_ROADMAP_TRIGGERS: tuple[str, ...] = ("PERIMETER_BREACH",)

ROADMAP_LABEL = "Roadmap — not yet live"

NIGHT_MULTIPLIER = 1.3
CROSS_ZONE_MULTIPLIER = 1.2


# ── Night mode override (demo control) ────────────────────────────────────────
# None = automatic (derive from the clock). True/False = forced, for demos and
# for exercising the multiplier without waiting for 20:00 local. Whichever
# path produced the answer is recorded in the breakdown as `night_source`, so
# a score computed under a manual override is never mistaken for one the
# clock produced.
_night_override: bool | None = None


def set_night_override(enabled: bool | None) -> None:
    """Force night mode on/off, or None to return to clock-based detection."""
    global _night_override
    _night_override = enabled
    logger.info(
        "SENTINEL IQ night mode override set to %s",
        "AUTO (clock)" if enabled is None else ("ON" if enabled else "OFF"),
    )


def get_night_override() -> bool | None:
    return _night_override


def _local_now(utc_dt: datetime) -> datetime:
    from backend.core.config import settings
    return utc_dt + timedelta(hours=settings.IQ_TIMEZONE_OFFSET_HOURS)


def _resolve_night(fired_at_utc: datetime) -> tuple[bool, str, dict[str, Any]]:
    """Return (is_night, night_source, detail) for the given UTC instant."""
    from backend.core.config import settings

    local = _local_now(fired_at_utc)
    start = settings.IQ_NIGHT_START_HOUR
    end = settings.IQ_NIGHT_END_HOUR
    hour = local.hour

    if start <= end:
        clock_says_night = start <= hour < end
    else:
        # Window wraps midnight — the normal case for a 20/6 default.
        clock_says_night = hour >= start or hour < end

    detail: dict[str, Any] = {
        "local_hour": hour,
        "window": f"{start:02d}:00-{end:02d}:00 local",
        "utc_offset_hours": settings.IQ_TIMEZONE_OFFSET_HOURS,
        "clock_says_night": clock_says_night,
    }

    override = get_night_override()
    if override is not None:
        return override, "manual_override", detail
    return clock_says_night, "clock", detail


# ── Data shapes ───────────────────────────────────────────────────────────────

@dataclass
class IQContext:
    """Everything compute_alert_iq() needs, gathered by the caller.

    `db` is the caller's OWN session, deliberately. The firing modules call
    this inside an open write transaction; opening a second session here
    would mean a second connection reading around a transaction that has not
    committed yet — and on SQLite, contending with its single-writer lock.

    Track identity: pass whichever the caller has. `track_db_id` (the `tracks`
    PK) is used directly; otherwise (camera_str_id, pipeline_track_id) is
    resolved to one. Detectors differ here — face-watchlist already holds a
    DB id, loitering holds the pipeline id — so the engine accepts both
    rather than forcing either caller to change what it stores.
    """
    db: Any
    camera_db_id: int | None = None
    camera_str_id: str | None = None
    track_db_id: int | None = None
    pipeline_track_id: int | None = None
    fired_at: datetime | None = None      # UTC; defaults to utcnow()


@dataclass
class IQMultiplier:
    name: str
    factor: float
    applied: bool
    reason: str
    detail: dict[str, Any] = field(default_factory=dict)


@dataclass
class IQBreakdown:
    alert_type: str
    base_score: float
    multipliers: list[IQMultiplier]
    total: float
    computed_at: str

    def to_json(self) -> str:
        return json.dumps(self.to_dict())

    def to_dict(self) -> dict[str, Any]:
        return {
            "alert_type": self.alert_type,
            "base_score": self.base_score,
            "multipliers": [
                {
                    "name": m.name,
                    "factor": m.factor,
                    "applied": m.applied,
                    "reason": m.reason,
                    "detail": m.detail,
                }
                for m in self.multipliers
            ],
            "total": self.total,
            "computed_at": self.computed_at,
        }

    @property
    def formula(self) -> str:
        """Human-readable trace, e.g. 'LOITERING 3.0 × Night 1.3 = 3.9'."""
        parts = [f"{self.alert_type} {self.base_score:g}"]
        for m in self.multipliers:
            if m.applied:
                parts.append(f"{m.name} {m.factor:g}")
        return " × ".join(parts) + f" = {self.total:g}"


@dataclass
class CameraIQSummary:
    camera_id: int
    camera_name: str | None
    camera_zone: str | None
    total_score: float
    alert_count: int
    window_minutes: int
    contributions: list[dict[str, Any]]
    unscored_count: int


# ── Multiplier: cross-zone travel ─────────────────────────────────────────────

def _resolve_track_db_id(ctx: IQContext) -> int | None:
    """Get the `tracks` PK for this alert's subject, if there is one."""
    if ctx.track_db_id is not None:
        return ctx.track_db_id
    if ctx.camera_str_id is None or ctx.pipeline_track_id is None:
        return None

    from backend.db.models import Track

    row = (
        ctx.db.query(Track)
        .filter(
            Track.camera_id == ctx.camera_str_id,
            Track.track_id == ctx.pipeline_track_id,
        )
        .first()
    )
    return row.id if row else None


def _cross_zone_multiplier(ctx: IQContext, fired_at: datetime) -> IQMultiplier:
    """×1.2 if this track's global identity was seen in 2+ distinct zones
    within IQ_CROSS_ZONE_WINDOW_MINUTES.

    Never raises and never silently always-applies: every path that cannot
    reach a verdict returns applied=False with a named reason, and those
    reasons are distinct on purpose. `not_applicable_no_track` (a crowd or
    abandoned-object alert, which has no person track to trace) is expected
    forever. `global_id_unresolved` (ReID has not yet resolved this track)
    should be RARE now that `tracks` rows exist and the ReID dispatch gate
    is reachable — if it is common in the logs, the Part A track-persistence
    fix is not actually working and this multiplier is quietly no-opping
    across the board.
    """
    from backend.core.config import settings
    from backend.db.models import Camera, Journey

    window_minutes = settings.IQ_CROSS_ZONE_WINDOW_MINUTES
    base_detail: dict[str, Any] = {"window_minutes": window_minutes}

    def _skip(reason: str, extra: dict[str, Any] | None = None) -> IQMultiplier:
        detail = {**base_detail, **(extra or {})}
        if reason == "global_id_unresolved":
            # Deliberately louder than debug: this is the signal that Part A
            # has regressed. A steady stream of these means every alert is
            # scoring at 1.0x cross-zone regardless of actual movement.
            logger.info(
                "SENTINEL IQ cross-zone: global identity unresolved for "
                "track_db_id=%s — multiplier no-op (1.0x). Expected to be rare; "
                "if frequent, check that ReID dispatch is firing.",
                detail.get("track_db_id"),
            )
        return IQMultiplier(
            name="Cross-zone travel", factor=CROSS_ZONE_MULTIPLIER,
            applied=False, reason=reason, detail=detail,
        )

    try:
        track_db_id = _resolve_track_db_id(ctx)
        if track_db_id is None:
            # Crowd anomaly and abandoned object are camera-level, not
            # person-level. There is no journey to trace, and there never
            # will be — this is not the "unresolved" case.
            return _skip("not_applicable_no_track")

        base_detail["track_db_id"] = track_db_id

        journey = (
            ctx.db.query(Journey)
            .filter(
                Journey.local_track_id == track_db_id,
                Journey.global_person_id.isnot(None),
            )
            .order_by(Journey.seen_at.desc())
            .first()
        )
        if journey is None:
            return _skip("global_id_unresolved")

        gp_id = journey.global_person_id
        base_detail["global_person_id"] = gp_id

        window_start = fired_at - timedelta(minutes=window_minutes)
        rows = (
            ctx.db.query(Camera.zone)
            .join(Journey, Journey.camera_db_id == Camera.id)
            .filter(
                Journey.global_person_id == gp_id,
                Journey.seen_at >= window_start,
                Camera.zone.isnot(None),
            )
            .distinct()
            .all()
        )
        zones = sorted({r[0] for r in rows if r[0]})
        base_detail["zones"] = zones

        if len(zones) >= 2:
            return IQMultiplier(
                name="Cross-zone travel", factor=CROSS_ZONE_MULTIPLIER,
                applied=True,
                reason=f"seen in {len(zones)} zones within {window_minutes} min",
                detail=base_detail,
            )
        return _skip("single_zone")

    except Exception as exc:
        # A scoring failure must never take down an alert that is otherwise
        # ready to fire. Degrade to 1.0x and say why.
        logger.warning("SENTINEL IQ cross-zone lookup failed: %s", exc)
        return _skip("lookup_failed", {"error": str(exc)})


# ── Public API ────────────────────────────────────────────────────────────────

def compute_alert_iq(alert_type: str, context: IQContext) -> IQBreakdown | None:
    """Score one alert at fire time.

    Returns None for a trigger with no base score (e.g. PERIMETER_BREACH).
    Callers MUST skip writing iq_contribution in that case — do not write 0.0.
    A stored 0.0 means "we scored it and it came to zero"; NULL means "not
    scored", and the score card renders the two differently on purpose.
    """
    base = IQ_BASE_SCORES.get(alert_type)
    if base is None:
        logger.debug(
            "SENTINEL IQ: no base score for alert_type=%r — not scoring it. "
            "This is expected for roadmap-only triggers.",
            alert_type,
        )
        return None

    fired_at = context.fired_at or datetime.utcnow()

    is_night, night_source, night_detail = _resolve_night(fired_at)
    night = IQMultiplier(
        name="Night", factor=NIGHT_MULTIPLIER, applied=is_night,
        reason=("night window" if is_night else "daytime"),
        detail={**night_detail, "night_source": night_source},
    )

    cross_zone = _cross_zone_multiplier(context, fired_at)

    multipliers = [night, cross_zone]
    total = base
    for m in multipliers:
        if m.applied:
            total *= m.factor

    breakdown = IQBreakdown(
        alert_type=alert_type,
        base_score=base,
        multipliers=multipliers,
        # 4dp keeps 3.0 x 1.3 x 1.2 = 4.68 exact rather than 4.680000000000001.
        total=round(total, 4),
        computed_at=fired_at.isoformat(),
    )
    logger.debug("SENTINEL IQ: %s", breakdown.formula)
    return breakdown


def _is_eligible_filter():
    """Filter clauses for 'alerts that currently count toward a camera score'.

    Dismissed and false-positive-marked alerts stop contributing immediately —
    that is what makes the aggregate drop live when an officer dismisses one.
    Acknowledged and escalated alerts still count: someone looking at it does
    not make the threat go away.
    """
    from backend.db.models import Alert

    return [
        Alert.is_deleted == False,             # noqa: E712
        Alert.lifecycle_status != "DISMISSED",
        Alert.status != "false_positive",
    ]


def get_camera_iq_score(
    camera_id: int,
    window_minutes: int,
    db: Any = None,
) -> CameraIQSummary:
    """Sum the STORED contributions of a camera's currently-active alerts.

    Never recomputes a contribution — one source of truth. An alert whose
    iq_contribution is NULL (roadmap trigger, or fired before this engine
    existed) is counted in `unscored_count` and contributes nothing, rather
    than being silently treated as 0.0.
    """
    from backend.db.models import Alert, Camera
    from backend.db.session import SessionLocal

    owns_session = db is None
    if owns_session:
        db = SessionLocal()
    try:
        window_start = datetime.utcnow() - timedelta(minutes=window_minutes)
        camera = db.query(Camera).filter(Camera.id == camera_id).first()

        rows = (
            db.query(Alert)
            .filter(
                Alert.camera_id == camera_id,
                Alert.created_at >= window_start,
                *_is_eligible_filter(),
            )
            .order_by(Alert.created_at.desc())
            .all()
        )

        contributions: list[dict[str, Any]] = []
        total = 0.0
        unscored = 0

        for r in rows:
            if r.iq_contribution is None:
                unscored += 1
                continue
            total += r.iq_contribution
            breakdown = None
            if r.iq_breakdown_json:
                try:
                    breakdown = json.loads(r.iq_breakdown_json)
                except (json.JSONDecodeError, TypeError):
                    breakdown = None
            contributions.append({
                "alert_id": r.id,
                "alert_type": r.alert_type,
                "subject_label": r.subject_label,
                "iq_contribution": round(r.iq_contribution, 4),
                "lifecycle_status": r.lifecycle_status,
                "created_at": r.created_at.isoformat() if r.created_at else "",
                "breakdown": breakdown,
            })

        return CameraIQSummary(
            camera_id=camera_id,
            camera_name=camera.name if camera else None,
            camera_zone=camera.zone if camera else None,
            total_score=round(total, 4),
            alert_count=len(contributions),
            window_minutes=window_minutes,
            contributions=contributions,
            unscored_count=unscored,
        )
    finally:
        if owns_session:
            db.close()


def get_all_camera_iq_scores(
    window_minutes: int,
    db: Any = None,
) -> list[CameraIQSummary]:
    """One summary per camera that has at least one active alert in the window.

    Cameras with no active alerts are omitted entirely rather than listed at
    0.0 — a score card full of zeroes invites reading "0" as "assessed and
    clear", which is not what it means.
    """
    from backend.db.models import Alert
    from backend.db.session import SessionLocal

    owns_session = db is None
    if owns_session:
        db = SessionLocal()
    try:
        window_start = datetime.utcnow() - timedelta(minutes=window_minutes)
        camera_ids = [
            row[0]
            for row in db.query(Alert.camera_id)
            .filter(
                Alert.camera_id.isnot(None),
                Alert.created_at >= window_start,
                *_is_eligible_filter(),
            )
            .distinct()
            .all()
        ]
        summaries = [
            get_camera_iq_score(cid, window_minutes, db=db) for cid in camera_ids
        ]
        return sorted(summaries, key=lambda s: s.total_score, reverse=True)
    finally:
        if owns_session:
            db.close()
