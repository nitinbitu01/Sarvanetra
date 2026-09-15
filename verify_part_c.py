"""
verify_part_c.py — Acceptance checks for the SENTINEL IQ scoring engine.

Exercises the real service code (sentinel_iq, loitering_detector,
crowd_detector, the alerts router's lifecycle transition, dashboard_ws)
against a scratch SQLite DB.

SUBSTITUTION, STATED UP FRONT
─────────────────────────────
Redis is not running in this environment, and the Day 8 behavior engine's
state layer requires it. Rather than skip the loitering checkpoint, this
script installs an in-memory BehaviorState that implements the same
interface — which backend/services/state.py explicitly anticipates
("the backend is swappable (e.g. for a test fake...)"). The loitering
detection math, the alert-firing path, and the IQ hook inside it are all
real production code; only the key-value store underneath is substituted.

Run:  python verify_part_c.py
"""
from __future__ import annotations

import asyncio
import json
import os
import sys
import tempfile
from datetime import datetime, timedelta
from pathlib import Path

_ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(_ROOT))

_tmpdir = tempfile.mkdtemp(prefix="verify_part_c_")
os.environ["DATABASE_URL"] = f"sqlite:///{_tmpdir}/verify_part_c.db"

PASS: list[str] = []
FAIL: list[str] = []


def check(name: str, cond: bool, detail: str = "") -> None:
    (PASS if cond else FAIL).append(name)
    print(f"{'PASS' if cond else 'FAIL'} — {name}{(': ' + detail) if detail else ''}")


# ── In-memory stand-in for the Redis-backed BehaviorState ────────────────────

class InMemoryBehaviorState:
    """Same interface as BehaviorState, backed by dicts. TTLs are ignored —
    nothing in this script runs long enough for expiry to matter."""

    def __init__(self) -> None:
        self._z: dict[str, dict[str, float]] = {}
        self._kv: dict[str, str] = {}

    async def ping(self) -> bool:
        return True

    async def zadd_point(self, key, score, member, ttl_seconds) -> bool:
        self._z.setdefault(key, {})[member] = score
        return True

    async def zremrangebyscore(self, key, min_score, max_score) -> bool:
        if key not in self._z:
            return True
        exclusive = isinstance(max_score, str) and max_score.startswith("(")
        hi = float(str(max_score).lstrip("("))
        self._z[key] = {
            m: s for m, s in self._z[key].items()
            if (s >= hi if exclusive else s > hi)
        }
        return True

    async def zrange_withscores(self, key):
        return sorted(self._z.get(key, {}).items(), key=lambda kv: kv[1])

    async def set_simple(self, key, value, ttl_seconds) -> bool:
        self._kv[key] = value
        return True

    async def get_simple(self, key):
        return self._kv.get(key)

    async def exists(self, key) -> bool:
        return key in self._kv

    async def delete(self, key) -> bool:
        self._kv.pop(key, None)
        return True

    async def get_flag(self, key):
        raw = self._kv.get(key)
        return json.loads(raw) if raw else None

    async def set_flag(self, key, value, ttl_seconds) -> bool:
        self._kv[key] = json.dumps(value)
        return True


class _FakeClient:
    def __init__(self):
        self.received = []

    async def send_json(self, message):
        self.received.append(message)


class _FakeRequest:
    class _Addr:
        host = "127.0.0.1"
    client = _Addr()


async def main() -> None:
    from backend.db.models import (
        Base, Alert, Camera, GlobalPerson, Journey, Track, User,
    )
    from backend.db.session import SessionLocal, engine
    from backend.services import state as state_module
    from backend.services.sentinel_iq import (
        IQ_BASE_SCORES, IQ_ROADMAP_TRIGGERS, IQContext, compute_alert_iq,
        get_all_camera_iq_scores, get_camera_iq_score, set_night_override,
    )

    Base.metadata.create_all(bind=engine)
    db = SessionLocal()

    state_module._state_instance = InMemoryBehaviorState()

    # ── Fixtures: two cameras in two different zones ────────────────────────
    cam_central = Camera(camera_id="CAM-IQ-01", name="MG Road Junction",
                         zone="Central", status="ONLINE")
    cam_east = Camera(camera_id="CAM-IQ-02", name="Railway Station Gate",
                      zone="East", status="ONLINE")
    db.add_all([cam_central, cam_east])
    db.commit()
    db.refresh(cam_central)
    db.refresh(cam_east)

    # ── 1. Base score table matches the specified values ───────────────────
    check("Base score table is exactly the four live triggers",
          IQ_BASE_SCORES == {
              "WATCHLIST_FACE_MATCH": 7.0, "LOITERING": 3.0,
              "CROWD_ANOMALY": 4.0, "ABANDONED_OBJECT": 5.0,
          }, str(IQ_BASE_SCORES))
    check("ABANDONED_OBJECT base score key matches Day 9's real alert_type string",
          "ABANDONED_OBJECT" in IQ_BASE_SCORES)

    # ── 2. PERIMETER_BREACH is unscored — None, never 0.0 ──────────────────
    pb = compute_alert_iq("PERIMETER_BREACH", IQContext(db=db))
    check("compute_alert_iq('PERIMETER_BREACH') returns None, not a 0.0 score",
          pb is None, repr(pb))
    check("PERIMETER_BREACH is declared as a roadmap trigger",
          "PERIMETER_BREACH" in IQ_ROADMAP_TRIGGERS)

    # ── 3. Loitering + night override = 3.0 × 1.3 = 3.9 ────────────────────
    from backend.services.loitering_detector import get_loitering_detector

    # A `tracks` row exists for this pipeline track, exactly as Part A's
    # persistence step creates one on first sighting. What it does NOT have
    # yet is a Journey — i.e. ReID has not resolved it to a global identity.
    # That is the case the cross-zone multiplier must report as
    # `global_id_unresolved`, distinct from "no track exists at all".
    db.add(Track(camera_id="CAM-IQ-01", track_id=9001,
                 first_seen_frame=1, last_seen_frame=60, best_confidence=0.88))
    db.commit()

    set_night_override(True)
    loiter = get_loitering_detector()
    r = None
    for t in (0, 20, 40, 60):
        r = await loiter.on_track_position(
            camera_str_id="CAM-IQ-01", camera_db_id=cam_central.id,
            track_id=9001, cx=300, cy=300, video_time=float(t),
        )
    check("Loitering alert fired", r.decision == "LOITERING_FIRED", r.decision)

    loiter_alert = (
        db.query(Alert).filter(Alert.alert_type == "LOITERING")
        .order_by(Alert.id.desc()).first()
    )
    check("Loitering alert has iq_contribution = 3.0 × 1.3 = 3.9",
          loiter_alert is not None and abs(loiter_alert.iq_contribution - 3.9) < 1e-9,
          str(loiter_alert.iq_contribution if loiter_alert else None))

    bd = json.loads(loiter_alert.iq_breakdown_json)
    night_mult = next(m for m in bd["multipliers"] if m["name"] == "Night")
    check("Breakdown records night_source = 'manual_override'",
          night_mult["detail"]["night_source"] == "manual_override",
          str(night_mult["detail"]))
    check("Breakdown records the night multiplier as applied", night_mult["applied"] is True)
    check("Breakdown stores base_score 3.0 traceably", bd["base_score"] == 3.0, str(bd["base_score"]))
    check("Stored total equals the stored parts multiplied out (no drift)",
          abs(bd["total"] - bd["base_score"] * 1.3) < 1e-9, str(bd["total"]))

    # ── 4. Cross-zone no-ops with a NAMED reason when nothing to resolve ────
    cz = next(m for m in bd["multipliers"] if m["name"] == "Cross-zone travel")
    check("Cross-zone multiplier no-opped (1.0×), not silently always-applied",
          cz["applied"] is False, str(cz))
    check("Cross-zone no-op carries a logged reason",
          bool(cz["reason"]), str(cz["reason"]))
    check("A loitering track with no ReID resolution reports 'global_id_unresolved' "
          "— distinct from the 'no track at all' case",
          cz["reason"] == "global_id_unresolved", cz["reason"])

    # ── 5. Cross-zone DOES apply for a genuinely cross-zone identity ────────
    set_night_override(False)   # isolate the cross-zone factor

    gp = GlobalPerson(
        representative_embedding=b"\x00" * 4, first_seen_at=datetime.utcnow(),
        last_seen_at=datetime.utcnow(), total_sightings=2,
        retention_expires_at=datetime.utcnow() + timedelta(days=90),
        retention_hold=False, legal_basis="verify", is_deleted=False,
    )
    db.add(gp)
    db.flush()

    track_row = Track(camera_id="CAM-IQ-01", track_id=9002,
                      first_seen_frame=1, last_seen_frame=10, best_confidence=0.9)
    db.add(track_row)
    db.flush()

    now = datetime.utcnow()
    db.add_all([
        Journey(global_person_id=gp.id, camera_db_id=cam_central.id,
                local_track_id=track_row.id, camera_id="CAM-IQ-01",
                seen_at=now - timedelta(minutes=10), confidence=0.9),
        Journey(global_person_id=gp.id, camera_db_id=cam_east.id,
                camera_id="CAM-IQ-02", seen_at=now - timedelta(minutes=5), confidence=0.9),
    ])
    db.commit()

    iq_cross = compute_alert_iq("LOITERING", IQContext(
        db=db, camera_db_id=cam_central.id, camera_str_id="CAM-IQ-01",
        pipeline_track_id=9002,
    ))
    cz2 = next(m for m in iq_cross.multipliers if m.name == "Cross-zone travel")
    check("Cross-zone multiplier APPLIES for an identity seen in 2 zones in-window",
          cz2.applied is True, f"{cz2.reason} zones={cz2.detail.get('zones')}")
    check("Cross-zone-only score = 3.0 × 1.2 = 3.6",
          abs(iq_cross.total - 3.6) < 1e-9, str(iq_cross.total))
    check("Cross-zone resolution went through the tracks row created by Part A "
          "(Journey.local_track_id FK) — proving the two parts connect",
          cz2.detail.get("track_db_id") == track_row.id, str(cz2.detail))

    # ── 6. Both multipliers stack ───────────────────────────────────────────
    set_night_override(True)
    iq_both = compute_alert_iq("LOITERING", IQContext(
        db=db, camera_db_id=cam_central.id, camera_str_id="CAM-IQ-01",
        pipeline_track_id=9002,
    ))
    check("Night + cross-zone stack: 3.0 × 1.3 × 1.2 = 4.68",
          abs(iq_both.total - 4.68) < 1e-9, str(iq_both.total))
    check("formula trace reads back the whole computation",
          "3 × Night 1.3 × Cross-zone travel 1.2 = 4.68" in iq_both.formula,
          iq_both.formula)
    # Pin OFF rather than releasing to the ambient clock. Releasing here made
    # the crowd assertion below time-dependent: the night window is
    # 20:00-06:00 IST, so an evening run scored the crowd alert 5.2 (4.0 x
    # 1.3) instead of 4.0 and reported a failure against a correct engine.
    # Night behaviour is asserted explicitly above, with the override ON.
    set_night_override(False)

    # ── 7. Crowd anomaly: camera-level, so cross-zone is 'not applicable' ───
    from backend.services.crowd_detector import get_crowd_detector

    crowd = get_crowd_detector()
    decisions = []
    for t, cnt in [(0, 2), (10, 2), (20, 2), (30, 9), (45, 9), (60, 9), (75, 9)]:
        cr = await crowd.on_frame_tick(
            camera_str_id="CAM-IQ-01", camera_db_id=cam_central.id,
            video_time=float(t), current_count=cnt,
        )
        decisions.append(cr.decision)
    # Check that a fire happened somewhere in the sequence, not that the LAST
    # tick fired — the detector goes into cooldown immediately after firing,
    # so the final tick is expected to report NO_SURGE.
    check("Crowd anomaly alert fired during the surge sequence",
          "CROWD_ANOMALY_FIRED" in decisions, " → ".join(decisions))

    crowd_alert = (
        db.query(Alert).filter(Alert.alert_type == "CROWD_ANOMALY")
        .order_by(Alert.id.desc()).first()
    )
    check("Crowd alert scored with base 4.0", crowd_alert is not None
          and abs(crowd_alert.iq_contribution - 4.0) < 1e-9,
          str(crowd_alert.iq_contribution if crowd_alert else None))
    crowd_bd = json.loads(crowd_alert.iq_breakdown_json)
    crowd_cz = next(m for m in crowd_bd["multipliers"] if m["name"] == "Cross-zone travel")
    check("Crowd alert's cross-zone reason is 'not_applicable_no_track', NOT "
          "'global_id_unresolved' — the two are distinguishable in the logs",
          crowd_cz["reason"] == "not_applicable_no_track", crowd_cz["reason"])

    # ── 8. Aggregate sums the stored contributions ──────────────────────────
    summary = get_camera_iq_score(cam_central.id, window_minutes=60, db=db)
    expected = round(loiter_alert.iq_contribution + crowd_alert.iq_contribution, 4)
    check("Camera aggregate = sum of its two alerts' stored contributions (3.9 + 4.0 = 7.9)",
          abs(summary.total_score - expected) < 1e-9,
          f"{summary.total_score} vs {expected}")
    check("Aggregate reports both contributing alerts", summary.alert_count == 2,
          str(summary.alert_count))

    # ── 9. Dismissing an alert drops it from the aggregate, live ────────────
    from backend.routers.v1.alerts import _apply_lifecycle_transition
    import backend.ws.dashboard_ws as dashboard_ws

    user = User(username="verify_iq_admin", hashed_password="x", role="admin", is_active=True)
    db.add(user)
    db.commit()
    db.refresh(user)

    tab1, tab2 = _FakeClient(), _FakeClient()
    dashboard_ws._clients.add(tab1)
    dashboard_ws._clients.add(tab2)
    try:
        await _apply_lifecycle_transition(
            crowd_alert.id, "DISMISSED", "ALERT_DISMISSED", db, user,
            _FakeRequest(), feedback="FALSE_POSITIVE",
        )
        db.commit()

        after = get_camera_iq_score(cam_central.id, window_minutes=60, db=db)
        check("Dismissing the crowd alert drops it out of the aggregate (7.9 → 3.9)",
              abs(after.total_score - loiter_alert.iq_contribution) < 1e-9,
              str(after.total_score))
        check("Dismissed alert no longer counted", after.alert_count == 1, str(after.alert_count))
        check("A second connected tab was told about the dismissal, so its score "
              "card refetches without a refresh",
              any(m.get("type") == "alert_status_update" and m.get("alert_id") == crowd_alert.id
                  for m in tab2.received))
    finally:
        dashboard_ws._clients.discard(tab1)
        dashboard_ws._clients.discard(tab2)

    # ── 10. Stored contribution is FROZEN — dismissal never rewrites it ────
    db.refresh(crowd_alert)
    check("Dismissed alert KEEPS its stored iq_contribution (frozen at fire "
          "time; the aggregate excludes it, it is not zeroed out)",
          abs(crowd_alert.iq_contribution - 4.0) < 1e-9,
          str(crowd_alert.iq_contribution))

    # ── 11. Night override AFTER firing does not rescore existing alerts ───
    set_night_override(True)
    db.refresh(loiter_alert)
    check("Flipping night mode does not retroactively rescore an already-fired alert",
          abs(loiter_alert.iq_contribution - 3.9) < 1e-9, str(loiter_alert.iq_contribution))
    set_night_override(None)

    # ── 12. API payload carries no numeric field for roadmap triggers ──────
    from backend.routers.v1.sentinel_iq import _roadmap_payload

    payload = _roadmap_payload()
    serialized = json.dumps(payload)
    check("Roadmap payload is label-only — no 'score', 'value', or numeric field",
          all(set(p.keys()) == {"alert_type", "status"} for p in payload), serialized)
    check("No 0.0 (or any number) appears anywhere in the roadmap payload",
          not any(isinstance(v, (int, float)) for p in payload for v in p.values()),
          serialized)
    check("PERIMETER_BREACH renders as the static roadmap label",
          any(p["alert_type"] == "PERIMETER_BREACH"
              and p["status"] == "Roadmap — not yet live" for p in payload),
          serialized)

    # ── 13. Cameras with nothing to score are omitted, not listed at 0 ─────
    all_scores = get_all_camera_iq_scores(window_minutes=60, db=db)
    listed_ids = {s.camera_id for s in all_scores}
    check("Camera with no active alerts is absent from the score card "
          "(not shown at 0.0, which would read as 'assessed and clear')",
          cam_east.id not in listed_ids, str(sorted(listed_ids)))

    db.close()

    print(f"\n{'=' * 68}\n{len(PASS)} passed, {len(FAIL)} failed\n{'=' * 68}")
    if FAIL:
        print("FAILED:", FAIL)
        sys.exit(1)


if __name__ == "__main__":
    asyncio.run(main())
