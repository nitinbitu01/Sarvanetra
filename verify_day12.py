"""
verify_day12.py — Day 12 (Human Review Queue) acceptance checks.

Covers the checkpoints that a script CAN settle. Checkpoint 2's frontend half
(ReviewQueue.jsx auto-populating off the reid_review_created WS broadcast)
needs a browser and is explicitly NOT claimed here — what IS covered is that
the broadcast is emitted, which is the half a script can prove.

Exercises the real HTTP surface via Starlette's TestClient (real routing, real
auth dependencies, real role normalisation), because the central risk in this
day's work is an AUTHORIZATION mismatch, and that only shows up through the
dependency stack — not by calling service functions directly.

Run:  python verify_day12.py
"""
from __future__ import annotations

import json
import os
import sys
import tempfile
from datetime import datetime, timedelta
from pathlib import Path

_ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(_ROOT))

_tmpdir = tempfile.mkdtemp(prefix="verify_day12_")
os.environ["DATABASE_URL"] = f"sqlite:///{_tmpdir}/verify_day12.db"
os.environ.pop("SENTINEL_SOURCE", None)

PASS: list[str] = []
FAIL: list[str] = []

ADMIN_PW = "day12-admin-pw"
OFFICER_PW = "day12-officer-pw"


def check(name: str, cond: bool, detail: str = "") -> None:
    (PASS if cond else FAIL).append(name)
    print(f"{'PASS' if cond else 'FAIL'} — {name}{(': ' + detail) if detail else ''}")


def seed():
    """Two users, two cameras in different zones, and two review items:
    one candidate WITH a multi-camera journey, one with none."""
    from backend.auth.routes import pwd_context
    from backend.db.models import (
        Base, Camera, GlobalPerson, Journey, ReIDCalibrationLog,
        ReIDReviewItem, Track, User,
    )
    from backend.db.session import SessionLocal, engine
    from backend.services.sentinel_iq import set_night_override

    # Nothing here asserts an IQ score, but pin the clock anyway: approve/
    # reject paths touch alert-adjacent code, and a test suite that changes
    # its answer depending on what time of day it runs is a trap. (Several
    # earlier scripts in this repo had exactly that bug — they passed all
    # day and failed after 20:00 IST when the night multiplier kicked in.)
    set_night_override(False)

    Base.metadata.create_all(bind=engine)
    db = SessionLocal()
    try:
        db.add_all([
            User(username="d12_admin", hashed_password=pwd_context.hash(ADMIN_PW),
                 role="admin", is_active=True),
            User(username="d12_officer", hashed_password=pwd_context.hash(OFFICER_PW),
                 role="officer", is_active=True),
        ])
        cam_a = Camera(camera_id="CAM-D12-A", name="MG Road", zone="Central", status="ONLINE")
        cam_b = Camera(camera_id="CAM-D12-B", name="Station Gate", zone="East", status="ONLINE")
        db.add_all([cam_a, cam_b])
        db.commit()
        db.refresh(cam_a)
        db.refresh(cam_b)

        now = datetime.utcnow()

        def _mk_gp(sightings):
            gp = GlobalPerson(
                representative_embedding=b"\x00" * 4,
                first_seen_at=now - timedelta(hours=2), last_seen_at=now,
                total_sightings=sightings,
                retention_expires_at=now + timedelta(days=90),
                retention_hold=False, legal_basis="verify", is_deleted=False,
            )
            db.add(gp)
            db.flush()
            return gp

        # Candidate WITH a journey across two cameras/zones.
        gp_with = _mk_gp(2)
        db.add_all([
            Journey(global_person_id=gp_with.id, camera_db_id=cam_a.id,
                    camera_id="CAM-D12-A", seen_at=now - timedelta(minutes=40),
                    confidence=0.91),
            Journey(global_person_id=gp_with.id, camera_db_id=cam_b.id,
                    camera_id="CAM-D12-B", seen_at=now - timedelta(minutes=10),
                    confidence=0.72,
                    confidence_caveat="Long gap since last sighting (7h). "
                                       "Appearance-based match only."),
        ])

        # Candidate with NO journey — the "first-time candidate" case.
        gp_without = _mk_gp(1)

        track = Track(camera_id="CAM-D12-A", track_id=9001,
                      first_seen_frame=1, last_seen_frame=40,
                      best_confidence=0.88,
                      best_crop_path="output/crops/track_9001/frame_1.jpg")
        db.add(track)
        db.flush()

        cal = ReIDCalibrationLog(similarity_score=0.78, decision="REVIEW_PENDING",
                                  time_gap_seconds=120)
        db.add(cal)
        db.flush()

        item_with = ReIDReviewItem(
            local_track_id=track.id, candidate_global_person_id=gp_with.id,
            similarity_score=0.78, crop_image_path=track.best_crop_path,
            status="PENDING", time_gap_seconds=120, calibration_log_id=cal.id,
        )
        item_without = ReIDReviewItem(
            local_track_id=track.id, candidate_global_person_id=gp_without.id,
            similarity_score=0.71, crop_image_path=track.best_crop_path,
            status="PENDING", time_gap_seconds=90, calibration_log_id=None,
        )
        db.add_all([item_with, item_without])
        db.commit()
        return {
            "item_with": item_with.id, "item_without": item_without.id,
            "gp_with": gp_with.id, "gp_without": gp_without.id,
            "track_id": track.id, "cal_id": cal.id,
        }
    finally:
        db.close()


class _FakeWSClient:
    """Stand-in for a dashboard WebSocket, matching what broadcast() calls."""

    def __init__(self):
        self.received: list[dict] = []

    async def send_json(self, message):
        self.received.append(message)


async def _check_review_item_created_by_pipeline() -> None:
    """Checkpoint 2 (backend half): a real in-band similarity resolved through
    resolve_identity() must create the ReIDReviewItem ITSELF and broadcast
    reid_review_created — not be inserted by test code.

    The 0.78 pair is CONSTRUCTED, not searched for: the stub ReID embedder's
    hash-derived vectors are near-orthogonal to any random vector, so nothing
    lands in the 0.65-0.85 review band by chance. Building
        base = t*v + sqrt(1-t^2)*u   (u ⟂ v, both unit)
    gives cos(v, base) = t exactly, so the band is exercised deterministically
    instead of depending on a lucky seed.
    """
    import numpy as np

    import backend.ws.dashboard_ws as dashboard_ws
    from backend.db.models import Camera, GlobalPerson, ReIDReviewItem, Track
    from backend.db.session import SessionLocal
    from backend.embedding_utils import encode_embedding
    from backend.services.reid_embedder import REID_EMBEDDING_DIM, get_embedder
    from backend.services.reid_index_manager import get_index_manager
    from backend.services.reid_matcher import resolve_identity

    db = SessionLocal()
    cam = db.query(Camera).filter(Camera.camera_id == "CAM-D12-A").first()

    idx = get_index_manager()
    await idx.initialize()

    TARGET = 0.78
    probe = np.full((128, 64, 3), 77, dtype=np.uint8)
    v = get_embedder().extract(probe).astype(np.float32)
    v /= np.linalg.norm(v)
    rng = np.random.default_rng(11)
    u = rng.standard_normal(REID_EMBEDDING_DIM).astype(np.float32)
    u -= np.dot(u, v) * v
    u /= np.linalg.norm(u)
    base = (TARGET * v + np.sqrt(1.0 - TARGET ** 2) * u).astype(np.float32)
    base /= np.linalg.norm(base)

    gp = GlobalPerson(
        representative_embedding=encode_embedding(base),
        first_seen_at=datetime.utcnow(), last_seen_at=datetime.utcnow(),
        total_sightings=1,
        retention_expires_at=datetime.utcnow() + timedelta(days=90),
        retention_hold=False, legal_basis="verify", is_deleted=False,
    )
    db.add(gp)
    db.flush()
    gp.faiss_index_position = await idx.add_person(gp.id, base)

    track = Track(camera_id="CAM-D12-A", track_id=7788, first_seen_frame=1,
                  last_seen_frame=1, best_confidence=0.9)
    db.add(track)
    db.commit()
    db.refresh(track)
    track_id = track.id
    db.close()

    tab = _FakeWSClient()
    dashboard_ws._clients.add(tab)
    try:
        result = await resolve_identity(
            track_db_id=track_id, camera_str_id="CAM-D12-A",
            camera_db_id=cam.id if cam else None,
            crop_bgr=probe, best_confidence=0.9,
        )
    finally:
        dashboard_ws._clients.discard(tab)

    check("A constructed 0.78 similarity resolves to REVIEW_QUEUE through the "
          "real resolve_identity() (band exercised deterministically)",
          result.get("decision") == "REVIEW_QUEUE", str(result.get("decision")))

    db = SessionLocal()
    created = db.query(ReIDReviewItem).filter(
        ReIDReviewItem.local_track_id == track_id
    ).first()
    check("resolve_identity() ITSELF created the ReIDReviewItem "
          "(no direct DB insert by test code)", created is not None)
    check("Track.body_embedding was written by the REVIEW_QUEUE branch — "
          "without it approve_review()'s embedding refresh is a silent no-op",
          db.query(Track).filter(Track.id == track_id).first().body_embedding is not None)
    db.close()

    check("reid_review_created was broadcast to connected dashboards "
          "(the event ReviewQueue.jsx refreshes on)",
          any(m.get("type") == "reid_review_created" for m in tab.received),
          str([m.get("type") for m in tab.received]))


def main() -> None:
    import asyncio

    from starlette.testclient import TestClient

    ids = seed()

    import backend.main as backend_main
    from backend.db.models import (
        GlobalPerson, Journey, JourneyQueryLog, ReIDCalibrationLog, ReIDReviewItem,
    )
    from backend.db.session import SessionLocal

    with TestClient(backend_main.app) as client:
        def _login(u, p):
            r = client.post("/api/v1/auth/login", json={"username": u, "password": p})
            return {"Authorization": f"Bearer {r.json()['access_token']}"}

        admin_h = _login("d12_admin", ADMIN_PW)
        officer_h = _login("d12_officer", OFFICER_PW)

        # ── The motivating bug: officers must be able to see this ───────────
        r_old = client.get(f"/api/v1/reid/journeys/{ids['gp_with']}", headers=officer_h)
        check("BASELINE: the investigative /reid/journeys endpoint still 403s "
              "for an officer on an unflagged candidate (Day 9 rule intact, "
              "NOT loosened by this work)",
              r_old.status_code == 403, f"HTTP {r_old.status_code}")

        r = client.get(
            f"/api/v1/reid/review-queue/{ids['item_with']}/candidate-journey",
            headers=officer_h,
        )
        check("OFFICER can load journey-so-far for a review candidate "
              "(the whole point — the queue is for officers)",
              r.status_code == 200, f"HTTP {r.status_code} {r.text[:160]}")

        # ── Checkpoint 1a: multi-camera journey renders ─────────────────────
        if r.status_code == 200:
            body = r.json()
            journey = body.get("journey", [])
            check("Journey-so-far returns the candidate's multi-camera journey",
                  len(journey) == 2, f"entries={len(journey)}")
            check("Journey entries are ordered oldest-first",
                  len(journey) == 2 and journey[0]["seen_at"] < journey[1]["seen_at"],
                  str([e.get("seen_at") for e in journey]))
            check("Journey entries carry camera_id, confidence and caveat "
                  "(the fields the timeline renders)",
                  all("camera_id" in e and "confidence" in e
                      and "confidence_caveat" in e for e in journey),
                  str(journey[0]) if journey else "")
            check("Confidence caveat preserved end-to-end (officer must see "
                  "that a prior link was itself a long-gap guess)",
                  any(e.get("confidence_caveat") for e in journey))

        # ── Checkpoint 1b: first-time candidate degrades cleanly ────────────
        r_empty = client.get(
            f"/api/v1/reid/review-queue/{ids['item_without']}/candidate-journey",
            headers=officer_h,
        )
        check("First-time candidate returns 200 with an EMPTY journey "
              "(clean 'no journey yet' state, not an error)",
              r_empty.status_code == 200 and r_empty.json().get("journey") == [],
              f"HTTP {r_empty.status_code} {r_empty.text[:120]}")

        # ── Authorization is scoped to the ITEM, not a free person id ───────
        r_404 = client.get(
            "/api/v1/reid/review-queue/999999/candidate-journey", headers=officer_h,
        )
        check("Unknown review item → 404 (no way to pass an arbitrary "
              "global_person_id to this endpoint)",
              r_404.status_code == 404, f"HTTP {r_404.status_code}")

        r_unauth = client.get(
            f"/api/v1/reid/review-queue/{ids['item_with']}/candidate-journey")
        check("Journey-so-far requires authentication",
              r_unauth.status_code in (401, 403), f"HTTP {r_unauth.status_code}")

        # ── Biometric-access register stays complete ────────────────────────
        db = SessionLocal()
        logs = db.query(JourneyQueryLog).filter(
            JourneyQueryLog.global_id_queried == ids["gp_with"]
        ).all()
        review_ctx = [l for l in logs if l.reason and "review item" in l.reason]
        check("Review-context journey reads ARE written to journey_query_log "
              "(register of who saw whose movements stays complete)",
              len(review_ctx) >= 1, f"rows={len(review_ctx)}")
        check("…and are tagged with the review item id, so a compliance "
              "reviewer can distinguish them from investigative lookups",
              any(f"#{ids['item_with']}" in (l.reason or "") for l in review_ctx),
              str([l.reason for l in review_ctx][:2]))
        db.close()

        # ── Checkpoint 3: approve → real Journey row + was_correct=True ─────
        r_ok = client.post(
            f"/api/v1/reid/review-queue/{ids['item_with']}/approve",
            headers={**officer_h, "X-Camera-Id": "CAM-D12-A"},
        )
        check("Approve returns 200", r_ok.status_code == 200,
              f"HTTP {r_ok.status_code} {r_ok.text[:160]}")

        db = SessionLocal()
        journeys_after = db.query(Journey).filter(
            Journey.global_person_id == ids["gp_with"]
        ).all()
        check("Approve created a real Journey row in the DB "
              "(queried directly, not inferred from the UI)",
              len(journeys_after) == 3, f"rows={len(journeys_after)}")

        cal_row = db.query(ReIDCalibrationLog).filter(
            ReIDCalibrationLog.id == ids["cal_id"]
        ).first()
        check("Approve flipped the original calibration log to was_correct=True",
              cal_row is not None and cal_row.was_correct is True,
              str(cal_row.was_correct if cal_row else None))

        approved = db.query(ReIDReviewItem).filter(
            ReIDReviewItem.id == ids["item_with"]
        ).first()
        check("Approved item is no longer PENDING",
              approved.status == "APPROVED", approved.status)
        db.close()

        # Resolved item must stop serving journey context.
        r_gone = client.get(
            f"/api/v1/reid/review-queue/{ids['item_with']}/candidate-journey",
            headers=officer_h,
        )
        check("Journey context is refused once the item is resolved — the "
              "justification for the lookup expires with the decision",
              r_gone.status_code == 404, f"HTTP {r_gone.status_code}")

        # ── Checkpoint 4: reject → new GlobalPerson + was_correct=False ─────
        db = SessionLocal()
        gp_before = db.query(GlobalPerson).count()
        db.close()

        r_rej = client.post(
            f"/api/v1/reid/review-queue/{ids['item_without']}/reject",
            headers={**officer_h, "X-Camera-Id": "CAM-D12-A"},
        )
        check("Reject returns 200", r_rej.status_code == 200,
              f"HTTP {r_rej.status_code} {r_rej.text[:160]}")

        db = SessionLocal()
        gp_after = db.query(GlobalPerson).count()
        check("Reject created a NEW GlobalPerson", gp_after == gp_before + 1,
              f"{gp_before} → {gp_after}")

        rejected_cals = db.query(ReIDCalibrationLog).filter(
            ReIDCalibrationLog.was_correct == False  # noqa: E712
        ).all()
        check("Reject wrote a calibration log with was_correct=False "
              "(the 'log false match' requirement)",
              len(rejected_cals) >= 1, f"rows={len(rejected_cals)}")
        db.close()

    # ── Checkpoint 2 (backend half) ────────────────────────────────────────
    asyncio.run(_check_review_item_created_by_pipeline())

    print(f"\n{'=' * 70}\n{len(PASS)} passed, {len(FAIL)} failed\n{'=' * 70}")
    if FAIL:
        print("FAILED:", FAIL)
        sys.exit(1)


if __name__ == "__main__":
    main()
