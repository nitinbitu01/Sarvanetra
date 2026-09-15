"""
backend/services/reid_matcher.py — ReID identity resolution engine.

This is the core decision function that runs for every confirmed track.
It orchestrates:
  1. FAISS similarity search against known GlobalPersons
  2. Threshold-based decision (auto-merge / review / new identity)
  3. Time-gap confidence caveat (clothing-change blind-spot mitigation)
  4. Journey record creation
  5. Audit log + calibration log writes
  6. WebSocket broadcast of resolution result

Threshold logic (all values from settings, not hardcoded):
  score >= REID_AUTO_MERGE_THRESHOLD  → AUTO_MERGE
  REID_REVIEW_LOWER_THRESHOLD <= score < REID_AUTO_MERGE_THRESHOLD → REVIEW_QUEUE
  score < REID_REVIEW_LOWER_THRESHOLD OR no results → NEW_IDENTITY

Appearance-staleness caveat:
  Even on AUTO_MERGE, if time_gap_seconds > REID_APPEARANCE_STALE_HOURS * 3600,
  the Journey row gets a confidence_caveat string. This is the concrete mechanism
  that prevents the system from silently overclaiming certainty across long gaps
  where clothing, lighting, or context may have changed.

Concurrency safety:
  Called via asyncio.create_task() from connection_manager — runs off the main
  detection loop. DB session is created internally (not from the request scope)
  to avoid session lifetime issues in background tasks.
"""
from __future__ import annotations

import asyncio
import json
import logging
from datetime import datetime, timedelta
from typing import Any

from sqlalchemy.orm import Session

from backend.core.config import settings
from backend.db.models import (
    AuditLog,
    GlobalPerson,
    Journey,
    ReIDCalibrationLog,
    ReIDReviewItem,
    Track,
)
from backend.db.session import SessionLocal
from backend.embedding_utils import decode_embedding, encode_embedding
from backend.services.reid_embedder import REID_EMBEDDING_DIM, get_embedder
from backend.services.reid_index_manager import get_index_manager
from backend.services.audit_logger import log_audit
from backend.services.geo_guard import get_geo_guard
from backend.core.reid_smoother import get_reid_smoother, consensus_vote

logger = logging.getLogger(__name__)

# Action strings used in AuditLog — kept as constants to catch typos at lint time
ACTION_AUTO_MERGE     = "REID_AUTO_MERGE"
ACTION_REVIEW_QUEUED  = "REID_REVIEW_QUEUED"
ACTION_REVIEW_APPROVED = "REID_REVIEW_APPROVED"
ACTION_REVIEW_REJECTED = "REID_REVIEW_REJECTED"
ACTION_NEW_IDENTITY   = "REID_NEW_IDENTITY"


def _stale_caveat(time_gap_seconds: int) -> str | None:
    """Return a confidence caveat string if the sighting gap is too long."""
    threshold_secs = settings.REID_APPEARANCE_STALE_HOURS * 3600
    if time_gap_seconds > threshold_secs:
        hours = time_gap_seconds // 3600
        return (
            f"⚠ Long gap since last sighting ({hours}h). "
            f"Appearance-based match only — clothing or context may have changed. "
            f"Officer verification recommended."
        )
    return None


def _compute_time_gap(gp: GlobalPerson) -> int:
    """Seconds between now and the candidate's last_seen_at."""
    now = datetime.utcnow()
    delta = now - gp.last_seen_at
    return max(0, int(delta.total_seconds()))


def _update_global_person(
    db: Session,
    gp: GlobalPerson,
    new_embedding: bytes,
) -> None:
    """Update a GlobalPerson's running-average embedding and sighting stats."""
    gp.last_seen_at = datetime.utcnow()
    gp.total_sightings = (gp.total_sightings or 0) + 1

    # Update representative embedding as exponential moving average (α=0.1)
    # This slowly adapts the reference to appearance changes while staying stable.
    if gp.representative_embedding:
        try:
            old_vec = decode_embedding(gp.representative_embedding, REID_EMBEDDING_DIM)
            new_vec = decode_embedding(new_embedding, REID_EMBEDDING_DIM)
            alpha = 0.1
            updated = (1 - alpha) * old_vec + alpha * new_vec
            # Normalize back to unit sphere
            norm = __import__("numpy").linalg.norm(updated)
            if norm > 1e-10:
                updated = updated / norm
            gp.representative_embedding = encode_embedding(updated)
        except Exception as exc:
            logger.warning("Failed to update embedding EMA for gp_id=%d: %s", gp.id, exc)
            gp.representative_embedding = new_embedding  # fallback: replace
    else:
        gp.representative_embedding = new_embedding


def _create_journey(
    db: Session,
    global_person_id: int,
    track_id: int,
    camera_db_id: int | None,
    camera_str_id: str,
    confidence: float,
    time_gap_seconds: int,
) -> Journey:
    """Create a Journey row for a resolved identity sighting."""
    caveat = _stale_caveat(time_gap_seconds)
    journey = Journey(
        global_person_id=global_person_id,
        local_track_id=track_id,
        camera_db_id=camera_db_id,
        camera_id=camera_str_id,            # keep legacy string col too
        seen_at=datetime.utcnow(),
        confidence=confidence,
        confidence_caveat=caveat,
    )
    db.add(journey)
    return journey


def _write_calibration_log(
    db: Session,
    similarity_score: float,
    decision: str,
    time_gap_seconds: int | None,
    was_correct: bool | None = None,
) -> ReIDCalibrationLog:
    """Write a calibration log row for every resolved decision."""
    cal = ReIDCalibrationLog(
        similarity_score=similarity_score,
        decision=decision,
        was_correct=was_correct,
        time_gap_seconds=time_gap_seconds,
    )
    db.add(cal)
    db.flush()
    return cal


async def resolve_identity(
    track_db_id: int,
    camera_str_id: str,
    camera_db_id: int | None,
    crop_bgr,            # np.ndarray — OpenCV BGR crop
    best_confidence: float,
) -> dict[str, Any]:
    """Resolve a confirmed track's identity against the global person index.

    This is the main ReID pipeline function. Creates its own DB session
    (safe to call as an asyncio.create_task from connection_manager).

    Args:
        track_db_id: DB primary key of the confirmed Track row.
        camera_str_id: String camera ID (legacy, for journey compat).
        camera_db_id: Integer FK to cameras.id (nullable if not registered).
        crop_bgr: Best-confidence BGR person crop from the track.
        best_confidence: Detection confidence of the crop (used in journey).

    Returns:
        Dict with keys: decision, global_person_id, similarity_score, caveat,
        review_item_id (if REVIEW_QUEUE), is_stub.
    """
    embedder = get_embedder()
    index_mgr = get_index_manager()

    geo_guard = get_geo_guard()
    smoother = get_reid_smoother()

    # 1. Extract embedding and add to temporal smoother buffer
    try:
        embedding = embedder.extract(crop_bgr)
        smoother.add_frame(track_db_id, embedding)
        search_vec = smoother.get_centroid_embedding(track_db_id)
        if search_vec is None:
            search_vec = embedding
    except Exception as exc:
        logger.error("Embedding extraction failed for track_db_id=%d: %s", track_db_id, exc)
        return {"decision": "ERROR", "error": str(exc)}

    embedding_blob = encode_embedding(embedding)

    # 2. Search FAISS (run in thread executor to avoid blocking event loop)
    loop = asyncio.get_event_loop()

    # Get set of deleted IDs to exclude from search
    db = SessionLocal()
    try:
        deleted_ids: set[int] = {
            row.id
            for row in db.query(GlobalPerson.id)
            .filter(GlobalPerson.is_deleted == True)  # noqa: E712
            .all()
        }

        candidates = await index_mgr.search(
            search_vec, k=5, exclude_deleted_ids=deleted_ids
        )

        top = candidates[0] if candidates else None
        top_score = top["score"] if top else 0.0
        top_gp_id = top["global_person_id"] if top else None

        # 3. Load candidate GlobalPerson for time-gap and spatial feasibility calculation
        candidate_gp = None
        time_gap_seconds = 0
        is_feasible = True
        geo_reason = "NO_CANDIDATE"

        if top_gp_id is not None:
            candidate_gp = db.query(GlobalPerson).filter(
                GlobalPerson.id == top_gp_id,
                GlobalPerson.is_deleted == False,  # noqa: E712
            ).first()
            if candidate_gp:
                time_gap_seconds = _compute_time_gap(candidate_gp)
                last_journey = (
                    db.query(Journey)
                    .filter(Journey.global_person_id == candidate_gp.id)
                    .order_by(Journey.seen_at.desc())
                    .first()
                )
                last_cam = last_journey.camera_id if last_journey else camera_str_id
                is_feasible, geo_reason = geo_guard.is_transit_feasible(
                    last_cam, camera_str_id, time_gap_seconds
                )
                if not is_feasible:
                    logger.warning(
                        "REID GEO_REJECT: track=%d candidate_gp=%d score=%.3f reason=%s",
                        track_db_id, candidate_gp.id, top_score, geo_reason
                    )

        # 4. Decision logic with adaptive consensus voting
        auto_threshold = settings.REID_AUTO_MERGE_THRESHOLD
        review_threshold = settings.REID_REVIEW_LOWER_THRESHOLD

        history_sims = [top_score]
        if candidate_gp and candidate_gp.representative_embedding:
            try:
                cand_vec = decode_embedding(candidate_gp.representative_embedding, REID_EMBEDDING_DIM)
                history_sims = smoother.get_history_similarities(track_db_id, cand_vec) or [top_score]
            except Exception:
                pass

        has_consensus = consensus_vote(
            history_sims, auto_threshold, s_geo=1.0 if is_feasible else 0.0, min_votes=3
        ) if len(history_sims) >= 3 else True

        result: dict[str, Any] = {
            "is_stub": embedder.is_stub,
            "similarity_score": top_score,
            "geo_feasible": is_feasible,
            "geo_reason": geo_reason,
        }

        if top_score >= auto_threshold and candidate_gp and is_feasible and has_consensus:
            # ── AUTO MERGE ─────────────────────────────────────────────────
            decision = "AUTO_MERGE"
            _update_global_person(db, candidate_gp, embedding_blob)
            journey = _create_journey(
                db, candidate_gp.id, track_db_id, camera_db_id,
                camera_str_id, best_confidence, time_gap_seconds,
            )
            cal = _write_calibration_log(
                db, top_score, ACTION_AUTO_MERGE, time_gap_seconds
            )
            log_audit(db, user=None, action=ACTION_AUTO_MERGE,
                      resource_type="global_person", resource_id=candidate_gp.id,
                      details={
                          "track_db_id": track_db_id,
                          "score": round(top_score, 4),
                          "camera": camera_str_id,
                          "time_gap_s": time_gap_seconds,
                          "geo_reason": geo_reason,
                          "caveat": journey.confidence_caveat,
                          "is_stub": embedder.is_stub,
                      })
            db.commit()
            result.update({
                "decision": decision,
                "global_person_id": candidate_gp.id,
                "caveat": journey.confidence_caveat,
                "calibration_log_id": cal.id,
            })
            logger.info(
                "REID AUTO_MERGE: track=%d → gp_id=%d score=%.3f gap=%ds%s",
                track_db_id, candidate_gp.id, top_score, time_gap_seconds,
                " [STUB]" if embedder.is_stub else "",
            )

        elif top_score >= review_threshold and candidate_gp:
            # ── REVIEW QUEUE ───────────────────────────────────────────────
            decision = "REVIEW_QUEUE"
            cal = _write_calibration_log(
                db, top_score, "REVIEW_PENDING", time_gap_seconds
            )
            track_row = db.query(Track).filter(Track.id == track_db_id).first()
            if track_row is not None:
                # Persist this sighting's embedding on the Track row.
                #
                # Without this, approve_review()'s "re-extract embedding from
                # the track's best crop" step is a silent no-op: it guards on
                # `if track and track.body_embedding`, and nothing ever wrote
                # that column, so approving a review updated the
                # GlobalPerson's sighting count but never actually folded the
                # newly-confirmed appearance into its representative
                # embedding. The merge was recorded but the identity never
                # learned from it.
                #
                # Only the REVIEW_QUEUE branch needs this today, because it is
                # the only branch whose result is consumed later by a separate
                # request (the officer's approve click). AUTO_MERGE and
                # NEW_IDENTITY both fold the embedding into the GlobalPerson
                # inline, in this same call, and never read it back off Track.
                track_row.body_embedding = embedding_blob

            review = ReIDReviewItem(
                local_track_id=track_db_id,
                candidate_global_person_id=candidate_gp.id,
                similarity_score=top_score,
                crop_image_path=track_row.best_crop_path if track_row else None,
                candidate_reference_image_path=None,  # set if we store ref images
                status="PENDING",
                time_gap_seconds=time_gap_seconds,
                calibration_log_id=cal.id,
            )
            db.add(review)
            db.flush()
            log_audit(db, user=None, action=ACTION_REVIEW_QUEUED,
                      resource_type="reid_review_item", resource_id=review.id,
                      details={
                          "track_db_id": track_db_id,
                          "candidate_gp_id": candidate_gp.id,
                          "score": round(top_score, 4),
                          "camera": camera_str_id,
                          "time_gap_s": time_gap_seconds,
                          "is_stub": embedder.is_stub,
                      })
            db.commit()
            result.update({
                "decision": decision,
                "global_person_id": candidate_gp.id,
                "review_item_id": review.id,
                "calibration_log_id": cal.id,
            })
            logger.info(
                "REID REVIEW_QUEUE: track=%d → candidate gp_id=%d score=%.3f",
                track_db_id, candidate_gp.id, top_score,
            )

        else:
            # ── NEW IDENTITY ────────────────────────────────────────────────
            decision = "NEW_IDENTITY"
            retention_expires = datetime.utcnow() + timedelta(
                days=settings.REID_RETENTION_DAYS
            )
            gp = GlobalPerson(
                representative_embedding=embedding_blob,
                first_seen_at=datetime.utcnow(),
                last_seen_at=datetime.utcnow(),
                total_sightings=1,
                retention_expires_at=retention_expires,
                retention_hold=False,
                legal_basis="routine_public_safety_monitoring",
                is_deleted=False,
            )
            db.add(gp)
            db.flush()   # populate gp.id

            # Add to FAISS (must happen after flush gives us gp.id)
            faiss_pos = await index_mgr.add_person(gp.id, embedding)
            gp.faiss_index_position = faiss_pos
            db.flush()

            journey = _create_journey(
                db, gp.id, track_db_id, camera_db_id,
                camera_str_id, best_confidence, 0,
            )
            cal = _write_calibration_log(
                db, top_score, ACTION_NEW_IDENTITY, time_gap_seconds
            )
            log_audit(db, user=None, action=ACTION_NEW_IDENTITY,
                      resource_type="global_person", resource_id=gp.id,
                      details={
                          "track_db_id": track_db_id,
                          "score": round(top_score, 4),
                          "camera": camera_str_id,
                          "faiss_pos": faiss_pos,
                          "is_stub": embedder.is_stub,
                      })
            db.commit()
            result.update({
                "decision": decision,
                "global_person_id": gp.id,
                "calibration_log_id": cal.id,
                "caveat": None,
            })
            logger.info(
                "REID NEW_IDENTITY: track=%d → new gp_id=%d",
                track_db_id, gp.id,
            )

        # 5. WebSocket broadcast
        await _broadcast_reid_event(result)
        return result

    except Exception as exc:
        db.rollback()
        logger.exception("resolve_identity failed for track_db_id=%d: %s", track_db_id, exc)
        return {"decision": "ERROR", "error": str(exc)}
    finally:
        db.close()


async def _broadcast_reid_event(result: dict[str, Any]) -> None:
    """Broadcast ReID resolution result to dashboard WebSocket clients."""
    try:
        from backend.ws.dashboard_ws import broadcast
        decision = result.get("decision", "UNKNOWN")
        if decision == "REVIEW_QUEUE":
            await broadcast({
                "type": "reid_review_created",
                "review_item_id": result.get("review_item_id"),
                "global_person_id": result.get("global_person_id"),
                "score": result.get("similarity_score"),
            })
        else:
            await broadcast({
                "type": "reid_identity_resolved",
                "decision": decision,
                "global_person_id": result.get("global_person_id"),
                "score": result.get("similarity_score"),
                "caveat": result.get("caveat"),
            })
    except Exception as exc:
        logger.debug("Failed to broadcast ReID event: %s", exc)


# ── Review queue approval / rejection ─────────────────────────────────────────

async def approve_review(
    db: Session, review_item_id: int, officer_user_id: int, camera_str_id: str,
    camera_db_id: int | None,
) -> dict[str, Any]:
    """Officer approves a review item: merge track into candidate GlobalPerson."""
    review = db.query(ReIDReviewItem).filter(
        ReIDReviewItem.id == review_item_id,
        ReIDReviewItem.status == "PENDING",
    ).first()
    if not review:
        return {"error": "Review item not found or already resolved."}

    gp = db.query(GlobalPerson).filter(
        GlobalPerson.id == review.candidate_global_person_id,
        GlobalPerson.is_deleted == False,  # noqa: E712
    ).first()
    if not gp:
        return {"error": "Candidate GlobalPerson not found or deleted."}

    track = db.query(Track).filter(Track.id == review.local_track_id).first()

    # Re-extract embedding from track's best crop (if available)
    if track and track.body_embedding:
        try:
            new_emb = decode_embedding(track.body_embedding, REID_EMBEDDING_DIM)
            _update_global_person(db, gp, encode_embedding(new_emb))
        except Exception:
            pass

    review.status = "APPROVED"
    review.reviewed_by_user_id = officer_user_id
    review.reviewed_at = datetime.utcnow()

    # Update calibration log: was_correct = True
    if review.calibration_log_id:
        cal = db.query(ReIDCalibrationLog).filter(
            ReIDCalibrationLog.id == review.calibration_log_id
        ).first()
        if cal:
            cal.was_correct = True
            cal.decision = ACTION_REVIEW_APPROVED

    # Write a new calibration log for this specific decision
    _write_calibration_log(
        db, review.similarity_score, ACTION_REVIEW_APPROVED,
        review.time_gap_seconds, was_correct=True,
    )

    # Create Journey entry
    _create_journey(
        db, gp.id, review.local_track_id or 0, camera_db_id,
        camera_str_id, review.similarity_score, review.time_gap_seconds or 0,
    )

    log_audit(db, user=None, action=ACTION_REVIEW_APPROVED,
              resource_type="reid_review_item", resource_id=review.id,
              details={
                  "global_person_id": gp.id,
                  "track_db_id": review.local_track_id,
                  "score": round(review.similarity_score, 4),
                  "officer_user_id": officer_user_id,
              })
    db.commit()
    return {"status": "APPROVED", "global_person_id": gp.id}


async def reject_review(
    db: Session, review_item_id: int, officer_user_id: int, camera_str_id: str,
    camera_db_id: int | None,
) -> dict[str, Any]:
    """Officer rejects a review item: create a new GlobalPerson for the track."""
    review = db.query(ReIDReviewItem).filter(
        ReIDReviewItem.id == review_item_id,
        ReIDReviewItem.status == "PENDING",
    ).first()
    if not review:
        return {"error": "Review item not found or already resolved."}

    review.status = "REJECTED"
    review.reviewed_by_user_id = officer_user_id
    review.reviewed_at = datetime.utcnow()

    # Update calibration log: was_correct = False
    if review.calibration_log_id:
        cal = db.query(ReIDCalibrationLog).filter(
            ReIDCalibrationLog.id == review.calibration_log_id
        ).first()
        if cal:
            cal.was_correct = False
            cal.decision = ACTION_REVIEW_REJECTED

    _write_calibration_log(
        db, review.similarity_score, ACTION_REVIEW_REJECTED,
        review.time_gap_seconds, was_correct=False,
    )

    # Create a new GlobalPerson for the rejected track
    track = db.query(Track).filter(Track.id == review.local_track_id).first()
    emb_blob: bytes | None = track.body_embedding if track else None
    retention_expires = datetime.utcnow() + timedelta(days=settings.REID_RETENTION_DAYS)

    gp = GlobalPerson(
        representative_embedding=emb_blob,
        first_seen_at=datetime.utcnow(),
        last_seen_at=datetime.utcnow(),
        total_sightings=1,
        retention_expires_at=retention_expires,
        retention_hold=False,
        legal_basis="routine_public_safety_monitoring",
        is_deleted=False,
    )
    db.add(gp)
    db.flush()

    # Add to FAISS (synchronous here; called from route handler on the event loop)
    if emb_blob:
        try:
            embedding = decode_embedding(emb_blob, REID_EMBEDDING_DIM)
            index_mgr = get_index_manager()
            loop = asyncio.get_event_loop()
            faiss_pos = await index_mgr.add_person(gp.id, embedding)
            gp.faiss_index_position = faiss_pos
            db.flush()
        except Exception as exc:
            logger.warning("Failed to add rejected track to FAISS: %s", exc)

    _create_journey(
        db, gp.id, review.local_track_id or 0, camera_db_id,
        camera_str_id, 0.0, 0,
    )

    log_audit(db, user=None, action=ACTION_REVIEW_REJECTED,
              resource_type="reid_review_item", resource_id=review.id,
              details={
                  "new_global_person_id": gp.id,
                  "candidate_gp_id": review.candidate_global_person_id,
                  "track_db_id": review.local_track_id,
                  "score": round(review.similarity_score, 4),
                  "officer_user_id": officer_user_id,
              })
    db.commit()
    return {"status": "REJECTED", "new_global_person_id": gp.id}
