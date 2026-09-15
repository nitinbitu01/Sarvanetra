"""
backend/services/face_watchlist_matcher.py — Face watchlist matching + alert firing (Day 7).

Two responsibilities, kept in one file to mirror Day 6's reid_matcher.py:

  1. FaceWatchlistMatcher — pure in-memory cosine-similarity matcher against
     the active watchlist_persons face embeddings.

  2. resolve_watchlist_face() — the orchestration function that runs for every
     confirmed track (parallel to Day 6's resolve_identity(), not sequential
     with it): extract a face embedding, match it, and if matched, create the
     CRITICAL alert + audit log + WebSocket broadcast.

Why NOT FAISS here:
  The watchlist is small (hundreds to low thousands of entries) and changes
  rarely. A plain numpy matrix + matrix-multiply cosine search is simpler to
  reason about, avoids running two separate vector index systems side-by-side
  with Day 6's ReID FAISS index, and is fast enough at this scale — a few
  thousand entries is a trivial dot product. Only introduce FAISS here if the
  watchlist genuinely grows into the tens of thousands.

Why this is a SEPARATE pipeline from Day 6's ReID engine:
  ReID builds new open-set identities for unknown people crossing cameras.
  This matches against a small, fixed, KNOWN list of wanted/missing persons.
  Same "embedding -> similarity -> threshold" shape, fundamentally different
  problem and different consequence of getting it wrong (see the threshold
  comment in core/config.py) — do not merge the two engines or their indexes.

Threshold logic:
  score >= settings.WATCHLIST_FACE_MATCH_THRESHOLD (default 0.80) -> MATCH,
  CRITICAL alert fires automatically. Below that -> no alert. There is no
  review-queue band here (unlike ReID) — a missed match just means no alert
  fires; a human never sees a "maybe" watchlist hit sitting in a queue. This
  is a deliberate simplification for Day 7, not an oversight.

Usage:
    from backend.services.face_watchlist_matcher import get_face_watchlist_matcher
    matcher = get_face_watchlist_matcher()
    matcher.reload_watchlist()               # call once at startup, and again
                                              # any time the watchlist is edited
    result = matcher.match(face_embedding)   # WatchlistMatchResult | None

    from backend.services.face_watchlist_matcher import resolve_watchlist_face
    result = await resolve_watchlist_face(
        track_db_id=..., camera_str_id=..., camera_db_id=..., crop_bgr=...,
    )
"""
from __future__ import annotations

import json
import logging
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any

import numpy as np

from backend.core.config import settings
from backend.services.face_embedder import FACE_EMBEDDING_DIM, get_face_embedder

logger = logging.getLogger(__name__)

ACTION_WATCHLIST_FACE_MATCH_FIRED = "WATCHLIST_FACE_MATCH_FIRED"

# Placeholder danger score for an auto-fired watchlist match. The real
# danger-scoring system referenced in the original problem statement has not
# been built yet (explicitly out of scope for Day 7) — this fixed high value
# just needs to sort a confirmed wanted-person match above everything else in
# the UI until that system exists. Replace with a computed score, not a
# tuned constant, once it does.
WATCHLIST_MATCH_DANGER_SCORE = 95.0

_PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
_ALERT_EVIDENCE_DIR = _PROJECT_ROOT / "output" / "alerts" / "watchlist_face"


# ── Pure matcher ──────────────────────────────────────────────────────────────

@dataclass(frozen=True)
class WatchlistMatchResult:
    watchlist_person_id: int
    name: str
    reason: str | None
    reference_photo_path: str | None
    similarity: float


class FaceWatchlistMatcher:
    """Holds the active watchlist's face embeddings in memory as one matrix."""

    def __init__(self) -> None:
        self._ids: list[int] = []
        self._names: list[str] = []
        self._reasons: list[str | None] = []
        self._reference_photos: list[str | None] = []
        # Shape (N, 512), each row L2-normalized. Empty until reload_watchlist()
        # is called — matching before that just returns no matches, not an error.
        self._matrix: np.ndarray = np.zeros((0, FACE_EMBEDDING_DIM), dtype=np.float32)

    def reload_watchlist(self) -> int:
        """Re-read active watchlist_persons from the DB into memory.

        Call once at startup, and again any time the watchlist is edited
        elsewhere (adding/removing/deactivating a person) so matching stays
        current without a process restart. Day 7 does not build the watchlist
        editing UI itself — this method is the hook a future editing endpoint
        calls after writing to watchlist_persons.

        Returns:
            Number of entries loaded.
        """
        from backend.db.models import WatchlistPerson
        from backend.db.session import SessionLocal
        from backend.embedding_utils import decode_embedding, embedding_shape_ok

        db = SessionLocal()
        try:
            rows = (
                db.query(WatchlistPerson)
                .filter(WatchlistPerson.active == True)  # noqa: E712
                .all()
            )

            ids: list[int] = []
            names: list[str] = []
            reasons: list[str | None] = []
            photos: list[str | None] = []
            vectors: list[np.ndarray] = []

            for row in rows:
                if not embedding_shape_ok(row.face_embedding, FACE_EMBEDDING_DIM):
                    logger.error(
                        "watchlist_persons.id=%s has a face_embedding of the wrong "
                        "size (stored embedding_dim=%s, expected %s) — skipping. "
                        "This entry will NOT be matched until re-embedded.",
                        row.id, getattr(row, "embedding_dim", "?"), FACE_EMBEDDING_DIM,
                    )
                    continue
                try:
                    vec = decode_embedding(row.face_embedding, FACE_EMBEDDING_DIM)
                except ValueError as exc:
                    logger.error(
                        "Failed to decode face_embedding for watchlist_persons.id=%s: %s",
                        row.id, exc,
                    )
                    continue

                ids.append(row.id)
                names.append(row.name)
                reasons.append(row.reason)
                photos.append(row.reference_photo_path)
                vectors.append(self._normalize(vec))

            matrix = (
                np.vstack(vectors).astype(np.float32)
                if vectors
                else np.zeros((0, FACE_EMBEDDING_DIM), dtype=np.float32)
            )

            # Build fully before swapping — an atomic reference reassignment is
            # safe under the GIL, so match() never sees a half-updated state.
            self._ids = ids
            self._names = names
            self._reasons = reasons
            self._reference_photos = photos
            self._matrix = matrix

            logger.info("Face watchlist loaded: %d active entries.", len(ids))
            return len(ids)
        finally:
            db.close()

    def match(self, face_embedding: np.ndarray) -> WatchlistMatchResult | None:
        """Return the best watchlist match above threshold, or None.

        Args:
            face_embedding: Raw or pre-normalized 512-d vector (normalized
                             internally before comparison).
        """
        if self._matrix.shape[0] == 0:
            return None

        query = self._normalize(face_embedding)
        scores = self._matrix @ query  # cosine similarity — all rows pre-normalized
        best_idx = int(np.argmax(scores))
        best_score = float(scores[best_idx])

        threshold = settings.WATCHLIST_FACE_MATCH_THRESHOLD
        if best_score < threshold:
            return None

        return WatchlistMatchResult(
            watchlist_person_id=self._ids[best_idx],
            name=self._names[best_idx],
            reason=self._reasons[best_idx],
            reference_photo_path=self._reference_photos[best_idx],
            similarity=best_score,
        )

    @property
    def size(self) -> int:
        return self._matrix.shape[0]

    @staticmethod
    def _normalize(vec: np.ndarray) -> np.ndarray:
        norm = np.linalg.norm(vec)
        if norm < 1e-10:
            return np.zeros(FACE_EMBEDDING_DIM, dtype=np.float32)
        return (vec / norm).astype(np.float32)


_matcher_instance: FaceWatchlistMatcher | None = None


def get_face_watchlist_matcher() -> FaceWatchlistMatcher:
    """Return the singleton FaceWatchlistMatcher.

    Call reload_watchlist() once at application startup before matching.
    """
    global _matcher_instance
    if _matcher_instance is None:
        _matcher_instance = FaceWatchlistMatcher()
    return _matcher_instance


# ── Orchestration: extract -> match -> alert -> audit -> broadcast ───────────

def _save_snapshot(crop_bgr: np.ndarray, track_db_id: int) -> str | None:
    """Save the matched crop as evidence. Returns a project-root-relative path."""
    import cv2

    try:
        _ALERT_EVIDENCE_DIR.mkdir(parents=True, exist_ok=True)
        fname = f"track{track_db_id}_{datetime.utcnow().strftime('%Y%m%d%H%M%S%f')}.jpg"
        path = _ALERT_EVIDENCE_DIR / fname
        ok = cv2.imwrite(str(path), crop_bgr)
        if not ok:
            logger.error("cv2.imwrite returned False for watchlist alert snapshot: %s", path)
            return None
        return str(path.relative_to(_PROJECT_ROOT)).replace("\\", "/")
    except Exception as exc:
        logger.error("Failed to save watchlist alert snapshot: %s", exc)
        return None


async def _broadcast_alert(alert_dict: dict[str, Any]) -> None:
    """Broadcast the new alert to dashboard WebSocket clients (reuses the Day 5 channel)."""
    try:
        from backend.ws.dashboard_ws import broadcast
        await broadcast({"type": "new_alert", "alert": alert_dict})
    except Exception as exc:
        logger.debug("Failed to broadcast watchlist alert: %s", exc)


async def resolve_watchlist_face(
    track_db_id: int,
    camera_str_id: str,
    camera_db_id: int | None,
    crop_bgr: np.ndarray,
) -> dict[str, Any]:
    """Face-watchlist-match a confirmed track's crop; fire a CRITICAL alert on match.

    Runs alongside (not instead of) Day 6's resolve_identity() for the same
    confirmed track — they answer different questions and neither depends on
    the other's result. Creates its own DB session (safe to call as an
    asyncio.create_task from connection_manager).

    Args:
        track_db_id: DB primary key of the confirmed Track row.
        camera_str_id: String camera ID (legacy, for logging/audit compat).
        camera_db_id: Integer FK to cameras.id (nullable if not registered).
        crop_bgr: Person crop from the track (NOT a pre-cropped face — the
                  embedder runs its own face detector over this).

    Returns:
        Dict with key "decision": one of NO_FACE, NO_MATCH, WATCHLIST_MATCH,
        ERROR. On WATCHLIST_MATCH, also includes alert_id, watchlist_person_id,
        similarity.
    """
    embedder = get_face_embedder()
    matcher = get_face_watchlist_matcher()

    try:
        embedding = embedder.extract(crop_bgr)
    except Exception as exc:
        logger.error(
            "Face embedding extraction failed for track_db_id=%d: %s", track_db_id, exc
        )
        return {"decision": "ERROR", "error": str(exc)}

    if embedding is None:
        # No face detected in this crop (back turned, occluded, too small).
        # Expected and common — not an error, just nothing to match here.
        return {"decision": "NO_FACE"}

    match = matcher.match(embedding)
    if match is None:
        return {"decision": "NO_MATCH"}

    # ── CRITICAL: watchlist match — create alert, audit, broadcast ───────────
    from backend.db.models import Alert, Camera
    from backend.db.session import SessionLocal
    from backend.services.audit_logger import log_audit
    from backend.services.evidence import hash_file, set_readonly
    from backend.services.sentinel_iq import IQContext, compute_alert_iq

    db = SessionLocal()
    try:
        snapshot_path = _save_snapshot(crop_bgr, track_db_id)
        evidence_hash = hash_file(snapshot_path) if snapshot_path else None
        if snapshot_path:
            set_readonly(_PROJECT_ROOT / snapshot_path)

        subject_label = (
            f"{match.name} ({match.reason})" if match.reason else match.name
        )

        # Day 10: this call site already holds the `tracks` PK, so it passes
        # track_db_id directly rather than the pipeline id.
        iq = compute_alert_iq("WATCHLIST_FACE_MATCH", IQContext(
            db=db, camera_db_id=camera_db_id, camera_str_id=camera_str_id,
            track_db_id=track_db_id,
        ))

        alert = Alert(
            alert_type="WATCHLIST_FACE_MATCH",
            camera_id=camera_str_id or str(camera_db_id or "CAM_01"),
            track_id=track_db_id,
            subject_label=subject_label,
            confidence=match.similarity,
            danger_score=WATCHLIST_MATCH_DANGER_SCORE,
            iq_contribution=iq.total if iq else None,
            iq_breakdown_json=iq.to_json() if iq else None,
            snapshot_path=snapshot_path,
            evidence_hash=evidence_hash,
            status="new",
            meta_json=json.dumps({
                "watchlist_person_id": match.watchlist_person_id,
                "reference_photo_path": match.reference_photo_path,
                "reason": match.reason,
                "is_stub": embedder.is_stub,
            }),
        )
        db.add(alert)
        db.flush()   # populate alert.id

        from backend.services.alert_dedup import apply_dedup
        from backend.services.evidence_capture import on_alert_fired
        from backend.services.zone_incident_manager import check_zone_incident

        primary_alert = apply_dedup(alert, db)
        await check_zone_incident(primary_alert, db)
        # Day 13: severity + evidence capture, on the PRIMARY alert — a
        # duplicate merged away by apply_dedup must not spawn its own
        # evidence folder for an event already being captured.
        await on_alert_fired(primary_alert, db, camera_str_id)

        log_audit(
            db, user=None, action=ACTION_WATCHLIST_FACE_MATCH_FIRED,
            resource_type="alert", resource_id=alert.id,
            details={
                "watchlist_person_id": match.watchlist_person_id,
                "watchlist_person_name": match.name,
                "similarity": round(match.similarity, 4),
                "camera": camera_str_id,
                "track_db_id": track_db_id,
                "is_stub": embedder.is_stub,
                "merged_into_alert_id": alert.merged_into_alert_id,
            },
        )
        db.commit()

        camera = (
            db.query(Camera).filter(Camera.id == camera_db_id).first()
            if camera_db_id else None
        )
        alert_dict = {
            "id": alert.id,
            "alert_type": alert.alert_type,
            "merged_into_alert_id": alert.merged_into_alert_id,
            "subject_label": alert.subject_label,
            "confidence": alert.confidence,
            "danger_score": alert.danger_score,
            "iq_contribution": alert.iq_contribution,
            "iq_breakdown": iq.to_dict() if iq else None,
            "evidence_hash": alert.evidence_hash,
            "snapshot_path": alert.snapshot_path,
            "status": alert.status,
            "false_positive_reason": None,
            "camera_name": camera.name if camera else None,
            "camera_zone": camera.zone if camera else None,
            "created_at": alert.created_at.isoformat() if alert.created_at else "",
            "metadata": {
                "watchlist_person_id": match.watchlist_person_id,
                "reference_photo_path": match.reference_photo_path,
                "reason": match.reason,
                # Carried on the live-pushed alert too, not just the stored
                # row — the UI shows a "this match is meaningless" banner off
                # this flag, and an alert arriving over the WebSocket needs it
                # as much as one loaded from /alerts does.
                "is_stub": embedder.is_stub,
            },
        }
        await _broadcast_alert(alert_dict)

        logger.info(
            "WATCHLIST_FACE_MATCH: track=%d -> watchlist_person_id=%d (%s) score=%.3f%s",
            track_db_id, match.watchlist_person_id, match.name, match.similarity,
            " [STUB]" if embedder.is_stub else "",
        )

        return {
            "decision": "WATCHLIST_MATCH",
            "alert_id": alert.id,
            "watchlist_person_id": match.watchlist_person_id,
            "similarity": match.similarity,
        }

    except Exception as exc:
        db.rollback()
        logger.exception(
            "resolve_watchlist_face failed for track_db_id=%d: %s", track_db_id, exc
        )
        return {"decision": "ERROR", "error": str(exc)}
    finally:
        db.close()
