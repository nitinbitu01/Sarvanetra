"""
backend/routers/v1/reid_review.py â€” Review queue and GlobalPerson list endpoints.

Endpoints:
  GET  /api/v1/reid/review-queue         â€” list PENDING items (admin + officer)
  GET  /api/v1/reid/review-queue/{id}/candidate-journey
                                         â€” journey-so-far for THIS item's
                                           candidate (admin + officer)
  POST /api/v1/reid/review-queue/{id}/approve  â€” approve merge (admin + officer)
  POST /api/v1/reid/review-queue/{id}/reject   â€” reject merge (admin + officer)
  GET  /api/v1/reid/persons              â€” list GlobalPersons (admin only)

Why candidate-journey exists rather than reusing GET /reid/journeys/{id}:
  That endpoint is the INVESTIGATIVE lookup, and Day 9 deliberately gates it â€”
  OPERATOR gets 403 unless the person already has a non-DISMISSED alert
  ("nobody browses movement histories of people who haven't been flagged for
  something"). A review-queue candidate is by definition someone the system
  is still deciding about; most have no alert at all. Pointing the review UI
  at that endpoint returns 403 for officers on virtually every card â€”
  verified, not assumed â€” which would make the feature work only for admins,
  i.e. not for the queue's stated audience ("admin + officer").

  The fix is NOT to loosen the investigative rule. It is to notice that these
  are two different questions with two different authorization stories:

    /reid/journeys/{id}   "show me this person's movements"  â€” user picks the
                          person; needs justification; ADMIN-or-flagged only.
    â€¦/{item}/candidate-journey
                          "show me the context for the decision YOU asked me
                          to make" â€” the SYSTEM picked the person, the user
                          can only reach candidates surfaced to them by a
                          PENDING review item they can already see and action.

  So authorization here derives from the review item, not from the person id:
  no item, no journey. There is no way to pass an arbitrary global_person_id.
  Access is still written to journey_query_log (that register is meant to be
  a COMPLETE record of who saw whose movements â€” a review-context read is
  still a read), tagged with the item id so a compliance reviewer can tell
  the two provenances apart instead of seeing an unexplained spike.
"""
from __future__ import annotations

from datetime import datetime
from typing import Any

from fastapi import APIRouter, Depends, HTTPException, Request, status
from sqlalchemy.orm import Session

from backend.auth.dependencies import get_current_user, normalize_role, require_role
from backend.db.models import GlobalPerson, Journey, ReIDReviewItem, User
from backend.db.session import get_db
from backend.routers.v1.reid_journeys import _get_client_ip, _log_query
from backend.services.reid_matcher import approve_review, reject_review

router = APIRouter(prefix="/reid", tags=["ReID"])

_any_user = get_current_user   # admin + officer both allowed


# â”€â”€ Review queue â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€

@router.get("/review-queue", summary="List PENDING review items")
async def get_review_queue(
    limit: int = 50,
    db: Session = Depends(get_db),
    current_user: User = Depends(_any_user),
) -> list[dict[str, Any]]:
    """Return all PENDING ReID review items, newest first."""
    items = (
        db.query(ReIDReviewItem)
        .filter(ReIDReviewItem.status == "PENDING")
        .order_by(ReIDReviewItem.created_at.desc())
        .limit(limit)
        .all()
    )
    return [
        {
            "id": item.id,
            "local_track_id": item.local_track_id,
            "candidate_global_person_id": item.candidate_global_person_id,
            "similarity_score": round(item.similarity_score, 4),
            "crop_image_path": item.crop_image_path,
            "candidate_reference_image_path": item.candidate_reference_image_path,
            "status": item.status,
            "time_gap_seconds": item.time_gap_seconds,
            "created_at": item.created_at.isoformat() if item.created_at else None,
        }
        for item in items
    ]


@router.get(
    "/review-queue/{item_id}/candidate-journey",
    summary="Journey-so-far for a pending review item's candidate identity",
)
async def get_review_candidate_journey(
    item_id: int,
    request: Request,
    db: Session = Depends(get_db),
    current_user: User = Depends(_any_user),
) -> dict[str, Any]:
    """Return the candidate GlobalPerson's journey, as context for THIS decision.

    Deliberately NOT fetched as part of GET /review-queue: that would make the
    list endpoint carry an N+1 (one journey query per pending item) to render
    something the officer may never expand. The frontend loads this lazily,
    per card.

    Authorization derives from the review item â€” see the module docstring.
    404 (not 403) when the item is missing or already resolved: once an item
    is approved/rejected it is no longer a decision this endpoint is
    providing context for, and leaving it readable would be a way to keep
    pulling a person's movements after the justification expired.
    """
    item = (
        db.query(ReIDReviewItem)
        .filter(ReIDReviewItem.id == item_id, ReIDReviewItem.status == "PENDING")
        .first()
    )
    if not item:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="Review item not found or already resolved.",
        )

    gp_id = item.candidate_global_person_id
    role = normalize_role(current_user)
    source_ip = _get_client_ip(request)

    gp = (
        db.query(GlobalPerson)
        .filter(GlobalPerson.id == gp_id, GlobalPerson.is_deleted == False)  # noqa: E712
        .first()
    )

    # Log the access either way â€” a read attempt against a deleted/missing
    # candidate is exactly as interesting to a compliance reviewer as a
    # successful one.
    _log_query(
        db, current_user, role, gp_id, source_ip,
        outcome="SUCCESS" if gp else "FORBIDDEN",
        reason=f"review-queue context for review item #{item_id}",
    )
    db.commit()

    if not gp:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Candidate GlobalPerson {gp_id} not found or deleted.",
        )

    entries = (
        db.query(Journey)
        .filter(Journey.global_person_id == gp_id)
        .order_by(Journey.seen_at.asc())
        .all()
    )

    # Same per-entry shape as GET /reid/journeys/{id} so the frontend renders
    # one timeline format, not two that can drift apart.
    return {
        "review_item_id": item_id,
        "global_person_id": gp_id,
        "total_sightings": gp.total_sightings,
        "first_seen_at": gp.first_seen_at.isoformat() if gp.first_seen_at else None,
        "last_seen_at": gp.last_seen_at.isoformat() if gp.last_seen_at else None,
        "journey": [
            {
                "journey_id": e.id,
                "camera_id": e.camera_id,
                "camera_db_id": e.camera_db_id,
                "local_track_id": e.local_track_id,
                "seen_at": e.seen_at.isoformat() if e.seen_at else None,
                "confidence": round(e.confidence, 4) if e.confidence else None,
                "confidence_caveat": e.confidence_caveat,
            }
            for e in entries
        ],
    }


@router.post(
    "/review-queue/{item_id}/approve",
    summary="Approve: merge track into candidate GlobalPerson",
)
async def approve_review_item(
    item_id: int,
    request: Request,
    db: Session = Depends(get_db),
    current_user: User = Depends(_any_user),
) -> dict[str, Any]:
    """Officer approves a review item â€” merges the track into the candidate identity."""
    camera_str = request.headers.get("X-Camera-Id", "unknown")
    result = await approve_review(
        db=db,
        review_item_id=item_id,
        officer_user_id=current_user.id,
        camera_str_id=camera_str,
        camera_db_id=None,
    )
    if "error" in result:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail=result["error"])
    return result


@router.post(
    "/review-queue/{item_id}/reject",
    summary="Reject: create new GlobalPerson for the track",
)
async def reject_review_item(
    item_id: int,
    request: Request,
    db: Session = Depends(get_db),
    current_user: User = Depends(_any_user),
) -> dict[str, Any]:
    """Officer rejects a review item â€” creates a new identity for the track."""
    camera_str = request.headers.get("X-Camera-Id", "unknown")
    result = await reject_review(
        db=db,
        review_item_id=item_id,
        officer_user_id=current_user.id,
        camera_str_id=camera_str,
        camera_db_id=None,
    )
    if "error" in result:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail=result["error"])
    return result


# â”€â”€ GlobalPerson list (admin only) â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€

@router.get("/persons", summary="List GlobalPersons (admin only)")
async def list_persons(
    limit: int = 100,
    include_deleted: bool = False,
    db: Session = Depends(get_db),
    current_user: User = Depends(require_role("admin")),
) -> list[dict[str, Any]]:
    """Return all (non-deleted by default) GlobalPerson records."""
    q = db.query(GlobalPerson)
    if not include_deleted:
        q = q.filter(GlobalPerson.is_deleted == False)  # noqa: E712
    persons = q.order_by(GlobalPerson.last_seen_at.desc()).limit(limit).all()
    return [
        {
            "id": p.id,
            "total_sightings": p.total_sightings,
            "first_seen_at": p.first_seen_at.isoformat() if p.first_seen_at else None,
            "last_seen_at": p.last_seen_at.isoformat() if p.last_seen_at else None,
            "retention_expires_at": (
                p.retention_expires_at.isoformat() if p.retention_expires_at else None
            ),
            "retention_hold": p.retention_hold,
            "legal_basis": p.legal_basis,
            "is_deleted": p.is_deleted,
            "faiss_index_position": p.faiss_index_position,
        }
        for p in persons
    ]
