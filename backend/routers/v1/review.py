"""
backend/routers/v1/review.py — Human-In-The-Loop (HITL) Officer Review Router.
Provides blind review endpoints, server-authoritative time-gating, and double-review escalations.
"""
import logging
from typing import List, Optional
from fastapi import APIRouter, Depends, HTTPException, status
from sqlalchemy.orm import Session

from backend.db.session import get_db
from backend.auth.dependencies import require_officer_auth
from backend.schemas.vault import (
    ReviewStartRequest, ReviewStartResponse,
    ReviewSubmitRequest, ReviewSubmitResponse,
)
from backend.services.review_service import ReviewService
from backend.db.models import ReviewPair

logger = logging.getLogger("sentinel.review_router")
router = APIRouter(prefix="/review", tags=["Active Learning Review"])


@router.get("/queue", summary="List pending alerts awaiting officer review")
async def get_pending_review_queue(
    limit: int = 50,
    db: Session = Depends(get_db),
    officer=Depends(require_officer_auth),
):
    """
    Returns pending active alerts for human-in-the-loop review.
    Populates the officer review queue with borderline, critical, and double-review alerts.
    """
    service = ReviewService(db=db)
    officer_id = str(getattr(officer, "id", "USER_OPS_01"))
    try:
        items = service.get_review_queue(limit=limit, officer_id=officer_id)
        return items
    except Exception as e:
        logger.error(f"Failed to fetch review queue: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/start", response_model=ReviewStartResponse)
async def start_review(
    request: ReviewStartRequest,
    db: Session = Depends(get_db),
    officer=Depends(require_officer_auth),
):
    """
    Start a blind review session with server-authoritative time gating.
    Hides AI confidence score to prevent officer confirmation bias.
    """
    service = ReviewService(db=db)
    try:
        response = await service.start_review_session(request, officer)
        return response
    except Exception as e:
        logger.error(f"Failed to start review session for alert {request.alert_id}: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/submit", response_model=ReviewSubmitResponse)
async def submit_review(
    request: ReviewSubmitRequest,
    db: Session = Depends(get_db),
    officer=Depends(require_officer_auth),
):
    """
    Submit an officer review decision with server-authoritative duration validation.
    Under-duration submissions are rejected to prevent fatigue speed-clicking.
    Borderline cases are automatically routed to a second reviewer.
    """
    service = ReviewService(db=db)
    try:
        response = await service.submit_review_decision(request)
        return response
    except Exception as e:
        logger.error(f"Failed to submit review decision {request.review_session_id}: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/escalations")
async def get_escalated_reviews(
    db: Session = Depends(get_db),
    officer=Depends(require_officer_auth),
):
    """
    Returns double-review cases where independent officers disagreed.
    Accessible to supervisors for tie-breaking.
    """
    pairs = (
        db.query(ReviewPair)
        .filter(ReviewPair.status == "disagreed_escalated")
        .order_by(ReviewPair.created_at.desc())
        .limit(50)
        .all()
    )
    return [
        {
            "review_pair_id": p.id,
            "alert_id": p.alert_id,
            "first_review_id": p.first_review_id,
            "second_review_id": p.second_review_id,
            "status": p.status,
            "resolved_at": p.resolved_at,
            "created_at": p.created_at,
        }
        for p in pairs
    ]
