"""
backend/routers/v1/vault.py — Active Learning Vault Router.
Endpoints for Vault health statistics, stratified uncertainty-sampling, and calibration evaluation.
"""
import logging
from typing import Optional, List, Dict, Any
from fastapi import APIRouter, Depends, HTTPException, status
from sqlalchemy.orm import Session

from backend.db.session import get_db
from backend.auth.dependencies import get_current_user, get_current_user_optional
from backend.schemas.vault import CalibrationJobRequest
from backend.services.vault_service import VaultService
from backend.services.anchor_evaluator import AnchorEvaluator

logger = logging.getLogger("sentinel.vault_router")
router = APIRouter(prefix="/vault", tags=["Active Learning Vault"])


@router.get("/stats")
async def get_vault_statistics(
    db: Session = Depends(get_db),
    user=Depends(get_current_user_optional),
):
    """
    Returns live statistics across Gold, Hard Negative, Hard Positive, and Anchor compartments.
    """
    service = VaultService(db=db)
    try:
        return service.get_vault_stats()
    except Exception as e:
        logger.error(f"Failed to fetch vault statistics: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/sample")
async def sample_retraining_dataset(
    total_samples: int = 500,
    entity_type: str = "vehicle",
    db: Session = Depends(get_db),
    user=Depends(get_current_user),
):
    """
    Extracts stratified active learning dataset using uncertainty sampling (|ai_score - threshold|)
    and per-camera diversity caps to prevent camera-specific model overfitting.
    """
    service = VaultService(db=db)
    try:
        return service.sample_for_retraining(total_samples=total_samples, entity_type=entity_type)
    except Exception as e:
        logger.error(f"Failed to sample retraining dataset: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/anchor-eval")
async def evaluate_anchor_regression(
    camera_id: str,
    lighting_condition: str = "day",
    proposed_threshold: float = 0.35,
    db: Session = Depends(get_db),
    user=Depends(get_current_user),
):
    """
    Evaluates proposed camera threshold against frozen anchor ground truth set.
    """
    evaluator = AnchorEvaluator(db=db)
    try:
        report = evaluator.evaluate_profile(
            camera_id=camera_id,
            proposed_thresholds={lighting_condition: {"vehicle": proposed_threshold}},
            lighting_condition=lighting_condition,
        )
        return report
    except Exception as e:
        logger.error(f"Anchor evaluation failed: {e}")
        raise HTTPException(status_code=500, detail=str(e))

