"""
backend/schemas/vault.py — Pydantic schemas for Active Learning Vault,
Officer HITL Reviews, Calibration, and Audit Verification.
"""
from uuid import UUID
from datetime import datetime
from typing import Optional, List, Dict, Any, Union
from pydantic import BaseModel, Field

from backend.core.vault_config import (
    VaultCompartment, LabelTrust, ReviewDecision, EntityType, LightingCondition
)


# ============================================================
# VAULT ENTRY SCHEMAS
# ============================================================
class VaultEntryCreate(BaseModel):
    model_config = {"protected_namespaces": ()}

    alert_id:            Optional[Union[int, str]] = None
    camera_id:           str
    captured_at:         Optional[datetime] = None
    compartment:         VaultCompartment = VaultCompartment.PROBATIONARY
    trust_level:         LabelTrust = LabelTrust.PROBATIONARY
    entity_type:         EntityType = EntityType.VEHICLE
    lighting_condition:  Optional[LightingCondition] = None
    vehicle_color_hsv:   Optional[Dict[str, Any]] = None
    vehicle_type:        Optional[str] = None
    plate_text:          Optional[str] = None
    ai_confidence:       Optional[float] = Field(None, ge=0.0, le=1.0)
    ai_cosine_distance:  Optional[float] = Field(None, ge=0.0, le=2.0)
    model_version:       Optional[str] = None
    camera_zone:         Optional[str] = None
    anchor_group_id:     Optional[str] = None


class VaultEntryResponse(BaseModel):
    id:                  str
    compartment:         str
    trust_level:         str
    entity_type:         str
    camera_id:           str
    captured_at:         datetime
    stored_at:           datetime
    crop_minio_path:     Optional[str] = None
    ai_confidence:       Optional[float] = None
    training_eligible:   bool
    used_in_training:    bool
    lighting_condition:  Optional[str] = None
    boundary_distance:   Optional[float] = None

    class Config:
        from_attributes = True


# ============================================================
# OFFICER REVIEW SCHEMAS
# ============================================================
class ReviewStartRequest(BaseModel):
    alert_id: Union[int, str]


class ReviewStartResponse(BaseModel):
    review_session_id:  str
    alert_id:           Union[int, str]
    probe_crop_url:     str
    candidate_crops:    List[Dict[str, Any]] = []
    entity_type:        str
    min_review_seconds: float
    server_started_at:  datetime
    is_second_review:   bool = False


class ReviewSubmitRequest(BaseModel):
    review_session_id:   str
    alert_id:            Union[int, str]
    decision:            ReviewDecision
    officer_notes:       Optional[str] = None
    client_started_at:   Optional[datetime] = None
    client_completed_at: Optional[datetime] = None


class ReviewSubmitResponse(BaseModel):
    success:                bool
    vault_entry_id:         Optional[str] = None
    decision:               ReviewDecision
    duration_sec:           float
    time_gate_passed:       bool
    requires_second_review: bool
    ai_confidence:          float
    duplicate_submission:   bool = False
    message:                str


# ============================================================
# CALIBRATION SCHEMAS
# ============================================================
class CalibrationProfileResponse(BaseModel):
    camera_id:                 str
    profile_version:           int
    calibrated_at:             datetime
    hsv_profiles:              Dict[str, Any]
    reid_thresholds:           Dict[str, Any]
    sample_count:              Optional[int] = None
    is_active:                 bool
    approved:                  bool
    pending_supervisor_review: bool
    anchor_eval_report:        Optional[Dict[str, Any]] = None

    class Config:
        from_attributes = True


class CalibrationJobRequest(BaseModel):
    camera_ids:     Optional[List[str]] = None
    triggered_by:   str = "scheduler"


# ============================================================
# VAULT STATS SCHEMA
# ============================================================
class VaultStatsResponse(BaseModel):
    total_entries:          int
    by_compartment:         Dict[str, int]
    by_entity_type:         Dict[str, int]
    by_lighting:            Dict[str, int]
    training_eligible:      int
    used_in_training:       int
    top_cameras:            List[Dict[str, Any]]
    expiring_soon_crops:    int


# ============================================================
# AUDIT INTEGRITY SCHEMA
# ============================================================
class AuditVerificationResponse(BaseModel):
    verified_count:  int
    broken_links:    List[Dict[str, Any]] = []
    integrity_ok:    bool
    skipped:         bool = False
