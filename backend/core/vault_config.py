"""
backend/core/vault_config.py — Configuration, enums, and constants for
the Active Learning Vault and Human-in-the-Loop (HITL) Officer Feedback Flywheel.
"""
from enum import Enum
from typing import Dict, Tuple

class VaultCompartment(str, Enum):
    GOLD = "gold"                      # Hand-verified reference identities (retained indefinitely)
    HARD_POSITIVE = "hard_positive"    # Confirmed matches with low AI confidence (retained for retraining)
    HARD_NEGATIVE = "hard_negative"    # False positives from officers (negative mining)
    ANCHOR = "anchor"                  # Frozen benchmark identity sets for regression testing
    PROBATIONARY = "probationary"      # Single unverified officer feedback before consensus

class LabelTrust(str, Enum):
    GOLD = "gold"                      # Multi-officer consensus or supervisor verified
    HIGH = "high"                      # 2 agreeing officers in double-review
    MEDIUM = "medium"                  # Single officer review outside ambiguous band
    PROBATIONARY = "probationary"      # Needs second review or verification

class ReviewDecision(str, Enum):
    CONFIRM_MATCH = "confirm_match"
    FALSE_POSITIVE = "false_positive"
    UNCERTAIN = "uncertain"

class EntityType(str, Enum):
    VEHICLE = "vehicle"
    PERSON = "person"

class LightingCondition(str, Enum):
    DAY = "day"
    NIGHT = "night"
    NIGHT_GLARE = "night_glare"
    MONSOON_RAIN = "monsoon_rain"
    DUST_STORM = "dust_storm"

# Server-authoritative time gates (seconds). Submissions under these durations are rejected.
REVIEW_TIME_GATES: Dict[str, float] = {
    "vehicle_reid": 4.0,
    "person_reid": 5.0,
    "critical_alert": 6.0,
    "default": 4.0,
}

# Ambiguous confidence band triggering double-review
DOUBLE_REVIEW_BAND: Tuple[float, float] = (0.65, 0.85)

# Minimum number of anchor identity groups required to validate a camera calibration profile
MIN_ANCHOR_GROUPS = 5

# Sampling ratios for stratified active learning retraining dataset construction
VAULT_SAMPLING_RATIOS = {
    VaultCompartment.GOLD: 0.20,
    VaultCompartment.HARD_POSITIVE: 0.40,
    VaultCompartment.HARD_NEGATIVE: 0.30,
    VaultCompartment.ANCHOR: 0.10,
}

# Maximum percentage of retraining samples allowed from a single camera (prevents camera overfitting)
VAULT_MAX_CAMERA_PERCENTAGE = 0.15

# MinIO Vault storage paths
MINIO_VAULT_BUCKET = "sentinel-evidence"
MINIO_VAULT_PATHS = {
    VaultCompartment.GOLD: "vault/gold/{year}/{month}/{entity_type}/{uuid}.jpg",
    VaultCompartment.HARD_POSITIVE: "vault/hard_positive/{year}/{month}/{entity_type}/{uuid}.jpg",
    VaultCompartment.HARD_NEGATIVE: "vault/hard_negative/{year}/{month}/{entity_type}/{uuid}.jpg",
    VaultCompartment.ANCHOR: "vault/anchor/{entity_type}/{uuid}.jpg",
    VaultCompartment.PROBATIONARY: "vault/probationary/{year}/{month}/{entity_type}/{uuid}.jpg",
}

# Retention policies in days
RETENTION_POLICIES = {
    "crop_default_days": 90,
    "embedding_days": 180,
    "probationary_days": 14,
}

# Kafka / Redis Stream topic names
KAFKA_TOPIC_REVIEW_COMPLETED = "sentinel.review.completed"
KAFKA_TOPIC_CALIBRATION_TRIGGERED = "sentinel.calibration.triggered"
KAFKA_TOPIC_AUDIT_INTEGRITY = "sentinel.audit.integrity"
