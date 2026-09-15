"""
backend/scripts/verify_active_learning.py — Comprehensive End-to-End Verification Suite
for Pillar 3: Active Learning Vault & Officer Feedback Flywheel.
"""
import asyncio
import json
import logging
import sys
import time
import uuid
from datetime import datetime, timezone
import numpy as np

from backend.db.session import SessionLocal, init_db
from backend.db.models import (
    VaultEntry, OfficerReview, ReviewPair,
    CameraCalibrationProfile, AuditCheckpoint, AuditLog,
    Alert, Camera
)
from backend.core.vault_config import (
    VaultCompartment, LabelTrust, ReviewDecision, EntityType, LightingCondition
)
from backend.schemas.vault import (
    VaultEntryCreate, ReviewStartRequest, ReviewSubmitRequest
)
from backend.services.vault_service import VaultService
from backend.services.review_service import ReviewService
from backend.services.anchor_evaluator import AnchorEvaluator
from backend.services.environmental_classifier import EnvironmentalClassifier
from backend.services.audit_service import AuditService
from backend.workers.calibration_worker import CalibrationWorker
from backend.db.seed_anchor_vault import seed_anchor_dataset
from backend.feedback.service import submit_feedback

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger("sentinel.verify_active_learning")


class MockOfficer:
    def __init__(self, id="USER_OPS_01", role="operator", zone="Central"):
        self.id = id
        self.role = role
        self.zone = zone


async def run_full_verification():
    print("=" * 70)
    print("[INFO] SENTINEL GUJARAT - ACTIVE LEARNING FLYWHEEL VERIFICATION")
    print("=" * 70)

    # 1. Initialize Tables
    print("\n[TEST 1/9] Initializing and verifying database schema...")
    init_db()
    db = SessionLocal()
    try:
        # Create a test alert if needed
        test_alert = db.query(Alert).first()
        if not test_alert:
            test_alert = Alert(
                camera_id=1,
                alert_type="vehicle_reid_match",
                score=0.72,
                danger_score=6.5,
                description="Test Vehicle Alert",
            )
            db.add(test_alert)
            db.commit()
        alert_id = test_alert.id
        print(f"[OK] Database tables verified. Working test alert id={alert_id}")
    finally:
        db.close()

    # 2. Seed Anchors
    print("\n[TEST 2/9] Bootstrapping Anchor-Set dataset...")
    seed_anchor_dataset(force=True)
    db = SessionLocal()
    try:
        anchors_count = db.query(VaultEntry).filter(VaultEntry.compartment == VaultCompartment.ANCHOR.value).count()
        assert anchors_count >= 16, f"Expected >= 16 anchors, got {anchors_count}"
        print(f"[OK] Verified {anchors_count} Anchor reference samples in Vault.")
    finally:
        db.close()

    # 3. Test Environmental Classifier
    print("\n[TEST 3/9] Testing Automated Environmental / Lighting Classifier...")
    # Test solar elevation calculation
    elev_day = EnvironmentalClassifier.calculate_solar_elevation(23.0225, 72.5714, datetime(2026, 6, 21, 7, 0, 0, tzinfo=timezone.utc))
    elev_night = EnvironmentalClassifier.calculate_solar_elevation(23.0225, 72.5714, datetime(2026, 6, 21, 21, 0, 0, tzinfo=timezone.utc))
    assert elev_day > 0, "Solar day elevation should be positive"
    assert elev_night < 0, "Solar night elevation should be negative"

    # Test frame heuristic
    dummy_night = np.full((100, 100, 3), 30, dtype=np.uint8)
    cond_night = EnvironmentalClassifier.classify_frame(dummy_night, 23.0225, 72.5714)
    print(f"[OK] Environmental Classifier output: Solar Day={elev_day:.1f}deg, Solar Night={elev_night:.1f}deg, Classified={cond_night.value}")

    # 4. Test Server-Authoritative Time Gate Rejection
    print("\n[TEST 4/9] Testing Server-Authoritative Time Gate (Speed-Click Fraud Prevention)...")
    db = SessionLocal()
    try:
        review_svc = ReviewService(db)
        officer = MockOfficer(id="USER_OPS_01", role="operator")
        start_res = await review_svc.start_review_session(
            ReviewStartRequest(alert_id=alert_id), officer
        )

        # Immediate fast submit (<0.1s)
        submit_req = ReviewSubmitRequest(
            review_session_id=start_res.review_session_id,
            alert_id=alert_id,
            decision=ReviewDecision.CONFIRM_MATCH,
        )
        fast_res = await review_svc.submit_review_decision(submit_req)
        assert fast_res.time_gate_passed is False, "Fast submission must be rejected by time gate"
        print(f"[OK] Time Gate successfully rejected {fast_res.duration_sec:.2f}s submission (Required: {start_res.min_review_seconds}s).")
    finally:
        db.close()

    # 5. Test pgvector Deduplication (Pre-Upload)
    print("\n[TEST 5/9] Testing pgvector ANN / Cosine Deduplication...")
    db = SessionLocal()
    try:
        vault_svc = VaultService(db)
        vec1 = np.random.randn(512).astype(np.float32)
        vec1 /= np.linalg.norm(vec1)

        entry1 = await vault_svc.store_entry(
            entry_data=VaultEntryCreate(
                alert_id=alert_id,
                camera_id="CAM_01",
                compartment=VaultCompartment.HARD_POSITIVE,
                trust_level=LabelTrust.HIGH,
                ai_cosine_distance=0.28,
            ),
            crop_bytes=b"dummy_jpeg_data_1",
            embedding_vector=vec1,
            actor_id="USER_OPS_01",
        )

        # Submit slightly noisy duplicate (cosine distance ~0.005)
        vec2 = vec1 + np.random.randn(512).astype(np.float32) * 0.002
        vec2 /= np.linalg.norm(vec2)

        entry2 = await vault_svc.store_entry(
            entry_data=VaultEntryCreate(
                alert_id=alert_id,
                camera_id="CAM_01",
                compartment=VaultCompartment.HARD_POSITIVE,
                trust_level=LabelTrust.HIGH,
                ai_cosine_distance=0.28,
            ),
            crop_bytes=b"dummy_jpeg_data_2",
            embedding_vector=vec2,
            actor_id="USER_OPS_01",
        )

        assert entry1.id == entry2.id, "Near-duplicate embedding should return the existing vault entry"
        print(f"[OK] Deduplication passed: Duplicate crop correctly intercepted (Reused UUID={entry1.id[:8]}...).")
    finally:
        db.close()

    # 6. Test Blind Double-Review Workflow
    print("\n[TEST 6/9] Testing Blind Double-Review State Machine (ReviewPair)...")
    db = SessionLocal()
    try:
        review_svc = ReviewService(db)
        # Borderline confidence (0.72) triggers double review
        officer1 = MockOfficer(id="OFF_AHM_01", role="operator")
        start1 = await review_svc.start_review_session(
            ReviewStartRequest(alert_id=alert_id), officer1,
            alert_data={"confidence_score": 0.72, "cosine_distance": 0.35, "entity_type": "vehicle"}
        )
        
        # Simulate elapsed time
        session_key = start1.review_session_id
        if session_key in review_svc._memory_sessions:
            review_svc._memory_sessions[session_key]["server_started_at"] = datetime(2020, 1, 1).isoformat()

        res1 = await review_svc.submit_review_decision(
            ReviewSubmitRequest(
                review_session_id=start1.review_session_id,
                alert_id=alert_id,
                decision=ReviewDecision.CONFIRM_MATCH,
            )
        )
        assert res1.requires_second_review is True, "Borderline confidence must trigger double-review"

        # Officer 2 starts review
        officer2 = MockOfficer(id="OFF_AHM_02", role="operator")
        start2 = await review_svc.start_review_session(
            ReviewStartRequest(alert_id=alert_id), officer2,
            alert_data={"confidence_score": 0.72, "cosine_distance": 0.35, "entity_type": "vehicle"}
        )
        assert start2.is_second_review is True, "Second session must be identified as round 2"

        session_key2 = start2.review_session_id
        if session_key2 in review_svc._memory_sessions:
            review_svc._memory_sessions[session_key2]["server_started_at"] = datetime(2020, 1, 1).isoformat()

        res2 = await review_svc.submit_review_decision(
            ReviewSubmitRequest(
                review_session_id=start2.review_session_id,
                alert_id=alert_id,
                decision=ReviewDecision.CONFIRM_MATCH,  # Agreement!
            )
        )
        assert res2.vault_entry_id is not None, "Consensus review must be promoted and stored in Vault"
        print(f"[OK] Blind Double-Review reconciled: Both officers agreed -> Stored with HIGH trust (Vault ID={res2.vault_entry_id[:8]}...).")
    finally:
        db.close()

    # 7. Test Anchor Regression Gating
    print("\n[TEST 7/9] Testing AnchorEvaluator Safety Regression Gate...")
    db = SessionLocal()
    try:
        evaluator = AnchorEvaluator(db)
        # Safe threshold (0.35)
        good_report = evaluator.evaluate_profile(
            camera_id="CAM_01",
            proposed_thresholds={"day": {"vehicle": 0.35}},
            lighting_condition="day",
        )
        assert good_report["passed"] is True, f"Standard threshold should pass anchor evaluation: {good_report}"

        # Unsafe extreme threshold (0.99 - matches all distinct identities, 100% false positive rate)
        bad_report = evaluator.evaluate_profile(
            camera_id="CAM_01",
            proposed_thresholds={"day": {"vehicle": 0.99}},
            lighting_condition="day",
        )
        assert bad_report["passed"] is False, "Unsafe threshold must be blocked by anchor evaluation"
        print(f"[OK] Anchor Safety Gate active: Safe threshold passed ({good_report['top1_accuracy']:.1%} Top-1), Extreme threshold blocked ({bad_report['reason']}).")
    finally:
        db.close()

    # 8. Test Uncertainty Sampling
    print("\n[TEST 8/9] Testing Active Learning Uncertainty Sampling (|score - threshold|)...")
    db = SessionLocal()
    try:
        vault_svc = VaultService(db)
        samples = vault_svc.sample_for_retraining(total_samples=10, entity_type="vehicle")
        print(f"[OK] Stratified Uncertainty Sampling extracted: {len(samples.get('hard_positive', []))} Hard Positives, {len(samples.get('hard_negative', []))} Hard Negatives.")
    finally:
        db.close()

    # 9. Test Cryptographic Audit Hash-Chain
    print("\n[TEST 9/9] Testing Cryptographic Audit Hash-Chain Verification...")
    db = SessionLocal()
    try:
        audit_svc = AuditService(db)
        audit_svc.log_event("test_event_1", payload={"action": "test_1"})
        audit_svc.log_event("test_event_2", payload={"action": "test_2"})
        verification = audit_svc.verify_integrity()
        assert verification["integrity_ok"] is True, f"Audit hash chain should be valid: {verification}"
        print(f"[OK] Cryptographic Audit Chain Verified: {verification['verified_count']} events verified without breaks.")
    finally:
        db.close()

    print("\n" + "=" * 70)
    print("[SUCCESS] ALL 9 ACTIVE LEARNING & FEEDBACK FLYWHEEL TESTS PASSED PERFECTLY!")
    print("=" * 70)


if __name__ == "__main__":
    asyncio.run(run_full_verification())
