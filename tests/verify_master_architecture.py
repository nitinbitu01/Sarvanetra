"""
tests/verify_master_architecture.py
Comprehensive End-to-End Verification Suite for Sentinel Gujarat Master Architecture.
"""

import sys
import os
import time
import asyncio
import numpy as np

# Reconfigure stdout for utf-8 on Windows
if sys.platform == "win32":
    try:
        sys.stdout.reconfigure(encoding="utf-8")
    except Exception:
        pass

# Add project root to sys.path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from backend.services.camera_link_model import CameraLinkModel, TravelTimeDistribution
from backend.services.vehicle_reid_engine import VehicleReIDEngine, VehicleColorClassifier, VehicleTypeClassifier
from backend.services.person_reid_engine import PersonReIDEngine, PersonAttributeClassifier
from backend.services.danger_score_engine import DangerScoreEngine, DangerContext, AlertPriority
from backend.services.evidence_vault import EvidenceVault
from backend.services.camera_adapters.factory import CameraAdapterFactory
from backend.services.camera_adapters.onvif_adapter import ONVIFAdapter
from backend.services.dispatch_router import DispatchRouter


async def main():
    print("=" * 80)
    print("[START] STARTING SENTINEL GUJARAT MASTER ARCHITECTURE VERIFICATION")
    print("=" * 80)

    # ─────────────────────────────────────────────────────────────
    # TEST 1: Camera Link Model (CLM)
    # ─────────────────────────────────────────────────────────────
    print("\n[TEST 1] Testing Camera Link Model (CLM) Spatio-Temporal Engine...")
    clm = CameraLinkModel(mode="vehicle")
    
    # 1.1 Seeding camera pair (Cam 1: Ellis Bridge -> Cam 2: Naroda, 10km apart)
    clm.seed_from_topology([
        {"cam_id": 1, "lat": 23.0225, "lon": 72.5714, "connects_to": [2], "road_type": "urban"},
        {"cam_id": 2, "lat": 23.0876, "lon": 72.6461, "connects_to": [1], "road_type": "urban"},
    ])
    
    # 1.2 Check physical violation (10km in 10 seconds = 3600 km/h -> MUST REJECT)
    res_impossible = clm.check_feasibility(cam_a_id=1, cam_b_id=2, time_delta_seconds=10.0)
    assert not res_impossible.is_feasible, "CLM failed to reject impossible travel speed!"
    print(f"  [OK] Correctly rejected physical impossibility: {res_impossible.reject_reason}")

    # 1.3 Self-supervised learning: Train CLM with 15 realistic ANPR trips (mean ~15 mins = 900s)
    for t in [880, 920, 900, 910, 890, 930, 905, 895, 915, 900, 925, 885, 905, 910, 900]:
        clm.learn_from_confirmed_match(cam_a_id=1, cam_b_id=2, travel_time_seconds=t, match_source="anpr")
    
    # 1.4 Test feasible sighting near mean (905 seconds) -> MUST ACCEPT WITH CONFIDENCE BOOST
    res_feasible = clm.check_feasibility(cam_a_id=1, cam_b_id=2, time_delta_seconds=905.0)
    assert res_feasible.is_feasible, "CLM rejected feasible travel time!"
    assert res_feasible.confidence_boost > 0.10, "CLM did not give Gaussian confidence boost!"
    print(f"  [OK] Feasible transition accepted! Boost: +{res_feasible.confidence_boost:.3f}, Expected mean: {res_feasible.expected_mean:.1f}s")

    # ─────────────────────────────────────────────────────────────
    # TEST 2: Vehicle ReID Engine (4-Gate Matcher)
    # ─────────────────────────────────────────────────────────────
    print("\n[TEST 2] Testing Vehicle ReID Engine (ResNet50 + Color + Type + CLM Gate)...")
    veh_engine = VehicleReIDEngine(clm)

    # 2.1 Color and Type Classifier tests
    dummy_crop = np.full((120, 200, 3), (250, 250, 250), dtype=np.uint8) # White SUV
    color, conf = veh_engine.color_clf.classify(dummy_crop)
    vtype, tconf = veh_engine.type_clf.classify(dummy_crop)
    assert color == "White", f"Expected White color, got {color}"
    assert vtype == "SUV/MUV", f"Expected SUV/MUV, got {vtype}"
    print(f"  [OK] Vehicle Classifiers: Color='{color}' (conf={conf:.2f}), Type='{vtype}' (conf={tconf:.2f})")

    # 2.2 Sighting 1 at Cam 1
    t0 = 1700000000.0
    emb1 = veh_engine.extract_embedding(dummy_crop, track_id="trk_1", camera_id=1, timestamp_utc=t0, plate_text=None)
    id1 = veh_engine.add_to_index(emb1)

    # 2.3 Sighting 2 at Cam 2 after 905 seconds (same appearance, muddy plate)
    emb2 = veh_engine.extract_embedding(dummy_crop, track_id="trk_2", camera_id=2, timestamp_utc=t0 + 905.0, plate_text=None)
    matches = veh_engine.search_vehicle(emb2)
    assert len(matches) > 0, "Vehicle ReID failed to match across cameras!"
    assert matches[0].feasibility.is_feasible, "Feasibility check failed on valid vehicle match"
    print(f"  [OK] Cross-camera match found without plate! Score: {matches[0].final_score:.3f} ({matches[0].match_type})")

    # ─────────────────────────────────────────────────────────────
    # TEST 3: Person ReID Engine
    # ─────────────────────────────────────────────────────────────
    print("\n[TEST 3] Testing Person ReID Engine (OSNet + Attributes + Attribute Gating)...")
    pers_engine = PersonReIDEngine(clm)
    person_crop = np.zeros((200, 100, 3), dtype=np.uint8)
    person_crop[:100, :] = [0, 0, 220] # Red shirt
    person_crop[100:, :] = [20, 20, 20] # Black pants
    
    attrs = pers_engine.attr.classify(person_crop)
    assert attrs["upper_color"] == "Red", f"Expected Red upper, got {attrs['upper_color']}"
    assert attrs["lower_color"] == "Black", f"Expected Black lower, got {attrs['lower_color']}"
    print(f"  [OK] Person Attributes: Upper={attrs['upper_color']}, Lower={attrs['lower_color']}, Bag={attrs['carrying_bag']}")

    p_emb = pers_engine.extract_embedding(person_crop, track_id="p_trk_1", camera_id=3, timestamp_utc=t0)
    pid = pers_engine.add_to_index(p_emb)
    assert pid.startswith("PERS_3_"), "Person ReID ID format incorrect"
    print(f"  [OK] Person indexed: {pid}")

    # ─────────────────────────────────────────────────────────────
    # TEST 4: Composite Danger Score Engine
    # ─────────────────────────────────────────────────────────────
    print("\n[TEST 4] Testing Composite Danger Score Engine...")
    danger_engine = DangerScoreEngine()

    # 4.1 Terrorist watchlist -> ALWAYS CRITICAL (1.0)
    ctx_terror = DangerContext(is_on_watchlist=True, watchlist_severity="TERRORIST")
    score_terror = danger_engine.compute(ctx_terror)
    assert score_terror.priority == AlertPriority.CRITICAL and score_terror.total == 1.0
    print(f"  [OK] Terrorist guard rail verified: Score={score_terror.total}, Priority={score_terror.priority.name}")

    # 4.2 Multi-factor Stolen Vehicle in High Risk Zone at Night
    ctx_stolen = DangerContext(
        is_stolen_vehicle=True,
        zone_risk_level=3,
        num_cameras_in_30min=4,
        match_type="plate_confirmed"
    )
    score_stolen = danger_engine.compute(ctx_stolen)
    assert score_stolen.total >= 0.70, f"Expected high score for stolen vehicle in risk zone, got {score_stolen.total}"
    print(f"  [OK] Stolen vehicle threat score: {score_stolen.total} ({score_stolen.priority.name}) | {score_stolen.explanation}")

    # ─────────────────────────────────────────────────────────────
    # TEST 5: Court Evidence Vault
    # ─────────────────────────────────────────────────────────────
    print("\n[TEST 5] Testing Court Evidence Vault (SHA-256 + HMAC + Integrity Verification)...")
    vault = EvidenceVault()
    sample_frames = [np.full((240, 320, 3), i * 10, dtype=np.uint8) for i in range(15)]
    
    # 5.1 Seal evidence
    rec = await vault.seal_evidence(
        alert_id="ALT_TEST_881",
        camera_id=42,
        frames=sample_frames,
        fps=15.0,
        camera_location="Ellis Bridge Junction"
    )
    assert os.path.exists(rec.storage_path), "Evidence video file not created on disk"
    assert len(rec.sha256_hash) == 64, "Invalid SHA-256 hash length"
    print(f"  [OK] Evidence Sealed: ID={rec.evidence_id} | SHA-256={rec.sha256_hash[:16]}... | HMAC={rec.hmac_signature[:16]}...")

    # 5.2 Verify integrity (Court Admissibility Proof)
    valid, msg = await vault.verify_integrity(
        evidence_id=rec.evidence_id,
        storage_path=rec.storage_path,
        expected_sha256=rec.sha256_hash,
        expected_hmac=rec.hmac_signature
    )
    assert valid, f"Integrity verification failed: {msg}"
    print(f"  [OK] Court Verification Succeeded: {msg}")

    # ─────────────────────────────────────────────────────────────
    # TEST 6: Universal Camera Adapters & Factory
    # ─────────────────────────────────────────────────────────────
    print("\n[TEST 6] Testing Universal Camera Adapters & Factory...")
    # 6.1 Supported adapters
    assert "cpplus" in CameraAdapterFactory.FALLBACK_CHAINS
    assert "hikvision" in CameraAdapterFactory.FALLBACK_CHAINS
    print(f"  [OK] Fallback chains registered for {len(CameraAdapterFactory.FALLBACK_CHAINS)} vendor families")

    # 6.2 ONVIF Discovery
    devices = await ONVIFAdapter.discover_devices()
    assert len(devices) > 0, "ONVIF discovery returned 0 devices"
    print(f"  [OK] ONVIF Discovery Probe discovered {len(devices)} cameras (Hikvision, Dahua, CP Plus)")

    # ─────────────────────────────────────────────────────────────
    # TEST 7: GPS Dispatch Router
    # ─────────────────────────────────────────────────────────────
    print("\n[TEST 7] Testing GPS Nearest Officer Dispatch Router...")
    router = DispatchRouter()
    dispatch_res = await router.dispatch_alert(
        alert_id="ALT_TEST_881",
        danger_score=score_stolen,
        camera_lat=23.0395,
        camera_lon=72.5797
    )
    assert dispatch_res.dispatched_to is not None, "Failed to dispatch nearest officer"
    print(f"  [OK] Dispatched to {dispatch_res.dispatched_to.name} ({dispatch_res.dispatched_to.badge_number}) | Dist: {dispatch_res.distance_km}km | ETA: {dispatch_res.estimated_arrival_minutes:.0f}min")

    print("\n" + "=" * 80)
    print("[SUCCESS] ALL 7 MASTER ARCHITECTURE COMPONENTS PASSED VERIFICATION WITH 100% SUCCESS!")
    print("=" * 80)


if __name__ == "__main__":
    asyncio.run(main())
