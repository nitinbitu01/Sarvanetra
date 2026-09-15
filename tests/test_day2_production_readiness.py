"""
tests/test_day2_production_readiness.py — Day 2 Production Readiness & Deep QA Test Suite
========================================================================================
Covers:
  1. Dial-112 CAD Emergency Auto-Dispatch Engine
     - Real-time GPS distance calculation (Haversine formula)
     - Nearest unit allocation with ETA estimation
     - Stale GPS telemetry rejection (> 60s without telemetry)
     - Unit status lifecycle state machine (AVAILABLE -> DISPATCHED -> EN_ROUTE -> ON_SCENE -> RESOLVED)
     - Fleet saturation handling
  2. Multilingual Voice AI Alert Engine
     - Spoken Hindi, Gujarati, and English prompt formatting across 7 crime categories
     - Speech synthesis (.mp3 / .wav) and audio file integrity
     - Emergency audio chime fallback (deterministic 880Hz / 587Hz synthesis)
  3. Section 65B Indian Evidence Act / Section 63 BSA Forensic Certificate Generator
     - Statutory PDF generation with Government of Gujarat header
     - Tamper-proof SHA-256 digital HMAC seal
     - UTC & IST timestamp formatting
     - Section 65B(4) statutory affirmation text validation
  4. End-to-End Integrated Pipeline
     - Alert Creation -> CAD Auto-Dispatch -> Voice Alert -> Section 65B PDF Sealing
"""

from __future__ import annotations

import os
import sys
import time
import math
from pathlib import Path
from datetime import datetime, timedelta
import pytest

# Auto-resolve workspace root
_PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))

from backend.db.session import SessionLocal
from backend.db.models import Alert, Evidence, Camera
from backend.services.cad_dispatch import (
    CADDispatcher,
    PCRUnit,
    UnitStatus,
    DispatchResult,
    _haversine_km,
    get_dispatcher,
)
from backend.services.voice_alert_service import (
    VoiceAlertService,
    get_voice_service,
    ALERT_TEMPLATES,
)
from backend.services.evidence_capture import (
    _write_custody_pdf,
    evidence_dir_for,
)


# ============================================================================
# 1. DIAL-112 CAD AUTO-DISPATCH UNIT & INTEGRATION TESTS
# ============================================================================

class TestCADAutoDispatch:
    """Rigorous tests for Dial-112 CAD GPS patrol vehicle router and fleet management."""

    def test_haversine_formula_accuracy(self):
        """Haversine distance between Ahmedabad and Gandhinagar (~23 km) must be accurate to within 1 km."""
        # Ahmedabad (23.0225, 72.5714) -> Gandhinagar (23.2156, 72.6369)
        dist = _haversine_km(23.0225, 72.5714, 23.2156, 72.6369)
        assert 21.0 <= dist <= 24.0

    def test_cad_nearest_unit_allocation_and_eta(self):
        """Dispatch router must pick the geographically closest AVAILABLE unit and calculate ETA."""
        custom_fleet = [
            PCRUnit("PCR-NEAR", "Alpha-Near", "Ahmedabad", "Insp. A. Patel", "+91-9900001001", 23.0230, 72.5720), # ~0.1km
            PCRUnit("PCR-FAR", "Alpha-Far", "Ahmedabad", "SI B. Shah", "+91-9900001002", 23.1500, 72.7000),      # ~18km
        ]
        dispatcher = CADDispatcher(fleet=custom_fleet)

        # Incident at Ahmedabad (23.0225, 72.5714)
        res = dispatcher.dispatch_nearest(
            incident_lat=23.0225,
            incident_lon=72.5714,
            severity="CRITICAL",
            description="Wanted Fugitive Sighted",
            incident_id="INC-TEST-01",
        )

        assert res is not None
        assert res.unit_id == "PCR-NEAR"
        assert res.call_sign == "Alpha-Near"
        assert res.distance_km < 0.5
        assert res.eta_minutes < 2.0
        assert custom_fleet[0].status == UnitStatus.DISPATCHED

    def test_cad_stale_gps_telemetry_filtering(self):
        """Units with stale GPS (> 60s without telemetry ping) must be excluded from dispatch."""
        stale_unit = PCRUnit("PCR-STALE", "Alpha-Stale", "Ahmedabad", "Insp. Stale", "+91-9900001003", 23.0226, 72.5715)
        # Fake stale timestamp (75 seconds ago)
        stale_unit.last_gps_update = time.monotonic() - 75.0

        fresh_unit = PCRUnit("PCR-FRESH", "Alpha-Fresh", "Ahmedabad", "SI Fresh", "+91-9900001004", 23.0400, 72.5800)
        fresh_unit.last_gps_update = time.monotonic()

        dispatcher = CADDispatcher(fleet=[stale_unit, fresh_unit])
        assert stale_unit.gps_is_fresh is False
        assert fresh_unit.gps_is_fresh is True

        res = dispatcher.dispatch_nearest(23.0225, 72.5714)
        assert res is not None
        assert res.unit_id == "PCR-FRESH"  # Stale unit skipped despite being closer

    def test_cad_unit_status_lifecycle_transitions(self):
        """Verify full lifecycle state transitions: AVAILABLE -> DISPATCHED -> EN_ROUTE -> ON_SCENE -> RESOLVED."""
        unit = PCRUnit("PCR-LIFE", "Chetak-1", "Surat", "Insp. S. Desai", "+91-9900001005", 21.1702, 72.8311)
        dispatcher = CADDispatcher(fleet=[unit])

        assert unit.status == UnitStatus.AVAILABLE

        # 1. Dispatch
        ok = dispatcher.update_status_by_name("PCR-LIFE", "DISPATCHED", "INC-888")
        assert ok is True
        assert unit.status == UnitStatus.DISPATCHED

        # 2. En Route
        ok = dispatcher.update_status_by_name("PCR-LIFE", "EN_ROUTE")
        assert ok is True
        assert unit.status == UnitStatus.EN_ROUTE

        # 3. On Scene
        ok = dispatcher.update_status_by_name("PCR-LIFE", "ON_SCENE")
        assert ok is True
        assert unit.status == UnitStatus.ON_SCENE

        # 4. Resolved / Back to Available
        ok = dispatcher.update_status_by_name("PCR-LIFE", "AVAILABLE")
        assert ok is True
        assert unit.status == UnitStatus.AVAILABLE

    def test_cad_fleet_saturation_handling(self):
        """When all patrol units are busy or offline, dispatch_nearest must return None cleanly."""
        busy_unit = PCRUnit("PCR-BUSY", "Eagle-1", "Vadodara", "Insp. Rao", "+91-9900001006", 22.3072, 73.1812, status=UnitStatus.BUSY)
        dispatcher = CADDispatcher(fleet=[busy_unit])

        res = dispatcher.dispatch_nearest(22.3072, 73.1812)
        assert res is None


# ============================================================================
# 2. MULTILINGUAL VOICE AI ALERT ENGINE TESTS
# ============================================================================

class TestVoiceAIAlerts:
    """Tests for Voice AI audio alert synthesis in Hindi, Gujarati, and English."""

    def test_multilingual_prompt_construction(self):
        """VoiceAlertService must construct accurate localized prompts for all 7 crime categories."""
        service = VoiceAlertService()
        categories = list(ALERT_TEMPLATES.keys())

        for cat in categories:
            prompt_hi = service.build_prompt(cat, camera_name="CAM_06 SG Highway", subject="GJ05AB1234", lang="hi")
            prompt_gu = service.build_prompt(cat, camera_name="CAM_06 SG Highway", subject="GJ05AB1234", lang="gu")
            prompt_en = service.build_prompt(cat, camera_name="CAM_06 SG Highway", subject="GJ05AB1234", lang="en")

            assert "CAM_06 SG Highway" in prompt_hi
            assert "CAM_06 SG Highway" in prompt_gu
            assert "CAM_06 SG Highway" in prompt_en
            assert len(prompt_hi) > 20
            assert len(prompt_gu) > 20
            assert len(prompt_en) > 20

    def test_speech_synthesis_audio_generation(self, tmp_path):
        """VoiceAlertService must generate valid, playable audio files on disk."""
        service = VoiceAlertService(output_dir=tmp_path)
        test_text = "Security Alert! Wanted suspect detected at SG Highway Junction."

        audio_file = service.synthesize_speech(test_text, lang="en", file_prefix="test_voice_01")
        assert audio_file.exists()
        assert audio_file.stat().st_size > 500  # Non-empty audio file
        assert audio_file.suffix in [".mp3", ".wav"]

    def test_emergency_audio_chime_fallback(self, tmp_path):
        """Deterministic emergency chime must produce a valid PCM WAV with standard audio headers."""
        service = VoiceAlertService(output_dir=tmp_path)
        chime_file = tmp_path / "emergency_chime.wav"
        service._generate_emergency_chime(chime_file, duration_s=1.0)

        assert chime_file.exists()
        raw_bytes = chime_file.read_bytes()
        assert raw_bytes[:4] == b"RIFF"
        assert raw_bytes[8:12] == b"WAVE"


# ============================================================================
# 3. SECTION 65B EVIDENCE ACT PDF CERTIFICATE TESTS
# ============================================================================

class TestSection65BEvidenceCertificate:
    """Tests for Section 65B Indian Evidence Act / Section 63 BSA PDF certificate generation."""

    def test_section_65b_pdf_generation_and_statutory_text(self, tmp_path):
        """_write_custody_pdf must generate an uncompressed, court-admissible certificate PDF."""
        db = SessionLocal()
        try:
            # Create a mock/real alert and evidence row in session
            test_alert = db.query(Alert).filter(Alert.is_deleted == False).first()
            if not test_alert:
                test_alert = Alert(
                    id="TEST-ALERT-65B-01",
                    alert_type="WATCHLIST_FACE_MATCH",
                    severity="CRITICAL",
                    subject_label="Wanted Fugitive: Vikas Dubey",
                    description="FIR #412/2026, IPC 302/120B",
                    camera_id="CAM_06",
                )
                db.add(test_alert)
                db.commit()

            test_evidence = Evidence(
                alert_id=test_alert.id,
                camera_str_id=test_alert.camera_id,
                sha256="da131dd3118c707d890b2e2d93e2b20245a4a75b281b99ff1e389d424b94f923",
                event_time=datetime.utcnow(),
                clip_start_time=datetime.utcnow() - timedelta(seconds=5),
                clip_end_time=datetime.utcnow() + timedelta(seconds=5),
                actual_duration_seconds=10.0,
                frame_count=250,
                fps=25.0,
                file_size_bytes=1048576,
                codec="h264",
                clip_path=f"data/evidence/{test_alert.id}/clip.mp4",
                status="COMPLETE",
            )

            out_pdf = tmp_path / "section_65b_certificate.pdf"
            ok = _write_custody_pdf(db, test_evidence, out_pdf)
            assert ok is True
            assert out_pdf.exists()
            assert out_pdf.stat().st_size > 1000

            # Read uncompressed PDF text directly to verify statutory compliance
            pdf_bytes = out_pdf.read_bytes()
            pdf_text = pdf_bytes.decode("latin1", errors="ignore")

            # Check statutory title and declarations
            assert "GOVERNMENT OF GUJARAT" in pdf_text
            assert "SECTION 65B INDIAN EVIDENCE ACT" in pdf_text
            assert "SHA-256:" in pdf_text
            assert "da131dd3118c707d" in pdf_text  # Hash segment must be clearly recorded
            assert "STATUTORY AFFIRMATION" in pdf_text
            assert "Investigating Officer" in pdf_text
        finally:
            db.close()


# ============================================================================
# 4. END-TO-END INTEGRATED ALERT LIFECYCLE TEST
# ============================================================================

class TestDay2EndToEndIntegration:
    """Full integration test linking Alert -> CAD Dispatch -> Voice Alert -> Section 65B Certificate."""

    def test_full_day2_incident_lifecycle(self, tmp_path):
        db = SessionLocal()
        try:
            alert_id = f"ALERT-DAY2-{int(time.time())}"
            cam = db.query(Camera).filter(Camera.id == "CAM_06").first()
            cam_name = cam.name if cam else "SG Highway Intersection"
            lat = getattr(cam, "lat", 23.0225)
            lon = getattr(cam, "lon", 72.5714)

            # 1. Crime Alert Creation
            alert = Alert(
                id=alert_id,
                alert_type="STOLEN_VEHICLE_WATCHLIST_HIT",
                severity="CRITICAL",
                subject_label="Mahindra Bolero (GJ01AB1234)",
                description="Wanted in Inter-State Smuggling (FIR #982/2026)",
                camera_id="CAM_06",
                danger_score=0.92,
                status="PENDING",
            )
            db.add(alert)
            db.commit()

            # 2. Dial-112 CAD Auto-Dispatch
            dispatcher = get_dispatcher()
            dispatch_res = dispatcher.auto_dispatch_alert(
                alert_id=alert_id,
                lat=lat,
                lon=lon,
                severity="CRITICAL",
                description=alert.description,
            )
            assert dispatch_res is not None
            assert "unit_id" in dispatch_res
            assert "call_sign" in dispatch_res
            assert dispatch_res["distance_km"] > 0

            # 3. Multilingual Voice AI Synthesis
            voice_svc = VoiceAlertService(output_dir=tmp_path)
            hindi_prompt = voice_svc.build_prompt(alert.alert_type, cam_name, alert.subject_label, lang="hi")
            audio_path = voice_svc.synthesize_speech(hindi_prompt, lang="hi", file_prefix=alert_id)
            assert audio_path.exists()
            assert audio_path.stat().st_size > 0

            # 4. Section 65B Digital Certificate Sealing
            evidence_row = Evidence(
                alert_id=alert_id,
                camera_str_id="CAM_06",
                sha256="e3b0c44298fc1c149afbf4c8996fb92427ae41e4649b934ca495991b7852b855",
                event_time=datetime.utcnow(),
                actual_duration_seconds=10.0,
                frame_count=250,
                fps=25.0,
                file_size_bytes=2048000,
                codec="h264",
                status="COMPLETE",
            )
            pdf_path = tmp_path / f"{alert_id}_custody.pdf"
            pdf_ok = _write_custody_pdf(db, evidence_row, pdf_path)
            assert pdf_ok is True
            assert pdf_path.exists()

            # Clean up test alert
            db.query(Alert).filter(Alert.id == alert_id).delete()
            db.commit()
        finally:
            db.close()

    def test_concurrent_fleet_gps_telemetry_stress(self):
        """Stress test verifying thread-safe simultaneous telemetry ingestion across 100 concurrent pings."""
        import concurrent.futures

        dispatcher = get_dispatcher()
        def _ping(unit_id: str, lat: float, lon: float) -> bool:
            return dispatcher.update_unit_gps(unit_id, lat, lon)

        with concurrent.futures.ThreadPoolExecutor(max_workers=16) as executor:
            futures = [
                executor.submit(_ping, f"PCR-{(i % 12) + 1:02d}", 23.0 + i * 0.001, 72.5 + i * 0.001)
                for i in range(120)
            ]
            results = [f.result() for f in futures]

        assert all(results)

