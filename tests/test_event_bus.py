import sys
import time
from pathlib import Path

WORKSPACE = Path(__file__).resolve().parent.parent
if str(WORKSPACE) not in sys.path:
    sys.path.insert(0, str(WORKSPACE))

import numpy as np
from backend.services.event_bus import AsyncEventBus
from backend.services.court_evidence_generator import CourtEvidenceGenerator, Section65BCertificate


def test_event_bus_lifecycle():
    bus = AsyncEventBus.get_instance()
    bus.start()
    
    received_events = []
    
    def on_stolen_vehicle(payload):
        received_events.append(payload)
        
    bus.subscribe("STOLEN_VEHICLE_ALERT", on_stolen_vehicle)
    
    success = bus.publish("STOLEN_VEHICLE_ALERT", {"plate": "GJ01AB1234", "camera": "CAM-04"}, priority=1)
    assert success is True
    
    time.sleep(0.5)
    assert len(received_events) == 1
    assert received_events[0]["plate"] == "GJ01AB1234"
    
    stats = bus.get_stats()
    assert stats["events_published"] >= 1
    assert stats["events_processed"] >= 1
    
    bus.stop()


def test_court_evidence_generator():
    dummy_frame = np.zeros((720, 1280, 3), dtype=np.uint8) + 100
    box = (100, 100, 400, 400)
    incident = {
        "incident_id": "INC-TEST-999",
        "camera_id": "CAM-08",
        "camera_location": "Majewadi Gate, Junagadh",
        "violation_name": "TRIPLE_RIDING",
        "statutory_law": "MV Act Sec 128",
        "fine_inr": 2000,
    }
    
    cert = CourtEvidenceGenerator.generate_evidence_package(
        full_frame=dummy_frame,
        target_box=box,
        incident_data=incident,
    )
    
    assert isinstance(cert, Section65BCertificate)
    assert cert.incident_id == "INC-TEST-999"
    assert len(cert.sha256_image_digest) == 64
    assert len(cert.hmac_signature) == 64
    assert Path(cert.evidence_image_path).exists()
