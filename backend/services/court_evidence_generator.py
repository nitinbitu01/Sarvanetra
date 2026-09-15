"""
backend/services/court_evidence_generator.py — Section 65B Indian Evidence Act Certificate Engine.
"""
from __future__ import annotations

import hashlib
import hmac
import json
import os
import time
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, Optional, Tuple

import cv2
import numpy as np

WORKSPACE = Path(__file__).resolve().parent.parent.parent
SECRET_SALT = os.getenv("SENTINEL_EVIDENCE_SALT", "GUJARAT_POLICE_NETRAM_SEC65B_SECRET_KEY")


@dataclass
class Section65BCertificate:
    certificate_id: str
    incident_id: str
    camera_id: str
    camera_location: str
    violation_name: str
    statutory_law: str
    fine_inr: int
    timestamp_utc: str
    timestamp_ist: str
    sha256_image_digest: str
    sha256_metadata_digest: str
    hmac_signature: str
    officer_badge_id: str
    terminal_id: str
    court_jurisdiction: str
    evidence_image_path: str


class CourtEvidenceGenerator:
    """
    Generates verified Section 65B evidence collages and cryptographic checksums.
    """

    @classmethod
    def generate_evidence_package(
        cls,
        full_frame: np.ndarray,
        target_box: Tuple[int, int, int, int],
        incident_data: Dict[str, Any],
        out_dir: Optional[Path] = None,
    ) -> Section65BCertificate:
        if out_dir is None:
            out_dir = WORKSPACE / "data" / "evidence" / "court_certificates"
        out_dir.mkdir(parents=True, exist_ok=True)

        now_utc = datetime.now(timezone.utc)
        now_ist = datetime.now()
        timestamp_utc_str = now_utc.isoformat()
        timestamp_ist_str = now_ist.strftime("%d-%m-%Y %H:%M:%S IST")

        incident_id = str(incident_data.get("incident_id", f"INC-{int(time.time()*1000)}"))
        camera_id = str(incident_data.get("camera_id", "CAM-01"))
        camera_loc = str(incident_data.get("camera_location", "Gujarat Police Surveillance Grid"))
        violation_name = str(incident_data.get("violation_name", "TRAFFIC_OFFENSE"))
        statutory_law = str(incident_data.get("statutory_law", "Motor Vehicles Act 1988"))
        fine_inr = int(incident_data.get("fine_inr", 1000))
        officer_badge = str(incident_data.get("officer_badge_id", "NETRAM-INSP-8421"))
        terminal_id = str(incident_data.get("terminal_id", "TRINETRA-ICCC-CONSOLE-04"))
        jurisdiction = str(incident_data.get("court_jurisdiction", "Gujarat Virtual Court / Traffic Judicial Magistrate"))

        h, w = full_frame.shape[:2]
        x1, y1, x2, y2 = target_box
        x1, y1 = max(0, x1), max(0, y1)
        x2, y2 = min(w, x2), min(h, y2)

        # Panel 1: Overview Scene
        p1 = full_frame.copy()
        cv2.rectangle(p1, (x1, y1), (x2, y2), (0, 0, 255), 3)

        # Panel 2: Target Crop
        p2_raw = full_frame[y1:y2, x1:x2]
        if p2_raw.size == 0 or p2_raw.shape[0] < 5 or p2_raw.shape[1] < 5:
            p2_raw = full_frame.copy()
        p2 = cv2.resize(p2_raw, (w // 2, h // 2))

        # Panel 3: Optical Contrast Enhancement
        gray = cv2.cvtColor(p2_raw, cv2.COLOR_BGR2GRAY)
        clahe = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(8, 8))
        enh_gray = clahe.apply(gray)
        p3 = cv2.cvtColor(cv2.resize(enh_gray, (w // 2, h // 2)), cv2.COLOR_GRAY2BGR)

        # Panel 4: Telemetry Card
        p4 = np.zeros((h // 2, w // 2, 3), dtype=np.uint8) + 20
        cv2.rectangle(p4, (10, 10), (w // 2 - 10, h // 2 - 10), (40, 40, 40), -1)
        cv2.putText(p4, "SECTION 65B EVIDENCE CERTIFICATE", (20, 40), cv2.FONT_HERSHEY_SIMPLEX, 0.65, (0, 255, 255), 2)
        cv2.putText(p4, f"INCIDENT ID:  {incident_id}", (20, 80), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (220, 220, 220), 1)
        cv2.putText(p4, f"CAMERA:       {camera_id} ({camera_loc})", (20, 110), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (220, 220, 220), 1)
        cv2.putText(p4, f"TIMESTAMP:    {timestamp_ist_str}", (20, 140), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (220, 220, 220), 1)
        cv2.putText(p4, f"OFFENSE:      {violation_name}", (20, 170), cv2.FONT_HERSHEY_SIMPLEX, 0.52, (0, 0, 255), 2)
        cv2.putText(p4, f"SECTION:      {statutory_law}", (20, 200), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1)
        cv2.putText(p4, f"PENALTY:      Rs. {fine_inr:,}", (20, 230), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0, 255, 255), 2)
        cv2.putText(p4, f"OFFICER:      {officer_badge} | {terminal_id}", (20, 260), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (180, 180, 180), 1)

        top_banner = np.zeros((70, w, 3), dtype=np.uint8) + 15
        cv2.putText(top_banner, "GUJARAT POLICE TRINETRA ICCC | SECTION 65B INDIAN EVIDENCE ACT RECORD", (30, 32), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 255), 2)
        cv2.putText(top_banner, f"Camera: {camera_id} | Location: {camera_loc} | Time: {timestamp_ist_str}", (30, 58), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (200, 200, 200), 1)

        half_w = w // 2
        half_h = h // 2
        p1_scaled = cv2.resize(p1, (w, half_h))
        bottom_row = np.hstack([cv2.resize(p2, (half_w // 2, half_h)), cv2.resize(p3, (half_w // 2, half_h)), cv2.resize(p4, (half_w, half_h))])

        composite = np.vstack([top_banner, p1_scaled, bottom_row])

        _, buffer = cv2.imencode(".jpg", composite, [cv2.IMWRITE_JPEG_QUALITY, 95])
        image_bytes = buffer.tobytes()
        sha256_img = hashlib.sha256(image_bytes).hexdigest()

        meta_dict = {
            "incident_id": incident_id,
            "camera_id": camera_id,
            "timestamp_utc": timestamp_utc_str,
            "violation": violation_name,
            "fine_inr": fine_inr,
            "sha256_image": sha256_img,
        }
        canonical_meta = json.dumps(meta_dict, sort_keys=True)
        sha256_meta = hashlib.sha256(canonical_meta.encode("utf-8")).hexdigest()
        hmac_sig = hmac.new(SECRET_SALT.encode("utf-8"), (sha256_img + sha256_meta).encode("utf-8"), hashlib.sha256).hexdigest()

        cert_id = f"CERT-65B-{sha256_img[:12].upper()}"
        cert_img_path = out_dir / f"{cert_id}.jpg"
        cert_json_path = out_dir / f"{cert_id}.json"

        with open(cert_img_path, "wb") as f:
            f.write(image_bytes)

        cert = Section65BCertificate(
            certificate_id=cert_id,
            incident_id=incident_id,
            camera_id=camera_id,
            camera_location=camera_loc,
            violation_name=violation_name,
            statutory_law=statutory_law,
            fine_inr=fine_inr,
            timestamp_utc=timestamp_utc_str,
            timestamp_ist=timestamp_ist_str,
            sha256_image_digest=sha256_img,
            sha256_metadata_digest=sha256_meta,
            hmac_signature=hmac_sig,
            officer_badge_id=officer_badge,
            terminal_id=terminal_id,
            court_jurisdiction=jurisdiction,
            evidence_image_path=str(cert_img_path),
        )

        with open(cert_json_path, "w") as f:
            json.dump(asdict(cert), f, indent=2)

        return cert
