import asyncio
import json
import logging
import os
import uuid
from datetime import datetime
from pathlib import Path
from typing import Dict, Optional

import cv2
import numpy as np

from backend.security.encryption import compute_sha256, encrypt_file

logger = logging.getLogger("sentinel.evidence_locker")


class EvidenceLocker:
    """
    Evidence Locker compliant with Indian Evidence Act:
    1. Saves original or AES-256 encrypted snapshot
    2. Computes SHA-256 hash
    3. Generates Chain of Custody JSON record
    """

    def __init__(self, config: Optional[dict] = None):
        self.config = config or {}
        evidence_cfg = self.config.get("evidence", {})
        self.storage_path = Path(evidence_cfg.get("storage_path", "data/evidence/"))
        self.storage_path.mkdir(parents=True, exist_ok=True)
        self.encrypt = evidence_cfg.get("encrypt", False)

    async def save(
        self,
        frame: np.ndarray,
        camera_id: str,
        timestamp: datetime,
        alert_type: str,
        extra: Optional[Dict] = None,
    ) -> Dict[str, str]:
        evidence_id = f"EVID_{str(uuid.uuid4())[:8].upper()}"
        evidence_dir = self.storage_path / evidence_id
        evidence_dir.mkdir(parents=True, exist_ok=True)

        snap_plain = evidence_dir / "snapshot_original.jpg"
        snap_enc = evidence_dir / "snapshot_encrypted.enc"

        # 1. Write snapshot
        loop = asyncio.get_event_loop()
        await loop.run_in_executor(
            None, cv2.imwrite, str(snap_plain), frame
        )

        # 2. SHA-256 Hash
        sha256_hash = await loop.run_in_executor(
            None, compute_sha256, str(snap_plain)
        )

        # 3. Optional AES-256 Encryption
        if self.encrypt:
            await loop.run_in_executor(
                None, encrypt_file, str(snap_plain), str(snap_enc)
            )
            try:
                snap_plain.unlink()
            except Exception:
                pass
            final_snap = str(snap_enc)
        else:
            final_snap = str(snap_plain)

        # 4. Chain of Custody
        custody = {
            "evidence_id": evidence_id,
            "camera_id": camera_id,
            "alert_type": alert_type,
            "timestamp_utc": timestamp.isoformat(),
            "timestamp_ist": timestamp.strftime("%d-%b-%Y %H:%M:%S IST"),
            "snapshot_path": final_snap,
            "snapshot_sha256": sha256_hash,
            "encrypted": self.encrypt,
            "sentinel_version": "3.0.0",
            "created_by": "SENTINEL GUJARAT AUTONOMOUS SYSTEM",
            "extra": extra or {},
        }

        custody_path = evidence_dir / "chain_of_custody.json"
        await loop.run_in_executor(
            None, lambda: custody_path.write_text(json.dumps(custody, indent=2), encoding="utf-8")
        )

        custody_hash = await loop.run_in_executor(
            None, compute_sha256, str(custody_path)
        )

        return {
            "evidence_id": evidence_id,
            "snapshot_path": final_snap,
            "sha256": sha256_hash,
            "custody_path": str(custody_path),
            "custody_sha256": custody_hash,
        }
