"""
backend/services/evidence_builder.py — Async 4-Panel Court-Admissible Evidence Collage.

Generates:
  - Panel 1: Context Wide Frame
  - Panel 2: Zoomed Motorcycle & Rider Envelopes
  - Panel 3: High-Res License Plate Zoom
  - Panel 4: Per-Rider Helmet Analysis Crops
  - SHA-256 hashes per panel & composite output file.
"""
import asyncio
import hashlib
import os
import time
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass

import cv2
import numpy as np

_executor = ThreadPoolExecutor(max_workers=2, thread_name_prefix="collage")


@dataclass
class EvidencePackage:
    collage_path: str
    collage_hash: str
    panel_source_hashes: dict[str, str]  # per-panel source frame SHA-256


def _sha256_array(img: np.ndarray) -> str:
    if img is None or img.size == 0:
        return ""
    return hashlib.sha256(img.tobytes()).hexdigest()


def _sha256_file(path: str) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(65536), b""):
            h.update(chunk)
    return h.hexdigest()


def _build_sync(
    p1: np.ndarray,
    p2: np.ndarray,
    p3: np.ndarray,
    p4: np.ndarray,
    meta: dict,
    out_path: str,
) -> EvidencePackage:
    # Per-panel hashes BEFORE compositing
    panel_hashes = {
        "panel1_context": _sha256_array(p1),
        "panel2_zoomed": _sha256_array(p2),
        "panel3_plate": _sha256_array(p3),
        "panel4_helmets": _sha256_array(p4),
    }

    TH, TW = 360, 640

    def ann(img: np.ndarray, txt: str) -> np.ndarray:
        if img is None or img.size == 0:
            img = np.zeros((TH, TW, 3), dtype=np.uint8)
        out = cv2.resize(img, (TW, TH))
        cv2.putText(out, txt, (8, 28), cv2.FONT_HERSHEY_SIMPLEX, 0.65, (0, 255, 0), 2)
        return out

    r = meta.get("majority_ratio", 0.0)
    panels = [
        ann(p1, f"CAM:{meta.get('camera_id','')} {meta.get('ts','')} {meta.get('speed',0):.0f}km/h"),
        ann(p2, f"Riders:{meta.get('rider_count',0)} Confidence:{r:.0%}"),
        ann(p3, f"PLATE:{meta.get('plate','')} Conf:{meta.get('plate_conf',0):.0%}"),
        ann(p4, "Helmet Analysis"),
    ]
    grid = np.vstack([np.hstack(panels[:2]), np.hstack(panels[2:])])

    footer = np.zeros((40, grid.shape[1], 3), dtype=np.uint8)
    p1_hash_prefix = panel_hashes["panel1_context"][:12] if panel_hashes["panel1_context"] else "N/A"
    ftext = f"SHA256(p1):{p1_hash_prefix}... | Status: PENDING REVIEW"
    cv2.putText(footer, ftext, (8, 26), cv2.FONT_HERSHEY_SIMPLEX, 0.42, (200, 200, 200), 1)
    collage = np.vstack([grid, footer])

    os.makedirs(os.path.dirname(os.path.abspath(out_path)), exist_ok=True)
    cv2.imwrite(out_path, collage, [cv2.IMWRITE_JPEG_QUALITY, 92])
    return EvidencePackage(
        collage_path=out_path,
        collage_hash=_sha256_file(out_path),
        panel_source_hashes=panel_hashes,
    )


async def build_collage_async(
    p1: np.ndarray,
    p2: np.ndarray,
    p3: np.ndarray,
    p4: np.ndarray,
    meta: dict,
    out_path: str,
) -> EvidencePackage:
    loop = asyncio.get_event_loop()
    return await loop.run_in_executor(
        _executor, _build_sync, p1, p2, p3, p4, meta, out_path
    )
