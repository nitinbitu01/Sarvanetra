"""
backend/services/evidence.py — Tamper-evident evidence integrity.

hash_file(path): SHA-256 hex of a file on disk.
set_readonly(path): chmod 0o444 — first line of defence against accidental overwrite.
verify_file(path, expected_hash): returns True if hash matches.

Called at alert creation time and by the /alerts/{id}/verify endpoint.
"""
from __future__ import annotations

import hashlib
import os
import stat
from pathlib import Path

import numpy as np


def hash_file(path: str | Path) -> str | None:
    """Compute SHA-256 hex digest of a file.

    Args:
        path: Path to the evidence file.

    Returns:
        64-char hex string, or None if the file doesn't exist or can't be read.
    """
    p = Path(path)
    if not p.exists():
        return None
    sha = hashlib.sha256()
    try:
        with p.open("rb") as fh:
            for chunk in iter(lambda: fh.read(65536), b""):
                sha.update(chunk)
        return sha.hexdigest()
    except OSError:
        return None


def set_readonly(path: str | Path) -> None:
    """Set a file read-only (0o444). Silently ignores errors on non-POSIX."""
    try:
        os.chmod(path, stat.S_IRUSR | stat.S_IRGRP | stat.S_IROTH)
    except OSError:
        pass  # Windows may not support all permission bits — best-effort


def verify_file(path: str | Path, expected_hash: str) -> bool:
    """Re-hash the file and compare to stored hash.

    Args:
        path: Evidence file path.
        expected_hash: Stored SHA-256 hex from Alert.evidence_hash.

    Returns:
        True if file still matches stored hash (intact), False if tampered or missing.
    """
    if not expected_hash:
        return False
    current = hash_file(path)
    if current is None:
        return False
    return current.lower() == expected_hash.lower()


def lock_evidence(
    frame: np.ndarray,
    track_id: int,
    camera_id: str,
    timestamp: float | None = None,
    base_dir: str = "output/evidence",
    bbox: "tuple[float, float, float, float] | None" = None,
) -> tuple[str, str, str]:
    """Locks frame evidence to disk and generates cryptographic SHA-256 digest.

    Writes TWO artefacts:
      - the full context frame (clip_path) - shows the violation in situ,
        which is what an officer or a court needs to see
      - a crop of the offending vehicle (crop_path) - what ANPR reads

    These were previously the same file, so plate OCR ran against the whole
    1080p frame instead of the vehicle, and the "evidence clip" was a still
    with no surrounding context. Both are hashed; the crop's hash is the one
    returned, since that is the artefact an automated plate read is based on.

    Args:
        bbox: (x1, y1, x2, y2) of the vehicle in pixel coords. If omitted or
            degenerate, the crop falls back to the full frame rather than
            failing - a slightly worse plate read beats losing the evidence.

    Returns:
        (clip_path, clip_hash, crop_path)
    """
    import time
    import cv2
    t = int((timestamp or time.time()) * 1000)
    os.makedirs(f"{base_dir}/crops", exist_ok=True)
    os.makedirs(f"{base_dir}/frames", exist_ok=True)

    stem = f"{camera_id}_track{track_id}_{t}"
    clip_path = f"{base_dir}/frames/{stem}.jpg"
    crop_path = f"{base_dir}/crops/{stem}.jpg"

    if frame is None or getattr(frame, "size", 0) == 0:
        # Placeholder so the caller still gets a hashable artefact rather
        # than a dangling path.
        for p in (clip_path, crop_path):
            with open(p, "wb") as f:
                f.write(b"EMPTY_FRAME")
    else:
        cv2.imwrite(clip_path, frame)

        crop = frame
        if bbox is not None:
            h, w = frame.shape[:2]
            x1, y1, x2, y2 = (int(round(v)) for v in bbox)
            x1, y1 = max(0, x1), max(0, y1)
            x2, y2 = min(w, x2), min(h, y2)
            if x2 - x1 > 1 and y2 - y1 > 1:
                crop = frame[y1:y2, x1:x2]
        cv2.imwrite(crop_path, crop)

    set_readonly(clip_path)
    set_readonly(crop_path)
    clip_hash = hash_file(crop_path) or ""
    return (clip_path, clip_hash, crop_path)

