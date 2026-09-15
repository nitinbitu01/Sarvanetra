"""
tests/test_evidence_builder.py — Evidence Builder & Per-Panel SHA-256 Tests (Tests 29-31).
"""
import os
import numpy as np
import pytest

from backend.services.evidence_builder import _build_sync, _sha256_file


def test_per_panel_hashes_present(tmp_path):
    p1 = np.full((100, 100, 3), 50, dtype=np.uint8)
    p2 = np.full((100, 100, 3), 100, dtype=np.uint8)
    p3 = np.full((100, 100, 3), 150, dtype=np.uint8)
    p4 = np.full((100, 100, 3), 200, dtype=np.uint8)

    meta = {"camera_id": "CAM_01", "ts": "2026-08-24 12:00:00", "speed": 35.0}
    out_file = str(tmp_path / "test_collage.jpg")

    pkg = _build_sync(p1, p2, p3, p4, meta, out_file)
    assert os.path.exists(pkg.collage_path)
    assert len(pkg.collage_hash) == 64
    assert len(pkg.panel_source_hashes) == 4
    assert "panel1_context" in pkg.panel_source_hashes
    assert "panel2_zoomed" in pkg.panel_source_hashes
    assert "panel3_plate" in pkg.panel_source_hashes
    assert "panel4_helmets" in pkg.panel_source_hashes


def test_panel_swap_detected(tmp_path):
    p1 = np.full((100, 100, 3), 50, dtype=np.uint8)
    p2_orig = np.full((100, 100, 3), 100, dtype=np.uint8)
    p2_tampered = np.full((100, 100, 3), 105, dtype=np.uint8)
    p3 = np.full((100, 100, 3), 150, dtype=np.uint8)
    p4 = np.full((100, 100, 3), 200, dtype=np.uint8)

    meta = {"camera_id": "CAM_01"}
    pkg1 = _build_sync(p1, p2_orig, p3, p4, meta, str(tmp_path / "c1.jpg"))
    pkg2 = _build_sync(p1, p2_tampered, p3, p4, meta, str(tmp_path / "c2.jpg"))

    # p2 hash must differ
    assert pkg1.panel_source_hashes["panel2_zoomed"] != pkg2.panel_source_hashes["panel2_zoomed"]
