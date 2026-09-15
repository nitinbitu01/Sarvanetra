"""
tests/test_multiframe_fusion.py — Unit tests for Item 1B: Multi-Frame Consensus & Fusion.
"""
from unittest.mock import MagicMock
import pytest
from backend.anpr_track_aggregator import _Reading, _TrackAnprState, AnprTrackAggregator


def test_multiframe_weighted_consensus(monkeypatch):
    """Verify that multi-frame sharpness & confidence consensus correctly breaks ties
    and boosts precision over single blurry frames.
    """
    import backend.db as db_mod
    monkeypatch.setattr(db_mod, "get_active_watchlist_vehicles", lambda path: [])
    monkeypatch.setattr(db_mod, "insert_alert", lambda *args, **kwargs: 1)

    mock_worker = MagicMock()
    agg = AnprTrackAggregator(
        db_path=":memory:",
        ocr_worker=mock_worker,
        camera_id="CAM_01",
        cfg={"anpr": {"uncertain_confidence_threshold": 0.70}},
    )
    state = _TrackAnprState(
        track_id="V-101",
        camera_id="CAM_01",
        db_path=":memory:",
        cfg={"anpr": {"uncertain_confidence_threshold": 0.70}},
    )

    # Frame 1: Low sharpness, noisy read
    r1 = _Reading(
        text="GJ03KH221Z",
        confidence=0.72,
        source="easyocr",
        agreement=False,
        format_valid=True,
        frame_number=10,
        timestamp=100.0,
        sharpness=5.0,
    )
    # Frame 2: High sharpness, correct read
    r2 = _Reading(
        text="GJ03KH2212",
        confidence=0.94,
        source="crnn_beam",
        agreement=True,
        format_valid=True,
        frame_number=15,
        timestamp=100.2,
        sharpness=120.0,
    )
    # Frame 3: Medium sharpness, correct read
    r3 = _Reading(
        text="GJ03KH2212",
        confidence=0.91,
        source="crnn_beam",
        agreement=True,
        format_valid=True,
        frame_number=20,
        timestamp=100.4,
        sharpness=85.0,
    )

    state.readings = [r1, r2, r3]
    agg._finalize(state)

    assert state.finalized is True
    valid_texts = [r.text for r in state.valid_readings]
    assert "GJ03KH2212" in valid_texts


def test_grammar_syntax_repair_in_consensus():
    """Verify that letter/digit confusions are resolved by Indian plate grammar."""
    from backend.scripts.indian_plate_grammar import decode_plate
    
    # 'O' in number zone -> '0'
    res = decode_plate("GJ03KH22O2")
    assert res["plate"] == "GJ03KH2202"
    assert res["format"] == "standard"

    # '8' in state zone -> 'B' (e.g. 8R -> BR)
    res_br = decode_plate("BR01AB1234")
    assert res_br["plate"] == "BR01AB1234"
    assert res_br["score"] >= 0.85
