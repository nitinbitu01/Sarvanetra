"""
tests/test_production_master.py — Master Production Certification & Regression Test Suite.

Verifies:
  1. GeoGuard Spatio-Temporal Road Graph (Teleportation rejection & valid transit).
  2. Thread-Safe Temporal ReID Smoother (Centroid pooling, outlier immunity, adaptive voting).
  3. 512-D Float32 BLOB Serialization (2048 bytes per embedding).
  4. Mathematical Danger Score Engine.
  5. FAISS HNSW Index Creation & Cluster Filtering.
"""
from datetime import datetime
import numpy as np
import pytest

from backend.core.reid_smoother import ReidSmoother, TrackBuffer, consensus_vote
from backend.embedding_utils import normalize_l2
from backend.faiss_index import (
    REID_EMBEDDING_DIM,
    FaissReIDIndex,
    blob_to_embedding,
    embedding_to_blob,
)
from backend.services.danger_calculator import calculate_danger_score
from backend.services.geo_guard import GeoGuard, haversine_distance



class TestGeoGuard:
    @pytest.fixture
    def geo_guard(self):
        # Initialize GeoGuard instance
        gg = GeoGuard()
        # Seed test cameras
        gg.camera_map = {
            "CAM_01": {"id": "CAM_01", "lat": 23.0395, "lon": 72.5797, "district": "Ahmedabad"},
            "CAM_02": {"id": "CAM_02", "lat": 23.0225, "lon": 72.5714, "district": "Ahmedabad"},
            "CAM_27": {"id": "CAM_27", "lat": 20.8500, "lon": 72.9200, "district": "Navsari"},
        }
        gg.cluster_map = {
            "CAM_01": "Ahmedabad",
            "CAM_02": "Ahmedabad",
            "CAM_27": "Navsari",
        }
        return gg

    def test_haversine_accuracy(self):
        # Ahmedabad to Janpath ~ 2.06 km
        dist = haversine_distance(23.0395, 72.5797, 23.0225, 72.5714)
        assert 1.8 < dist < 2.5

    def test_teleportation_rejected(self, geo_guard):
        # Ahmedabad (CAM_01) to Navsari (CAM_27) ~240 km in 30 seconds -> V = 28,800 km/h
        is_valid, reason = geo_guard.is_transit_feasible("CAM_01", "CAM_27", time_gap_seconds=30.0)
        assert not is_valid
        assert "VIOLATION" in reason

    def test_valid_intracity_transit(self, geo_guard):
        # CAM_01 to CAM_02 (~2.06 km) in 6 minutes (360s) -> V ~ 20.6 km/h (valid vehicular speed)
        is_valid, reason = geo_guard.is_transit_feasible("CAM_01", "CAM_02", time_gap_seconds=360.0)
        assert is_valid
        assert "VALID_TRANSIT" in reason

    def test_same_camera_always_valid(self, geo_guard):
        is_valid, reason = geo_guard.is_transit_feasible("CAM_01", "CAM_01", time_gap_seconds=5.0)
        assert is_valid
        assert reason == "SAME_CAMERA"


class TestTemporalSmoother:
    def test_centroid_pooling(self):
        smoother = ReidSmoother(ttl_seconds=60)
        v1 = np.ones(512, dtype=np.float32)
        v2 = np.ones(512, dtype=np.float32)
        smoother.add_frame(1, v1)
        smoother.add_frame(1, v2)
        centroid = smoother.get_centroid_embedding(1)
        assert centroid is not None
        assert centroid.shape == (512,)
        assert np.isclose(np.linalg.norm(centroid), 1.0, atol=1e-5)

    def test_adaptive_consensus_voting(self):
        # Case 1: Less than 3 frames -> always False (routed to review)
        assert not consensus_vote([0.95, 0.95], theta=0.85, s_geo=1.0, min_votes=3)

        # Case 2: Teleportation violation (s_geo=0.0) -> always False
        assert not consensus_vote([0.95, 0.95, 0.95, 0.95], theta=0.85, s_geo=0.0, min_votes=3)

        # Case 3: 4 frames, 3 exceed threshold -> True
        assert consensus_vote([0.90, 0.88, 0.86, 0.40], theta=0.85, s_geo=1.0, min_votes=3)

        # Case 4: 5 frames, only 2 exceed threshold -> False
        assert not consensus_vote([0.90, 0.88, 0.40, 0.35, 0.30], theta=0.85, s_geo=1.0, min_votes=3)


class TestEmbeddingSerialization:
    def test_blob_roundtrip(self):
        original = np.random.randn(512).astype(np.float32)
        norm_orig = normalize_l2(original)
        blob = embedding_to_blob(norm_orig)
        assert len(blob) == 2048  # 512 * 4 bytes
        restored = blob_to_embedding(blob)
        assert restored.shape == (512,)
        assert restored.dtype == np.float32
        assert np.allclose(norm_orig, restored, atol=1e-6)

    def test_invalid_shape_raises(self):
        with pytest.raises(ValueError):
            embedding_to_blob(np.zeros(256, dtype=np.float32))


class TestDangerCalculator:
    def test_danger_score_multipliers(self):
        # Person (1.0) + Loiter >=300s (2.5) + Watchlist (3.0) = 7.5
        res = calculate_danger_score(
            class_name="person",
            dwell_time_seconds=320.0,
            is_watchlist_match=True,
            nearby_crowd_count=1,
            is_restricted_zone=False,
            is_high_crime_zone=False,
            timestamp=datetime(2026, 8, 24, 14, 0, 0),
        )
        assert res["danger_score"] == 7.5
        assert res["alert_level"] == "HIGH"
        assert "persistent_loitering" in res["threat_tags"]
        assert "watchlist_suspect" in res["threat_tags"]

    def test_critical_alert_level(self):
        # Person + Loiter (2.5) + Watchlist (3.0) + Restricted Zone (1.9) = 14.25
        res = calculate_danger_score(
            class_name="person",
            dwell_time_seconds=320.0,
            is_watchlist_match=True,
            nearby_crowd_count=1,
            is_restricted_zone=True,
            is_high_crime_zone=False,
            timestamp=datetime(2026, 8, 24, 14, 0, 0),
        )
        assert res["danger_score"] == 14.25
        assert res["alert_level"] == "CRITICAL"


class TestFaissHNSW:
    def test_hnsw_creation_and_search(self):
        index = FaissReIDIndex(dim=512, use_hnsw=True)
        assert index.dim == 512

        # Add 10 synthetic vectors
        for i in range(10):
            vec = np.random.randn(512).astype(np.float32)
            index.add_vector(vec, {"clip_id": f"clip_{i}", "district": "Ahmedabad"})

        assert index.ntotal == 10
        query = np.random.randn(512).astype(np.float32)
        results = index.search(query, k=3, cluster_filter="Ahmedabad")
        assert len(results) == 3
        assert "score" in results[0]
        assert "clip_id" in results[0]
