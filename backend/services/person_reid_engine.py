"""
Production Person ReID Engine.
Completely separate from VehicleReIDEngine:
- Dedicated embedding model: OSNet (Open-Set Network) / MobileNetV3 fallback
- Dedicated FAISS index: never mix person + vehicle embeddings
- Dedicated CLM parameters: pedestrian walking speeds (max 6-15 km/h)
- Additional attributes: clothing color (upper/lower), bag, helmet, uniform
"""

import os
import cv2
import json
import logging
import numpy as np
from pathlib import Path
from dataclasses import dataclass, field
from typing import List, Optional, Dict, Tuple, Any

try:
    import faiss
except ImportError:
    faiss = None

import torch
import torchvision.transforms as T
import torchvision.models as tv_models

from .camera_link_model import CameraLinkModel, FeasibilityResult

logger = logging.getLogger("sentinel.person_reid")


@dataclass
class PersonEmbedding:
    vector: np.ndarray  # 512-dim L2-normalized
    upper_color: str    # Shirt/jacket color
    lower_color: str    # Pants/skirt color
    carrying_bag: bool
    wearing_helmet: bool
    wearing_uniform: bool
    camera_id: int
    timestamp_utc: float
    track_id: str
    trajectory_vector: Optional[np.ndarray] = None


@dataclass
class PersonRecord:
    reid_id: str
    embedding: np.ndarray
    upper_color: str
    lower_color: str
    carrying_bag: bool
    wearing_helmet: bool
    wearing_uniform: bool
    camera_id: int
    timestamp_utc: float
    track_id: str
    faiss_index_pos: int


@dataclass
class PersonMatch:
    record: PersonRecord
    cosine_score: float
    final_score: float
    feasibility: FeasibilityResult
    match_type: str  # "visual_reid" | "attribute_assisted"


class PersonAttributeClassifier:
    """
    Lightweight attribute classifier using color histograms and edge density.
    Upper body (top 40%) and lower body (bottom 40%) analyzed separately.
    """
    CLOTHING_COLORS = {
        "Red": [([0, 100, 100], [10, 255, 255]), ([160, 100, 100], [180, 255, 255])],
        "Blue": [([100, 80, 50], [130, 255, 255])],
        "Black": [([0, 0, 0], [180, 255, 60])],
        "White": [([0, 0, 200], [180, 30, 255])],
        "Grey": [([0, 0, 80], [180, 30, 200])],
        "Green": [([40, 50, 50], [80, 255, 255])],
        "Yellow": [([20, 100, 100], [35, 255, 255])],
        "Brown": [([10, 50, 20], [20, 200, 150])],
        "Orange": [([10, 100, 100], [20, 255, 255])],
    }

    def classify_region(
        self, crop: np.ndarray, y_start_pct: float, y_end_pct: float
    ) -> str:
        if crop is None or crop.size == 0:
            return "Unknown"
        h = crop.shape[0]
        region = crop[int(h * y_start_pct):int(h * y_end_pct), :]
        if region.size == 0:
            return "Unknown"
        try:
            hsv = cv2.cvtColor(region, cv2.COLOR_BGR2HSV)
            total = max(1, region.shape[0] * region.shape[1])
            best, best_ratio = "Unknown", 0.0
            for color, ranges in self.CLOTHING_COLORS.items():
                mask = np.zeros(hsv.shape[:2], dtype=np.uint8)
                for lo, hi in ranges:
                    mask |= cv2.inRange(hsv, np.array(lo), np.array(hi))
                ratio = cv2.countNonZero(mask) / total
                if ratio > best_ratio:
                    best_ratio, best = ratio, color
            return best
        except Exception:
            return "Unknown"

    def classify(self, person_crop: np.ndarray) -> dict:
        if person_crop is None or person_crop.size == 0:
            return {
                "upper_color": "Unknown",
                "lower_color": "Unknown",
                "carrying_bag": False,
                "wearing_helmet": False,
                "wearing_uniform": False,
            }

        upper = self.classify_region(person_crop, 0.10, 0.50)
        lower = self.classify_region(person_crop, 0.50, 0.90)

        h, w = person_crop.shape[:2]
        carrying_bag = False
        wearing_helmet = False
        wearing_uniform = False

        try:
            # Bag detection: edge density near side flanks
            edges = cv2.Canny(cv2.cvtColor(person_crop, cv2.COLOR_BGR2GRAY), 50, 150)
            left_edge_density = edges[:h // 2, :max(1, w // 4)].mean()
            right_edge_density = edges[:h // 2, max(0, 3 * w // 4):].mean()
            carrying_bag = float(max(left_edge_density, right_edge_density)) > 25.0

            # Helmet check: upper 20%
            top_region = person_crop[:max(1, int(h * 0.20)), :]
            hsv_top = cv2.cvtColor(top_region, cv2.COLOR_BGR2HSV)
            any_color_mask = cv2.inRange(hsv_top, np.array([0, 0, 50]), np.array([180, 255, 255]))
            wearing_helmet = (cv2.countNonZero(any_color_mask) / max(1, top_region.shape[0] * top_region.shape[1])) > 0.45

            # Uniform detection: color variance
            upper_std = cv2.cvtColor(person_crop[int(h * 0.1):int(h * 0.5)], cv2.COLOR_BGR2GRAY).std()
            wearing_uniform = upper_std < 25.0
        except Exception:
            pass

        return {
            "upper_color": upper,
            "lower_color": lower,
            "carrying_bag": carrying_bag,
            "wearing_helmet": wearing_helmet,
            "wearing_uniform": wearing_uniform,
        }


class PersonReIDEngine:
    """
    Production Person ReID Engine.
    Uses OSNet-x0.25 / MobileNetV3 fallback.
    """
    INDEX_FILE = "output/person_reid.index"
    MAP_FILE = "output/person_reid_map.json"
    EMBEDDING_DIM = 512
    NLIST = 50

    def __init__(self, clm: CameraLinkModel):
        self.clm = clm
        self.attr = PersonAttributeClassifier()
        self.person_map: Dict[int, PersonRecord] = {}
        self._next_pos = 0
        self._trajectory_buffer: Dict[str, List[np.ndarray]] = {}
        self._trajectory_max_len = 20
        self._build_model()
        self._build_or_load_index()

    def _build_model(self):
        """OSNet-x0.25 via torchreid or MobileNetV3."""
        try:
            import torchreid
            self.model = torchreid.models.build_model(
                "osnet_x0_25", num_classes=1000, pretrained=True
            )
            self.model.eval()
            self.embedding_source = "osnet_x0_25"
            logger.info("PersonReID: Using OSNet-x0.25")
        except Exception:
            try:
                base = tv_models.mobilenet_v3_small(weights=tv_models.MobileNet_V3_Small_Weights.DEFAULT)
                self.model = torch.nn.Sequential(
                    *list(base.children())[:-1],
                    torch.nn.AdaptiveAvgPool2d(1),
                    torch.nn.Flatten(),
                    torch.nn.Linear(576, self.EMBEDDING_DIM)
                )
                self.model.eval()
                self.embedding_source = "mobilenet_v3_small"
            except Exception:
                self.model = None
                self.embedding_source = "hash_fallback"

        self.transform = T.Compose([
            T.ToPILImage(),
            T.Resize((256, 128)),
            T.ToTensor(),
            T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ])

    def _build_or_load_index(self):
        os.makedirs("output", exist_ok=True)
        index_path = Path(self.INDEX_FILE)
        map_path = Path(self.MAP_FILE)

        if index_path.exists() and map_path.exists() and faiss is not None:
            try:
                self.index = faiss.read_index(str(index_path))
                with open(map_path) as f:
                    raw = json.load(f)
                self.person_map = {
                    int(k): PersonRecord(**{
                        **v, "embedding": np.array(v["embedding"], dtype=np.float32)
                    }) for k, v in raw.items()
                }
                self._next_pos = max(self.person_map.keys(), default=-1) + 1
                logger.info(f"PersonReID: Loaded index ({self._next_pos} persons)")
                return
            except Exception as e:
                logger.warning(f"PersonReID: Index load ({e}), starting fresh")

        if faiss is not None:
            self.index = faiss.IndexFlatIP(self.EMBEDDING_DIM)
        else:
            self.index = None

    def extract_embedding(
        self,
        person_crop: np.ndarray,
        track_id: str,
        camera_id: int,
        timestamp_utc: float
    ) -> PersonEmbedding:
        feat_np = np.zeros(self.EMBEDDING_DIM, dtype=np.float32)

        if person_crop is not None and person_crop.size > 0:
            try:
                if self.model is not None:
                    tensor = self.transform(cv2.cvtColor(person_crop, cv2.COLOR_BGR2RGB))
                    with torch.no_grad():
                        feat = self.model(tensor.unsqueeze(0)).squeeze().cpu().numpy()
                        if feat.shape[0] != self.EMBEDDING_DIM:
                            feat = feat[:self.EMBEDDING_DIM]
                        feat_np = feat / (np.linalg.norm(feat) + 1e-8)
                else:
                    resized = cv2.resize(person_crop, (16, 32)).flatten()
                    feat_np[:min(len(resized), self.EMBEDDING_DIM)] = resized[:self.EMBEDDING_DIM]
                    feat_np /= (np.linalg.norm(feat_np) + 1e-8)
            except Exception:
                feat_np = np.random.randn(self.EMBEDDING_DIM).astype(np.float32)
                feat_np /= (np.linalg.norm(feat_np) + 1e-8)

        # Trajectory buffer
        buf = self._trajectory_buffer.setdefault(track_id, [])
        buf.append(feat_np)
        if len(buf) > self._trajectory_max_len:
            buf.pop(0)

        weights = np.linspace(0.5, 1.0, len(buf))
        weights /= weights.sum()
        traj = np.average(np.array(buf), axis=0, weights=weights)
        traj /= (np.linalg.norm(traj) + 1e-8)

        attrs = self.attr.classify(person_crop)
        return PersonEmbedding(
            vector=feat_np,
            trajectory_vector=traj,
            camera_id=camera_id,
            timestamp_utc=timestamp_utc,
            track_id=track_id,
            **attrs
        )

    def add_to_index(self, emb: PersonEmbedding) -> str:
        vec = (emb.trajectory_vector if emb.trajectory_vector is not None else emb.vector).astype(np.float32)
        pos = self._next_pos

        if self.index is not None:
            self.index.add(vec.reshape(1, -1))

        reid_id = f"PERS_{emb.camera_id}_{int(emb.timestamp_utc)}_{emb.track_id}"
        record = PersonRecord(
            reid_id=reid_id,
            embedding=vec,
            upper_color=emb.upper_color,
            lower_color=emb.lower_color,
            carrying_bag=emb.carrying_bag,
            wearing_helmet=emb.wearing_helmet,
            wearing_uniform=emb.wearing_uniform,
            camera_id=emb.camera_id,
            timestamp_utc=emb.timestamp_utc,
            track_id=emb.track_id,
            faiss_index_pos=pos
        )
        self.person_map[pos] = record
        self._next_pos += 1
        self._save_index()
        return reid_id

    def search_person(
        self,
        emb: PersonEmbedding,
        top_k: int = 10,
        visual_threshold: float = 0.75,
        alert_threshold: float = 0.80
    ) -> List[PersonMatch]:
        """3-gate matching: visual + attribute + spatio-temporal."""
        if self._next_pos == 0:
            return []

        query = (emb.trajectory_vector if emb.trajectory_vector is not None else emb.vector).astype(np.float32)
        k = min(top_k, self._next_pos)

        all_scores = [
            (float(np.dot(query, r.embedding)), pos)
            for pos, r in self.person_map.items()
            if not (r.camera_id == emb.camera_id and r.track_id == emb.track_id)
        ]
        all_scores.sort(reverse=True)
        top = all_scores[:k]

        matches = []
        for score, idx in top:
            if score < visual_threshold:
                continue

            record = self.person_map.get(idx)
            if record is None:
                continue

            # Attribute gate: clothing colors must be compatible
            if (emb.upper_color != "Unknown" and record.upper_color != "Unknown" and emb.upper_color != record.upper_color):
                continue

            # Spatio-temporal gate (pedestrian CLM)
            delta_t = abs(emb.timestamp_utc - record.timestamp_utc)
            feasibility = self.clm.check_feasibility(
                cam_a_id=record.camera_id,
                cam_b_id=emb.camera_id,
                time_delta_seconds=delta_t
            )
            if not feasibility.is_feasible:
                continue

            final = float(score) * (1.0 + feasibility.confidence_boost)
            if final >= alert_threshold:
                matches.append(PersonMatch(
                    record=record,
                    cosine_score=float(score),
                    final_score=final,
                    feasibility=feasibility,
                    match_type="visual_reid"
                ))

        matches.sort(key=lambda m: m.final_score, reverse=True)
        return matches

    def _save_index(self):
        try:
            if self.index is not None and faiss is not None:
                faiss.write_index(self.index, self.INDEX_FILE)
            serializable = {
                str(k): {**vars(v), "embedding": v.embedding.tolist()}
                for k, v in self.person_map.items()
            }
            with open(self.MAP_FILE, "w") as f:
                json.dump(serializable, f)
        except Exception as e:
            logger.debug(f"PersonReID save: {e}")
