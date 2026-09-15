import asyncio
import logging
import time
import uuid
from datetime import datetime
from typing import Optional, Tuple

import cv2
import faiss
import numpy as np

from backend.metrics.prometheus_metrics import (
    FAISS_INDEX_SIZE, REID_MATCHES, REID_SEARCH_LATENCY
)

logger = logging.getLogger("sentinel.reid")

THRESHOLDS = {
    "same_zone":   0.78,
    "cross_zone":  0.85,
    "watchlist":   0.82,
    "general":     0.70,
}

EMBEDDING_DIM = 512


class ReIDEngine:
    def __init__(self):
        self.index   = faiss.IndexFlatIP(EMBEDDING_DIM)
        self.id_map:  list[str]  = []
        self.zone_map: list[str] = []
        self._lock   = asyncio.Lock()
        self._model  = None
        self._ready  = False

    async def initialize(self):
        if self._ready:
            return
        try:
            import torchreid
            self._model = torchreid.models.build_model(
                name="osnet_ibn_x1_0", num_classes=1000, pretrained=True
            )
            self._model.eval()
            logger.info("ReID Engine ✅ OSNet-IBN loaded")
        except Exception:
            logger.info("ReID Engine running in HOG/Color Histogram fallback mode")
            self._model = None
        self._ready = True

    async def identify(
        self,
        person_crop: np.ndarray,
        camera_id: str,
        current_zone: str = "",
        prev_zone: str = "",
    ) -> Tuple[str, bool, float]:
        """
        Returns: (reid_id, is_known_match, similarity_score)
        """
        await self.initialize()

        embedding = await asyncio.get_event_loop().run_in_executor(
            None, self._extract, person_crop
        )
        if embedding is None:
            return f"REID_{datetime.utcnow().strftime('%Y%m%d')}_{str(uuid.uuid4())[:8].upper()}", False, 0.0

        threshold = (
            THRESHOLDS["cross_zone"]
            if current_zone and prev_zone and current_zone != prev_zone
            else THRESHOLDS["general"]
        )

        t0 = time.monotonic()
        async with self._lock:
            result = await self._search(embedding, threshold)
        elapsed_ms = (time.monotonic() - t0) * 1000
        REID_SEARCH_LATENCY.observe(elapsed_ms)

        if result:
            REID_MATCHES.labels(match_type="known_match").inc()
            return result[0], True, result[1]

        reid_id = (
            f"REID_{datetime.utcnow().strftime('%Y%m%d')}_{str(uuid.uuid4())[:8].upper()}"
        )
        async with self._lock:
            normed = self._normalize(embedding)
            self.index.add(normed)
            self.id_map.append(reid_id)
            self.zone_map.append(current_zone)
            FAISS_INDEX_SIZE.set(self.index.ntotal)

        REID_MATCHES.labels(match_type="new_identity").inc()
        return reid_id, False, 0.0

    async def _search(
        self, embedding: np.ndarray, threshold: float
    ) -> Optional[Tuple[str, float]]:
        if self.index.ntotal == 0:
            return None
        normed = self._normalize(embedding)
        D, I = self.index.search(normed, k=1)
        score, idx = float(D[0][0]), int(I[0][0])
        if score >= threshold and idx >= 0:
            return self.id_map[idx], score
        return None

    def _normalize(self, embedding: np.ndarray) -> np.ndarray:
        arr = embedding.reshape(1, -1).astype(np.float32)
        faiss.normalize_L2(arr)
        return arr

    def _extract(self, crop: np.ndarray) -> Optional[np.ndarray]:
        if crop is None or crop.size == 0:
            return None
        try:
            return self._osnet(crop) if self._model else self._hog(crop)
        except Exception as e:
            logger.error(f"Embedding extraction error: {e}")
            return None

    def _osnet(self, crop: np.ndarray) -> np.ndarray:
        import torch
        import torchvision.transforms as T

        tf = T.Compose([
            T.ToPILImage(),
            T.Resize((256, 128)),
            T.ToTensor(),
            T.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
        ])
        img = tf(cv2.cvtColor(crop, cv2.COLOR_BGR2RGB)).unsqueeze(0)
        with torch.no_grad():
            return self._model(img).squeeze().numpy()

    def _hog(self, crop: np.ndarray) -> np.ndarray:
        resized = cv2.resize(crop, (64, 128))
        gray    = cv2.cvtColor(resized, cv2.COLOR_BGR2GRAY)
        hog     = cv2.HOGDescriptor(
            (64, 128), (16, 16), (8, 8), (8, 8), 9
        )
        desc = hog.compute(gray).flatten()
        if len(desc) >= EMBEDDING_DIM:
            return desc[:EMBEDDING_DIM]
        padded = np.zeros(EMBEDDING_DIM, dtype=np.float32)
        padded[:len(desc)] = desc
        return padded
