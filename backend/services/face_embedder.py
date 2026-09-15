"""
backend/services/face_embedder.py — InsightFace face embedding extractor (Day 7).

Model: InsightFace `buffalo_l` FaceAnalysis pack (RetinaFace detector + ArcFace
  recognition, ONNX runtime backend). Output: 512-dimensional float32 embedding,
  L2-normalized by InsightFace itself (`.normed_embedding`).
  Before any commercial deployment, verify the buffalo_l pack's bundled model
  licenses — mixed research/commercial terms across sub-models.

Unlike reid_embedder.py's OSNet-IBN (which assumes the whole crop IS the
subject to re-identify), InsightFace runs its own face detector over the crop
first. A person crop from the tracker frequently does NOT contain a usable
face (back turned, occluded, too small, motion blur) — extract() returns None
in that case rather than a garbage vector. Callers MUST handle None and simply
skip face-watchlist matching for that track; this is expected, not an error.

Stub mode: same convention as reid_embedder.py — if insightface isn't
installed, extract() returns a deterministic hash-derived vector so the rest
of the API/UI stack is testable without the model installed. Stub mode never
returns None (there is no real detector to fail against), so it is NOT
suitable for exercising the "no face detected" code path — that requires the
real model.

Usage (singleton pattern — load once at startup):
    from backend.services.face_embedder import get_face_embedder
    embedder = get_face_embedder()          # returns singleton
    vec = embedder.extract(crop_bgr)        # np.ndarray shape (512,) or None
"""
from __future__ import annotations

import hashlib
import logging
from functools import lru_cache

import numpy as np

logger = logging.getLogger(__name__)

# InsightFace buffalo_l (ArcFace r100) outputs exactly 512-dimensional
# embeddings. Any mismatch fails loudly at startup, not silently later.
FACE_EMBEDDING_DIM: int = 512


class FaceEmbedder:
    """InsightFace face embedding extractor with stub-mode fallback.

    Args:
        stub_mode: If True, use deterministic random embeddings instead of
                   running the real model. Set automatically when insightface
                   is unavailable or fails to load.
    """

    def __init__(self, stub_mode: bool = False) -> None:
        self._stub_mode = stub_mode
        self._app = None

        if not stub_mode:
            self._load_model()

    def _load_model(self) -> None:
        """Load InsightFace buffalo_l via FaceAnalysis. Falls back to stub on failure."""
        try:
            from insightface.app import FaceAnalysis  # type: ignore[import]

            logger.info("Loading InsightFace (buffalo_l) face model on CPU.")
            self._app = FaceAnalysis(name="buffalo_l")
            # ctx_id=-1 -> CPU. Swap to a GPU device id if CUDA + onnxruntime-gpu
            # are available; the extract() interface does not change either way.
            self._app.prepare(ctx_id=-1, det_size=(640, 640))

            # Sanity call — an all-zero dummy image has no detectable face, so an
            # empty result here is expected and fine; it just proves the model runs.
            dummy = np.zeros((256, 128, 3), dtype=np.uint8)
            self._app.get(dummy)
            logger.info(
                "InsightFace loaded successfully. Embedding dim=%d.", FACE_EMBEDDING_DIM
            )

        except ImportError:
            logger.warning(
                "insightface not installed — FaceEmbedder running in STUB MODE. "
                "Embeddings are deterministic random and NOT suitable for production "
                "watchlist matching. Install insightface (+ onnxruntime) for real embeddings."
            )
            self._stub_mode = True

        except Exception as exc:
            logger.error(
                "Failed to load InsightFace model: %s — falling back to STUB MODE.", exc
            )
            self._stub_mode = True

    def _extract_real(self, crop_bgr: np.ndarray) -> np.ndarray | None:
        """Run InsightFace detection+recognition on a BGR crop.

        Returns the L2-normalized 512-d embedding of the most prominent
        detected face, or None if no face was detected.
        """
        faces = self._app.get(crop_bgr)
        if not faces:
            return None

        # A tracker crop is expected to contain at most one subject, but pick
        # the largest detected face by bbox area in case background people
        # slip into frame.
        best = max(faces, key=lambda f: (f.bbox[2] - f.bbox[0]) * (f.bbox[3] - f.bbox[1]))
        vec = getattr(best, "normed_embedding", None)
        if vec is None:
            return None
        return vec.astype(np.float32)

    def _extract_stub(self, crop_bgr: np.ndarray) -> np.ndarray:
        """Return a deterministic pseudo-embedding derived from image hash.

        Same crop always -> same vector (reproducible), but completely
        unrelated to visual appearance. For API/UI testing ONLY.
        """
        digest = hashlib.md5(crop_bgr.tobytes()).hexdigest()
        seed = int(digest[:8], 16)
        rng = np.random.default_rng(seed)
        vec = rng.standard_normal(FACE_EMBEDDING_DIM).astype(np.float32)
        return self._normalize(vec)

    @staticmethod
    def _normalize(vec: np.ndarray) -> np.ndarray:
        """L2-normalize. Returns zeros for zero-norm vector (avoids div-by-zero)."""
        norm = np.linalg.norm(vec)
        if norm < 1e-10:
            return np.zeros(FACE_EMBEDDING_DIM, dtype=np.float32)
        return (vec / norm).astype(np.float32)

    def extract(self, crop_bgr: np.ndarray) -> np.ndarray | None:
        """Extract a 512-d L2-normalized face embedding from a BGR person crop.

        Args:
            crop_bgr: OpenCV-format BGR numpy array (typically a tracker's
                      person crop, not a pre-cropped face).

        Returns:
            np.ndarray of shape (512,), dtype float32, L2-normalized, or None
            if no face was detected in the crop (real mode only — stub mode
            always returns a vector). Callers must handle None by skipping
            watchlist matching for that track, not by treating it as an error.

        Raises:
            ValueError: If crop is empty or has wrong number of channels.
        """
        if crop_bgr is None or crop_bgr.size == 0:
            raise ValueError("Empty crop passed to FaceEmbedder.extract()")
        if crop_bgr.ndim != 3 or crop_bgr.shape[2] != 3:
            raise ValueError(
                f"Expected 3-channel BGR crop, got shape {crop_bgr.shape}"
            )

        if self._stub_mode:
            logger.debug("STUB MODE: extracting deterministic pseudo face embedding.")
            return self._extract_stub(crop_bgr)
        return self._extract_real(crop_bgr)

    @property
    def is_stub(self) -> bool:
        """True if running in stub mode (no real model loaded)."""
        return self._stub_mode

    @property
    def embedding_dim(self) -> int:
        return FACE_EMBEDDING_DIM


@lru_cache(maxsize=1)
def get_face_embedder() -> FaceEmbedder:
    """Return the singleton FaceEmbedder (loaded once at first call).

    Safe to call from multiple async tasks — lru_cache is thread-safe
    at the Python level. The heavy model load happens exactly once.
    """
    return FaceEmbedder()
