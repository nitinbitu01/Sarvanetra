"""
embedding_utils.py — Single embedding serialization convention for Sentinel Gujarat.

CRITICAL: Every module that stores or retrieves an embedding (face or body)
MUST use encode_embedding() / decode_embedding() from this file.
Do NOT invent per-file alternatives — inconsistent byte ordering or dtype
casting will silently corrupt round-trip accuracy and break FAISS similarity.

Convention:
  - All embeddings are stored as float32 numpy arrays.
  - Serialized to raw bytes with .astype(np.float32).tobytes()
  - Deserialized with np.frombuffer(..., dtype=np.float32).reshape(dim)
  - The embedding_dim is stored alongside every BLOB column in the DB so
    decode_embedding() never has to guess shape.

Usage:
    from embedding_utils import encode_embedding, decode_embedding, normalize_l2

    blob   = encode_embedding(vector)          # store in DB
    vector = decode_embedding(blob, dim=512)   # retrieve from DB
    vector = normalize_l2(vector)              # before FAISS insert/query
"""

from __future__ import annotations

import numpy as np


def encode_embedding(vector: np.ndarray) -> bytes:
    """Serialize a float32 numpy vector for SQLite BLOB storage.

    Args:
        vector: 1-D numpy array of any float dtype.

    Returns:
        Raw bytes in float32 little-endian layout.

    Example:
        >>> blob = encode_embedding(np.array([0.1, 0.2, 0.3]))
        >>> len(blob)  # 3 floats × 4 bytes
        12
    """
    return vector.astype(np.float32).tobytes()


def decode_embedding(blob: bytes, dim: int) -> np.ndarray:
    """Deserialize a SQLite BLOB back into a float32 numpy vector.

    Args:
        blob: Raw bytes produced by encode_embedding().
        dim:  Expected vector length (used to validate and reshape).

    Returns:
        1-D numpy float32 array of length `dim`.

    Raises:
        ValueError: If byte count doesn't match dim × 4 bytes.

    Example:
        >>> v = np.array([0.1, 0.2, 0.3], dtype=np.float32)
        >>> assert np.allclose(decode_embedding(encode_embedding(v), 3), v)
    """
    expected_bytes = dim * 4  # float32 = 4 bytes per element
    if len(blob) != expected_bytes:
        raise ValueError(
            f"decode_embedding: expected {expected_bytes} bytes for dim={dim}, "
            f"got {len(blob)} bytes. Wrong dim or corrupted blob."
        )
    return np.frombuffer(blob, dtype=np.float32).reshape(dim)


def normalize_l2(vector: np.ndarray) -> np.ndarray:
    """L2-normalize a vector so that inner product equals cosine similarity.

    This MUST be applied before adding to or querying a faiss.IndexFlatIP
    index. Skipping normalization silently produces wrong similarity scores.

    Args:
        vector: Any float numpy array.

    Returns:
        Unit-norm float32 array. If norm is 0 (zero vector), returns zeros
        to avoid division-by-zero rather than raising.

    Example:
        >>> v = normalize_l2(np.array([3.0, 4.0]))
        >>> round(np.linalg.norm(v), 6)
        1.0
    """
    norm = np.linalg.norm(vector)
    if norm < 1e-10:
        return np.zeros_like(vector, dtype=np.float32)
    return (vector / norm).astype(np.float32)


def embedding_shape_ok(blob: bytes, dim: int) -> bool:
    """Quick sanity check — does this BLOB decode to the expected dimension?

    Use before passing a blob to decode_embedding() when you're not 100% sure
    of its origin (e.g., reading from a DB row written by a different day's code).
    """
    if blob is None or not isinstance(blob, (bytes, bytearray)):
        return False
    return len(blob) == dim * 4


def is_valid_embedding(vector: np.ndarray) -> bool:
    """Return True if the vector looks like a plausible embedding.

    Checks: no NaNs, no all-zeros, finite values only.
    """
    return (
        vector.ndim == 1
        and np.isfinite(vector).all()
        and not np.allclose(vector, 0.0)
    )
