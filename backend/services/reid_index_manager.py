"""
backend/services/reid_index_manager.py — Concurrency-safe FAISS index for GlobalPerson ReID.

Architecture:
  - Separate from the legacy FaissReIDIndex (backend/faiss_index.py, Day 3).
    That index is clip-keyed and seeded; this one is global_person_id-keyed
    and populated entirely by live ReID decisions.
  - Index type: faiss.IndexFlatIP (inner product on L2-normalized vectors =
    cosine similarity). Range: [-1, 1], 1.0 = identical.
  - Scale upgrade path (documented, not built yet):
    When global_person count grows large (>100k), switch to IndexIVFFlat or
    HNSW for sub-linear search time. Only the index type changes; the manager
    interface stays identical.
  - Deletion: IndexFlatIP does NOT support cheap deletion. Strategy chosen:
    mark deleted in global_persons.is_deleted=True, filter results at search
    time against deleted IDs. Periodic rebuild_excluding_deleted() is called
    by the retention job to reclaim FAISS space after bulk purges.
    This is documented here so a future refactor to HNSW knows the trade-off.

Concurrency:
  - A single asyncio.Lock() guards all reads and writes.
  - All public methods are `async` — call only from the async event loop.
  - For multi-worker deployments (Gunicorn + uvicorn workers), replace the
    asyncio.Lock with a cross-process lock (e.g. filelock) and move index
    persistence to a shared filesystem. Day 6 stays single-process.

Persistence:
  - Index file:  data/faiss_index/global_persons.index
  - ID map file: data/faiss_index/global_persons_idmap.json
    Format: list of {global_person_id: int} indexed by FAISS position.
  - Written on every add and after every rebuild. Read on startup.

Usage:
    from backend.services.reid_index_manager import get_index_manager
    mgr = await get_index_manager()            # singleton, initialized once
    await mgr.add_person(global_person_id=42, embedding=vec)
    results = await mgr.search(query_vec, k=5)
    # → [{"global_person_id": 42, "score": 0.91, "faiss_position": 0}, ...]
"""
from __future__ import annotations

import asyncio
import json
import logging
from pathlib import Path
from typing import Any

import faiss
import numpy as np

from backend.services.reid_embedder import REID_EMBEDDING_DIM

logger = logging.getLogger(__name__)

# Paths for the ReID-specific FAISS index (separate from legacy Day 3 index)
_INDEX_DIR = Path(__file__).resolve().parent.parent.parent / "data" / "faiss_index"
_INDEX_FILE = _INDEX_DIR / "global_persons.index"
_IDMAP_FILE = _INDEX_DIR / "global_persons_idmap.json"


class ReIDIndexManager:
    """Asyncio-locked FAISS IndexFlatIP manager for GlobalPerson vectors.

    All public methods acquire a lock before touching FAISS state.
    Never call from a background thread without wrapping in run_in_executor.
    """

    def __init__(self) -> None:
        self._lock = asyncio.Lock()
        self._index: faiss.IndexFlatIP = faiss.IndexFlatIP(REID_EMBEDDING_DIM)
        # id_map[i] = {"global_person_id": int} for FAISS position i
        self._id_map: list[dict[str, Any]] = []

    # ──────────────────────────────────────────────────────────────────────
    # Startup / persistence
    # ──────────────────────────────────────────────────────────────────────

    async def initialize(self) -> None:
        """Load persisted index from disk on startup. Safe to call multiple times."""
        async with self._lock:
            _INDEX_DIR.mkdir(parents=True, exist_ok=True)

            if _INDEX_FILE.exists() and _IDMAP_FILE.exists():
                try:
                    loaded_index = faiss.read_index(str(_INDEX_FILE))
                    with _IDMAP_FILE.open("r", encoding="utf-8") as fh:
                        loaded_idmap = json.load(fh)

                    if loaded_index.ntotal != len(loaded_idmap):
                        raise ValueError(
                            f"FAISS index ({loaded_index.ntotal} vectors) and "
                            f"id_map ({len(loaded_idmap)} entries) are out of sync. "
                            f"Delete {_INDEX_DIR} to rebuild from DB."
                        )
                    if loaded_index.d != REID_EMBEDDING_DIM:
                        raise ValueError(
                            f"FAISS index dimension mismatch: "
                            f"index.d={loaded_index.d}, expected {REID_EMBEDDING_DIM}. "
                            f"Delete {_INDEX_DIR} to rebuild."
                        )

                    self._index = loaded_index
                    self._id_map = loaded_idmap
                    logger.info(
                        "ReID FAISS index loaded: %d vectors, dim=%d.",
                        self._index.ntotal, REID_EMBEDDING_DIM,
                    )
                except Exception as exc:
                    logger.error(
                        "Failed to load ReID FAISS index: %s — starting fresh. "
                        "Existing global_person records in DB are still intact; "
                        "run rebuild_from_db() to restore the index.",
                        exc,
                    )
                    self._index = faiss.IndexFlatIP(REID_EMBEDDING_DIM)
                    self._id_map = []
            else:
                logger.info(
                    "No persisted ReID FAISS index found — starting empty. "
                    "This is expected on first run."
                )

    def _save(self) -> None:
        """Persist index + id_map. Must be called within the lock."""
        _INDEX_DIR.mkdir(parents=True, exist_ok=True)
        faiss.write_index(self._index, str(_INDEX_FILE))
        with _IDMAP_FILE.open("w", encoding="utf-8") as fh:
            json.dump(self._id_map, fh, indent=2)
        logger.debug(
            "ReID FAISS index persisted: %d vectors.", self._index.ntotal
        )

    # ──────────────────────────────────────────────────────────────────────
    # Public API
    # ──────────────────────────────────────────────────────────────────────

    async def add_person(
        self, global_person_id: int, embedding: np.ndarray
    ) -> int:
        """Add a GlobalPerson's embedding to the index.

        Args:
            global_person_id: PK of the GlobalPerson DB row.
            embedding: L2-normalized 512-d float32 vector.

        Returns:
            FAISS position (0-based) of the new vector.
        """
        async with self._lock:
            norm_vec = self._normalize(embedding).reshape(1, REID_EMBEDDING_DIM)
            self._index.add(norm_vec)
            position = len(self._id_map)
            self._id_map.append({"global_person_id": global_person_id})
            logger.debug(
                "ReID add: global_person_id=%d → faiss_position=%d",
                global_person_id, position,
            )
            self._save()
            return position

    async def search(
        self,
        query_embedding: np.ndarray,
        k: int = 5,
        exclude_deleted_ids: set[int] | None = None,
    ) -> list[dict[str, Any]]:
        """Find k nearest GlobalPersons to query_embedding.

        Args:
            query_embedding: Raw or pre-normalized 512-d vector (normalized inside).
            k: Number of candidates to return.
            exclude_deleted_ids: Set of global_person_id values to filter out
                (soft-deleted records). Pass the set from a DB query for correctness.

        Returns:
            List of result dicts sorted by score descending:
            [{"global_person_id": int, "score": float, "faiss_position": int}, ...]
            Empty list if index has no vectors.
        """
        async with self._lock:
            if self._index.ntotal == 0:
                return []

            k_actual = min(k, self._index.ntotal)
            norm_vec = self._normalize(query_embedding).reshape(1, REID_EMBEDDING_DIM)
            scores, indices = self._index.search(norm_vec, k_actual)

            results = []
            for score, idx in zip(scores[0], indices[0]):
                if idx < 0:   # FAISS padding sentinel
                    continue
                gp_id = self._id_map[idx]["global_person_id"]
                if exclude_deleted_ids and gp_id in exclude_deleted_ids:
                    continue
                results.append({
                    "global_person_id": gp_id,
                    "score": float(score),
                    "faiss_position": int(idx),
                })

            logger.debug(
                "ReID search: top score=%.4f gp_id=%s",
                results[0]["score"] if results else 0.0,
                results[0].get("global_person_id", "—") if results else "—",
            )
            return results

    async def rebuild_excluding_deleted(self, db) -> int:
        """Rebuild the FAISS index excluding soft-deleted GlobalPersons.

        Called by the retention job after bulk purges. Reconstructs the index
        entirely from all non-deleted GlobalPerson rows in the DB.

        Args:
            db: Active SQLAlchemy Session.

        Returns:
            Number of vectors in the rebuilt index.
        """
        from backend.db.models import GlobalPerson
        from backend.embedding_utils import decode_embedding

        async with self._lock:
            new_index = faiss.IndexFlatIP(REID_EMBEDDING_DIM)
            new_id_map: list[dict[str, Any]] = []

            persons = (
                db.query(GlobalPerson)
                .filter(
                    GlobalPerson.is_deleted == False,  # noqa: E712
                    GlobalPerson.representative_embedding.isnot(None),
                )
                .all()
            )

            for person in persons:
                try:
                    vec = decode_embedding(
                        person.representative_embedding, REID_EMBEDDING_DIM
                    )
                    norm_vec = self._normalize(vec).reshape(1, REID_EMBEDDING_DIM)
                    new_index.add(norm_vec)
                    new_position = len(new_id_map)
                    new_id_map.append({"global_person_id": person.id})
                    # Update faiss_index_position in DB
                    person.faiss_index_position = new_position
                except Exception as exc:
                    logger.error(
                        "Failed to rebuild vector for global_person_id=%d: %s",
                        person.id, exc,
                    )

            db.flush()

            self._index = new_index
            self._id_map = new_id_map
            self._save()

            logger.info(
                "ReID FAISS index rebuilt: %d vectors (excluded deleted).",
                new_index.ntotal,
            )
            return new_index.ntotal

    @property
    def ntotal(self) -> int:
        """Number of vectors currently in the index (no lock — read-only int)."""
        return self._index.ntotal

    @staticmethod
    def _normalize(vec: np.ndarray) -> np.ndarray:
        """L2-normalize. Zero vector stays zero."""
        norm = np.linalg.norm(vec)
        if norm < 1e-10:
            return np.zeros(REID_EMBEDDING_DIM, dtype=np.float32)
        return (vec / norm).astype(np.float32)


# ── Singleton ─────────────────────────────────────────────────────────────────

_manager_instance: ReIDIndexManager | None = None


def get_index_manager() -> ReIDIndexManager:
    """Return the singleton ReIDIndexManager.

    Call initialize() once at application startup before using.
    """
    global _manager_instance
    if _manager_instance is None:
        _manager_instance = ReIDIndexManager()
    return _manager_instance
