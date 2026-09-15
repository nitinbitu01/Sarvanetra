"""
backend/seed_data.py — Idempotent database and FAISS seeding script.

Safe to run multiple times — it will never create duplicate rows or
duplicate FAISS vectors. On each run it logs a summary of what's in
the DB and index so you can verify the expected counts at a glance.

What this script does:
  1. Creates all SQLite tables (CREATE TABLE IF NOT EXISTS via init_db).
  2. Loads / generates the InsightFace face embedding for the demo
     wanted person → inserts into watchlist_persons if not already present.
  3. Inserts the demo stolen plate if not already present.
  4. Registers 6 demo cameras from config.yaml.
  5. Loads / generates 15 OSNet body embeddings → adds to FAISS if not
     already present (checked by clip_id in id_map.json).
  6. Saves FAISS index + id_map.
  7. Logs summary table.

Pre-computed .npy files (Day 0 prep):
  - data/prep/watchlist_face_001.npy     — InsightFace face embedding (512-d)
  - data/prep/osnet_ref_clip_XX.npy      — 15 OSNet body embeddings (512-d)
  If these files are absent (common on a fresh repo clone), this script
  generates reproducible synthetic embeddings so the schema and index are
  always valid. Synthetic embeddings are seeded from a fixed random state
  so re-runs produce identical vectors.

Run:
    python -m backend.seed_data
  or:
    python backend/seed_data.py
"""

from __future__ import annotations

import logging
import sys
from pathlib import Path

import numpy as np
import yaml

# ── Path setup ────────────────────────────────────────────────────────────────
_PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))

import importlib.util

_db_file = _PROJECT_ROOT / "backend" / "db.py"
_spec = importlib.util.spec_from_file_location("backend_db_module", _db_file)
db = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(db)

from backend.embedding_utils import encode_embedding, decode_embedding, is_valid_embedding
from backend.faiss_index import FaissReIDIndex

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
logger = logging.getLogger("seed_data")


# ── Demo watchlist data ───────────────────────────────────────────────────────
WATCHLIST_PERSON = {
    "name": "Rahul Demo-Suspect",
    "reason": "wanted",
    "reference_photo": "data/prep/watchlist_face_ref_001.jpg",
    "npy_file": "data/prep/watchlist_face_001.npy",
}

WATCHLIST_VEHICLE = {
    "plate_number": "GJ05AB1234",
    "reason": "stolen",
}

# Clip IDs for the 15 reference OSNet embeddings
OSNET_CLIP_IDS = [f"clip_{i:02d}" for i in range(1, 16)]
OSNET_LABELS = [
    "Suspect crossing MG Road",
    "Unknown male near railway gate",
    "Person loitering at stadium",
    "Suspect re-entry GIFT City",
    "Unknown female Sabarmati south",
    "Male suspect Surat bourse",
    "Unidentified person clip 07",
    "Reference track clip 08",
    "Reference track clip 09",
    "Reference track clip 10",
    "Reference track clip 11",
    "Reference track clip 12",
    "Reference track clip 13",
    "Reference track clip 14",
    "Reference track clip 15",
]

OSNET_CAMERA_IDS = [
    "CAM-01", "CAM-02", "CAM-03", "CAM-04", "CAM-05", "CAM-06",
    "CAM-01", "CAM-02", "CAM-03", "CAM-04", "CAM-05", "CAM-06",
    "CAM-01", "CAM-02", "CAM-03",
]


# ── Helpers ───────────────────────────────────────────────────────────────────

def _load_or_generate_embedding(
    npy_path: str,
    dim: int,
    seed: int,
    label: str,
) -> np.ndarray:
    """Load a .npy embedding or generate a reproducible synthetic one.

    If the .npy file exists: load and validate it.
    If missing: generate a random float32 vector from a fixed seed, log a warning,
    and save it so future runs are consistent.

    Args:
        npy_path: Path to the .npy file.
        dim: Expected embedding dimension.
        seed: RNG seed for reproducible synthetic generation.
        label: Human-readable label for log messages.

    Returns:
        float32 numpy array of length dim.
    """
    p = _PROJECT_ROOT / npy_path
    if p.exists():
        vec = np.load(str(p)).astype(np.float32).flatten()
        if len(vec) != dim:
            logger.warning(
                "%s: .npy has dim=%d but expected dim=%d — regenerating synthetic.",
                label, len(vec), dim,
            )
            vec = _synthetic_embedding(dim, seed)
        elif not is_valid_embedding(vec):
            logger.warning("%s: .npy contains invalid values — regenerating synthetic.", label)
            vec = _synthetic_embedding(dim, seed)
        else:
            logger.debug("Loaded real embedding: %s (dim=%d)", npy_path, dim)
        return vec
    else:
        logger.warning(
            "Day 0 prep file missing: %s — generating reproducible synthetic embedding "
            "(seed=%d). Replace with real embedding before live demo.",
            npy_path, seed,
        )
        vec = _synthetic_embedding(dim, seed)
        p.parent.mkdir(parents=True, exist_ok=True)
        np.save(str(p), vec)
        return vec


def _synthetic_embedding(dim: int, seed: int) -> np.ndarray:
    """Generate a reproducible random unit-norm float32 vector."""
    rng = np.random.default_rng(seed)
    vec = rng.standard_normal(dim).astype(np.float32)
    return vec / np.linalg.norm(vec)


# ── Main seeding function ─────────────────────────────────────────────────────

def seed(cfg: dict) -> None:
    """Run the full idempotent seeding process.

    Args:
        cfg: Parsed config.yaml dict.
    """
    db_path = str(_PROJECT_ROOT / cfg["database"]["path"])
    dim: int = cfg["faiss"]["embedding_dim"]

    logger.info("=" * 60)
    logger.info("Sentinel Gujarat — Day 3 Data Seeding")
    logger.info("DB: %s | FAISS dim: %d", db_path, dim)
    logger.info("=" * 60)

    # ── 1. Init DB schema ────────────────────────────────────────────────
    logger.info("[1/5] Initialising database schema...")
    db.init_db(db_path)
    assert db.verify_wal_mode(db_path), "WAL mode failed to enable!"
    logger.info("      WAL mode: ✓")

    # ── 2. Seed cameras ─────────────────────────────────────────────────
    logger.info("[2/5] Registering demo cameras...")
    cameras_cfg = cfg.get("demo_cameras", [])
    for cam in cameras_cfg:
        cam_id = cam.get("id") or cam.get("camera_id")
        lat = cam.get("lat") if cam.get("lat") is not None else cam.get("gps_lat")
        lon = cam.get("lon") if cam.get("lon") is not None else cam.get("gps_lon")
        db.insert_camera(
            db_path,
            camera_id=cam_id,
            name=cam["name"],
            zone=cam["zone"],
            gps_lat=lat,
            gps_lon=lon,
            status="ONLINE",
        )
    logger.info("      Registered %d cameras.", len(cameras_cfg))

    # ── 3. Seed watchlist person (InsightFace face embedding) ────────────
    logger.info("[3/5] Seeding watchlist person...")
    existing_persons = db.get_active_watchlist_persons(db_path)
    person_names = [row["name"] for row in existing_persons]

    if WATCHLIST_PERSON["name"] not in person_names:
        face_vec = _load_or_generate_embedding(
            npy_path=WATCHLIST_PERSON["npy_file"],
            dim=dim,
            seed=42,
            label=WATCHLIST_PERSON["name"],
        )
        blob = encode_embedding(face_vec)
        db.insert_watchlist_person(
            db_path,
            name=WATCHLIST_PERSON["name"],
            reason=WATCHLIST_PERSON["reason"],
            face_embedding=blob,
            embedding_dim=dim,
            reference_photo_path=WATCHLIST_PERSON.get("reference_photo"),
        )
        logger.info("      Watchlist person added: %s", WATCHLIST_PERSON["name"])
    else:
        logger.info("      Watchlist person already present (skip): %s", WATCHLIST_PERSON["name"])

    # ── 4. Seed watchlist vehicle ────────────────────────────────────────
    logger.info("[4/5] Seeding watchlist vehicle...")
    existing_veh = db.is_plate_on_watchlist(db_path, WATCHLIST_VEHICLE["plate_number"])
    if existing_veh is None:
        db.insert_watchlist_vehicle(
            db_path,
            plate_number=WATCHLIST_VEHICLE["plate_number"],
            reason=WATCHLIST_VEHICLE["reason"],
        )
        logger.info("      Watchlist vehicle added: %s", WATCHLIST_VEHICLE["plate_number"])
    else:
        logger.info("      Watchlist vehicle already present (skip): %s", WATCHLIST_VEHICLE["plate_number"])

    # ── 5. Seed FAISS reference index ────────────────────────────────────
    logger.info("[5/5] Seeding FAISS ReID reference index...")
    index_path = _PROJECT_ROOT / cfg["faiss"]["index_path"]
    id_map_path = _PROJECT_ROOT / cfg["faiss"]["id_map_path"]
    index_path.parent.mkdir(parents=True, exist_ok=True)

    # Load existing index or create fresh
    if index_path.exists() and id_map_path.exists():
        try:
            faiss_idx = FaissReIDIndex.load(cfg)
            logger.info("      Loaded existing index with %d vectors.", faiss_idx.ntotal)
        except (ValueError, FileNotFoundError) as exc:
            logger.warning("      Existing index corrupt (%s) — rebuilding.", exc)
            faiss_idx = FaissReIDIndex(dim)
    else:
        faiss_idx = FaissReIDIndex(dim)
        logger.info("      Created fresh FAISS index (dim=%d).", dim)

    added_count = 0
    for i, clip_id in enumerate(OSNET_CLIP_IDS):
        if faiss_idx.has_clip_id(clip_id):
            logger.debug("      FAISS: clip_id=%s already present — skip.", clip_id)
            continue

        npy_file = f"data/prep/osnet_ref_{clip_id}.npy"
        vec = _load_or_generate_embedding(
            npy_path=npy_file,
            dim=dim,
            seed=1000 + i,   # reproducible per-clip seed
            label=f"OSNet {clip_id}",
        )

        metadata = {
            "clip_id": clip_id,
            "label": OSNET_LABELS[i],
            "camera_id": OSNET_CAMERA_IDS[i],
            "npy_path": npy_file,
        }
        faiss_idx.add_vector(vec, metadata)
        added_count += 1

    faiss_idx.save(cfg)
    logger.info(
        "      FAISS: added %d new vectors → total %d vectors in index.",
        added_count, faiss_idx.ntotal,
    )

    # ── Summary ─────────────────────────────────────────────────────────
    counts = db.get_table_counts(db_path)
    logger.info("")
    logger.info("=" * 60)
    logger.info("SEEDING COMPLETE — Summary")
    logger.info("  Cameras:            %d", counts["cameras"])
    logger.info("  Watchlist persons:  %d", counts["watchlist_persons"])
    logger.info("  Watchlist vehicles: %d", counts["watchlist_vehicles"])
    logger.info("  FAISS vectors:      %d", faiss_idx.ntotal)
    logger.info("  Tracks (DB):        %d  (grows at runtime)", counts["tracks"])
    logger.info("  Alerts (DB):        %d  (grows at runtime)", counts["alerts"])
    logger.info("=" * 60)

    # ── Self-verification ────────────────────────────────────────────────
    _run_self_verification(db_path, faiss_idx, dim, cfg)


def _run_self_verification(
    db_path: str,
    faiss_idx: FaissReIDIndex,
    dim: int,
    cfg: dict,
) -> None:
    """Run the mandatory correctness checks documented in the spec.

    1. Embedding round-trip: stored BLOB → decode → check shape/dtype/validity.
    2. FAISS self-query: seeded vector queries itself → score ≈ 1.0.
    3. WAL mode confirmed.
    """
    logger.info("")
    logger.info("Running self-verification checks...")
    errors: list[str] = []

    # Check 1: Embedding round-trip
    persons = db.get_active_watchlist_persons(db_path)
    if persons:
        row = persons[0]
        blob = bytes(row["face_embedding"])
        stored_dim = row["embedding_dim"]
        try:
            vec = decode_embedding(blob, stored_dim)
            if vec.shape != (stored_dim,):
                errors.append(f"Embedding shape mismatch: {vec.shape} vs ({stored_dim},)")
            elif vec.dtype != np.float32:
                errors.append(f"Embedding dtype wrong: {vec.dtype}")
            elif not is_valid_embedding(vec):
                errors.append("Embedding contains NaN / zeros / inf")
            else:
                logger.info("  [✓] Embedding round-trip: shape=%s dtype=%s first_3=%s",
                            vec.shape, vec.dtype, vec[:3].tolist())
        except ValueError as exc:
            errors.append(f"decode_embedding failed: {exc}")
    else:
        errors.append("No watchlist persons in DB — cannot verify round-trip")

    # Check 2: FAISS self-query
    if faiss_idx.ntotal > 0:
        # Pick the first seeded clip, reload its .npy (same vector that was inserted)
        first_meta = faiss_idx.get_id_map()[0]
        npy_file = str(_PROJECT_ROOT / first_meta["npy_path"])
        test_vec = np.load(npy_file).astype(np.float32).flatten()
        results = faiss_idx.search(test_vec, k=1)
        if results:
            score = results[0]["score"]
            returned_clip = results[0].get("clip_id", "?")
            expected_clip = first_meta["clip_id"]
            if abs(score - 1.0) > 0.01:
                errors.append(f"FAISS self-query score={score:.4f} (expected ≈1.0) — normalization bug?")
            elif returned_clip != expected_clip:
                errors.append(f"FAISS self-query returned clip_id={returned_clip} (expected {expected_clip})")
            else:
                logger.info("  [✓] FAISS self-query: clip_id=%s score=%.6f ≈ 1.0", returned_clip, score)
        else:
            errors.append("FAISS search returned no results")
    else:
        errors.append("FAISS index is empty — cannot self-verify")

    # Check 3: WAL mode
    if db.verify_wal_mode(db_path):
        logger.info("  [✓] SQLite WAL mode: active")
    else:
        errors.append("SQLite WAL mode NOT active")

    # Result
    if errors:
        logger.error("SELF-VERIFICATION FAILED:")
        for e in errors:
            logger.error("  ✗ %s", e)
        raise RuntimeError("Seeding self-verification failed — see errors above.")
    else:
        logger.info("  ALL CHECKS PASSED ✓")


if __name__ == "__main__":
    cfg_path = _PROJECT_ROOT / "config.yaml"
    with cfg_path.open("r", encoding="utf-8") as fh:
        cfg = yaml.safe_load(fh)
    seed(cfg)
