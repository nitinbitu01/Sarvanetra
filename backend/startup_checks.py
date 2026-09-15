"""
backend/startup_checks.py — Pre-flight validator for Sentinel Gujarat.

Run this at process startup (before any thread is spawned) to catch
configuration errors, missing files, and broken dependencies with clear,
actionable error messages rather than obscure tracebacks mid-run.

Usage (called automatically by main.py and backend/main.py at startup):
    from backend.startup_checks import run_startup_checks
    run_startup_checks(cfg, source=args.source)  # exits with code 1 on failure

Each check is independent. All failures are collected and reported together
so the operator can fix everything in one pass instead of discovering errors
one by one.
"""

from __future__ import annotations

import logging
import sys
from pathlib import Path
from typing import Any

logger = logging.getLogger(__name__)

# ── Required config keys and their expected types ─────────────────────────────
# Format: "section.key" → expected Python type
_REQUIRED_KEYS: list[tuple[str, type]] = [
    ("model.path",                          str),
    ("model.device",                        str),
    ("detection.confidence_threshold",      float),
    ("detection.class_filter",              list),
    ("tracking.tracker_config",             str),
    ("tracking.min_confirmation_frames",    int),
    ("processing.fps",                      int),
    ("processing.input_width",              int),
    ("output.events_file",                  str),
    ("output.crops_dir",                    str),
    ("output.crop_save_interval_frames",    int),
    ("camera.id",                           str),
    ("camera.loop_video",                   bool),
    ("database.path",                       str),
    ("faiss.index_path",                    str),
    ("faiss.id_map_path",                   str),
    ("faiss.embedding_dim",                 int),
    ("anpr.read_interval_frames",           int),
    ("anpr.max_readings_per_track",         int),
    ("anpr.min_readings_before_finalize",   int),
    ("anpr.uncertain_confidence_threshold", float),
    ("anpr.plate_regex",                    str),
    ("anpr.ocr_worker_queue_maxsize",       int),
]


# Module-level sentinel so _get_nested can safely detect missing keys
_MISSING = object()


def _get_nested(cfg: dict, dotted_key: str) -> Any:
    """Traverse a nested dict using dot-notation key. Returns _MISSING on absence."""
    parts = dotted_key.split(".")
    node = cfg
    for part in parts:
        if not isinstance(node, dict) or part not in node:
            return _MISSING
        node = node[part]
    return node


def check_config_schema(cfg: dict[str, Any]) -> list[str]:
    """Validate all required config keys exist and have correct types.

    Returns:
        List of error strings (empty = no errors).
    """
    errors: list[str] = []

    for dotted_key, expected_type in _REQUIRED_KEYS:
        val = _get_nested(cfg, dotted_key)
        if val is _MISSING:
            errors.append(f"config.yaml: missing required key '{dotted_key}'")
        elif not isinstance(val, expected_type):
            errors.append(
                f"config.yaml: '{dotted_key}' must be {expected_type.__name__}, "
                f"got {type(val).__name__} (value={val!r})"
            )

    # Range checks
    fps = _get_nested(cfg, "processing.fps")
    if fps is not _MISSING and isinstance(fps, int) and not (1 <= fps <= 120):
        errors.append(f"config.yaml: 'processing.fps' must be 1-120, got {fps}")

    conf = _get_nested(cfg, "detection.confidence_threshold")
    if conf is not _MISSING and isinstance(conf, float) and not (0.0 < conf < 1.0):
        errors.append(
            f"config.yaml: 'detection.confidence_threshold' must be in (0, 1), got {conf}"
        )

    dim = _get_nested(cfg, "faiss.embedding_dim")
    if dim is not _MISSING and isinstance(dim, int) and dim not in (128, 256, 512, 1024, 2048):
        errors.append(
            f"config.yaml: 'faiss.embedding_dim'={dim} is unusual -- expected 128/256/512/1024/2048"
        )

    return errors


def check_video_source(source: str) -> list[str]:
    """Verify the video source exists (file path) or is a valid webcam index.

    Returns:
        List of error strings (empty = no errors).
    """
    errors: list[str] = []
    if source.isdigit():
        # Webcam — can't validate without opening, just note it
        logger.debug("startup_checks: webcam source (index %s) — skipping file check", source)
        return errors

    if any(source.startswith(prefix) for prefix in ("rtsp://", "http://", "https://", "rtmp://")):
        logger.debug("startup_checks: remote stream URL detected (%s) — accepted", source)
        return errors

    p = Path(source)
    if not p.exists():
        errors.append(f"Video source not found: '{source}' — check --source argument")
    elif not p.is_file():
        errors.append(f"Video source is not a file: '{source}'")
    elif p.suffix.lower() not in {".mp4", ".avi", ".mov", ".mkv", ".webm", ".ts", ".m4v"}:
        # Warning, not error — OpenCV may still support it
        logger.warning(
            "startup_checks: video extension '%s' is unusual — OpenCV may not support it",
            p.suffix,
        )
    return errors


def check_model_file(cfg: dict[str, Any]) -> list[str]:
    """Check that the YOLOv8 model file exists or is a known auto-downloadable name.

    Returns:
        List of error strings (empty = no errors).
    """
    errors: list[str] = []
    model_path = cfg.get("model", {}).get("path", "")
    p = Path(model_path)

    auto_downloadable = {"yolov8n.pt", "yolov8s.pt", "yolov8m.pt", "yolov8l.pt", "yolov8x.pt"}
    if p.name in auto_downloadable:
        if not p.exists():
            logger.info(
                "startup_checks: model '%s' not found locally — ultralytics will auto-download it on first run.",
                model_path,
            )
        # Not an error — ultralytics downloads automatically
    elif not p.exists():
        errors.append(
            f"Model file not found: '{model_path}'. "
            f"Use a standard name (yolov8n.pt etc.) for auto-download, or fix the path."
        )
    return errors


def check_database(cfg: dict[str, Any]) -> list[str]:
    """Verify the SQLite database exists and is readable.

    Returns:
        List of error strings (empty = no errors).
    """
    errors: list[str] = []
    db_path = cfg.get("database", {}).get("path", "")
    if not db_path:
        errors.append("config.yaml: 'database.path' is empty")
        return errors

    p = Path(db_path)
    if not p.exists():
        errors.append(
            f"Database not found: '{db_path}'. "
            f"Run 'python -m backend.seed_data' to initialise it."
        )
        return errors

    # Quick read-only connectivity check
    try:
        import sqlite3
        conn = sqlite3.connect(str(p))
        tables = conn.execute(
            "SELECT name FROM sqlite_master WHERE type='table'"
        ).fetchall()
        conn.close()
        table_names = {row[0] for row in tables}
        required_tables = {"cameras", "tracks", "alerts", "watchlist_persons", "watchlist_vehicles"}
        missing = required_tables - table_names
        if missing:
            errors.append(
                f"Database '{db_path}' is missing tables: {sorted(missing)}. "
                f"Run 'python -m backend.seed_data' to re-initialise."
            )
    except Exception as exc:
        errors.append(f"Database '{db_path}' could not be opened: {exc}")

    return errors


def check_faiss_index(cfg: dict[str, Any]) -> list[str]:
    """Verify the FAISS index and id_map exist and have the correct dimension.

    Returns:
        List of error strings (empty = no errors).
    """
    errors: list[str] = []
    faiss_cfg = cfg.get("faiss", {})
    index_path = faiss_cfg.get("index_path", "")
    id_map_path = faiss_cfg.get("id_map_path", "")
    expected_dim = faiss_cfg.get("embedding_dim", 512)

    if not Path(index_path).exists():
        errors.append(
            f"FAISS index not found: '{index_path}'. "
            f"Run 'python -m backend.seed_data' to create it."
        )
    else:
        # Check dimension matches config
        try:
            import faiss
            idx = faiss.read_index(index_path)
            if idx.d != expected_dim:
                errors.append(
                    f"FAISS index dim mismatch: index has d={idx.d} "
                    f"but config says faiss.embedding_dim={expected_dim}. "
                    f"Delete the index and re-run seed_data.py."
                )
        except Exception as exc:
            errors.append(f"Could not load FAISS index '{index_path}': {exc}")

    if not Path(id_map_path).exists():
        errors.append(
            f"FAISS id_map not found: '{id_map_path}'. "
            f"Run 'python -m backend.seed_data' to create it."
        )
    else:
        try:
            import json
            data = json.loads(Path(id_map_path).read_text())
            if not isinstance(data, list):
                errors.append(f"id_map '{id_map_path}' is not a JSON array — re-run seed_data.py")
        except Exception as exc:
            errors.append(f"Could not parse id_map '{id_map_path}': {exc}")

    return errors


def check_tracker_config(cfg: dict[str, Any]) -> list[str]:
    """Verify the BoT-SORT YAML config exists.

    Returns:
        List of error strings (empty = no errors).
    """
    errors: list[str] = []
    tracker_cfg = cfg.get("tracking", {}).get("tracker_config", "")
    if tracker_cfg and not Path(tracker_cfg).exists():
        errors.append(
            f"Tracker config not found: '{tracker_cfg}'. "
            f"Ensure botsort.yaml is in the project root."
        )
    return errors


def run_startup_checks(
    cfg: dict[str, Any],
    source: str | None = None,
    fail_fast: bool = True,
) -> bool:
    """Run all pre-flight checks. Log errors and optionally exit.

    Args:
        cfg: Parsed config dict.
        source: Video source path (optional — skipped if None).
        fail_fast: If True, call sys.exit(1) on any failure.
                   If False, return False (for testing).

    Returns:
        True if all checks pass, False if any fail (only when fail_fast=False).
    """
    logger.info("Running startup pre-flight checks...")
    all_errors: list[str] = []

    all_errors.extend(check_config_schema(cfg))
    all_errors.extend(check_model_file(cfg))
    all_errors.extend(check_tracker_config(cfg))

    if source is not None:
        all_errors.extend(check_video_source(source))

    all_errors.extend(check_database(cfg))
    all_errors.extend(check_faiss_index(cfg))

    if all_errors:
        logger.error("Startup pre-flight FAILED — %d error(s):", len(all_errors))
        for i, err in enumerate(all_errors, 1):
            logger.error("  [%d] %s", i, err)
        if fail_fast:
            sys.exit(1)
        return False

    logger.info("Startup pre-flight checks passed (%d checks).", len(_REQUIRED_KEYS) + 3)
    return True
