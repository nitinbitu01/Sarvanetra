"""
backend/config.py — Config loader for the FastAPI server.

Loads config.yaml from the project root (parent of this file's directory).
Single source of truth — the same config.yaml used by Day 1's CLI.
"""

from __future__ import annotations

import sys
from pathlib import Path
from typing import Any

import yaml

# Project root is one level above backend/
PROJECT_ROOT = Path(__file__).resolve().parent.parent


def load_config(config_path: str | None = None) -> dict[str, Any]:
    """Load config.yaml from the project root.

    Args:
        config_path: Override path. Defaults to PROJECT_ROOT/config.yaml.

    Returns:
        Parsed config dict.

    Raises:
        SystemExit: If the file is missing or malformed.
    """
    path = Path(config_path) if config_path else PROJECT_ROOT / "config.yaml"

    if not path.exists():
        print(f"[ERROR] Config file not found: {path}", file=sys.stderr)
        sys.exit(1)

    try:
        with path.open("r", encoding="utf-8") as fh:
            return yaml.safe_load(fh)
    except yaml.YAMLError as exc:
        print(f"[ERROR] Failed to parse config.yaml: {exc}", file=sys.stderr)
        sys.exit(1)
