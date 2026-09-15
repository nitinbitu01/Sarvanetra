"""
backend/workers/main.py
=======================
Docker worker entrypoint — launched by the worker service in docker-compose.yml.

Wraps the existing MultiProcessCoordinator so the same Python package can serve
as both the FastAPI process (api service) and the background processing process
(worker service), keeping the image identical but the command different.
"""
import asyncio
import logging
import os
import sys
from pathlib import Path

# Ensure project root is importable
_ROOT = Path(__file__).resolve().parent.parent.parent
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from backend.core.logging import configure_logging, get_logger
from backend.workers.coordinator import MultiProcessCoordinator
from backend.config import load_config

configure_logging()
logger = get_logger(__name__)


async def main():
    logger.info("Sentinel worker process starting...")

    config_path = Path(os.getenv("CONFIG_PATH", "config.yaml"))
    try:
        config = load_config(str(config_path))
    except Exception as exc:
        logger.warning("Could not load config.yaml (%s) — using defaults.", exc)
        config = {}

    coordinator = MultiProcessCoordinator(config)
    try:
        await coordinator.run()
    except (KeyboardInterrupt, asyncio.CancelledError):
        logger.info("Worker shutdown requested.")
    finally:
        coordinator.stop()
        logger.info("Worker stopped cleanly.")


if __name__ == "__main__":
    asyncio.run(main())
