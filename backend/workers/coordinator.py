import asyncio
import logging
import multiprocessing as mp
import os
import sys
import threading
from typing import Optional, List, Dict
import yaml

logger = logging.getLogger("sentinel.coordinator")


def _worker_entrypoint(worker_id: int, cam_configs: list, config: dict):
    """Entry point for each AI worker process"""
    import asyncio
    from backend.workers.ai_worker import AIWorker

    os.environ["METRICS_PORT"] = str(9090 + worker_id)
    worker = AIWorker(worker_id, cam_configs, config)
    asyncio.run(worker.run())


class MultiProcessCoordinator:
    """
    Coordinator managing 3 workers (10 cameras each).
    Supports multi-process or async task execution.
    """

    WORKER_COUNT = int(os.getenv("WORKER_COUNT", "3"))

    def __init__(self, config_or_path: str | dict = "config.yaml"):
        if isinstance(config_or_path, dict):
            self.config = config_or_path
        elif isinstance(config_or_path, str) and os.path.exists(config_or_path):
            with open(config_or_path, "r", encoding="utf-8") as f:
                self.config = yaml.safe_load(f) or {}
        else:
            self.config = {}

        self._processes: list[mp.Process] = []
        self._async_tasks: list[asyncio.Task] = []
        self._running = False

    def start(self, loop: Optional[asyncio.AbstractEventLoop] = None):
        cameras = self.config.get("demo_cameras", [])
        if not cameras:
            logger.warning("No demo_cameras in config — using standard 30 cameras default")
            cameras = [{"id": f"CAM_{i:02d}", "name": f"Camera {i:02d}", "rtsp_url": f"demo://cam{i:02d}"} for i in range(1, 31)]

        chunk_size = max(1, len(cameras) // self.WORKER_COUNT)
        chunks = [
            cameras[i * chunk_size: (i + 1) * chunk_size]
            for i in range(self.WORKER_COUNT)
        ]
        remainder = cameras[self.WORKER_COUNT * chunk_size:]
        if remainder and chunks:
            chunks[-1].extend(remainder)

        self._running = True

        from backend.workers.ai_worker import AIWorker
        for i, chunk in enumerate(chunks, start=1):
            worker = AIWorker(i, chunk, self.config)
            task = asyncio.create_task(worker.run(), name=f"ai_worker_{i}")
            self._async_tasks.append(task)
            logger.info(
                f"✅ AI Worker {i} active — {len(chunk)} cameras: {[c['id'] for c in chunk]}"
            )

    async def run(self):
        """Run worker coordinator loop"""
        self.start()
        try:
            while self._running:
                await asyncio.sleep(2.0)
        except (asyncio.CancelledError, KeyboardInterrupt):
            self.stop()

    def stop(self):
        self._running = False
        for task in self._async_tasks:
            task.cancel()
        for p in self._processes:
            try:
                p.terminate()
            except Exception:
                pass
        logger.info("All AI workers stopped")

    def status(self) -> list[dict]:
        return [
            {
                "worker_id": i + 1,
                "status": "running" if not task.done() else "stopped",
            }
            for i, task in enumerate(self._async_tasks)
        ]
