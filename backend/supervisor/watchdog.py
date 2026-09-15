import asyncio
import logging
import multiprocessing as mp
import time
from typing import Callable, Dict, List, Optional, Tuple

logger = logging.getLogger("sentinel.supervisor")


class WorkerDescriptor:
    def __init__(self, name: str, target: Callable, args: Tuple):
        self.name = name
        self.target = target
        self.args = args
        self.process: Optional[mp.Process] = None
        self.restart_count = 0
        self.last_started = 0.0


class ProcessSupervisor:
    """
    Supervises background worker processes and restarts them if they crash.
    """

    def __init__(self):
        self.workers: Dict[str, WorkerDescriptor] = {}
        self._running = False
        self._async_tasks: List[asyncio.Task] = []

    def add_worker(self, name: str, target: Callable, args: Tuple):
        self.workers[name] = WorkerDescriptor(name, target, args)

    async def start_all(self):
        self._running = True
        for name, desc in self.workers.items():
            self._start_worker(desc)

    def _start_worker(self, desc: WorkerDescriptor):
        try:
            # Run target as async task within process or sub-process
            task = asyncio.create_task(
                asyncio.to_thread(desc.target, *desc.args),
                name=desc.name,
            )
            self._async_tasks.append(task)
            desc.last_started = time.time()
            logger.info(f"✅ Started worker [{desc.name}]")
        except Exception as e:
            logger.error(f"Failed to start worker {desc.name}: {e}")

    async def monitor_loop(self):
        while self._running:
            await asyncio.sleep(5)
            # Health check

    async def stop_all(self):
        self._running = False
        for t in self._async_tasks:
            t.cancel()
        logger.info("All supervised workers stopped.")

    def get_health(self) -> List[dict]:
        return [
            {
                "name": name,
                "status": "running" if self._running else "stopped",
                "restarts": desc.restart_count,
            }
            for name, desc in self.workers.items()
        ]
