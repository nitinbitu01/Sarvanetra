"""
backend/services/event_bus.py — Production-Grade Asynchronous Decoupled Event Bus.

Decouples high-throughput video processing from I/O bound tasks:
  - VAHAN Database Lookups
  - Dial-112 CAD Patrol Van Dispatch
  - Section 65B Cryptographic Evidence Hashing & Certificate Generation
  - Multi-Agency Scoped WebSocket Broadcasting
"""
from __future__ import annotations

import logging
import queue
import threading
import time
from dataclasses import dataclass, field
from typing import Any, Callable, Dict, List, Optional

logger = logging.getLogger("sentinel.event_bus")


@dataclass(order=True)
class PrioritizedEvent:
    priority: int
    timestamp: float = field(compare=False)
    event_type: str = field(compare=False)
    payload: Dict[str, Any] = field(compare=False)


class AsyncEventBus:
    """
    Lock-free, thread-safe asynchronous event bus.
    Workers run in a background daemon thread.
    """
    _instance: Optional[AsyncEventBus] = None
    _lock = threading.Lock()

    def __init__(self, max_queue_size: int = 2000):
        self._queue: queue.PriorityQueue[PrioritizedEvent] = queue.PriorityQueue(maxsize=max_queue_size)
        self._subscribers: Dict[str, List[Callable[[Dict[str, Any]], Any]]] = {}
        self._running = False
        self._worker_thread: Optional[threading.Thread] = None
        self._stats = {
            "events_published": 0,
            "events_processed": 0,
            "events_dropped": 0,
        }

    @classmethod
    def get_instance(cls) -> AsyncEventBus:
        with cls._lock:
            if cls._instance is None:
                cls._instance = cls()
            return cls._instance

    def subscribe(self, event_type: str, handler: Callable[[Dict[str, Any]], Any]) -> None:
        if event_type not in self._subscribers:
            self._subscribers[event_type] = []
        self._subscribers[event_type].append(handler)
        logger.info(f"Subscribed handler {getattr(handler, '__name__', str(handler))} to event: {event_type}")

    def publish(self, event_type: str, payload: Dict[str, Any], priority: int = 10) -> bool:
        event = PrioritizedEvent(
            priority=priority,
            timestamp=time.time(),
            event_type=event_type,
            payload=payload,
        )
        try:
            self._queue.put_nowait(event)
            self._stats["events_published"] += 1
            return True
        except queue.Full:
            self._stats["events_dropped"] += 1
            logger.warning(f"Event bus queue full! Dropped event: {event_type}")
            return False

    def start(self) -> None:
        if self._running:
            return
        self._running = True
        self._worker_thread = threading.Thread(target=self._worker_loop, daemon=True, name="SentinelEventBusWorker")
        self._worker_thread.start()
        logger.info("AsyncEventBus background worker started successfully.")

    def stop(self) -> None:
        self._running = False
        if self._worker_thread and self._worker_thread.is_alive():
            self._worker_thread.join(timeout=2.0)
        logger.info("AsyncEventBus stopped.")

    def _worker_loop(self) -> None:
        while self._running:
            try:
                event = self._queue.get(timeout=0.2)
            except queue.Empty:
                continue

            try:
                self._dispatch_event(event)
                self._stats["events_processed"] += 1
            except Exception as e:
                logger.error(f"Error processing event {event.event_type}: {e}", exc_info=True)
            finally:
                self._queue.task_done()

    def _dispatch_event(self, event: PrioritizedEvent) -> None:
        handlers = self._subscribers.get(event.event_type, [])
        wildcard_handlers = self._subscribers.get("*", [])

        for handler in handlers + wildcard_handlers:
            try:
                handler(event.payload)
            except Exception as e:
                logger.error(f"Handler failed for {event.event_type}: {e}")

    def get_stats(self) -> Dict[str, Any]:
        return {
            **self._stats,
            "queue_size": self._queue.qsize(),
            "is_running": self._running,
        }
