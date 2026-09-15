import asyncio
import json
import logging
import os
import time
from typing import Dict, List, Optional
import cv2
import numpy as np

logger = logging.getLogger("sentinel.redis_queue")


class SentinelRedisQueue:
    """
    High-throughput frame and alert queue.
    Supports Redis Streams if Redis is running, or high-performance in-memory fallback.
    """

    def __init__(self, redis_url: str = "redis://localhost:6379/0"):
        self.redis_url = redis_url
        self._redis = None
        self._is_connected = False
        self._mem_frames: Dict[str, bytes] = {}
        self._mem_alerts: List[dict] = []
        self._cam_statuses: Dict[str, dict] = {}

    async def connect(self):
        try:
            import redis.asyncio as aioredis
            self._redis = aioredis.from_url(
                self.redis_url, decode_responses=False, socket_timeout=2.0
            )
            await self._redis.ping()
            self._is_connected = True
            logger.info("Connected to Redis Streams queue")
        except Exception as e:
            logger.info(f"Using high-performance in-memory queue fallback ({e})")
            self._is_connected = False

    async def push_frame(self, camera_id: str, frame: np.ndarray, timestamp: float):
        try:
            _, buffer = cv2.imencode(".jpg", frame, [int(cv2.IMWRITE_JPEG_QUALITY), 80])
            jpeg_bytes = buffer.tobytes()

            if self._is_connected and self._redis:
                await self._redis.xadd(
                    f"sentinel:frames:{camera_id}",
                    {"frame": jpeg_bytes, "ts": str(timestamp)},
                    maxlen=30,
                    approximate=True,
                )
            else:
                self._mem_frames[camera_id] = jpeg_bytes

            self._cam_statuses[camera_id] = {
                "camera_id": camera_id,
                "is_online": True,
                "last_seen": timestamp,
                "total_frames": self._cam_statuses.get(camera_id, {}).get("total_frames", 0) + 1,
            }
        except Exception:
            pass

    async def get_latest_frame(self, camera_id: str) -> Optional[bytes]:
        if self._is_connected and self._redis:
            try:
                msgs = await self._redis.xrevrange(f"sentinel:frames:{camera_id}", count=1)
                if msgs:
                    return msgs[0][1].get(b"frame")
            except Exception:
                pass
        return self._mem_frames.get(camera_id)

    async def publish_alert(self, alert_dict: dict):
        self._mem_alerts.insert(0, alert_dict)
        if len(self._mem_alerts) > 200:
            self._mem_alerts = self._mem_alerts[:200]

        if self._is_connected and self._redis:
            try:
                raw = json.dumps(alert_dict, default=str)
                await self._redis.publish("sentinel:alerts", raw)
                await self._redis.lpush("sentinel:alert_feed", raw)
                await self._redis.ltrim("sentinel:alert_feed", 0, 199)
            except Exception:
                pass

    async def get_recent_alerts(self, limit: int = 20) -> List[dict]:
        if self._is_connected and self._redis:
            try:
                items = await self._redis.lrange("sentinel:alert_feed", 0, limit - 1)
                return [json.loads(x.decode() if isinstance(x, bytes) else x) for x in items]
            except Exception:
                pass
        return self._mem_alerts[:limit]

    async def get_all_camera_statuses(self) -> List[dict]:
        now = time.time()
        res = []
        for cam_id, stat in self._cam_statuses.items():
            is_active = (now - stat.get("last_seen", 0)) < 15
            res.append({**stat, "is_online": is_active})
        return res

    async def close(self):
        if self._redis:
            try:
                await self._redis.aclose()
            except Exception:
                pass
