"""
backend/services/camera_adapters/dahua_adapter.py
Dahua & CP PLUS CGI / RTSP Stream Adapter.
"""

import asyncio
import logging
import cv2
import numpy as np
from typing import AsyncGenerator, Optional, Tuple
from .base import BaseCameraAdapter
from .rtsp_adapter import RTSPAdapter

logger = logging.getLogger("sentinel.adapter.dahua")


class DahuaAdapter(BaseCameraAdapter):
    def __init__(self, camera_config: dict):
        super().__init__(camera_config)
        self.underlying_adapter = None

    async def connect(self) -> bool:
        user = self.username or "admin"
        pwd = self.password or "admin123"
        ip = self.ip_address or "127.0.0.1"
        channel = self.config.get("channel", 1)

        # Standard Dahua / CP Plus RTSP format
        rtsp_url = f"rtsp://{user}:{pwd}@{ip}:554/cam/realmonitor?channel={channel}&subtype=0"
        if self.stream_url:
            rtsp_url = self.stream_url

        config = {**self.config, "stream_url": rtsp_url}
        self.underlying_adapter = RTSPAdapter(config)
        self.is_connected = await self.underlying_adapter.connect()
        return self.is_connected

    async def read_frame(self) -> Tuple[bool, Optional[np.ndarray]]:
        if self.underlying_adapter:
            return await self.underlying_adapter.read_frame()
        return False, None

    async def stream(self) -> AsyncGenerator[Tuple[bool, Optional[np.ndarray]], None]:
        if self.underlying_adapter:
            async for success, frame in self.underlying_adapter.stream():
                yield success, frame

    async def disconnect(self) -> None:
        self.is_connected = False
        if self.underlying_adapter:
            await self.underlying_adapter.disconnect()
