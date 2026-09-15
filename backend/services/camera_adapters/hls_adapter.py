"""
backend/services/camera_adapters/hls_adapter.py
HLS (.m3u8) and HTTP Stream Adapter.
"""

import asyncio
import logging
import cv2
import numpy as np
from typing import AsyncGenerator, Optional, Tuple
from .base import BaseCameraAdapter

logger = logging.getLogger("sentinel.adapter.hls")


class HLSAdapter(BaseCameraAdapter):
    def __init__(self, camera_config: dict):
        super().__init__(camera_config)
        self.cap = None

    async def connect(self) -> bool:
        url = self.stream_url
        if not url:
            cid = self.config.get("channel_num") or self.camera_id
            url = f"https://live.corp8.cloud/camera/{cid}"

        try:
            self.cap = cv2.VideoCapture(url)
            self.is_connected = self.cap.isOpened()
            if self.is_connected:
                self._resolved_url = url
            return self.is_connected
        except Exception as e:
            logger.debug(f"HLSAdapter connect: {e}")
            self.is_connected = False
            return False

    async def read_frame(self) -> Tuple[bool, Optional[np.ndarray]]:
        if not self.is_connected or self.cap is None:
            return False, None
        try:
            ret, frame = self.cap.read()
            return ret, frame
        except Exception:
            return False, None

    async def stream(self) -> AsyncGenerator[Tuple[bool, Optional[np.ndarray]], None]:
        while self.is_connected and self.cap is not None:
            ret, frame = await self.read_frame()
            if not ret:
                await asyncio.sleep(0.1)
                continue
            yield True, frame
            await asyncio.sleep(0.02)

    async def disconnect(self) -> None:
        self.is_connected = False
        if self.cap is not None:
            try:
                self.cap.release()
            except Exception:
                pass
            self.cap = None
