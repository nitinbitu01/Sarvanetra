"""
backend/services/camera_adapters/base.py
Abstract base class for all camera stream adapters.
"""

from abc import ABC, abstractmethod
from typing import AsyncGenerator, Optional, Tuple, Dict, Any
import numpy as np


class BaseCameraAdapter(ABC):
    def __init__(self, camera_config: dict):
        self.config = camera_config
        self.camera_id = camera_config.get("id", "unknown")
        self.name = camera_config.get("name", str(self.camera_id))
        self.stream_url = camera_config.get("url") or camera_config.get("stream_url", "")
        self.ip_address = camera_config.get("ip_address", "")
        self.username = camera_config.get("username", "admin")
        self.password = camera_config.get("password", "")
        self.is_connected = False

    @abstractmethod
    async def connect(self) -> bool:
        """Connect to stream and verify frame availability."""
        pass

    @abstractmethod
    async def read_frame(self) -> Tuple[bool, Optional[np.ndarray]]:
        """Read a single BGR frame."""
        pass

    @abstractmethod
    async def stream(self) -> AsyncGenerator[Tuple[bool, Optional[np.ndarray]], None]:
        """Continuous frame generator."""
        pass

    @abstractmethod
    async def disconnect(self) -> None:
        """Gracefully close sockets/processes."""
        pass

    @property
    def resolved_url(self) -> str:
        """The stream URL this adapter actually connected with.

        This is the interoperability payload. A department hands over a
        vendor, an IP and credentials - not a stream URL - and each adapter
        knows its vendor's convention:
            Hikvision  rtsp://user:pass@ip:554/Streaming/Channels/101
            Dahua      rtsp://user:pass@ip:554/cam/realmonitor?...
            ONVIF      discovered from the device
        Exposing the negotiated URL lets a synchronous consumer reuse the
        result without adopting the async adapter interface.

        Vendor adapters delegate to an inner RTSPAdapter, so walk to it when
        present; otherwise fall back to whatever connect() settled on.
        """
        inner = getattr(self, "underlying_adapter", None)
        if inner is not None:
            return inner.resolved_url
        return getattr(self, "_resolved_url", "") or self.stream_url

    def get_health(self) -> Dict[str, Any]:
        return {
            "camera_id": self.camera_id,
            "is_connected": self.is_connected,
            "adapter_type": self.__class__.__name__,
            "resolved_url": self.resolved_url,
        }
