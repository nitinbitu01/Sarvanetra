"""
Camera Adapters Package
"""

from .base import BaseCameraAdapter
from .rtsp_adapter import RTSPAdapter
from .onvif_adapter import ONVIFAdapter
from .hikvision_adapter import HikvisionAdapter
from .dahua_adapter import DahuaAdapter
from .hls_adapter import HLSAdapter
from .factory import CameraAdapterFactory, CameraConnectionError

__all__ = [
    "BaseCameraAdapter",
    "RTSPAdapter",
    "ONVIFAdapter",
    "HikvisionAdapter",
    "DahuaAdapter",
    "HLSAdapter",
    "CameraAdapterFactory",
    "CameraConnectionError",
]
