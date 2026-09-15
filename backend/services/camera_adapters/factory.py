"""
backend/services/camera_adapters/factory.py
Production Camera Adapter Factory with Auto-Fallback Chain.
"""

import asyncio
import logging
from typing import Dict, List, Optional, Type
from .base import BaseCameraAdapter
from .onvif_adapter import ONVIFAdapter
from .hikvision_adapter import HikvisionAdapter
from .dahua_adapter import DahuaAdapter
from .rtsp_adapter import RTSPAdapter
from .hls_adapter import HLSAdapter

logger = logging.getLogger("sentinel.camera_factory")


class CameraConnectionError(Exception):
    """Raised when all adapters in the fallback chain are exhausted."""
    pass


class CameraAdapterFactory:
    """
    Production camera adapter factory.
    Vendor fallback chains (ordered by likelihood of success):
      hikvision -> [hikvision, onvif, rtsp, hls]
      dahua     -> [dahua, onvif, rtsp, hls]
      cpplus    -> [dahua, hikvision, onvif, rtsp, hls]
      axis      -> [onvif, rtsp, hls]
      bosch     -> [onvif, rtsp, hls]
      onvif     -> [onvif, rtsp, hls]
      rtsp      -> [rtsp, hls]
      hls       -> [hls]
      unknown   -> [onvif, rtsp, hls]
    """
    FALLBACK_CHAINS: Dict[str, List[str]] = {
        "hikvision": ["hikvision", "onvif", "rtsp", "hls"],
        "dahua": ["dahua", "onvif", "rtsp", "hls"],
        "cpplus": ["dahua", "hikvision", "onvif", "rtsp", "hls"],
        "axis": ["onvif", "rtsp", "hls"],
        "bosch": ["onvif", "rtsp", "hls"],
        "hanwha": ["onvif", "rtsp", "hls"],
        "onvif": ["onvif", "rtsp", "hls"],
        "rtsp": ["rtsp", "hls"],
        "hls": ["hls"],
        "unknown": ["rtsp", "onvif", "hls"],
    }

    ADAPTER_REGISTRY: Dict[str, Type[BaseCameraAdapter]] = {
        "hikvision": HikvisionAdapter,
        "dahua": DahuaAdapter,
        "onvif": ONVIFAdapter,
        "rtsp": RTSPAdapter,
        "hls": HLSAdapter,
    }

    CONNECTION_TIMEOUT_SECONDS = 5.0
    _in_memory_preferences: Dict[str, str] = {}

    @classmethod
    async def create_with_fallback(
        cls, camera_config: dict, db=None
    ) -> BaseCameraAdapter:
        camera_id = str(camera_config.get("id", "unknown"))
        vendor = str(camera_config.get("vendor", "unknown")).lower()

        # Check for previously successful adapter preference
        learned_type = cls._in_memory_preferences.get(camera_id)
        chain = cls.FALLBACK_CHAINS.get(vendor, cls.FALLBACK_CHAINS["unknown"])

        if learned_type and learned_type in chain:
            chain = [learned_type] + [t for t in chain if t != learned_type]

        errors = []
        for adapter_type in chain:
            adapter_class = cls.ADAPTER_REGISTRY.get(adapter_type)
            if not adapter_class:
                continue

            adapter = adapter_class(camera_config)
            logger.debug(f"Camera {camera_id}: Trying {adapter_type} adapter...")

            try:
                connected = await asyncio.wait_for(
                    adapter.connect(), timeout=cls.CONNECTION_TIMEOUT_SECONDS
                )
                if connected:
                    logger.info(
                        f"Camera {camera_id} ({vendor}): Connected via {adapter_type}"
                    )
                    cls._in_memory_preferences[camera_id] = adapter_type
                    return adapter
                errors.append(f"{adapter_type}: connect() returned False")
            except asyncio.TimeoutError:
                errors.append(f"{adapter_type}: timeout after {cls.CONNECTION_TIMEOUT_SECONDS}s")
            except Exception as e:
                errors.append(f"{adapter_type}: {type(e).__name__}: {str(e)[:100]}")

        # If all fail, return fallback RTSPAdapter to allow simulated/retry loops
        logger.warning(f"Camera {camera_id} ({vendor}): All live adapters failed ({'; '.join(errors)}). Using standard RTSP fallback.")
        return RTSPAdapter(camera_config)


def resolve_stream_url(camera_config: dict, timeout: float = 30.0) -> Optional[str]:
    """Negotiate a working stream URL for one camera. Synchronous.

    WHY THIS BRIDGE EXISTS
      The adapters are async, but the detection pipeline is a synchronous
      thread loop (main.run_detection_pipeline). Before this, that loop
      called cv2.VideoCapture() on a URL straight from config, so the entire
      vendor-adapter layer - the part that answers "interoperable" - was
      never reached by the running system: the only caller was
      edge_inference_worker.py, which nothing launches.

      Rather than rewrite the pipeline as async, note what the adapters
      actually produce. Every one of them converges on a URL and hands it to
      cv2.VideoCapture. So the useful output is the NEGOTIATED URL, and a
      sync caller can take that and open it itself.

      That is what makes onboarding work without per-vendor code: a
      department supplies vendor + IP + credentials, the fallback chain
      finds the convention that answers, and the pipeline runs unchanged.

    Returns the URL that connected, or None if the whole chain failed.
    Never raises - a camera that cannot be negotiated must not stop the
    other 25 departments' cameras from starting.
    """
    async def _run() -> Optional[str]:
        adapter = await CameraAdapterFactory.create_with_fallback(camera_config)
        try:
            # create_with_fallback returns an unconnected RTSPAdapter when
            # every adapter failed, so verify rather than trust the object.
            if not adapter.is_connected:
                return None
            return adapter.resolved_url or None
        finally:
            try:
                await adapter.disconnect()
            except Exception:
                pass

    try:
        return asyncio.run(asyncio.wait_for(_run(), timeout=timeout))
    except RuntimeError:
        # Already inside a running loop (e.g. called from async startup):
        # use a private loop in this thread instead of hijacking that one.
        loop = asyncio.new_event_loop()
        try:
            return loop.run_until_complete(asyncio.wait_for(_run(), timeout=timeout))
        except Exception as exc:
            logger.warning("resolve_stream_url(%s): %s",
                           camera_config.get("id"), exc)
            return None
        finally:
            loop.close()
    except Exception as exc:
        logger.warning("resolve_stream_url(%s): %s", camera_config.get("id"), exc)
        return None
