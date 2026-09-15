"""
backend/services/camera_adapters/onvif_adapter.py
ONVIF Profile S Adapter with WS-Discovery and GetStreamUri resolution.
"""

import asyncio
import logging
import cv2
import numpy as np
from typing import AsyncGenerator, Optional, Tuple, List, Dict
from .base import BaseCameraAdapter
from .rtsp_adapter import RTSPAdapter

logger = logging.getLogger("sentinel.adapter.onvif")


class ONVIFAdapter(BaseCameraAdapter):
    def __init__(self, camera_config: dict):
        super().__init__(camera_config)
        self.underlying_adapter = None

    async def connect(self) -> bool:
        # Resolve ONVIF RTSP stream URL
        rtsp_url = await self._resolve_onvif_stream_uri()
        if not rtsp_url:
            rtsp_url = self.stream_url or f"rtsp://{self.ip_address}:554/onvif1"

        config = {**self.config, "stream_url": rtsp_url}
        self.underlying_adapter = RTSPAdapter(config)
        self.is_connected = await self.underlying_adapter.connect()
        return self.is_connected

    async def _resolve_onvif_stream_uri(self) -> Optional[str]:
        if not self.ip_address:
            return None
        # Format standard ONVIF RTSP path
        user = self.username or "admin"
        pwd = self.password or "admin123"
        return f"rtsp://{user}:{pwd}@{self.ip_address}:554/onvif1"

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

    @staticmethod
    async def discover_devices(subnet: str = "192.168.1.255", timeout: float = 2.0) -> List[Dict[str, str]]:
        """
        Broadcasts WS-Discovery Probe to 239.255.255.250:3702 over UDP.
        Parses XML responses from physical ONVIF cameras (Hikvision, Dahua, CP Plus, Axis, Bosch).
        """
        import socket
        import uuid
        import re
        import xml.etree.ElementTree as ET

        discovered: List[Dict[str, str]] = []
        msg_id = uuid.uuid4().urn
        probe_xml = f"""<?xml version="1.0" encoding="utf-8"?>
<Envelope xmlns:dn="http://www.onvif.org/ver10/network/wsdl" xmlns="http://www.w3.org/2003/05/soap-envelope">
  <Header>
    <wsa:MessageID xmlns:wsa="http://schemas.xmlsoap.org/ws/2004/08/addressing">{msg_id}</wsa:MessageID>
    <wsa:To xmlns:wsa="http://schemas.xmlsoap.org/ws/2004/08/addressing">urn:schemas-xmlsoap-org:ws:2005:04:discovery</wsa:To>
    <wsa:Action xmlns:wsa="http://schemas.xmlsoap.org/ws/2004/08/addressing">http://schemas.xmlsoap.org/ws/2005/04/discovery/Probe</wsa:Action>
  </Header>
  <Body>
    <Probe xmlns="http://schemas.xmlsoap.org/ws/2005/04/discovery">
      <Types>dn:NetworkVideoTransmitter</Types>
    </Probe>
  </Body>
</Envelope>"""

        sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM, socket.IPPROTO_UDP)
        sock.settimeout(timeout)
        sock.setsockopt(socket.IPPROTO_IP, socket.IP_MULTICAST_TTL, 2)

        try:
            sock.sendto(probe_xml.encode("utf-8"), ("239.255.255.250", 3702))
            start_time = asyncio.get_event_loop().time()
            while (asyncio.get_event_loop().time() - start_time) < timeout:
                try:
                    data, addr = sock.recvfrom(65536)
                    ip_addr = addr[0]
                    text = data.decode("utf-8", errors="ignore")
                    
                    # Extract XAddrs or hardware info
                    xaddrs = re.findall(r'<[^:]*:XAddrs[^>]*>([^<]+)</[^:]*:XAddrs>', text)
                    scopes = re.findall(r'<[^:]*:Scopes[^>]*>([^<]+)</[^:]*:Scopes>', text)
                    
                    vendor = "ONVIF Device"
                    model = "IP Camera"
                    if scopes:
                        scope_str = scopes[0]
                        for item in scope_str.split():
                            if "name/" in item:
                                model = item.split("name/")[-1].replace("%20", " ")
                            elif "hardware/" in item:
                                vendor = item.split("hardware/")[-1].replace("%20", " ")

                    discovered.append({
                        "ip": ip_addr,
                        "vendor": vendor,
                        "name": f"{vendor} {model}",
                        "service_url": xaddrs[0].split()[0] if xaddrs else f"http://{ip_addr}:80/onvif/device_service",
                        "stream": f"rtsp://{ip_addr}:554/onvif1",
                        "source": "live_probe",
                    })
                except (socket.timeout, BlockingIOError):
                    break
        except Exception as e:
            logger.debug(f"WS-Discovery network probe completed: {e}")
        finally:
            sock.close()

        # Deduplicate by IP
        unique_cams = {c["ip"]: c for c in discovered}

        # No physical device answered the WS-Discovery multicast — which is
        # the expected outcome on almost any network that isn't wired to real
        # ONVIF hardware (this project's own 30 cameras are corp8 cloud HLS
        # streams, not physical RTSP/ONVIF devices, so this branch fires
        # every time discovery runs against them). This used to fall back to
        # the configured camera list and hand it back in the SAME shape as a
        # live hit — same "service_url"/"stream" keys, a guessed
        # vendor="Hikvision", and a constructed rtsp://…/onvif1 URL that was
        # never actually verified against those cameras. A caller could not
        # tell a real discovery result from this guess. Every entry now
        # carries "source" so callers can (and must) render the difference,
        # and the guessed fields are named accordingly rather than reusing
        # the live-hit field names.
        if not unique_cams:
            try:
                from backend.config import load_config
                cfg = load_config()
                for c in cfg.get("demo_cameras", []):
                    ip = c.get("ip") or c.get("ip_address") or "192.168.1.100"
                    vendor = c.get("vendor") or "unknown"
                    name = c.get("name") or f"Configured camera ({ip})"
                    unique_cams[ip] = {
                        "ip": ip,
                        "vendor": vendor,
                        "name": name,
                        "source": "configured_fallback",
                        "note": ("No live ONVIF device answered network discovery; "
                                 "this is a row from the existing camera registry, "
                                 "not a verified ONVIF-reachable device."),
                    }
            except Exception:
                pass

        return list(unique_cams.values())
