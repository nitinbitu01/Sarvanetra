"""
backend/routers/v1/detections_ws.py â€” Real-Time Bounding Box & Telemetry WebSocket.

Exposes:
  WebSocket /api/v1/ws/detections/{camera_id}
Broadcasts:
  {
    "type": "DETECTION_FRAME",
    "camera_id": "CAM_01",
    "timestamp": float,
    "frame_seq": int,
    "pipeline_latency_ms": float,
    "gpu_utilization_pct": int,
    "active_track_count": int,
    "geo_guard_rejections_last_60s": int,
    "detections": [
       {"track_id": int, "global_id": str, "class_name": str, "confidence": float,
        "bbox": [x1, y1, x2, y2], "danger_score": float, "threat_tags": list, "color": str}
    ]
  }
"""
from __future__ import annotations

import asyncio
import json
import logging
import time
from typing import Any

from fastapi import APIRouter, WebSocket, WebSocketDisconnect

logger = logging.getLogger(__name__)

router = APIRouter(tags=["Detections Live Stream"])

class DetectionConnectionManager:
    def __init__(self) -> None:
        # camera_id -> set of active WebSockets
        self._connections: dict[str, set[WebSocket]] = {}
        self._lock = asyncio.Lock()
        self.geo_rejections_60s = 0

    async def connect(self, camera_id: str, websocket: WebSocket) -> None:
        await websocket.accept()
        async with self._lock:
            if camera_id not in self._connections:
                self._connections[camera_id] = set()
            self._connections[camera_id].add(websocket)
        logger.info("Detections WebSocket client connected for camera %s", camera_id)

    async def disconnect(self, camera_id: str, websocket: WebSocket) -> None:
        async with self._lock:
            if camera_id in self._connections:
                self._connections[camera_id].discard(websocket)
                if not self._connections[camera_id]:
                    del self._connections[camera_id]
        logger.info("Detections WebSocket client disconnected from camera %s", camera_id)

    async def broadcast_detections(
        self,
        camera_id: str,
        detections: list[dict[str, Any]],
        frame_seq: int = 0,
        latency_ms: float = 15.6,
        gpu_util_pct: int = 68,
    ) -> None:
        async with self._lock:
            sockets = list(self._connections.get(camera_id, []))

        if not sockets:
            return

        payload = {
            "type": "DETECTION_FRAME",
            "camera_id": camera_id,
            "timestamp": time.time(),
            "frame_seq": frame_seq,
            "pipeline_latency_ms": round(latency_ms, 1),
            "gpu_utilization_pct": gpu_util_pct,
            "active_track_count": len(detections),
            "geo_guard_rejections_last_60s": self.geo_rejections_60s,
            "detections": detections,
        }
        msg = json.dumps(payload)

        stale = []
        for ws in sockets:
            try:
                await ws.send_text(msg)
            except Exception:
                stale.append(ws)

        if stale:
            async with self._lock:
                for ws in stale:
                    self._connections.get(camera_id, set()).discard(ws)


detection_ws_mgr = DetectionConnectionManager()


@router.websocket("/ws/detections/{camera_id}")
async def ws_detections_endpoint(websocket: WebSocket, camera_id: str):
    await detection_ws_mgr.connect(camera_id, websocket)
    try:
        while True:
            # Keepalive receiver
            data = await websocket.receive_text()
            if data == "ping":
                await websocket.send_text("pong")
    except WebSocketDisconnect:
        await detection_ws_mgr.disconnect(camera_id, websocket)
    except Exception as exc:
        logger.debug("Detections WS disconnect %s: %s", camera_id, exc)
        await detection_ws_mgr.disconnect(camera_id, websocket)
