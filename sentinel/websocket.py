# sentinel/websocket.py
from typing import Any
from backend.ws.dashboard_ws import broadcast

class WebSocketManagerProxy:
    async def broadcast(self, message: dict[str, Any]) -> None:
        await broadcast(message)

manager = WebSocketManagerProxy()

__all__ = ["manager", "broadcast"]
