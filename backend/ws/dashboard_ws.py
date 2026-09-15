"""
backend/ws/dashboard_ws.py — Auth-gated WebSocket handler for Day 5.

Clients send { "token": "<jwt>" } as the first message.
Server validates the token and closes with code 4001 if invalid.
After auth, the client receives detection events as before, plus
camera_added and new_alert events.
"""
from __future__ import annotations

import asyncio
import json
import logging
from typing import Any

from fastapi import WebSocket, WebSocketDisconnect
from jose import JWTError

from backend.auth.jwt_utils import decode_access_token

logger = logging.getLogger(__name__)

# Set of all authenticated, connected WebSocket clients
_clients: set[WebSocket] = set()


async def ws_dashboard_handler(websocket: WebSocket) -> None:
    """Handle an auth-gated WebSocket connection."""
    await websocket.accept()

    # ── Step 1: wait for auth message (timeout 10s) ───────────────────────
    try:
        raw = await asyncio.wait_for(websocket.receive_text(), timeout=10.0)
        msg = json.loads(raw)
        token = msg.get("token", "")
    except (asyncio.TimeoutError, json.JSONDecodeError, Exception):
        await websocket.close(code=4001)
        return

    try:
        payload = decode_access_token(token)
        user_id = payload.get("sub")
        if not user_id:
            raise JWTError("no sub")
    except JWTError:
        logger.warning("WebSocket auth failed — invalid token")
        await websocket.close(code=4001)
        return

    # ── Step 2: join the broadcast set ───────────────────────────────────
    _clients.add(websocket)
    logger.info("WebSocket client authenticated", extra={"user_id": user_id,
                                                          "active": len(_clients)})

    try:
        await websocket.send_json({"type": "authenticated", "user_id": user_id})

        # Keep alive — discard incoming messages (Day 9 adds commands)
        while True:
            await websocket.receive_text()

    except WebSocketDisconnect:
        pass
    except Exception as exc:
        logger.debug("WebSocket error: %s", exc)
    finally:
        _clients.discard(websocket)
        logger.info("WebSocket client disconnected", extra={"active": len(_clients)})


async def broadcast(message: dict[str, Any]) -> None:
    """Broadcast a message to all authenticated clients.

    Dead connections are silently removed from the set.
    Uses asyncio.wait_for with a 1.5s timeout per client (matches Day 4 hardening).
    """
    if not _clients:
        return
    dead: set[WebSocket] = set()
    for ws in list(_clients):
        try:
            await asyncio.wait_for(ws.send_json(message), timeout=1.5)
        except Exception:
            dead.add(ws)
    _clients.difference_update(dead)
