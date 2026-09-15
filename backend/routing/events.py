"""
backend/routing/events.py — routing WebSocket payloads (Day 14).

Every datetime crosses the wire through to_iso8601(). A raw SQLite
'2024-01-15 14:23:01' is rejected by Safari's Date() constructor and renders
as "Invalid Date" on iOS/macOS while working fine in Chrome — a bug that only
appears on the reviewer's laptop.

Broadcasts are best-effort: a dashboard that is not connected must never
prevent a dispatch decision from being committed. The DB write is the source
of truth; these events are how a connected client avoids polling for it.
"""
from __future__ import annotations

import logging
from typing import Any

from backend.routing.utils import to_iso8601

logger = logging.getLogger(__name__)


async def _broadcast(payload: dict[str, Any]) -> None:
    try:
        from backend.ws.dashboard_ws import broadcast
        await broadcast(payload)
    except Exception as exc:
        logger.debug("Routing broadcast failed (%s): %s", payload.get("type"), exc)


async def broadcast_routed(alert_id: int, ra_id: int, officer_id: int,
                            officer_name: str | None, assigned_at: Any,
                            timeout_seconds: int) -> None:
    await _broadcast({
        "type": "alert.routed",
        "alert_id": alert_id,
        "routed_alert_id": ra_id,
        "officer_id": officer_id,
        "officer_name": officer_name,
        "assigned_at": to_iso8601(assigned_at),
        "timeout_seconds": timeout_seconds,
    })


async def broadcast_ack(alert_id: int, ra_id: int, ack_at: Any,
                         officer_id: int | None) -> None:
    await _broadcast({
        "type": "alert.ack",
        "alert_id": alert_id,
        "routed_alert_id": ra_id,
        "ack_at": to_iso8601(ack_at),
        "officer_id": officer_id,
    })


async def broadcast_escalated(alert_id: int, ra_id: int,
                               new_officer_id: int | None,
                               new_officer_name: str | None,
                               new_assigned_at: Any,
                               final_status: str) -> None:
    await _broadcast({
        "type": "alert.escalated",
        "alert_id": alert_id,
        "routed_alert_id": ra_id,
        "new_officer_id": new_officer_id,
        "new_officer_name": new_officer_name,
        "new_assigned_at": to_iso8601(new_assigned_at),
        "reason": "timeout",
        "final_status": final_status,
    })


async def broadcast_unrouted(alert_id: int, ra_id: int,
                              reason: str = "No officers available") -> None:
    await _broadcast({
        "type": "alert.unrouted",
        "alert_id": alert_id,
        "routed_alert_id": ra_id,
        "reason": reason,
    })


async def broadcast_officer_status(officer_id: int, status: str,
                                    current_alert_id: int | None) -> None:
    await _broadcast({
        "type": "officer.status",
        "officer_id": officer_id,
        "status": status,
        "current_alert_id": current_alert_id,
    })
