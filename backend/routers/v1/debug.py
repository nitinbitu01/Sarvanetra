"""
backend/routers/v1/debug.py — GET /api/v1/debug/stats (Day 8, §4).

Placeholder for a real Prometheus /metrics endpoint (explicitly out of scope
for this pass — see Day 8 prompt §7). Exposes the same counters a Prometheus
exporter would, in a plain JSON shape, so `frames_processed_total`,
`alerts_fired_total`, `behavior_detector_latency_ms`, and `redis_errors_total`
are inspectable without standing up a metrics stack.

No auth required (matches the existing unauthenticated /health convention) —
this is operational telemetry, not sensitive data. Revisit before any real
deployment, same caveat as the existing /media static mounts in main.py.
"""
from __future__ import annotations

from fastapi import APIRouter

from backend.services.metrics import counters

router = APIRouter(prefix="/debug", tags=["debug"])


@router.get("/stats")
async def get_stats():
    return counters.snapshot()
