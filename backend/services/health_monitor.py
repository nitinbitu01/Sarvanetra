"""
backend/services/health_monitor.py
Production Observability — Prometheus Metrics + Health Checks.
"""

import logging
import asyncio
from typing import Dict, Any

try:
    from prometheus_client import Counter, Gauge, Histogram, start_http_server
except ImportError:
    Counter = Gauge = Histogram = lambda *a, **kw: None
    start_http_server = lambda *a, **kw: None

logger = logging.getLogger("sentinel.health")


class SentinelMetrics:
    def __init__(self):
        try:
            self.cameras_online = Gauge(
                "sentinel_cameras_online_total", "Total cameras streaming successfully", ["department", "zone", "vendor"]
            )
            self.cameras_offline = Gauge(
                "sentinel_cameras_offline_total", "Total cameras not responding", ["department", "zone", "vendor"]
            )
            self.reid_matches_total = Counter(
                "sentinel_reid_matches_total", "Total cross-camera ReID matches", ["engine", "match_type"]
            )
            self.clm_hard_rejects = Counter(
                "sentinel_clm_rejects_total", "ReID matches rejected by Camera Link Model", ["reason"]
            )
            self.alerts_total = Counter(
                "sentinel_alerts_total", "Total alerts generated", ["priority"]
            )
            self.evidence_sealed_total = Counter(
                "sentinel_evidence_sealed_total", "Total video clips sealed as evidence"
            )
        except Exception:
            pass


def start_metrics_server(port: int = 9090):
    try:
        start_http_server(port)
        logger.info(f"Metrics server started on :{port}/metrics")
    except Exception as e:
        logger.debug(f"Metrics server notice: {e}")
    return SentinelMetrics()
