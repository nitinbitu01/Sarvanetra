import os
import logging
from prometheus_client import Counter, Gauge, Histogram, start_http_server

logger = logging.getLogger("sentinel.metrics")

# ── Counters ──────────────────────────────────────────────────────────────────
FRAMES_PROCESSED = Counter(
    "sentinel_frames_processed_total",
    "Total frames processed by AI pipeline",
    ["camera_id", "district", "zone"],
)

DETECTIONS_TOTAL = Counter(
    "sentinel_detections_total",
    "Total objects detected",
    ["camera_id", "object_type"],
)

ALERTS_GENERATED = Counter(
    "sentinel_alerts_total",
    "Total alerts generated",
    ["severity", "alert_type", "zone"],
)

REID_MATCHES = Counter(
    "sentinel_reid_matches_total",
    "Cross-camera ReID matches",
    ["match_type"],
)

ANPR_READS = Counter(
    "sentinel_anpr_reads_total",
    "Number plate reads",
    ["confidence_band", "is_watchlisted"],
)

STREAM_RECONNECTS = Counter(
    "sentinel_stream_reconnects_total",
    "Stream reconnect attempts",
    ["camera_id"],
)

JOURNEY_ANOMALIES = Counter(
    "sentinel_journey_anomalies_total",
    "Journey anomalies detected",
    ["anomaly_type"],
)

OFFICERS_DISPATCHED = Counter(
    "sentinel_officers_dispatched_total",
    "Officers dispatched to alerts",
    ["zone"],
)

EVIDENCE_SAVED = Counter(
    "sentinel_evidence_saved_total",
    "Evidence packages saved",
    ["alert_type"],
)

# ── Gauges ────────────────────────────────────────────────────────────────────
CAMERAS_ONLINE = Gauge(
    "sentinel_cameras_online",
    "Cameras currently online",
)

CAMERAS_OFFLINE = Gauge(
    "sentinel_cameras_offline",
    "Cameras currently offline",
)

FRAME_QUEUE_DEPTH = Gauge(
    "sentinel_frame_queue_depth",
    "Frames waiting in stream per camera",
    ["camera_id"],
)

ACTIVE_ALERTS = Gauge(
    "sentinel_active_alerts",
    "Unresolved alerts by severity",
    ["severity"],
)

OFFICERS_AVAILABLE = Gauge(
    "sentinel_officers_available",
    "Available officers by zone",
    ["zone"],
)

GPU_MEMORY_PERCENT = Gauge(
    "sentinel_gpu_memory_percent",
    "GPU memory utilization",
    ["device"],
)

FAISS_INDEX_SIZE = Gauge(
    "sentinel_faiss_index_size",
    "Number of ReID embeddings in FAISS index",
)

WEBSOCKET_CONNECTIONS = Gauge(
    "sentinel_websocket_connections",
    "Active WebSocket dashboard connections",
)

# ── Histograms ────────────────────────────────────────────────────────────────
DETECTION_LATENCY = Histogram(
    "sentinel_detection_latency_ms",
    "YOLOv8 batch inference latency",
    buckets=[5, 10, 25, 50, 100, 200, 500, 1000],
)

ALERT_TO_DISPATCH_LATENCY = Histogram(
    "sentinel_alert_dispatch_latency_ms",
    "Time from detection to officer notification",
    buckets=[100, 250, 500, 1000, 2000, 5000],
)

REID_SEARCH_LATENCY = Histogram(
    "sentinel_reid_search_latency_ms",
    "FAISS ReID search latency",
    buckets=[1, 2, 5, 10, 25, 50],
)

ANPR_LATENCY = Histogram(
    "sentinel_anpr_latency_ms",
    "ANPR plate reading latency",
    buckets=[10, 25, 50, 100, 200, 500],
)

_server_started = False

def start_metrics_server():
    global _server_started
    if _server_started:
        return
    port = int(os.getenv("METRICS_PORT", "9090"))
    try:
        start_http_server(port)
        _server_started = True
        logger.info(f"Prometheus metrics HTTP server started on port {port}")
    except Exception as e:
        logger.warning(f"Prometheus metrics port {port} unavailable: {e}")
