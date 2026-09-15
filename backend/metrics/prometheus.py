import os
from prometheus_client import (
    Counter, Histogram, Gauge, start_http_server
)

FRAMES_PROCESSED = Counter(
    "sentinel_frames_total",
    "Total frames processed",
    ["camera_id", "zone"]
)

ALERTS_GENERATED = Counter(
    "sentinel_alerts_total",
    "Total alerts generated",
    ["severity", "crime_type", "zone"]
)

DETECTION_LATENCY = Histogram(
    "sentinel_inference_ms",
    "YOLOv8 batch inference latency ms",
    buckets=[5, 10, 25, 50, 100, 200, 500, 1000]
)

REID_MATCHES = Counter(
    "sentinel_reid_matches_total",
    "Cross-camera ReID matches",
    ["is_new", "is_wanted"]
)

ANPR_READS = Counter(
    "sentinel_anpr_reads_total",
    "Number plate reads",
    ["is_watchlisted"]
)

CAMERAS_ONLINE = Gauge(
    "sentinel_cameras_online",
    "Cameras currently streaming"
)

GPU_MEMORY = Gauge(
    "sentinel_gpu_memory_pct",
    "GPU memory utilization %",
    ["device"]
)

STREAM_RECONNECTS = Counter(
    "sentinel_stream_reconnects_total",
    "Stream reconnection attempts",
    ["camera_id"]
)

PIPELINE_LATENCY = Histogram(
    "sentinel_pipeline_latency_ms",
    "End-to-end pipeline latency ms per frame",
    buckets=[5, 10, 15, 20, 25, 30, 50]
)

TELEPORT_REJECTIONS = Counter(
    "sentinel_geo_teleport_rejections_total",
    "Total GeoGuard teleportation violations blocked"
)

GPU_THROTTLE_EVENTS = Counter(
    "sentinel_gpu_throttle_events_total",
    "GPU OOM throttle activations"
)

ACTIVE_GLOBAL_PERSONS = Gauge(
    "sentinel_active_global_persons",
    "Current GlobalPerson registry size in FAISS"
)

_server_started = False

def start_metrics_server():
    global _server_started
    if _server_started:
        return
    port = int(os.getenv("PROMETHEUS_PORT", "9091"))
    try:
        start_http_server(port)
        _server_started = True
    except Exception:
        pass
