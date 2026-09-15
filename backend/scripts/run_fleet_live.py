"""Run the 30-camera RTSP live fleet supervisor with verified settings."""
import os
import sys
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

# Set UTF-8 encoding
os.environ["PYTHONUTF8"] = "1"
os.environ["PYTHONIOENCODING"] = "utf-8"
os.environ["PYTHONUNBUFFERED"] = "1"

# RTSP streaming parameters
os.environ.setdefault("CORP8_RTSP_HOST", "103.250.160.189")
os.environ.setdefault("CORP8_RTSP_PORT", "8554")
os.environ.setdefault(
    "OPENCV_FFMPEG_CAPTURE_OPTIONS",
    "rtsp_transport;tcp|analyzeduration;1000000|probesize;1000000|fflags;nobuffer|flags;low_delay|max_delay;500000"
)

# Load corp8 credentials for any API queries
creds_path = ROOT / "config" / "corp8_credentials.json"
if creds_path.exists():
    try:
        creds = json.loads(creds_path.read_text(encoding="utf-8"))
        os.environ.setdefault("CORP8_EMAIL", creds.get("email", ""))
        os.environ.setdefault("CORP8_PASSWORD", creds.get("password", ""))
        os.environ.setdefault("CORP8_BASE", creds.get("base", "https://cctv.corp8.cloud"))
    except Exception as exc:
        print(f"[WARN] Failed to load credentials: {exc}")

# Stream preference: prefer live RTSP, allow failover to clips on network failure
os.environ["SENTINEL_FORCE_CLIPS"] = "0"
os.environ["SENTINEL_STRICT_LIVE"] = "0"

# Fleet sizing & FPS
os.environ.setdefault("SENTINEL_FLEET_WORKERS", "10")
os.environ.setdefault("SENTINEL_FLEET_READER_FPS", "5")
os.environ.setdefault("SENTINEL_FOCUS_READER_FPS", "10")
os.environ.setdefault("SENTINEL_FOCUS_CAMERAS", "CAM_09")
os.environ["SENTINEL_ANPR_CAMERAS"] = "ALL"

# Inference & publishing tuning for 30 cameras on RTX 4070
os.environ.setdefault("SENTINEL_YOLO_IMGSZ", "640")
os.environ.setdefault("SENTINEL_YOLO_CONF", "0.25")
os.environ.setdefault("SENTINEL_PLATE_DET_CONF", "0.40")
os.environ.setdefault("SENTINEL_SPEED_MAX_PX_RES_M", "2.5")
os.environ.pop("SENTINEL_WORKER_GPU_FRACTION", None)
os.environ.setdefault("SENTINEL_PUBLISH_ASYNC", "1")
os.environ.setdefault("SENTINEL_PUBLISH_MAX_WIDTH", "960")
os.environ.setdefault("SENTINEL_PUBLISH_QUALITY", "75")

if __name__ == "__main__":
    from backend.scripts.fleet_supervisor import main
    sys.exit(main())
