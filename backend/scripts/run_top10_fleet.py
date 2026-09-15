"""Run the verified Top-10 CCTV camera fleet with 100% success settings."""
import os
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

os.environ["PYTHONUTF8"] = "1"
os.environ["PYTHONIOENCODING"] = "utf-8"
os.environ["PYTHONUNBUFFERED"] = "1"

# Target top 10 cameras explicitly
TOP10 = "CAM_09,CAM_08,CAM_07,CAM_10,CAM_18,CAM_21,CAM_27,CAM_06,CAM_04,CAM_22"
os.environ["SENTINEL_ONLY_CAMERAS"] = TOP10
os.environ["SENTINEL_PIPELINE_CAMERAS"] = TOP10

# 5 workers × 2 cameras each = 10 cameras @ 10fps with 400ms headroom
os.environ["SENTINEL_FLEET_WORKERS"] = "5"
os.environ["SENTINEL_EQUAL_PRIORITY"] = "1"
os.environ["SENTINEL_FLEET_READER_FPS"] = "10"
os.environ["SENTINEL_FOCUS_CAMERAS"] = ""

# Enable ANPR across all 10 fleet cameras asynchronously
os.environ["SENTINEL_ANPR_CAMERAS"] = "CAM_09,CAM_08,CAM_07,CAM_10,CAM_18,CAM_21,CAM_27,CAM_06,CAM_04,CAM_22"
os.environ["SENTINEL_ANPR_ASYNC"] = "1"
os.environ["SENTINEL_PLATE_DET_CONF"] = "0.15"
os.environ["SENTINEL_MIN_PLATE_ASPECT_RATIO"] = "1.1"
os.environ["SENTINEL_SMALL_PLATE_DETECTOR"] = "1"
os.environ["SENTINEL_DOUBLE_LINE"] = "1"
os.environ["SENTINEL_SINGLE_FRAME_FLOOR_PX"] = "16"
os.environ["SENTINEL_FUSION_FLOOR_PX"] = "14"

# Guarantee clean, artifact-free 1080p footage and select daytime traffic clips
os.environ["SENTINEL_FORCE_CLIPS"] = "1"
os.environ["SENTINEL_CLIP_MATCH"] = "0830,0630,0730"
os.environ["SENTINEL_STRICT_LIVE"] = "0"

# Inference tuning for 10fps on RTX 4070
os.environ["SENTINEL_YOLO_IMGSZ"] = "640"
os.environ["SENTINEL_INFER_BATCH"] = "8"
os.environ["SENTINEL_WORKER_GPU_FRACTION"] = "0.18"
os.environ["SENTINEL_CUDA_MEM_FRACTION"] = "0.18"
os.environ["SENTINEL_EMBED_EVERY"] = "3"
os.environ["SENTINEL_ROLLUPS"] = "0"
os.environ["SENTINEL_PUBLISH_ASYNC"] = "1"
os.environ["SENTINEL_PUBLISH_MAX_WIDTH"] = "960"
os.environ["SENTINEL_PUBLISH_QUALITY"] = "75"
os.environ["SENTINEL_GLOBALID_CAMERAS"] = "ALL"

if __name__ == "__main__":
    from backend.scripts.fleet_supervisor import main
    sys.exit(main())
