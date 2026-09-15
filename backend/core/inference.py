# backend/core/inference.py
"""
GPU Inference API routes.
Wired into backend/main.py as an APIRouter.
"""

from typing import Optional
import base64
import logging

import cv2
import numpy as np
from fastapi import APIRouter, HTTPException

from backend.core.gpu_inference_engine import SentinelGPUEngine

logger = logging.getLogger("sentinel.inference_router")
router = APIRouter(prefix="/api/inference", tags=["inference"])

# Single engine instance shared across all requests
_engine: Optional[SentinelGPUEngine] = None


def get_engine() -> SentinelGPUEngine:
    """Lazy-load GPU engine on first request."""
    global _engine
    if _engine is None:
        try:
            _engine = SentinelGPUEngine("yolov8n.onnx")
            logger.info(
                f"GPU engine loaded. "
                f"Active Provider: {_engine.active_provider} | GPU Active: {_engine.gpu_active}"
            )
        except Exception as e:
            logger.error(f"GPU engine load failed: {e}")
            raise HTTPException(
                status_code=503,
                detail=f"GPU engine unavailable: {e}"
            )
    return _engine


@router.get("/status")
def get_inference_status():
    """Live GPU inference stats."""
    engine = get_engine()
    return engine.get_performance_stats()


@router.post("/detect/frame")
async def detect_in_frame(payload: dict):
    """
    Run GPU detection on a base64-encoded frame.
    """
    engine = get_engine()

    if "frame_b64" not in payload:
        raise HTTPException(
            status_code=400,
            detail="Missing field: frame_b64"
        )

    try:
        img_data  = base64.b64decode(payload["frame_b64"])
        img_array = np.frombuffer(img_data, dtype=np.uint8)
        frame     = cv2.imdecode(img_array, cv2.IMREAD_COLOR)
    except Exception as e:
        raise HTTPException(
            status_code=400,
            detail=f"Frame decode failed: {e}"
        )

    if frame is None:
        raise HTTPException(
            status_code=400,
            detail="Invalid frame data — could not decode image"
        )

    result = engine.process_frame(
        camera_id=payload.get("camera_id", "soc_live"),
        frame=frame,
        frame_index=payload.get("frame_index", 0),
        apply_dedup=False
    )

    return {
        "camera_id":    result.camera_id,
        "timestamp":    result.timestamp,
        "inference_ms": result.inference_ms,
        "total_ms":     result.total_ms,
        "provider":     engine.active_provider,
        "gpu_active":   engine.gpu_active,
        "detections": [
            {
                "class_name":  d.class_name,
                "confidence":  round(d.confidence, 3),
                "bbox_norm":   d.bbox_norm,
                "bbox_pixels": d.bbox_pixels
            }
            for d in result.detections
        ]
    }


@router.get("/benchmark")
def get_benchmark_results():
    """
    Live measured benchmark results measured dynamically at call time.
    """
    engine = get_engine()
    return engine.benchmark_live(num_runs=50)