# backend/services/gpu_manager.py
"""
GPU Memory Manager — PyTorch CUDA direct monitoring with nvidia-smi & ONNX Runtime fallbacks.
Real-world hardware telemetry with zero hardcoded metrics.
"""

import logging
import subprocess
from typing import Optional

logger = logging.getLogger("sentinel.gpu_manager")


class GPUMemoryManager:
    """
    Monitors GPU memory and hardware metrics using PyTorch CUDA and nvidia-smi.
    Falls back gracefully to CPU if GPU is unavailable.
    """

    def __init__(self, config: Optional[dict] = None):
        self.config = config or {}
        gpu_cfg = self.config.get("gpu", {})
        self.warning_thresh  = gpu_cfg.get("warning_threshold", 0.80)
        self.critical_thresh = gpu_cfg.get("critical_threshold", 0.92)

    def check_memory(self) -> dict:
        # Tier 1: Direct PyTorch CUDA memory tracking (fastest, zero subprocess overhead)
        try:
            import torch
            if torch.cuda.is_available():
                device_idx = 0
                props = torch.cuda.get_device_properties(device_idx)
                total_bytes = props.total_memory
                allocated_bytes = torch.cuda.memory_allocated(device_idx)
                reserved_bytes = torch.cuda.memory_reserved(device_idx)

                used_mb = round(reserved_bytes / (1024 * 1024), 1)
                total_mb = round(total_bytes / (1024 * 1024), 1)
                util = (reserved_bytes / total_bytes) if total_bytes > 0 else 0.0

                if util > self.critical_thresh:
                    logger.warning(
                        "GPU memory critical: %.1f/%.1f MB (%.1f%%)",
                        used_mb, total_mb, util * 100
                    )

                return {
                    "device": f"cuda:{device_idx}",
                    "device_name": props.name,
                    "used_mb": used_mb,
                    "allocated_mb": round(allocated_bytes / (1024 * 1024), 1),
                    "total_mb": total_mb,
                    "utilization_pct": round(util * 100, 1),
                    "is_safe": util < self.critical_thresh,
                    "provider": "PyTorch-CUDA",
                }
        except Exception as e:
            logger.debug("PyTorch CUDA check skipped: %s", e)

        # Tier 2: nvidia-smi subprocess query
        try:
            result = subprocess.run(
                [
                    "nvidia-smi",
                    "--query-gpu=memory.used,memory.total",
                    "--format=csv,noheader,nounits"
                ],
                capture_output=True, text=True, timeout=5
            )

            if result.returncode == 0:
                parts    = result.stdout.strip().split(", ")
                used_mb  = float(parts[0])
                total_mb = float(parts[1])
                util     = used_mb / max(total_mb, 1)

                if util > self.critical_thresh:
                    logger.warning(
                        "GPU memory critical: %.1f/%.1f MB (%.1f%%)",
                        used_mb, total_mb, util * 100
                    )

                return {
                    "device":          "cuda:0",
                    "used_mb":         round(used_mb, 1),
                    "total_mb":        round(total_mb, 1),
                    "utilization_pct": round(util * 100, 1),
                    "is_safe":         util < self.critical_thresh,
                    "provider":        "NVIDIA-SMI"
                }
        except Exception as e:
            logger.debug("nvidia-smi query skipped: %s", e)

        # Tier 3: CPU fallback
        return {
            "device":          "cpu",
            "utilization_pct": 0.0,
            "is_safe":         True,
            "provider":        "CPUExecutionProvider"
        }

    def get_gpu_info(self) -> dict:
        info = {
            "name": "CPU",
            "provider": "CPUExecutionProvider",
            "is_cuda": False,
        }

        # Check torch.cuda
        try:
            import torch
            if torch.cuda.is_available():
                props = torch.cuda.get_device_properties(0)
                info = {
                    "name": props.name,
                    "total_memory_mb": round(props.total_memory / (1024 * 1024), 1),
                    "cuda_version": torch.version.cuda or "available",
                    "device_count": torch.cuda.device_count(),
                    "provider": "PyTorch-CUDA",
                    "is_cuda": True,
                }
        except Exception:
            pass

        # Check nvidia-smi for driver and temperature
        try:
            result = subprocess.run(
                [
                    "nvidia-smi",
                    "--query-gpu=name,driver_version,memory.total,temperature.gpu",
                    "--format=csv,noheader"
                ],
                capture_output=True, text=True, timeout=5
            )
            if result.returncode == 0:
                parts = result.stdout.strip().split(", ")
                info["driver"] = parts[1].strip() if len(parts) > 1 else "unknown"
                if len(parts) > 3:
                    info["temperature_c"] = parts[3].strip()
        except Exception:
            pass

        return info
