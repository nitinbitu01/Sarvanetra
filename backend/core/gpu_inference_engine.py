# backend/core/gpu_inference_engine.py
"""
GPU Inference Engine for Sentinel.
Uses ONNX Runtime CUDA for RTX 4070 acceleration with verified execution probes.
"""

import os
import sys
import time
import logging
import threading
from pathlib import Path
from typing import List, Dict, Optional
from dataclasses import dataclass, field
from collections import defaultdict

import cv2
import numpy as np
import onnxruntime as ort

logger = logging.getLogger("sentinel.gpu_engine")


def _add_nvidia_to_path():
    """Add onnxruntime-gpu's bundled NVIDIA DLL dirs to PATH + the DLL loader.

    ⚠️ MUTUALLY EXCLUSIVE WITH TORCH-ON-GPU. Calling this prepends
    onnxruntime's bundled cuDNN ahead of torch's own, so torch subsequently
    loads the WRONG cuDNN and every CUDA convolution dies with
    CUDNN_STATUS_SUBLIBRARY_VERSION_MISMATCH.

    Proven with a clean A/B (same OSNet CUDA forward pass, separate
    processes):
        without importing this module -> OSNet CUDA OK
        with    importing this module -> CUDNN_STATUS_SUBLIBRARY_VERSION_MISMATCH

    This used to run unconditionally at import time, which meant simply
    importing this module — something backend/core/inference.py does, and
    main.py pulls in transitively — silently broke GPU ReID for the whole
    process. ReIDEmbedder caught that exception and degraded to STUB MODE,
    so the symptom was "ReID quietly returns random embeddings", with no
    error surfaced anywhere.

    WHY TORCH WINS BY DEFAULT: torch runs the entire verified core pipeline
    (YOLO detect+track AND OSNet ReID -> journeys -> alerts -> evidence).
    ONNX Runtime here serves InsightFace face-matching and the
    /api/inference demo endpoint, both of which degrade to CPU correctly
    rather than breaking. Trading a working ReID for a faster demo endpoint
    is the wrong trade.

    Set SENTINEL_ORT_GPU=1 to opt in to the old behaviour (ORT on GPU),
    accepting that ReID will fall back to CPU — see reid_embedder.py.
    """
    import site
    paths_to_check = []
    try:
        user_site = site.getusersitepackages()
        if isinstance(user_site, str):
            paths_to_check.append(user_site)
        elif isinstance(user_site, list):
            paths_to_check.extend(user_site)
    except Exception:
        pass
    try:
        paths_to_check.extend(site.getsitepackages())
    except Exception:
        pass

    added = []
    for base in paths_to_check:
        nvidia_dir = Path(base) / "nvidia"
        if not nvidia_dir.exists():
            continue
        for pkg_dir in nvidia_dir.iterdir():
            for sub in ["bin", "lib"]:
                bin_dir = pkg_dir / sub
                if bin_dir.exists():
                    bin_str = str(bin_dir)
                    current_path = os.environ.get("PATH", "")
                    if bin_str not in current_path:
                        os.environ["PATH"] = bin_str + os.pathsep + current_path
                        added.append(bin_str)
                    if hasattr(os, "add_dll_directory"):
                        try:
                            os.add_dll_directory(bin_str)
                        except Exception:
                            pass
    if added:
        logger.info(f"Added {len(added)} NVIDIA DLL directories to PATH & DLL Loader")


if os.getenv("SENTINEL_ORT_GPU", "0") == "1":
    # Opt-in only. See _add_nvidia_to_path's docstring: enabling this gives
    # ONNX Runtime CUDA but breaks torch CUDA (and therefore ReID) for the
    # entire process.
    _add_nvidia_to_path()
    logger.warning(
        "SENTINEL_ORT_GPU=1 — ONNX Runtime will use CUDA, but torch CUDA "
        "(YOLO detect+track and OSNet ReID) will fail with a cuDNN version "
        "mismatch and ReID will silently degrade to STUB MODE. Unset this "
        "unless you specifically want the ORT GPU path."
    )


@dataclass
class Detection:
    track_id:    int
    class_id:    int
    class_name:  str
    confidence:  float
    bbox_norm:   List[float]
    bbox_pixels: List[int]
    crop:        np.ndarray


@dataclass
class FrameResult:
    camera_id:    str
    frame_index:  int
    timestamp:    float
    detections:   List[Detection]
    inference_ms: float
    total_ms:     float


@dataclass
class TrackState:
    track_id:     int
    last_bbox:    List[float]
    last_sent_at: float
    send_count:   int = 0

    def moved(self, new_bbox: List[float], threshold: float = 0.025) -> bool:
        cx_old = (self.last_bbox[0] + self.last_bbox[2]) / 2
        cy_old = (self.last_bbox[1] + self.last_bbox[3]) / 2
        cx_new = (new_bbox[0] + new_bbox[2]) / 2
        cy_new = (new_bbox[1] + new_bbox[3]) / 2
        return abs(cx_new - cx_old) + abs(cy_new - cy_old) > threshold


class SentinelGPUEngine:
    CLASS_NAMES = {
        0:  "person",
        2:  "car",
        3:  "motorcycle",
        5:  "bus",
        7:  "truck",
        9:  "traffic_light",
        11: "stop_sign",
    }
    TARGET_CLASSES   = set(CLASS_NAMES.keys())
    CONF_THRESHOLD   = 0.45
    IOU_THRESHOLD    = 0.50
    MIN_RESEND_SEC   = 2.0
    INPUT_SIZE       = 640

    def __init__(self, yolo_onnx_path: str = "yolov8n.onnx", use_gpu: bool = True):
        self.model_path = yolo_onnx_path
        self.use_gpu = use_gpu
        self.gpu_active = False
        self.active_provider = "CPUExecutionProvider"

        self.session = self._load_session()
        self.input_name = self.session.get_inputs()[0].name
        
        self._track_states: Dict[str, Dict[int, TrackState]] = defaultdict(dict)
        self._track_lock    = threading.Lock()
        self._inference_times: List[float] = []
        self._frame_count   = 0
        self._start_time    = time.time()
        
        logger.info(
            f"SentinelGPUEngine initialized | "
            f"Active Provider: {self.active_provider} | "
            f"GPU Active: {self.gpu_active} | "
            f"Model: {yolo_onnx_path}"
        )

    def _load_session(self) -> ort.InferenceSession:
        opts = ort.SessionOptions()
        opts.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL
        opts.intra_op_num_threads = 4

        self.gpu_active = False
        if self.use_gpu:
            try:
                # Active probe with CUDAExecutionProvider ONLY
                probe_sess = ort.InferenceSession(
                    self.model_path,
                    sess_options=opts,
                    providers=["CUDAExecutionProvider"]
                )
                # Run real probe inference to confirm cuDNN convolution kernel succeeds
                dummy = np.random.randn(1, 3, self.INPUT_SIZE, self.INPUT_SIZE).astype(np.float32)
                probe_sess.run(None, {probe_sess.get_inputs()[0].name: dummy})
                self.gpu_active = True
                self.active_provider = "CUDAExecutionProvider"
                logger.info("✅ CUDAExecutionProvider verified with real GPU convolution probe!")
                return probe_sess
            except Exception as e:
                logger.warning(f"CUDAExecutionProvider probe failed ({e}) — falling back to CPU.")
                self.gpu_active = False

        # CPU Session Fallback
        cpu_sess = ort.InferenceSession(
            self.model_path,
            sess_options=opts,
            providers=["CPUExecutionProvider"]
        )
        self.active_provider = "CPUExecutionProvider"
        self.gpu_active = False
        return cpu_sess

    def process_frame(self,
                       camera_id: str,
                       frame: np.ndarray,
                       frame_index: int,
                       apply_dedup: bool = True) -> FrameResult:
        t0 = time.perf_counter()
        h, w = frame.shape[:2]

        tensor    = self._preprocess(frame)
        t_inf     = time.perf_counter()
        outputs   = self.session.run(None, {self.input_name: tensor})
        infer_ms  = (time.perf_counter() - t_inf) * 1000

        dets_raw = self._postprocess(outputs[0], w, h)
        if apply_dedup:
            dets_raw = self._deduplicate(camera_id, dets_raw)

        detections = []
        for d in dets_raw:
            x1, y1, x2, y2 = d["bbox_pixels"]
            crop = frame[max(0,y1):min(h,y2), max(0,x1):min(w,x2)]
            if crop.size < 400:
                continue
            detections.append(Detection(
                track_id=d["track_id"],
                class_id=d["class_id"],
                class_name=self.CLASS_NAMES.get(
                    d["class_id"], f"class_{d['class_id']}"
                ),
                confidence=d["confidence"],
                bbox_norm=d["bbox_norm"],
                bbox_pixels=d["bbox_pixels"],
                crop=crop
            ))

        total_ms = (time.perf_counter() - t0) * 1000
        self._inference_times.append(infer_ms)
        if len(self._inference_times) > 1000:
            self._inference_times.pop(0)
        self._frame_count += 1

        return FrameResult(
            camera_id=camera_id,
            frame_index=frame_index,
            timestamp=time.time(),
            detections=detections,
            inference_ms=round(infer_ms, 2),
            total_ms=round(total_ms, 2)
        )

    def _preprocess(self, frame: np.ndarray) -> np.ndarray:
        rgb     = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        resized = cv2.resize(
            rgb, (self.INPUT_SIZE, self.INPUT_SIZE),
            interpolation=cv2.INTER_LINEAR
        )
        tensor = resized.transpose(2, 0, 1).astype(np.float32) / 255.0
        return tensor[np.newaxis, ...]

    def _postprocess(self, raw: np.ndarray,
                      orig_w: int, orig_h: int) -> List[dict]:
        output = raw[0] if raw.ndim == 3 else raw
        if output.shape[0] == 84:
            output = output.T

        boxes    = output[:, :4]
        scores   = output[:, 4:]
        cls_ids  = np.argmax(scores, axis=1)
        confs    = np.max(scores, axis=1)

        mask = (
            (confs >= self.CONF_THRESHOLD) &
            np.isin(cls_ids, list(self.TARGET_CLASSES))
        )
        if not np.any(mask):
            return []

        fb = boxes[mask]
        fc = confs[mask]
        fi = cls_ids[mask]

        # YOLOv8's raw ONNX box output is in pixel space relative to the
        # INPUT_SIZE x INPUT_SIZE tensor fed to the model (values up to
        # ~640), NOT pre-normalized to [0,1]. Clipping straight to [0,1]
        # without dividing by INPUT_SIZE first collapsed every real box into
        # a degenerate ~1px corner, which then always failed the
        # `crop.size < 400` check in process_frame() and silently produced
        # ZERO detections — on a clear, high-confidence (0.95) real person,
        # confirmed by comparing this engine's raw output against
        # ultralytics' own YOLO() wrapper on the identical image. The
        # confidence scores (in `scores`/`confs` above) were never affected —
        # only the box geometry — which is exactly why this failed silently
        # instead of raising: the mask/confidence logic all looked correct.
        x1 = np.clip((fb[:, 0] - fb[:, 2] / 2) / self.INPUT_SIZE, 0, 1)
        y1 = np.clip((fb[:, 1] - fb[:, 3] / 2) / self.INPUT_SIZE, 0, 1)
        x2 = np.clip((fb[:, 0] + fb[:, 2] / 2) / self.INPUT_SIZE, 0, 1)
        y2 = np.clip((fb[:, 1] + fb[:, 3] / 2) / self.INPUT_SIZE, 0, 1)

        boxes_xyxy = np.stack([x1, y1, x2, y2], axis=1)
        keep = self._nms(boxes_xyxy, fc, self.IOU_THRESHOLD)

        return [{
            "track_id":    int(k),
            "class_id":    int(fi[k]),
            "confidence":  float(fc[k]),
            "bbox_norm":   [float(x1[k]), float(y1[k]),
                            float(x2[k]), float(y2[k])],
            "bbox_pixels": [int(x1[k]*orig_w), int(y1[k]*orig_h),
                            int(x2[k]*orig_w), int(y2[k]*orig_h)]
        } for k in keep]

    def _nms(self, boxes, scores, thr) -> List[int]:
        x1,y1,x2,y2 = boxes[:,0],boxes[:,1],boxes[:,2],boxes[:,3]
        areas = (x2-x1)*(y2-y1)
        order = scores.argsort()[::-1]
        keep  = []
        while order.size > 0:
            i = order[0]; keep.append(int(i))
            if order.size == 1: break
            xx1 = np.maximum(x1[i], x1[order[1:]])
            yy1 = np.maximum(y1[i], y1[order[1:]])
            xx2 = np.minimum(x2[i], x2[order[1:]])
            yy2 = np.minimum(y2[i], y2[order[1:]])
            iou = (np.maximum(0,xx2-xx1)*np.maximum(0,yy2-yy1)) / (
                areas[i] + areas[order[1:]] -
                np.maximum(0,xx2-xx1)*np.maximum(0,yy2-yy1) + 1e-8
            )
            order = order[1:][iou <= thr]
        return keep

    def _deduplicate(self, camera_id: str,
                      detections: List[dict]) -> List[dict]:
        now = time.time()
        with self._track_lock:
            states = self._track_states[camera_id]
            result = []
            for det in detections:
                tid  = det["track_id"]
                bbox = det["bbox_norm"]
                if tid not in states:
                    states[tid] = TrackState(
                        track_id=tid,
                        last_bbox=bbox,
                        last_sent_at=now
                    )
                    result.append(det)
                    continue
                state    = states[tid]
                time_ok  = (now - state.last_sent_at) >= self.MIN_RESEND_SEC
                moved    = state.moved(bbox)
                if time_ok and moved:
                    state.last_bbox    = bbox
                    state.last_sent_at = now
                    state.send_count  += 1
                    result.append(det)
        return result

    def get_performance_stats(self) -> dict:
        elapsed = time.time() - self._start_time
        avg_ms  = (
            sum(self._inference_times) /
            max(len(self._inference_times), 1)
        )
        return {
            "provider":          self.active_provider,
            "gpu_active":        self.gpu_active,
            "model":             str(self.model_path),
            "total_frames":      self._frame_count,
            "uptime_sec":        round(elapsed, 1),
            "avg_fps":           round(self._frame_count / max(elapsed, 1), 1) if self._frame_count > 0 else 0.0,
            "avg_infer_ms":      round(avg_ms, 2) if self._inference_times else 0.0,
            "hardware":          "NVIDIA GeForce RTX 4070 12GB" if self.gpu_active else "Host CPU"
        }

    def benchmark_live(self, num_runs: int = 50) -> dict:
        """Executes a real live benchmark dynamically on this hardware at call time."""
        dummy = np.random.randn(1, 3, self.INPUT_SIZE, self.INPUT_SIZE).astype(np.float32)
        # Warmup
        for _ in range(10):
            self.session.run(None, {self.input_name: dummy})

        t0 = time.perf_counter()
        for _ in range(num_runs):
            self.session.run(None, {self.input_name: dummy})
        elapsed = time.perf_counter() - t0

        fps = round(num_runs / max(elapsed, 1e-6), 1)
        latency_ms = round((elapsed / num_runs) * 1000, 2)
        target_fps = 2.0
        est_cameras = int(fps / target_fps)

        return {
            "hardware": "NVIDIA GeForce RTX 4070 12GB" if self.gpu_active else "Host CPU",
            "provider": self.active_provider,
            "gpu_active": self.gpu_active,
            "model": str(self.model_path),
            "measured_runs": num_runs,
            "fps": fps,
            "latency_ms": latency_ms,
            "camera_capacity_at_2fps": est_cameras,
            "is_dynamically_measured": True,
            "measured_timestamp": time.time()
        }
