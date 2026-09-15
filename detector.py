"""
detector.py — YOLOv8 wrapper for Sentinel Gujarat.

Day 4 change: ONE model.track() call per frame with combined class filter
  [person, car, motorcycle, bus, truck]. Results are split AFTER inference
  using split_by_class() into separate person and vehicle lists. This avoids
  doubling inference cost by never calling model.track() twice on the same frame.

  Before Day 4: class_filter=[0] (persons only)
  After Day 4:  class_filter=[0,2,3,5,7] (configured in config.yaml)
  Results are split by class_id AFTER the single inference call.

The existing PersonDetector interface is UNCHANGED — callers that only need
person tracks call track_frame() and use split_by_class() to filter.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import cv2
import numpy as np

logger = logging.getLogger(__name__)

# COCO class IDs for vehicles — used by split_by_class() and config validation
VEHICLE_CLASS_IDS: frozenset[int] = frozenset([2, 3, 5, 7])
PERSON_CLASS_IDS: frozenset[int] = frozenset([0])

# COCO class IDs for the object classes the abandoned-object detector cares
# about (backend/services/abandoned_object_detector.py). Added when that
# detector was wired into the live pipeline — previously class_filter never
# included these, so no object tracks existed for it to consume at all.
OBJECT_CLASS_IDS: frozenset[int] = frozenset([24, 26, 28])
OBJECT_CLASS_NAMES: dict[int, str] = {24: "backpack", 26: "handbag", 28: "suitcase"}


@dataclass
class RawTrackResult:
    """One raw detection+track row as returned by BoT-SORT for a single frame.

    These are NOT confirmed tracks. tracker.py decides what is confirmed.
    """
    track_id: int
    bbox: list[float]       # [x1, y1, x2, y2] in pixel coordinates
    confidence: float
    class_id: int


def split_by_class(
    raw_tracks: list[RawTrackResult],
    person_classes: list[int] | None = None,
    vehicle_classes: list[int] | None = None,
    object_classes: list[int] | None = None,
) -> tuple[list[RawTrackResult], list[RawTrackResult], list[RawTrackResult]]:
    """Split a combined detection result into person, vehicle, and object lists.

    Called immediately after track_frame() so each list can be fed to its
    own TrackStateManager instance. This is the sole place class splitting
    happens — never duplicate the model.track() call instead.

    Args:
        raw_tracks: Output of PersonDetector.track_frame() (all classes).
        person_classes: Class IDs to route to persons list. Defaults to {0}.
        vehicle_classes: Class IDs to route to vehicles list. Defaults to
                         VEHICLE_CLASS_IDS {2, 3, 5, 7}.
        object_classes: Class IDs to route to objects list (bags/suitcases
                         for the abandoned-object detector). Defaults to
                         OBJECT_CLASS_IDS {24, 26, 28}.

    Returns:
        (person_tracks, vehicle_tracks, object_tracks) — three separate lists.
    """
    p_cls = frozenset(person_classes) if person_classes else PERSON_CLASS_IDS
    v_cls = frozenset(vehicle_classes) if vehicle_classes else VEHICLE_CLASS_IDS
    o_cls = frozenset(object_classes) if object_classes else OBJECT_CLASS_IDS

    persons: list[RawTrackResult] = []
    vehicles: list[RawTrackResult] = []
    objects: list[RawTrackResult] = []

    for t in raw_tracks:
        if t.class_id in p_cls:
            persons.append(t)
        elif t.class_id in v_cls:
            vehicles.append(t)
        elif t.class_id in o_cls:
            objects.append(t)
        # else: unknown class — silently ignored

    return persons, vehicles, objects


class PersonDetector:
    """Wraps YOLOv8 + BoT-SORT tracking.

    Day 4: Runs a single combined detection pass for both persons and vehicles.
    The class_filter in config.yaml should be [0, 2, 3, 5, 7] as of Day 4.
    Use split_by_class() on the result to separate person and vehicle tracks.

    GPU / TensorRT swap: change ``device`` in config.yaml to "cuda:0"
    or "tensorrt". No code changes required — ultralytics handles it.
    """

    def __init__(self, cfg: dict[str, Any]) -> None:
        """Load model and validate configuration.

        Args:
            cfg: Full parsed config dict (entire config.yaml contents).
        """
        self._model_path = cfg["model"]["path"]
        self._device = cfg["model"]["device"]
        # half (FP16) only makes sense on CUDA - ultralytics raises on CPU.
        self._half = bool(cfg["model"].get("half", False)) and str(self._device).startswith("cuda")
        self._conf_threshold = cfg["detection"]["confidence_threshold"]
        self._class_filter = cfg["detection"]["class_filter"]
        self._tracker_config = cfg["tracking"]["tracker_config"]
        self._input_width = cfg["processing"]["input_width"]

        self._model = self._load_model()
        self._class_filter = self._resolve_class_filter(self._class_filter)

    def _resolve_class_filter(self, class_filter: Any) -> list[int] | None:
        """Map config's class NAMES to the integer indices Ultralytics wants.

        `detection.class_filter` in config.yaml is written as readable names
        (["person", "car", ...]) but ultralytics' `.track(classes=...)` takes
        COCO integer indices and feeds them straight into
        `torch.tensor(classes)`. Passing strings raised
        `ValueError: too many dimensions 'str'` on the FIRST frame of every
        source, file or live - the detection pipeline could not process a
        single frame.

        Indices are resolved from the loaded model's own `names` mapping
        rather than a hardcoded table, so this stays correct if the model is
        ever swapped for one with a different class set.
        """
        if not class_filter:
            return None  # ultralytics treats None as "all classes"

        # Already integers (someone configured indices directly) - pass through.
        if all(isinstance(c, int) for c in class_filter):
            return list(class_filter)

        names: dict[int, str] = getattr(self._model, "names", {}) or {}
        name_to_idx = {str(v).lower(): int(k) for k, v in names.items()}

        resolved: list[int] = []
        unknown: list[str] = []
        for entry in class_filter:
            if isinstance(entry, int):
                resolved.append(entry)
                continue
            idx = name_to_idx.get(str(entry).strip().lower())
            if idx is None:
                unknown.append(str(entry))
            else:
                resolved.append(idx)

        if unknown:
            # Loud, not silent: a typo'd class name would otherwise just
            # narrow what the system detects, with no indication why.
            logger.error(
                "detection.class_filter contains %d name(s) this model does "
                "not know: %s. They will NOT be detected. Known classes: %s",
                len(unknown), ", ".join(unknown),
                ", ".join(sorted(name_to_idx)[:12]) + " ...",
            )

        logger.info("class_filter resolved to indices %s", resolved)
        return resolved or None

    def _load_model(self) -> Any:
        """Load YOLOv8 model from disk, raising a clear error on failure."""
        from ultralytics import YOLO  # deferred so import errors surface clearly

        model_path = Path(self._model_path)
        logger.info("Loading model from '%s' on device '%s'", model_path, self._device)

        try:
            model = YOLO(str(model_path))
        except Exception as exc:
            raise RuntimeError(
                f"Failed to load YOLOv8 model from '{model_path}'. "
                f"Ensure the path is correct or the model will be auto-downloaded. "
                f"Original error: {exc}"
            ) from exc

        logger.info("Model loaded successfully: %s", model_path.name)
        return model

    def track_frame(
        self, frame: np.ndarray, frame_number: int
    ) -> list[RawTrackResult]:
        """Run ONE combined detection + tracking pass on a single frame.

        Returns ALL detected classes (persons + vehicles) in one call.
        Call split_by_class() on the result to separate them — do NOT
        call this function twice for the same frame.

        Args:
            frame: BGR image array from OpenCV.
            frame_number: Current frame index (used for debug logging only).

        Returns:
            List of raw track results for ALL configured classes.
            May be empty. Confidence filtering is applied here.
        """
        # 'half=' triggers a LOGGER.warning deprecation notice on every single
        # call (it goes through get_cfg's arg-merge, not a deduped
        # warnings.warn) - at 5fps across 30 cameras that's ~150 log lines/sec
        # of pure noise. 'quantize=16' is the current equivalent and takes
        # the non-deprecated branch, so it's only passed when actually
        # wanted rather than passing quantize=None on every FP32 call too.
        extra_kwargs: dict[str, Any] = {"quantize": 16} if self._half else {}

        results = self._model.track(
            source=frame,
            tracker=self._tracker_config,
            classes=self._class_filter,
            conf=self._conf_threshold,
            imgsz=self._input_width,
            device=self._device,
            persist=True,       # Required: tells BoT-SORT to keep state across calls
            verbose=False,      # Suppress ultralytics per-frame stdout
            **extra_kwargs,
        )

        tracks: list[RawTrackResult] = []
        if not results or results[0].boxes is None:
            logger.debug("Frame %d: no detections.", frame_number)
            return tracks

        boxes = results[0].boxes
        if boxes.id is None:
            # Tracker returned detections but assigned no IDs yet (can happen on
            # the very first frame before BoT-SORT stabilises)
            logger.debug("Frame %d: detections present but no track IDs assigned yet.", frame_number)
            return tracks

        ids  = boxes.id.cpu().numpy().astype(int)
        xyxy = boxes.xyxy.cpu().numpy()
        confs = boxes.conf.cpu().numpy()
        clses = boxes.cls.cpu().numpy().astype(int)

        for tid, box, conf, cls in zip(ids, xyxy, confs, clses):
            tracks.append(
                RawTrackResult(
                    track_id=int(tid),
                    bbox=box.tolist(),
                    confidence=float(conf),
                    class_id=int(cls),
                )
            )
            logger.debug(
                "Frame %d: raw track id=%d class=%d conf=%.2f bbox=%s",
                frame_number, tid, cls, conf, box.tolist(),
            )

        return tracks
