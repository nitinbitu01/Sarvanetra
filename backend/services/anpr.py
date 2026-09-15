"""
backend/services/anpr.py — 3-Stage Indian License Plate Recognition (ANPR) Pipeline.

Stack:
  Stage 1 — Plate Detection:   YOLOv8-nano fine-tuned on IND plates (or plate region crop)
  Stage 2 — Character OCR:     PaddleOCR / EasyOCR ensemble
  Stage 3 — Format Validation: IND plate regex (with 0.5x confidence penalty on invalid formats)
"""
from __future__ import annotations

import logging
import os
import re
import cv2
import numpy as np
from dataclasses import dataclass
from typing import Optional

from backend.scripts.indian_plate_grammar import (
    decode_plate,
    resolve_state_code,
    STATE_CODES,
    SPECIAL_PREFIX,
)

logger = logging.getLogger(__name__)

# ── Indian plate format regex (standard + BH series) ──────────────────
_IND_PLATE_RE = re.compile(
    r'^(?:[A-Z]{2}\d{1,2}[A-Z]{0,3}\d{4}|\d{2}BH\d{4}[A-Z]{1,2})$'
)


@dataclass
class PlateResult:
    text: Optional[str]          # None if no plate found
    confidence: float            # combined confidence 0.0 – 1.0
    crop_path: Optional[str]     # saved plate-region crop
    format_valid: bool           # True if matches IND plate regex


# ── Module-level model singletons ─────────────────────────────────────
_detector: Optional[object] = None
_ocr: Optional[object] = None
_yolo_available = False
_paddle_available = False

try:
    from ultralytics import YOLO as _YOLO
    _yolo_available = True
except ImportError:
    pass

try:
    from paddleocr import PaddleOCR as _PaddleOCR
    _paddle_available = True
except ImportError:
    pass

try:
    import easyocr as _easyocr_mod
    _easyocr_available = True
except ImportError:
    _easyocr_available = False


def _get_easyocr():
    """Lazily build (and cache) an EasyOCR reader.

    PaddleOCR is not installed in this deployment, which left Stage 2 with
    no engine at all: `_ocr` stayed None, `raw_text` stayed empty, and every
    call returned PlateResult(None, 0.0, ...) - i.e. the pipeline reported
    "no plate" for every vehicle, silently. EasyOCR IS installed and is what
    the main ANPR worker already uses, so it is the correct fallback.

    Loading costs ~15s, so it happens on first use rather than at import,
    and is cached for the process lifetime.
    """
    global _ocr
    if _ocr is not None or not _easyocr_available:
        return _ocr
    try:
        logger.info("ANPR: loading EasyOCR reader (first use, ~15s)...")
        _ocr = _easyocr_mod.Reader(["en"], gpu=False, verbose=False)
        logger.info("ANPR: EasyOCR reader ready.")
    except Exception as exc:
        logger.warning("Could not initialize EasyOCR: %s", exc)
    return _ocr


def load_models(detector_weights: str = "models/yolov8n_ind_plate.pt", ocr_lang: str = "en") -> None:
    """Call once at startup (from main.py lifespan)."""
    global _detector, _ocr
    if _yolo_available and os.path.exists(detector_weights):
        try:
            _detector = _YOLO(detector_weights)
        except Exception as e:
            logger.warning("Could not load YOLO plate detector from %s: %s", detector_weights, e)
    if _paddle_available:
        try:
            _ocr = _PaddleOCR(use_angle_cls=True, lang=ocr_lang, show_log=False)
        except Exception as e:
            logger.warning("Could not initialize PaddleOCR: %s", e)


def _validate_format(text: str) -> tuple[bool, float]:
    """Returns (is_valid, confidence_multiplier). Invalid format receives 0.5x multiplier.
    Enforces that the plate prefix is a valid Indian State/UT or BH code.
    """
    cleaned = text.replace(" ", "").replace("-", "").upper()
    if not bool(_IND_PLATE_RE.match(cleaned)):
        return False, 0.5
    
    # Check if first 2 characters are a genuine Indian state/UT code
    prefix = cleaned[:2]
    if prefix in STATE_CODES or prefix in SPECIAL_PREFIX:
        return True, 1.0
    
    # If it's a BH series format (e.g. 24BH1234A)
    if len(cleaned) >= 4 and cleaned[2:4] == "BH":
        return True, 1.0

    return False, 0.5


def _save_crop(crop_img: np.ndarray, label: str, base_dir: str) -> str:
    os.makedirs(base_dir, exist_ok=True)
    stamp = int(__import__('time').time() * 1000)
    filename = f"plate_{label}_{stamp}.jpg"
    path = os.path.join(base_dir, filename)
    cv2.imwrite(path, crop_img)
    return path


def read_plate(
    vehicle_crop_path: str,
    min_conf: float = 0.75,
    plate_crop_dir: str = "output/evidence/plate_crops",
) -> PlateResult:
    """Full 3-stage ANPR pipeline."""
    if not os.path.exists(vehicle_crop_path):
        return PlateResult(None, 0.0, None, False)

    vehicle_img = cv2.imread(vehicle_crop_path)
    if vehicle_img is None or vehicle_img.size == 0:
        return PlateResult(None, 0.0, None, False)

    # ── Stage 1: Plate Bounding Box Detection ──────────────────────────
    detect_conf = 0.85
    plate_crop = vehicle_img

    if _detector is not None:
        try:
            results = _detector(vehicle_img, verbose=False)
            if results and len(results[0].boxes) > 0:
                boxes = results[0].boxes
                best_idx = int(boxes.conf.argmax())
                detect_conf = float(boxes.conf[best_idx])
                x1, y1, x2, y2 = map(int, boxes.xyxy[best_idx].tolist())
                h, w = vehicle_img.shape[:2]
                x1, y1 = max(0, x1), max(0, y1)
                x2, y2 = min(w, x2), min(h, y2)
                if (x2 - x1) > 10 and (y2 - y1) > 5:
                    plate_crop = vehicle_img[y1:y2, x1:x2]
        except Exception:
            pass

    # ── Stage 2: OCR ──────────────────────────────────────────────────
    # Two engines with different result shapes, so branch on the method the
    # object actually exposes rather than assuming PaddleOCR:
    #   PaddleOCR .ocr()      -> [[ [box, (text, conf)], ... ]]
    #   EasyOCR   .readtext() -> [ (box, text, conf), ... ]
    raw_text = ""
    ocr_conf = 0.80

    ocr = _ocr if _ocr is not None else _get_easyocr()

    if ocr is not None and plate_crop.size > 0:
        confs: list[float] = []
        try:
            if hasattr(ocr, "ocr"):
                ocr_results = ocr.ocr(plate_crop, cls=True)
                if ocr_results and ocr_results[0]:
                    for line in ocr_results[0]:
                        raw_text += line[1][0]
                        confs.append(float(line[1][1]))
            elif hasattr(ocr, "readtext"):
                for _box, txt, conf in ocr.readtext(plate_crop):
                    raw_text += txt
                    confs.append(float(conf))
        except Exception as exc:
            logger.debug("OCR failed on %s: %s", vehicle_crop_path, exc)
        if confs:
            ocr_conf = sum(confs) / len(confs)

    # If OCR produced no text, return empty
    if not raw_text:
        return PlateResult(None, 0.0, None, False)

    # ── Stage 3: Format validation & State Disambiguation ─────────────
    decoded = decode_plate(raw_text)
    if decoded.get("plate"):
        cleaned_text = decoded["plate"]
    else:
        cleaned_text = raw_text.replace(" ", "").replace("-", "").upper()
        if len(cleaned_text) >= 2:
            st, _ = resolve_state_code(cleaned_text[:2])
            cleaned_text = st + cleaned_text[2:]

    is_valid, format_mult = _validate_format(cleaned_text)
    final_conf = float(detect_conf * ocr_conf * format_mult)

    crop_path = _save_crop(plate_crop, cleaned_text, plate_crop_dir)

    return PlateResult(
        text=cleaned_text,
        confidence=round(final_conf, 3),
        crop_path=crop_path,
        format_valid=is_valid,
    )


def should_auto_populate(result: PlateResult, min_conf: float = 0.75) -> bool:
    return result.text is not None and result.confidence >= min_conf
