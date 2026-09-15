"""
backend/services/ensemble_ocr.py

Two-engine OCR with agreement boosting.

Fast path  (crnn_conf >= 0.85): return CRNN result directly — no EasyOCR call.
Fallback   (crnn_conf <  0.85): run EasyOCR on the raw BGR crop.
Agreement  (both agree on text): confidence boosted by +0.10, capped at 1.0.
Disagree   (different texts):    return the engine with higher confidence.

Changes vs sketch in upgrade notes:
  • EasyOCR reader loaded lazily on first fallback call (saves ~3 GB RAM if
    CRNN is always confident enough).
  • EasyOCR allowlist restricted to plate charset (0-9 A-Z) — reduces noise.
  • Empty EasyOCR result handled gracefully (returns CRNN result).
  • Preprocessing option: can feed either raw BGR or pre-enhanced grey to EasyOCR.
"""
from __future__ import annotations

import logging
from typing import TYPE_CHECKING, Optional

import numpy as np

if TYPE_CHECKING:
    from backend.services.anpr_engine import ANPREngine

log = logging.getLogger(__name__)

EASYOCR_ALLOWLIST = "ABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789"


class EnsembleOCR:
    """
    Primary: fine-tuned CRNN (fast, accurate on training distribution).
    Fallback: EasyOCR (slower, handles out-of-distribution crops).
    """

    FAST_PATH_CONF = 0.85   # CRNN confidence above which EasyOCR is skipped

    def __init__(self, crnn_engine: "ANPREngine", gpu: bool | None = None):
        self.crnn         = crnn_engine
        self._use_gpu     = gpu
        self._easy_reader = None   # Lazy init

    # ── Lazy EasyOCR loader ───────────────────────────────────────────────────

    def _get_easy_reader(self):
        if self._easy_reader is None:
            try:
                import easyocr
                import torch
                use_gpu = (
                    self._use_gpu
                    if self._use_gpu is not None
                    else torch.cuda.is_available()
                )
                self._easy_reader = easyocr.Reader(
                    ["en"],
                    gpu=use_gpu,
                    verbose=False,
                )
                log.info("EasyOCR reader initialised (gpu=%s)", use_gpu)
            except Exception as exc:
                log.warning("EasyOCR unavailable: %s", exc)
                self._easy_reader = None
        return self._easy_reader

    # ── Public API ────────────────────────────────────────────────────────────

    def recognize(
        self,
        plate_crop:       np.ndarray,
        preprocessed_crop: np.ndarray,
    ) -> tuple[str, float, str]:
        """
        Returns (plate_text, confidence, source).
        source ∈ {"crnn", "easyocr", "agreed", "crnn_fallback"}
        """
        # ── Primary: CRNN ────────────────────────────────────────────────────
        crnn_text, crnn_conf = self.crnn.recognize(preprocessed_crop)
        crnn_text = (crnn_text or "").upper().replace(" ", "")

        # Fast path — CRNN very confident
        if crnn_conf >= self.FAST_PATH_CONF:
            return crnn_text, crnn_conf, "crnn"

        # ── Fallback: EasyOCR ─────────────────────────────────────────────────
        reader = self._get_easy_reader()
        if reader is None:
            return crnn_text, crnn_conf, "crnn_fallback"

        try:
            import cv2
            h_c = plate_crop.shape[0]
            ocr_input = plate_crop
            if h_c < 30 and plate_crop.shape[1] > 10:
                ocr_input = cv2.resize(
                    plate_crop,
                    (plate_crop.shape[1] * 2, h_c * 2),
                    interpolation=cv2.INTER_CUBIC,
                )

            easy_results = reader.readtext(
                ocr_input,
                detail=1,
                allowlist=EASYOCR_ALLOWLIST,
            )
        except Exception as exc:
            log.debug("EasyOCR inference failed: %s", exc)
            return crnn_text, crnn_conf, "crnn_fallback"

        if not easy_results:
            return crnn_text, crnn_conf, "crnn_fallback"

        # Sort detected word boxes top-to-bottom by y-center so 2-line plates read in proper order (Line 1 + Line 2)
        sorted_results = sorted(
            easy_results,
            key=lambda r: (r[0][0][1] + r[0][2][1]) / 2.0,
        )
        easy_text = "".join(r[1] for r in sorted_results).upper().replace(" ", "")
        easy_conf = float(max(r[2] for r in sorted_results))

        # ── Agreement check ───────────────────────────────────────────────────
        if crnn_text and easy_text and crnn_text == easy_text:
            boosted = min(1.0, (crnn_conf + easy_conf) / 2.0 + 0.10)
            return crnn_text, boosted, "agreed"

        # ── Disagreement — pick higher confidence ─────────────────────────────
        if crnn_conf >= easy_conf:
            return crnn_text, crnn_conf, "crnn"
        return easy_text, easy_conf, "easyocr"