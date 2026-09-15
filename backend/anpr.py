"""
backend/anpr.py — 3-engine OCR ensemble for ANPR.

This module contains ONLY the OCR logic: running EasyOCR, running PaddleOCR,
running the domain-specific CRNN with constrained beam search, combining
results, preprocessing crops. It is called exclusively by the OCR worker
thread (anpr_worker.py).

CRITICAL CONTRACT:
  - read_plate() and its helpers take pre-loaded reader instances as arguments.
  - They NEVER construct new easyocr.Reader() or PaddleOCR() or CRNN objects.
  - All model construction happens once, in anpr_worker.py at thread startup.
  - This prevents per-call model load latency (several seconds each time).

3-ENGINE ENSEMBLE STRATEGY:
  The original 2-engine ensemble (EasyOCR + PaddleOCR) is upgraded to include
  a domain-specific CRNN recogniser trained on Gujarat CCTV footage, decoded
  with grammar-constrained beam search.

  The CRNN is the TIEBREAKER: it resolves disagreements between the two
  general-purpose engines because it is specialised for exactly this glyph
  set, camera network, and plate format.

  Hierarchy of trust:
    1. All 3 agree → maximum confidence (no penalty).
    2. CRNN agrees with one general engine → medium-high confidence (0.95×).
    3. 2 general engines agree, CRNN disagrees → use generals (CRNN may be
       wrong on non-Gujarat plates), slight penalty (0.92×).
    4. All 3 disagree → CRNN wins (domain specialist), penalty (0.82×).
    5. Only 1 or 2 engines succeed → fallback logic by score.

  RECTIFICATION (Item 1C) is applied before recognition: deskew + perspective
  correction via plate_rectify.rectify(). Fail-safe: returns input unchanged
  if angle or quad detection fails, so it cannot make things worse.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import cv2
import numpy as np

from backend.plate_utils import normalize_plate_text

logger = logging.getLogger(__name__)

# ── CRNN + beam search constants (loaded once, used per-call) ─────────────────
# These must match the training configuration exactly.
_CRNN_IMG_H = 32
_CRNN_IMG_W = 128
_CKPT_PATH = Path("models/plate_recognizer/best.pt")


@dataclass
class _OcrResult:
    """Raw result from a single OCR engine call."""
    text: str
    conf: float
    engine: str


# ── Preprocessing ─────────────────────────────────────────────────────────────

def preprocess_crop(crop: np.ndarray) -> np.ndarray:
    """Prepare a vehicle crop for OCR.

    Steps:
      1. Resize so shorter side >= 300px (INTER_CUBIC for upscaling).
      2. CLAHE contrast enhancement on the L channel (LAB color space).

    This is the FULL preprocessing budget. No super-resolution.

    Args:
        crop: BGR image array from OpenCV.

    Returns:
        Preprocessed BGR image array.
    """
    if crop is None or crop.size == 0:
        return crop

    h, w = crop.shape[:2]
    min_side = min(h, w)
    if min_side < 300:
        scale = 300.0 / min_side
        new_w, new_h = int(w * scale), int(h * scale)
        crop = cv2.resize(crop, (new_w, new_h), interpolation=cv2.INTER_CUBIC)

    # CLAHE on L channel
    lab = cv2.cvtColor(crop, cv2.COLOR_BGR2LAB)
    l_ch, a_ch, b_ch = cv2.split(lab)
    clahe = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(8, 8))
    l_enhanced = clahe.apply(l_ch)
    enhanced = cv2.merge([l_enhanced, a_ch, b_ch])
    return cv2.cvtColor(enhanced, cv2.COLOR_LAB2BGR)


def _preprocess_for_crnn(crop: np.ndarray) -> np.ndarray:
    """Prepare a plate crop specifically for the CRNN recogniser.

    Steps:
      1. Rectify: deskew + perspective unwarp (fail-safe — returns input if
         detection fails). This undoes camera angle distortion.
      2. Convert to grayscale.
      3. Resize to 32×128 (CRNN's training resolution).
      4. Normalize to [-1, 1] float32.

    Args:
        crop: BGR or grayscale image of the plate region.

    Returns:
        Grayscale float32 array of shape (32, 128) normalised to [-1, 1].
    """
    # Rectification: deskew + perspective correction, fail-safe
    try:
        from backend.scripts.plate_rectify import rectify
        gray = cv2.cvtColor(crop, cv2.COLOR_BGR2GRAY) if crop.ndim == 3 else crop
        gray = rectify(gray, mode="full")
    except Exception:
        # If rectification import or execution fails, proceed with raw gray
        gray = cv2.cvtColor(crop, cv2.COLOR_BGR2GRAY) if crop.ndim == 3 else crop

    gray = cv2.resize(gray, (_CRNN_IMG_W, _CRNN_IMG_H),
                      interpolation=cv2.INTER_AREA)
    return gray.astype(np.float32) / 127.5 - 1.0


# ── Individual engine runners ─────────────────────────────────────────────────

def run_easyocr(crop: np.ndarray, reader: Any) -> "_OcrResult | None":
    """Run EasyOCR on a preprocessed crop.

    Args:
        crop: Preprocessed BGR image.
        reader: Pre-loaded easyocr.Reader instance (NOT constructed here).

    Returns:
        _OcrResult with raw text and confidence, or None on failure/empty.
    """
    try:
        results = reader.readtext(crop, detail=1, paragraph=False)
        if not results:
            logger.debug("EasyOCR: no text detected")
            return None

        # readtext returns list of (bbox, text, confidence)
        # Take the highest-confidence result
        best = max(results, key=lambda r: r[2])
        raw_text, conf = best[1], float(best[2])
        logger.debug("EasyOCR raw: text=%r conf=%.3f", raw_text, conf)
        return _OcrResult(text=raw_text, conf=conf, engine="easyocr")

    except Exception as exc:
        logger.warning("EasyOCR exception (crop ignored): %s", exc)
        return None


def run_paddleocr(crop: np.ndarray, reader: Any) -> "_OcrResult | None":
    """Run PaddleOCR on a preprocessed crop.

    Args:
        crop: Preprocessed BGR image.
        reader: Pre-loaded PaddleOCR instance (NOT constructed here).

    Returns:
        _OcrResult with raw text and confidence, or None on failure/empty.
    """
    if reader is None:
        return None
    try:
        result = reader.ocr(crop, cls=False)
        # PaddleOCR returns nested list: result[0] = list of [bbox, (text, conf)]
        if not result or not result[0]:
            logger.debug("PaddleOCR: no text detected")
            return None

        lines = result[0]
        # Take highest-confidence line
        best = max(lines, key=lambda r: r[1][1])
        raw_text = best[1][0]
        conf = float(best[1][1])
        logger.debug("PaddleOCR raw: text=%r conf=%.3f", raw_text, conf)
        return _OcrResult(text=raw_text, conf=conf, engine="paddleocr")

    except Exception as exc:
        logger.warning("PaddleOCR exception (crop ignored): %s", exc)
        return None


def run_crnn_beam(
    crop: np.ndarray,
    crnn_model: Any,
    crnn_device: str,
) -> "_OcrResult | None":
    """Run the domain-specific CRNN recogniser with grammar-constrained beam search.

    This is the engine that resolves confusable pairs (8/B, 0/O, 1/I, D/0)
    using plate format constraints DURING the CTC search — not after it.

    Args:
        crop: BGR or grayscale plate crop (rectification + resize done here).
        crnn_model: Pre-loaded CRNN model (nn.Module in eval mode).
        crnn_device: Device string ("cuda" or "cpu").

    Returns:
        _OcrResult with beam-decoded text and confidence, or None on failure.
    """
    if crnn_model is None:
        return None
    try:
        import torch
        from backend.scripts.plate_beam_decode import beam_decode
        from backend.scripts.train_plate_recognizer import BLANK, ITOS

        # Prepare input
        preprocessed = _preprocess_for_crnn(crop)
        x = torch.from_numpy(preprocessed).unsqueeze(0).unsqueeze(0)  # (1, 1, H, W)
        x = x.to(crnn_device)

        # Forward pass — single crop, no gradient
        with torch.no_grad():
            logits = crnn_model(x)  # (1, T, C)

        # Grammar-constrained beam search (beam_width=24, topk=6)
        candidates = beam_decode(
            logits[0],  # (T, C)
            itos=ITOS,
            blank=BLANK,
            beam_width=24,
            topk=6,
            constrain=True,
        )

        if not candidates:
            logger.debug("CRNN beam: no candidates produced")
            return None

        best_text, best_logp = candidates[0]
        if not best_text:
            return None

        # Convert log-probability to a 0-1 confidence score.
        # beam_decode returns log-probabilities which can be very negative.
        # We use a calibrated sigmoid mapping: logp > -2 → high confidence.
        import math
        # Clamp to prevent overflow
        clamped = max(best_logp, -50.0)
        conf = 1.0 / (1.0 + math.exp(-0.5 * (clamped + 5.0)))
        # Scale into [0.3, 1.0] range to avoid artificially low scores
        conf = 0.3 + 0.7 * conf

        logger.debug(
            "CRNN beam: text=%r logp=%.2f conf=%.3f (from %d candidates)",
            best_text, best_logp, conf, len(candidates),
        )
        return _OcrResult(text=best_text, conf=conf, engine="crnn_beam")

    except Exception as exc:
        logger.warning("CRNN beam exception (crop ignored): %s", exc)
        return None


# ── 3-Engine Ensemble ─────────────────────────────────────────────────────────

def read_plate(
    crop: np.ndarray,
    easyocr_reader: Any,
    paddleocr_reader: Any,
    crnn_model: Any = None,
    crnn_device: str = "cpu",
) -> dict[str, Any]:
    """Run the full 3-engine OCR ensemble on a vehicle crop.

    This function takes pre-loaded reader/model instances and does NOT
    construct any new reader objects. It runs all available engines,
    normalizes results, and combines them according to the hierarchy:

      1. All 3 agree         → max confidence, no penalty
      2. CRNN + 1 general    → 0.95× penalty (strong domain agreement)
      3. 2 generals agree    → 0.92× penalty (CRNN overruled by consensus)
      4. All 3 disagree      → CRNN wins at 0.82× (domain specialist)
      5. Fewer engines avail  → fallback by score with appropriate penalty

    Args:
        crop: Raw BGR vehicle crop (preprocessing applied inside).
        easyocr_reader: Pre-loaded easyocr.Reader.
        paddleocr_reader: Pre-loaded PaddleOCR (or None).
        crnn_model: Pre-loaded CRNN model (or None to fall back to 2-engine).
        crnn_device: Device string for CRNN inference.

    Returns:
        Dict with keys:
          "text"      : str | None — normalized plate text
          "confidence": float       — combined confidence
          "source"    : str         — ensemble decision description
          "agreement" : bool        — True only when ≥2 engines agree
    """
    processed = preprocess_crop(crop)

    # ── Run all 3 engines in parallel-ready fashion ──────────────────────────
    easy_result   = run_easyocr(processed, easyocr_reader)
    paddle_result = run_paddleocr(processed, paddleocr_reader)
    crnn_result   = run_crnn_beam(crop, crnn_model, crnn_device)

    # Normalize all outputs
    easy_norm   = normalize_plate_text(easy_result.text)   if easy_result   else None
    paddle_norm = normalize_plate_text(paddle_result.text) if paddle_result else None
    crnn_norm   = normalize_plate_text(crnn_result.text)   if crnn_result   else None

    # Compute crop sharpness metric for multi-frame weighting
    try:
        sharpness = float(cv2.Laplacian(processed, cv2.CV_64F).var())
    except Exception:
        sharpness = 1.0
    sharpness = max(sharpness, 0.1)

    # Collect successful reads into a list for systematic comparison
    reads = []
    if easy_norm:
        reads.append(("easyocr", easy_norm, easy_result.conf))
    if paddle_norm:
        reads.append(("paddleocr", paddle_norm, paddle_result.conf))
    if crnn_norm:
        reads.append(("crnn_beam", crnn_norm, crnn_result.conf))

    # ── No reads at all ──────────────────────────────────────────────────────
    if not reads:
        logger.debug("Ensemble: NO RESULT from any engine")
        return {
            "text": None,
            "confidence": 0.0,
            "source": "none",
            "agreement": False,
            "sharpness": sharpness,
        }

    # ── Only 1 engine succeeded ──────────────────────────────────────────────
    if len(reads) == 1:
        engine, text, conf = reads[0]
        # Single engine penalty (EasyOCR/PaddleOCR get 0.85×, CRNN gets 0.90×)
        penalty = 0.90 if engine == "crnn_beam" else 0.85
        penalized = conf * penalty
        logger.debug(
            "Ensemble: SINGLE ENGINE (%s) %r conf=%.3f (from %.3f)",
            engine, text, penalized, conf,
        )
        return {
            "text": text,
            "confidence": penalized,
            "source": f"single_{engine}",
            "agreement": False,
            "sharpness": sharpness,
        }

    # ── 2 or 3 engines succeeded — check agreement ──────────────────────────
    # Build agreement groups
    from collections import defaultdict
    text_votes: dict[str, list[tuple[str, float]]] = defaultdict(list)
    for engine, text, conf in reads:
        text_votes[text].append((engine, conf))

    # Find the reading with the most votes
    best_text = max(text_votes, key=lambda t: (len(text_votes[t]),
                                                max(c for _, c in text_votes[t])))
    voters = text_votes[best_text]
    voter_names = {v[0] for v in voters}
    max_conf = max(c for _, c in voters)

    # ── All engines agree ────────────────────────────────────────────────────
    if len(text_votes) == 1:
        logger.debug(
            "Ensemble: ALL AGREE (%d engines) text=%r conf=%.3f",
            len(reads), best_text, max_conf,
        )
        return {
            "text": best_text,
            "confidence": max_conf,  # No penalty — unanimous
            "source": "all_agree" if len(reads) == 3 else "both_agree",
            "agreement": True,
            "sharpness": sharpness,
        }

    # ── Majority (2 of 3 agree) ──────────────────────────────────────────────
    if len(voters) >= 2:
        # Who is the majority?
        crnn_in_majority = "crnn_beam" in voter_names

        if crnn_in_majority:
            # CRNN + one general engine agree → strong domain signal
            penalized = max_conf * 0.95
            source = "crnn_plus_general"
        else:
            # Two generals agree, CRNN disagrees → generals win but lower conf
            penalized = max_conf * 0.92
            source = "generals_agree"

        logger.debug(
            "Ensemble: MAJORITY %r voters=%s conf=%.3f (from %.3f)",
            best_text, voter_names, penalized, max_conf,
        )
        return {
            "text": best_text,
            "confidence": penalized,
            "source": source,
            "agreement": True,
            "sharpness": sharpness,
        }

    # ── All 3 disagree (each read something different) ───────────────────────
    # CRNN is the domain specialist — it wins the tiebreak
    if crnn_norm:
        penalized = crnn_result.conf * 0.82
        logger.debug(
            "Ensemble: ALL DISAGREE — CRNN tiebreak text=%r conf=%.3f",
            crnn_norm, penalized,
        )
        return {
            "text": crnn_norm,
            "confidence": penalized,
            "source": "crnn_tiebreak",
            "agreement": False,
        }

    # ── 2 engines, both disagree (no CRNN) → original logic ──────────────────
    winner = max(reads, key=lambda r: r[2])
    penalized = winner[2] * 0.85
    logger.debug(
        "Ensemble: DISAGREE (2-engine) %r conf=%.3f",
        winner[1], penalized,
    )
    return {
        "text": winner[1],
        "confidence": penalized,
        "source": "disagreement",
        "agreement": False,
    }
