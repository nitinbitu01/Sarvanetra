"""backend/services/vehicle_appearance.py — does the vehicle at camera B look
like the vehicle at camera A?

WHY THIS EXISTS, WRITTEN AFTER IT WAS NEEDED
  Journeys were first linked on plate likelihood and a speed check alone. Five
  were found and inspection showed at least two were wrong:

    GJ11BR5201  a white municipal waste truck at one camera, a grey van at
                the other
    GJ02ED8004  a black Tata hatchback at one, a silver Hyundai at the other

  Both passed the speed check - one was two hours apart at 0.4 km/h, which a
  parked car would produce, and the other implied 122 km/h, under the limit.
  Neither could have survived a glance at the colour.

  Plate likelihood alone cannot carry this. The recogniser is 93% accurate per
  character, so two different vehicles do occasionally read alike, and when
  they do nothing else in the pipeline objects. Appearance is the independent
  signal that makes the mistake detectable.

WHAT IS COMPARED, AND WHY IT IS DELIBERATELY SIMPLE
  Colour and coarse shape, not a learned embedding. A vehicle re-identification
  network would be better and needs training data, tuning and evidence that it
  transfers to these cameras - none of which exists yet. Colour histograms in
  HSV plus an aspect ratio need none of that and catch the failures actually
  observed: black against silver, a truck against a van.

  The comparison is deliberately LENIENT. The same car photographed from
  behind at one camera and in front at another, under different light, will
  not match closely; demanding a high similarity would reject true journeys.
  The bar is set to reject what is plainly a different vehicle, not to confirm
  that two crops are the same one.

  Sky, road and background dominate a vehicle crop's edges, so the centre
  region is weighted: it is mostly bodywork.
"""
from __future__ import annotations

from pathlib import Path

import cv2
import numpy as np


def _core(img: np.ndarray) -> np.ndarray:
    """Centre region, where the bodywork is rather than the road."""
    h, w = img.shape[:2]
    return img[int(h * .22):int(h * .82), int(w * .18):int(w * .82)]


def signature(path: str | Path) -> dict | None:
    """Colour and shape signature of one vehicle crop."""
    img = cv2.imread(str(path))
    if img is None or img.size == 0:
        return None
    core = _core(img)
    if core.size == 0:
        return None
    hsv = cv2.cvtColor(core, cv2.COLOR_BGR2HSV)
    # Hue and saturation only. Value carries the lighting, which differs
    # between cameras for the same vehicle and would dominate the distance.
    hist = cv2.calcHist([hsv], [0, 1], None, [24, 8], [0, 180, 0, 256])
    cv2.normalize(hist, hist, 0, 1, cv2.NORM_MINMAX)
    grey = cv2.cvtColor(core, cv2.COLOR_BGR2GRAY)
    return {
        "hist": hist.flatten(),
        "aspect": img.shape[1] / max(img.shape[0], 1),
        # Mean lightness separates a white vehicle from a black one even when
        # the hue histograms are both near-neutral and therefore similar.
        "light": float(grey.mean()) / 255.0,
    }


def similarity(a: dict, b: dict) -> dict:
    """0 to 1, with the components that produced it."""
    if not a or not b:
        return {"score": 0.5, "colour": None, "light": None,
                "note": "a crop was unreadable; no appearance evidence"}
    colour = float(cv2.compareHist(a["hist"].astype(np.float32),
                                   b["hist"].astype(np.float32),
                                   cv2.HISTCMP_CORREL))
    colour = max(0.0, min(1.0, (colour + 1) / 2))
    # Lightness difference is the single most reliable cue across cameras: a
    # black car and a silver one differ here no matter the angle or the hue.
    dl = abs(a["light"] - b["light"])
    light = max(0.0, 1.0 - dl / 0.45)
    da = abs(a["aspect"] - b["aspect"]) / max(a["aspect"], b["aspect"], 1e-6)
    shape = max(0.0, 1.0 - da)
    score = 0.40 * colour + 0.45 * light + 0.15 * shape
    return {"score": round(score, 3), "colour": round(colour, 3),
            "light": round(light, 3), "shape": round(shape, 3),
            "light_delta": round(dl, 3)}


def compare_crops(paths_a: list, paths_b: list) -> dict:
    """Best appearance match between two sightings' available crops.

    Best rather than average: each sighting may include a frame where the
    vehicle is half out of shot, and one bad crop on either side should not
    condemn a true link.
    """
    sa = [s for s in (signature(p) for p in paths_a[:3]) if s]
    sb = [s for s in (signature(p) for p in paths_b[:3]) if s]
    if not sa or not sb:
        return {"score": 0.5, "note": "no usable crops"}
    best = max((similarity(x, y) for x in sa for y in sb),
               key=lambda d: d["score"])
    return best
