"""
backend/services/traffic_registry.py — builds WrongWayDetector instances from
on-disk calibration, and refuses to build one without it.

WHY THIS IS FAIL-CLOSED
  Wrong-way detection is only meaningful in world-plane METRES. The heading
  of a vehicle and its speed are both derived from a homography H that maps
  image pixels to ground coordinates. H cannot be guessed, defaulted, or
  approximated from the video alone - it requires at least 4 surveyed
  ground-control points whose real-world separation was physically measured
  at the camera site.

  So a camera with no calibration gets NO detector, not a detector with an
  identity matrix. With an identity H, "metres" are really pixels: a vehicle
  near the camera appears to move many times faster than an identical
  vehicle further away, speed gates fire arbitrarily, and the direction
  cosine is compared against a flow vector drawn in the wrong space. The
  engine would emit confident, official-looking, entirely fabricated
  violations - the worst possible failure mode for evidence that could be
  put in front of an officer or a court.

  Returning None is therefore the correct behaviour, not a limitation.

CALIBRATION FILE FORMAT  (config/traffic/<CAMERA_ID>.json)
  {
    "camera_id": "CAM_08",
    "calibrated_by": "name",
    "calibrated_on": "2026-08-24",
    "image_points":   [[x,y], [x,y], [x,y], [x,y]],
    "world_points_m": [[X,Y], [X,Y], [X,Y], [X,Y]],
    "validation": {"image": [x,y], "world_m": [X,Y]},
    "zones": [
      {
        "lane_id": "NB_LEFT",
        "polygon_world_m": [[X,Y], ...],
        "geometry_type": "straight",
        "flow_vector": [0.0, -1.0],
        "speed_limit_kmh": 60.0
      }
    ]
  }

  `validation` is optional but strongly recommended: it is a 5th point held
  out of the fit, used to measure real reprojection error. Without it a
  homography that fits its own 4 points perfectly can still be badly wrong.
"""
from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Any

import numpy as np

from .calibration import compute_homography, validate_homography
from .track_state import Zone
from .wrong_way_detector import WrongWayDetector

logger = logging.getLogger(__name__)

CALIB_DIR = Path("config/traffic")
MAX_REPROJ_ERROR_M = 0.30


def _build_zones(raw_zones: list[dict[str, Any]]) -> list[Zone]:
    zones: list[Zone] = []
    for z in raw_zones:
        flow = z.get("flow_vector")
        poly = z.get("flow_polyline_world_m")
        zones.append(
            Zone(
                lane_id=z["lane_id"],
                polygon_world_m=[tuple(p) for p in z["polygon_world_m"]],
                geometry_type=z.get("geometry_type", "straight"),
                flow_vector=tuple(flow) if flow else None,
                flow_polyline_world_m=[tuple(p) for p in poly] if poly else None,
                speed_limit_kmh=float(z.get("speed_limit_kmh", 60.0)),
            )
        )
    return zones


def load_detector(
    camera_id: str,
    config: dict[str, Any] | None = None,
    calib_dir: Path | None = None,
) -> WrongWayDetector | None:
    """Build a detector for one camera, or return None if it cannot be trusted.

    Returns None (never a partly-configured detector) when calibration is
    absent, malformed, or fails its held-out reprojection check.
    """
    d = calib_dir or CALIB_DIR
    path = d / f"{camera_id}.json"
    if not path.is_file():
        return None

    try:
        spec = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        logger.error("Traffic calibration for %s is unreadable: %s", camera_id, exc)
        return None

    try:
        img_pts = [tuple(p) for p in spec["image_points"]]
        world_pts = [tuple(p) for p in spec["world_points_m"]]
        raw_zones = spec["zones"]
    except KeyError as exc:
        logger.error("Traffic calibration for %s missing key %s", camera_id, exc)
        return None

    if not raw_zones:
        logger.error(
            "Traffic calibration for %s defines no lane zones - a wrong-way "
            "check needs at least one zone with a legal flow direction.",
            camera_id,
        )
        return None

    try:
        H = compute_homography(img_pts, world_pts)
    except ValueError as exc:
        logger.error("Homography for %s could not be computed: %s", camera_id, exc)
        return None

    if H is None or not np.all(np.isfinite(H)):
        logger.error(
            "Homography for %s is degenerate (collinear control points?) - "
            "choose points spread across the road plane, not along one line.",
            camera_id,
        )
        return None

    # Held-out validation. Optional in the file, but if present a failure is
    # fatal: a detector built on a bad homography is worse than none.
    val = spec.get("validation")
    if val:
        try:
            err = validate_homography(
                H, tuple(val["image"]), tuple(val["world_m"]), MAX_REPROJ_ERROR_M
            )
            logger.info(
                "Traffic calibration %s validated: reprojection error %.3f m",
                camera_id, err,
            )
        except (ValueError, KeyError) as exc:
            logger.error("Traffic calibration for %s REJECTED: %s", camera_id, exc)
            return None
    else:
        logger.warning(
            "Traffic calibration for %s has no held-out validation point. "
            "Accuracy is unverified - add 'validation' before trusting any "
            "violation from this camera.",
            camera_id,
        )

    try:
        zones = _build_zones(raw_zones)
    except (KeyError, TypeError) as exc:
        logger.error("Zone definition for %s is malformed: %s", camera_id, exc)
        return None

    logger.info(
        "Wrong-way detection ENABLED for %s (%d zone(s)).", camera_id, len(zones)
    )
    return WrongWayDetector(camera_id, H, zones, config)


def calibrated_cameras(calib_dir: Path | None = None) -> list[str]:
    """Camera ids that have a calibration file present (not yet validated)."""
    d = calib_dir or CALIB_DIR
    if not d.is_dir():
        return []
    return sorted(p.stem for p in d.glob("*.json"))
