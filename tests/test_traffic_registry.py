"""
tests/test_traffic_registry.py — fail-closed calibration loading.

The point of these tests is that the registry must NEVER hand back a
detector it cannot vouch for. An uncalibrated camera silently producing
wrong-way "violations" in pixel space would generate official-looking
evidence that is entirely fabricated, so every rejection path is asserted
explicitly rather than assumed.
"""
import json

import numpy as np
import pytest

from backend.services.evidence import hash_file
from backend.services.traffic_registry import calibrated_cameras, load_detector


def _write(tmp_path, camera_id, spec):
    p = tmp_path / f"{camera_id}.json"
    p.write_text(json.dumps(spec), encoding="utf-8")
    return tmp_path


# A 1px = 0.1m scaling homography, matching the unit-test fixture scale.
_GOOD = {
    "camera_id": "CAM_T",
    "image_points": [[0, 0], [100, 0], [100, 100], [0, 100]],
    "world_points_m": [[0, 0], [10, 0], [10, 10], [0, 10]],
    "validation": {"image": [50, 50], "world_m": [5.0, 5.0]},
    "zones": [
        {
            "lane_id": "NB",
            "polygon_world_m": [[0, 0], [20, 0], [20, 200], [0, 200]],
            "geometry_type": "straight",
            "flow_vector": [0.0, -1.0],
            "speed_limit_kmh": 60.0,
        }
    ],
}


def test_missing_calibration_returns_none(tmp_path):
    assert load_detector("NO_SUCH_CAM", calib_dir=tmp_path) is None


def test_valid_calibration_builds_detector(tmp_path):
    _write(tmp_path, "CAM_T", _GOOD)
    det = load_detector("CAM_T", calib_dir=tmp_path)
    assert det is not None
    assert det.camera_id == "CAM_T"
    assert len(det.zones) == 1


def test_zero_zones_rejected(tmp_path):
    spec = dict(_GOOD, zones=[])
    _write(tmp_path, "CAM_T", spec)
    assert load_detector("CAM_T", calib_dir=tmp_path) is None


def test_bad_validation_point_rejected(tmp_path):
    # Claims pixel (50,50) is at (99,99)m; true projection is (5,5)m, so the
    # reprojection error is far over the 0.30m limit.
    spec = dict(_GOOD, validation={"image": [50, 50], "world_m": [99.0, 99.0]})
    _write(tmp_path, "CAM_T", spec)
    assert load_detector("CAM_T", calib_dir=tmp_path) is None


def test_malformed_json_rejected(tmp_path):
    (tmp_path / "CAM_T.json").write_text("{not json", encoding="utf-8")
    assert load_detector("CAM_T", calib_dir=tmp_path) is None


def test_missing_required_key_rejected(tmp_path):
    spec = {k: v for k, v in _GOOD.items() if k != "world_points_m"}
    _write(tmp_path, "CAM_T", spec)
    assert load_detector("CAM_T", calib_dir=tmp_path) is None


def test_calibrated_cameras_lists_files(tmp_path):
    _write(tmp_path, "CAM_A", _GOOD)
    _write(tmp_path, "CAM_B", _GOOD)
    assert calibrated_cameras(tmp_path) == ["CAM_A", "CAM_B"]


def test_end_to_end_violation_through_registry(tmp_path):
    """A registry-built detector must actually detect, and its evidence crop
    must be a real hashable file - this is what the pipeline wiring calls."""
    _write(tmp_path, "CAM_T", _GOOD)
    det = load_detector("CAM_T", calib_dir=tmp_path)
    assert det is not None

    frame = np.zeros((480, 640, 3), dtype=np.uint8)
    frame[50:100, 50:100] = 255
    events = []
    t = 0.0
    for i in range(35):
        t += 0.05
        px_y = 100.0 + (i * 10.0)   # travelling +y, against flow (0,-1)
        det_in = [{"track_id": 3, "bbox": [40, px_y - 10, 60, px_y + 10],
                   "vehicle_class": "car"}]
        events.extend(det.process_detections(det_in, frame, now=t))

    assert len(events) >= 1
    ev = events[0]
    assert ev.zone_id == "NB"
    assert ev.discrepancy_angle_deg >= 170.0
    # Evidence crop is a genuine file whose stored hash verifies.
    assert hash_file(ev.crop_path) == ev.evidence_clip_hash


def test_evidence_crop_is_vehicle_not_full_frame(tmp_path):
    """Regression: lock_evidence used to save the whole frame as the 'crop',
    so ANPR read a full 1080p scene instead of the offending vehicle."""
    import cv2

    _write(tmp_path, "CAM_T", _GOOD)
    det = load_detector("CAM_T", calib_dir=tmp_path)
    frame = np.zeros((480, 640, 3), dtype=np.uint8)

    events = []
    t = 0.0
    for i in range(35):
        t += 0.05
        px_y = 100.0 + (i * 10.0)
        det_in = [{"track_id": 4, "bbox": [40, px_y - 10, 60, px_y + 10],
                   "vehicle_class": "car"}]
        events.extend(det.process_detections(det_in, frame, now=t))

    assert events
    crop = cv2.imread(events[0].crop_path)
    assert crop is not None
    # bbox is 20px wide / 20px tall, so the crop must be far smaller than
    # the 640x480 source frame.
    assert crop.shape[0] < 480 and crop.shape[1] < 640
    # And the full-context frame is still preserved separately.
    full = cv2.imread(events[0].evidence_clip_path)
    assert full is not None and full.shape[:2] == (480, 640)
