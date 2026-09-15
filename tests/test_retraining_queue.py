"""
tests/test_retraining_queue.py — Active Learning Feedback Loop Verification.
"""
import os
import sqlite3
import cv2
import numpy as np
import pytest

from training.retraining_queue import route_to_retraining


def test_retraining_queue_true_positive(tmp_path):
    crop_path = str(tmp_path / "crop_101.jpg")
    cv2.imwrite(crop_path, np.zeros((50, 50, 3), dtype=np.uint8))
    db_file = str(tmp_path / "retrain.db")

    route_to_retraining(
        violation_id=101,
        crop_path=crop_path,
        label="true_positive",
        rider_count=3,
        majority_ratio=0.85,
        camera_id="CAM_01",
        weather_regime="NORMAL",
        db_path=db_file,
    )

    conn = sqlite3.connect(db_file)
    rows = conn.execute("SELECT * FROM retraining_items WHERE violation_id = 101").fetchall()
    conn.close()

    assert len(rows) == 1
    assert rows[0][3] == "true_positive"
    assert rows[0][4] == 3


def test_retraining_queue_false_positive(tmp_path):
    crop_path = str(tmp_path / "crop_102.jpg")
    cv2.imwrite(crop_path, np.zeros((50, 50, 3), dtype=np.uint8))
    db_file = str(tmp_path / "retrain.db")

    route_to_retraining(
        violation_id=102,
        crop_path=crop_path,
        label="false_positive",
        rider_count=2,
        majority_ratio=0.68,
        camera_id="CAM_02",
        weather_regime="RAIN",
        db_path=db_file,
    )

    conn = sqlite3.connect(db_file)
    rows = conn.execute("SELECT * FROM retraining_items WHERE violation_id = 102").fetchall()
    conn.close()

    assert len(rows) == 1
    assert rows[0][3] == "false_positive"
