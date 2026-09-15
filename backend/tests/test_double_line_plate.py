import unittest
import numpy as np
import cv2
from backend.services.anpr_track_aggregator_v2 import split_stacked_plate, TrackletPlateBuffer


class TestDoubleLinePlate(unittest.TestCase):
    def setUp(self):
        # Create a synthetic 2:1 double-line plate (e.g. 60 height, 110 width)
        self.stacked_crop = np.full((60, 110, 3), 230, dtype=np.uint8)
        # Top line: GJ 01
        cv2.putText(self.stacked_crop, "GJ 01", (15, 22), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (20, 20, 20), 2)
        # Bottom line: AB 1234
        cv2.putText(self.stacked_crop, "AB 1234", (10, 50), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (20, 20, 20), 2)

        # Create a synthetic 4.5:1 single-line plate (e.g. 35 height, 160 width)
        self.single_crop = np.full((35, 160, 3), 230, dtype=np.uint8)
        cv2.putText(self.single_crop, "GJ01AB1234", (10, 25), cv2.FONT_HERSHEY_SIMPLEX, 0.65, (20, 20, 20), 2)

    def test_split_stacked_plate(self):
        top_strip, bot_strip = split_stacked_plate(self.stacked_crop)
        self.assertIsNotNone(top_strip)
        self.assertIsNotNone(bot_strip)
        # Both strips must have positive height and preserve width
        self.assertGreater(top_strip.shape[0], 10)
        self.assertGreater(bot_strip.shape[0], 10)
        self.assertEqual(top_strip.shape[1], self.stacked_crop.shape[1])
        self.assertEqual(bot_strip.shape[1], self.stacked_crop.shape[1])
        # Sum of heights should equal original height
        self.assertEqual(top_strip.shape[0] + bot_strip.shape[0], self.stacked_crop.shape[0])

    def test_tracklet_buffer_layout_classification(self):
        buf = TrackletPlateBuffer(max_crops=8)
        
        # Vehicle 1: Car with single-line plate (~4.5:1)
        buf.add_crop(track_id=101, crop=self.single_crop)
        self.assertEqual(buf.get_layout(101), "single_line")

        # Vehicle 2: Scooter with double-line plate (~2:1)
        buf.add_crop(track_id=102, crop=self.stacked_crop)
        self.assertEqual(buf.get_layout(102), "double_line")

        # Verify ring buffer max length
        for _ in range(12):
            buf.add_crop(track_id=101, crop=self.single_crop)
        self.assertLessEqual(len(buf.get_crops(101)), 8)


if __name__ == "__main__":
    unittest.main()
