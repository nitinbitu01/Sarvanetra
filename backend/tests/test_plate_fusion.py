import unittest
import numpy as np
import cv2
from backend.scripts.plate_multiframe_fusion import fuse, _to_gray


class TestPlateFusion(unittest.TestCase):
    def setUp(self):
        # Create a synthetic sharp plate pattern
        self.canvas = np.full((48, 160), 220, dtype=np.uint8)
        cv2.putText(self.canvas, "GJ01AB1234", (10, 32), cv2.FONT_HERSHEY_SIMPLEX, 0.8, 20, 2)

    def test_fuse_subpixel_shifts(self):
        # Generate 4 sub-pixel translated & slightly blurred views
        crops = []
        shifts = [(0.0, 0.0), (0.7, 0.4), (-0.5, 0.6), (0.3, -0.4)]
        for dx, dy in shifts:
            M = np.float32([[1, 0, dx], [0, 1, dy]])
            shifted = cv2.warpAffine(self.canvas, M, (160, 48), flags=cv2.INTER_LINEAR)
            blurred = cv2.GaussianBlur(shifted, (3, 3), 0.5)
            crops.append(blurred)

        fused, used = fuse(crops)
        self.assertIsNotNone(fused)
        self.assertGreaterEqual(used, 2)
        self.assertEqual(fused.shape[0], 48 * 3)  # Canonical height * UPSCALE (3)

    def test_fuse_bgr_support(self):
        # Test that 3-channel BGR crops work without error
        bgr_crops = [cv2.cvtColor(self.canvas, cv2.COLOR_GRAY2BGR) for _ in range(3)]
        fused, used = fuse(bgr_crops)
        self.assertIsNotNone(fused)
        self.assertGreaterEqual(used, 1)

    def test_fuse_fallback_on_incompatible(self):
        # Test with random noise images that cannot align
        noise_crops = [np.random.randint(0, 256, (48, 160), dtype=np.uint8) for _ in range(3)]
        fused, used = fuse(noise_crops)
        # Should gracefully fall back to sharpest single crop (used=1) rather than crashing
        self.assertIsNotNone(fused)
        self.assertEqual(used, 1)


if __name__ == "__main__":
    unittest.main()
