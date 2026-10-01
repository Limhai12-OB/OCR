import unittest
from unittest.mock import patch

import numpy as np

from ocr.preprocessing import _deskew


class DeskewCompatibilityTests(unittest.TestCase):
    def test_accepts_both_opencv_hough_line_shapes(self):
        gray = np.full((200, 200), 255, dtype=np.uint8)
        variants = (
            np.array([[0, 0, 100, 3], [0, 30, 100, 33]], dtype=np.int32),
            np.array([[[0, 0, 100, 3]], [[0, 30, 100, 33]]], dtype=np.int32),
        )
        for lines in variants:
            with self.subTest(shape=lines.shape), patch(
                "ocr.preprocessing.cv2.HoughLinesP", return_value=lines
            ):
                corrected, inverse = _deskew(gray)
                self.assertEqual(corrected.shape, gray.shape)
                self.assertEqual(inverse.shape, (3, 3))


if __name__ == "__main__":
    unittest.main()
