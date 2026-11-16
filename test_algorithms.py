"""Chạy: python -m unittest -v."""
import unittest
import cv2
import numpy as np
from contours import find_contours, contour_area, contour_perimeter, sort_contours, min_area_rect, box_points
from io_utils import order_points
from morphology import closing, fill_holes
from spatial_filters import gaussian_blur
from edge_detection import canny_edge_detection
from measurement_utils import prepare_measurement


class AlgorithmTests(unittest.TestCase):
    def check_shape(self, mask):
        contours = find_contours(mask)
        expected, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
        self.assertEqual(len(contours), len(expected))
        contours, _ = sort_contours(contours)
        expected, _ = sort_contours(expected)
        for c, e in zip(contours, expected):
            self.assertAlmostEqual(contour_area(c), cv2.contourArea(e), places=5)
            self.assertAlmostEqual(contour_perimeter(c), cv2.arcLength(e, True), places=4)

    def test_rectangle_and_concave(self):
        mask = np.zeros((100, 150), np.uint8)
        mask[10:40, 10:50] = 255
        mask[50:90, 70:90] = 255
        mask[70:90, 70:130] = 255
        self.check_shape(mask)

    def test_circle(self):
        y, x = np.indices((80, 80))
        mask = (((x - 40) ** 2 + (y - 40) ** 2) <= 25 ** 2).astype(np.uint8) * 255
        self.check_shape(mask)

    def test_diagonal_connection(self):
        mask = np.zeros((50, 50), np.uint8)
        mask[5:20, 5:20] = 255
        mask[20:35, 20:35] = 255
        self.check_shape(mask)

    def test_rotated_box(self):
        box = order_points(np.array([[10, 0], [20, 10], [10, 20], [0, 10]], np.float32))
        self.assertEqual(len(np.unique(box, axis=0)), 4)
        np.testing.assert_allclose(np.linalg.norm(box - np.roll(box, 1, axis=0), axis=1), np.sqrt(200))
        for angle in (0, 15, 45, 75):
            pts = box_points(((60, 60), (40, 20), angle)).reshape(-1, 1, 2)
            np.testing.assert_allclose(sorted(min_area_rect(pts)[1]), [20, 40], atol=0.001)

    def test_empty_and_invalid_reference(self):
        self.assertEqual(sort_contours([])[0], [])
        c = np.array([[[0, 0]], [[20, 0]], [[20, 20]], [[0, 20]]])
        for value in (0, -1, np.nan, np.inf):
            with self.assertRaises(ValueError):
                prepare_measurement([c], value)
        with self.assertRaises(ValueError):
            prepare_measurement([], 1)
        with self.assertRaises(ValueError):
            prepare_measurement([c], 1, 2)
        _, ppm = prepare_measurement([c], 2)
        self.assertAlmostEqual(ppm, 10)

    def test_holes_and_closing(self):
        mask = np.zeros((50, 50), np.uint8)
        mask[5:45, 5:45] = 255
        mask[15:35, 15:35] = 0
        self.assertEqual(np.count_nonzero(fill_holes(mask)), 1600)
        kernel = np.ones((3, 3), np.uint8)
        np.testing.assert_array_equal(closing(mask, kernel), cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel))

    def test_constant_and_object_pipeline(self):
        blank = np.full((70, 90), 128, np.uint8)
        np.testing.assert_allclose(gaussian_blur(blank), blank, atol=1)
        self.assertEqual(np.count_nonzero(canny_edge_detection(blank)[2:-2, 2:-2]), 0)
        mask = np.zeros((70, 90), np.uint8)
        mask[15:55, 20:70] = 255
        edges = canny_edge_detection(gaussian_blur(mask))
        filled = fill_holes(closing(edges, np.ones((3, 3), np.uint8)))
        found = find_contours(filled)
        self.assertEqual(len(found), 1)
        self.assertGreater(contour_area(found[0]), 1700)


if __name__ == "__main__":
    unittest.main()
