import json
import math
from pathlib import Path
import tempfile
import unittest
import zipfile
import cv2
import numpy as np
from PIL import Image
from pipeline import Settings, process, measurements, export_bundle, load_image
from contours import find_contours, contour_area, sort_contours
from morphology import dilation, erosion
from spatial_filters import gaussian_blur
from segmentation import grabcut_mask


class PipelineTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.folder = Path(self.temp.name)

    def tearDown(self):
        self.temp.cleanup()

    def save(self, array, name='ảnh đo tiếng Việt.png'):
        p = self.folder/name
        Image.fromarray(array).save(p)
        return p

    def rectangles(self):
        a = np.zeros((400, 700, 3), np.uint8)
        a[70:171, 60:161] = 255
        a[220:301, 300:501] = 255
        return self.save(a)

    def test_known_dimensions_and_calibration(self):
        r = process(self.rectangles(), Settings(method='Otsu light', blur=1, close=1))
        self.assertEqual(len(r.objects), 2)
        rows = measurements(r, 1, 10, 'mm')
        self.assertAlmostEqual(rows[0]['long'], 10, places=3)
        self.assertAlmostEqual(rows[1]['long'], 20, delta=.2)
        self.assertAlmostEqual(rows[1]['short'], 8, delta=.2)

    def test_resize_preserves_pixel_dimensions(self):
        path = self.rectangles()
        a = measurements(process(path, Settings(method='Otsu light', blur=1, close=1)))
        b = measurements(process(path, Settings(method='Otsu light', blur=1, close=1, max_side=350)))
        self.assertEqual(len(b), 2)
        self.assertAlmostEqual(a[1]['long'], b[1]['long'], delta=3)

    def test_rotated_box(self):
        a = np.zeros((400, 500, 3), np.uint8)
        box = cv2.boxPoints(((250, 200), (180, 60), 37)).astype(np.int32)
        cv2.fillPoly(a, [box], (255, 255, 255))
        r = process(self.save(a), Settings(method='Otsu light', blur=1, close=1))
        self.assertEqual(len(r.objects), 1)
        self.assertAlmostEqual(r.objects[0]['long_px'], 180, delta=3)
        self.assertAlmostEqual(r.objects[0]['short_px'], 60, delta=3)

    def test_blank_and_empty_contours(self):
        for value in (0, 255):
            for method in ('Canny', 'Otsu light', 'Otsu dark'):
                r = process(self.save(np.full((300, 300, 3), value, np.uint8)), Settings(method=method))
                self.assertEqual(r.objects, [])
        self.assertEqual(sort_contours([])[1].shape, (0, 4))

    def test_ordered_area(self):
        a = np.zeros((200, 200), np.uint8)
        a[30:131, 20:121] = 255
        self.assertEqual(contour_area(find_contours(a)[0]), 10000)

    def test_morphology_matches_explicit_zero_border(self):
        rng = np.random.default_rng(42)
        a = (rng.random((40, 60)) > .5).astype(np.uint8)*255
        k = np.ones((3, 3), np.uint8)
        np.testing.assert_array_equal(dilation(a, k), cv2.dilate(a, k, borderType=cv2.BORDER_CONSTANT, borderValue=0))
        np.testing.assert_array_equal(erosion(a, k), cv2.erode(a, k, borderType=cv2.BORDER_CONSTANT, borderValue=0))

    def test_no_false_gaussian_border(self):
        blur = gaussian_blur(np.full((40, 50), 200, np.uint8))
        self.assertLessEqual(int(blur.max())-int(blur.min()), 1)
        self.assertGreaterEqual(int(blur.min()), 199)

    def test_input_modes(self):
        for array in [np.full((200, 200), 128, np.uint8), np.zeros((200, 200, 4), np.uint8)]:
            image = load_image(self.save(array))
            self.assertEqual(image.shape, (200, 200, 3))
        self.assertTrue(np.all(image == 255))

    def test_invalid_inputs(self):
        for config in [Settings(low=120, high=50), Settings(min_area=math.nan), Settings(blur=4), Settings(max_side=9999)]:
            with self.assertRaises(ValueError):
                config.validate()
        bad = self.folder/'bad.png'
        bad.write_text('not an image')
        with self.assertRaises(OSError):
            process(bad)
        r = process(self.rectangles())
        for size in (0, -1, math.inf, math.nan):
            with self.assertRaises(ValueError):
                measurements(r, 1, size)
        with self.assertRaises(ValueError):
            measurements(r, 999, 10)

    def test_export(self):
        r = process(self.rectangles())
        out = self.folder/'output.zip'
        export_bundle(out, r, measurements(r))
        with zipfile.ZipFile(out) as z:
            self.assertEqual(len(z.namelist()), 9)
            meta = json.loads(z.read('metadata.json'))
            self.assertEqual(len(meta['measurements']), 2)
            self.assertTrue(z.read('measurements.csv').startswith(b'\xef\xbb\xbf'))
            self.assertIsNone(z.testzip())

    def test_samples(self):
        folder = Path(__file__).resolve().parents[1]/'images'
        for name, count in [('example_01.png', 6), ('example_02.png', 6), ('example_03.png', 4)]:
            self.assertEqual(len(process(folder/name).objects), count)

    def test_guided_grabcut_selects_one_object(self):
        image = np.full((120, 160, 3), 25, np.uint8)
        image[25:100, 45:120] = (180, 190, 205)
        selection = {'rect': [30, 15, 135, 110], 'strokes': [(True, [(60, 40), (90, 70)])]}
        mask = grabcut_mask(image, selection)
        self.assertGreater(np.count_nonzero(mask[25:100, 45:120]), 5000)
        self.assertLess(np.count_nonzero(mask[:15]), 20)

    def test_manual_polygon_mask(self):
        image = np.zeros((100, 120, 3), np.uint8)
        mask = grabcut_mask(image, {'polygon': [(10, 10), (80, 10), (80, 60), (10, 60)]})
        self.assertEqual(np.count_nonzero(mask), 71 * 51)

    def test_grabcut_requires_user_roi(self):
        with self.assertRaises(ValueError):
            grabcut_mask(np.zeros((30, 30, 3), np.uint8), None)


if __name__ == '__main__':
    unittest.main()
