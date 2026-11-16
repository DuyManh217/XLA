"""Kiểm tra ứng dụng trên ảnh biết trước, không thay thế số đo vật thật."""
import json
import tempfile
import unittest
import zipfile
from pathlib import Path
import cv2
import numpy as np
from PIL import Image
from pipeline import Settings, process, apply_grabcut, measurements, export_bundle
from segmentation import grabcut_mask


class ApplicationTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.path = Path(self.temp.name) / 'test.png'
        self.rgb = np.full((300, 400, 3), 30, np.uint8)
        self.rgb[60:241, 100:281] = [180, 190, 210]
        # Chi tiết bên trong không trở thành vật riêng ở chế độ một vật.
        self.rgb[100:130, 140:170] = 30
        Image.fromarray(self.rgb).save(self.path)

    def tearDown(self):
        self.temp.cleanup()

    def test_grabcut_one_object(self):
        selection = dict(rect=[80, 40, 300, 260], strokes=[(True, [(110, 80), (120, 90)])])
        r = process(self.path, Settings(method='GrabCut'), selection)
        self.assertEqual(len(r.objects), 1)
        self.assertAlmostEqual(r.objects[0]['long_px'], 180, delta=3)
        rows = measurements(r, 1, 18, 'cm')
        self.assertAlmostEqual(rows[0]['area'], 324, delta=10)
        self.assertAlmostEqual(rows[0]['perimeter'], 72, delta=2)

    def test_manual_polygon_and_line_calibration(self):
        selection = dict(polygon=[(100,60), (280,60), (280,240), (100,240)])
        r = process(self.path, Settings(method='GrabCut'), selection)
        rows = measurements(r, reference_size=10, reference_points=[(10,10),(110,10)], unit='mm')
        self.assertEqual(rows[0]['long'], 18)
        self.assertEqual(rows[0]['area'], 324)
        self.assertEqual(rows[0]['perimeter'], 72)
        self.assertEqual(measurements(r)[0]['unit'], 'px')

    def test_resize_calibration(self):
        r = process(self.path, Settings(method='Otsu light', blur=1, close=1, max_side=200))
        self.assertAlmostEqual(r.objects[0]['long_px'], 180, delta=4)
        rows = measurements(r, reference_size=10, reference_points=[(5,5),(55,5)])
        self.assertAlmostEqual(rows[0]['long'], 18, delta=.4)

    def test_preview_and_invalid_inputs(self):
        self.assertEqual(process(self.path, Settings(method='GrabCut')).objects, [])
        with self.assertRaises(ValueError):
            grabcut_mask(self.rgb, dict(rect=[1,1,2,2]))
        r = process(self.path)
        for size in (0, -1, np.nan, np.inf):
            with self.assertRaises(ValueError):
                measurements(r, reference_size=size, reference_points=[(0,0),(10,0)])
        with self.assertRaises(ValueError):
            measurements(r, reference_size=1, reference_points=[(0,0),(0,0)])

    def test_blank(self):
        Image.fromarray(np.zeros_like(self.rgb)).save(self.path)
        for method in ('Canny', 'Otsu light', 'Otsu dark'):
            self.assertEqual(process(self.path, Settings(method=method)).objects, [])

    def test_export(self):
        selection = dict(polygon=[(100,60),(280,60),(280,240),(100,240)])
        r = process(self.path, Settings(method='GrabCut'), selection)
        dest = Path(self.temp.name) / 'test.zip'
        export_bundle(dest, r, measurements(r))
        with zipfile.ZipFile(dest) as z:
            self.assertIsNone(z.testzip())
            meta = json.loads(z.read('metadata.json'))
            self.assertEqual(meta['selection']['polygon'][0], [100,60])
            self.assertIn('perimeter', meta['measurements'][0])
            self.assertTrue(z.read('measurements.csv').startswith(b'\xef\xbb\xbf'))

    def test_sample_counts(self):
        folder = Path(__file__).parent / 'images'
        for name, count in [('example_01.png',6), ('example_02.png',6), ('example_03.png',4)]:
            self.assertEqual(len(process(folder/name).objects), count)

    def test_hybrid_grabcut_keeps_existing_objects(self):
        # Ảnh gồm 2 vật: vật trái (tốt) và vật phải (có chi tiết gây tách mảnh).
        rgb = np.full((300, 500, 3), 30, np.uint8)
        rgb[50:150, 40:140] = [200, 200, 200]
        rgb[50:220, 260:440] = [180, 190, 210]
        rgb[90:140, 300:360] = 30
        Image.fromarray(rgb).save(self.path)
        auto = process(self.path, Settings(method='Canny'))
        left_id = auto.objects[0]['id']
        # Dùng GrabCut sửa vật bên phải trong khi giữ vật bên trái.
        selection = dict(rect=[245, 35, 455, 235], strokes=[(True, [(275, 65), (420, 200)])])
        merged = apply_grabcut(auto, Settings(method='GrabCut'), selection, keep_existing=True)
        ids = [o['id'] for o in merged.objects]
        self.assertIn(left_id, ids)
        self.assertEqual(len(merged.objects), 2)
        self.assertAlmostEqual(merged.objects[-1]['long_px'], 180, delta=4)

    def test_polygon_with_strokes_runs_grabcut(self):
        # Vẽ đa giác rộng quanh vật rồi tô nét Loại nền lên vùng thừa bên trong đa giác.
        selection = dict(
            polygon=[(70, 35), (320, 35), (320, 265), (70, 265)],
            strokes=[(False, [(80, 45), (310, 45)]), (True, [(120, 80), (260, 220)])],
        )
        r = process(self.path, Settings(method='GrabCut'), selection)
        self.assertEqual(len(r.objects), 1)
        self.assertAlmostEqual(r.objects[0]['long_px'], 180, delta=4)


if __name__ == '__main__':
    unittest.main()
