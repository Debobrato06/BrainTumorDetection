"""Tests for the 2D tumor analysis (src/seg2d.py, src/pipeline/analysis_2d.py) and its use in
the app. Synthetic fixtures check geometry and conventions; segmentation accuracy is measured
by src/evaluate_seg2d.py.

    python -m unittest discover -s tests -p test_seg2d.py -v
"""
import base64
import io
import os
import sys
import unittest

import numpy as np
from PIL import Image

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
sys.path.insert(0, os.path.join(ROOT, 'src'))

from seg2d import CLASSES, INPUT_SIZE, TumorSegNet2D, prepare_slice  # noqa: E402
from pipeline import analysis_2d as A2  # noqa: E402

HAS_SEG2D = os.path.exists(os.path.join(ROOT, 'models', 'seg2d_best.pth'))
GLIOMA = np.array([0.02, 0.9, 0.05, 0.03])


def synthetic_head(tumor_center=None, radius=15, size=256, head=(100, 80)):
    """Uniform elliptical 'head' (half-axes head = (a_x, b_y)) on black, optional bright disk.
    Returns (uint8 image, tumor probability map)."""
    yy, xx = np.mgrid[:size, :size]
    c = size / 2
    img = np.zeros((size, size), np.float32)
    img[((xx - c) / head[0]) ** 2 + ((yy - c) / head[1]) ** 2 <= 1] = 100
    prob = np.zeros((size, size), np.float32)
    if tumor_center is not None:
        disk = (xx - tumor_center[0]) ** 2 + (yy - tumor_center[1]) ** 2 <= radius ** 2
        img[disk] = 220
        prob[disk] = 0.95
    return img.astype(np.uint8), prob


class TestPrepareSlice(unittest.TestCase):
    def test_non_square_is_padded_centered(self):
        img = Image.fromarray(np.full((100, 200), 255, np.uint8))
        out, info = prepare_slice(img)
        self.assertEqual(out.shape, (INPUT_SIZE, INPUT_SIZE))
        self.assertEqual(info['pad_top'], 50)
        self.assertEqual(info['orig_px_per_input_px'], 200 / INPUT_SIZE)
        # top and bottom quarter are padding
        self.assertEqual(out[:INPUT_SIZE // 4 - 2].max(), 0)
        self.assertGreater(out[INPUT_SIZE // 2].min(), 200)

    def test_model_output_shapes(self):
        import torch
        m = TumorSegNet2D(pretrained=False).eval()
        with torch.no_grad():
            seg, cls = m(torch.rand(1, 1, 128, 128))
        self.assertEqual(tuple(seg.shape), (1, 1, 128, 128))
        self.assertEqual(tuple(cls.shape), (1, len(CLASSES)))


class TestAnalysis(unittest.TestCase):
    def analyze(self, center, **kw):
        img, prob = synthetic_head(center, **{k: v for k, v in kw.items() if k in ('radius', 'head')})
        return A2.analyze_slice(img, prob, GLIOMA, CLASSES, pixel_spacing_mm=kw.get('spacing'))

    def test_side_uses_radiological_convention(self):
        # image left = patient right
        self.assertEqual(self.analyze((70, 128))['side'], 'Right side of the brain')
        self.assertEqual(self.analyze((186, 128))['side'], 'Left side of the brain')
        self.assertTrue(self.analyze((128, 128))['side'].startswith('Midline'))

    def test_diameter_with_known_spacing(self):
        a = self.analyze((90, 120), radius=15, spacing=0.5)
        self.assertAlmostEqual(a['longest_mm'], 31 * 0.5, delta=1.0)
        self.assertAlmostEqual(a['perpendicular_mm'], 31 * 0.5, delta=1.0)
        self.assertAlmostEqual(a['area_mm2'], np.pi * 15 ** 2 * 0.25, delta=15)
        self.assertEqual(a['scale_source'], 'provided')

    def test_scale_estimated_from_head_width(self):
        a = self.analyze((90, 120))
        self.assertEqual(a['view'], 'axial_or_coronal')
        self.assertEqual(a['scale_source'], 'estimated_from_head_width')
        # head is 2*100+1 px wide; smoothing of the head mask may add a pixel per side
        self.assertAlmostEqual(a['mm_per_px'], A2.HEAD_WIDTH_MM / 201, delta=0.02 * A2.HEAD_WIDTH_MM / 201)

    def test_no_tumor(self):
        img, _ = synthetic_head(None)
        a = A2.analyze_slice(img, np.zeros(img.shape, np.float32), GLIOMA, CLASSES)
        self.assertFalse(a['tumor_found'])

    def test_mask_outside_head_ignored(self):
        img, _ = synthetic_head(None)
        prob = np.zeros(img.shape, np.float32)
        prob[2:20, 2:20] = 0.9            # corner, outside the head
        self.assertFalse(A2.analyze_slice(img, prob, GLIOMA, CLASSES)['tumor_found'])

    def test_risk_orders_cases(self):
        big_glioma = self.analyze((90, 128), radius=30, spacing=1.0)
        big_glioma['risk'] = A2.risk_estimate_2d(big_glioma)
        img, prob = synthetic_head((90, 128), radius=5)
        small_men = A2.analyze_slice(img, prob, np.array([0.02, 0.03, 0.9, 0.05]), CLASSES, pixel_spacing_mm=1.0)
        small_men['risk'] = A2.risk_estimate_2d(small_men)
        self.assertGreater(big_glioma['risk']['total_score'], small_men['risk']['total_score'])
        for r in (big_glioma['risk'], small_men['risk']):
            self.assertEqual(sum(f['score'] for f in r['factor_breakdown']), r['total_score'])
            self.assertIn(r['risk_level'], ('Low', 'Moderate', 'High', 'Critical'))

    def test_overlay_matches_original_size(self):
        img, prob = synthetic_head((80, 128))
        pil = Image.fromarray(img).resize((300, 300))
        _, info = prepare_slice(pil)
        a = A2.analyze_slice(img, prob, GLIOMA, CLASSES)
        png = Image.open(io.BytesIO(base64.b64decode(A2.render_overlay(pil, a, info, INPUT_SIZE))))
        self.assertEqual(png.size, (300, 300))
        changed = np.abs(np.asarray(png).astype(int) - np.asarray(pil.convert('RGB')).astype(int)).sum(-1) > 30
        # tumor drawn on the image-left half only
        self.assertGreater(changed[:, :150].sum(), 50)
        self.assertEqual(changed[:, 170:].sum(), 0)


@unittest.skipUnless(HAS_SEG2D, 'models/seg2d_best.pth not present')
class TestAppWith2DModel(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        import app
        cls.A = app
        cls.client = app.app.test_client()

    def post(self, path, **form):
        with open(path, 'rb') as f:
            data = dict(form, file=(io.BytesIO(f.read()), os.path.basename(path)))
            return self.client.post('/predict', data=data, content_type='multipart/form-data').get_json()

    def test_tumor_image_gets_size_side_type_risk(self):
        p = os.path.join(ROOT, 'dummy_data', 'Testing', 'glioma', 'Te-gl_0100.jpg')
        if not os.path.exists(p):
            self.skipTest('Kaggle test image not present')
        d = self.post(p)
        self.assertTrue(d['label'].startswith('POSITIVE'))
        self.assertTrue(d['overlay_b64'])
        m = d['measurements']
        self.assertGreater(m['diameter_mm'], 3)
        self.assertLess(m['diameter_mm'], 120)
        self.assertGreaterEqual(m['diameter_mm'], m['perpendicular_mm'])
        t = d['tumor_analysis_2d']
        self.assertTrue(t['side'])
        self.assertIn(d['risk']['risk_level'], ('Low', 'Moderate', 'High', 'Critical'))
        self.assertIn(t['tumor_type'], CLASSES[1:])

    def test_pixel_spacing_form_field_scales_sizes(self):
        p = os.path.join(ROOT, 'dummy_data', 'Testing', 'glioma', 'Te-gl_0100.jpg')
        if not os.path.exists(p):
            self.skipTest('Kaggle test image not present')
        a = self.post(p, pixel_spacing_mm='0.5')
        b = self.post(p, pixel_spacing_mm='1.0')
        self.assertEqual(a['tumor_analysis_2d']['scale_source'], 'provided')
        self.assertAlmostEqual(b['measurements']['diameter_mm'], 2 * a['measurements']['diameter_mm'], delta=0.2)

    def test_low_outline_confidence_is_flagged(self):
        img, prob = synthetic_head((80, 128))
        a = A2.analyze_slice(img, prob * 0.6, GLIOMA, CLASSES, threshold=0.5)  # mean prob 0.57
        a['risk'], a['overlay_b64'] = A2.risk_estimate_2d(a), ''
        r = self.A.build_2d_report(a, 0.9)
        self.assertTrue(r['tumor_analysis_2d']['outline_low_confidence'])
        self.assertTrue(r['impression'].startswith('CAUTION'))

    def test_no_tumor_image(self):
        p = os.path.join(ROOT, 'dummy_data', 'Testing', 'notumor', 'Te-no_0100.jpg')
        if not os.path.exists(p):
            self.skipTest('Kaggle test image not present')
        d = self.post(p)
        self.assertTrue(d['label'].startswith('NEGATIVE'))
        self.assertEqual(d['overlay_b64'], '')
        self.assertNotIn('tumor_analysis_2d', d)


if __name__ == '__main__':
    unittest.main()
