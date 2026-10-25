"""Tests for the web app's inference pipeline (app.py). Synthetic fixtures check software
behaviour only; they say nothing about detection accuracy (see src/evaluate_app_model.py).

    python -m unittest discover -s tests -p test_app.py -v
"""
import io
import os
import sys
import tempfile
import unittest

import numpy as np
import torch
from PIL import Image

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
sys.path.insert(0, os.path.join(ROOT, 'src'))

import app as A  # noqa: E402  (loads best_model.pth if present)
from dataset import Image2DDataset, image_to_volume  # noqa: E402
from model import BrainTumorClassifier  # noqa: E402
from pipeline.segmentation import apply_threshold  # noqa: E402

HAS_CKPT = os.path.exists(os.path.join(ROOT, 'best_model.pth'))


def png_bytes(size=(128, 128), seed=0):
    rng = np.random.default_rng(seed)
    img = Image.fromarray(rng.integers(0, 255, size=size, dtype=np.uint8))
    buf = io.BytesIO()
    img.save(buf, format='PNG')
    buf.seek(0)
    return buf


def post(client, data, name):
    return client.post('/predict', data={'file': (data, name)}, content_type='multipart/form-data')


def write_nifti(data, affine, path):
    import nibabel as nib
    nib.save(nib.Nifti1Image(data.astype(np.float32), affine), path)


class StubModel(torch.nn.Module):
    """Stands in for a model with a trained segmentation head: predicts "tumor" and a cube."""
    def __init__(self, cube):
        super().__init__()
        self.cube = cube
        self.dummy = torch.nn.Parameter(torch.zeros(1))

    def forward(self, x, return_attention=False):
        seg = torch.full((1, 3, *x.shape[2:]), -10.0)
        (d0, d1), (h0, h1), (w0, w1) = self.cube
        seg[:, :, d0:d1, h0:h1, w0:w1] = 10.0
        cls = torch.tensor([[-5.0, 5.0]])
        out = (cls, seg, torch.zeros(1, 2), torch.zeros(1, 5))
        return out + ({'T1': .5, 'T1c': .5, 'T2': .5, 'FLAIR': .5},) if return_attention else out[:3]

    def forward_heads(self, x):
        return torch.tensor([[-5.0, 5.0]]), torch.zeros(1, 2), torch.zeros(1, 5), torch.full((1, 4), .5)


class TestInputValidation(unittest.TestCase):
    def setUp(self):
        self.client = A.app.test_client()

    def test_unsupported_extension_rejected(self):
        r = post(self.client, io.BytesIO(b'hello'), 'notes.txt')
        self.assertEqual(r.status_code, 400)
        self.assertIn('Unsupported', r.get_json()['error'])

    def test_corrupted_image_rejected(self):
        r = post(self.client, io.BytesIO(b'\x89PNG\r\n\x1a\n not really a png'), 'scan.png')
        self.assertEqual(r.status_code, 400)

    def test_tiny_image_rejected(self):
        r = post(self.client, png_bytes((8, 8)), 'scan.png')
        self.assertEqual(r.status_code, 400)

    def test_bad_nifti_rejected(self):
        r = post(self.client, io.BytesIO(b'not a nifti'), 'scan.nii')
        self.assertEqual(r.status_code, 400)

    def test_nifti_with_wrong_channel_count_rejected(self):
        with tempfile.TemporaryDirectory() as d:
            p = os.path.join(d, 'x.nii.gz')
            write_nifti(np.random.rand(16, 16, 16, 3), np.eye(4), p)
            with open(p, 'rb') as f:
                r = post(self.client, io.BytesIO(f.read()), 'x.nii.gz')
        self.assertEqual(r.status_code, 400)

    def test_no_temp_files_left(self):
        temp_files = lambda: {f for f in os.listdir(os.path.join(ROOT, 'data')) if f.startswith('temp_upload')}
        before = temp_files()
        post(self.client, io.BytesIO(b'\x89PNG broken'), 'scan.png')
        post(self.client, png_bytes(), 'scan.png')
        self.assertEqual(temp_files() - before, set())


class TestPreprocessing(unittest.TestCase):
    def test_image_to_volume_shape_dtype_range(self):
        v = image_to_volume(Image.fromarray(np.full((200, 150), 255, np.uint8)), (64, 64, 64))
        self.assertEqual(tuple(v.shape), (4, 64, 64, 64))
        self.assertEqual(v.dtype, torch.float32)
        self.assertAlmostEqual(float(v.max()), 1.0)       # [0, 1], not z-scored
        self.assertGreaterEqual(float(v.min()), 0.0)

    def test_color_image_becomes_grayscale(self):
        rgb = Image.fromarray(np.dstack([np.full((64, 64), c, np.uint8) for c in (255, 0, 0)]))
        v = image_to_volume(rgb, (8, 64, 64))
        self.assertTrue(torch.allclose(v[0, 0], v[3, 7]))  # same slice in every channel / depth

    def test_app_input_matches_training_input(self):
        with tempfile.TemporaryDirectory() as d:
            p = os.path.join(d, 'x.png')
            with open(p, 'wb') as f:
                f.write(png_bytes((97, 131)).read())
            app_in = A.model_input_2d(p)[0]
            train_in = Image2DDataset(d, img_size=A.MODEL_IMG_SIZE[1:], files=[(p, 1)],
                                      depth=A.MODEL_IMG_SIZE[0])[0][0]
        self.assertTrue(torch.equal(app_in, train_in))


class TestNiftiLoading(unittest.TestCase):
    """The app arranges NIfTI data as (D=axial inferior->superior, H=anterior->posterior,
    W=patient right->left) whatever the file's orientation."""

    def marker_volume(self):
        vol = np.zeros((40, 50, 30), np.float32) + 1.0  # (x: L->R, y: P->A, z: I->S) in RAS
        vol[34:38, 44:48, 25:28] = 100.0                # right, anterior, superior corner
        return vol

    def locate(self, path):
        t, spacing = A.load_nifti_for_app(path, img_size=(32, 32, 32))
        idx = np.unravel_index(int(t[0, 0].argmax()), t.shape[2:])
        return np.array(idx) / 32.0, t, spacing

    def test_orientation_ras(self):
        with tempfile.TemporaryDirectory() as d:
            p = os.path.join(d, 'ras.nii.gz')
            write_nifti(self.marker_volume(), np.diag([1.0, 1.0, 1.0, 1.0]), p)
            (dz, hy, wx), t, _ = self.locate(p)
        self.assertEqual(tuple(t.shape), (1, 4, 32, 32, 32))
        self.assertGreater(dz, 0.75)   # superior -> high slice index
        self.assertLess(hy, 0.25)      # anterior -> top rows
        self.assertLess(wx, 0.25)      # patient right -> left columns (radiological)

    def test_orientation_independent_of_storage_order(self):
        # Same anatomy stored LPS (x and y flipped): result must be identical
        vol = self.marker_volume()[::-1, ::-1, :].copy()
        affine = np.diag([-1.0, -1.0, 1.0, 1.0])
        affine[:3, 3] = [39, 49, 0]
        with tempfile.TemporaryDirectory() as d:
            p = os.path.join(d, 'lps.nii.gz')
            write_nifti(vol, affine, p)
            (dz, hy, wx), _, _ = self.locate(p)
        self.assertGreater(dz, 0.75)
        self.assertLess(hy, 0.25)
        self.assertLess(wx, 0.25)

    def test_grid_spacing(self):
        with tempfile.TemporaryDirectory() as d:
            p = os.path.join(d, 's.nii.gz')
            write_nifti(self.marker_volume(), np.diag([0.5, 2.0, 3.0, 1.0]), p)
            _, _, s = self.locate(p)
        self.assertAlmostEqual(s['pixel_spacing_x'], 0.5 * 40 / 32)
        self.assertAlmostEqual(s['pixel_spacing_y'], 2.0 * 50 / 32)
        self.assertAlmostEqual(s['slice_thickness'], 3.0 * 30 / 32)


class TestModel(unittest.TestCase):
    def test_forward_heads_matches_forward(self):
        torch.manual_seed(0)
        m = BrainTumorClassifier(in_channels=4).eval()
        x = torch.randn(1, 4, 32, 32, 32)
        with torch.no_grad():
            full, heads = m(x, return_attention=True), m.forward_heads(x)
        self.assertTrue(torch.allclose(full[0], heads[0], atol=1e-6))
        self.assertTrue(torch.allclose(full[2], heads[1], atol=1e-6))
        self.assertTrue(torch.allclose(full[3], heads[2], atol=1e-6))

    @unittest.skipUnless(HAS_CKPT, 'best_model.pth not present')
    def test_checkpoint_loads_strictly(self):
        ckpt = torch.load(os.path.join(ROOT, 'best_model.pth'), map_location='cpu', weights_only=False)
        m = BrainTumorClassifier(in_channels=4, num_classes=2)
        m.load_state_dict(ckpt['model_state_dict'], strict=True)
        with torch.no_grad():
            out = m.forward_heads(torch.zeros(1, 4, 32, 32, 32))
        self.assertEqual(tuple(out[0].shape), (1, 2))

    def test_threshold_masks_are_nested_booleans(self):
        logits = torch.randn(1, 3, 8, 8, 8) * 3
        wt, prob, sub = apply_threshold(logits, 0.5)
        for k in ('ET', 'TC', 'WT'):
            self.assertEqual(sub[k].dtype, bool)
        self.assertTrue(np.all(sub['ET'] <= sub['TC']) and np.all(sub['TC'] <= sub['WT']))


class TestPredictOutputs(unittest.TestCase):
    def setUp(self):
        self.client = A.app.test_client()
        self.saved = (A.model, set(A.MODEL_TRAINED_HEADS), A.TORCH_AVAILABLE, A.MODEL_IMG_SIZE, A.SEG2D_MODEL)

    def tearDown(self):
        A.model, A.MODEL_TRAINED_HEADS, A.TORCH_AVAILABLE, A.MODEL_IMG_SIZE, A.SEG2D_MODEL = self.saved

    @unittest.skipUnless(HAS_CKPT, 'best_model.pth not present')
    def test_no_localization_without_trained_segmentation(self):
        # best_model.pth trains detection only: without the 2D segmentation model there is no
        # overlay, mask, location, size, type or risk
        self.assertNotIn('seg', A.MODEL_TRAINED_HEADS)
        A.SEG2D_MODEL = None
        r = post(self.client, png_bytes(), 'scan.png')
        self.assertEqual(r.status_code, 200)
        d = r.get_json()
        self.assertEqual(d['overlay_b64'], '')
        self.assertFalse(d['model_status']['segmentation_available'])
        self.assertIsNone(d['measurements']['volume_ml'])
        self.assertIn(d['measurements']['voxel_count'], (None, 0))
        self.assertEqual(d['radiomics'], {})
        self.assertEqual(d['risk']['risk_level'], 'N/A')
        self.assertIn(d['label'].split(':')[0], ('POSITIVE', 'NEGATIVE'))
        if d['label'].startswith('POSITIVE'):
            self.assertIn('does not predict tumor location', d['structured_report']['clinical_impression'])

    def test_model_unavailable_is_indeterminate(self):
        A.model = None
        A.SEG2D_MODEL = None
        d = post(self.client, png_bytes(), 'scan.png').get_json()
        self.assertTrue(d['is_indeterminate'])
        self.assertTrue(d['label'].startswith('INDETERMINATE'))
        self.assertEqual(d['overlay_b64'], '')

    @unittest.skipUnless(A.SEG2D_MODEL is not None, 'models/seg2d_best.pth not present')
    def test_2d_model_alone_gives_a_decision(self):
        A.model = None
        d = post(self.client, png_bytes(), 'scan.png').get_json()
        self.assertFalse(d['is_indeterminate'])
        self.assertIn(d['label'].split(':')[0], ('POSITIVE', 'NEGATIVE'))

    def test_trained_segmentation_mask_is_aligned_and_measured(self):
        # With a (stub) trained segmentation head, the mask lives on the model grid and is
        # measured with the grid's real spacing.
        A.model = StubModel(((40, 48), (10, 18), (20, 28)))
        A.MODEL_TRAINED_HEADS = {'cls', 'seg'}
        A.MODEL_IMG_SIZE = (64, 64, 64)  # the stub's box and the expected spacing assume a 64^3 grid
        affine = np.diag([1.0, 1.0, 2.0, 1.0])
        with tempfile.TemporaryDirectory() as d:
            p = os.path.join(d, 'x.nii.gz')
            write_nifti(np.random.rand(128, 128, 64) + 1, affine, p)
            with open(p, 'rb') as f:
                r = post(self.client, io.BytesIO(f.read()), 'x.nii.gz')
        self.assertEqual(r.status_code, 200)
        out = r.get_json()
        self.assertTrue(out['model_status']['segmentation_available'])
        self.assertTrue(out['overlay_b64'])
        self.assertEqual(out['measurements']['voxel_count'], 8 * 8 * 8)
        # one grid voxel = (128/64 mm) * (128/64 mm) * (64*2/64 mm) = 8 mm^3 -> 512 voxels = 4.10 mL
        self.assertAlmostEqual(out['measurements']['volume_ml'], 512 * 8 / 1000, places=2)
        bb = out['measurements']['bounding_box']
        self.assertTrue(bb)


@unittest.skipUnless(HAS_CKPT and os.path.isdir(os.path.join(ROOT, 'dummy_data', 'Testing')),
                     'checkpoint or Kaggle test images not present')
class TestPredictionRegression(unittest.TestCase):
    """Held-out Kaggle test images whose expected class matches their ground-truth label and
    best_model.pth's recorded prediction (runs/app_model_eval). Catches preprocessing drift."""
    CASES = {
        'dummy_data/Testing/glioma/Te-gl_0100.jpg': 'POSITIVE',
        'dummy_data/Testing/pituitary/Te-pi_0100.jpg': 'POSITIVE',
        'dummy_data/Testing/meningioma/Te-me_0100.jpg': 'POSITIVE',
        'dummy_data/Testing/notumor/Te-no_0100.jpg': 'NEGATIVE',
        'dummy_data/Testing/notumor/Te-no_0010.jpg': 'NEGATIVE',
    }

    def test_known_cases(self):
        client = A.app.test_client()
        for rel, expected in self.CASES.items():
            with open(os.path.join(ROOT, rel), 'rb') as f:
                d = post(client, io.BytesIO(f.read()), os.path.basename(rel)).get_json()
            self.assertTrue(d['label'].startswith(expected), f'{rel}: {d["label"]}')


if __name__ == '__main__':
    unittest.main()
