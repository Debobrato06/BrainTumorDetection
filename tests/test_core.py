"""
Fast CPU tests for the research code. Run from the project root:

    python -m unittest discover -s tests -v      (or: pytest tests)
"""
import os
import sys
import tempfile
import unittest

import numpy as np
import torch

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

from brats_dataset import BraTSEvalDataset, BraTSTrainDataset, brats_label_to_regions, cases_for, zscore_nonzero  # noqa: E402
from hybrid_model import HybridTumorModel, enable_mc_dropout, mc_dropout_predict  # noqa: E402
from losses import MultiTaskLoss, TverskyLoss  # noqa: E402
from metrics import HD95_EMPTY_PENALTY, dice, expected_calibration_error, hd95  # noqa: E402

SIZE = (32, 32, 32)


def small_model(**kw):
    return HybridTumorModel(img_size=SIZE, patch_size=4, embed_dim=32, depth=2, num_heads=2, **kw)


class TestHybridModel(unittest.TestCase):
    def setUp(self):
        torch.manual_seed(0)

    def test_output_shapes(self):
        m = small_model()
        cls, seg, grade = m(torch.randn(2, 4, *SIZE))
        self.assertEqual(cls.shape, (2, 2))
        self.assertEqual(seg.shape, (2, 3, *SIZE))
        self.assertEqual(grade.shape, (2, 2))

    def test_segmentation_uses_transformer(self):
        """The old model ignored the transformer output in the decoder."""
        m = small_model()
        _, seg, _ = m(torch.randn(1, 4, *SIZE))
        seg.sum().backward()
        grad = m.blocks[-1].mlp[-2].weight.grad
        self.assertIsNotNone(grad)
        self.assertGreater(grad.abs().sum().item(), 0)

    def test_return_attention(self):
        m = small_model()
        out = m(torch.randn(1, 4, *SIZE), return_attention=True)
        self.assertEqual(len(out), 4)
        self.assertEqual(len(out[3]["attention"]), 2)
        self.assertEqual(out[3]["sequence_weights"].shape, (1, 4))

    def test_ablation_switches(self):
        for kw in ({"use_skips": False}, {"use_se": False}, {"depth": 0}):
            with self.subTest(**kw):
                m = HybridTumorModel(img_size=SIZE, patch_size=4, embed_dim=32,
                                     depth=kw.pop("depth", 2), num_heads=2, **kw)
                self.assertEqual(m(torch.randn(1, 4, *SIZE))[1].shape, (1, 3, *SIZE))

    def test_wrong_input_size_raises(self):
        with self.assertRaises(ValueError):
            small_model()(torch.randn(1, 4, 48, 48, 48))

    def test_mim_loss_only_on_masked_patches(self):
        m = small_model()
        x = torch.randn(1, 4, *SIZE)
        loss = m.forward_mim(x, mask_ratio=0.75)
        self.assertTrue(torch.isfinite(loss))
        loss.backward()
        self.assertGreater(m.mask_token.grad.abs().sum().item(), 0)
        mask = m.random_token_mask(1, 0.75, x.device)
        self.assertEqual(int(mask.sum()), round(m.num_patches * 0.75))
        self.assertEqual(m._patchify(x).shape, (1, m.num_patches, 4 * 8 ** 3))

    def test_patchify_matches_mask_layout(self):
        """Token i of _patchify must cover the voxels that mask token i zeroes."""
        m = small_model()
        mask = torch.zeros(1, m.num_patches)
        mask[0, 5] = 1
        vox = m._token_mask_to_voxels(mask)
        patches = m._patchify(vox.expand(1, 4, *SIZE))
        self.assertTrue(torch.all(patches[0, 5] == 1))
        self.assertEqual(patches.sum().item(), patches[0, 5].numel())

    def test_mc_dropout_keeps_batchnorm_frozen(self):
        m = small_model()
        bn = m.stem1[1]
        before = bn.running_mean.clone()
        mean, var, cls_prob = mc_dropout_predict(m, torch.randn(1, 4, *SIZE), num_samples=4)
        self.assertTrue(torch.equal(before, bn.running_mean))
        self.assertGreater(var.max().item(), 0)
        enable_mc_dropout(m)
        self.assertFalse(bn.training)

    def test_overfits_one_batch(self):
        m = small_model(dropout=0.0, decoder_dropout=0.0)
        x = torch.randn(2, 4, *SIZE)
        seg = torch.zeros(2, 3, *SIZE)
        seg[:, :, 8:20, 8:20, 8:20] = 1
        label = torch.tensor([1, 1])
        grade = torch.tensor([0, -1])
        loss_fn = MultiTaskLoss()
        opt = torch.optim.Adam(m.parameters(), lr=3e-3)
        losses = []
        for _ in range(25):
            opt.zero_grad()
            loss, _ = loss_fn(m(x), seg=seg, label=label, grade=grade)
            loss.backward()
            opt.step()
            losses.append(loss.item())
        self.assertLess(losses[-1], losses[0] * 0.6)


class TestLosses(unittest.TestCase):
    def test_tversky_beta_penalizes_false_negatives(self):
        target = torch.zeros(1, 1, 10, 10, 10)
        target[..., :5, :, :] = 1
        miss = torch.full_like(target, -10.0)               # predicts nothing: all FN
        over = torch.full_like(target, 10.0)                # predicts everything: all FP
        tl = TverskyLoss(alpha=0.3, beta=0.7)
        self.assertGreater(tl(miss, target).item(), tl(over, target).item())

    def test_unknown_grade_is_ignored(self):
        loss_fn = MultiTaskLoss()
        out = (torch.randn(2, 2), torch.randn(2, 3, 4, 4, 4), torch.randn(2, 2))
        _, terms = loss_fn(out, seg=torch.zeros(2, 3, 4, 4, 4), label=torch.tensor([0, 1]), grade=torch.tensor([-1, -1]))
        self.assertEqual(terms["grade"].item(), 0.0)

    def test_segmentation_only_baseline(self):
        loss_fn = MultiTaskLoss()
        out = (None, torch.randn(1, 3, 4, 4, 4), None)
        total, terms = loss_fn(out, seg=torch.zeros(1, 3, 4, 4, 4))
        self.assertTrue(torch.isfinite(total))


class TestMetrics(unittest.TestCase):
    def test_dice(self):
        a = np.zeros((10, 10, 10), bool)
        a[2:6, 2:6, 2:6] = True
        self.assertAlmostEqual(dice(a, a), 1.0)
        self.assertEqual(dice(np.zeros_like(a), np.zeros_like(a)), 1.0)
        self.assertEqual(dice(a, np.zeros_like(a)), 0.0)

    def test_hd95(self):
        a = np.zeros((20, 20, 20), bool)
        a[5:10, 5:10, 5:10] = True
        b = np.roll(a, 3, axis=0)
        self.assertEqual(hd95(a, a), 0.0)
        self.assertAlmostEqual(hd95(a, b), 3.0, places=5)
        self.assertEqual(hd95(a, np.zeros_like(a)), HD95_EMPTY_PENALTY)

    def test_ece(self):
        rng = np.random.default_rng(0)
        p = rng.uniform(size=50_000)
        y = rng.uniform(size=p.size) < p        # perfectly calibrated
        self.assertLess(expected_calibration_error(p, y), 0.02)
        self.assertGreater(expected_calibration_error(np.full(1000, 0.9), np.zeros(1000)), 0.8)


class TestBraTSData(unittest.TestCase):
    def test_label_regions(self):
        seg = np.array([0, 1, 2, 4])
        r = brats_label_to_regions(seg)
        np.testing.assert_array_equal(r[0], [0, 1, 1, 1])  # WT
        np.testing.assert_array_equal(r[1], [0, 1, 0, 1])  # TC
        np.testing.assert_array_equal(r[2], [0, 0, 0, 1])  # ET

    def test_zscore_ignores_background(self):
        v = np.zeros((4, 4, 4), np.float32)
        v[1:3, 1:3, 1:3] = [[[1, 2], [3, 4]], [[5, 6], [7, 8]]]
        z = zscore_nonzero(v)
        self.assertEqual(z[0, 0, 0], 0)
        self.assertAlmostEqual(float(z[v != 0].mean()), 0.0, places=5)

    def test_splits_are_disjoint(self):
        rows = [{"case_id": f"c{i}", "split": s, "grade": ""}
                for i, s in enumerate(["0", "1", "2", "test", "0", "1"])]
        train = {c for c, _ in cases_for(rows, "train", 0)}
        val = {c for c, _ in cases_for(rows, "val", 0)}
        test = {c for c, _ in cases_for(rows, "test", 0)}
        self.assertFalse(train & val or train & test or val & test)
        self.assertEqual(len(train | val | test), len(rows))

    def test_datasets_from_cache(self):
        with tempfile.TemporaryDirectory() as d:
            img = np.random.randn(4, 40, 36, 30).astype(np.float16)
            seg = np.zeros((3, 40, 36, 30), np.uint8)
            seg[:, 10:20, 10:20, 10:20] = 1
            np.save(os.path.join(d, "c0_img.npy"), img)
            np.save(os.path.join(d, "c0_seg.npy"), seg)
            ds = BraTSTrainDataset(d, [("c0", 1)], crop_size=SIZE, fg_prob=1.0, seed=0)
            item = ds[0]
            self.assertEqual(item["image"].shape, (4, *SIZE))
            self.assertEqual(item["seg"].shape, (3, *SIZE))
            self.assertEqual(int(item["label"]), 1)
            self.assertEqual(int(item["grade"]), 1)
            full = BraTSEvalDataset(d, [("c0", -1)])[0]
            self.assertEqual(full["image"].shape, (4, 40, 36, 30))


class TestLightningModule(unittest.TestCase):
    def test_training_steps(self):
        try:
            from lightning_module import TumorResearchModule
        except ImportError as e:
            self.skipTest(f"pytorch_lightning missing: {e}")
        cfg = {
            "data": {"crop_size": list(SIZE)},
            "model": {"name": "hybrid", "kwargs": {"patch_size": 4, "embed_dim": 32, "depth": 1, "num_heads": 2}},
            "loss": {},
            "train": {"lr": 1e-4, "epochs": 2, "mask_ratio": 0.5},
        }
        batch = {"image": torch.randn(1, 4, *SIZE), "seg": torch.zeros(1, 3, *SIZE),
                 "label": torch.tensor([0]), "grade": torch.tensor([-1])}
        for mode in ("pretrain", "finetune"):
            with self.subTest(mode=mode):
                module = TumorResearchModule(cfg, mode=mode)
                module.log = lambda *a, **k: None
                loss = module.training_step(batch, 0)
                self.assertTrue(torch.isfinite(loss))


if __name__ == "__main__":
    unittest.main()
