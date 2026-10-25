"""The deployed model must only be replaced by a finished, more accurate candidate.

    python -m unittest discover -s tests -p test_safe_training.py -v
"""
import os
import sys
import tempfile
import unittest
from unittest import mock

import torch

sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), 'src'))
import train as T  # noqa: E402


class TestPromoteIfBetter(unittest.TestCase):
    def setUp(self):
        self.dir = tempfile.TemporaryDirectory()
        self.deploy = os.path.join(self.dir.name, 'best_model.pth')
        self.cand = os.path.join(self.dir.name, 'best_model.candidate.pth')
        torch.save({'tag': 'old'}, self.deploy)
        self.logs = []

    def tearDown(self):
        self.dir.cleanup()

    def run_promote(self, finished, acc_new, acc_old):
        cand = {'tag': 'new'}
        if finished:
            cand['temperature'] = 1.2
        torch.save(cand, self.cand)
        accs = {self.cand: acc_new, self.deploy: acc_old}
        with mock.patch.object(T, 'val_accuracy', side_effect=lambda p, *a: accs[p]):
            return T.promote_if_better(self.cand, self.deploy, None, 'cpu', self.logs.append)

    def deployed_tag(self):
        return torch.load(self.deploy, weights_only=False)['tag']

    def backups(self):
        return [f for f in os.listdir(self.dir.name) if '.backup-' in f]

    def test_unfinished_candidate_not_promoted(self):
        self.assertFalse(self.run_promote(False, 0.99, 0.50))
        self.assertEqual(self.deployed_tag(), 'old')

    def test_worse_candidate_not_promoted(self):
        self.assertFalse(self.run_promote(True, 0.83, 0.99))
        self.assertEqual(self.deployed_tag(), 'old')
        self.assertTrue(os.path.exists(self.cand))       # kept for inspection

    def test_equal_candidate_not_promoted(self):
        self.assertFalse(self.run_promote(True, 0.90, 0.90))
        self.assertEqual(self.deployed_tag(), 'old')

    def test_better_candidate_promoted_with_backup(self):
        self.assertTrue(self.run_promote(True, 0.97, 0.83))
        self.assertEqual(self.deployed_tag(), 'new')
        self.assertFalse(os.path.exists(self.cand))
        self.assertEqual(len(self.backups()), 1)
        backup = os.path.join(self.dir.name, self.backups()[0])
        self.assertEqual(torch.load(backup, weights_only=False)['tag'], 'old')

    def test_unreadable_deployed_model_replaced(self):
        self.assertTrue(self.run_promote(True, 0.80, None))
        self.assertEqual(self.deployed_tag(), 'new')


class TestValAccuracy(unittest.TestCase):
    def test_real_checkpoint_scored_and_bad_file_is_none(self):
        torch.manual_seed(0)
        from PIL import Image
        m = T.BrainTumorClassifier(input_channels=4, num_classes=2)
        with tempfile.TemporaryDirectory() as d:
            files = []
            for i in range(2):
                p = os.path.join(d, f'{i}.png')
                Image.new('L', (40, 40), color=60 * i).save(p)
                files.append((p, i))
            good, bad = os.path.join(d, 'g.pth'), os.path.join(d, 'b.pth')
            torch.save({'model_state_dict': m.state_dict(), 'img_size': (16, 32, 32)}, good)
            torch.save({'model_state_dict': {'x': torch.zeros(1)}}, bad)
            acc = T.val_accuracy(good, files, torch.device('cpu'))
            self.assertIn(acc, (0.0, 0.5, 1.0))
            self.assertIsNone(T.val_accuracy(bad, files, torch.device('cpu')))


if __name__ == '__main__':
    unittest.main()
