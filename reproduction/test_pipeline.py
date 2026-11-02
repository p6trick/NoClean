"""Numerical checks for attack/rectification direction and data isolation."""
import unittest
import tempfile
from pathlib import Path
from types import SimpleNamespace

import numpy as np
from PIL import Image
import torch
from torch import nn

from models import Denoiser
from run import attack, rectify, split_entries, attach_saved_paths, prepare_pairs


class ThresholdClassifier(nn.Module):
    def forward(self, x):
        value = x.flatten(1).mean(1) - 0.5
        return torch.stack((-value, value), dim=1) * 10


class PipelineTests(unittest.TestCase):
    def setUp(self):
        torch.set_num_threads(1)
        self.model = ThresholdClassifier()
        self.args = SimpleNamespace(
            attack='fgsm', attack_epsilon=0.3, attack_alpha=0.1, attack_steps=5,
            gradient_loss='logits-ce', rectify_epsilon=0.4, rectify_alpha=0.1,
            rectify_steps=10, rectify_stop='correct')

    def test_attack_changes_class_and_stays_within_budget(self):
        clean = torch.full((1, 1, 4, 4), 0.7)
        corrupted = attack(self.model, clean, torch.tensor([1]), self.args)
        self.assertEqual(self.model(corrupted).argmax(1).item(), 0)
        self.assertLessEqual((corrupted - clean).abs().max().item(), 0.300001)
        self.assertTrue(torch.equal(clean, torch.full_like(clean, 0.7)))

    def test_rectification_restores_label_and_preserves_already_correct(self):
        source = torch.stack((torch.full((1, 4, 4), 0.3), torch.full((1, 4, 4), 0.8)))
        target, steps = rectify(self.model, source, torch.tensor([1, 1]), self.args)
        self.assertEqual(self.model(target).argmax(1).tolist(), [1, 1])
        self.assertEqual(steps[1].item(), 0)
        self.assertTrue(torch.equal(source[1], target[1]))
        self.assertLessEqual((target - source).abs().max().item(), 0.400001)

    def test_failed_rectification_stops_at_limit(self):
        self.args.rectify_epsilon = 0.01
        self.args.rectify_steps = 3
        source = torch.full((1, 1, 4, 4), 0.1)
        target, steps = rectify(self.model, source, torch.tensor([1]), self.args)
        self.assertEqual(steps.item(), 3)
        self.assertEqual(self.model(target).argmax(1).item(), 0)
        self.assertLessEqual((target - source).abs().max().item(), 0.010001)

    def test_split_is_disjoint_and_reproducible(self):
        entries = [{'id': str(i)} for i in range(10000)]
        args = SimpleNamespace(seed=17, train_size=100, validation_size=20, eval_size=100)
        a, b = split_entries(entries, args), split_entries(entries, args)
        self.assertEqual(a, b)
        fit_ids = {int(e['id']) for e in a['train'] + a['validation']}
        eval_ids = {int(e['id']) for e in a['evaluation']}
        self.assertTrue(all(i < 5000 for i in fit_ids))
        self.assertTrue(all(i >= 5000 for i in eval_ids))
        self.assertFalse(fit_ids & eval_ids)

    def test_archived_unet_accepts_28px_and_receives_gradients(self):
        model = Denoiser()
        source = torch.rand(2, 3, 28, 28)
        result = model(source)
        self.assertEqual(result.shape, source.shape)
        result.square().mean().backward()
        self.assertTrue(all(p.grad is not None and torch.isfinite(p.grad).all()
                            for p in model.parameters()))

    def test_saved_split_uses_last_half_for_fitting(self):
        entries = [{'id': str(i)} for i in range(10000)]
        args = SimpleNamespace(seed=17, train_size=100, validation_size=20,
                               eval_size=100, fit_half='last')
        groups = split_entries(entries, args)
        self.assertTrue(all(int(e['id']) >= 5000 for e in groups['train'] + groups['validation']))
        self.assertTrue(all(int(e['id']) < 5000 for e in groups['evaluation']))

    def test_saved_pairs_use_exact_files_without_attack_or_clean_target(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            for name, value in [('attacks', 51), ('targets', 102)]:
                (root / name).mkdir()
                Image.fromarray(np.full((28, 28), value, np.uint8)).save(root / name / 'image_5000.png')
            args = SimpleNamespace(input_mode='saved', batch_size=1,
                                   saved_train_attacks=str(root / 'attacks'),
                                   saved_eval_attacks=str(root / 'attacks'),
                                   saved_targets=str(root / 'targets'))
            groups = {'train': [{'id': 'image_5000.png', 'label': 3, 'path': 'nonexistent-clean.png'}]}
            attach_saved_paths(groups, args)
            pairs = prepare_pairs(None, groups['train'], args, torch.device('cpu'), 'test')
            self.assertTrue(torch.allclose(pairs['source'], torch.full((1, 1, 28, 28), 0.2)))
            self.assertTrue(torch.allclose(pairs['target'], torch.full((1, 1, 28, 28), 0.4)))
            self.assertEqual(pairs['rectify_iterations'].item(), -1)
            (root / 'targets/image_5000.png').unlink()
            with self.assertRaises(FileNotFoundError):
                attach_saved_paths(groups, args)


if __name__ == '__main__':
    unittest.main()
