"""Check paired initialization/order and fixed-epoch checkpoint selection."""
from pathlib import Path
import tempfile
import json
from types import SimpleNamespace
import unittest

import torch

from models import Denoiser
from mnist_target_ablation import seed_all, state_hash
from run import loader_for_pairs, train_denoiser


class PairedTrainingTests(unittest.TestCase):
    def setUp(self):
        torch.set_num_threads(1)

    def test_pair_order_independent_of_target(self):
        args = SimpleNamespace(seed=2026, batch_size=3)
        source = torch.arange(12).float().view(12, 1)
        left = {'source': source, 'target': source + 10}
        right = {'source': source, 'target': source + 20}
        for _ in range(2):
            a = list(loader_for_pairs(left, args, True))
            b = list(loader_for_pairs(right, args, True))
            self.assertTrue(torch.equal(torch.cat([v[0] for v in a]), torch.cat([v[0] for v in b])))

    def test_same_targets_reproduce_same_training_and_final_epoch(self):
        args = SimpleNamespace(seed=7, batch_size=2, epochs=2, learning_rate=.001,
                               checkpoint_selection='last')
        seed_all(7)
        initial = {k: v.clone() for k, v in Denoiser().state_dict().items()}
        initial_hash = state_hash(initial)
        pairs = {'source': torch.rand(4, 1, 28, 28), 'target': torch.rand(4, 1, 28, 28)}
        with tempfile.TemporaryDirectory() as directory:
            result = []
            for condition in ['clean', 'rectified']:
                out = Path(directory) / condition
                out.mkdir()
                seed_all(7)
                model, epoch = train_denoiser(pairs, pairs, args, torch.device('cpu'), out,
                                              initial_state=initial)
                self.assertEqual(epoch, 2)
                result.append(state_hash(model.state_dict()))
            self.assertEqual(result[0], result[1])
            self.assertEqual(state_hash(initial), initial_hash)
            self.assertNotEqual(result[0], initial_hash)

    def test_validation_selection_and_last_checkpoint_are_separate(self):
        args = SimpleNamespace(seed=7, batch_size=2, epochs=2, learning_rate=.001,
                               checkpoint_selection='validation-mse', keep_checkpoint_variants=True)
        seed_all(7)
        initial = {k: v.clone() for k, v in Denoiser().state_dict().items()}
        pairs = {'source': torch.rand(4, 1, 28, 28), 'target': torch.rand(4, 1, 28, 28)}
        with tempfile.TemporaryDirectory() as directory:
            out = Path(directory)
            seed_all(7)
            _, epoch = train_denoiser(pairs, pairs, args, torch.device('cpu'), out, initial_state=initial)
            history = json.loads((out / 'training_history.json').read_text())
            self.assertEqual(epoch, min(history, key=lambda row: row['validation_mse'])['epoch'])
            selected = torch.load(out / 'denoiser.pt', weights_only=True)
            best = torch.load(out / 'best_validation_denoiser.pt', weights_only=True)
            last = torch.load(out / 'last_denoiser.pt', weights_only=True)
            self.assertEqual(state_hash(selected['state_dict']), state_hash(best['state_dict']))
            self.assertEqual(last['epoch'], 2)


if __name__ == '__main__':
    unittest.main()
