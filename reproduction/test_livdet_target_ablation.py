"""Check target routing and deterministic paired LivDet fitting."""
import json
from pathlib import Path
import tempfile
from types import SimpleNamespace
import unittest

import numpy as np
from PIL import Image
import torch

from livdet import PairDataset, fit
from mnist_target_ablation import seed_all, state_hash
from models import Denoiser


class LivDetTargetTests(unittest.TestCase):
    def setUp(self):
        torch.set_num_threads(1)

    def test_clean_target_reads_clean_path(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            Image.fromarray(np.full((581, 581), 64, dtype='uint8')).save(root / 'clean.png')
            Image.fromarray(np.full((581, 581), 192, dtype='uint8')).save(root / 'attack.png')
            entry = {'path': str(root / 'clean.png'), 'attack_path': str(root / 'attack.png'),
                     'target_path': str(root / 'missing_rectified.png')}
            source, target = PairDataset([entry], 32, True, target_key='path')[0]
            self.assertAlmostEqual(source.mean().item(), 192/255, places=5)
            self.assertAlmostEqual(target.mean().item(), 64/255, places=5)
            with self.assertRaisesRegex(OSError, 'missing_rectified'):
                PairDataset([entry], 32, True)[0]

    def test_identical_targets_pair_training_and_checkpoint_selection(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            array = (np.arange(581*581).reshape(581, 581) % 256).astype('uint8')
            Image.fromarray(array).save(root / 'image.png')
            path = str(root / 'image.png')
            entries = [{'path': path, 'attack_path': path, 'target_path': path} for _ in range(2)]
            seed_all(2026)
            initial = {key: value.clone() for key, value in Denoiser(first_pool=2).state_dict().items()}
            results = []
            for target in ['clean', 'rectified']:
                args = SimpleNamespace(seed=2026, first_pool=2, learning_rate=.001, scheduler='none',
                    crop_size=32, batch_size=1, epochs=2, training_target=target, keep_checkpoint_variants=True)
                out = root / target
                out.mkdir()
                seed_all(2026)
                model, epoch = fit({'train': entries, 'validation': entries}, args, torch.device('cpu'),
                                    out, lambda line: None, initial_state=initial)
                history = json.loads((out / 'training_history.json').read_text())
                self.assertEqual(epoch, min(history, key=lambda row: row['validation_mse'])['epoch'])
                self.assertTrue((out / 'best_validation_denoiser.pt').is_file())
                self.assertEqual(torch.load(out / 'last_denoiser.pt', weights_only=True)['epoch'], 2)
                results.append(state_hash(model.state_dict()))
            self.assertEqual(results[0], results[1])


if __name__ == '__main__':
    unittest.main()
