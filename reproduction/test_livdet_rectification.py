from pathlib import Path
import tempfile
from types import SimpleNamespace
import unittest

import numpy as np
from PIL import Image
import torch

from livdet import read_tensor, regenerate_targets, scan_folders, verify_cached_targets


class ThresholdClassifier(torch.nn.Module):
    def forward(self, images):
        value = images.flatten(1).mean(1) - .5
        return torch.stack((-value, value), 1) * 10


class RectificationCacheTests(unittest.TestCase):
    def setUp(self):
        torch.set_num_threads(1)

    def test_cache_respects_budget_stopping_source_bytes_and_reuse_checks(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            records = []
            for identity, pixel, label in [('eco_001_0_0.png', 25, 0),
                                            ('alive_001_0_0.png', 125, 1),
                                            ('alive_002_0_0.png', 25, 1)]:
                path = root / identity
                Image.fromarray(np.full((32, 32), pixel, np.uint8)).save(path)
                records.append({'id': identity, 'label': label, 'attack_path': str(path),
                                'path': 'clean_image_must_not_be_read.png'})
            originals = {e['id']: Path(e['attack_path']).read_bytes() for e in records}
            groups = {'train': records[:2], 'validation': [], 'evaluation': records[2:]}
            args = SimpleNamespace(rectify_epsilon=.03, rectify_alpha=2 / 255,
                                   rectify_steps=40, rectify_batch_size=2,
                                   rectify_stop='correct', gradient_loss='logits-ce')
            summary = regenerate_targets(ThresholdClassifier().eval(), groups, args,
                                         torch.device('cpu'), root, 'test-classifier-sha')
            self.assertEqual(summary['count'], 3)
            self.assertEqual(summary['zero_iterations'], 1)
            self.assertEqual(summary['unrectified'], 1)
            self.assertEqual(summary['reached_iteration_limit'], 1)
            self.assertLessEqual(summary['max_linf'], .030001)
            for entry in records:
                self.assertEqual(Path(entry['attack_path']).read_bytes(), originals[entry['id']])
                target = read_tensor(entry['target_path'])
                source = read_tensor(entry['attack_path'])
                self.assertLessEqual((target - source).abs().max().item(), .030001)
                self.assertEqual(read_tensor(entry['target_path'], rgb=True).shape, (3, 581, 581))
            self.assertTrue(torch.equal(read_tensor(records[0]['target_path']),
                                        read_tensor(records[0]['attack_path'])))
            saved = torch.load(records[1]['target_path'], weights_only=True)
            self.assertLess(saved['iterations'], 40)
            self.assertGreater(saved['iterations'], 0)
            mapping = scan_folders(root, ['rectified_targets'], tensor_targets=True)
            self.assertEqual(set(mapping), {e['id'] for e in records})
            verified = verify_cached_targets(groups, 'test-classifier-sha')
            self.assertEqual(verified['verified_count'], 3)
            self.assertEqual(verified['mean_evaluation_iterations'], 40)
            with self.assertRaisesRegex(ValueError, 'classifier mismatch'):
                verify_cached_targets(groups, 'different-classifier')
            Image.fromarray(np.full((32, 32), 26, np.uint8)).save(records[0]['attack_path'])
            with self.assertRaisesRegex(ValueError, 'does not match'):
                verify_cached_targets(groups, 'test-classifier-sha')

    def test_invalid_float_target_fails_instead_of_loading_model_data(self):
        with tempfile.TemporaryDirectory() as temporary:
            path = Path(temporary) / 'bad.pt'
            torch.save({'image': torch.zeros(1, 10, 10)}, path)
            with self.assertRaisesRegex(ValueError, 'Invalid float'):
                read_tensor(path)


if __name__ == '__main__':
    unittest.main()
