from pathlib import Path
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import patch
import io

import numpy as np
from PIL import Image, ImageFile
import torch

from livdet import canonical_id, decode_image, fit, make_scheduler, PairDataset, scan_folders, stratified_take
from models import Denoiser


class LivDetTests(unittest.TestCase):
    def setUp(self):
        torch.set_num_threads(1)

    def test_identity_normalization_preserves_material_and_label(self):
        self.assertEqual(canonical_id('013_0_3.png'), 'alive_013_0_3.png')
        self.assertEqual(canonical_id('Alive_013_0_3.png'), 'alive_013_0_3.png')
        self.assertNotEqual(canonical_id('rtv_013_0_3.png'), canonical_id('eco_013_0_3.png'))
        with self.assertRaises(ValueError):
            canonical_id('unknown_013.png')

    def test_duplicate_normalized_ids_fail(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            for name in ['a', 'b']:
                (root / name).mkdir()
            (root / 'a/013_0_3.png').touch()
            (root / 'b/alive_013_0_3.png').touch()
            with self.assertRaisesRegex(ValueError, 'Duplicate'):
                scan_folders(root, ['a', 'b'])

    def test_stratification_is_disjoint_and_reproducible(self):
        entries = [{'id': str(i), 'label': int(i < 500)} for i in range(1250)]
        validation, remainder = stratified_take(entries, 125, 2026)
        train, _ = stratified_take(remainder, 1125, 2027)
        self.assertEqual(sum(e['label'] for e in validation), 50)
        self.assertEqual(sum(e['label'] for e in train), 450)
        self.assertFalse({e['id'] for e in validation} & {e['id'] for e in train})
        self.assertEqual(validation, stratified_take(entries, 125, 2026)[0])

    def test_paired_crop_uses_identical_coordinates_without_clean_images(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            array = np.arange(581 * 581, dtype=np.int64).reshape(581, 581) % 200 + 20
            Image.fromarray(array.astype('uint8')).save(root / 'source.png')
            Image.fromarray((255 - array).astype('uint8')).save(root / 'target.png')
            dataset = PairDataset([{'attack_path': str(root / 'source.png'),
                                    'target_path': str(root / 'target.png')}], 128, True)
            for _ in range(3):
                source, target = dataset[0]
                self.assertEqual(source.shape, (3, 128, 128))
                self.assertTrue(torch.allclose(source + target, torch.ones_like(source), atol=1e-6))

    def test_fingerprint_unet_handles_full_odd_sized_image(self):
        model = Denoiser(first_pool=2).eval()
        with torch.no_grad():
            result = model(torch.zeros(1, 3, 581, 581))
        self.assertEqual(result.shape, (1, 3, 581, 581))
        self.assertTrue(torch.isfinite(result).all())

    def test_training_checkpoint_can_be_reloaded_with_weights_only(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            Image.fromarray(np.full((32, 32), 100, np.uint8)).save(root / 'source.png')
            Image.fromarray(np.full((32, 32), 101, np.uint8)).save(root / 'target.png')
            entry = {'attack_path': str(root / 'source.png'), 'target_path': str(root / 'target.png')}
            args = SimpleNamespace(seed=1, learning_rate=.001, crop_size=32, batch_size=1,
                                   epochs=1, pm_root=Path('/unused'), output=root)
            model, epoch = fit({'train': [entry], 'validation': [entry]}, args,
                               torch.device('cpu'), root, lambda _: None)
            saved = torch.load(root / 'denoiser.pt', weights_only=True)
            self.assertEqual(epoch, 1)
            self.assertEqual(saved['model_configuration'], {'first_pool': 2})
            self.assertEqual(saved['arguments']['pm_root'], '/unused')
            self.assertFalse(model.training)

    def test_truncated_image_retry_retains_sample_and_restores_pillow_setting(self):
        with tempfile.TemporaryDirectory() as temporary:
            stream = io.BytesIO()
            Image.fromarray(np.full((32, 32), 100, np.uint8)).save(stream, format='JPEG')
            path = Path(temporary) / 'incomplete.jpg'
            path.write_bytes(stream.getvalue()[:-2])  # Missing JPEG end marker.
            with patch.object(ImageFile, 'LOAD_TRUNCATED_IMAGES', False):
                with self.assertWarnsRegex(RuntimeWarning, 'sample retained'):
                    result = decode_image(path, 'L')
                self.assertEqual(result.size, (32, 32))
                self.assertFalse(ImageFile.LOAD_TRUNCATED_IMAGES)

    def test_unrecoverable_image_fails_with_path_without_skipping(self):
        with tempfile.TemporaryDirectory() as temporary:
            path = Path(temporary) / 'invalid.png'
            path.write_bytes(b'not an image')
            with patch.object(ImageFile, 'LOAD_TRUNCATED_IMAGES', False):
                with self.assertRaisesRegex(OSError, 'invalid.png'):
                    decode_image(path, 'L')
                self.assertFalse(ImageFile.LOAD_TRUNCATED_IMAGES)

    def test_plateau_scheduler_reduces_learning_rate_after_validation_stalls(self):
        optimizer = torch.optim.Adam([torch.nn.Parameter(torch.zeros(1))], lr=.001)
        scheduler = make_scheduler(optimizer, SimpleNamespace(
            scheduler='plateau', scheduler_patience=1, epochs=300))
        scheduler.step(1.0)
        scheduler.step(1.0)
        self.assertEqual(optimizer.param_groups[0]['lr'], .001)
        scheduler.step(1.0)
        self.assertEqual(optimizer.param_groups[0]['lr'], .0005)

    def test_scheduler_keeps_previous_default_and_legacy_patience(self):
        optimizer = torch.optim.Adam([torch.nn.Parameter(torch.zeros(1))], lr=.001)
        self.assertIsNone(make_scheduler(optimizer, SimpleNamespace(epochs=30)))
        scheduler = make_scheduler(optimizer, SimpleNamespace(scheduler='plateau', epochs=300))
        self.assertEqual(scheduler.patience, 75)

    def test_training_saves_selected_pooling_topology(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            Image.fromarray(np.full((32, 32), 100, np.uint8)).save(root / 'source.png')
            Image.fromarray(np.full((32, 32), 101, np.uint8)).save(root / 'target.png')
            entry = {'attack_path': str(root / 'source.png'), 'target_path': str(root / 'target.png')}
            args = SimpleNamespace(seed=1, learning_rate=.001, crop_size=32, batch_size=1,
                                   epochs=1, first_pool=1)
            model, _ = fit({'train': [entry], 'validation': []}, args,
                           torch.device('cpu'), root, lambda _: None)
            saved = torch.load(root / 'denoiser.pt', weights_only=True)
            self.assertEqual(saved['model_configuration']['first_pool'], 1)
            self.assertEqual(model._block1[-1].kernel_size, 1)


if __name__ == '__main__':
    unittest.main()
