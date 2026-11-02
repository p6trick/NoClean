import contextlib
import io
from pathlib import Path
from types import SimpleNamespace
import unittest
from unittest.mock import patch

from run import parse_args, split_entries
from suite import load_experiments, summary_row


class SuiteTests(unittest.TestCase):
    def test_experiment_manifests_do_not_share_mutable_entries(self):
        entries = [{'id': str(i)} for i in range(10000)]
        args = SimpleNamespace(seed=2026, fit_half='last', train_size=10,
                               validation_size=5, eval_size=10)
        a = split_entries(entries, args)
        b = split_entries(entries, args)
        a['train'][0]['attack_path'] = 'pgd/file.png'
        b['train'][0]['attack_path'] = 'cw/file.png'
        self.assertEqual(a['train'][0]['attack_path'], 'pgd/file.png')
        self.assertFalse(any('attack_path' in e for e in entries))

    def test_all_archived_attack_labels_are_supported(self):
        for name in ['fgsm', 'pgd', 'cw', 'ba', 'bim', 'aa']:
            with self.subTest(attack=name), patch('sys.argv', [
                    'run.py', '--output', 'unused', '--input-mode', 'saved',
                    '--attack', name, '--saved-train-attacks', 'train',
                    '--saved-eval-attacks', 'evaluation', '--saved-targets', 'targets']):
                self.assertEqual(parse_args().attack, name)

    def test_unimplemented_generators_fail_instead_of_using_pgd(self):
        for name in ['cw', 'ba', 'bim', 'aa']:
            with self.subTest(attack=name), patch('sys.argv', [
                    'run.py', '--output', 'unused', '--attack', name]), \
                    contextlib.redirect_stderr(io.StringIO()):
                with self.assertRaises(SystemExit) as error:
                    parse_args()
                self.assertEqual(error.exception.code, 2)

    def test_summary_uses_matching_paper_attack_and_signed_delta(self):
        spec = load_experiments()['cw']
        metrics = {'count': 5000, 'selected_epoch': 10, 'accuracy_percent': {
            'clean': 98.58, 'attacked': 2.0, 'defended_attacked': 97.0, 'defended_clean': 95.0}}
        row = summary_row('cw', spec, metrics, Path('cw-run'))
        self.assertEqual(row['attack'], 'CW')
        self.assertEqual(row['paper_defended'], 97.79)
        self.assertEqual(row['defended_delta_pp'], -0.79)


if __name__ == '__main__':
    unittest.main()
