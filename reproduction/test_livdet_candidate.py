from pathlib import Path
import tempfile
import unittest

from audit_livdet_candidate import index_candidate, label_from_name, summarize


class CandidateAuditTests(unittest.TestCase):
    def test_candidate_labels_are_explicit(self):
        self.assertEqual(label_from_name('023_2_9.png'), (1, 'Alive'))
        self.assertEqual(label_from_name('LiquidEcoflex_250.png'), (0, 'liquidecoflex'))
        with self.assertRaises(ValueError):
            label_from_name('unknown_1.png')

    def test_index_does_not_silently_skip_unknown_labels(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            (root / 'unknown_1.png').touch()
            with self.assertRaises(ValueError):
                index_candidate(root)

    def test_accuracy_and_balanced_accuracy_remain_separate(self):
        entries = [{'label': 0, 'material': 'latex'} for _ in range(3)] + [
            {'label': 1, 'material': 'Alive'} for _ in range(2)]
        result = summarize(entries, [0] * 5)
        self.assertEqual(result['accuracy_percent'], 60)
        self.assertEqual(result['balanced_accuracy_percent'], 50)
        self.assertEqual(result['confusion_matrix'], [[3, 0], [2, 0]])


if __name__ == '__main__':
    unittest.main()
