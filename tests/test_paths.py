"""Check manifest relocation without loading models or requiring archive access."""
import json
import os
from pathlib import Path
import subprocess
import sys
import unittest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'reproduction'))
from paths import loads_json, PM_ROOT


class RelocationTests(unittest.TestCase):
    def test_nested_manifest_values_and_keys(self):
        data = {'evaluation': [{'path': '${N2N_PM_ROOT}/test/image.png'}],
                '${N2N_PM_ROOT}/key': {'label': 1}}
        resolved = loads_json(json.dumps(data))
        self.assertEqual(resolved['evaluation'][0]['path'], str(PM_ROOT / 'test/image.png'))
        self.assertEqual(resolved[str(PM_ROOT / 'key')], {'label': 1})

    def test_provenance_placeholders_preserved(self):
        self.assertEqual(loads_json('"${N2N_ORIGINAL_WORKSPACE}/run"'),
                         '${N2N_ORIGINAL_WORKSPACE}/run')

    def test_environment_override_and_asset_defaults(self):
        env = dict(os.environ, N2N_ASSET_ROOT='/tmp/n2n_test_assets',
                   N2N_PM_ROOT='/tmp/n2n_test_archive',
                   PYTHONPATH=str(Path(__file__).resolve().parents[1] / 'reproduction'))
        env.pop('N2N_CHECKPOINT_ROOT', None)
        env.pop('N2N_AT_MODEL_ROOT', None)
        source = 'from paths import *; print(CHECKPOINT_ROOT); print(AT_MODEL_ROOT); print(PM_ROOT)'
        result = subprocess.check_output([sys.executable, '-c', source], env=env, text=True)
        self.assertEqual(result.splitlines(), ['/tmp/n2n_test_assets/checkpoints',
                         '/tmp/n2n_test_assets/at_models', '/tmp/n2n_test_archive'])


if __name__ == '__main__':
    unittest.main()
