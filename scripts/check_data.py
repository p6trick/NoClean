"""Check original clean/attack files referenced by the downloaded evaluation manifests."""
import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'reproduction'))
from paths import CHECKPOINT_ROOT, PM_ROOT, loads_json


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--datasets', nargs='+', choices=['mnist', 'livdet'], default=['mnist', 'livdet'])
    parser.add_argument('--include-training', action='store_true')
    parser.add_argument('--include-targets', action='store_true')
    args = parser.parse_args()
    paths = set()
    for dataset in args.datasets:
        for attack in ['fgsm', 'pgd', 'cw', 'ba', 'bim', 'aa']:
            manifest = CHECKPOINT_ROOT / dataset / attack / 'source_data_manifest.json'
            data = loads_json(manifest.read_text())
            if isinstance(data, dict):
                entries = data['evaluation']
                if args.include_training:
                    entries = entries + data['train'] + data['validation']
            else:
                entries = data
                if args.include_training:
                    raise SystemExit('LivDet bundle manifests preserve evaluation IDs only; use the archive registry for training.')
            for entry in entries:
                for key in ['path', 'attack_path'] + (['target_path'] if args.include_targets else []):
                    paths.add(Path(entry[key]))
    missing = [str(path) for path in sorted(paths) if not path.is_file()]
    print('PM_ROOT', PM_ROOT)
    print('UNIQUE_INPUT_FILES', len(paths), 'MISSING', len(missing))
    if missing:
        print('\n'.join(missing[:20]))
        print('Provide the matching archived data; checkpoints alone are insufficient.')
        raise SystemExit(1)
    print('PASS all referenced files exist; content hashes were not checked')


if __name__ == '__main__':
    main()
