"""Check separately downloaded weights without importing a deep-learning framework."""
import argparse
import hashlib
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'reproduction'))
from paths import ASSET_ROOT, CHECKPOINT_ROOT, AT_MODEL_ROOT


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--verify-hashes', action='store_true')
    args = parser.parse_args()
    missing, checked = [], 0
    records = []
    for manifest, kind in [(CHECKPOINT_ROOT / 'manifest.json', 'n2n'), (AT_MODEL_ROOT / 'manifest.json', 'at')]:
        if not manifest.is_file():
            missing.append(str(manifest))
            continue
        data = json.loads(manifest.read_text())
        if kind == 'n2n':
            records.extend((CHECKPOINT_ROOT / r['copy'], r) for r in data['copied_files']
                           if Path(r['copy']).suffix in ['.pt', '.pth'])
        else:
            records.extend((AT_MODEL_ROOT / r['file'], r) for r in data['models'])
    for path, record in records:
        if not path.is_file() or path.stat().st_size != record['size_bytes']:
            missing.append(str(path))
            continue
        if args.verify_hashes:
            digest = hashlib.sha256()
            with path.open('rb') as stream:
                for block in iter(lambda: stream.read(4 * 1024 * 1024), b''):
                    digest.update(block)
            if digest.hexdigest() != record['sha256']:
                raise SystemExit(f'Hash mismatch: {path}')
        checked += 1
    print('ASSET_ROOT', ASSET_ROOT)
    print('CHECKED_WEIGHT_FILES', checked)
    if missing:
        print('Missing or wrong-size files:\n' + '\n'.join(missing))
        raise SystemExit(1)
    print('PASS', 'SHA256 verified' if args.verify_hashes else 'existence/size verified')


if __name__ == '__main__':
    main()
