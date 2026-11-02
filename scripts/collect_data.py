"""Copy reproduction inputs into a separate PM tree, retaining source files."""
import argparse
from collections import Counter
from concurrent.futures import ThreadPoolExecutor
import csv
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import sys
import time

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'reproduction'))
from paths import CHECKPOINT_ROOT, PM_ROOT, PROJECT_ROOT


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--source-root', type=Path, default=PM_ROOT)
    parser.add_argument('--checkpoint-root', type=Path, default=CHECKPOINT_ROOT)
    parser.add_argument('--destination', type=Path, required=True,
                        help='Data bundle root; copies are placed under its PM/ subfolder')
    parser.add_argument('--workers', type=int, default=4)
    parser.add_argument('--plan-only', action='store_true')
    args = parser.parse_args()
    if not 1 <= args.workers <= 16:
        parser.error('workers must be between 1 and 16')
    source = args.source_root.resolve()
    destination = args.destination.resolve()
    if source == destination or source in destination.parents or destination in source.parents:
        parser.error('Source and destination must be separate directory trees')
    files = {}

    def add(relative, role):
        relative = Path(relative)
        if relative.is_absolute() or '..' in relative.parts:
            raise ValueError(f'Unsafe relative path: {relative}')
        files.setdefault(str(relative), set()).add(role)

    attacks = ['fgsm', 'pgd', 'cw', 'ba', 'bim', 'aa']
    for attack in attacks:
        manifest = json.loads((args.checkpoint_root / 'mnist' / attack /
                               'source_data_manifest.json').read_text())
        for split, entries in manifest.items():
            for entry in entries:
                for key in ['path', 'attack_path', 'target_path']:
                    raw = entry[key]
                    if raw.startswith('${N2N_PM_ROOT}/'):
                        relative = raw[len('${N2N_PM_ROOT}/'):]
                    else:
                        relative = str(Path(raw).relative_to(source))
                    add(relative, f'mnist/{attack}/{split}/{key}')
    add('FGSM/image_names.txt', 'mnist/canonical_order')

    clean = ['PGD/data/2015/GreenBit/Alive_half', 'PGD/data/2015/GreenBit/Alive',
             'Self2Self/noise2noise/temporary/data/ECCV/Fake_all/fk/Fake',
             'Self2Self/noise2noise/temporary/data/ECCV/Fake_all/fk_val/Fake']
    specs = json.loads((PROJECT_ROOT / 'reproduction/livdet_experiments.json').read_text())
    directories = [(folder, 'livdet/clean') for folder in clean]
    directories += [(folder, f'livdet/{attack}/{kind}') for attack, spec in specs.items()
                    for kind in ['train', 'evaluation', 'targets'] for folder in spec[kind]]
    for folder, role in directories:
        directory = source / folder
        if not directory.is_dir():
            raise FileNotFoundError(directory)
        paths = sorted(directory.glob('*.png'))
        if not paths:
            raise ValueError(f'No PNG inputs: {directory}')
        for path in paths:
            add(path.relative_to(source), role)

    dip = json.loads((PROJECT_ROOT / 'reproduction/baseline_saved_dip_inventory.json').read_text())
    for attack, spec in dip.items():
        for folder in spec['folders']:
            directory = source / folder.removeprefix('${N2N_PM_ROOT}/')
            paths = sorted(directory.rglob('*.png'))
            if not paths:
                raise ValueError(f'No saved DIP outputs: {directory}')
            for path in paths:
                add(path.relative_to(source), f'livdet/{attack}/saved_dip')

    relative_paths = sorted(files)
    print('PLANNED_FILES', len(relative_paths), flush=True)
    def size_of(relative):
        return (source / relative).stat().st_size
    with ThreadPoolExecutor(args.workers) as executor:
        sizes = list(executor.map(size_of, relative_paths))
    required = sum(sizes)
    print('PLANNED_BYTES', required, flush=True)
    if args.plan_only:
        return
    import shutil
    destination.mkdir(parents=True, exist_ok=True)
    if shutil.disk_usage(destination).free < required:
        raise OSError('Insufficient free space for the complete copy')
    (destination / 'PM').mkdir(exist_ok=True)
    started = time.monotonic()

    def copy_one(item):
        relative, size = item
        src, dest = source / relative, destination / 'PM' / relative
        payload = src.read_bytes()
        if len(payload) != size:
            raise ValueError(f'Source changed during collection: {relative}')
        digest = hashlib.sha256(payload).hexdigest()
        dest.parent.mkdir(parents=True, exist_ok=True)
        reused = dest.exists()
        if reused:
            if hashlib.sha256(dest.read_bytes()).hexdigest() != digest:
                raise FileExistsError(f'Different existing destination; refusing overwrite: {dest}')
        else:
            temporary = dest.with_name(dest.name + '.partial')
            temporary.write_bytes(payload)
            if hashlib.sha256(temporary.read_bytes()).hexdigest() != digest:
                raise ValueError(f'Copy hash mismatch: {relative}')
            temporary.replace(dest)
        return [relative, size, digest, ';'.join(sorted(files[relative]))], reused

    copied_bytes = reused_count = 0
    with (destination / 'files.csv.partial').open('w', newline='') as stream:
        writer = csv.writer(stream)
        writer.writerow(['relative_path', 'size_bytes', 'sha256', 'roles'])
        with ThreadPoolExecutor(args.workers) as executor:
            for index, (row, reused) in enumerate(executor.map(copy_one, zip(relative_paths, sizes)), 1):
                writer.writerow(row)
                copied_bytes += row[1]
                reused_count += reused
                if index % 1000 == 0 or index == len(relative_paths):
                    stream.flush()
                    print(f'COPIED_VERIFIED {index}/{len(relative_paths)} bytes={copied_bytes} '
                          f'elapsed={time.monotonic()-started:.1f}s', flush=True)
    (destination / 'files.csv.partial').replace(destination / 'files.csv')
    roles = Counter(role for entries in files.values() for role in entries)
    report = {'completed_utc': datetime.now(timezone.utc).isoformat(),
              'file_count': len(files), 'image_count': len(files)-1,
              'total_bytes': copied_bytes, 'reused_verified_files': reused_count,
              'source_retained': True, 'sha256_verified': True,
              'source_root': str(source), 'role_reference_counts': dict(sorted(roles.items())),
              'elapsed_seconds': time.monotonic()-started}
    (destination / 'collection.json').write_text(json.dumps(report, indent=2)+'\n')
    print('COMPLETE', destination, json.dumps(report), flush=True)


if __name__ == '__main__':
    main()
