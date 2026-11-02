"""Audit an archived FGSM candidate without training or changing the data."""
from paths import CHECKPOINT_ROOT
import argparse
import json
import re
import time
from collections import Counter
from datetime import datetime
from pathlib import Path

import torch

import livdet
from run import PM, fingerprint, load_classifier, write_json


HERE = Path(__file__).resolve().parent
FAKE_MATERIALS = {'woodglue', 'ecoflex00-50', 'latex', 'rtv', 'gelatine', 'liquidecoflex'}


def label_from_name(name):
    stem = Path(name).stem.lower()
    if re.fullmatch(r'(?:alive_)?\d{3}_\d+_\d+', stem):
        return 1, 'Alive'
    material, separator, index = stem.rpartition('_')
    if separator and material in FAKE_MATERIALS and index.isdigit():
        return 0, material
    raise ValueError(f'Unrecognized candidate image label: {name}')


def index_candidate(directory):
    if not directory.is_dir():
        raise FileNotFoundError(directory)
    entries = []
    seen = set()
    for path in sorted(directory.glob('*.png'), key=lambda p: p.name.lower()):
        if path.name.lower() in seen:
            raise ValueError(f'Duplicate candidate ID: {path.name}')
        seen.add(path.name.lower())
        label, material = label_from_name(path.name)
        entries.append({'id': path.name, 'path': str(path), 'label': label, 'material': material})
    if not entries:
        raise ValueError(f'No candidate PNG files found in {directory}')
    return entries


def summarize(entries, predictions):
    if len(entries) != len(predictions) or not entries:
        raise ValueError('Each nonempty input set must have one prediction per image')
    matrix = [[0, 0], [0, 0]]
    materials = {}
    for entry, prediction in zip(entries, predictions):
        if prediction not in [0, 1]:
            raise ValueError(f'Unexpected classifier class: {prediction}')
        matrix[entry['label']][prediction] += 1
        stats = materials.setdefault(entry['material'], {'count': 0, 'correct': 0})
        stats['count'] += 1
        stats['correct'] += int(entry['label'] == prediction)
    classes = {}
    for label, name in [(0, 'Fake'), (1, 'Alive')]:
        total = sum(matrix[label])
        classes[name] = {'count': total, 'correct': matrix[label][label],
                         'accuracy_percent': 100 * matrix[label][label] / total if total else None}
    for stats in materials.values():
        stats['accuracy_percent'] = 100 * stats['correct'] / stats['count']
    correct = matrix[0][0] + matrix[1][1]
    return {'count': len(entries), 'correct': correct, 'accuracy_percent': 100 * correct / len(entries),
            'per_class': classes, 'per_material': materials,
            'balanced_accuracy_percent': sum(c['accuracy_percent'] for c in classes.values()) / 2
            if all(c['count'] for c in classes.values()) else None,
            'confusion_matrix': matrix, 'confusion_matrix_order': ['Fake', 'Alive']}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--data-dir', type=Path, default=PM / 'Self2Self/noise2noise/temporary/data/new_ictc/Total_FGSM')
    parser.add_argument('--classifier', type=Path, default=CHECKPOINT_ROOT / 'livdet/classifier.pth')
    parser.add_argument('--device', choices=['cpu', 'cuda'], default='cuda')
    parser.add_argument('--batch-size', type=int, default=2)
    parser.add_argument('--threads', type=int, default=4)
    parser.add_argument('--max-images', type=int, default=0, help='0 uses all images; positive values are diagnostic subsets')
    parser.add_argument('--output', type=Path)
    parser.add_argument('--check-only', action='store_true')
    args = parser.parse_args()
    if min(args.batch_size, args.threads) < 1 or args.max_images < 0:
        parser.error('Batch size and threads must be positive; max-images must be nonnegative')
    all_entries = index_candidate(args.data_dir)
    print('FOUND', len(all_entries), 'images;', dict(Counter(e['label'] for e in all_entries)),
          '(Fake=0, Alive=1)', flush=True)
    if args.check_only:
        return
    if args.device == 'cuda' and not torch.cuda.is_available():
        raise SystemExit('CUDA unavailable. Activate conda n2n and use your GPU terminal.')
    entries = all_entries[:args.max_images] if args.max_images else all_entries
    out = args.output or HERE / 'runs' / ('fgsm_candidate_' + datetime.now().strftime('%Y%m%d_%H%M%S_%f'))
    out.mkdir(parents=True, exist_ok=False)
    livdet.RECOVERED_IMAGES.clear()
    livdet.RECOVERY_REPORT_PATH = out / 'image_load_recoveries.json'
    write_json(livdet.RECOVERY_REPORT_PATH, {})
    write_json(out / 'manifest.json', entries)
    write_json(out / 'config.json', {
        'arguments': {k: str(v) if isinstance(v, Path) else v for k, v in vars(args).items()},
        'classifier_sha256': fingerprint(args.classifier), 'torch': str(torch.__version__),
        'cuda_runtime': torch.version.cuda, 'dataset_total': len(all_entries),
        'processing': 'Same LivDet PIL grayscale, bilinear resize to 581, divide by 255; frozen classifier',
        'scope': 'Archived attack classification only; no denoiser training or target pairing',
        'comparison_note': 'Candidate whole-folder accuracy; paper evaluation split correspondence unverified'})
    torch.set_num_threads(args.threads)
    torch.use_deterministic_algorithms(True)
    torch.backends.cudnn.benchmark = False
    device = torch.device(args.device)
    model = load_classifier(args.classifier, device)
    started = time.monotonic()
    predictions = []
    for start in range(0, len(entries), args.batch_size):
        chunk = entries[start:start + args.batch_size]
        images = torch.stack([livdet.read_tensor(e['path']) for e in chunk]).to(device)
        with torch.no_grad():
            predictions.extend(model(images).argmax(1).cpu().tolist())
        if len(predictions) % (args.batch_size * 50) == 0 or len(predictions) == len(entries):
            print(f'evaluated {len(predictions)}/{len(entries)}', flush=True)
    metrics = summarize(entries, predictions)
    metrics.update(partial_evaluation=len(entries) < len(all_entries), dataset_total=len(all_entries),
                   paper_fgsm_attacked_accuracy_percent=50.0,
                   previous_fgsm_attacked_accuracy_percent=60.0,
                   delta_to_paper_pp=metrics['accuracy_percent'] - 50.0,
                   comparison_note='All candidate images; exact paper evaluation split is unverified.',
                   elapsed_seconds=time.monotonic() - started)
    write_json(out / 'predictions.json', [dict(e, prediction=p) for e, p in zip(entries, predictions)])
    write_json(out / 'metrics.json', metrics)
    print(json.dumps(metrics, ensure_ascii=False, indent=2), flush=True)
    print('RESULTS', out, flush=True)


if __name__ == '__main__':
    main()
