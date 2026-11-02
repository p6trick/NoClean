"""Evaluate archived LivDet DIP PNGs, without regenerating DIP or attacks."""
from paths import CHECKPOINT_ROOT, loads_json
import argparse
import csv
import json
import time
from datetime import datetime
from pathlib import Path

import torch

import livdet
from run import fingerprint, load_classifier, write_json


PAPER = {'fgsm': 40.00, 'pgd': 39.91, 'cw': 40.00, 'ba': 40.00}


def resolve_entries(collection, spec, attack, count):
    paths = {}
    for folder in spec['folders']:
        for path in Path(folder).rglob('*.png'):
            identity = livdet.canonical_id(path.name)
            if identity in paths:
                raise ValueError(f'Duplicate DIP ID: {identity}')
            paths[identity] = str(path)
    entries = loads_json((collection / 'livdet' / attack / 'source_data_manifest.json').read_text())
    expected = {e['id'] for e in entries}
    if len(expected) != len(entries) or expected != set(paths):
        raise ValueError(f'DIP/evaluation ID mismatch: {attack}')
    return [{**e, 'dip_path': paths[e['id']]} for e in entries[:count]]


def main():
    here = Path(__file__).resolve().parent
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--attacks', nargs='+', choices=list(PAPER), default=list(PAPER))
    parser.add_argument('--device', choices=['cpu', 'cuda'], default='cuda')
    parser.add_argument('--eval-size', type=int, default=1250)
    parser.add_argument('--batch-size', type=int, default=1)
    parser.add_argument('--threads', type=int, default=4)
    parser.add_argument('--checkpoint-root', type=Path, default=CHECKPOINT_ROOT)
    parser.add_argument('--output', type=Path)
    args = parser.parse_args()
    if not 1 <= args.eval_size <= 1250 or min(args.batch_size, args.threads) < 1:
        parser.error('Invalid sizes or thread count')
    if len(set(args.attacks)) != len(args.attacks):
        parser.error('Duplicate attack')
    if args.device == 'cuda' and not torch.cuda.is_available():
        parser.error('CUDA unavailable; use --device cpu')
    torch.set_num_threads(args.threads)
    device = torch.device(args.device)
    inventory = loads_json((here / 'baseline_saved_dip_inventory.json').read_text())
    groups = {a: resolve_entries(args.checkpoint_root, inventory[a], a, args.eval_size) for a in args.attacks}
    output = args.output or here / 'runs' / datetime.now().strftime('saved_dip_eval_%Y%m%d_%H%M%S_%f')
    output.mkdir(parents=True, exist_ok=False)
    livdet.RECOVERY_REPORT_PATH = output / 'image_load_recoveries.json'
    classifier_path = args.checkpoint_root / 'livdet/classifier.pth'
    collection = loads_json((args.checkpoint_root / 'manifest.json').read_text())
    expected_sha = next(r['sha256'] for r in collection['copied_files'] if r['copy'] == 'livdet/classifier.pth')
    if fingerprint(classifier_path) != expected_sha:
        raise ValueError('Classifier hash mismatch')
    write_json(output / 'config.json', {
        'paper': str(here.parent / 'N2N_Journal.pdf'), 'method': 'archived DIP outputs, LivDet',
        'arguments': {k: str(v) if isinstance(v, Path) else v for k, v in vars(args).items()},
        'classifier_sha256': expected_sha, 'image_processing': 'PIL L / resize581 bilinear if necessary /255',
        'limitations': ['DIP generation parameters/version are not fully verified.',
                        'Historical evaluator used Keras; this uses the recovered ONNX-converted classifier.',
                        'Only four attacks have located saved DIP outputs; no full A.7 average yet.',
                        'DIP is a per-image fitting method; no source-attack denoiser is selected here.']})
    classifier = load_classifier(classifier_path, device)
    rows = []
    for attack, entries in groups.items():
        folder = output / attack
        folder.mkdir()
        write_json(folder / 'manifest.json', entries)
        keys = {'clean': 'path', 'attacked': 'attack_path', 'dip': 'dip_path'}
        correct, confusions = dict.fromkeys(keys, 0), {k: [[0, 0], [0, 0]] for k in keys}
        predictions, difference, mean_intensity, count = [], 0.0, 0.0, 0
        started = time.monotonic()
        with torch.inference_mode():
            for start in range(0, len(entries), args.batch_size):
                chunk = entries[start:start + args.batch_size]
                labels = torch.tensor([e['label'] for e in chunk], device=device)
                images = {k: torch.stack([livdet.read_tensor(e[path]) for e in chunk]).to(device)
                          for k, path in keys.items()}
                preds = {k: classifier(img).argmax(1) for k, img in images.items()}
                difference += (images['dip'] - images['attacked']).square().flatten(1).mean(1).sum().item()
                mean_intensity += images['dip'].flatten(1).mean(1).sum().item()
                for key in keys:
                    correct[key] += preds[key].eq(labels).sum().item()
                    for label, pred in zip(labels.tolist(), preds[key].tolist()):
                        confusions[key][label][pred] += 1
                for i, e in enumerate(chunk):
                    predictions.append({'id': e['id'], 'label': e['label'],
                                        **{k: p[i].item() for k, p in preds.items()}})
                count += len(chunk)
                if count % (args.batch_size * 100) == 0 or count == len(entries):
                    print(attack, f'{count}/{len(entries)}', flush=True)
        metrics = {
            'attack': attack, 'count': count, 'partial_evaluation': count != 1250,
            'correct': correct, 'accuracy_percent': {k: 100 * n / count for k, n in correct.items()},
            'confusion_matrix_order': ['Fake', 'Alive'], 'confusion_matrices': confusions,
            'dip_per_class_accuracy_percent': {name: 100 * confusions['dip'][i][i] / sum(confusions['dip'][i])
                                              if sum(confusions['dip'][i]) else None
                                              for i, name in enumerate(['Fake', 'Alive'])},
            'paper_dip_accuracy_percent': PAPER[attack],
            'delta_to_paper_pp': 100 * correct['dip'] / count - PAPER[attack],
            'mean_dip_mse_to_attack': difference / count, 'mean_dip_intensity': mean_intensity / count,
            'elapsed_seconds': time.monotonic() - started}
        write_json(folder / 'metrics.json', metrics)
        write_json(folder / 'predictions.json', predictions)
        rows.append({'attack': attack.upper(), 'count': count,
                     'clean': metrics['accuracy_percent']['clean'], 'attacked': metrics['accuracy_percent']['attacked'],
                     'dip': metrics['accuracy_percent']['dip'], 'paper_dip': PAPER[attack],
                     'delta_pp': metrics['delta_to_paper_pp']})
        write_json(output / 'summary.json', rows)
        with (output / 'summary.csv').open('w') as stream:
            writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
            writer.writeheader()
            writer.writerows(rows)
        print(json.dumps(metrics), flush=True)
    print('RESULTS', output, flush=True)


if __name__ == '__main__':
    main()
