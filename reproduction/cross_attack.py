"""N2N_Journal.pdf Table3(LivDet) / Table4(MNIST): average five source models."""
from paths import CHECKPOINT_ROOT, loads_json
import argparse
import csv
import json
from datetime import datetime
from pathlib import Path

import torch

from evaluate_legacy_livdet import classifier_input
from livdet import read_tensor
from models import Denoiser, grayscale
from run import fingerprint, image_batches, load_classifier, write_json


ATTACKS = ['fgsm', 'pgd', 'cw', 'ba', 'bim', 'aa']
PAPER = {
    'livdet': [65.88, 72.67, 78.92, 78.53, 77.03, 85.57],
    'mnist': [95.95, 80.18, 94.67, 98.18, 95.97, 97.01],
}


def cross_summary(matrix):
    """Rows: source model; columns: target attack. Exclude diagonal, average column."""
    return {target: sum(matrix[source][target] for source in ATTACKS if source != target) / 5
            for target in ATTACKS}


def save_tables(folder, dataset, matrix):
    with (folder / 'matrix.csv').open('w') as stream:
        writer = csv.writer(stream)
        writer.writerow(['model_label / target_attack', *[a.upper() for a in ATTACKS]])
        for source in ATTACKS:
            writer.writerow([source.upper(), *[matrix[source].get(target, '') for target in ATTACKS]])
    write_json(folder / 'matrix.json', matrix)
    if any(len(row) != 6 for row in matrix.values()):
        return
    summary = cross_summary(matrix)
    rows = [{'target_attack': a.upper(), 'cross_attack_accuracy_percent': summary[a],
             'paper_accuracy_percent': PAPER[dataset][i],
             'delta_pp': summary[a] - PAPER[dataset][i]} for i, a in enumerate(ATTACKS)]
    rows.append({'target_attack': 'MEAN',
                 'cross_attack_accuracy_percent': sum(summary.values()) / 6,
                 'paper_accuracy_percent': sum(PAPER[dataset]) / 6,
                 'delta_pp': (sum(summary.values()) - sum(PAPER[dataset])) / 6})
    with (folder / 'comparison.csv').open('w') as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    write_json(folder / 'comparison.json', rows)


def main():
    here = Path(__file__).resolve().parent
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--datasets', nargs='+', choices=['mnist', 'livdet'], default=['mnist', 'livdet'])
    parser.add_argument('--matched-only', action='store_true',
                        help='Evaluate attack-matched diagonal only (Tables1/2), skipping cross-attack pairs')
    parser.add_argument('--checkpoint-root', type=Path, default=CHECKPOINT_ROOT)
    parser.add_argument('--device', choices=['cpu', 'cuda'], default='cuda')
    parser.add_argument('--mnist-eval-size', type=int, default=5000)
    parser.add_argument('--livdet-eval-size', type=int, default=1250)
    parser.add_argument('--mnist-batch-size', type=int, default=32)
    parser.add_argument('--livdet-batch-size', type=int, default=1)
    parser.add_argument('--threads', type=int, default=4)
    parser.add_argument('--output', type=Path)
    args = parser.parse_args()
    if not 1 <= args.mnist_eval_size <= 5000 or not 1 <= args.livdet_eval_size <= 1250:
        parser.error('Invalid evaluation count')
    if min(args.mnist_batch_size, args.livdet_batch_size, args.threads) < 1:
        parser.error('Batch sizes and threads must be positive')
    if len(set(args.datasets)) != len(args.datasets):
        parser.error('Duplicate dataset')
    if args.device == 'cuda' and not torch.cuda.is_available():
        parser.error('CUDA unavailable; use --device cpu')
    torch.set_num_threads(args.threads)
    device = torch.device(args.device)
    collection = loads_json((args.checkpoint_root / 'manifest.json').read_text())
    settings = {(r['dataset'], r['attack']): r for r in collection['models']}
    output = args.output or here / 'runs' / datetime.now().strftime('cross_attack_%Y%m%d_%H%M%S_%f')
    output.mkdir(parents=True, exist_ok=False)
    write_json(output / 'config.json', {
        'paper': 'N2N_Journal.pdf', 'tables': ({'livdet': 1, 'mnist': 2} if args.matched_only else
                                            {'livdet': 3, 'mnist': 4}),
        'arguments': {k: str(v) if isinstance(v, Path) else v for k, v in vars(args).items()},
        'protocol': ('Attack-matched diagonal only; no cross-attack averaging' if args.matched_only else
                     'Six source models by six target attacks; column mean excluding diagonal, no voting.'),
        'limitations': ['LivDet PGD-labelled candidate training attack is unconfirmed; exclusion condition conditional.',
                        'MNIST uses newly trained models; LivDet uses recovered historical weights.',
                        'Saved attack versions are not fully verified, notably MNIST CW/BA training-target pairing.',
                        'Inference-only non-adaptive evaluation; other defense baselines are not reimplemented.']})
    for dataset in args.datasets:
        folder = output / dataset
        folder.mkdir()
        limit = args.mnist_eval_size if dataset == 'mnist' else args.livdet_eval_size
        batch_size = args.mnist_batch_size if dataset == 'mnist' else args.livdet_batch_size
        classifier_file = args.checkpoint_root / dataset / 'classifier.pth'
        recorded = next(r for r in collection['copied_files'] if r['copy'] == f'{dataset}/classifier.pth')
        if fingerprint(classifier_file) != recorded['sha256']:
            raise ValueError(f'Classifier hash mismatch: {classifier_file}')
        classifier = load_classifier(classifier_file, device)
        models, provenance, groups = {}, {}, {}
        for source in ATTACKS:
            cfg = settings[dataset, source]
            path = args.checkpoint_root / cfg['checkpoint']
            if fingerprint(path) != cfg['sha256']:
                raise ValueError(f'Checkpoint hash mismatch: {path}')
            saved = torch.load(path, map_location='cpu', weights_only=True)
            state = saved['state_dict'] if cfg['checkpoint_format'] == 'wrapped_state_dict' else saved
            model = Denoiser(first_pool=cfg['first_pool'])
            model.load_state_dict(state, strict=True)
            models[source] = model.to(device).eval().requires_grad_(False)
            provenance[source] = cfg
            manifest = loads_json((args.checkpoint_root / dataset / source / 'source_data_manifest.json').read_text())
            entries = manifest['evaluation'] if isinstance(manifest, dict) else manifest
            if len(entries) < limit or len({e['id'] for e in entries}) != len(entries):
                raise ValueError('Insufficient entries or duplicate evaluation IDs')
            groups[source] = entries
        reference = {e['id']: (e['label'], e['path']) for e in groups['fgsm']}
        for target in ATTACKS:
            if {e['id']: (e['label'], e['path']) for e in groups[target]} != reference:
                raise ValueError('Target attack evaluation identity/label mismatch')
        write_json(folder / 'checkpoints.json', provenance)
        write_json(folder / 'evaluation_info.json', {
            'count': limit, 'partial_evaluation': limit != (5000 if dataset == 'mnist' else 1250),
            'output_processing': 'float grayscale' if dataset == 'mnist' else 'RGB clamp / uint8 PIL / grayscale',
            'diagonal': 'Attack-matched controls, excluded from Table3/4 averages.'})
        matrix = {source: {} for source in ATTACKS}
        matched_rows = []
        for target in ATTACKS:
            entries = groups[target][:limit]
            write_json(folder / f'{target}_manifest.json', entries)
            correct, baseline, count, predictions = dict.fromkeys([target] if args.matched_only else ATTACKS, 0), 0, 0, []
            with torch.inference_mode():
                if dataset == 'mnist':
                    batches = ((gray.repeat(1, 3, 1, 1), gray, labels)
                               for gray, labels in image_batches(entries, batch_size, device, 'attack_path'))
                else:
                    def livdet_batches():
                        for start in range(0, len(entries), batch_size):
                            chunk = entries[start:start + batch_size]
                            rgb = torch.stack([read_tensor(e['attack_path'], True) for e in chunk]).to(device)
                            gray = torch.stack([read_tensor(e['attack_path']) for e in chunk]).to(device)
                            yield rgb, gray, torch.tensor([e['label'] for e in chunk], device=device)
                    batches = livdet_batches()
                for rgb, gray, labels in batches:
                    raw_preds = classifier(gray).argmax(1)
                    baseline += raw_preds.eq(labels).sum().item()
                    preds = {}
                    for source, denoiser in models.items():
                        if args.matched_only and source != target:
                            continue
                        result = denoiser(rgb)
                        if not torch.isfinite(result).all():
                            raise ValueError(f'Nonfinite output: {dataset}/{source}/{target}')
                        if dataset == 'mnist':
                            defended = grayscale(result)
                        else:
                            defended = torch.cat([classifier_input(result[i:i+1], True)
                                                  for i in range(len(result))])
                        preds[source] = classifier(defended).argmax(1)
                        correct[source] += preds[source].eq(labels).sum().item()
                    for i in range(len(labels)):
                        predictions.append({'id': entries[count + i]['id'], 'label': labels[i].item(),
                                            'attacked': raw_preds[i].item(),
                                            **{s: p[i].item() for s, p in preds.items()}})
                    count += len(labels)
                    if count % (batch_size * 25) == 0 or count == limit:
                        print(f'{dataset}/{target}: {count}/{limit}', flush=True)
            write_json(folder / f'{target}_predictions.json', predictions)
            write_json(folder / f'{target}_metrics.json', {
                'count': count, 'attacked_correct': baseline, 'correct': correct,
                'accuracy_percent': {s: 100 * n / count for s, n in correct.items()}})
            for source in ATTACKS:
                if args.matched_only and source != target:
                    continue
                matrix[source][target] = 100 * correct[source] / count
            save_tables(folder, dataset, matrix)
            value = matrix[target][target]
            paper = settings[dataset, target]['paper_accuracy_percent']
            matched_rows.append({'attack': target.upper(), 'count': count,
                                 'accuracy_percent': value, 'paper_accuracy_percent': paper,
                                 'delta_pp': value - paper})
            with (folder / 'matched_comparison.csv').open('w') as stream:
                writer = csv.DictWriter(stream, fieldnames=list(matched_rows[0]))
                writer.writeheader()
                writer.writerows(matched_rows)
            print(dataset, target, {s: matrix[s][target] for s in ATTACKS if target in matrix[s]}, flush=True)
        print('COMPARISON', folder / ('matched_comparison.csv' if args.matched_only else 'comparison.csv'), flush=True)
        del classifier, models
    print('RESULTS', output, flush=True)


if __name__ == '__main__':
    main()
