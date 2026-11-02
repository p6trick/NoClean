"""N2N_Journal.pdf Table6: fixed same-dataset denoisers on clean evaluation images."""
from paths import CHECKPOINT_ROOT, loads_json
import argparse
import csv
import json
from datetime import datetime
from pathlib import Path

import torch

from livdet import read_tensor
from models import Denoiser, grayscale
from run import fingerprint, image_batches, load_classifier, write_json
from table5_transfer import png_grayscale


ATTACKS = ['fgsm', 'pgd', 'cw', 'ba', 'bim', 'aa']
PAPER = {
    'livdet': dict(zip(ATTACKS, [88.88, 89.12, 89.12, 89.12, 89.12, 89.12])),
    'mnist': dict(zip(ATTACKS, [98.64, 97.32, 95.86, 95.95, 97.97, 97.10])),
}


def clean_batches(entries, dataset, batch_size, device):
    if dataset == 'mnist':
        yield from image_batches(entries, batch_size, device, 'path')
    else:
        for start in range(0, len(entries), batch_size):
            chunk = entries[start:start + batch_size]
            yield torch.stack([read_tensor(e['path']) for e in chunk]).to(device), torch.tensor(
                [e['label'] for e in chunk], device=device)


def write_comparison(output, rows):
    write_json(output / 'comparison.json', rows)
    if rows:
        with (output / 'comparison.csv').open('w') as stream:
            writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
            writer.writeheader()
            writer.writerows(rows)


def main():
    here = Path(__file__).resolve().parent
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--checkpoint-root', type=Path, default=CHECKPOINT_ROOT)
    parser.add_argument('--datasets', nargs='+', choices=['mnist', 'livdet'], default=['mnist', 'livdet'])
    parser.add_argument('--device', choices=['cpu', 'cuda'], default='cuda')
    parser.add_argument('--mnist-eval-size', type=int, default=5000)
    parser.add_argument('--livdet-eval-size', type=int, default=1250)
    parser.add_argument('--mnist-batch-size', type=int, default=32)
    parser.add_argument('--livdet-batch-size', type=int, default=1)
    parser.add_argument('--threads', type=int, default=4)
    parser.add_argument('--output', type=Path)
    args = parser.parse_args()
    if not 1 <= args.mnist_eval_size <= 5000 or not 1 <= args.livdet_eval_size <= 1250:
        parser.error('Invalid evaluation counts')
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
    output = args.output or here / 'runs' / datetime.now().strftime('table6_clean_%Y%m%d_%H%M%S_%f')
    output.mkdir(parents=True, exist_ok=False)
    write_json(output / 'config.json', {
        'paper': str(here.parent / 'N2N_Journal.pdf'), 'table': 6,
        'arguments': {k: str(v) if isinstance(v, Path) else v for k, v in vars(args).items()},
        'protocol': 'Same-dataset models applied to original clean held-out IDs; no attacks or training.',
        'input': 'PIL grayscale /255 repeated3 channels; MNIST28, LivDet581',
        'primary_output': {'mnist': 'float', 'livdet': 'png'},
        'limitations': ['MNIST models are newly trained; LivDet models are recovered historical weights.',
                        'PGD-labelled LivDet model training attack remains unconfirmed.',
                        'Exact paper-final checkpoint versions and preprocessing remain partly unverified.']})
    rows = []
    for dataset in args.datasets:
        folder = output / dataset
        folder.mkdir()
        size = args.mnist_eval_size if dataset == 'mnist' else args.livdet_eval_size
        batch_size = args.mnist_batch_size if dataset == 'mnist' else args.livdet_batch_size
        entries = None
        for attack in ATTACKS:
            manifest = loads_json((args.checkpoint_root / dataset / attack / 'source_data_manifest.json').read_text())
            group = manifest['evaluation'] if isinstance(manifest, dict) else manifest
            current = {e['id']: (e['label'], e['path']) for e in group}
            if len(current) != len(group):
                raise ValueError('Duplicate evaluation IDs')
            if entries is None:
                reference = current
                entries = group[:size]
            elif current != reference:
                raise ValueError('Model evaluation IDs/labels/clean paths differ')
        if len(entries) != size:
            raise ValueError('Insufficient evaluation samples')
        write_json(folder / 'manifest.json', entries)
        write_json(folder / 'evaluation_info.json', {
            'count': size, 'partial_evaluation': size != (5000 if dataset == 'mnist' else 1250)})
        classifier_path = args.checkpoint_root / dataset / 'classifier.pth'
        recorded = next(r for r in collection['copied_files'] if r['copy'] == f'{dataset}/classifier.pth')
        if fingerprint(classifier_path) != recorded['sha256']:
            raise ValueError('Classifier hash mismatch')
        classifier = load_classifier(classifier_path, device)
        with torch.inference_mode():
            before = sum(classifier(images).argmax(1).eq(labels).sum().item()
                         for images, labels in clean_batches(entries, dataset, batch_size, device))
        baseline = {'count': size, 'correct': before, 'accuracy_percent': 100 * before / size,
                    'paper_accuracy_percent': 98.58 if dataset == 'mnist' else 89.12}
        write_json(folder / 'baseline.json', baseline)
        print(dataset, 'BASELINE', json.dumps(baseline), flush=True)
        for attack in ATTACKS:
            cfg = settings[dataset, attack]
            path = args.checkpoint_root / cfg['checkpoint']
            if fingerprint(path) != cfg['sha256']:
                raise ValueError(f'Checkpoint hash mismatch: {path}')
            saved = torch.load(path, map_location='cpu', weights_only=True)
            state = saved['state_dict'] if cfg['checkpoint_format'] == 'wrapped_state_dict' else saved
            model = Denoiser(first_pool=cfg['first_pool'])
            model.load_state_dict(state, strict=True)
            model.to(device).eval().requires_grad_(False)
            correct, count, predictions = {'float': 0, 'png': 0}, 0, []
            with torch.inference_mode():
                for images, labels in clean_batches(entries, dataset, batch_size, device):
                    result = model(images.repeat(1, 3, 1, 1))
                    if not torch.isfinite(result).all():
                        raise ValueError(f'Nonfinite output: {dataset}/{attack}')
                    variants = {'float': grayscale(result), 'png': png_grayscale(result)}
                    preds = {k: classifier(v).argmax(1) for k, v in variants.items()}
                    for key in correct:
                        correct[key] += preds[key].eq(labels).sum().item()
                    for i in range(len(labels)):
                        predictions.append({'id': entries[count + i]['id'], 'label': labels[i].item(),
                                            **{k: p[i].item() for k, p in preds.items()}})
                    count += len(labels)
                    if count % (batch_size * 25) == 0 or count == size:
                        print(dataset, attack, f'{count}/{size}', flush=True)
            primary = 'float' if dataset == 'mnist' else 'png'
            accuracy = 100 * correct[primary] / count
            row = {'dataset': dataset, 'model_label': attack.upper(), 'count': count,
                   'before_accuracy_percent': baseline['accuracy_percent'],
                   'float_accuracy_percent': 100 * correct['float'] / count,
                   'png_accuracy_percent': 100 * correct['png'] / count,
                   'reported_output': primary, 'reported_accuracy_percent': accuracy,
                   'paper_accuracy_percent': PAPER[dataset][attack],
                   'delta_pp': accuracy - PAPER[dataset][attack], 'first_pool': cfg['first_pool'],
                   'checkpoint': str(path), 'sha256': cfg['sha256']}
            write_json(folder / f'{attack}_predictions.json', predictions)
            write_json(folder / f'{attack}_metrics.json', {**row, 'correct': correct})
            rows.append(row)
            write_comparison(output, rows)
            print(json.dumps(row), flush=True)
            del model
        del classifier
    print('RESULTS', output, flush=True)


if __name__ == '__main__':
    main()
