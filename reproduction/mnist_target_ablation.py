"""Paired clean-vs-rectified supervision on fixed archived MNIST attacks."""
import argparse
import csv
from datetime import datetime
import hashlib
import json
import os
import random
import statistics
import time
from pathlib import Path
from types import SimpleNamespace

os.environ.setdefault('CUBLAS_WORKSPACE_CONFIG', ':4096:8')
import numpy as np
import torch

from models import Denoiser
from paths import CHECKPOINT_ROOT, loads_json
from run import evaluate, fingerprint, image_batches, load_classifier, train_denoiser, write_json

ATTACKS = ['fgsm', 'pgd', 'cw', 'ba', 'bim', 'aa']


def state_hash(state):
    digest = hashlib.sha256()
    for key, value in sorted(state.items()):
        digest.update(key.encode())
        digest.update(value.detach().cpu().contiguous().numpy().tobytes())
    return digest.hexdigest()


def seed_all(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)


def load_split(entries, batch_size, device):
    arrays = {}
    for name, key in [('source', 'attack_path'), ('clean', 'path'), ('rectified', 'target_path')]:
        arrays[name] = torch.cat([images.cpu() for images, _ in
                                 image_batches(entries, batch_size, device, key)])
    return arrays


def write_tables(output, rows, attacks):
    with (output / 'per_seed.csv').open('w', newline='') as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    for metric, filename in [('defended_attacked', 'attack_accuracy.csv'),
                             ('defended_clean', 'clean_input_accuracy.csv')]:
        with (output / filename).open('w', newline='') as stream:
            writer = csv.writer(stream)
            writer.writerow(['Target', *[a.upper() if a != 'aa' else 'EA (archive AA)' for a in attacks], 'Mean'])
            for target in ['clean', 'rectified']:
                cells, means = [], []
                for attack in attacks:
                    values = [row[metric] for row in rows if row['target'] == target and row['attack'] == attack]
                    if values:
                        mean = statistics.mean(values)
                        means.append(mean)
                        cells.append(f'{mean:.2f}' if len(values) == 1 else
                                     f'{mean:.2f} ± {statistics.stdev(values):.2f}')
                    else:
                        cells.append('')
                writer.writerow([target, *cells, f'{statistics.mean(means):.2f}' if len(means) == len(attacks) else ''])


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--attacks', nargs='+', choices=ATTACKS, default=ATTACKS)
    parser.add_argument('--seeds', nargs='+', type=int, default=[2026])
    parser.add_argument('--checkpoint-root', type=Path, default=CHECKPOINT_ROOT)
    parser.add_argument('--device', choices=['cpu', 'cuda'], default='cuda')
    parser.add_argument('--epochs', type=int, default=30)
    parser.add_argument('--batch-size', type=int, default=32)
    parser.add_argument('--learning-rate', type=float, default=.001)
    parser.add_argument('--checkpoint-selection', choices=['validation-mse', 'last'],
                        default='validation-mse',
                        help='Select each condition by its own validation MSE, or use the final epoch')
    parser.add_argument('--threads', type=int, default=4)
    parser.add_argument('--train-size', type=int, default=4500)
    parser.add_argument('--validation-size', type=int, default=500)
    parser.add_argument('--eval-size', type=int, default=5000)
    parser.add_argument('--check-only', action='store_true')
    parser.add_argument('--output', type=Path)
    args = parser.parse_args()
    if min(args.epochs, args.batch_size, args.threads, args.train_size, args.validation_size, args.eval_size) < 1 or args.learning_rate <= 0:
        parser.error('Sizes, epochs, threads and learning rate must be positive')
    if len(set(args.attacks)) != len(args.attacks) or len(set(args.seeds)) != len(args.seeds):
        parser.error('Duplicate attacks/seeds')
    groups, reference = {}, None
    for attack in args.attacks:
        manifest = loads_json((args.checkpoint_root / 'mnist' / attack / 'source_data_manifest.json').read_text())
        limits = {'train': args.train_size, 'validation': args.validation_size, 'evaluation': args.eval_size}
        if any(limits[key] > len(manifest[key]) for key in limits):
            parser.error(f'{attack}: requested size exceeds recorded split')
        selected = {key: manifest[key][:limit] for key, limit in limits.items()}
        identity = {key: [(e['id'], e['label'], e['path']) for e in entries] for key, entries in selected.items()}
        if reference is not None and identity != reference:
            raise ValueError('Cross-attack clean IDs/labels/splits differ')
        reference = identity
        ids = [e['id'] for entries in selected.values() for e in entries]
        if len(set(ids)) != len(ids):
            raise ValueError('Overlapping split IDs')
        for entries in selected.values():
            for entry in entries:
                for key in ['path', 'attack_path', 'target_path']:
                    if not Path(entry[key]).is_file():
                        raise FileNotFoundError(entry[key])
        groups[attack] = selected
        print('VERIFIED', attack, {key: len(value) for key, value in selected.items()}, flush=True)
    if args.check_only:
        return
    if args.device == 'cuda' and not torch.cuda.is_available():
        parser.error('CUDA unavailable; run from a GPU terminal or use --device cpu')
    torch.set_num_threads(args.threads)
    torch.use_deterministic_algorithms(True)
    torch.backends.cudnn.benchmark = False
    device = torch.device(args.device)
    output = args.output or Path(__file__).resolve().parent / 'runs' / datetime.now().strftime('mnist_target_ablation_%Y%m%d_%H%M%S_%f')
    output.mkdir(parents=True, exist_ok=False)
    classifier_file = args.checkpoint_root / 'mnist/classifier.pth'
    classifier_hash = fingerprint(classifier_file)
    collection = loads_json((args.checkpoint_root / 'manifest.json').read_text())
    expected = next(row['sha256'] for row in collection['copied_files'] if row['copy'] == 'mnist/classifier.pth')
    if classifier_hash != expected:
        raise ValueError('Classifier hash mismatch')
    write_json(output / 'config.json', {
        'arguments': {k: str(v) if isinstance(v, Path) else v for k, v in vars(args).items()},
        'classifier_sha256': classifier_hash, 'torch': str(torch.__version__),
        'cuda_runtime': torch.version.cuda, 'device': str(device),
        'gpu_name': torch.cuda.get_device_name(device) if device.type == 'cuda' else None,
        'architecture': 'Denoiser first_pool=1; identical architecture for both targets',
        'selection': ('minimum own-target validation MSE; final evaluation never used for selection'
                      if args.checkpoint_selection == 'validation-mse' else 'fixed final epoch'),
        'split': 'fixed archived manifests; training seeds never change split IDs',
        'pairing': 'identical initial state, batch order, source tensors, Adam, L2 and training budget',
        'augmentation': 'none', 'evaluation_processing': 'float RGB to weighted grayscale then clamp',
        'partial_evaluation': args.eval_size != 5000,
        'limitations': ['Archived attack/rectification generation settings unverified.',
                       'EA tentatively maps to archive AA; standard AutoAttack not established.',
                       'Not an adaptive attack on the complete defense.']})
    classifier = load_classifier(classifier_file, device)
    rows = []
    started = time.monotonic()
    for attack, group in groups.items():
        folder = output / attack
        folder.mkdir()
        write_json(folder / 'manifest.json', group)
        hashes = {e['id']: {key: fingerprint(e[key]) for key in ['path', 'attack_path', 'target_path']}
                  for entries in group.values() for e in entries}
        write_json(folder / 'input_hashes.json', hashes)
        cached = {split: load_split(group[split], args.batch_size, device) for split in ['train', 'validation']}
        for seed in args.seeds:
            seed_all(seed)
            initial = {key: value.detach().cpu().clone() for key, value in Denoiser().state_dict().items()}
            initial_hash = state_hash(initial)
            for target in ['clean', 'rectified']:
                seed_all(seed)
                run_folder = folder / f'seed_{seed}' / target
                run_folder.mkdir(parents=True)
                train_args = SimpleNamespace(seed=seed, batch_size=args.batch_size, epochs=args.epochs,
                    learning_rate=args.learning_rate, checkpoint_selection=args.checkpoint_selection,
                    keep_checkpoint_variants=True, training_target=target,
                    input_mode='saved', saved_targets='archived rectification; evaluation diagnostic only',
                    attack=attack, fit_half='last')
                pairs = {split: {'source': value['source'], 'target': value[target]}
                         for split, value in cached.items()}
                write_json(run_folder / 'pairing.json', {'initial_state_sha256': initial_hash,
                           'seed': seed, 'target': target, 'batch_order_seed': seed})
                model, epoch = train_denoiser(pairs['train'], pairs['validation'], train_args, device,
                                             run_folder, initial_state=initial)
                metrics = evaluate(classifier, model, group['evaluation'], train_args, device, run_folder)
                metrics.update(training_target=target, selected_epoch=epoch, seed=seed,
                               checkpoint_selection=args.checkpoint_selection,
                               initial_state_sha256=initial_hash, partial_evaluation=args.eval_size != 5000)
                write_json(run_folder / 'metrics.json', metrics)
                accuracy = metrics['accuracy_percent']
                rows.append({'attack': attack, 'target': target, 'seed': seed, 'count': metrics['count'],
                             'clean': accuracy['clean'], 'attacked': accuracy['attacked'],
                             'defended_attacked': accuracy['defended_attacked'],
                             'defended_clean': accuracy['defended_clean'], 'selected_epoch': epoch,
                             'initial_state_sha256': initial_hash,
                             'checkpoint_selection': args.checkpoint_selection})
                write_tables(output, rows, args.attacks)
                print('RESULT', json.dumps(rows[-1]), flush=True)
                del model
        del cached
    write_json(output / 'completion.json', {'completed_runs': len(rows),
               'elapsed_seconds': time.monotonic()-started})
    print('RESULTS', output, flush=True)


if __name__ == '__main__':
    main()
