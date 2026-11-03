"""Paired clean/rectified target training on fixed archived LivDet attacks."""
import argparse
from datetime import datetime
import json
import os
from pathlib import Path
import time
from types import SimpleNamespace

os.environ.setdefault('CUBLAS_WORKSPACE_CONFIG', ':4096:8')
import torch

import livdet
from mnist_target_ablation import ATTACKS, seed_all, state_hash, write_tables
from models import Denoiser
from paths import CHECKPOINT_ROOT, PM_ROOT, loads_json
from run import fingerprint, load_classifier, write_json


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--attacks', nargs='+', choices=ATTACKS, default=['fgsm'])
    parser.add_argument('--seeds', nargs='+', type=int, default=[2026])
    parser.add_argument('--pm-root', type=Path, default=PM_ROOT)
    parser.add_argument('--checkpoint-root', type=Path, default=CHECKPOINT_ROOT)
    parser.add_argument('--device', choices=['cuda', 'cpu'], default='cuda')
    parser.add_argument('--epochs', type=int, default=100)
    parser.add_argument('--batch-size', type=int, default=1)
    parser.add_argument('--learning-rate', type=float, default=.001)
    parser.add_argument('--crop-size', type=int, default=581)
    parser.add_argument('--eval-batch-size', type=int, default=1)
    parser.add_argument('--threads', type=int, default=4)
    parser.add_argument('--train-size', type=int, default=1125)
    parser.add_argument('--validation-size', type=int, default=125)
    parser.add_argument('--eval-size', type=int, default=1250)
    parser.add_argument('--output-processing', choices=['png', 'float'], default='png')
    parser.add_argument('--check-only', action='store_true')
    parser.add_argument('--output', type=Path)
    args = parser.parse_args()
    if min(args.epochs, args.batch_size, args.eval_batch_size, args.train_size,
           args.validation_size, args.eval_size, args.threads) < 1 or args.learning_rate <= 0:
        parser.error('Counts and learning rate must be positive')
    if not 32 <= args.crop_size <= 581:
        parser.error('crop-size must be between 32 and 581')
    if len(set(args.attacks)) != len(args.attacks) or len(set(args.seeds)) != len(args.seeds):
        parser.error('Duplicate attack/seed')
    if any(seed < 0 or seed >= 2**32 for seed in args.seeds):
        parser.error('Seeds must be in [0, 2**32)')
    specs = loads_json((Path(__file__).parent / 'livdet_experiments.json').read_text())
    fixed = SimpleNamespace(seed=2026, train_size=1125, validation_size=125, eval_size=1250)
    limits = {'train': args.train_size, 'validation': args.validation_size, 'evaluation': args.eval_size}
    groups, reference = {}, None
    for attack in args.attacks:
        full = livdet.resolve_groups(args.pm_root, specs[attack], fixed)
        recorded = loads_json((args.checkpoint_root / 'livdet' / attack / 'source_data_manifest.json').read_text())
        if [(e['id'], e['label'], e['path']) for e in full['evaluation']] != [
                (e['id'], e['label'], e['path']) for e in recorded]:
            raise ValueError(f'Evaluation partition differs from the archived checkpoint manifest: {attack}')
        if any(limits[key] > len(full[key]) for key in limits):
            parser.error('Requested size exceeds fixed split')
        group = {key: full[key][:limit] for key, limit in limits.items()}
        identity = {key: [(e['id'], e['label'], e['path']) for e in entries] for key, entries in group.items()}
        if reference is not None and reference != identity:
            raise ValueError('Cross-attack split identities differ')
        reference = identity
        ids = [e['id'] for entries in group.values() for e in entries]
        if len(ids) != len(set(ids)):
            raise ValueError('Overlapping train/validation/evaluation IDs')
        for entries in group.values():
            for entry in entries:
                for key in ['path', 'attack_path', 'target_path']:
                    if not Path(entry[key]).is_file():
                        raise FileNotFoundError(entry[key])
        groups[attack] = group
        print('VERIFIED', attack, {key: len(value) for key, value in group.items()}, flush=True)
    if args.check_only:
        return
    if args.device == 'cuda' and not torch.cuda.is_available():
        parser.error('CUDA unavailable; use a GPU terminal or --device cpu for a small smoke test')
    torch.set_num_threads(args.threads)
    torch.use_deterministic_algorithms(True)
    torch.backends.cudnn.benchmark = False
    device = torch.device(args.device)
    output = args.output or Path(__file__).parent / 'runs' / datetime.now().strftime('livdet_target_ablation_%Y%m%d_%H%M%S_%f')
    output.mkdir(parents=True, exist_ok=False)
    classifier_path = args.checkpoint_root / 'livdet/classifier.pth'
    collection = loads_json((args.checkpoint_root / 'manifest.json').read_text())
    expected = next(r['sha256'] for r in collection['copied_files'] if r['copy'] == 'livdet/classifier.pth')
    if fingerprint(classifier_path) != expected:
        raise ValueError('Classifier hash mismatch')
    write_json(output / 'config.json', {
        'arguments': {key: str(value) if isinstance(value, Path) else value for key, value in vars(args).items()},
        'classifier_sha256': expected, 'torch': str(torch.__version__), 'cuda_runtime': torch.version.cuda,
        'gpu_name': torch.cuda.get_device_name(device) if device.type == 'cuda' else None,
        'selection': 'minimum own-target validation MSE; never final evaluation accuracy',
        'architecture': 'shared Denoiser first_pool=2; recovered-checkpoint topology',
        'pairing': 'same initial state, batch order, crop coordinates, L2, Adam and fixed learning rate',
        'split': 'fixed seed2026; 1125 train +125 validation carved from 1250 fit pool;1250 evaluation',
        'augmentation': 'paired random crop only when crop_size<581; validation center crop',
        'clean_input': 'PIL grayscale repeated to RGB, matching prior clean evaluation',
        'partial_evaluation': args.eval_size != 1250,
        'limitations': ['100 epochs/batch1/lr.001 follow latest manuscript; other details not fully verified.',
                       'Full-image training and first_pool=2 are declared implementation choices.',
                       'Internal validation reduces fitting images from manuscript1250 to1125.',
                       'Archived attack/rectification generation parameters unverified.',
                       'Both models retrained; recovered denoisers are not used as a comparison arm.',
                       'EA tentatively maps to archive AA; not an adaptive defense evaluation.']})
    classifier = load_classifier(classifier_path, device)
    rows, hashes = [], {}
    started = time.monotonic()
    for attack, group in groups.items():
        folder = output / attack
        folder.mkdir()
        write_json(folder / 'manifest.json', group)
        image_hashes = {}
        for entries in group.values():
            for entry in entries:
                image_hashes[entry['id']] = {}
                for key in ['path', 'attack_path', 'target_path']:
                    path = entry[key]
                    if path not in hashes:
                        hashes[path] = fingerprint(path)
                    image_hashes[entry['id']][key] = hashes[path]
        write_json(folder / 'input_hashes.json', image_hashes)
        for seed in args.seeds:
            seed_all(seed)
            initial = {key: value.detach().cpu().clone() for key, value in Denoiser(first_pool=2).state_dict().items()}
            initial_hash = state_hash(initial)
            for target in ['clean', 'rectified']:
                seed_all(seed)
                run_folder = folder / f'seed_{seed}' / target
                run_folder.mkdir(parents=True)
                livdet.RECOVERED_IMAGES.clear()
                livdet.RECOVERY_REPORT_PATH = run_folder / 'image_decode_recoveries.json'
                train_args = SimpleNamespace(seed=seed, epochs=args.epochs, batch_size=args.batch_size,
                    eval_batch_size=args.eval_batch_size, crop_size=args.crop_size, first_pool=2,
                    learning_rate=args.learning_rate, scheduler='none', training_target=target,
                    keep_checkpoint_variants=True, output_processing=args.output_processing)
                write_json(run_folder / 'pairing.json', {'seed': seed, 'target': target,
                           'initial_state_sha256': initial_hash, 'batch_and_crop_seed': seed})
                emit = lambda line: print(f'{attack}/{seed}/{target}: {line}', flush=True)
                model, epoch = livdet.fit(group, train_args, device, run_folder, emit, initial_state=initial)
                metrics = livdet.evaluate(classifier, model, group['evaluation'], train_args,
                                          device, run_folder, emit)
                metrics.update(training_target=target, seed=seed, selected_epoch=epoch,
                               initial_state_sha256=initial_hash, partial_evaluation=args.eval_size != 1250,
                               checkpoint_selection='validation-mse')
                write_json(run_folder / 'metrics.json', metrics)
                accuracy = metrics['accuracy_percent']
                rows.append({'attack': attack, 'target': target, 'seed': seed, 'count': metrics['count'],
                    'clean': accuracy['clean'], 'attacked': accuracy['attacked'],
                    'defended_attacked': accuracy['defended_attacked'], 'defended_clean': accuracy['defended_clean'],
                    'selected_epoch': epoch, 'initial_state_sha256': initial_hash,
                    'checkpoint_selection': 'validation-mse', 'output_processing': args.output_processing})
                write_tables(output, rows, args.attacks)
                print('RESULT', json.dumps(rows[-1]), flush=True)
                del model
    write_json(output / 'completion.json', {'completed_runs': len(rows),
               'elapsed_seconds': time.monotonic()-started})
    print('RESULTS', output, flush=True)


if __name__ == '__main__':
    main()
