"""Diagnose unidentified legacy checkpoints on the original PGD training partition."""
from paths import CHECKPOINT_ROOT, loads_json
import argparse
import csv
import json
import time
from datetime import datetime
from pathlib import Path
from types import SimpleNamespace

import torch
from torch.nn import functional as F

from evaluate_legacy_livdet import classifier_input
from livdet import HERE, read_tensor, resolve_groups
from models import Denoiser
from run import PM, fingerprint, load_classifier, write_json


def candidate_paths(inventory, noted, include_mapped=False):
    mapped = {(r['family'], r['run']) for r in noted['checkpoints']}
    controls = {('temporary', run) for run in ['adv-0115', 'adv-2006', 'adv-0924']}
    result = []
    for run in inventory['runs']:
        identity = (run['family'], run['run'])
        known = identity in mapped
        if known and not include_mapped and identity not in controls:
            continue
        fallback = Path(run['best_validation_checkpoint'])
        hundred = sorted(fallback.parent.glob('n2n-epoch100-*.pt'))
        if len(hundred) > 1:
            raise ValueError(f'Ambiguous epoch100: {fallback.parent}')
        path = hundred[0] if hundred else fallback
        if not path.is_file():
            raise FileNotFoundError(path)
        result.append({'family': run['family'], 'run': run['run'], 'checkpoint': str(path),
                       'selection': 'epoch100' if hundred else 'historical minimum validation loss',
                       'role': 'mapped control' if identity in controls else
                               'mapped candidate' if known else 'unidentified candidate'})
    return result


def save_ranking(output, rows):
    # Diagnostic ranking only; it does not establish training attack or paper provenance.
    ranked = sorted(rows, key=lambda r: (-r['png_accuracy_percent'], r['target_mse'], r['checkpoint']))
    write_json(output / 'ranking.json', ranked)
    if ranked:
        with (output / 'ranking.csv').open('w') as stream:
            writer = csv.DictWriter(stream, fieldnames=list(ranked[0]))
            writer.writeheader()
            writer.writerows(ranked)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--device', choices=['cpu', 'cuda'], default='cuda')
    parser.add_argument('--sample-size', type=int, default=32)
    parser.add_argument('--first-pools', nargs='+', type=int, choices=[1, 2], default=[2])
    parser.add_argument('--runs', nargs='+', help='Optional run IDs, e.g. adv-0133 adv-2027')
    parser.add_argument('--include-mapped', action='store_true')
    parser.add_argument('--seed', type=int, default=2026)
    parser.add_argument('--threads', type=int, default=4)
    parser.add_argument('--output', type=Path)
    parser.add_argument('--list-only', action='store_true')
    args = parser.parse_args()
    if not 1 <= args.sample_size < 1250 or args.threads < 1:
        parser.error('sample-size must be 1..1249; threads must be positive')
    project = HERE.parent
    inventory = loads_json((project / 'checkpoint_inventory.json').read_text())
    noted = loads_json((project / 'legacy_checkpoint_note_verification.json').read_text())
    candidates = candidate_paths(inventory, noted, args.include_mapped)
    if args.runs:
        candidates = [r for r in candidates if r['run'] in args.runs]
        missing = set(args.runs) - {r['run'] for r in candidates}
        if missing:
            parser.error(f'Unavailable or excluded runs: {sorted(missing)}; use --include-mapped if needed')
    if not candidates:
        parser.error('No candidates')
    print(json.dumps(candidates, indent=2), flush=True)
    if args.list_only:
        return
    if args.device == 'cuda' and not torch.cuda.is_available():
        parser.error('CUDA unavailable; use --device cpu')
    torch.set_num_threads(args.threads)
    device = torch.device(args.device)
    spec = loads_json((HERE / 'livdet_experiments.json').read_text())['pgd']
    selection = SimpleNamespace(train_size=1250 - args.sample_size, validation_size=args.sample_size,
                                eval_size=1250, seed=args.seed, regenerate_targets=False)
    groups = resolve_groups(PM, spec, selection)
    entries = groups['validation']
    if {e['id'] for e in entries} & {e['id'] for e in groups['evaluation']}:
        raise ValueError('Screening/evaluation ID overlap')
    output = args.output or HERE / 'runs' / datetime.now().strftime('pgd_checkpoint_screen_%Y%m%d_%H%M%S_%f')
    output.mkdir(parents=True, exist_ok=False)
    write_json(output / 'manifest.json', entries)
    classifier_path = CHECKPOINT_ROOT / 'livdet/classifier.pth'
    write_json(output / 'config.json', {
        'arguments': {k: str(v) if isinstance(v, Path) else v for k, v in vars(args).items()},
        'candidates': candidates, 'classifier_sha256': fingerprint(classifier_path),
        'split': 'stratified subset of original PGD training partition; final1250 excluded',
        'limitations': ['Legacy models may have trained on these IDs; this is not held-out performance.',
                        'High PGD accuracy does not establish PGD training provenance.',
                        'One representative epoch per run; other epochs are not excluded.',
                        'Pooling2/full581 with local skip alignment; historical configuration unverified.',
                        'PNG path uses PIL; historical evaluation may use TensorFlow resize.']})
    classifier = load_classifier(classifier_path, device)
    cached = [{**e, 'rgb': read_tensor(e['attack_path'], True)[None],
               'source': read_tensor(e['attack_path'])[None],
               'target': read_tensor(e['target_path'])[None],
               'clean': read_tensor(e['path'])[None]} for e in entries]
    baseline = {key: 0 for key in ['clean', 'source', 'target']}
    with torch.inference_mode():
        for e in cached:
            for key in baseline:
                baseline[key] += int(classifier(e[key].to(device)).argmax(1).item() == e['label'])
    write_json(output / 'baseline.json', {
        'count': len(entries), 'correct': baseline,
        'accuracy_percent': {k: 100 * v / len(entries) for k, v in baseline.items()},
        'identity_target_mse': sum(F.mse_loss(e['source'], e['target']).item() for e in cached) / len(entries)})
    rows, errors = [], []
    for candidate in candidates:
        path = Path(candidate['checkpoint'])
        try:
            state = torch.load(path, map_location='cpu', weights_only=True)
        except Exception as exc:
            errors.append({**candidate, 'error': f'{type(exc).__name__}: {exc}'})
            write_json(output / 'errors.json', errors)
            print('CHECKPOINT_ERROR', errors[-1], flush=True)
            continue
        sha = fingerprint(path)
        for pool in args.first_pools:
            model = Denoiser(first_pool=pool)
            try:
                model.load_state_dict(state, strict=True)
            except (RuntimeError, TypeError) as exc:
                errors.append({**candidate, 'first_pool': pool, 'error': f'{type(exc).__name__}: {exc}'})
                write_json(output / 'errors.json', errors)
                print('CHECKPOINT_ERROR', errors[-1], flush=True)
                continue
            model.to(device).eval().requires_grad_(False)
            correct = {'float': 0, 'png': 0}
            per_class = {key: {0: 0, 1: 0} for key in correct}
            predictions, mse = [], 0.0
            started = time.monotonic()
            with torch.inference_mode():
                for e in cached:
                    rgb = model(e['rgb'].to(device))
                    if not torch.isfinite(rgb).all():
                        raise ValueError(f'Nonfinite output: {path}, {e["id"]}')
                    images = {key: classifier_input(rgb, key == 'png') for key in correct}
                    preds = {key: classifier(value).argmax(1).item() for key, value in images.items()}
                    mse += F.mse_loss(images['float'], e['target'].to(device)).item()
                    for key, pred in preds.items():
                        success = int(pred == e['label'])
                        correct[key] += success
                        per_class[key][e['label']] += success
                    predictions.append({'id': e['id'], 'label': e['label'], **preds})
            row = {**candidate, 'sha256': sha, 'first_pool': pool, 'count': len(entries),
                   'float_accuracy_percent': 100 * correct['float'] / len(entries),
                   'png_accuracy_percent': 100 * correct['png'] / len(entries),
                   'png_fake_accuracy_percent': 100 * per_class['png'][0] / sum(e['label'] == 0 for e in cached)
                   if any(e['label'] == 0 for e in cached) else None,
                   'png_alive_accuracy_percent': 100 * per_class['png'][1] / sum(e['label'] == 1 for e in cached)
                   if any(e['label'] == 1 for e in cached) else None,
                   'target_mse': mse / len(entries), 'elapsed_seconds': time.monotonic() - started}
            write_json(output / f'{candidate["family"]}_{candidate["run"]}_pool{pool}_predictions.json', predictions)
            rows.append(row)
            save_ranking(output, rows)
            print(json.dumps(row), flush=True)
            del model
    print('RESULTS', output, flush=True)


if __name__ == '__main__':
    main()
