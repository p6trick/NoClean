"""Validate, audit or run independent MNIST denoisers for archived attacks."""
from paths import CHECKPOINT_ROOT, loads_json
import argparse
import csv
import json
import subprocess
import sys
from datetime import datetime
from pathlib import Path
from types import SimpleNamespace

import torch

from run import PM, attach_saved_paths, image_batches, index_images, load_classifier, split_entries, write_json


HERE = Path(__file__).resolve().parent


def load_experiments(path=HERE / 'saved_experiments.json'):
    return loads_json(Path(path).read_text())


def resolve_groups(entries, spec, args):
    settings = SimpleNamespace(seed=args.seed, fit_half='last', train_size=args.train_size,
                               validation_size=args.validation_size, eval_size=args.eval_size,
                               saved_train_attacks=str(PM / spec['train_attacks']),
                               saved_eval_attacks=str(PM / spec['eval_attacks']),
                               saved_targets=str(PM / spec['targets']))
    groups = split_entries(entries, settings)
    return attach_saved_paths(groups, settings)


def classify_entries(model, entries, key, device, batch_size):
    correct, count = 0, 0
    for images, labels in image_batches(entries, batch_size, device, key):
        with torch.no_grad():
            correct += model(images).argmax(1).eq(labels).sum().item()
        count += len(labels)
    return {'count': count, 'correct': correct, 'accuracy_percent': 100 * correct / count}


def audit(attacks, specs, groups, args, output):
    device = torch.device(args.device)
    model = load_classifier(CHECKPOINT_ROOT / 'mnist/classifier.pth', device)
    report = {'note': 'Read-only audit; targets assessed on training IDs, not chosen on evaluation scores.',
              'experiments': {}}
    clean = classify_entries(model, groups[attacks[0]]['evaluation'], 'path', device, args.batch_size)
    report['clean'] = clean
    print('CLEAN', clean, flush=True)
    for attack in attacks:
        group = groups[attack]
        sample = group['train'][:args.audit_sample_size]
        result = {
            'paths': specs[attack],
            'evaluation_attack': classify_entries(model, group['evaluation'], 'attack_path', device, args.batch_size),
            'training_attack_sample': classify_entries(model, sample, 'attack_path', device, args.batch_size),
            'training_target_sample': classify_entries(model, sample, 'target_path', device, args.batch_size),
        }
        result['baseline_delta_pp'] = result['evaluation_attack']['accuracy_percent'] - specs[attack]['paper_attacked']
        report['experiments'][attack] = result
        write_json(output / 'audit.json', report)
        print(attack.upper(), json.dumps(result), flush=True)
    return report


def summary_row(attack, spec, metrics, run_path):
    accuracy = metrics['accuracy_percent']
    return {'attack': attack.upper(), 'count': metrics['count'],
            'clean': accuracy['clean'], 'attacked': accuracy['attacked'],
            'paper_attacked': spec['paper_attacked'],
            'defended_attacked': accuracy['defended_attacked'],
            'paper_defended': spec['paper_defended'],
            'defended_delta_pp': round(accuracy['defended_attacked'] - spec['paper_defended'], 4),
            'defended_clean': accuracy['defended_clean'],
            'paper_defended_clean': spec['paper_defended_clean'],
            'selected_epoch': metrics['selected_epoch'], 'run': str(run_path)}


def write_summary(output, rows):
    write_json(output / 'summary.json', rows)
    with (output / 'summary.csv').open('w', newline='') as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--attacks', nargs='+', choices=['fgsm', 'pgd', 'cw', 'ba', 'bim', 'aa'],
                        default=['pgd', 'cw', 'ba', 'bim', 'aa'])
    parser.add_argument('--device', choices=['cpu', 'cuda'], default='cuda')
    parser.add_argument('--output', default=None)
    parser.add_argument('--train-size', type=int, default=4500)
    parser.add_argument('--validation-size', type=int, default=500)
    parser.add_argument('--eval-size', type=int, default=5000)
    parser.add_argument('--epochs', type=int, default=30)
    parser.add_argument('--batch-size', type=int, default=32)
    parser.add_argument('--seed', type=int, default=2026)
    parser.add_argument('--threads', type=int, default=4)
    parser.add_argument('--audit-sample-size', type=int, default=256)
    mode = parser.add_mutually_exclusive_group()
    mode.add_argument('--check-only', action='store_true')
    mode.add_argument('--audit-only', action='store_true')
    args = parser.parse_args()
    if len(set(args.attacks)) != len(args.attacks):
        parser.error('Attack names must be unique')
    if min(args.epochs, args.batch_size, args.threads, args.audit_sample_size) < 1:
        parser.error('Epochs, batch size, threads and audit sample size must be positive')
    if args.validation_size < 0:
        parser.error('Validation size must be nonnegative')
    specs = load_experiments()
    entries = index_images(PM / 'PGD/MNIST/data', PM / 'FGSM/image_names.txt')
    # Check every selected input/target for every requested attack before training.
    groups = {name: resolve_groups(entries, specs[name], args) for name in args.attacks}
    for name, group in groups.items():
        print('VERIFIED', name.upper(), {k: len(v) for k, v in group.items()}, flush=True)
    if args.check_only:
        return
    if args.device == 'cuda' and not torch.cuda.is_available():
        raise SystemExit('CUDA unavailable. Activate conda n2n and run on a GPU-accessible terminal.')
    torch.set_num_threads(args.threads)
    output = Path(args.output) if args.output else HERE / 'runs' / (
        'mnist_suite_' + datetime.now().strftime('%Y%m%d_%H%M%S_%f'))
    output.mkdir(parents=True, exist_ok=False)
    write_json(output / 'suite_config.json', {'arguments': vars(args), 'experiments':
               {name: specs[name] for name in args.attacks},
               'protocol': 'independent model per attack; last-half fit, first-half evaluation',
               'provenance': 'historical generation parameters and final-paper target version unverified'})
    if args.audit_only:
        audit(args.attacks, specs, groups, args, output)
        return
    rows = []
    for name in args.attacks:
        spec = specs[name]
        run_path = output / name
        command = [sys.executable, '-u', str(HERE / 'run.py'), '--output', str(run_path),
                   '--device', args.device, '--input-mode', 'saved', '--fit-half', 'last',
                   '--attack', name, '--saved-train-attacks', str(PM / spec['train_attacks']),
                   '--saved-eval-attacks', str(PM / spec['eval_attacks']),
                   '--saved-targets', str(PM / spec['targets']),
                   '--train-size', str(args.train_size), '--validation-size', str(args.validation_size),
                   '--eval-size', str(args.eval_size), '--epochs', str(args.epochs),
                   '--batch-size', str(args.batch_size), '--seed', str(args.seed),
                   '--threads', str(args.threads)]
        write_json(output / f'{name}_command.json', command)
        with (output / f'{name}.log').open('w') as log:
            process = subprocess.Popen(command, stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
                                       text=True, bufsize=1)
            try:
                for line in process.stdout:
                    print(line, end='', flush=True)
                    log.write(line)
                    log.flush()
                status = process.wait()
            except BaseException:
                process.terminate()
                process.wait()
                raise
        if status:
            raise SystemExit(f'{name.upper()} failed (exit {status}); see {output / (name + ".log")}')
        metrics = loads_json((run_path / 'metrics.json').read_text())
        rows.append(summary_row(name, spec, metrics, run_path))
        write_summary(output, rows)
        print('RESULT', json.dumps(rows[-1]), flush=True)
    print('SUMMARY', output / 'summary.csv', flush=True)


if __name__ == '__main__':
    main()
