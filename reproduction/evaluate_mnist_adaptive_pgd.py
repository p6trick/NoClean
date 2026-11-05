"""Non-adaptive and end-to-end adaptive L-infinity PGD on fixed MNIST models."""
import argparse
import csv
from datetime import datetime
import json
import math
import os
from pathlib import Path
import time

os.environ.setdefault('CUBLAS_WORKSPACE_CONFIG', ':4096:8')
import torch
from torch import nn
from torch.nn import functional as F

from models import Denoiser, grayscale
from paths import CHECKPOINT_ROOT, loads_json
from run import fingerprint, image_batches, load_classifier, project, write_json


class DefendedMNIST(nn.Module):
    """Same float forward as MNIST evaluation, with an intact input gradient."""
    def __init__(self, classifier, denoiser):
        super().__init__()
        self.classifier, self.denoiser = classifier, denoiser

    def forward(self, images):
        return self.classifier(grayscale(self.denoiser(images.repeat(1, 3, 1, 1))))


def candidate_score(model, images, labels):
    with torch.no_grad():
        logits = model(images)
        if not torch.isfinite(logits).all():
            raise ValueError('Nonfinite attack logits')
        return F.cross_entropy(logits, labels, reduction='none'), logits.argmax(1).ne(labels)


def pgd_candidates(model, clean, labels, starts, epsilon, alpha, steps):
    """Retain any successful iterate; otherwise largest CE. All budgets center on clean."""
    best = clean.detach().clone()
    scores, successes = candidate_score(model, best, labels)
    diagnostics = {'gradient_calls': 0, 'zero_gradient_samples': 0, 'max_gradient_linf': 0.0}

    def retain(candidate):
        nonlocal best, scores, successes
        values, failed = candidate_score(model, candidate, labels)
        take = (failed & ~successes) | ((failed == successes) & (values > scores))
        best = torch.where(take[:, None, None, None], candidate.detach(), best)
        scores = torch.where(take, values, scores)
        successes = torch.where(take, failed, successes)

    for start in starts:
        value = project(start.detach(), clean, epsilon)
        retain(value)
        for _ in range(steps):
            value = value.detach().requires_grad_(True)
            logits = model(value)
            loss = F.cross_entropy(logits, labels)
            gradient, = torch.autograd.grad(loss, value)
            if not torch.isfinite(gradient).all():
                raise ValueError('Nonfinite input gradient')
            magnitudes = gradient.detach().abs().flatten(1).amax(1)
            diagnostics['gradient_calls'] += 1
            diagnostics['zero_gradient_samples'] += magnitudes.eq(0).sum().item()
            diagnostics['max_gradient_linf'] = max(diagnostics['max_gradient_linf'], magnitudes.max().item())
            value = project(value.detach() + alpha * gradient.sign(), clean, epsilon).detach()
            retain(value)
    delta = (best-clean).abs().max().item()
    if delta > epsilon + 1e-6 or best.min() < 0 or best.max() > 1:
        raise ValueError('PGD candidate exceeds threat-model constraints')
    return best, diagnostics


def load_denoiser(path, device):
    saved = torch.load(path, map_location='cpu', weights_only=True)
    state = saved.get('state_dict', saved)
    config = saved.get('config', {})
    if config.get('training_target') == 'clean' and config.get('attack', 'pgd') != 'pgd':
        raise ValueError('Clean-target checkpoint was trained on another attack')
    if config.get('fit_half', 'last') != 'last':
        raise ValueError('Checkpoint trained on the incompatible MNIST half')
    model = Denoiser(first_pool=1)
    model.load_state_dict(state, strict=True)
    return model.to(device).eval().requires_grad_(False), {
        'path': str(path), 'sha256': fingerprint(path), 'epoch': saved.get('epoch'),
        'training_config': config}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--checkpoint-root', type=Path, default=CHECKPOINT_ROOT)
    parser.add_argument('--rectified-checkpoint', type=Path,
                        help='Default: collected MNIST PGD denoiser, with manifest hash verification')
    parser.add_argument('--clean-checkpoint', type=Path, help='Optional newly trained PGD clean-target checkpoint')
    parser.add_argument('--device', choices=['cuda', 'cpu'], default='cuda')
    parser.add_argument('--epsilon', type=float, default=.4)
    parser.add_argument('--alpha', type=float, default=.01)
    parser.add_argument('--steps', type=int, default=40)
    parser.add_argument('--random-start', action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument('--restarts', type=int, default=1)
    parser.add_argument('--seed', type=int, default=2026)
    parser.add_argument('--eval-size', type=int, default=5000)
    parser.add_argument('--batch-size', type=int, default=32)
    parser.add_argument('--threads', type=int, default=4)
    parser.add_argument('--check-only', action='store_true')
    parser.add_argument('--output', type=Path)
    args = parser.parse_args()
    if (not all(math.isfinite(v) and v > 0 for v in [args.epsilon, args.alpha]) or
            min(args.steps, args.restarts, args.batch_size, args.threads) < 1 or
            not 1 <= args.eval_size <= 5000 or not 0 <= args.seed < 2**32):
        parser.error('Invalid attack parameters, sizes or seed')
    if not args.random_start and args.restarts != 1:
        parser.error('Multiple identical restarts without random start are redundant; use --restarts 1')
    manifest_path = args.checkpoint_root / 'mnist/pgd/source_data_manifest.json'
    manifest = loads_json(manifest_path.read_text())
    entries = manifest['evaluation'][:args.eval_size]
    all_ids = [e['id'] for split in ['train', 'validation', 'evaluation'] for e in manifest[split]]
    if len(entries) != args.eval_size or len(all_ids) != len(set(all_ids)):
        raise ValueError('Insufficient evaluation images or overlapping manifest splits')
    for entry in entries:
        if not Path(entry['path']).is_file():
            raise FileNotFoundError(entry['path'])
    collection = loads_json((args.checkpoint_root / 'manifest.json').read_text())
    classifier_path = args.checkpoint_root / 'mnist/classifier.pth'
    expected = next(r['sha256'] for r in collection['copied_files'] if r['copy'] == 'mnist/classifier.pth')
    if fingerprint(classifier_path) != expected:
        raise ValueError('Classifier hash mismatch')
    rectified_path = args.rectified_checkpoint or args.checkpoint_root / 'mnist/pgd/denoiser.pt'
    checkpoints = {'rectified': rectified_path}
    if args.clean_checkpoint:
        checkpoints = {'clean': args.clean_checkpoint, **checkpoints}
    for target, path in checkpoints.items():
        if not path.is_file():
            raise FileNotFoundError(path)
        if target == 'rectified' and args.rectified_checkpoint is None:
            recorded = next(r for r in collection['models'] if r['dataset'] == 'mnist' and r['attack'] == 'pgd')
            if fingerprint(path) != recorded['sha256']:
                raise ValueError('Collected rectified PGD checkpoint hash mismatch')
    print('VERIFIED', len(entries), 'evaluation clean images;', 'models:', ','.join(checkpoints), flush=True)
    if args.check_only:
        return
    if args.device == 'cuda' and not torch.cuda.is_available():
        parser.error('CUDA unavailable; use a GPU terminal or --device cpu for a small smoke test')
    torch.set_num_threads(args.threads)
    torch.manual_seed(args.seed)
    torch.use_deterministic_algorithms(True)
    torch.backends.cudnn.benchmark = False
    device = torch.device(args.device)
    output = args.output or Path(__file__).parent / 'runs' / datetime.now().strftime('mnist_adaptive_pgd_%Y%m%d_%H%M%S_%f')
    output.mkdir(parents=True, exist_ok=False)
    classifier = load_classifier(classifier_path, device)
    models, provenance = {}, {}
    for target, path in checkpoints.items():
        denoiser, info = load_denoiser(path, device)
        if target == 'clean' and info['training_config'].get('training_target') != 'clean':
            raise ValueError('--clean-checkpoint must identify a clean-target trained checkpoint')
        if target == 'rectified' and info['training_config'].get('training_target') == 'clean':
            raise ValueError('Rectified model is marked clean-target')
        models[target] = DefendedMNIST(classifier, denoiser).eval().requires_grad_(False)
        provenance[target] = info
    write_json(output / 'config.json', {
        'arguments': {key: str(v) if isinstance(v, Path) else v for key, v in vars(args).items()},
        'classifier_sha256': expected, 'models': provenance, 'torch': str(torch.__version__),
        'gpu': torch.cuda.get_device_name(device) if device.type == 'cuda' else None,
        'threat_model': 'Untargeted L-infinity, clean-centered projection and [0,1] clamp',
        'loss': 'Cross entropy on classifier logits; frozen models, intact input gradients',
        'non_adaptive': 'Attack classifier only, then evaluate denoiser+classifier',
        'adaptive': 'Attack denoiser+classifier directly; exact float forward/backward; no BPDA needed',
        'candidate_selection': 'Any successful iterate/start retained, otherwise largest CE; include clean',
        'reporting': 'Adaptive combined column unions clean, non-adaptive and adaptive candidates per image; raw adaptive also reported',
        'starts': 'Same per-batch seeded random initial perturbations for all attack objectives/models',
        'partial_evaluation': args.eval_size != 5000,
        'reference': {'paper_nonadaptive_pgd_percent': 94.26, 'archived_reevaluation_percent': 91.54},
        'limitations': ['Requested PGD .4/.01/40/random-start is explicit; exact archived generation settings unverified.',
                       'Fresh float attacks may differ from stored PNG attacks; reference numbers are diagnostic only.',
                       'Clean and rectified checkpoints may have different historical training provenance.',
                       'Single CE PGD configuration is not a certificate of robustness.']})
    write_json(output / 'manifest.json', entries)
    write_json(output / 'clean_image_hashes.json', {e['id']: fingerprint(e['path']) for e in entries})
    counts = {target: dict.fromkeys(['clean_classifier', 'clean_defended', 'nonadaptive_classifier',
                  'nonadaptive_defended', 'adaptive_only', 'adaptive_combined'], 0) for target in models}
    diagnostics = {target: {'zero_gradient_samples': 0, 'gradient_calls': 0,
                           'max_gradient_linf': 0.0} for target in ['nonadaptive', *models]}
    predictions, maxima = [], {'nonadaptive_linf': 0.0, 'adaptive_linf': 0.0}
    started = time.monotonic()
    for batch_index, (clean, labels) in enumerate(image_batches(entries, args.batch_size, device)):
        generator = torch.Generator(device=device).manual_seed(args.seed + batch_index)
        starts = [project(clean + torch.empty_like(clean).uniform_(-args.epsilon, args.epsilon,
                    generator=generator), clean, args.epsilon) if args.random_start else clean.clone()
                  for _ in range(args.restarts)]
        nonadaptive, diag = pgd_candidates(classifier, clean, labels, starts, args.epsilon, args.alpha, args.steps)
        maxima['nonadaptive_linf'] = max(maxima['nonadaptive_linf'], (nonadaptive-clean).abs().max().item())
        def merge(name, values):
            for key, value in values.items():
                diagnostics[name][key] = max(diagnostics[name][key], value) if key.startswith('max_') else diagnostics[name][key]+value
        merge('nonadaptive', diag)
        with torch.no_grad():
            raw_clean = classifier(clean).argmax(1)
            raw_nonadaptive = classifier(nonadaptive).argmax(1)
        batch_records = [{'id': entries[batch_index*args.batch_size+i]['id'], 'label': labels[i].item(),
                          'clean_classifier': raw_clean[i].item(), 'nonadaptive_classifier': raw_nonadaptive[i].item()}
                         for i in range(len(labels))]
        for target, model in models.items():
            adaptive, diag = pgd_candidates(model, clean, labels, starts, args.epsilon, args.alpha, args.steps)
            merge(target, diag)
            maxima['adaptive_linf'] = max(maxima['adaptive_linf'], (adaptive-clean).abs().max().item())
            with torch.no_grad():
                defended_clean = model(clean).argmax(1)
                defended_nonadaptive = model(nonadaptive).argmax(1)
                defended_adaptive = model(adaptive).argmax(1)
            correct_clean = defended_clean.eq(labels)
            correct_nonadaptive = defended_nonadaptive.eq(labels)
            correct_adaptive = defended_adaptive.eq(labels)
            values = {'clean_classifier': raw_clean.eq(labels), 'clean_defended': correct_clean,
                      'nonadaptive_classifier': raw_nonadaptive.eq(labels),
                      'nonadaptive_defended': correct_nonadaptive, 'adaptive_only': correct_adaptive,
                      'adaptive_combined': correct_clean & correct_nonadaptive & correct_adaptive}
            for key, value in values.items():
                counts[target][key] += value.sum().item()
            for i, record in enumerate(batch_records):
                record[target] = {'clean': defended_clean[i].item(), 'nonadaptive': defended_nonadaptive[i].item(),
                                 'adaptive': defended_adaptive[i].item(),
                                 'combined_correct': bool(values['adaptive_combined'][i])}
        predictions.extend(batch_records)
        print('EVALUATED', len(predictions), '/', len(entries), flush=True)
    rows = []
    for target, count in counts.items():
        accuracy = {key:100*value/len(entries) for key,value in count.items()}
        rows.append({'supervision': target, 'count': len(entries),
                     'clean_defended_percent': accuracy['clean_defended'],
                     'nonadaptive_pgd_percent': accuracy['nonadaptive_defended'],
                     'adaptive_pgd_only_percent': accuracy['adaptive_only'],
                     'adaptive_pgd_combined_percent': accuracy['adaptive_combined'],
                     'classifier_nonadaptive_percent': accuracy['nonadaptive_classifier'],
                     'adaptive_combined_minus_nonadaptive_pp': accuracy['adaptive_combined']-accuracy['nonadaptive_defended']})
    with (output / 'summary.csv').open('w', newline='') as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0])); writer.writeheader(); writer.writerows(rows)
    write_json(output / 'metrics.json', {'count':len(entries), 'correct':counts, 'summary':rows,
               'diagnostics':diagnostics, 'max_perturbations':maxima,
               'elapsed_seconds':time.monotonic()-started, 'partial_evaluation':args.eval_size != 5000})
    write_json(output / 'predictions.json', predictions)
    print('SUMMARY', json.dumps(rows), flush=True)
    print('RESULTS', output, flush=True)


if __name__ == '__main__':
    main()
