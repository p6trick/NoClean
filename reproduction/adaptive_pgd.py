"""Shared non-adaptive and adaptive PGD evaluation for MNIST and LivDet."""
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


def png_bpda(rgb):
    """Exact legacy PNG forward; clipped weighted grayscale backward surrogate."""
    from evaluate_legacy_livdet import classifier_input
    with torch.no_grad():
        exact = torch.cat([classifier_input(value[None], True) for value in rgb])
    surrogate = grayscale(rgb.clamp(0, 1))
    # Parentheses keep the forward exactly equal to the legacy PIL output.
    return exact + (surrogate - surrogate.detach())


class DefendedLivDet(DefendedMNIST):
    def forward(self, images):
        return self.classifier(png_bpda(self.denoiser(images.repeat(1, 3, 1, 1))))


def livdet_image_batches(entries, batch_size, device):
    from livdet import batches
    yield from batches(entries, 'path', batch_size, device)


def candidate_score(model, images, labels, loss_mode='logits'):
    with torch.no_grad():
        logits = model(images)
        if not torch.isfinite(logits).all():
            raise ValueError('Nonfinite attack logits')
        output = logits.softmax(1) if loss_mode == 'probability' else logits
        return F.cross_entropy(output, labels, reduction='none'), logits.argmax(1).ne(labels)


def pgd_candidates(model, clean, labels, starts, epsilon, alpha, steps,
                   loss_mode='logits', selection='best'):
    """Retain any successful iterate; otherwise largest CE. All budgets center on clean."""
    if loss_mode not in ['logits', 'probability'] or selection not in ['best', 'last']:
        raise ValueError('Invalid loss or candidate selection')
    if selection == 'last' and len(starts) != 1:
        raise ValueError('Last-iterate reference evaluation requires one start')
    best = clean.detach().clone()
    scores, successes = candidate_score(model, best, labels, loss_mode)
    diagnostics = {'gradient_calls': 0, 'zero_gradient_samples': 0, 'max_gradient_linf': 0.0}

    def retain(candidate):
        nonlocal best, scores, successes
        values, failed = candidate_score(model, candidate, labels, loss_mode)
        take = (failed & ~successes) | ((failed == successes) & (values > scores))
        best = torch.where(take[:, None, None, None], candidate.detach(), best)
        scores = torch.where(take, values, scores)
        successes = torch.where(take, failed, successes)

    for start in starts:
        value = project(start.detach(), clean, epsilon)
        if selection == 'best':
            retain(value)
        for _ in range(steps):
            value = value.detach().requires_grad_(True)
            logits = model(value)
            loss = F.cross_entropy(logits.softmax(1) if loss_mode == 'probability' else logits, labels)
            gradient, = torch.autograd.grad(loss, value)
            if not torch.isfinite(gradient).all():
                raise ValueError('Nonfinite input gradient')
            magnitudes = gradient.detach().abs().flatten(1).amax(1)
            diagnostics['gradient_calls'] += 1
            diagnostics['zero_gradient_samples'] += magnitudes.eq(0).sum().item()
            diagnostics['max_gradient_linf'] = max(diagnostics['max_gradient_linf'], magnitudes.max().item())
            value = project(value.detach() + alpha * gradient.sign(), clean, epsilon).detach()
            if selection == 'best':
                retain(value)
        if selection == 'last':
            best = value.detach()
    delta = (best-clean).abs().max().item()
    if delta > epsilon + 1e-6 or best.min() < 0 or best.max() > 1:
        raise ValueError('PGD candidate exceeds threat-model constraints')
    return best, diagnostics


def load_denoiser(path, device, dataset='mnist'):
    saved = torch.load(path, map_location='cpu', weights_only=True)
    state = saved.get('state_dict', saved)
    config = saved.get('config', saved.get('arguments', {}))
    if config.get('attack', 'pgd') != 'pgd':
        raise ValueError('Clean-target checkpoint was trained on another attack')
    if dataset == 'mnist' and config.get('fit_half', 'last') != 'last':
        raise ValueError('Checkpoint trained on the incompatible MNIST half')
    first_pool = 1 if dataset == 'mnist' else 2
    if saved.get('model_configuration', {}).get('first_pool', first_pool) != first_pool:
        raise ValueError('Checkpoint first_pool differs from the selected pipeline')
    model = Denoiser(first_pool=first_pool)
    model.load_state_dict(state, strict=True)
    return model.to(device).eval().requires_grad_(False), {
        'path': str(path), 'sha256': fingerprint(path), 'epoch': saved.get('epoch'),
        'training_config': config}


def main(dataset='mnist'):
    if dataset not in ['mnist', 'livdet']:
        raise ValueError(dataset)
    total = 5000 if dataset == 'mnist' else 1250
    parser = argparse.ArgumentParser(description=f'{dataset}: '+__doc__)
    parser.add_argument('--checkpoint-root', type=Path, default=CHECKPOINT_ROOT)
    parser.add_argument('--rectified-checkpoint', type=Path,
                        help='Default: collected PGD denoiser for the selected dataset; hash verified')
    parser.add_argument('--clean-checkpoint', type=Path, help='Optional newly trained PGD clean-target checkpoint')
    parser.add_argument('--clean-only', action='store_true', help='Evaluate clean-target model only; requires --clean-checkpoint')
    parser.add_argument('--device', choices=['cuda', 'cpu'], default='cuda')
    parser.add_argument('--attack-profile', choices=['standard', 'legacy-reference'], default='standard',
                        help='legacy-reference fixes LivDet .03,2/255,40,no random start,probability CE,last iterate,batch1')
    parser.add_argument('--epsilon', type=float, default=.4 if dataset == 'mnist' else .03)
    parser.add_argument('--alpha', type=float, default=.01)
    parser.add_argument('--steps', type=int, default=40)
    parser.add_argument('--random-start', action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument('--restarts', type=int, default=1)
    parser.add_argument('--seed', type=int, default=2026)
    parser.add_argument('--eval-size', type=int, default=total)
    parser.add_argument('--batch-size', type=int, default=32 if dataset == 'mnist' else 1)
    parser.add_argument('--threads', type=int, default=4)
    parser.add_argument('--check-only', action='store_true')
    parser.add_argument('--output', type=Path)
    args = parser.parse_args()
    if args.attack_profile == 'legacy-reference':
        if dataset != 'livdet':
            parser.error('legacy-reference currently supports LivDet only')
        if args.batch_size != 1:
            parser.error('legacy-reference requires --batch-size 1 to match the reference evaluation')
        args.epsilon, args.alpha, args.steps = .03, 2/255, 40
        args.random_start, args.restarts = False, 1
    loss_mode = 'probability' if args.attack_profile == 'legacy-reference' else 'logits'
    selection = 'last' if args.attack_profile == 'legacy-reference' else 'best'
    if args.clean_only and not args.clean_checkpoint:
        parser.error('--clean-only requires --clean-checkpoint')
    if (not all(math.isfinite(v) and v > 0 for v in [args.epsilon, args.alpha]) or
            min(args.steps, args.restarts, args.batch_size, args.threads) < 1 or
            not 1 <= args.eval_size <= total or not 0 <= args.seed < 2**32):
        parser.error('Invalid attack parameters, sizes or seed')
    if not args.random_start and args.restarts != 1:
        parser.error('Multiple identical restarts without random start are redundant; use --restarts 1')
    manifest_path = args.checkpoint_root / dataset / 'pgd/source_data_manifest.json'
    manifest = loads_json(manifest_path.read_text())
    entries = (manifest['evaluation'] if isinstance(manifest, dict) else manifest)[:args.eval_size]
    all_ids = ([e['id'] for split in ['train', 'validation', 'evaluation'] for e in manifest[split]]
               if isinstance(manifest, dict) else [e['id'] for e in manifest])
    if len(entries) != args.eval_size or len(all_ids) != len(set(all_ids)):
        raise ValueError('Insufficient evaluation images or overlapping manifest splits')
    for entry in entries:
        if not Path(entry['path']).is_file():
            raise FileNotFoundError(entry['path'])
    collection = loads_json((args.checkpoint_root / 'manifest.json').read_text())
    classifier_path = args.checkpoint_root / dataset / 'classifier.pth'
    expected = next(r['sha256'] for r in collection['copied_files'] if r['copy'] == f'{dataset}/classifier.pth')
    if fingerprint(classifier_path) != expected:
        raise ValueError('Classifier hash mismatch')
    rectified_path = args.rectified_checkpoint or args.checkpoint_root / dataset / 'pgd/denoiser.pt'
    checkpoints = {'rectified': rectified_path}
    if args.clean_checkpoint:
        checkpoints = {'clean': args.clean_checkpoint, **checkpoints}
    if args.clean_only:
        checkpoints = {'clean': args.clean_checkpoint}
    for target, path in checkpoints.items():
        if not path.is_file():
            raise FileNotFoundError(path)
        if target == 'rectified' and args.rectified_checkpoint is None:
            recorded = next(r for r in collection['models'] if r['dataset'] == dataset and r['attack'] == 'pgd')
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
    output = args.output or Path(__file__).parent / 'runs' / datetime.now().strftime(dataset+'_adaptive_pgd_%Y%m%d_%H%M%S_%f')
    output.mkdir(parents=True, exist_ok=False)
    classifier = load_classifier(classifier_path, device)
    models, provenance = {}, {}
    for target, path in checkpoints.items():
        denoiser, info = load_denoiser(path, device, dataset)
        if target == 'clean' and info['training_config'].get('training_target') != 'clean':
            raise ValueError('--clean-checkpoint must identify a clean-target trained checkpoint')
        if target == 'rectified' and info['training_config'].get('training_target') == 'clean':
            raise ValueError('Rectified model is marked clean-target')
        wrapper = DefendedMNIST if dataset == 'mnist' else DefendedLivDet
        models[target] = wrapper(classifier, denoiser).eval().requires_grad_(False)
        provenance[target] = info
    write_json(output / 'config.json', {
        'arguments': {key: str(v) if isinstance(v, Path) else v for key, v in vars(args).items()},
        'dataset': dataset, 'classifier_sha256': expected, 'models': provenance, 'torch': str(torch.__version__),
        'gpu': torch.cuda.get_device_name(device) if device.type == 'cuda' else None,
        'threat_model': 'Untargeted L-infinity, clean-centered projection and [0,1] clamp',
        'loss': ('CE(softmax(logits),labels); fixed legacy reference hypothesis' if loss_mode == 'probability' else
                 'Cross entropy on classifier logits; frozen models, input gradients enabled'),
        'non_adaptive': 'Attack classifier only, then evaluate denoiser+classifier',
        'adaptive': ('Attack denoiser+classifier directly; exact float forward/backward; no BPDA needed'
                     if dataset == 'mnist' else
                     'BPDA-PGD: exact RGB clamp/uint8/PIL grayscale forward; clipped weighted grayscale backward'),
        'output_processing': 'float' if dataset == 'mnist' else 'png-bpda',
        'input_domain': 'Grayscale clean input repeated to RGB inside denoiser; perturbation budget is grayscale',
        'candidate_selection': ('Last iterate only; no best/clean candidate selection' if selection == 'last' else
                                'Any successful iterate/start retained, otherwise largest CE; include clean'),
        'reporting': 'Adaptive combined column unions clean, non-adaptive and adaptive candidates per image; raw adaptive also reported',
        'starts': ('Same per-batch seeded random initial perturbations for all attack objectives/models'
                   if args.random_start else 'Clean input; no random start'),
        'partial_evaluation': args.eval_size != total,
        'reference': {'paper_nonadaptive_pgd_percent': 94.26 if dataset == 'mnist' else 91.36,
                      'archived_reevaluation_percent': 91.54 if dataset == 'mnist' else 89.76},
        'limitations': ['Requested attack settings are explicit; exact archived generation settings unverified.',
                       'Fresh float attacks may differ from stored PNG attacks; reference numbers are diagnostic only.',
                       'Clean and rectified checkpoints may have different historical training provenance.',
                       'Single CE PGD configuration is not a certificate of robustness.'] +
                      (['legacy-reference is a fixed historical hypothesis, not a verified archived generator.',
                        'Weaker legacy loss/start settings do not replace standard adaptive robustness evaluation.']
                       if args.attack_profile == 'legacy-reference' else []) +
                      (['LivDet collected PGD-labelled checkpoint training attack is unconfirmed.',
                        'LivDet backward is a BPDA surrogate, not the derivative of quantization.',
                        'Fresh grayscale attacks may differ from archived RGB attack inputs.'] if dataset == 'livdet' else [])})
    write_json(output / 'manifest.json', entries)
    write_json(output / 'clean_image_hashes.json', {e['id']: fingerprint(e['path']) for e in entries})
    counts = {target: dict.fromkeys(['clean_classifier', 'clean_defended', 'nonadaptive_classifier',
                  'nonadaptive_defended', 'adaptive_only', 'adaptive_combined'], 0) for target in models}
    diagnostics = {target: {'zero_gradient_samples': 0, 'gradient_calls': 0,
                           'max_gradient_linf': 0.0} for target in ['nonadaptive', *models]}
    predictions, maxima = [], {'nonadaptive_linf': 0.0, 'adaptive_linf': 0.0}
    started = time.monotonic()
    batch_reader = image_batches if dataset == 'mnist' else livdet_image_batches
    if dataset == 'livdet':
        import livdet
        livdet.RECOVERED_IMAGES.clear()
        livdet.RECOVERY_REPORT_PATH = output / 'image_decode_recoveries.json'
    for batch_index, (clean, labels) in enumerate(batch_reader(entries, args.batch_size, device)):
        generator = torch.Generator(device=device).manual_seed(args.seed + batch_index)
        starts = [project(clean + torch.empty_like(clean).uniform_(-args.epsilon, args.epsilon,
                    generator=generator), clean, args.epsilon) if args.random_start else clean.clone()
                  for _ in range(args.restarts)]
        nonadaptive, diag = pgd_candidates(classifier, clean, labels, starts, args.epsilon, args.alpha, args.steps,
                                          loss_mode=loss_mode, selection=selection)
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
            adaptive, diag = pgd_candidates(model, clean, labels, starts, args.epsilon, args.alpha, args.steps,
                                            loss_mode=loss_mode, selection=selection)
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
               'elapsed_seconds':time.monotonic()-started, 'partial_evaluation':args.eval_size != total})
    write_json(output / 'predictions.json', predictions)
    print('SUMMARY', json.dumps(rows), flush=True)
    print('RESULTS', output, flush=True)


if __name__ == '__main__':
    main()
