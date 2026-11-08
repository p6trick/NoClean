"""Epsilon-only sweep of the recorded MNIST legacy adaptive-PGD experiment."""
import argparse
import csv
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
import sys
import time

os.environ.setdefault('CUBLAS_WORKSPACE_CONFIG', ':4096:8')
import numpy as np
import torch

from adaptive_pgd import DefendedMNIST, load_denoiser, pgd_candidates
from epsilon_sweep_io import EPSILONS, SUPERVISIONS, condition_dir, write_aggregates, write_json
from models import grayscale
from paths import CHECKPOINT_ROOT, loads_json
from run import fingerprint, image_batches, load_classifier

HERE = Path(__file__).resolve().parent


def digest_json(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(',', ':'), allow_nan=False).encode()).hexdigest()


def read_reference(folder, size):
    config = json.loads((folder / 'config.json').read_text())
    metrics = json.loads((folder / 'metrics.json').read_text())
    manifest = json.loads((folder / 'manifest.json').read_text())
    predictions = json.loads((folder / 'predictions.json').read_text())
    hashes = json.loads((folder / 'clean_image_hashes.json').read_text())
    args = config['arguments']
    required = {'attack_profile': 'legacy-reference', 'epsilon': .4, 'alpha': 2/255,
                'steps': 40, 'random_start': False, 'restarts': 1, 'seed': 2026,
                'batch_size': 1, 'eval_size': 5000}
    if config.get('dataset') != 'mnist' or any(args.get(k) != v for k, v in required.items()):
        raise ValueError('Reference must be the recorded MNIST 2/255, no-random, probability-CE, last-iterate run')
    if set(config['models']) != set(SUPERVISIONS) or len(manifest) != 5000 or len(predictions) != 5000:
        raise ValueError('Reference must contain both models and the full 5000-image evaluation')
    identities = [(e['id'], e['label']) for e in manifest]
    if len({e[0] for e in identities}) != 5000 or identities != [(e['id'], e['label']) for e in predictions]:
        raise ValueError('Reference IDs/order/labels are inconsistent')
    expected = {}
    for target in SUPERVISIONS:
        full = sum(e[target]['adaptive'] == e['label'] for e in predictions)
        if full != metrics['correct'][target]['adaptive_only']:
            raise ValueError('Reference predictions disagree with metrics')
        if full != {'clean': 995, 'rectified': 438}[target]:
            raise ValueError('Reference is not the requested 19.90%/8.76% full-evaluation anchor')
        count = sum(e[target]['adaptive'] == e['label'] for e in predictions[:size])
        expected[target] = {'correct_count': count, 'accuracy_percent': 100*count/size,
                            'full_reference_correct_count': full,
                            'full_reference_accuracy_percent': 100*full/5000}
    fixed = {'alpha': args['alpha'], 'steps': args['steps'], 'random_start': args['random_start'],
             'restarts': args['restarts'], 'seed': args['seed'], 'batch_size': args['batch_size'],
             'threads': args['threads'], 'loss_mode': 'probability', 'selection': 'last',
             'input_range': [0, 1], 'norm': 'Linf', 'output_processing': 'float weighted grayscale then clamp'}
    source_hashes = {name: fingerprint(folder / name) for name in [
        'config.json', 'metrics.json', 'manifest.json', 'predictions.json', 'clean_image_hashes.json']}
    snapshot = {'source_run_name': folder.name, 'source_file_sha256': source_hashes,
                'full_reference_count': 5000, 'selected_count': size, 'expected': expected,
                'recorded_alpha': args['alpha'], 'requested_text_alpha_0_01_is_not_this_reference': True}
    return config, manifest[:size], predictions[:size], hashes, fixed, snapshot


def evaluate_condition(root, epsilon, target, inputs, classifier, model, config, allow_partial):
    final = condition_dir(root, epsilon, target)
    if final.exists():
        metric = json.loads((final / 'metrics.json').read_text())
        if metric['experiment_signature'] != config['experiment_signature']:
            raise ValueError(f'Completed condition signature mismatch: {final}')
        if fingerprint(final / 'images.npz') != metric['images_sha256']:
            raise ValueError(f'Completed condition image cache changed: {final}')
        if fingerprint(final / 'per_image.csv') != metric['per_image_sha256']:
            raise ValueError(f'Completed per-image records changed: {final}')
        print('REUSE_COMPLETED', epsilon, target, flush=True)
        return metric
    work = final.with_name(final.name + '.partial')
    work.mkdir(parents=True, exist_ok=True)
    fixed = config['attack']
    n = config['sample_count']
    device = next(model.denoiser.parameters()).device
    adversarial = np.empty_like(inputs['clean'])
    denoised = np.empty_like(inputs['clean'])
    pred_attack = np.empty(n, dtype=np.int64)
    pred_defended = np.empty(n, dtype=np.int64)
    pred_clean = np.empty(n, dtype=np.int64)
    pred_original = np.empty(n, dtype=np.int64)
    infs, l2s, rmses = [], [], []
    diagnostics = {'gradient_calls': 0, 'zero_gradient_samples': 0, 'max_gradient_linf': 0.0}
    minimum, maximum = 1., 0.
    started = time.monotonic()
    fields = ['id', 'label', 'epsilon', 'supervision', 'seed', 'clean_classifier_prediction',
              'clean_defended_prediction', 'adversarial_classifier_prediction', 'adaptive_prediction',
              'initially_correct', 'attack_success_from_correct', 'linf', 'l2', 'denoised_clean_rmse']
    with (work / 'per_image.csv').open('w', newline='') as stream:
        writer = csv.DictWriter(stream, fieldnames=fields)
        writer.writeheader()
        for offset in range(0, n, fixed['batch_size']):
            stop = min(offset + fixed['batch_size'], n)
            clean = torch.from_numpy(inputs['clean'][offset:stop]).to(device)
            labels = torch.from_numpy(inputs['labels'][offset:stop]).to(device)
            # This is the same no-random initial point and attack function as the reference.
            attacked, diag = pgd_candidates(model, clean, labels, [clean.clone()],
                epsilon, fixed['alpha'], fixed['steps'],
                loss_mode=fixed['loss_mode'], selection=fixed['selection'])
            for key, value in diag.items():
                diagnostics[key] = max(diagnostics[key], value) if key.startswith('max_') else diagnostics[key] + value
            if not torch.isfinite(attacked).all():
                raise ValueError('Nonfinite adversarial image')
            delta = attacked.double() - clean.double()
            linf = delta.abs().flatten(1).amax(1)
            l2 = delta.square().flatten(1).sum(1).sqrt()
            if linf.max().item() > epsilon + 1e-6 or attacked.min() < 0 or attacked.max() > 1:
                raise ValueError('Attack exceeds clean-centered epsilon or [0,1] bounds')
            with torch.no_grad():
                clean_logits = model(clean)
                raw_logits = classifier(attacked)
                original_logits = classifier(clean)
                rgb = model.denoiser(attacked.repeat(1, 3, 1, 1))
                output = grayscale(rgb)
                logits = classifier(output)
                if not all(torch.isfinite(v).all() for v in [clean_logits, raw_logits, original_logits, rgb, output, logits]):
                    raise ValueError('Nonfinite model forward')
                clean_pred, raw_pred = clean_logits.argmax(1), raw_logits.argmax(1)
                final_pred, original_pred = logits.argmax(1), original_logits.argmax(1)
                distance = (output.double()-clean.double()).square().flatten(1).mean(1).sqrt()
            adversarial[offset:stop] = attacked.cpu().numpy()
            denoised[offset:stop] = output.cpu().numpy()
            pred_attack[offset:stop] = raw_pred.cpu().numpy()
            pred_defended[offset:stop] = final_pred.cpu().numpy()
            pred_clean[offset:stop] = clean_pred.cpu().numpy()
            pred_original[offset:stop] = original_pred.cpu().numpy()
            infs.extend(linf.cpu().tolist()); l2s.extend(l2.cpu().tolist()); rmses.extend(distance.cpu().tolist())
            minimum = min(minimum, attacked.min().item())
            maximum = max(maximum, attacked.max().item())
            for i in range(stop-offset):
                index = offset+i
                correct_before = clean_pred[i].item() == labels[i].item()
                writer.writerow({'id': str(inputs['ids'][index]), 'label': labels[i].item(),
                    'epsilon': epsilon, 'supervision': target, 'seed': fixed['seed'],
                    'clean_classifier_prediction': original_pred[i].item(),
                    'clean_defended_prediction': clean_pred[i].item(),
                    'adversarial_classifier_prediction': raw_pred[i].item(),
                    'adaptive_prediction': final_pred[i].item(), 'initially_correct': correct_before,
                    'attack_success_from_correct': correct_before and final_pred[i].item() != labels[i].item(),
                    'linf': linf[i].item(), 'l2': l2[i].item(), 'denoised_clean_rmse': distance[i].item()})
            if stop % 100 == 0 or stop == n:
                stream.flush()
                print(f'epsilon={epsilon:.2f} {target}: {stop}/{n}', flush=True)
    np.savez_compressed(work / 'images.npz', adversarial=adversarial, denoised=denoised,
        pred_attack_classifier=pred_attack, pred_defended=pred_defended,
        pred_clean_defended=pred_clean, pred_original_classifier=pred_original)
    labels = inputs['labels']
    clean_correct = pred_clean == labels
    attacked_correct = pred_defended == labels
    correct = int(attacked_correct.sum())
    asr_denominator = int(clean_correct.sum())
    asr_numerator = int((clean_correct & ~attacked_correct).sum())
    expected = config['reference']['expected'][target]
    metric = {'epsilon': epsilon, 'supervision': target, 'sample_count': n,
        'correct_count': correct, 'accuracy_percent': 100*correct/n,
        'clean_defended_correct': asr_denominator, 'clean_defended_accuracy_percent': 100*asr_denominator/n,
        'clean_inclusive_robust_accuracy_percent': 100*int((clean_correct & attacked_correct).sum())/n,
        'asr_clean_correct_percent': 100*asr_numerator/asr_denominator if asr_denominator else None,
        'asr_numerator': asr_numerator, 'asr_denominator': asr_denominator,
        'all_sample_error_percent': 100*(n-correct)/n,
        'avg_linf': float(np.mean(infs)), 'max_linf': max(infs),
        'avg_l2': float(np.mean(l2s)), 'max_l2': max(l2s),
        'denoised_clean_rmse_mean': float(np.mean(rmses)),
        'input_min': minimum, 'input_max': maximum, 'constraint_violations': 0,
        'seed': fixed['seed'], 'alpha': fixed['alpha'], 'steps': fixed['steps'],
        'random_start': fixed['random_start'], 'restarts': fixed['restarts'],
        'batch_size': fixed['batch_size'], 'loss_mode': fixed['loss_mode'], 'candidate_selection': fixed['selection'],
        'classifier_sha256': config['classifier_sha256'], 'denoiser_sha256': config['models'][target]['sha256'],
        **diagnostics,
        'nonzero_input_gradient_observed': diagnostics['max_gradient_linf'] > 0,
        'reference_accuracy_difference_pp': 100*correct/n-expected['accuracy_percent'] if epsilon == .4 else None,
        'reference_prediction_mismatch_count': int((pred_defended != inputs['reference_adaptive_'+target]).sum()) if epsilon == .4 else None,
        'clean_baseline_prediction_mismatches': int((pred_clean != inputs['reference_clean_'+target]).sum()),
        'raw_clean_prediction_mismatches': int((pred_original != inputs['reference_original']).sum()),
        'partial_evaluation': allow_partial, 'elapsed_seconds': time.monotonic()-started,
        'images_sha256': fingerprint(work/'images.npz'), 'per_image_sha256': fingerprint(work/'per_image.csv'),
        'experiment_signature': config['experiment_signature']}
    write_json(work/'metrics.json', metric)
    work.rename(final)
    return metric


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--reference-run', type=Path,
        default=HERE/'runs/mnist_adaptive_pgd_20261105_172406_795278')
    parser.add_argument('--checkpoint-root', type=Path, default=CHECKPOINT_ROOT)
    parser.add_argument('--clean-checkpoint', type=Path, required=True)
    parser.add_argument('--rectified-checkpoint', type=Path)
    parser.add_argument('--epsilons', nargs='+', type=float, choices=EPSILONS, default=EPSILONS)
    parser.add_argument('--eval-size', type=int, default=5000)
    parser.add_argument('--device', choices=['cpu','cuda'], default='cuda')
    parser.add_argument('--check-only', action='store_true')
    parser.add_argument('--allow-reference-mismatch', action='store_true',
        help='Continue after a reported epsilon0.4 mismatch; never tune parameters automatically')
    output_group = parser.add_mutually_exclusive_group()
    output_group.add_argument('--output', type=Path)
    output_group.add_argument('--resume', type=Path)
    args = parser.parse_args()
    if not 1 <= args.eval_size <= 5000 or len(set(args.epsilons)) != len(args.epsilons):
        parser.error('Invalid evaluation size or duplicate epsilons')
    ref_config, ref_entries, ref_predictions, ref_hashes, fixed, snapshot = read_reference(args.reference_run, args.eval_size)
    manifest = loads_json((args.checkpoint_root/'mnist/pgd/source_data_manifest.json').read_text())
    entries = manifest['evaluation'][:args.eval_size]
    ids = [e['id'] for key in ['train','validation','evaluation'] for e in manifest[key]]
    if len(ids) != len(set(ids)) or len(entries) != args.eval_size:
        raise ValueError('Manifest fitting/evaluation split overlap or insufficient images')
    if [(e['id'],e['label']) for e in entries] != [(e['id'],e['label']) for e in ref_entries]:
        raise ValueError('Evaluation IDs/order/labels differ from the reference')
    clean_hashes = {}
    for e in entries:
        value = fingerprint(e['path'])
        if value != ref_hashes[e['id']]:
            raise ValueError(f"Clean image changed: {e['id']}")
        clean_hashes[e['id']] = value
    classifier_path = args.checkpoint_root/'mnist/classifier.pth'
    if fingerprint(classifier_path) != ref_config['classifier_sha256']:
        raise ValueError('Classifier differs from reference')
    model_paths = {'clean': args.clean_checkpoint,
                   'rectified': args.rectified_checkpoint or args.checkpoint_root/'mnist/pgd/denoiser.pt'}
    for target, path in model_paths.items():
        if fingerprint(path) != ref_config['models'][target]['sha256']:
            raise ValueError(f'{target} checkpoint differs from reference')
    print('VERIFIED reference, model hashes and clean image hashes/IDs/order:', len(entries), flush=True)
    print('FIXED_ATTACK', json.dumps(fixed), flush=True)
    print('EPSILONS', args.epsilons, flush=True)
    if args.check_only:
        return
    if args.device == 'cuda' and not torch.cuda.is_available():
        parser.error('CUDA unavailable')
    torch.set_num_threads(fixed['threads'])
    torch.manual_seed(fixed['seed'])
    torch.use_deterministic_algorithms(True)
    torch.backends.cudnn.benchmark = False
    device = torch.device(args.device)
    environment = {'python': sys.version, 'torch': str(torch.__version__), 'numpy': np.__version__,
        'cuda_runtime': torch.version.cuda, 'device': str(device),
        'gpu': torch.cuda.get_device_name(device) if device.type=='cuda' else None,
        'cudnn_version': torch.backends.cudnn.version(),
        'cublas_workspace_config': os.environ.get('CUBLAS_WORKSPACE_CONFIG'),
        'matmul_allow_tf32': torch.backends.cuda.matmul.allow_tf32,
        'cudnn_allow_tf32': torch.backends.cudnn.allow_tf32,
        'reference_torch': ref_config.get('torch'), 'reference_gpu': ref_config.get('gpu')}
    code_hashes = {name: fingerprint(HERE/name) for name in [
        'adaptive_pgd.py','models.py','run.py','mnist_epsilon_sweep.py','epsilon_sweep_io.py']}
    identity = {'epsilons': sorted(args.epsilons), 'sample_count': len(entries), 'attack': fixed,
        'classifier_sha256': ref_config['classifier_sha256'],
        'model_sha256': {k: ref_config['models'][k]['sha256'] for k in SUPERVISIONS},
        'evaluation_identity_sha256': digest_json([(e['id'],e['label'],clean_hashes[e['id']]) for e in entries]),
        'reference_files': snapshot['source_file_sha256'], 'code_sha256': code_hashes, 'environment': environment}
    signature = digest_json(identity)
    root = args.resume or args.output or HERE/'runs'/datetime.now().strftime('mnist_epsilon_sensitivity_%Y%m%d_%H%M%S_%f')
    if args.resume:
        config = json.loads((root/'config.json').read_text())
        if config['experiment_signature'] != signature:
            raise ValueError('Resume requires identical data, checkpoints, code, environment and attack settings')
    else:
        root.mkdir(parents=True, exist_ok=False)
        config = {**identity, 'experiment_signature': signature,
            'created_utc': datetime.now(timezone.utc).isoformat(),
            'reference': snapshot, 'models': ref_config['models'],
            'paths_used': {'reference_run': str(args.reference_run.resolve()),
                           'checkpoints': {k:str(v.resolve()) for k,v in model_paths.items()}},
            'partial_evaluation': len(entries)!=5000,
            'baseline_note': '19.90/8.76 came from alpha2/255, not0.01. Epsilon only varies here.',
            'resume_policy': 'Completed cells reused after SHA256 verification; incomplete .partial cell restarts from image0'}
        write_json(root/'config.json', config)
        write_json(root/'manifest.json', entries)
        write_json(root/'input_hashes.json', clean_hashes)
    # Cache exact existing grayscale preprocessing; no resizing, augmentation or quantization changes.
    clean = torch.cat([x for x,_ in image_batches(entries,fixed['batch_size'],torch.device('cpu'))]).numpy()
    inputs = {'clean': clean, 'labels': np.array([e['label'] for e in entries],dtype=np.int64),
              'ids': np.array([e['id'] for e in entries]),
              'reference_original': np.array([e['clean_classifier'] for e in ref_predictions],dtype=np.int64)}
    for target in SUPERVISIONS:
        inputs['reference_adaptive_'+target] = np.array([e[target]['adaptive'] for e in ref_predictions],dtype=np.int64)
        inputs['reference_clean_'+target] = np.array([e[target]['clean'] for e in ref_predictions],dtype=np.int64)
    if args.resume and (root/'inputs.npz').exists():
        with np.load(root/'inputs.npz',allow_pickle=False) as old:
            if set(old.files) != set(inputs) or any(not np.array_equal(old[k],v) for k,v in inputs.items()):
                raise ValueError('Resume input tensor cache mismatch')
        cache_hash = fingerprint(root/'inputs.npz')
        if (root/'input_cache.json').exists():
            if json.loads((root/'input_cache.json').read_text())['sha256'] != cache_hash:
                raise ValueError('Resume input cache hash mismatch')
        else:
            write_json(root/'input_cache.json', {'sha256': cache_hash})
    else:
        np.savez_compressed(root/'inputs.npz',**inputs)
        write_json(root/'input_cache.json', {'sha256': fingerprint(root/'inputs.npz')})
    classifier = load_classifier(classifier_path,device)
    models = {}
    for target, path in model_paths.items():
        denoiser, info = load_denoiser(path,device,'mnist')
        if target=='clean' and info['training_config'].get('training_target')!='clean':
            raise ValueError('Clean model is not marked clean-target')
        if target=='rectified' and info['training_config'].get('training_target')=='clean':
            raise ValueError('Rectified model is marked clean-target')
        models[target] = DefendedMNIST(classifier,denoiser).eval().requires_grad_(False)
        if any(parameter.requires_grad for parameter in models[target].parameters()):
            raise ValueError('Model parameters are not frozen')
    # Check the existing epsilon0.4 anchor first without changing any attack parameters.
    order = ([.4] if .4 in args.epsilons else []) + [e for e in sorted(args.epsilons) if e!=.4]
    for epsilon in order:
        for target in SUPERVISIONS:
            metric = evaluate_condition(root,epsilon,target,inputs,classifier,models[target],config,len(entries)!=5000)
            write_aggregates(root)
            print('RESULT', json.dumps(metric), flush=True)
        if epsilon==.4:
            comparisons,_ = write_aggregates(root)
            mismatch = any(r['status']=='MISMATCH' for r in comparisons)
            if mismatch and not args.allow_reference_mismatch:
                raise SystemExit('Epsilon0.4 differs from reference. Results retained: '+str(root)+
                    '; inspect reference_reproduction.csv/config.json before continuing. '
                    'Explicit continuation: repeat command with --resume RUN_PATH --allow-reference-mismatch.')
    comparisons,missing = write_aggregates(root)
    if missing:
        raise RuntimeError(f'Incomplete conditions: {missing}')
    write_json(root/'completion.json',{'completed_conditions':len(args.epsilons)*len(SUPERVISIONS),
        'reference_accuracy_match':all(r['status']=='PASS' for r in comparisons) if .4 in args.epsilons else None})
    print('RESULTS',root,flush=True)
    print('PLOT_COMMAND',f'python reproduction/analyze_mnist_epsilon_sweep.py --run {root}',flush=True)


if __name__=='__main__':
    main()
