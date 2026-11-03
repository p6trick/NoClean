"""LivDet GreenBit: audit archived pairs or train independent denoisers."""
from paths import CHECKPOINT_ROOT, loads_json
import argparse
import json
import random
import time
import warnings
from datetime import datetime
from pathlib import Path

import numpy as np
from PIL import Image, ImageFile
import torch
from torch.nn import functional as F
from torch.utils.data import DataLoader, Dataset

from models import Denoiser, grayscale
from run import PM, fingerprint, load_classifier, rectify, write_json
from suite import summary_row, write_summary


HERE = Path(__file__).resolve().parent
IMAGE_SIZE = 581
CLEAN_FOLDERS = [
    'PGD/data/2015/GreenBit/Alive_half', 'PGD/data/2015/GreenBit/Alive',
    'Self2Self/noise2noise/temporary/data/ECCV/Fake_all/fk/Fake',
    'Self2Self/noise2noise/temporary/data/ECCV/Fake_all/fk_val/Fake']
MATERIALS = {'eco', 'gelatine', 'latex', 'liq', 'rtv', 'woodglue'}
RECOVERED_IMAGES = {}
RECOVERY_REPORT_PATH = None


def canonical_id(name):
    name = name.lower()
    if name[0].isdigit():
        name = 'alive_' + name
    if name.split('_')[0] not in MATERIALS | {'alive'}:
        raise ValueError(f'Unrecognized fingerprint label prefix: {name}')
    return name


def scan_folders(root, folders, tensor_targets=False):
    result = {}
    for folder in folders:
        directory = root / folder
        if not directory.is_dir():
            raise FileNotFoundError(directory)
        patterns = ['*.png', '*.pt'] if tensor_targets else ['*.png']
        paths = [path for pattern in patterns for path in directory.glob(pattern)]
        for path in sorted(paths):
            identity = canonical_id(path.stem + '.png')
            if identity in result:
                raise ValueError(f'Duplicate fingerprint ID {identity}: {result[identity]} and {path}')
            result[identity] = str(path)
    if not result:
        raise ValueError(f'No images in {folders}')
    return result


def stratified_take(entries, count, seed):
    if not 0 <= count <= len(entries):
        raise ValueError(f'Requested {count} fingerprints from a pool of {len(entries)}')
    by_label = {label: [dict(e) for e in entries if e['label'] == label] for label in [0, 1]}
    rng = random.Random(seed)
    for group in by_label.values():
        rng.shuffle(group)
    n_alive = round(count * len(by_label[1]) / len(entries)) if entries else 0
    selected = by_label[0][:count - n_alive] + by_label[1][:n_alive]
    selected_ids = {e['id'] for e in selected}
    remaining = [dict(e) for e in entries if e['id'] not in selected_ids]
    rng.shuffle(selected)
    return selected, remaining


def resolve_groups(root, spec, args):
    clean = scan_folders(root, CLEAN_FOLDERS)
    sources = {phase: scan_folders(root, spec[phase]) for phase in ['train', 'evaluation']}
    generating = getattr(args, 'regenerate_targets', False)
    targets = {} if generating else scan_folders(root, spec['targets'], tensor_targets=True)
    overlap = sources['train'].keys() & sources['evaluation'].keys()
    if overlap:
        raise ValueError(f'LivDet train/evaluation overlap: {len(overlap)} IDs, e.g. {sorted(overlap)[:3]}')
    pools = {}
    for phase, source in sources.items():
        missing = source.keys() - (clean.keys() if generating else clean.keys() & targets.keys())
        if missing:
            raise ValueError(f'Missing clean/target for {phase}: {sorted(missing)[:5]}')
        pools[phase] = [{'id': identity, 'label': int(identity.startswith('alive_')),
                         'path': clean[identity], 'attack_path': path,
                         **({'target_path': targets[identity]} if not generating else {})}
                        for identity, path in sorted(source.items())]
    # Selection never uses classifier or denoiser performance.
    valid, fit_pool = stratified_take(pools['train'], args.validation_size, args.seed)
    train, _ = stratified_take(fit_pool, args.train_size, args.seed + 1)
    evaluation, _ = stratified_take(pools['evaluation'], args.eval_size, args.seed + 2)
    return {'train': train, 'validation': valid, 'evaluation': evaluation}


def decode_image(path, mode):
    """Match the legacy loader for recoverable streams, without skipping samples."""
    path = str(path)
    try:
        with Image.open(path) as image:
            return image.convert(mode)
    except OSError as strict_error:
        # The original notebooks enabled this globally. Scope the compatibility
        # setting to this retry and always restore it, including on failure.
        previous = ImageFile.LOAD_TRUNCATED_IMAGES
        try:
            ImageFile.LOAD_TRUNCATED_IMAGES = True
            with Image.open(path) as image:
                decoded = image.convert(mode)
        except OSError as legacy_error:
            raise OSError(f'Cannot decode image {path}; strict loader: {strict_error}; '
                          f'legacy tolerant loader: {legacy_error}') from legacy_error
        finally:
            ImageFile.LOAD_TRUNCATED_IMAGES = previous
        if path not in RECOVERED_IMAGES:
            RECOVERED_IMAGES[path] = str(strict_error)
            if RECOVERY_REPORT_PATH is not None:
                write_json(RECOVERY_REPORT_PATH, RECOVERED_IMAGES)
            warnings.warn(f'Legacy tolerant image decode used; sample retained: {path}',
                          RuntimeWarning, stacklevel=2)
        return decoded


def read_tensor(path, rgb=False):
    if Path(path).suffix.lower() == '.pt':
        record = torch.load(path, map_location='cpu', weights_only=True)
        image = record.get('image') if isinstance(record, dict) else None
        if (not isinstance(image, torch.Tensor) or image.shape != (1, IMAGE_SIZE, IMAGE_SIZE)
                or image.dtype != torch.float32 or not torch.isfinite(image).all()
                or image.min() < 0 or image.max() > 1):
            raise ValueError(f'Invalid float rectification target: {path}')
        return image.repeat(3, 1, 1) if rgb else image
    image = decode_image(path, 'RGB' if rgb else 'L')
    if image.size != (IMAGE_SIZE, IMAGE_SIZE):
        image = image.resize((IMAGE_SIZE, IMAGE_SIZE), Image.Resampling.BILINEAR)
    array = np.array(image, dtype=np.float32) / 255
    if rgb:
        return torch.from_numpy(array).permute(2, 0, 1)
    return torch.from_numpy(array).unsqueeze(0)


def regenerate_targets(model, groups, args, device, folder, classifier_sha256):
    """Use only archived attacks and labels; never inspect clean image pixels."""
    cache = folder / 'rectified_targets'
    cache.mkdir(exist_ok=False)
    settings = {'epsilon': args.rectify_epsilon, 'alpha': args.rectify_alpha,
                'maximum_iterations': args.rectify_steps, 'stop': 'correct',
                'gradient_loss': 'logits-ce', 'projection_center': 'archived attack image',
                'failure_policy': 'keep final iterate', 'storage': 'float32 grayscale tensors',
                'classifier_sha256': classifier_sha256}
    write_json(cache / 'rectification_settings.json', settings)
    rows, seen = [], set()
    summary = {'count': 0, 'correct_before': 0, 'correct_after': 0, 'zero_iterations': 0,
               'reached_iteration_limit': 0, 'unrectified': 0, 'max_linf': 0.0,
               'total_iterations': 0, 'per_split': {}}
    for split, entries in groups.items():
        split_before = split_after = split_count = split_iterations = 0
        for start in range(0, len(entries), args.rectify_batch_size):
            chunk = entries[start:start + args.rectify_batch_size]
            if (len({e['id'] for e in chunk}) != len(chunk)
                    or any(e['id'] in seen for e in chunk)):
                raise ValueError('Duplicate target ID across splits')
            source_hashes = {e['id']: fingerprint(e['attack_path']) for e in chunk}
            source = torch.stack([read_tensor(e['attack_path']) for e in chunk]).to(device)
            labels = torch.tensor([e['label'] for e in chunk], device=device)
            with torch.no_grad():
                before = model(source).argmax(1)
            # run.rectify implements early stopping independently for each sample.
            target, iterations = rectify(model, source, labels, args)
            with torch.no_grad():
                after = model(target).argmax(1)
            delta = (target - source).abs().flatten(1).max(1).values
            if not torch.isfinite(target).all() or (delta > args.rectify_epsilon + 1e-6).any():
                raise ValueError('Generated target violates finite-value or L-infinity constraint')
            for index, entry in enumerate(chunk):
                if fingerprint(entry['attack_path']) != source_hashes[entry['id']]:
                    raise ValueError(f"Attack file changed during generation: {entry['attack_path']}")
                path = cache / (Path(entry['id']).stem + '.pt')
                image = target[index].detach().cpu().clone().contiguous()
                torch.save({'image': image, 'id': entry['id'], 'label': entry['label'],
                            'source_sha256': source_hashes[entry['id']],
                            'iterations': iterations[index].item(), 'settings': settings}, path)
                entry['target_path'] = str(path.resolve())
                row = {'id': entry['id'], 'split': split, 'label': entry['label'],
                       'source_path': entry['attack_path'], 'source_sha256': source_hashes[entry['id']],
                       'target_path': entry['target_path'], 'target_sha256': fingerprint(path),
                       'prediction_before': before[index].item(), 'prediction_after': after[index].item(),
                       'iterations': iterations[index].item(), 'linf': delta[index].item()}
                rows.append(row)
                seen.add(entry['id'])
                summary['zero_iterations'] += int(row['iterations'] == 0)
                summary['reached_iteration_limit'] += int(row['iterations'] == args.rectify_steps)
                summary['unrectified'] += int(row['prediction_after'] != row['label'])
                summary['max_linf'] = max(summary['max_linf'], row['linf'])
                summary['total_iterations'] += row['iterations']
            split_before += before.eq(labels).sum().item()
            split_after += after.eq(labels).sum().item()
            split_iterations += iterations.sum().item()
            split_count += len(chunk)
            print(f'rectification {split} {split_count}/{len(entries)}', flush=True)
        summary['per_split'][split] = {'count': split_count, 'correct_before': split_before,
                                      'correct_after': split_after,
                                      'accuracy_before_percent': 100 * split_before / split_count if split_count else None,
                                      'accuracy_after_percent': 100 * split_after / split_count if split_count else None,
                                      'mean_iterations': split_iterations / split_count if split_count else None}
        summary['count'] += split_count
        summary['correct_before'] += split_before
        summary['correct_after'] += split_after
        write_json(cache / 'manifest.json', rows)
    summary['mean_iterations'] = summary['total_iterations'] / summary['count']
    write_json(cache / 'summary.json', summary)
    write_json(folder / 'manifest.json', groups)
    print('RECTIFICATION', json.dumps(summary), flush=True)
    return summary


def verify_cached_targets(groups, classifier_sha256):
    """Reject a reused float cache if its classifier, source bytes or targets changed."""
    catalogs, selected, evaluation_iterations = {}, [], []
    for split, entries in groups.items():
        for entry in entries:
            path = Path(entry['target_path'])
            if path.suffix.lower() != '.pt':
                continue
            directory = path.parent
            if directory not in catalogs:
                settings = loads_json((directory / 'rectification_settings.json').read_text())
                if settings['classifier_sha256'] != classifier_sha256:
                    raise ValueError(f'Cached target classifier mismatch: {directory}')
                rows = loads_json((directory / 'manifest.json').read_text())
                catalog = {row['id']: row for row in rows}
                if len(catalog) != len(rows):
                    raise ValueError(f'Duplicate IDs in cache manifest: {directory}')
                catalogs[directory] = (settings, catalog)
            row = catalogs[directory][1].get(entry['id'])
            if (row is None or row['label'] != entry['label']
                    or row['source_sha256'] != fingerprint(entry['attack_path'])
                    or row['target_sha256'] != fingerprint(path)):
                raise ValueError(f'Cached target does not match selected attack/label: {entry["id"]}')
            selected.append(row)
            if split == 'evaluation':
                evaluation_iterations.append(row['iterations'])
    return {'settings': [item[0] for item in catalogs.values()], 'verified_count': len(selected),
            'mean_evaluation_iterations': sum(evaluation_iterations) / len(evaluation_iterations)
            if evaluation_iterations else None} if selected else None


class PairDataset(Dataset):
    def __init__(self, entries, crop_size, training, target_key='target_path'):
        self.entries, self.crop_size, self.training = entries, crop_size, training
        self.target_key = target_key

    def __len__(self):
        return len(self.entries)

    def __getitem__(self, index):
        entry = self.entries[index]
        source, target = read_tensor(entry['attack_path'], True), read_tensor(entry[self.target_key], True)
        if self.crop_size:
            margin = IMAGE_SIZE - self.crop_size
            if self.training:
                top, left = torch.randint(margin + 1, (2,)).tolist()
            else:
                top = left = margin // 2
            source = source[:, top:top + self.crop_size, left:left + self.crop_size]
            target = target[:, top:top + self.crop_size, left:left + self.crop_size]
        return source, target


def batches(entries, key, batch_size, device, rgb=False):
    for start in range(0, len(entries), batch_size):
        chunk = entries[start:start + batch_size]
        yield torch.stack([read_tensor(e[key], rgb) for e in chunk]).to(device), torch.tensor(
            [e['label'] for e in chunk], device=device)


def classify(model, entries, key, args, device):
    correct = count = 0
    for images, labels in batches(entries, key, args.eval_batch_size, device):
        with torch.no_grad():
            correct += model(images).argmax(1).eq(labels).sum().item()
        count += len(labels)
    return {'count': count, 'correct': correct, 'accuracy_percent': 100 * correct / count}


def pair_diagnostics(entries, sample_size, args, model, device):
    selected, _ = stratified_take(entries, min(sample_size, len(entries)), args.seed + 3)
    linf, squares, count, equal = [], 0.0, 0, 0
    for start in range(0, len(selected), args.eval_batch_size):
        chunk = selected[start:start + args.eval_batch_size]
        source = torch.stack([read_tensor(e['attack_path']) for e in chunk])
        target = torch.stack([read_tensor(e['target_path']) for e in chunk])
        delta = (source - target).abs().flatten(1)
        linf.extend(delta.max(1).values.tolist())
        equal += delta.max(1).values.eq(0).sum().item()
        squares += delta.square().mean(1).sum().item()
        count += len(chunk)
    return {'training_sample_count': count, 'sample_ids': [e['id'] for e in selected],
            'identity_target_count': equal, 'mean_source_target_mse': squares / count,
            'max_source_target_linf': max(linf), 'count_linf_over_0_031': sum(v > .031 for v in linf),
            'attack_accuracy': classify(model, selected, 'attack_path', args, device),
            'target_accuracy': classify(model, selected, 'target_path', args, device),
            'note': 'Saved target provenance is unverified; distances diagnose possible version mismatch.'}


def audit_attack(model, group, spec, args, device, output):
    evaluation = group['evaluation']
    if args.audit_eval_size:
        evaluation, _ = stratified_take(evaluation, min(args.audit_eval_size, len(evaluation)), args.seed + 4)
    record = {'paths': spec,
              'clean': classify(model, evaluation, 'path', args, device),
              'attacked': classify(model, evaluation, 'attack_path', args, device),
              'training_pairs': pair_diagnostics(group['train'], args.audit_sample_size, args, model, device),
              'partial_evaluation': len(evaluation) != len(group['evaluation']),
              'paper_attacked': spec['paper_attacked']}
    record['baseline_delta_pp'] = record['attacked']['accuracy_percent'] - spec['paper_attacked']
    write_json(output / 'audit.json', record)
    return record


def make_scheduler(optimizer, args):
    if getattr(args, 'scheduler', 'none') == 'none':
        return None
    patience = getattr(args, 'scheduler_patience', None)
    if patience is None:
        patience = max(1, args.epochs // 4)
    return torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode='min', patience=patience, factor=.5)


def fit(group, args, device, output, emit, initial_state=None):
    first_pool = getattr(args, 'first_pool', 2)
    model = Denoiser(first_pool=first_pool).to(device)
    if initial_state is not None:
        model.load_state_dict(initial_state, strict=True)
    optimizer = torch.optim.Adam(model.parameters(), lr=args.learning_rate, betas=(.9, .99), eps=1e-8)
    scheduler = make_scheduler(optimizer, args)
    generator = torch.Generator().manual_seed(args.seed)
    target_key = 'path' if getattr(args, 'training_target', 'rectified') == 'clean' else 'target_path'
    loader = DataLoader(PairDataset(group['train'], args.crop_size, True, target_key),
                        batch_size=args.batch_size, shuffle=True, num_workers=0, generator=generator)
    valid_loader = DataLoader(PairDataset(group['validation'], args.crop_size, False, target_key),
                              batch_size=args.batch_size, num_workers=0) if group['validation'] else None
    history, best, selected = [], float('inf'), None
    for epoch in range(1, args.epochs + 1):
        model.train()
        total = count = 0
        for source, target in loader:
            source, target = source.to(device), target.to(device)
            optimizer.zero_grad(set_to_none=True)
            loss = F.mse_loss(model(source), target)
            if not torch.isfinite(loss):
                raise ValueError('Nonfinite LivDet training loss')
            loss.backward()
            optimizer.step()
            total += loss.item() * len(source)
            count += len(source)
        record = {'epoch': epoch, 'train_mse': total / count,
                  'learning_rate': optimizer.param_groups[0]['lr']}
        model.eval()
        if valid_loader:
            total = count = 0
            with torch.no_grad():
                for source, target in valid_loader:
                    source, target = source.to(device), target.to(device)
                    loss = F.mse_loss(model(source), target)
                    total += loss.item() * len(source)
                    count += len(source)
            record['validation_mse'] = total / count
            save = record['validation_mse'] < best
            if save:
                best = record['validation_mse']
        else:
            save = True
        if scheduler is not None:
            scheduler.step(record.get('validation_mse', record['train_mse']))
        record['next_learning_rate'] = optimizer.param_groups[0]['lr']
        snapshot = {'state_dict': model.state_dict(), 'epoch': epoch,
                    'model_configuration': {'first_pool': first_pool},
                    'arguments': {key: str(value) if isinstance(value, Path) else value
                                  for key, value in vars(args).items()}}
        if getattr(args, 'keep_checkpoint_variants', False):
            if save:
                torch.save(snapshot, output / 'best_validation_denoiser.pt')
            if epoch == args.epochs:
                torch.save(snapshot, output / 'last_denoiser.pt')
        if save:
            selected = epoch
            torch.save(snapshot, output / 'denoiser.pt')
        history.append(record)
        write_json(output / 'training_history.json', history)
        emit(json.dumps(record))
    saved = torch.load(output / 'denoiser.pt', map_location=device, weights_only=True)
    model.load_state_dict(saved['state_dict'])
    return model.eval(), selected


def evaluate(model, denoiser, group, args, device, output, emit):
    names = ['clean', 'attacked', 'rectified_oracle', 'defended_attacked', 'defended_clean']
    correct = dict.fromkeys(names, 0)
    count, attack_linf, target_linf, predictions = 0, 0.0, 0.0, []
    for start in range(0, len(group), args.eval_batch_size):
        chunk = group[start:start + args.eval_batch_size]
        labels = torch.tensor([e['label'] for e in chunk], device=device)
        clean = torch.stack([read_tensor(e['path']) for e in chunk]).to(device)
        source_rgb = torch.stack([read_tensor(e['attack_path'], True) for e in chunk]).to(device)
        # PIL grayscale for stored inputs mirrors the legacy evaluation loader.
        source = torch.stack([read_tensor(e['attack_path']) for e in chunk]).to(device)
        target = torch.stack([read_tensor(e['target_path']) for e in chunk]).to(device)
        with torch.no_grad():
            result = denoiser(source_rgb)
            clean_result = denoiser(clean.repeat(1, 3, 1, 1))
            if not torch.isfinite(result).all() or not torch.isfinite(clean_result).all():
                raise ValueError('Nonfinite LivDet denoiser output')
            if getattr(args, 'output_processing', 'float') == 'png':
                from evaluate_legacy_livdet import classifier_input
                defended = torch.cat([classifier_input(value[None], True) for value in result])
                defended_clean = torch.cat([classifier_input(value[None], True) for value in clean_result])
            else:
                defended = grayscale(result)
                defended_clean = grayscale(clean_result)
            images = dict(zip(names, [clean, source, target, defended, defended_clean]))
            preds = {key: model(image).argmax(1) for key, image in images.items()}
            for key in names:
                correct[key] += preds[key].eq(labels).sum().item()
            for index, entry in enumerate(chunk):
                predictions.append({'id': entry['id'], 'label': entry['label'],
                                    **{key: value[index].item() for key, value in preds.items()}})
        attack_linf = max(attack_linf, (source - clean).abs().max().item())
        target_linf = max(target_linf, (target - source).abs().max().item())
        if start == 0:
            # Compact contact sheet: rows are examples, columns follow metric names.
            rows = []
            for index in range(min(4, len(chunk))):
                rows.append(torch.cat([images[key][index, 0] for key in names], dim=1))
            grid = (torch.cat(rows).detach().cpu().clamp(0, 1).numpy() * 255).round().astype(np.uint8)
            Image.fromarray(grid).resize((1450, 290 * len(rows))).save(output / 'examples.png')
        count += len(chunk)
        if count % max(args.eval_batch_size * 25, 1) == 0 or count == len(group):
            emit(f'evaluation {count}/{len(group)}')
    metrics = {'count': count, 'correct': correct,
               'accuracy_percent': {key: 100 * value / count for key, value in correct.items()},
               'max_attack_linf': attack_linf, 'max_rectify_linf': target_linf,
               'mean_rectify_iterations': None, 'input_mode': 'saved', 'dataset': 'livdet2015_greenbit',
               'output_processing': getattr(args, 'output_processing', 'float'),
               'attack_scope': 'archived classifier attacks; not adaptive defense evaluation',
               'rectified_oracle_note': 'saved target accuracy; historical rectification settings unverified',
               'example_columns': names}
    write_json(output / 'predictions.json', predictions)
    return metrics


def main():
    global RECOVERY_REPORT_PATH
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--attacks', nargs='+', choices=['fgsm', 'pgd', 'cw', 'ba', 'bim', 'aa'],
                        default=['fgsm', 'pgd', 'cw', 'ba', 'bim', 'aa'])
    parser.add_argument('--device', choices=['cpu', 'cuda'], default='cuda')
    parser.add_argument('--pm-root', type=Path, default=PM)
    parser.add_argument('--output', type=Path)
    parser.add_argument('--train-size', type=int, default=1125)
    parser.add_argument('--validation-size', type=int, default=125)
    parser.add_argument('--eval-size', type=int, default=1250)
    parser.add_argument('--epochs', type=int, default=30)
    parser.add_argument('--batch-size', type=int, default=4)
    parser.add_argument('--eval-batch-size', type=int, default=2)
    parser.add_argument('--crop-size', type=int, default=128)
    parser.add_argument('--first-pool', type=int, choices=[1, 2], default=2,
                        help='1 matches the active archived finger U-Net; 2 preserves previous runs')
    parser.add_argument('--training-diagnostic-size', type=int, default=0,
                        help='Classify this many fitted training images after training; 0 disables')
    parser.add_argument('--learning-rate', type=float, default=.001)
    parser.add_argument('--scheduler', choices=['none', 'plateau'], default='none')
    parser.add_argument('--scheduler-patience', type=int,
                        help='Plateau patience; default epochs//4, matching legacy code')
    parser.add_argument('--seed', type=int, default=2026)
    parser.add_argument('--threads', type=int, default=4)
    parser.add_argument('--audit-sample-size', type=int, default=32)
    parser.add_argument('--audit-eval-size', type=int, default=0, help='0 audits the entire selected evaluation split')
    parser.add_argument('--target-dir', help='Override targets for a single selected attack, relative to PM or absolute')
    parser.add_argument('--regenerate-targets', action='store_true',
                        help='Rectify archived attacks into a new float32 cache; never regenerate attacks')
    parser.add_argument('--rectify-epsilon', type=float, default=.03)
    parser.add_argument('--rectify-alpha', type=float, default=2 / 255)
    parser.add_argument('--rectify-steps', type=int, default=40)
    parser.add_argument('--rectify-batch-size', type=int, default=1)
    parser.add_argument('--hash-images', action='store_true', help='Hash all selected image bytes; can be slow on NAS')
    mode = parser.add_mutually_exclusive_group()
    mode.add_argument('--check-only', action='store_true')
    mode.add_argument('--audit-only', action='store_true')
    mode.add_argument('--targets-only', action='store_true', help='Generate new targets and stop before training')
    args = parser.parse_args()
    args.rectify_stop = 'correct'
    args.gradient_loss = 'logits-ce'
    if len(set(args.attacks)) != len(args.attacks):
        parser.error('Attack names must be unique')
    if min(args.train_size, args.eval_size, args.epochs, args.batch_size, args.eval_batch_size,
           args.threads, args.audit_sample_size) < 1 or args.validation_size < 0 or args.audit_eval_size < 0 or args.training_diagnostic_size < 0:
        parser.error('Invalid size or epoch count')
    if not 32 <= args.crop_size <= IMAGE_SIZE or args.learning_rate <= 0:
        parser.error('crop-size must be between 32 and 581; learning rate must be positive')
    if args.scheduler_patience is not None and args.scheduler_patience < 0:
        parser.error('scheduler-patience must be nonnegative')
    if min(args.rectify_epsilon, args.rectify_alpha, args.rectify_steps, args.rectify_batch_size) <= 0:
        parser.error('Rectification parameters must be positive')
    if args.regenerate_targets and args.target_dir:
        parser.error('Use either --regenerate-targets or --target-dir, not both')
    if args.targets_only and not args.regenerate_targets:
        parser.error('--targets-only requires --regenerate-targets')
    if args.target_dir and len(args.attacks) != 1:
        parser.error('--target-dir requires exactly one attack')
    specs = loads_json((HERE / 'livdet_experiments.json').read_text())
    if args.target_dir:
        specs[args.attacks[0]]['targets'] = [args.target_dir]
    groups = {name: resolve_groups(args.pm_root, specs[name], args) for name in args.attacks}
    for name, group in groups.items():
        print('VERIFIED', name.upper(), {key: {'count': len(entries),
              'alive': sum(e['label'] == 1 for e in entries), 'fake': sum(e['label'] == 0 for e in entries)}
              for key, entries in group.items()}, flush=True)
    if args.check_only:
        return
    if args.device == 'cuda' and not torch.cuda.is_available():
        raise SystemExit('CUDA unavailable. Activate conda n2n and use your GPU terminal.')
    torch.set_num_threads(args.threads)
    torch.use_deterministic_algorithms(True)
    torch.backends.cudnn.benchmark = False
    device = torch.device(args.device)
    output = args.output or HERE / 'runs' / ('livdet_suite_' + datetime.now().strftime('%Y%m%d_%H%M%S_%f'))
    output.mkdir(parents=True, exist_ok=False)
    RECOVERED_IMAGES.clear()
    RECOVERY_REPORT_PATH = output / 'image_load_recoveries.json'
    write_json(RECOVERY_REPORT_PATH, {})
    classifier_path = CHECKPOINT_ROOT / 'livdet/classifier.pth'
    arguments = {key: str(value) if isinstance(value, Path) else value for key, value in vars(args).items()}
    configuration = {'arguments': arguments, 'classifier_sha256': fingerprint(classifier_path),
                     'registry_sha256': fingerprint(HERE / 'livdet_experiments.json'),
                     'torch': str(torch.__version__), 'cuda_runtime': torch.version.cuda,
                     'gpu': torch.cuda.get_device_name(device) if device.type == 'cuda' else None,
                     'protocol': 'archive train/eval split; class-stratified validation from train only',
                     'image_processing': 'PIL bilinear resize to 581; RGB paired patches for training; full image evaluation',
                     'image_loading_policy': 'strict first, legacy LOAD_TRUNCATED_IMAGES retry on OSError; retained and logged; unrecoverable files raise with path',
                     'targets': ('regenerated from archived attacks; true labels, eps/step/max-iterations recorded'
                                 if args.regenerate_targets else
                                 'archived rectified images; no clean training targets; final-paper version unverified'),
                     'rectification_settings': {'epsilon': args.rectify_epsilon, 'alpha': args.rectify_alpha,
                                                'maximum_iterations': args.rectify_steps, 'stop': 'correct',
                                                'gradient_loss': 'logits-ce'}
                     if args.regenerate_targets else None,
                     'model': f'legacy shared U-Net; first pooling stride {args.first_pool}',
                     'scheduler_effective_patience': (args.scheduler_patience if args.scheduler_patience is not None
                                                     else max(1, args.epochs // 4))
                     if args.scheduler == 'plateau' else None,
                     'image_hashes_recorded': args.hash_images,
                     'experiments': {name: specs[name] for name in args.attacks}}
    write_json(output / 'suite_config.json', configuration)
    classifier = load_classifier(classifier_path, device)
    cache_info = {}
    if not args.regenerate_targets:
        cache_info = {name: verify_cached_targets(group, configuration['classifier_sha256'])
                      for name, group in groups.items()}
        if any(cache_info.values()):
            configuration['targets'] = 'reused float targets; classifier, attack bytes, labels and target hashes verified'
            configuration['verified_float_target_caches'] = cache_info
            write_json(output / 'suite_config.json', configuration)
    rows, audits = [], {}
    for name in args.attacks:
        folder = output / name
        folder.mkdir()
        target_summary = None
        if args.regenerate_targets:
            write_json(folder / 'source_manifest.json', groups[name])
            target_summary = regenerate_targets(classifier, groups[name], args, device, folder,
                                                configuration['classifier_sha256'])
        write_json(folder / 'manifest.json', groups[name])
        if args.hash_images:
            hashes = {e['id']: {key: fingerprint(e[key]) for key in ['path', 'attack_path', 'target_path']}
                      for entries in groups[name].values() for e in entries}
            write_json(folder / 'image_sha256.json', hashes)
        if args.targets_only:
            continue
        if args.audit_only:
            audits[name] = audit_attack(classifier, groups[name], specs[name], args, device, folder)
            write_json(output / 'audit_summary.json', audits)
            print('AUDIT', name.upper(), json.dumps(audits[name]), flush=True)
            continue
        random.seed(args.seed)
        np.random.seed(args.seed)
        torch.manual_seed(args.seed)
        started = time.monotonic()
        with (output / f'{name}.log').open('w') as stream:
            def emit(message):
                print(f'[{name.upper()}] {message}', flush=True)
                stream.write(message + '\n')
                stream.flush()
            denoiser, epoch = fit(groups[name], args, device, folder, emit)
            metrics = evaluate(classifier, denoiser, groups[name]['evaluation'], args, device, folder, emit)
            if target_summary is not None:
                metrics['target_mode'] = 'regenerated_from_archived_attacks'
                metrics['mean_rectify_iterations'] = target_summary['per_split']['evaluation']['mean_iterations']
                metrics['rectified_oracle_note'] = 'new true-label targets; diagnostic only; not used by defended inference'
            elif cache_info.get(name):
                metrics['target_mode'] = 'verified_float_cache'
                metrics['mean_rectify_iterations'] = cache_info[name]['mean_evaluation_iterations']
                metrics['rectified_oracle_note'] = 'verified cached true-label targets; diagnostic only; not used by defended inference'
            metrics.update(attack=name, selected_epoch=epoch, elapsed_seconds=time.monotonic() - started)
            write_json(folder / 'metrics.json', metrics)
            emit(json.dumps(metrics))
            if args.training_diagnostic_size:
                diagnostic_folder = folder / 'training_diagnostic'
                diagnostic_folder.mkdir()
                selected, _ = stratified_take(groups[name]['train'],
                    min(args.training_diagnostic_size, len(groups[name]['train'])), args.seed + 5)
                write_json(diagnostic_folder / 'manifest.json', selected)
                diagnostic = evaluate(classifier, denoiser, selected, args, device,
                                      diagnostic_folder, emit)
                diagnostic.update(attack=name, selected_epoch=epoch, split='fitted_training_subset',
                                  target_mode=metrics.get('target_mode', 'archived'),
                                  rectified_oracle_note=metrics['rectified_oracle_note'],
                                  note='Diagnostic on fitted training IDs; not held-out performance. '
                                       'Full-image classification can differ from patch-level fit.')
                write_json(diagnostic_folder / 'metrics.json', diagnostic)
                emit('TRAINING_DIAGNOSTIC ' + json.dumps(diagnostic))
        rows.append(summary_row(name, specs[name], metrics, folder))
        write_summary(output, rows)
        del denoiser
    print('RESULTS', output, flush=True)


if __name__ == '__main__':
    main()
