"""MNIST adversarial/rectified-pair reproduction; no clean denoising targets."""
from paths import PM_ROOT, CHECKPOINT_ROOT
import argparse
import hashlib
import json
import os
import random
import time
from pathlib import Path

# Set before CUDA initialization for deterministic CUDA matrix multiplication.
os.environ.setdefault('CUBLAS_WORKSPACE_CONFIG', ':4096:8')

import numpy as np
from PIL import Image
import torch
from torch import nn
from torch.nn import functional as F
from torch.utils.data import DataLoader, TensorDataset

from models import Denoiser, grayscale


PM = PM_ROOT


def write_json(path, obj):
    path.write_text(json.dumps(obj, ensure_ascii=False, indent=2) + '\n')


def fingerprint(path):
    h = hashlib.sha256()
    with open(path, 'rb') as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b''):
            h.update(chunk)
    return h.hexdigest()


def load_classifier(path, device):
    # This is the user's historical full FX model, not a weights-only file.
    # Loading it requires the original onnx2torch module classes.
    model = torch.load(path, map_location='cpu', weights_only=False)
    if not hasattr(model, 'Softmax_0') or not isinstance(model.Softmax_0, nn.Softmax):
        raise ValueError('Expected an archived ONNX-converted classifier with terminal Softmax_0')
    model.Softmax_0 = nn.Identity()
    model.to(device).eval().requires_grad_(False)
    return model


def loss_for_gradient(logits, labels, mode):
    # Historical PGD code applied CE to already-softmaxed model outputs.
    if mode == 'legacy-probability-ce':
        logits = logits.softmax(dim=1)
    return F.cross_entropy(logits, labels, reduction='sum')


def project(value, origin, epsilon):
    return torch.maximum(torch.minimum(value, origin + epsilon), origin - epsilon).clamp(0, 1)


def attack(model, clean, labels, args):
    result = clean.detach().clone()
    if args.attack == 'pgd':
        result = project(result + torch.empty_like(result).uniform_(
            -args.attack_epsilon, args.attack_epsilon), clean, args.attack_epsilon)
    steps = 1 if args.attack == 'fgsm' else args.attack_steps
    alpha = args.attack_epsilon if args.attack == 'fgsm' else args.attack_alpha
    for _ in range(steps):
        result.requires_grad_(True)
        gradient, = torch.autograd.grad(loss_for_gradient(model(result), labels,
                                                         args.gradient_loss), result)
        result = project(result.detach() + alpha * gradient.sign(), clean,
                         args.attack_epsilon).detach()
    return result


def rectify(model, corrupted, labels, args):
    result = corrupted.detach().clone()
    iterations = torch.zeros(len(labels), dtype=torch.int64, device=labels.device)
    for _ in range(args.rectify_steps):
        with torch.no_grad():
            active = model(result).argmax(1).ne(labels)
        if args.rectify_stop == 'fixed':
            active = torch.ones_like(active)
        if not active.any():
            break
        # Update only misclassified examples. Projection is around the attack,
        # never around a clean image (which is unavailable during rectification).
        current = result[active].detach().requires_grad_(True)
        gradient, = torch.autograd.grad(loss_for_gradient(model(current), labels[active],
                                                         args.gradient_loss), current)
        updated = project(current.detach() - args.rectify_alpha * gradient.sign(),
                          corrupted[active], args.rectify_epsilon)
        result[active] = updated.detach()
        iterations[active] += 1
    return result.detach(), iterations


def index_images(data_root, order_file):
    entries = {}
    for label in range(10):
        directory = data_root / f'{label}_' / str(label)
        for path in directory.glob('*.png'):
            if path.name in entries:
                raise ValueError(f'Duplicate image ID: {path.name}')
            entries[path.name] = {'id': path.name, 'label': label, 'path': str(path)}
    names = order_file.read_text().splitlines()
    if len(names) != 10000 or len(set(names)) != 10000:
        raise ValueError('Expected 10000 unique canonical test IDs in order file')
    missing = set(names) - entries.keys()
    if missing:
        raise ValueError(f'Missing {len(missing)} images, for example {sorted(missing)[:3]}')
    return [entries[name] for name in names]


def split_entries(entries, args):
    # Saved FGSM folders use the last half for fitting, first half for evaluation.
    generator = random.Random(args.seed)
    # Each experiment must own its entry dicts: saved-path attachment mutates them.
    fit_pool = [dict(entry) for entry in entries[:5000]]
    evaluation = [dict(entry) for entry in entries[5000:]]
    if getattr(args, 'fit_half', 'first') == 'last':
        fit_pool, evaluation = evaluation, fit_pool
    generator.shuffle(fit_pool)
    generator.shuffle(evaluation)
    if args.train_size + args.validation_size > 5000:
        raise ValueError('train-size + validation-size must be <= 5000')
    if not 1 <= args.eval_size <= 5000 or args.train_size < 1:
        raise ValueError('Invalid train/eval size')
    groups = {'train': fit_pool[:args.train_size],
              'validation': fit_pool[args.train_size:args.train_size + args.validation_size],
              'evaluation': evaluation[:args.eval_size]}
    ids = [e['id'] for group in groups.values() for e in group]
    if len(ids) != len(set(ids)):
        raise ValueError('Split overlap detected')
    return groups


def attach_saved_paths(groups, args):
    for group_name, entries in groups.items():
        root = Path(args.saved_eval_attacks if group_name == 'evaluation'
                    else args.saved_train_attacks)
        for entry in entries:
            paths = {'attack_path': root / entry['id']}
            if args.saved_targets:
                paths['target_path'] = Path(args.saved_targets) / entry['id']
            for key, path in paths.items():
                if not path.is_file():
                    raise FileNotFoundError(f'Missing {group_name} {key}: {path}')
                entry[key] = str(path)
    return groups


def image_batches(entries, batch_size, device, path_key='path'):
    for start in range(0, len(entries), batch_size):
        chunk = entries[start:start + batch_size]
        images = []
        for entry in chunk:
            with Image.open(entry[path_key]) as image:
                array = np.array(image.convert('L'), dtype=np.float32) / 255.0
            if array.shape != (28, 28):
                raise ValueError(f"Expected 28x28 image: {entry[path_key]}: {array.shape}")
            images.append(torch.from_numpy(array).unsqueeze(0))
        yield torch.stack(images).to(device), torch.tensor(
            [e['label'] for e in chunk], device=device)


def prepare_pairs(model, entries, args, device, name):
    sources, targets, ys, counts = [], [], [], []
    saved = args.input_mode == 'saved'
    for i, (images, labels) in enumerate(image_batches(
            entries, args.batch_size, device, 'attack_path' if saved else 'path')):
        corrupted = images if saved else attack(model, images, labels, args)
        chunk = entries[i * args.batch_size:(i + 1) * args.batch_size]
        if saved and args.saved_targets:
            target, _ = next(image_batches(chunk, len(chunk), device, 'target_path'))
            iterations = torch.full_like(labels, -1)  # Historical iteration count unknown.
        else:
            target, iterations = rectify(model, corrupted, labels, args)
        if not torch.isfinite(target).all():
            raise ValueError('Nonfinite rectification target')
        sources.append(corrupted.cpu())
        targets.append(target.cpu())
        ys.append(labels.cpu())
        counts.append(iterations.cpu())
        print(f'{name} pairs {min((i + 1) * args.batch_size, len(entries))}/{len(entries)}', flush=True)
    return {'source': torch.cat(sources), 'target': torch.cat(targets),
            'labels': torch.cat(ys), 'rectify_iterations': torch.cat(counts),
            'ids': [e['id'] for e in entries]}


def loader_for_pairs(pairs, args, shuffle):
    return DataLoader(TensorDataset(pairs['source'], pairs['target']),
                      batch_size=args.batch_size, shuffle=shuffle,
                      generator=torch.Generator().manual_seed(args.seed))


def train_denoiser(train_pairs, valid_pairs, args, device, out):
    denoiser = Denoiser().to(device)
    optimizer = torch.optim.Adam(denoiser.parameters(), lr=args.learning_rate,
                                 betas=(0.9, 0.99), eps=1e-8)
    train_loader = loader_for_pairs(train_pairs, args, True)
    valid_loader = loader_for_pairs(valid_pairs, args, False) if valid_pairs else None
    history, best_loss, selected_epoch = [], float('inf'), None
    for epoch in range(1, args.epochs + 1):
        denoiser.train()
        total, count = 0.0, 0
        for source, target in train_loader:
            source, target = source.to(device).repeat(1, 3, 1, 1), target.to(device).repeat(1, 3, 1, 1)
            optimizer.zero_grad(set_to_none=True)
            loss = F.mse_loss(denoiser(source), target)
            if not torch.isfinite(loss):
                raise ValueError('Nonfinite training loss')
            loss.backward()
            optimizer.step()
            total += loss.item() * len(source)
            count += len(source)
        record = {'epoch': epoch, 'train_mse': total / count}
        denoiser.eval()
        if valid_loader:
            total, count = 0.0, 0
            with torch.no_grad():
                for source, target in valid_loader:
                    source, target = source.to(device).repeat(1, 3, 1, 1), target.to(device).repeat(1, 3, 1, 1)
                    total += F.mse_loss(denoiser(source), target).item() * len(source)
                    count += len(source)
            record['validation_mse'] = total / count
            select = record['validation_mse'] < best_loss
            if select:
                best_loss = record['validation_mse']
        else:
            select = True  # Fixed epoch training; no evaluation-set selection.
        if select:
            selected_epoch = epoch
            torch.save({'state_dict': denoiser.state_dict(), 'epoch': epoch,
                        'config': vars(args)}, out / 'denoiser.pt')
        history.append(record)
        write_json(out / 'training_history.json', history)
        print(json.dumps(record), flush=True)
    checkpoint = torch.load(out / 'denoiser.pt', map_location=device, weights_only=True)
    denoiser.load_state_dict(checkpoint['state_dict'])
    return denoiser.eval(), selected_epoch


def save_examples(out, batches):
    columns = []
    for name, tensor in batches.items():
        array = (tensor[:12, 0].detach().cpu().clamp(0, 1).numpy() * 255).round().astype(np.uint8)
        columns.append(np.concatenate(list(array), axis=0))
    grid = np.concatenate(columns, axis=1)
    Image.fromarray(grid).resize((grid.shape[1] * 3, grid.shape[0] * 3)).save(out / 'examples.png')


def evaluate(model, denoiser, entries, args, device, out):
    names = ['clean', 'attacked', 'rectified_oracle', 'defended_attacked', 'defended_clean']
    correct = dict.fromkeys(names, 0)
    count, attack_delta, rectify_delta, iteration_sum = 0, 0.0, 0.0, 0
    predictions = []
    saved = args.input_mode == 'saved'
    for i, (clean, labels) in enumerate(image_batches(entries, args.batch_size, device)):
        chunk = entries[i * args.batch_size:(i + 1) * args.batch_size]
        if saved:
            corrupted, _ = next(image_batches(chunk, len(chunk), device, 'attack_path'))
        else:
            corrupted = attack(model, clean, labels, args)
        if saved and args.saved_targets:
            rectified, _ = next(image_batches(chunk, len(chunk), device, 'target_path'))
            iterations = torch.full_like(labels, -1)
        else:
            rectified, iterations = rectify(model, corrupted, labels, args)
        with torch.no_grad():
            defended = grayscale(denoiser(corrupted.repeat(1, 3, 1, 1)))
            defended_clean = grayscale(denoiser(clean.repeat(1, 3, 1, 1)))
            values = dict(zip(names, [clean, corrupted, rectified, defended, defended_clean]))
            labels_pred = {name: model(value).argmax(1) for name, value in values.items()}
            for name, pred in labels_pred.items():
                correct[name] += pred.eq(labels).sum().item()
            for j in range(len(labels)):
                predictions.append({'id': entries[count + j]['id'], 'label': labels[j].item(),
                                    **{name: pred[j].item() for name, pred in labels_pred.items()}})
        attack_delta = max(attack_delta, (corrupted - clean).abs().max().item())
        rectify_delta = max(rectify_delta, (rectified - corrupted).abs().max().item())
        iteration_sum += iterations.sum().item()
        if i == 0:
            save_examples(out, values)
        count += len(labels)
        print(f'evaluation {count}/{len(entries)}', flush=True)
    metrics = {'count': count, 'correct': correct,
               'accuracy_percent': {name: 100 * n / count for name, n in correct.items()},
               'max_attack_linf': attack_delta, 'max_rectify_linf': rectify_delta,
               'mean_rectify_iterations': None if saved and args.saved_targets else iteration_sum / count,
               'input_mode': args.input_mode, 'attack': args.attack,
               'attack_scope': ('archived attacks; historical generation parameters unverified' if saved else
                                'classifier-only; not an adaptive attack on classifier+denoiser'),
               'rectified_oracle_note': ('archived target accuracy; historical rectification settings unverified'
                                        if saved and args.saved_targets else
                                        'uses true labels; diagnostic only, not deployable defense'),
               'example_columns': names}
    write_json(out / 'predictions.json', predictions)
    return metrics


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--data-root', default=str(PM / 'PGD/MNIST/data'))
    parser.add_argument('--order-file', default=str(PM / 'FGSM/image_names.txt'))
    parser.add_argument('--classifier', default=str(CHECKPOINT_ROOT / 'mnist/classifier.pth'))
    parser.add_argument('--input-mode', choices=['generated', 'saved'], default='generated')
    parser.add_argument('--fit-half', choices=['first', 'last'], default='first')
    parser.add_argument('--saved-train-attacks')
    parser.add_argument('--saved-eval-attacks')
    parser.add_argument('--saved-targets', help='Optional existing target folder; must cover all selected IDs')
    parser.add_argument('--output', required=True)
    parser.add_argument('--seed', type=int, default=2026)
    parser.add_argument('--device', choices=['auto', 'cpu', 'cuda'], default='auto')
    parser.add_argument('--threads', type=int, default=4)
    parser.add_argument('--batch-size', type=int, default=32)
    parser.add_argument('--train-size', type=int, default=4500)
    parser.add_argument('--validation-size', type=int, default=500)
    parser.add_argument('--eval-size', type=int, default=5000)
    parser.add_argument('--epochs', type=int, default=30)
    parser.add_argument('--learning-rate', type=float, default=0.001)
    parser.add_argument('--attack', choices=['fgsm', 'pgd', 'cw', 'ba', 'bim', 'aa'], default='fgsm')
    parser.add_argument('--attack-epsilon', type=float, default=0.15)
    parser.add_argument('--attack-alpha', type=float, default=2 / 255)
    parser.add_argument('--attack-steps', type=int, default=40)
    parser.add_argument('--rectify-epsilon', type=float, default=0.4)
    parser.add_argument('--rectify-alpha', type=float, default=2 / 255)
    parser.add_argument('--rectify-steps', type=int, default=100)
    parser.add_argument('--rectify-stop', choices=['correct', 'fixed'], default='correct')
    parser.add_argument('--gradient-loss', choices=['logits-ce', 'legacy-probability-ce'], default='logits-ce')
    parser.add_argument('--evaluate-only', action='store_true')
    parser.add_argument('--denoiser-checkpoint', help='Required with --evaluate-only')
    args = parser.parse_args()
    if min(args.batch_size, args.epochs, args.threads, args.attack_steps, args.rectify_steps) < 1:
        parser.error('Batch size, epochs, threads and step counts must be positive')
    if args.validation_size < 0 or min(args.attack_epsilon, args.rectify_epsilon,
                                       args.attack_alpha, args.rectify_alpha, args.learning_rate) <= 0:
        parser.error('Invalid size, perturbation, step size or learning rate')
    if args.evaluate_only and not args.denoiser_checkpoint:
        parser.error('--evaluate-only requires --denoiser-checkpoint')
    if args.input_mode == 'saved' and not (args.saved_train_attacks and args.saved_eval_attacks):
        parser.error('Saved mode requires --saved-train-attacks and --saved-eval-attacks')
    if args.input_mode == 'generated' and any([args.saved_train_attacks, args.saved_eval_attacks, args.saved_targets]):
        parser.error('Saved paths require --input-mode saved')
    if args.input_mode == 'generated' and args.attack not in ['fgsm', 'pgd']:
        parser.error('Generated mode supports fgsm/pgd only; other attacks require saved mode')
    return args


def main():
    args = parse_args()
    out = Path(args.output)
    out.mkdir(parents=True, exist_ok=False)
    start = time.monotonic()
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.set_num_threads(args.threads)
    device = torch.device('cuda' if args.device == 'auto' and torch.cuda.is_available()
                          else 'cpu' if args.device == 'auto' else args.device)
    torch.use_deterministic_algorithms(True)
    torch.backends.cudnn.benchmark = False
    groups = split_entries(index_images(Path(args.data_root), Path(args.order_file)), args)
    if args.input_mode == 'saved':
        attach_saved_paths(groups, args)
    write_json(out / 'manifest.json', groups)
    metadata = {'arguments': vars(args), 'device': str(device), 'torch': torch.__version__,
                'cuda_runtime': torch.version.cuda,
                'gpu_name': torch.cuda.get_device_name(device) if device.type == 'cuda' else None,
                'numpy': np.__version__, 'classifier_sha256': fingerprint(args.classifier),
                'order_file_sha256': fingerprint(args.order_file),
                'protocol': f'MNIST test {args.fit_half} half fit pool / opposite half evaluation; seeded subsampling',
                'clean_data_use': ('label lookup and evaluation only; training loads saved attacks and targets'
                                   if args.input_mode == 'saved' else
                                   'attack generation and evaluation only; denoiser targets are rectified attacks'),
                'saved_generation_parameters': 'unknown' if args.input_mode == 'saved' else None,
                'paper_comparison': 'Reconstruction of method, not a verified exact historical run'}
    # Pin every selected image so a later NAS edit cannot silently change data.
    metadata['selected_image_sha256'] = {e['id']: fingerprint(e['path'])
                                        for group in groups.values() for e in group}
    if args.input_mode == 'saved':
        metadata['saved_image_sha256'] = {
            e['id']: {key: fingerprint(e[key]) for key in ['attack_path', 'target_path'] if key in e}
            for group in groups.values() for e in group}
    write_json(out / 'config.json', metadata)
    model = load_classifier(args.classifier, device)
    if args.evaluate_only:
        checkpoint = torch.load(args.denoiser_checkpoint, map_location=device, weights_only=True)
        trained_half = checkpoint.get('config', {}).get('fit_half', 'first')
        if trained_half != args.fit_half:
            raise ValueError('Checkpoint fit-half differs: evaluation may overlap its training IDs. '
                             'Train a new model with the requested split.')
        denoiser = Denoiser().to(device)
        denoiser.load_state_dict(checkpoint['state_dict'])
        denoiser.eval()
        selected_epoch = checkpoint['epoch']
    else:
        train_pairs = prepare_pairs(model, groups['train'], args, device, 'train')
        torch.save(train_pairs, out / 'train_pairs.pt')
        valid_pairs = prepare_pairs(model, groups['validation'], args, device, 'validation') if groups['validation'] else None
        if valid_pairs:
            torch.save(valid_pairs, out / 'validation_pairs.pt')
        denoiser, selected_epoch = train_denoiser(train_pairs, valid_pairs, args, device, out)
    # Independent evaluation RNG also permits reproducible PGD evaluate-only runs.
    torch.manual_seed(args.seed + 1)
    metrics = evaluate(model, denoiser, groups['evaluation'], args, device, out)
    metrics.update(selected_epoch=selected_epoch, elapsed_seconds=time.monotonic() - start)
    write_json(out / 'metrics.json', metrics)
    print(json.dumps(metrics, ensure_ascii=False, indent=2), flush=True)


if __name__ == '__main__':
    main()
