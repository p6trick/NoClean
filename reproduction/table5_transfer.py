"""N2N_Journal.pdf Table5: LivDet denoisers on saved AA-attacked MNIST.

Input handling is unverified historically. Report each explicit hypothesis;
never silently resize inputs or change pooling to make the native model run.
"""
from paths import CHECKPOINT_ROOT, loads_json
import argparse
import csv
import json
from datetime import datetime
from pathlib import Path

import torch
from torch.nn import functional as F

from models import Denoiser, grayscale
from run import fingerprint, image_batches, load_classifier, write_json
from torchvision.transforms.functional import to_pil_image
import numpy as np


ATTACKS = ['fgsm', 'pgd', 'cw', 'ba', 'bim', 'aa']
PAPER = dict(zip(ATTACKS, [85.64, 83.61, 90.46, 89.27, 81.66, 82.26]))
DESCRIPTIONS = {
    'native': 'pool2, direct28 input; no adaptation; expected pooling-size failure',
    'pool1': 'pool1 instead of collected pool2; weights unchanged, architecture setting changed',
    'pad32': 'pool2; zero-pad2 pixels each side to32, remove border after denoising; no resize',
}


def forward_mode(model, rgb, mode):
    if mode == 'pad32':
        return model(F.pad(rgb, (2, 2, 2, 2), value=0))[:, :, 2:-2, 2:-2]
    return model(rgb)


def png_grayscale(rgb):
    # PIL conversion matches the archived RGB clamp -> uint8 image -> L path.
    values = [torch.from_numpy(np.array(to_pil_image(x.detach().cpu().clamp(0, 1)).convert('L'),
                                         dtype=np.float32) / 255)[None] for x in rgb]
    return torch.stack(values).to(rgb.device)


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
    parser.add_argument('--device', choices=['cpu', 'cuda'], default='cuda')
    parser.add_argument('--eval-size', type=int, default=5000)
    parser.add_argument('--batch-size', type=int, default=32)
    parser.add_argument('--threads', type=int, default=4)
    parser.add_argument('--modes', nargs='+', choices=list(DESCRIPTIONS), default=list(DESCRIPTIONS))
    parser.add_argument('--output', type=Path)
    args = parser.parse_args()
    if not 1 <= args.eval_size <= 5000 or min(args.batch_size, args.threads) < 1:
        parser.error('Invalid sizes or thread count')
    if len(set(args.modes)) != len(args.modes):
        parser.error('Duplicate modes')
    if args.device == 'cuda' and not torch.cuda.is_available():
        parser.error('CUDA unavailable; use --device cpu')
    device = torch.device(args.device)
    torch.set_num_threads(args.threads)
    collection = loads_json((args.checkpoint_root / 'manifest.json').read_text())
    manifest = loads_json((args.checkpoint_root / 'mnist/aa/source_data_manifest.json').read_text())
    entries = manifest['evaluation'][:args.eval_size]
    if len(entries) != args.eval_size or len({e['id'] for e in entries}) != len(entries):
        raise ValueError('Missing or duplicate evaluation images')
    if {e['id'] for e in entries} & {e['id'] for e in manifest['train'] + manifest['validation']}:
        raise ValueError('MNIST source experiment split overlap')
    output = args.output or here / 'runs' / datetime.now().strftime('table5_transfer_%Y%m%d_%H%M%S_%f')
    output.mkdir(parents=True, exist_ok=False)
    write_json(output / 'manifest.json', entries)
    write_json(output / 'config.json', {
        'paper': str(here.parent / 'N2N_Journal.pdf'), 'table': 5,
        'arguments': {k: str(v) if isinstance(v, Path) else v for k, v in vars(args).items()},
        'modes': DESCRIPTIONS, 'strict_paper_reproduction_verified': False,
        'limitations': ['Original Table5 input handling/architecture is unverified.',
                        'pool1/pad32 are explicit diagnostic adaptations, not established paper settings.',
                        'LivDet PGD-labelled checkpoint training attack remains unconfirmed.',
                        'No denoiser is trained or selected on MNIST here; weights are fixed.',
                        'All modes are reported; do not select a best mode and claim exact reproduction.'],
        'count': len(entries), 'partial_evaluation': len(entries) != 5000})
    classifier_file = args.checkpoint_root / 'mnist/classifier.pth'
    classifier_record = next(x for x in collection['copied_files'] if x['copy'] == 'mnist/classifier.pth')
    if fingerprint(classifier_file) != classifier_record['sha256']:
        raise ValueError('MNIST classifier hash mismatch')
    classifier = load_classifier(classifier_file, device)
    baseline = {}
    with torch.inference_mode():
        for key, name in [('path', 'clean'), ('attack_path', 'aa_attacked')]:
            count = correct = 0
            for images, labels in image_batches(entries, args.batch_size, device, key):
                correct += classifier(images).argmax(1).eq(labels).sum().item()
                count += len(labels)
            baseline[name] = {'count': count, 'correct': correct, 'accuracy_percent': 100 * correct / count}
    write_json(output / 'baseline.json', baseline)
    print('BASELINE', json.dumps(baseline), flush=True)
    rows = []
    for source in ATTACKS:
        cfg = next(x for x in collection['models'] if x['dataset'] == 'livdet' and x['attack'] == source)
        checkpoint = args.checkpoint_root / cfg['checkpoint']
        if fingerprint(checkpoint) != cfg['sha256']:
            raise ValueError(f'Checkpoint hash mismatch: {checkpoint}')
        state = torch.load(checkpoint, map_location='cpu', weights_only=True)
        for mode in args.modes:
            model = Denoiser(first_pool=1 if mode == 'pool1' else 2)
            model.load_state_dict(state, strict=True)
            model.to(device).eval().requires_grad_(False)
            base = {'model_label': source.upper(), 'mode': mode, 'checkpoint': str(checkpoint),
                    'sha256': cfg['sha256'], 'paper_accuracy_percent': PAPER[source],
                    'count': len(entries), 'status': 'ok', 'float_accuracy_percent': None,
                    'png_accuracy_percent': None, 'float_delta_pp': None, 'png_delta_pp': None,
                    'error': ''}
            with torch.inference_mode():
                try:
                    test = forward_mode(model, torch.zeros(1, 3, 28, 28, device=device), mode)
                    if test.shape != (1, 3, 28, 28):
                        raise ValueError(f'Unexpected output shape {test.shape}')
                except (RuntimeError, ValueError) as exc:
                    base.update(status='unsupported', error=f'{type(exc).__name__}: {exc}')
                    rows.append(base)
                    write_comparison(output, rows)
                    print(source, mode, 'UNSUPPORTED', base['error'], flush=True)
                    continue
                correct = {'float': 0, 'png': 0}
                records, count = [], 0
                for images, labels in image_batches(entries, args.batch_size, device, 'attack_path'):
                    result = forward_mode(model, images.repeat(1, 3, 1, 1), mode)
                    if not torch.isfinite(result).all():
                        raise ValueError(f'Nonfinite output: {source}/{mode}')
                    variants = {'float': grayscale(result), 'png': png_grayscale(result)}
                    preds = {key: classifier(value).argmax(1) for key, value in variants.items()}
                    for key in correct:
                        correct[key] += preds[key].eq(labels).sum().item()
                    for i in range(len(labels)):
                        records.append({'id': entries[count + i]['id'], 'label': labels[i].item(),
                                        **{key: pred[i].item() for key, pred in preds.items()}})
                    count += len(labels)
                    if count % (args.batch_size * 25) == 0 or count == len(entries):
                        print(source, mode, f'{count}/{len(entries)}', flush=True)
                for key in correct:
                    base[f'{key}_accuracy_percent'] = 100 * correct[key] / count
                    base[f'{key}_delta_pp'] = base[f'{key}_accuracy_percent'] - PAPER[source]
                write_json(output / f'{source}_{mode}_predictions.json', records)
                rows.append(base)
                write_comparison(output, rows)
                print(json.dumps(base), flush=True)
    print('RESULTS', output, flush=True)


if __name__ == '__main__':
    main()
