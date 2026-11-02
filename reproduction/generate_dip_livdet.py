"""Regenerate missing LivDet DIP BIM/AA outputs using archived notebook settings."""
from paths import CHECKPOINT_ROOT, loads_json
import argparse
import csv
import hashlib
import json
import math
import random
import time
from datetime import datetime
from pathlib import Path

import numpy as np
from PIL import Image
import torch
from torch.nn import functional as F

from legacy_dip import get_net
import livdet
from run import fingerprint, load_classifier, write_json


def image_seed(seed, attack, identity):
    return int.from_bytes(hashlib.sha256(f'{seed}:{attack}:{identity}'.encode()).digest()[:4], 'big')


def generate(source, args, seed, emit):
    """Fixed Adam steps; retain archived noisy-input PSNR backtracking and final train-mode output."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if source.is_cuda:
        torch.cuda.manual_seed_all(seed)
    net = get_net(32, 'skip', 'reflection', upsample_mode='bilinear',
                  skip_n33d=128, skip_n33u=128, skip_n11=4, num_scales=5).to(source.device).train()
    optimizer = torch.optim.Adam(net.parameters(), lr=args.learning_rate)
    # Archived get_noise samples on CPU, scales by .1, then moves to device.
    saved_input = (torch.rand(1, 32, source.shape[-2], source.shape[-1]) * .1).to(source.device)
    noise = saved_input.clone()
    last_net, psnr_last, closure_index, fallbacks = None, 0.0, 0, 0
    history = []
    last_input = saved_input
    for step in range(args.iterations):
        optimizer.zero_grad(set_to_none=True)
        last_input = saved_input + noise.normal_() * args.reg_noise_std
        result = net(last_input)
        if result.shape != source.shape:
            raise ValueError(f'DIP output shape mismatch: {result.shape}, {source.shape}')
        loss = F.mse_loss(result, source)
        if not torch.isfinite(loss):
            raise ValueError('Nonfinite DIP loss')
        loss.backward()
        # Equivalent PSNR to skimage with float images in [0,1]; no clean image used.
        psnr = -10 * math.log10(max(float(loss.detach()), 1e-30))
        reverted = False
        # Preserve notebook's condition literally: this is NOT "every100 steps".
        if closure_index % 100:
            if psnr - psnr_last < -5:
                if last_net is None:
                    raise ValueError('Missing legacy backtracking snapshot')
                with torch.no_grad():
                    for parameter, previous in zip(net.parameters(), last_net):
                        parameter.copy_(previous.to(parameter.device))
                fallbacks += 1
                reverted = True
            else:
                # .cpu() in the archived CUDA run makes a copy; clone also preserves this on CPU.
                last_net = [p.detach().cpu().clone() for p in net.parameters()]
                psnr_last = psnr
        if not reverted:
            closure_index += 1
        # Archived optimize() still steps using existing gradients after a revert.
        optimizer.step()
        if (step + 1) % 100 == 0 or step + 1 == args.iterations:
            row = {'adam_step': step + 1, 'mse_to_attack': float(loss.detach()),
                   'psnr_to_attack': psnr, 'fallbacks': fallbacks}
            history.append(row)
            emit(row)
    # Notebook saves net(net_input) after the last optimizer step, NOT out_avg or eval-mode output.
    with torch.no_grad():
        final = net(last_input).detach().cpu()[0]
    if not torch.isfinite(final).all():
        raise ValueError('Nonfinite DIP output')
    del net, optimizer
    return final, {'history': history, 'fallbacks': fallbacks, 'adam_steps': args.iterations,
                   'closure_index': closure_index}


def main():
    here = Path(__file__).resolve().parent
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--attacks', nargs='+', choices=['bim', 'aa'], default=['bim', 'aa'])
    parser.add_argument('--eval-size', type=int, default=1250)
    parser.add_argument('--iterations', type=int, default=801)
    parser.add_argument('--learning-rate', type=float, default=.01)
    parser.add_argument('--reg-noise-std', type=float, default=1 / 30)
    parser.add_argument('--seed', type=int, default=2026)
    parser.add_argument('--threads', type=int, default=4)
    parser.add_argument('--device', choices=['cuda', 'cpu'], default='cuda')
    parser.add_argument('--checkpoint-root', type=Path, default=CHECKPOINT_ROOT)
    parser.add_argument('--output', type=Path)
    parser.add_argument('--resume', type=Path, help='Continue same run; completed verified PNGs are reused')
    parser.add_argument('--extend', action='store_true', help='Allow resume to extend evaluation count, preserving prior results')
    args = parser.parse_args()
    if not 1 <= args.eval_size <= 1250 or min(args.iterations, args.threads) < 1:
        parser.error('Invalid counts')
    if args.learning_rate <= 0 or args.reg_noise_std < 0:
        parser.error('Invalid learning parameters')
    if len(set(args.attacks)) != len(args.attacks) or args.output and args.resume:
        parser.error('Duplicate attack or both output/resume provided')
    if args.extend and not args.resume:
        parser.error('--extend requires --resume')
    if args.device == 'cuda' and not torch.cuda.is_available():
        parser.error('CUDA unavailable; use --device cpu only for small checks')
    torch.set_num_threads(args.threads)
    torch.backends.cudnn.enabled = True
    torch.backends.cudnn.benchmark = True
    device = torch.device(args.device)
    output = args.resume or args.output or here / 'runs' / datetime.now().strftime('dip_regenerated_%Y%m%d_%H%M%S_%f')
    classifier_file = args.checkpoint_root / 'livdet/classifier.pth'
    collection = loads_json((args.checkpoint_root / 'manifest.json').read_text())
    expected = next(r['sha256'] for r in collection['copied_files'] if r['copy'] == 'livdet/classifier.pth')
    if fingerprint(classifier_file) != expected:
        raise ValueError('Classifier hash mismatch')
    sources = loads_json((here / 'legacy_dip/SOURCE.json').read_text())
    for item in sources:
        if fingerprint(here / 'legacy_dip' / item['copy']) != item['sha256']:
            raise ValueError('Legacy DIP snapshot changed')
    protocol = {'attacks': args.attacks, 'eval_size': args.eval_size, 'iterations': args.iterations,
                'learning_rate': args.learning_rate, 'reg_noise_std': args.reg_noise_std,
                'seed': args.seed, 'device': args.device, 'torch': torch.__version__,
                'classifier_sha256': expected, 'legacy_model_sources': sources,
                'architecture': 'archived skip128x5/skip4/input32/reflection/bilinear/sigmoid',
                'optimizer': 'Adam defaults, optimize net parameters only',
                'backtracking': 'archived closure-index %100; PSNR drop>5; parameter restore then Adam step',
                'saved_output': 'final net(last noisy input), train mode; RGB8bit floor, no EMA',
                'note': 'New DIP generation; not archived outputs. Notebook801 steps vs manuscript800.',
                'random_seed_note': 'Historical seeds unknown; independent deterministic seed per attack/ID.'}
    groups = {}
    for attack in args.attacks:
        entries = loads_json((args.checkpoint_root / 'livdet' / attack / 'source_data_manifest.json').read_text())[:args.eval_size]
        if len(entries) != args.eval_size or len({e['id'] for e in entries}) != len(entries):
            raise ValueError('Invalid evaluation entries')
        groups[attack] = entries
    if args.resume:
        previous = loads_json((output / 'config.json').read_text())
        if previous != protocol and args.extend:
            same_settings = {k: v for k, v in previous.items() if k != 'eval_size'} == {
                k: v for k, v in protocol.items() if k != 'eval_size'}
            if not same_settings or previous['eval_size'] >= args.eval_size:
                raise ValueError('Extension must only increase evaluation count; hyperparameters must match')
            # Check all saved ID/label/path prefixes before changing the run configuration.
            for attack, entries in groups.items():
                path = output / attack / 'manifest.json'
                if path.exists():
                    old = loads_json(path.read_text())
                    if old != entries[:len(old)]:
                        raise ValueError('Extension changes previously selected inputs')
            history_path = output / 'extension_history.json'
            history = loads_json(history_path.read_text()) if history_path.exists() else []
            history.append({'previous_config': previous, 'new_eval_size': args.eval_size})
            write_json(history_path, history)
            write_json(output / 'config.json', protocol)
            previous = protocol
        if not output.is_dir() or previous != protocol:
            raise ValueError('Resume configuration does not match; repeat original options')
    else:
        output.mkdir(parents=True, exist_ok=False)
        write_json(output / 'config.json', protocol)
    classifier = load_classifier(classifier_file, device)
    livdet.RECOVERY_REPORT_PATH = output / 'image_load_recoveries.json'
    rows = []
    for attack in args.attacks:
        folder = output / attack
        (folder / 'images').mkdir(parents=True, exist_ok=True)
        (folder / 'records').mkdir(exist_ok=True)
        entries = groups[attack]
        manifest_path = folder / 'manifest.json'
        if manifest_path.exists():
            old = loads_json(manifest_path.read_text())
            if old != entries and not (args.extend and old == entries[:len(old)]):
                raise ValueError('Resume evaluation manifest changed')
        write_json(manifest_path, entries)
        records = []
        for index, entry in enumerate(entries):
            path = folder / 'images' / entry['id']
            record_file = folder / 'records' / (entry['id'] + '.json')
            source_sha = fingerprint(entry['attack_path'])
            seed = image_seed(args.seed, attack, entry['id'])
            if record_file.exists():
                record = loads_json(record_file.read_text())
                if (record['source_sha256'] != source_sha or record['seed'] != seed or
                        record['label'] != entry['label'] or record['output_sha256'] != fingerprint(path)):
                    raise ValueError(f'Resume source/output mismatch: {entry["id"]}')
                print('REUSE', attack, entry['id'], flush=True)
            else:
                started = time.monotonic()
                image = livdet.decode_image(entry['attack_path'], 'RGB')
                if image.size != (581, 581):
                    raise ValueError('Archived DIP protocol expects native581x581; no implicit resize')
                source = torch.from_numpy(np.asarray(image, dtype=np.float32).transpose(2, 0, 1) / 255)[None].to(device)
                print('START', attack, f'{index + 1}/{len(entries)}', entry['id'], flush=True)
                final, details = generate(source, args, seed, lambda row: print(attack, entry['id'], json.dumps(row), flush=True))
                # Matplotlib imsave's opaque RGB data conversion truncates channel values to8bit.
                array = (final.permute(1, 2, 0).numpy().clip(0, 1) * 255).astype(np.uint8)
                Image.fromarray(array).save(path)
                if fingerprint(entry['attack_path']) != source_sha:
                    raise ValueError('Attack source changed during generation')
                with torch.inference_mode():
                    pred = classifier(livdet.read_tensor(path)[None].to(device)).argmax(1).item()
                    before = classifier(livdet.read_tensor(entry['attack_path'])[None].to(device)).argmax(1).item()
                record = {'id': entry['id'], 'label': entry['label'], 'source': entry['attack_path'],
                          'source_sha256': source_sha, 'output': str(path.resolve()),
                          'output_sha256': fingerprint(path), 'seed': seed,
                          'attacked_prediction': before, 'dip_prediction': pred,
                          'elapsed_seconds': time.monotonic() - started, **details}
                write_json(record_file, record)
                print('DONE', attack, entry['id'], 'seconds', round(record['elapsed_seconds'], 2),
                      'dip_pred', pred, 'label', entry['label'], flush=True)
                del source
            records.append(record)
            count = len(records)
            confused = [[0, 0], [0, 0]]
            for r in records:
                confused[r['label']][r['dip_prediction']] += 1
            metrics = {'attack': attack, 'count': count, 'requested_count': len(entries),
                       'partial_evaluation': count != 1250, 'new_dip_outputs': True,
                       'attacked_accuracy_percent': 100 * sum(r['attacked_prediction'] == r['label'] for r in records) / count,
                       'dip_accuracy_percent': 100 * sum(r['dip_prediction'] == r['label'] for r in records) / count,
                       'paper_dip_accuracy_percent': 40.0, 'confusion_matrix': confused,
                       'confusion_matrix_order': ['Fake', 'Alive'],
                       'mean_generation_seconds': sum(r['elapsed_seconds'] for r in records) / count}
            write_json(folder / 'metrics.json', metrics)
        rows.append(metrics)
        write_json(output / 'summary.json', rows)
        with (output / 'summary.csv').open('w') as stream:
            writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
            writer.writeheader()
            writer.writerows(rows)
    print('RESULTS', output, flush=True)


if __name__ == '__main__':
    main()
