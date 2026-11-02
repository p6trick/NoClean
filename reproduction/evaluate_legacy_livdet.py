"""Evaluate note-mapped legacy checkpoints without training or changing NAS files."""
from paths import CHECKPOINT_ROOT, loads_json
import argparse
import json
import time
from datetime import datetime
from pathlib import Path
from types import SimpleNamespace

import torch
from PIL import Image
from torch.nn import functional as F
from torchvision.transforms.functional import to_pil_image

from livdet import HERE, read_tensor, resolve_groups
from models import Denoiser, grayscale
from run import PM, fingerprint, load_classifier, write_json


PRESETS = {
    # Directly named in historical notes; these are candidates, not all paper-final models.
    'fgsm': ('adv-0115', 'n2n-epoch100-0.00017.pt', 91.11),
    'cw': ('adv-0943', 'n2n-epoch300-0.00030.pt', 89.27),
    'ba': ('adv-0502', 'n2n-epoch300-0.00013.pt', 80.55),
    'bim': ('adv-2006', 'n2n-epoch100-0.00016.pt', 90.16),
    'aa': ('adv-0924', 'n2n-epoch100-0.00017.pt', 88.95),
}

# Additional candidates from the same notes; not selected on evaluation accuracy.
FOLLOWUP_PRESETS = {
    'cw': ('adv-2231', 'n2n-epoch100-0.00006.pt', 89.27),
    'ba': ('adv-2113', 'n2n-epoch99-0.00013.pt', 80.55),
}

# Fixed using only the original training-side 125-image diagnostic, not final scores.
PGD_CANDIDATE_PRESETS = {
    'pgd': ('adv-0218', 'n2n-epoch100-0.00012.pt', 91.36),
}


def classifier_input(rgb, png):
    if png:
        # Match legacy clamp -> torchvision uint8 RGB PIL -> grayscale loading.
        pil = to_pil_image(rgb[0].detach().cpu().clamp(0, 1)).convert('L')
        pil = pil.resize((581, 581), Image.Resampling.BILINEAR)
        import numpy as np
        return torch.from_numpy(np.array(pil, dtype='float32') / 255)[None, None].to(rgb.device)
    gray = grayscale(rgb)
    if gray.shape[-2:] != (581, 581):
        gray = F.interpolate(gray, size=(581, 581), mode='bilinear', align_corners=False)
    return gray


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--attacks', nargs='+', choices=[*PRESETS, 'pgd'], default=list(PRESETS))
    parser.add_argument('--checkpoint-profile', choices=['baseline', 'followup', 'pgd-candidate'], default='baseline',
                        help='followup: CW/BA alternatives; pgd-candidate: training-screen-selected adv-0218 epoch100')
    parser.add_argument('--device', choices=['cpu', 'cuda'], default='cuda')
    parser.add_argument('--eval-size', type=int, default=1250)
    parser.add_argument('--first-pools', nargs='+', type=int, choices=[1, 2], default=[1, 2])
    parser.add_argument('--input-modes', nargs='+', choices=['full', 'center512'], default=['full'])
    parser.add_argument('--seed', type=int, default=2026)
    parser.add_argument('--threads', type=int, default=4)
    parser.add_argument('--pm-root', type=Path, default=PM)
    parser.add_argument('--output', type=Path)
    args = parser.parse_args()
    presets = {'baseline': PRESETS, 'followup': FOLLOWUP_PRESETS,
               'pgd-candidate': PGD_CANDIDATE_PRESETS}[args.checkpoint_profile]
    if any(attack not in presets for attack in args.attacks):
        parser.error(f'{args.checkpoint_profile} profile supports only --attacks {" ".join(presets)}')
    if not 1 <= args.eval_size <= 1250 or args.threads < 1:
        parser.error('eval-size must be 1..1250 and threads must be positive')
    if args.device == 'cuda' and not torch.cuda.is_available():
        parser.error('CUDA unavailable; use --device cpu')
    torch.set_num_threads(args.threads)
    device = torch.device(args.device)
    output = args.output or HERE / 'runs' / datetime.now().strftime('legacy_livdet_eval_%Y%m%d_%H%M%S_%f')
    output.mkdir(parents=True, exist_ok=False)
    classifier_path = CHECKPOINT_ROOT / 'livdet/classifier.pth'
    classifier = load_classifier(classifier_path, device)
    configuration = {
        'arguments': {k: str(v) if isinstance(v, Path) else v for k, v in vars(args).items()},
        'classifier_sha256': fingerprint(classifier_path),
        'checkpoint_mapping': ('PGD candidate fixed from training-side125 diagnostic; training attack unknown'
                               if args.checkpoint_profile == 'pgd-candidate' else
                               'user historical notes; not an independently verified paper model'),
        'limitations': ['Pooling stride is absent from state_dict; both are explicit hypotheses.',
                        'Full 581 uses local skip-size interpolation adaptation.',
                        'center512 is a deterministic control, not a verified historical crop.',
                        'PNG path uses PIL bilinear resize, not verified historical TensorFlow resize.',
                        'Attacks were generated on the classifier; no adaptive defense evaluation.'],
    }
    write_json(output / 'config.json', configuration)
    registry = loads_json((HERE / 'livdet_experiments.json').read_text())
    summaries = []
    for attack in args.attacks:
        folder = output / attack
        folder.mkdir()
        selection = SimpleNamespace(train_size=1250, validation_size=0, eval_size=args.eval_size,
                                    seed=args.seed, regenerate_targets=False)
        entries = resolve_groups(args.pm_root, registry[attack], selection)['evaluation']
        write_json(folder / 'manifest.json', entries)
        run, filename, paper = presets[attack]
        checkpoint = args.pm_root / 'Self2Self/noise2noise/temporary/ckpts' / run / filename
        state = torch.load(checkpoint, map_location='cpu', weights_only=True)
        # Strict loading rejects missing keys, unexpected keys and incompatible shapes.
        models = {}
        for pool in args.first_pools:
            model = Denoiser(first_pool=pool)
            model.load_state_dict(state, strict=True)
            models[pool] = model.to(device).eval().requires_grad_(False)
        checkpoint_info = {'path': str(checkpoint), 'sha256': fingerprint(checkpoint),
                           'strict_state_dict_load': True, 'tensor_count': len(state),
                           'mapping_evidence': configuration['checkpoint_mapping']}
        write_json(folder / 'checkpoint.json', checkpoint_info)
        print(attack, checkpoint_info, flush=True)
        records = []
        started = time.monotonic()
        with torch.inference_mode():
            for index, entry in enumerate(entries):
                rgb = read_tensor(entry['attack_path'], True)[None].to(device)
                tensors = {'clean': read_tensor(entry['path'])[None].to(device),
                           'attacked': read_tensor(entry['attack_path'])[None].to(device),
                           'rectified': read_tensor(entry['target_path'])[None].to(device)}
                for mode in args.input_modes:
                    source = rgb if mode == 'full' else rgb[:, :, 34:546, 34:546]
                    if mode != 'full':
                        tensors[f'{mode}_input_png'] = classifier_input(source, True)
                    for pool, denoiser in models.items():
                        result = denoiser(source)
                        if not torch.isfinite(result).all():
                            raise ValueError(f'Nonfinite model output: {attack}, pool {pool}, {entry["id"]}')
                        for png in [False, True]:
                            key = f'{mode}_pool{pool}_{"png" if png else "float"}'
                            tensors[key] = classifier_input(result, png)
                record = {'id': entry['id'], 'label': entry['label'],
                          'predictions': {key: classifier(value).argmax(1).item()
                                          for key, value in tensors.items()}}
                records.append(record)
                if (index + 1) % 4 == 0 or index + 1 == len(entries):
                    print(f'{attack}: {index + 1}/{len(entries)} ({time.monotonic() - started:.1f}s)', flush=True)
        keys = records[0]['predictions']
        correct = {key: sum(r['predictions'][key] == r['label'] for r in records) for key in keys}
        metrics = {'attack': attack, 'count': len(records), 'partial_evaluation': len(records) != 1250,
                   'checkpoint': checkpoint_info, 'correct': correct,
                   'accuracy_percent': {k: 100 * v / len(records) for k, v in correct.items()},
                   'predicted_alive_count': {k: sum(r['predictions'][k] == 1 for r in records) for k in keys},
                   'per_class_count': {str(label): sum(r['label'] == label for r in records) for label in [0, 1]},
                   'paper_defended_accuracy_percent': paper,
                   'elapsed_seconds': time.monotonic() - started}
        write_json(folder / 'predictions.json', records)
        write_json(folder / 'metrics.json', metrics)
        summaries.append(metrics)
        write_json(output / 'summary.json', summaries)
        print(json.dumps(metrics), flush=True)
        del models
    print('RESULTS', output, flush=True)


if __name__ == '__main__':
    main()
