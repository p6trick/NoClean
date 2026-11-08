"""Create curves and paired image figures from completed sweep caches; no inference."""
import argparse
from contextlib import ExitStack
import hashlib
import json
from pathlib import Path

import numpy as np

from epsilon_sweep_io import LABELS, SUPERVISIONS, condition_dir, read_conditions, write_aggregates, write_json


def sha256_file(path):
    digest = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for block in iter(lambda: stream.read(4*1024*1024), b''):
            digest.update(block)
    return digest.hexdigest()


def choose_samples(inputs, caches, epsilons, count, manual_ids):
    ids = inputs['ids'].tolist()
    if manual_ids:
        if len(set(manual_ids)) != len(manual_ids) or any(name not in ids for name in manual_ids):
            raise ValueError('Visualization IDs must be unique and present in the evaluation manifest')
        return [ids.index(name) for name in manual_ids], 'Manually requested IDs; same samples at all visualization epsilons'
    n = len(ids)
    labels = inputs['labels']
    failures = {target: np.stack([caches[e,target]['pred_defended'] != labels for e in epsilons])
                for target in SUPERVISIONS}
    transitions = sum(np.count_nonzero(np.diff(failures[t].astype(int), axis=0), axis=0) for t in SUPERVISIONS)
    disagreements = np.count_nonzero(failures['clean'] != failures['rectified'], axis=0)
    initially_correct_both = ((inputs['reference_clean_clean'] == labels) &
                              (inputs['reference_clean_rectified'] == labels))
    ranking = sorted(range(n), key=lambda i: (-int(initially_correct_both[i]),
                     -int(transitions[i]), -int(disagreements[i]), i))
    count = min(count, n)
    # Always include up to two fixed, non-selected initial samples, then qualitative transition cases.
    selected = list(range(min(2, count)))
    used_labels = {int(labels[i]) for i in selected}
    for i in ranking:
        if len(selected) == count:
            break
        if i not in selected and int(labels[i]) not in used_labels:
            selected.append(i); used_labels.add(int(labels[i]))
    for i in ranking:
        if len(selected) == count:
            break
        if i not in selected:
            selected.append(i)
    return selected, ('First up to2 fixed samples plus post-hoc examples prioritizing initially correct samples, '
                      'success changes across epsilon, model disagreements and label diversity. '
                      'Qualitative selection only; all accuracy/statistics use the full evaluation subset.')


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--run', type=Path, required=True)
    parser.add_argument('--visual-epsilons', nargs='+', type=float, default=[.05,.10,.20,.40])
    parser.add_argument('--visualization-count', type=int, default=8)
    parser.add_argument('--sample-ids', nargs='+', help='Optional exact IDs instead of automatic example selection')
    args = parser.parse_args()
    if args.visualization_count < 1 or len(set(args.visual_epsilons)) != len(args.visual_epsilons):
        parser.error('Invalid visualization count or duplicate epsilons')
    config, rows, missing = read_conditions(args.run)
    if missing:
        raise SystemExit(f'Sweep is incomplete. Resume evaluation first. Missing: {missing}')
    if any(e not in config['epsilons'] for e in args.visual_epsilons):
        parser.error('Requested visualization epsilon is not in the completed run')
    try:
        import matplotlib
        matplotlib.use('Agg')
        import matplotlib.pyplot as plt
    except ImportError as error:
        raise SystemExit('Plotting needs matplotlib. Install reproduction/requirements_analysis.txt yourself, then rerun analysis.') from error
    input_digest = json.loads((args.run/'input_cache.json').read_text())['sha256']
    if sha256_file(args.run/'inputs.npz') != input_digest:
        raise ValueError('Input cache hash mismatch')
    with np.load(args.run/'inputs.npz', allow_pickle=False) as source:
        inputs = {key: source[key] for key in source.files}
    manifest = json.loads((args.run/'manifest.json').read_text())
    if (inputs['ids'].tolist() != [e['id'] for e in manifest] or
            inputs['labels'].tolist() != [e['label'] for e in manifest]):
        raise ValueError('Input cache does not match manifest ID/order/labels')
    lookup = {(row['epsilon'], row['supervision']):row for row in rows}
    epsilons = sorted(config['epsilons'])
    caches = {}
    with ExitStack() as stack:
        for row in rows:
            folder = condition_dir(args.run,row['epsilon'],row['supervision'])
            if sha256_file(folder/'images.npz') != row['images_sha256']:
                raise ValueError(f'Image cache changed: {folder}')
            if sha256_file(folder/'per_image.csv') != row['per_image_sha256']:
                raise ValueError(f'Per-image records changed: {folder}')
            archive = stack.enter_context(np.load(folder/'images.npz', allow_pickle=False))
            caches[row['epsilon'],row['supervision']] = {
                'archive':archive, 'pred_defended':archive['pred_defended'],
                'pred_clean_defended':archive['pred_clean_defended'],
                'pred_attack_classifier':archive['pred_attack_classifier'],
                'pred_original_classifier':archive['pred_original_classifier']}
        for row in rows:
            cache = caches[row['epsilon'],row['supervision']]
            measured = int((cache['pred_defended']==inputs['labels']).sum())
            if measured != row['correct_count'] or len(cache['pred_defended']) != config['sample_count']:
                raise ValueError('Cached predictions disagree with accuracy metrics')
        fig, ax = plt.subplots(figsize=(8,5))
        for target, color, marker, shift in [('clean','#2463b0','o',9),('rectified','#d46b18','s',-14)]:
            values = [lookup[e,target]['accuracy_percent'] for e in epsilons]
            ax.plot(epsilons, values, color=color, marker=marker, linewidth=2, label=LABELS[target])
            for e,value in zip(epsilons,values):
                ax.annotate(f'{value:.2f}', (e,value), xytext=(0,shift), textcoords='offset points',
                            ha='center', fontsize=8, color=color, annotation_clip=False)
        ax.set(xlabel='Epsilon (L-infinity budget)',ylabel='Adaptive PGD accuracy (%)',
               title='MNIST adaptive PGD epsilon sensitivity',ylim=(0,100))
        ax.set_xticks(epsilons)
        ax.set_xlim(min(epsilons)-.015,max(epsilons)+.015)
        ax.grid(alpha=.25)
        ax.legend()
        fig.tight_layout()
        fig.savefig(args.run/'robust_accuracy_curve.png',dpi=220,bbox_inches='tight')
        plt.close(fig)
        selected, policy = choose_samples(inputs,caches,sorted(args.visual_epsilons),args.visualization_count,args.sample_ids)
        visual_dir = args.run/'visualizations'
        visual_dir.mkdir(exist_ok=True)
        # Load float images once per visualized condition; keep identical intensity scale across all panels.
        visual_images = {(e,t): {key:caches[e,t]['archive'][key] for key in ['adversarial','denoised']}
                         for e in args.visual_epsilons for t in SUPERVISIONS}
        visual_rows = []
        for index in selected:
            label = int(inputs['labels'][index])
            fig, axes = plt.subplots(len(args.visual_epsilons),5,
                                     figsize=(12,2.5*len(args.visual_epsilons)), squeeze=False)
            for j,e in enumerate(args.visual_epsilons):
                panels = [(inputs['clean'][index,0],
                           f'Original clean\nf(x)={int(caches[e,"clean"]["pred_original_classifier"][index])}; y={label}')]
                for target in SUPERVISIONS:
                    cache = caches[e,target]
                    raw_pred = int(cache['pred_attack_classifier'][index])
                    prediction = int(cache['pred_defended'][index])
                    initial_ok = int(cache['pred_clean_defended'][index])==label
                    success = ('yes' if prediction!=label else 'no') if initial_ok else 'initially wrong'
                    name = 'Clean-target' if target=='clean' else 'Rectified-target'
                    panels += [(visual_images[e,target]['adversarial'][index,0],
                                f'Adaptive to {name}\nf(x_adv)={raw_pred}; ASR success={success}'),
                               (visual_images[e,target]['denoised'][index,0],
                                f'{name} denoised\nf(g(x_adv))={prediction}')]
                for col,(pixels,title) in enumerate(panels):
                    axes[j,col].imshow(pixels,cmap='gray',vmin=0,vmax=1,interpolation='nearest')
                    axes[j,col].set_title(title,fontsize=8)
                    axes[j,col].set_axis_off()
                axes[j,0].text(-.25,.5,f'epsilon={e:.2f}',rotation=90,va='center',ha='right',
                               transform=axes[j,0].transAxes,fontsize=10)
            fig.suptitle(f'{str(inputs["ids"][index])} | true label={label} | float images, common [0,1] display',fontsize=11)
            fig.tight_layout(rect=(0,.01,1,.96))
            name = f'sample_{index:05d}.png'
            fig.savefig(visual_dir/name,dpi=180,bbox_inches='tight')
            plt.close(fig)
            visual_rows.append({'index':index,'id':str(inputs['ids'][index]),'label':label,'file':'visualizations/'+name})
    write_json(args.run/'visualization_manifest.json',{
        'policy':policy,'epsilons':args.visual_epsilons,'samples':visual_rows,
        'prediction_semantics':'f(x_adv) is the standalone classifier; f(g(x_adv)) is defended prediction and defines attack success',
        'no_inference_in_analysis':True})
    write_aggregates(args.run)
    with (args.run/'summary.md').open('a') as stream:
        stream.write('\n## 시각화\n\n![측정 곡선](robust_accuracy_curve.png)\n\n')
        stream.write('같은 샘플을 모든 시각화 epsilon에서 비교한다. 선택 정책은 visualization_manifest.json에 기록했다.\n\n')
        for row in visual_rows:
            stream.write(f"- [{row['id']}]({row['file']})\n")
    print('ANALYSIS',args.run/'summary.md')
    print('CURVE',args.run/'robust_accuracy_curve.png')
    print('VISUALIZATIONS',visual_dir)


if __name__=='__main__':
    main()
