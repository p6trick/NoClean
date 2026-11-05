"""Held-out clean-distance analysis of newly generated rectification; no denoiser training."""
import argparse
import csv
from datetime import datetime
import json
import math
from pathlib import Path
import time
from types import SimpleNamespace

from run import fingerprint, image_batches, load_classifier, rectify, write_json
from paths import CHECKPOINT_ROOT, loads_json
import torch

ATTACKS = ['fgsm', 'pgd', 'cw', 'ba', 'bim', 'aa']


def rmse(left, right):
    """One RMSE per image, in the original [0,1] grayscale pixel space."""
    return (left.double()-right.double()).square().flatten(1).mean(1).sqrt()


def summarize(records, tolerance):
    if not records:
        raise ValueError('Empty analysis')
    n = len(records)
    succeeded = [r for r in records if r['success']]
    failed = [r for r in records if not r['success']]
    initially_wrong = [r for r in records if not r['initially_correct']]
    recovered = [r for r in initially_wrong if r['success']]
    def mean(rows, key):
        return sum(r[key] for r in rows)/len(rows) if rows else None
    return {
        'count': n,
        'success_percent': 100*len(succeeded)/n,
        'failure_percent': 100*len(failed)/n,
        'initially_correct_percent': 100*sum(r['initially_correct'] for r in records)/n,
        'initially_wrong_count': len(initially_wrong),
        'recovery_among_initially_wrong_percent': 100*len(recovered)/len(initially_wrong) if initially_wrong else None,
        'avg_iterations_success_including_zero': mean(succeeded, 'iterations'),
        'avg_iterations_success_positive_only': mean(recovered, 'iterations'),
        'adv_clean_rmse': mean(records, 'adv_clean_rmse'),
        'rect_clean_rmse': mean(records, 'rect_clean_rmse'),
        'closer_percent': 100*sum(r['rect_clean_rmse'] < r['adv_clean_rmse']-tolerance for r in records)/n,
        'closer_among_initially_wrong_percent': 100*sum(r['rect_clean_rmse'] < r['adv_clean_rmse']-tolerance for r in initially_wrong)/len(initially_wrong) if initially_wrong else None,
        'closer_among_recovered_percent': 100*sum(r['rect_clean_rmse'] < r['adv_clean_rmse']-tolerance for r in recovered)/len(recovered) if recovered else None,
        'recovered_adv_clean_rmse': mean(recovered, 'adv_clean_rmse'),
        'recovered_rect_clean_rmse': mean(recovered, 'rect_clean_rmse'),
        'max_rectify_linf': max(r['rectify_linf'] for r in records),
        'max_observed_attack_linf': max(r['attack_linf'] for r in records),
        'failure_count_at_iteration_limit': sum(r['failure_at_limit'] for r in records),
    }


def save_summary(folder, rows):
    write_json(folder/'summary.json', rows)
    with (folder/'summary.csv').open('w', newline='') as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader(); writer.writerows(rows)


def get_entries(root, dataset, attack, limit):
    manifest = loads_json((root/dataset/attack/'source_data_manifest.json').read_text())
    if isinstance(manifest, dict):
        full = manifest['evaluation']
        fit_ids = {e['id'] for key in ['train', 'validation'] for e in manifest[key]}
        if fit_ids & {e['id'] for e in full}:
            raise ValueError('Evaluation overlaps manifest fitting IDs')
    else:
        full = manifest
    if len(full) < limit or len({e['id'] for e in full}) != len(full):
        raise ValueError('Insufficient or duplicate evaluation IDs')
    return full[:limit]


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--datasets', nargs='+', choices=['mnist', 'livdet'], default=['mnist', 'livdet'])
    parser.add_argument('--attacks', nargs='+', choices=ATTACKS, default=ATTACKS)
    parser.add_argument('--checkpoint-root', type=Path, default=CHECKPOINT_ROOT)
    parser.add_argument('--device', choices=['cpu', 'cuda'], default='cuda')
    parser.add_argument('--mnist-eval-size', type=int, default=5000)
    parser.add_argument('--livdet-eval-size', type=int, default=1250)
    parser.add_argument('--mnist-batch-size', type=int, default=32)
    parser.add_argument('--livdet-batch-size', type=int, default=1)
    parser.add_argument('--mnist-rectify-epsilon', type=float, default=.3)
    parser.add_argument('--livdet-rectify-epsilon', type=float, default=.03)
    parser.add_argument('--rectify-alpha', type=float, default=2/255)
    parser.add_argument('--rectify-steps', type=int, default=40)
    parser.add_argument('--gradient-loss', choices=['logits-ce','legacy-probability-ce'], default='logits-ce')
    parser.add_argument('--closer-tolerance', type=float, default=1e-12)
    parser.add_argument('--threads', type=int, default=4)
    parser.add_argument('--save-rectified', action='store_true', help='Save float tensors for analysis only; consumes disk space')
    parser.add_argument('--check-only', action='store_true')
    parser.add_argument('--output', type=Path)
    args = parser.parse_args()
    if len(set(args.datasets)) != len(args.datasets) or len(set(args.attacks)) != len(args.attacks):
        parser.error('Duplicate datasets or attacks')
    if not (1 <= args.mnist_eval_size <= 5000 and 1 <= args.livdet_eval_size <= 1250):
        parser.error('Invalid evaluation sizes')
    if min(args.mnist_batch_size,args.livdet_batch_size,args.rectify_steps,args.threads)<1:
        parser.error('Batch/step/thread counts must be positive')
    if not all(math.isfinite(v) and v>0 for v in [args.mnist_rectify_epsilon,args.livdet_rectify_epsilon,args.rectify_alpha]):
        parser.error('Perturbation and step bounds must be finite and positive')
    if not math.isfinite(args.closer_tolerance) or args.closer_tolerance<0:
        parser.error('Closer tolerance must be finite and nonnegative')
    groups = {}
    collection = loads_json((args.checkpoint_root/'manifest.json').read_text())
    hashes = {}
    for dataset in args.datasets:
        limit = args.mnist_eval_size if dataset=='mnist' else args.livdet_eval_size
        reference = None
        for attack in args.attacks:
            entries = get_entries(args.checkpoint_root,dataset,attack,limit)
            identity = [(e['id'],e['label'],e['path']) for e in entries]
            if reference is not None and identity != reference:
                raise ValueError('Cross-attack evaluation ID/order/label/clean-path mismatch')
            reference = identity
            for e in entries:
                for key in ['path','attack_path']:
                    if not Path(e[key]).is_file():
                        raise FileNotFoundError(e[key])
            groups[dataset,attack] = entries
            print('VERIFIED',dataset,attack,len(entries),flush=True)
        classifier = args.checkpoint_root/dataset/'classifier.pth'
        expected = next(r['sha256'] for r in collection['copied_files'] if r['copy']==f'{dataset}/classifier.pth')
        if fingerprint(classifier) != expected:
            raise ValueError('Classifier hash mismatch')
        hashes[dataset] = expected
    if args.check_only:
        return
    if args.device=='cuda' and not torch.cuda.is_available():
        parser.error('CUDA unavailable; use a GPU terminal or a small --device cpu check')
    torch.set_num_threads(args.threads)
    torch.manual_seed(2026)
    torch.use_deterministic_algorithms(True)
    torch.backends.cudnn.benchmark = False
    device = torch.device(args.device)
    output = args.output or Path(__file__).parent/'runs'/datetime.now().strftime('rectification_mechanism_%Y%m%d_%H%M%S_%f')
    output.mkdir(parents=True,exist_ok=False)
    write_json(output/'config.json',{
        'arguments':{key:str(v) if isinstance(v,Path) else v for key,v in vars(args).items()},
        'classifier_sha256':hashes,'torch':str(torch.__version__),
        'gpu':torch.cuda.get_device_name(device) if device.type=='cuda' else None,
        'protocol':'Fixed stored attacks; new float rectification; no denoiser loaded/trained',
        'rectification':'True-label gradient descent; projection around archived attack; [0,1] clamp; no random start; stop on correct class',
        'success':'Classifier(final float rectification)==true label; initially correct images count as success at0 steps',
        'failure':'Still misclassified after maximum allowed updates; last iterate retained',
        'avg_iterations':'Successful samples including0; positive-only recovered average also reported',
        'rmse':'Arithmetic mean of per-image grayscale RMSE in[0,1], all samples; no PNG quantization of new outputs',
        'closer':'rect_clean_rmse < adv_clean_rmse - closer_tolerance',
        'limitations':['Uses labels for diagnostic rectification, not a deployable defense.',
                       'No generated rectification is used as a training target in this script.',
                       'Defaults follow latest manuscript rectification .3 MNIST/.03 LivDet,2/255,40; historical target generation differs/unverified.',
                       'LivDet operates on PIL grayscale, not a reconstructed historical RGB generator.',
                       'Distances alone do not establish classifier specificity; another classifier is needed to test transfer.']})
    rows = []
    all_started = time.monotonic()
    input_hash_cache = {}
    for dataset in args.datasets:
        folder = output/dataset
        folder.mkdir()
        model = load_classifier(args.checkpoint_root/dataset/'classifier.pth',device)
        batch_size = args.mnist_batch_size if dataset=='mnist' else args.livdet_batch_size
        epsilon = args.mnist_rectify_epsilon if dataset=='mnist' else args.livdet_rectify_epsilon
        settings = SimpleNamespace(rectify_epsilon=epsilon,rectify_alpha=args.rectify_alpha,
            rectify_steps=args.rectify_steps,rectify_stop='correct',gradient_loss=args.gradient_loss)
        for attack in args.attacks:
            attack_folder = folder/attack
            attack_folder.mkdir()
            entries = groups[dataset,attack]
            write_json(attack_folder/'manifest.json',entries)
            pinned = {}
            for e in entries:
                pinned[e['id']] = {}
                for key in ['path','attack_path']:
                    path = e[key]
                    if path not in input_hash_cache:
                        input_hash_cache[path] = fingerprint(path)
                    pinned[e['id']][key] = input_hash_cache[path]
            write_json(attack_folder/'input_hashes.json',pinned)
            if dataset=='livdet':
                import livdet
                livdet.RECOVERED_IMAGES.clear()
                livdet.RECOVERY_REPORT_PATH = attack_folder/'image_decode_recoveries.json'
                def batches():
                    for start in range(0,len(entries),batch_size):
                        chunk = entries[start:start+batch_size]
                        clean = torch.stack([livdet.read_tensor(e['path']) for e in chunk]).to(device)
                        attacked = torch.stack([livdet.read_tensor(e['attack_path']) for e in chunk]).to(device)
                        yield clean,attacked,torch.tensor([e['label'] for e in chunk],device=device)
            else:
                def batches():
                    for start in range(0,len(entries),batch_size):
                        chunk = entries[start:start+batch_size]
                        clean,labels = next(image_batches(chunk,len(chunk),device))
                        attacked,_ = next(image_batches(chunk,len(chunk),device,'attack_path'))
                        yield clean,attacked,labels
            records = []
            if args.save_rectified:
                (attack_folder/'analysis_only_rectified').mkdir()
            started = time.monotonic()
            with (attack_folder/'per_image.csv').open('w',newline='') as stream:
                writer = None
                for clean,attacked,labels in batches():
                    with torch.no_grad():
                        before = model(attacked).argmax(1)
                        clean_pred = model(clean).argmax(1)
                    rectified,iterations = rectify(model,attacked,labels,settings)
                    if not torch.isfinite(rectified).all():
                        raise ValueError('Nonfinite rectification')
                    linf = (rectified-attacked).abs().flatten(1).amax(1)
                    if linf.max().item()>epsilon+1e-6 or rectified.min()<0 or rectified.max()>1:
                        raise ValueError('Rectification violates attack-centered projection constraints')
                    with torch.no_grad():
                        after = model(rectified).argmax(1)
                    d_adv,d_rec = rmse(attacked,clean),rmse(rectified,clean)
                    perturb = (attacked-clean).abs().flatten(1).amax(1)
                    for i in range(len(labels)):
                        e = entries[len(records)]
                        success = after[i].item()==labels[i].item()
                        initial = before[i].item()==labels[i].item()
                        count = iterations[i].item()
                        if (initial and count!=0) or (not success and count!=args.rectify_steps):
                            raise ValueError('Inconsistent stopping/iteration accounting')
                        record = {'id':e['id'],'label':e['label'],'clean_prediction':clean_pred[i].item(),
                            'attack_prediction':before[i].item(),'rectified_prediction':after[i].item(),
                            'initially_correct':initial,'success':success,'iterations':count,
                            'failure_at_limit':not success and count==args.rectify_steps,
                            'adv_clean_rmse':d_adv[i].item(),'rect_clean_rmse':d_rec[i].item(),
                            'closer':d_rec[i].item()<d_adv[i].item()-args.closer_tolerance,
                            'rectify_linf':linf[i].item(),'attack_linf':perturb[i].item()}
                        if writer is None:
                            writer = csv.DictWriter(stream,fieldnames=list(record));writer.writeheader()
                        writer.writerow(record)
                        if args.save_rectified:
                            name = e['id']
                            if Path(name).name!=name:
                                raise ValueError('Unsafe sample ID')
                            torch.save({'image':rectified[i].detach().cpu(),'id':name,'label':e['label'],
                                        'analysis_only':True,'not_for_training':True},
                                       attack_folder/'analysis_only_rectified'/f'{name}.pt')
                        records.append(record)
                    stream.flush()
                    if len(records)%max(25*batch_size,1)==0 or len(records)==len(entries):
                        print(dataset,attack,len(records),'/',len(entries),flush=True)
            metrics = summarize(records,args.closer_tolerance)
            metrics.update(dataset=dataset,attack=attack,epsilon=epsilon,
                partial_evaluation=len(entries)!=(5000 if dataset=='mnist' else 1250),
                elapsed_seconds=time.monotonic()-started)
            write_json(attack_folder/'metrics.json',metrics)
            rows.append(metrics)
            save_summary(output,rows)
            save_summary(folder,[row for row in rows if row['dataset']==dataset])
            print('RESULT',json.dumps(metrics),flush=True)
        del model
    write_json(output/'completion.json',{'completed_conditions':len(rows),'elapsed_seconds':time.monotonic()-all_started})
    print('RESULTS',output,flush=True)


if __name__=='__main__':
    main()
