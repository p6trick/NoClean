"""Training-side diagnostics for archived-vs-fresh PGD; no final-set tuning."""
import argparse
import csv
from datetime import datetime
import hashlib
import json
import os
from pathlib import Path
from types import SimpleNamespace

os.environ.setdefault('CUBLAS_WORKSPACE_CONFIG', ':4096:8')
import torch
from torch import nn
from torch.nn import functional as F

from adaptive_pgd import load_denoiser, pgd_candidates
from evaluate_legacy_livdet import classifier_input
import livdet
from paths import CHECKPOINT_ROOT, PM_ROOT, loads_json
from run import fingerprint, load_classifier, project, write_json


def start_for_id(clean, identity, seed):
    value = int.from_bytes(hashlib.sha256(f'{seed}:{identity}'.encode()).digest()[:8], 'big')
    generator = torch.Generator(device=clean.device).manual_seed(value)
    return project(clean + torch.empty_like(clean).uniform_(-.03, .03, generator=generator), clean, .03)


def generate(model, clean, labels, alpha, random_start, probability_ce, best, start, steps):
    if best:
        return pgd_candidates(model, clean, labels, [start if random_start else clean], .03, alpha, steps)[0]
    value = start.detach().clone() if random_start else clean.detach().clone()
    for _ in range(steps):
        value = value.detach().requires_grad_(True)
        output = model(value)
        if probability_ce:
            output = output.softmax(1)
        loss = F.cross_entropy(output, labels)
        grad, = torch.autograd.grad(loss, value)
        if not torch.isfinite(grad).all():
            raise ValueError('Nonfinite gradient')
        value = project(value.detach()+alpha*grad.sign(), clean, .03).detach()
    return value


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--device', choices=['cpu','cuda'], default='cuda')
    parser.add_argument('--sample-size', type=int, default=16, help='Even number of training-side images, balanced by class')
    parser.add_argument('--steps', type=int, default=40)
    parser.add_argument('--seed', type=int, default=2026)
    parser.add_argument('--threads', type=int, default=4)
    parser.add_argument('--output', type=Path)
    parser.add_argument('--variants', nargs='+', help='Optional subset of diagnostic variant names')
    args = parser.parse_args()
    if args.sample_size < 2 or args.sample_size % 2 or args.steps < 1 or args.threads < 1:
        parser.error('sample-size must be positive/even >=2; steps and threads must be positive')
    if args.device=='cuda' and not torch.cuda.is_available():
        parser.error('CUDA unavailable')
    torch.set_num_threads(args.threads)
    torch.manual_seed(args.seed)
    torch.use_deterministic_algorithms(True)
    torch.backends.cudnn.benchmark = False
    device = torch.device(args.device)
    output = args.output or Path(__file__).parent/'runs'/datetime.now().strftime('livdet_pgd_generation_audit_%Y%m%d_%H%M%S_%f')
    output.mkdir(parents=True, exist_ok=False)
    livdet.RECOVERY_REPORT_PATH = output/'image_decode_recoveries.json'
    registry = loads_json((Path(__file__).parent/'livdet_experiments.json').read_text())
    group = livdet.resolve_groups(PM_ROOT, registry['pgd'],SimpleNamespace(seed=2026,train_size=1125,validation_size=125,eval_size=1250))
    # Fixed training-side sample only, not selected by attack or defense performance.
    entries = []
    for label in [0,1]:
        entries += [e for e in group['train'] if e['label']==label][:args.sample_size//2]
    if len(entries)!=args.sample_size:
        parser.error('Insufficient training-side images')
    final_ids={e['id'] for e in group['evaluation']}
    assert not final_ids.intersection(e['id'] for e in entries)
    write_json(output/'manifest.json',entries)
    classifier_path=CHECKPOINT_ROOT/'livdet/classifier.pth'
    denoiser_path=CHECKPOINT_ROOT/'livdet/pgd/denoiser.pt'
    collection=loads_json((CHECKPOINT_ROOT/'manifest.json').read_text())
    assert fingerprint(classifier_path)==next(r['sha256'] for r in collection['copied_files'] if r['copy']=='livdet/classifier.pth')
    assert fingerprint(denoiser_path)==next(r['sha256'] for r in collection['models'] if r['dataset']=='livdet' and r['attack']=='pgd')
    classifier=load_classifier(classifier_path,device)
    denoiser,_=load_denoiser(denoiser_path,device,'livdet')
    variants=[
        ('current_best',.01,True,False,True),
        ('current_last',.01,True,False,False),
        ('no_random_last',.01,False,False,False),
        ('alpha_2_255_last',2/255,True,False,False),
        ('probability_ce_last',.01,True,True,False),
        ('legacy_hypothesis',2/255,False,True,False),
        ('no_random_alpha2_last',2/255,False,False,False),
        ('no_random_probability_last',.01,False,True,False),
        ('alpha2_probability_last',2/255,True,True,False),
    ]
    if args.variants:
        names={row[0] for row in variants}
        if set(args.variants)-names:
            parser.error('Unknown variant: '+','.join(sorted(set(args.variants)-names)))
        variants=[row for row in variants if row[0] in args.variants]
    write_json(output/'config.json',{'arguments':{k:str(v) if isinstance(v,Path) else v for k,v in vars(args).items()},
        'sampling':'fixed balanced training-side images; final evaluation excluded',
        'starts':'per-ID hash seed; identical perturbations across diagnostic variants, independent of batching',
        'variants':variants,'epsilon':.03,'classifier_sha256':fingerprint(classifier_path),
        'denoiser_sha256':fingerprint(denoiser_path),
        'limitations':['Small diagnostic is not final accuracy and does not select best attack parameters.',
                       'legacy_hypothesis is not verified as the archived generator.',
                       'Round/truncate variants model common input PNG encodings; historical encoder unconfirmed.',
                       'Training-side performance may be optimistic because denoiser may have seen these IDs.']})
    records=[]
    def measure(entry, name, rgb, clean):
        with torch.no_grad():
            gray=classifier_input(rgb,True)
            # For fresh_float we must avoid re-quantizing attack input before the classifier.
            if name.endswith('/float'):
                gray=rgb[:,:1]
            out=denoiser(rgb)
            prediction=classifier(classifier_input(out,True)).argmax(1).item()
            before=classifier(gray).argmax(1).item()
            result={'id':entry['id'],'label':entry['label'],'variant':name,
                    'classifier_correct':int(before==entry['label']),
                    'defended_correct':int(prediction==entry['label']),
                    'input_linf':(gray-clean).abs().max().item(),
                    'input_mse':(gray-clean).square().mean().item()}
            records.append(result)
    for i,e in enumerate(entries):
        clean=livdet.read_tensor(e['path'])[None].to(device)
        labels=torch.tensor([e['label']],device=device)
        archived=livdet.read_tensor(e['attack_path'],True)[None].to(device)
        measure(e,'archived_rgb',archived,clean)
        archived_gray=livdet.read_tensor(e['attack_path'])[None].to(device).repeat(1,3,1,1)
        measure(e,'archived_gray_repeated',archived_gray,clean)
        measure(e,'clean',clean.repeat(1,3,1,1),clean)
        start=start_for_id(clean,e['id'],args.seed)
        for name,alpha,random,prob,best in variants:
            value=generate(classifier,clean,labels,alpha,random,prob,best,start,args.steps)
            for mode,tensor in [('float',value),('round8',value.mul(255).round().clamp(0,255)/255),
                                ('truncate8',value.mul(255).floor().clamp(0,255)/255)]:
                measure(e,name+'/'+mode,tensor.repeat(1,3,1,1),clean)
        print('DIAGNOSED',i+1,'/',len(entries),e['id'],flush=True)
    write_json(output/'per_image.json',records)
    rows=[]
    for name in dict.fromkeys(r['variant'] for r in records):
        values=[r for r in records if r['variant']==name]
        rows.append({'variant':name,'count':len(values),
                     'classifier_accuracy_percent':100*sum(r['classifier_correct'] for r in values)/len(values),
                     'defended_accuracy_percent':100*sum(r['defended_correct'] for r in values)/len(values),
                     'max_input_linf':max(r['input_linf'] for r in values),
                     'mean_input_mse':sum(r['input_mse'] for r in values)/len(values)})
    with (output/'summary.csv').open('w',newline='') as stream:
        writer=csv.DictWriter(stream,fieldnames=list(rows[0]));writer.writeheader();writer.writerows(rows)
    print(json.dumps(rows,indent=2),flush=True)
    print('RESULTS',output,flush=True)


if __name__=='__main__':
    main()
