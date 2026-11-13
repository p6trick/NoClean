"""Fixed LivDet PGD candidate, with optional adaptive BPDA evaluation."""
import argparse
import csv
from datetime import datetime
import json
from pathlib import Path
from types import SimpleNamespace

from audit_livdet_pgd_generation import generate
from adaptive_pgd import DefendedLivDet, load_denoiser, pgd_candidates
from epsilon_sweep_io import write_csv, write_json
from evaluate_legacy_livdet import classifier_input
import livdet
from paths import CHECKPOINT_ROOT, PM_ROOT, loads_json
from run import fingerprint, load_classifier
from search_livdet_pgd_parameters import encode, signature
import torch
from torchvision.transforms.functional import rgb_to_grayscale


ATTACK = dict(epsilon=8/255, alpha=2/255, steps=40, random_start=False,
              loss='probability', preprocessing='tensor-gray', selection='last',
              encoding='truncate8', batch_size=1)


class QuantizedDefense(torch.nn.Module):
    """Exact truncate8 input and legacy PNG output; identity BPDA for input Q."""
    def __init__(self, classifier, denoiser):
        super().__init__()
        self.defense=DefendedLivDet(classifier,denoiser)

    def forward(self, images):
        exact=encode(images.detach(),'truncate8')
        quantized=exact+(images-images.detach())
        return self.defense(quantized)


def summarize(output, records, models, total):
    if not records:
        return
    rows=[]
    for name in ['classifier', *models]:
        values={key:100*sum(r[name+'_'+key+'_correct'] for r in records)/len(records)
                for key in ['clean','archived','candidate']}
        reference={'classifier':3.76,'rectified':89.76,'clean':89.04}[name]
        rows.append(dict(model=name, count=len(records), expected_count=total,
                         complete=len(records)==total, clean_accuracy_percent=values['clean'],
                         archived_attack_accuracy_percent=values['archived'],
                         candidate_attack_accuracy_percent=values['candidate'],
                         historical_archived_reference_percent=reference,
                         candidate_minus_same_sample_archived_pp=values['candidate']-values['archived']))
    if 'adaptive' in records[0]:
        for row in rows:
            name=row['model']
            if name=='classifier':
                row.update(adaptive_pgd_percent='',adaptive_clean_correct_asr_percent='',
                           adaptive_baseline_accuracy_percent='',combined_accuracy_percent='')
                continue
            items=[r['adaptive'][name] for r in records]
            eligible=sum(v['baseline_correct'] for v in items)
            row.update(adaptive_pgd_percent=100*sum(v['correct'] for v in items)/len(items),
                adaptive_clean_correct_asr_percent=(100*sum(v['baseline_correct'] and not v['correct'] for v in items)/eligible if eligible else None),
                adaptive_baseline_accuracy_percent=100*eligible/len(items),
                combined_accuracy_percent=100*sum(v['correct'] and v['baseline_correct'] and r[name+'_candidate_correct'] for r,v in zip(records,items))/len(items))
        # Nested per-model diagnostics are flattened in the CSV, retained in sample JSON.
    flat=[]
    for r in records:
        item={k:v for k,v in r.items() if k!='adaptive'}
        for name,values in r.get('adaptive',{}).items():
            item.update({name+'_adaptive_'+k:v for k,v in values.items()})
        flat.append(item)
    write_csv(output/'per_image.csv',flat,list(flat[0]))
    write_csv(output/'summary.csv',rows,list(rows[0]))
    write_json(output/'metrics.json',dict(sample_count=len(records), expected_count=total,
        complete=len(records)==total, attack=ATTACK, results=rows,
        adaptive_diagnostics={name:dict(
            gradient_calls=sum(r['adaptive'][name]['gradient_calls'] for r in records),
            zero_gradient_samples=sum(r['adaptive'][name]['zero_gradient_samples'] for r in records),
            max_float_linf=max(r['adaptive'][name]['float_linf'] for r in records),
            max_encoded_linf_from_canonical=max(r['adaptive'][name]['encoded_linf'] for r in records))
            for name in models if 'adaptive' in records[0]},
        max_float_linf_from_attack_center=max(r['float_linf'] for r in records),
        max_encoded_linf_from_canonical_clean=max(r['encoded_linf'] for r in records),
        expected_classifier_candidate_percent=2.88,
        classifier_candidate_delta_to_search_pp=rows[0]['candidate_attack_accuracy_percent']-2.88,
        note='2.88% refers to the full1250 search on RTX3090/cu118, not a guaranteed value on other runtimes.'))


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--adaptive',action='store_true',help='Also attack each complete quantized defense using BPDA')
    p.add_argument('--reference-run',type=Path,help='Required for adaptive mode: completed non-adaptive candidate run')
    p.add_argument('--clean-checkpoint',type=Path,
                   help='Optional PGD clean-target model from the first target-ablation experiment, not the legacy bundle')
    p.add_argument('--device',choices=['cuda','cpu'],default='cuda')
    p.add_argument('--threads',type=int,default=4)
    p.add_argument('--seed',type=int,default=2026)
    p.add_argument('--eval-size',type=int,default=1250)
    p.add_argument('--check-only',action='store_true')
    g=p.add_mutually_exclusive_group()
    g.add_argument('--output',type=Path)
    g.add_argument('--resume',type=Path)
    args=p.parse_args()
    if not 1<=args.eval_size<=1250 or args.threads<1 or not 0<=args.seed<2**32:
        p.error('Invalid evaluation size, threads or seed')
    if args.adaptive and not args.reference_run:
        p.error('--adaptive requires --reference-run to verify the non-adaptive control')
    registry=loads_json((Path(__file__).parent/'livdet_experiments.json').read_text())
    entries=livdet.resolve_groups(PM_ROOT,registry['pgd'],SimpleNamespace(
        seed=2026,train_size=1125,validation_size=125,eval_size=1250))['evaluation']
    recorded=loads_json((CHECKPOINT_ROOT/'livdet/pgd/source_data_manifest.json').read_text())
    if [(e['id'],e['label']) for e in entries]!=[(e['id'],e['label']) for e in recorded]:
        raise ValueError('Evaluation IDs/order/labels differ from the recorded reference')
    entries=entries[:args.eval_size]
    collection=loads_json((CHECKPOINT_ROOT/'manifest.json').read_text())
    classifier_path=CHECKPOINT_ROOT/'livdet/classifier.pth'
    rectified_path=CHECKPOINT_ROOT/'livdet/pgd/denoiser.pt'
    if fingerprint(classifier_path)!=next(r['sha256'] for r in collection['copied_files'] if r['copy']=='livdet/classifier.pth'):
        raise ValueError('Classifier hash mismatch')
    if fingerprint(rectified_path)!=next(r['sha256'] for r in collection['models'] if r['dataset']=='livdet' and r['attack']=='pgd'):
        raise ValueError('Rectified checkpoint hash mismatch')
    checkpoints={'rectified':rectified_path}
    if args.clean_checkpoint:
        checkpoints={'clean':args.clean_checkpoint,**checkpoints}
    hashes={e['id']:{k:fingerprint(e[k]) for k in ['path','attack_path']} for e in entries}
    identity=dict(attack=ATTACK,seed=args.seed,
        samples=[(e['id'],e['label'],hashes[e['id']]) for e in entries],
        classifier_sha256=fingerprint(classifier_path),
        checkpoint_sha256={k:fingerprint(v) for k,v in checkpoints.items()},
        code_sha256={name:fingerprint(Path(__file__).parent/name) for name in [
            'evaluate_livdet_pgd_candidate.py','audit_livdet_pgd_generation.py',
            'adaptive_pgd.py','search_livdet_pgd_parameters.py','livdet.py',
            'models.py','evaluate_legacy_livdet.py','run.py']})
    reference=None
    if args.reference_run:
        ref_config=json.loads((args.reference_run/'config.json').read_text())
        ref_completion=json.loads((args.reference_run/'completion.json').read_text())
        if ref_completion['experiment_signature']!=ref_config['experiment_signature'] or ref_completion['sample_count']!=len(entries):
            raise ValueError('Reference is not a complete matching run')
        if fingerprint(args.reference_run/'per_image.csv')!=ref_completion['per_image_sha256']:
            raise ValueError('Reference per-image hash mismatch')
        for key in ['attack','seed','samples','classifier_sha256','checkpoint_sha256']:
            if signature(identity[key])!=signature(ref_config[key]):
                raise ValueError('Reference differs: '+key+'; use exactly the completed reference models/data/settings')
        reference=list(csv.DictReader((args.reference_run/'per_image.csv').open()))
        if len(reference)!=len(entries):
            raise ValueError('Reference sample count mismatch')
        identity['reference_per_image_sha256']=ref_completion['per_image_sha256']
    identity['adaptive']=args.adaptive
    identity['adaptive_method']='Input truncate8 identity BPDA + legacy output PNG BPDA; probability CE,last iterate,40 steps,no random start' if args.adaptive else None
    print('VERIFIED',len(entries),'images; models:',','.join(checkpoints),'; attack:',ATTACK,flush=True)
    if args.check_only:
        return
    if args.device=='cuda' and not torch.cuda.is_available():
        p.error('CUDA unavailable')
    torch.set_num_threads(args.threads)
    torch.manual_seed(args.seed)
    torch.use_deterministic_algorithms(True)
    torch.backends.cudnn.benchmark=False
    device=torch.device(args.device)
    identity['runtime']=dict(torch=str(torch.__version__),cuda=torch.version.cuda,
        cudnn=torch.backends.cudnn.version(),device=str(device),threads=args.threads,
        gpu=torch.cuda.get_device_name(device) if device.type=='cuda' else None,
        matmul_allow_tf32=torch.backends.cuda.matmul.allow_tf32,
        cudnn_allow_tf32=torch.backends.cudnn.allow_tf32)
    if reference is not None and identity['runtime']!=ref_config['runtime']:
        raise ValueError('Use the same runtime/GPU as the reference, or measure a new non-adaptive reference there first')
    classifier=load_classifier(classifier_path,device)
    models={};provenance={}
    for name,path in checkpoints.items():
        model,info=load_denoiser(path,device,'livdet')
        target=info['training_config'].get('training_target')
        if name=='clean' and target!='clean':
            raise ValueError('Clean checkpoint must identify training_target=clean')
        if name=='rectified' and target=='clean':
            raise ValueError('Rectified checkpoint is marked clean-target')
        models[name]=model;provenance[name]=info
    wrappers={k:QuantizedDefense(classifier,m).eval().requires_grad_(False) for k,m in models.items()} if args.adaptive else {}
    stamp=signature(identity)
    output=args.resume or args.output or Path(__file__).parent/'runs'/datetime.now().strftime('livdet_pgd_candidate_%Y%m%d_%H%M%S_%f')
    if args.resume:
        if json.loads((output/'config.json').read_text())['experiment_signature']!=stamp:
            raise ValueError('Resume needs identical samples, checkpoints, code, runtime and settings')
    else:
        output.mkdir(parents=True,exist_ok=False)
        write_json(output/'config.json',dict(experiment_signature=stamp,**identity,
            model_provenance=provenance,mode='non-adaptive + adaptive BPDA PGD; no training' if args.adaptive else 'non-adaptive classifier-only PGD; no training',
            denoiser_postprocessing='Legacy RGB uint8 truncation -> PIL grayscale -> classifier',
            missing_clean_target='clean' not in models,
            adaptive_asr_denominator='Samples correct under quantized defended clean input Q(tensor-gray clean); separate from canonical PIL clean baseline',
            adaptive_table_metric='adaptive_pgd_percent: accuracy on final iterate of adaptive attack alone; combined_accuracy_percent is reported separately',
            source_case_id='case_42f43d74a4339e9d'))
        write_json(output/'manifest.json',entries)
        write_json(output/'input_hashes.json',hashes)
    (output/'samples').mkdir(exist_ok=True)
    livdet.RECOVERY_REPORT_PATH=output/'image_decode_recoveries.json'
    records=[]
    for index,e in enumerate(entries):
        saved=output/'samples'/f'{index:04d}.json'
        if saved.is_file():
            r=json.loads(saved.read_text())
            if r['id']!=e['id'] or r['label']!=e['label']:
                raise ValueError('Cached sample identity mismatch')
            records.append(r)
            continue
        canonical=livdet.read_tensor(e['path'])[None].to(device)
        center=rgb_to_grayscale(livdet.read_tensor(e['path'],True)[None].to(device),num_output_channels=1)
        archived=livdet.read_tensor(e['attack_path'])[None].to(device)
        labels=torch.tensor([e['label']],device=device)
        generated=generate(classifier,center,labels,ATTACK['alpha'],False,True,False,
                           center,ATTACK['steps'],epsilon=ATTACK['epsilon'])
        if not torch.isfinite(generated).all() or (generated-center).abs().max().item()>ATTACK['epsilon']+1e-6:
            raise ValueError('Invalid generated image or float budget violation')
        candidate=encode(generated,'truncate8')
        if candidate.min()<0 or candidate.max()>1:
            raise ValueError('Encoded image outside[0,1]')
        r=dict(id=e['id'],label=e['label'],float_linf=(generated-center).abs().max().item(),
               encoded_linf=(candidate-canonical).abs().max().item())
        with torch.no_grad():
            for source,image in [('clean',canonical),('archived',archived),('candidate',candidate)]:
                for name,model in [('classifier',None),*models.items()]:
                    value=image if model is None else classifier_input(model(image.repeat(1,3,1,1)),True)
                    logits=classifier(value)
                    if not torch.isfinite(logits).all():
                        raise ValueError('Nonfinite classifier output')
                    pred=logits.argmax(1).item()
                    r[name+'_'+source+'_prediction']=pred
                    r[name+'_'+source+'_correct']=int(pred==e['label'])
        if reference is not None:
            expected=reference[index]
            if expected['id']!=e['id'] or int(expected['label'])!=e['label']:
                raise ValueError('Reference sample identity/order mismatch')
            for key,value in r.items():
                if key.endswith('_prediction') and int(expected[key])!=value:
                    raise ValueError(f'Non-adaptive control differs for {e["id"]}: {key}; stop before mixing conditions')
        if args.adaptive:
            r['adaptive']={}
            for name,wrapper in wrappers.items():
                with torch.no_grad():
                    baseline=wrapper(center).argmax(1).item()
                adv,diagnostics=pgd_candidates(wrapper,center,labels,[center],
                    ATTACK['epsilon'],ATTACK['alpha'],ATTACK['steps'],
                    loss_mode=ATTACK['loss'],selection=ATTACK['selection'])
                with torch.no_grad():
                    logits=wrapper(adv)
                    if not torch.isfinite(logits).all():
                        raise ValueError('Nonfinite adaptive logits')
                    pred=logits.argmax(1).item()
                quantized=encode(adv,'truncate8')
                r['adaptive'][name]=dict(prediction=pred,correct=int(pred==e['label']),
                    baseline_prediction=baseline,baseline_correct=int(baseline==e['label']),
                    float_linf=(adv-center).abs().max().item(),
                    encoded_linf=(quantized-canonical).abs().max().item(),**diagnostics)
        write_json(saved,r)
        records.append(r)
        if (index+1)%25==0 or index+1==len(entries):
            summarize(output,records,models,len(entries))
            print('PROGRESS',index+1,'/',len(entries),flush=True)
    summarize(output,records,models,len(entries))
    write_json(output/'completion.json',dict(experiment_signature=stamp,sample_count=len(records),
        summary_sha256=fingerprint(output/'summary.csv'),per_image_sha256=fingerprint(output/'per_image.csv')))
    print('RESULTS',output,flush=True)


if __name__=='__main__':
    main()
