"""Fixed non-adaptive LivDet PGD candidate: tensor-gray, 8/255, truncate8."""
import argparse
from datetime import datetime
import json
from pathlib import Path
from types import SimpleNamespace

from audit_livdet_pgd_generation import generate
from adaptive_pgd import load_denoiser
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
    write_csv(output/'per_image.csv',records,list(records[0]))
    write_csv(output/'summary.csv',rows,list(rows[0]))
    write_json(output/'metrics.json',dict(sample_count=len(records), expected_count=total,
        complete=len(records)==total, attack=ATTACK, results=rows,
        max_float_linf_from_attack_center=max(r['float_linf'] for r in records),
        max_encoded_linf_from_canonical_clean=max(r['encoded_linf'] for r in records),
        expected_classifier_candidate_percent=2.88,
        classifier_candidate_delta_to_search_pp=rows[0]['candidate_attack_accuracy_percent']-2.88,
        note='2.88% refers to the full1250 search on RTX3090/cu118, not a guaranteed value on other runtimes.'))


def main():
    p=argparse.ArgumentParser(description=__doc__)
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
    stamp=signature(identity)
    output=args.resume or args.output or Path(__file__).parent/'runs'/datetime.now().strftime('livdet_pgd_candidate_%Y%m%d_%H%M%S_%f')
    if args.resume:
        if json.loads((output/'config.json').read_text())['experiment_signature']!=stamp:
            raise ValueError('Resume needs identical samples, checkpoints, code, runtime and settings')
    else:
        output.mkdir(parents=True,exist_ok=False)
        write_json(output/'config.json',dict(experiment_signature=stamp,**identity,
            model_provenance=provenance,mode='non-adaptive classifier-only PGD; no training',
            denoiser_postprocessing='Legacy RGB uint8 truncation -> PIL grayscale -> classifier',
            missing_clean_target='clean' not in models,
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
