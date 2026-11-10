"""Compare LivDet PGD configurations with archived attacks on a fixed split."""
import argparse
from datetime import datetime
import hashlib
import itertools
import json
from pathlib import Path
import time
from types import SimpleNamespace

from audit_livdet_pgd_generation import generate, start_for_id
import livdet
from paths import CHECKPOINT_ROOT, PM_ROOT, loads_json
from run import fingerprint, load_classifier
from epsilon_sweep_io import write_csv, write_json
import torch
from torchvision.transforms.functional import rgb_to_grayscale


def signature(value):
    return hashlib.sha256(json.dumps(value,sort_keys=True,separators=(',',':')).encode()).hexdigest()


def encode(images, name):
    if name=='float':
        return images
    if name=='save_image':
        # torchvision.utils.save_image's non-normalized multiply/add/clamp conversion.
        return (images*255+.5).clamp(0,255).floor()/255
    if name=='truncate8':
        return images.clamp(0,1).mul(255).floor()/255
    raise ValueError(name)


def aggregate(output, cases):
    rows=[]
    for case in cases:
        path=output/'cases'/case['case_id']/'summary.json'
        if path.is_file() and (path.parent/'completion.json').is_file():
            rows.extend(json.loads(path.read_text()))
    if not rows:
        return
    write_csv(output/'summary.csv',rows,list(rows[0]))
    comparison=sorted(rows,key=lambda r:(abs(r['delta_to_archived_pp']),r['case_id'],r['encoding']))
    write_csv(output/'accuracy_comparison.csv',
              [{'accuracy_rank':i+1,**r} for i,r in enumerate(comparison)],
              ['accuracy_rank',*list(rows[0])])
    # Fit image content, not the desired final-evaluation accuracy. Keep source-budget controls separate.
    ranked=sorted((r for r in rows if r['budget_role']=='archive_range_hypothesis'),
                  key=lambda r:(r['mean_rmse_to_archived'], -r['pixel_exact_image_percent'],
                                -r['classifier_prediction_agreement_percent'],r['case_id'],r['encoding']))
    write_csv(output/'ranking.csv',[{'rank':i+1,**row} for i,row in enumerate(ranked)],['rank',*list(rows[0])])


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--device',choices=['cpu','cuda'],default='cuda')
    parser.add_argument('--split',choices=['train','validation','evaluation'],default='train')
    parser.add_argument('--sample-size',type=int,default=8,
                        help='Balanced sample count; 0 uses every sample in the chosen split (original class ratio)')
    parser.add_argument('--seed',type=int,default=2026)
    parser.add_argument('--threads',type=int,default=4)
    parser.add_argument('--include-source-budget-controls',action='store_true',
                        help='Also examine notebook epsilon0.3; label as a different-budget control, not a compatible candidate')
    parser.add_argument('--candidate-selections',nargs='+',choices=['last','best'],default=['last'])
    parser.add_argument('--check-only',action='store_true')
    group=parser.add_mutually_exclusive_group()
    group.add_argument('--output',type=Path)
    group.add_argument('--resume',type=Path)
    args=parser.parse_args()
    if args.sample_size<0 or args.sample_size%2 or args.threads<1 or not 0<=args.seed<2**32:
        parser.error('Use sample size 0 (full split) or a positive even count, positive threads and a valid seed')
    if len(set(args.candidate_selections))!=len(args.candidate_selections):
        parser.error('Duplicate candidate selections')
    registry=loads_json((Path(__file__).parent/'livdet_experiments.json').read_text())
    # Split selection remains fixed even if the attack seed changes.
    groups=livdet.resolve_groups(PM_ROOT,registry['pgd'],SimpleNamespace(
        seed=2026,train_size=1125,validation_size=125,eval_size=1250))
    if args.sample_size==0:
        selected=list(groups[args.split])
    else:
        selected=[]
        for label in [0,1]:
            selected += [e for e in groups[args.split] if e['label']==label][:args.sample_size//2]
        if len(selected)!=args.sample_size:
            parser.error('Sample request exceeds the selected class pool')
    if args.split!='evaluation' and {e['id'] for e in selected}&{e['id'] for e in groups['evaluation']}:
        raise ValueError('Fitting samples overlap final evaluation')
    pinned={e['id']:{key:fingerprint(e[key]) for key in ['path','attack_path']} for e in selected}
    classifier_path=CHECKPOINT_ROOT/'livdet/classifier.pth'
    classifier_hash=fingerprint(classifier_path)
    collection=loads_json((CHECKPOINT_ROOT/'manifest.json').read_text())
    if classifier_hash!=next(r['sha256'] for r in collection['copied_files'] if r['copy']=='livdet/classifier.pth'):
        raise ValueError('Classifier hash mismatch')
    epsilons=[.03,8/255]+([.3] if args.include_source_budget_controls else [])
    cases=[]
    for epsilon,alpha,steps,random_start,loss,preprocessing,selection in itertools.product(
        epsilons,[2/255,.01],[7,40],[False,True],['probability','logits'],
        ['pil-l','tensor-gray'],args.candidate_selections):
        case={'epsilon':epsilon,'alpha':alpha,'steps':steps,'random_start':random_start,
              'loss':loss,'preprocessing':preprocessing,'selection':selection,
              'budget_role':'source_epsilon_0_3_control' if epsilon==.3 else 'archive_range_hypothesis'}
        case['case_id']='case_'+signature(case)[:16]
        cases.append(case)
    code_hashes={name:fingerprint(Path(__file__).parent/name) for name in [
        'search_livdet_pgd_parameters.py','audit_livdet_pgd_generation.py',
        'adaptive_pgd.py','livdet.py','run.py']}
    identity={'split':args.split,'sample_size':args.sample_size,'seed':args.seed,
              'classifier_sha256':classifier_hash,'samples':[(e['id'],e['label'],pinned[e['id']]) for e in selected],
              'cases':cases,'encodings':['float','save_image','truncate8'],'code_sha256':code_hashes}
    print('SEARCH_PLAN',json.dumps({'split':args.split,'images':len(selected),'batch_size':1,
        'attack_configurations':len(cases),'encoded_outputs':3*len(cases),
        'final_evaluation_used':args.split=='evaluation','classifier_hash':classifier_hash}),flush=True)
    if args.check_only:
        return
    if args.device=='cuda' and not torch.cuda.is_available():
        parser.error('CUDA unavailable')
    torch.set_num_threads(args.threads)
    torch.manual_seed(args.seed)
    torch.use_deterministic_algorithms(True)
    torch.backends.cudnn.benchmark=False
    device=torch.device(args.device)
    identity['runtime']={'torch':str(torch.__version__),'device':str(device),'threads':args.threads,
                         'gpu':torch.cuda.get_device_name(device) if device.type=='cuda' else None,
                         'cudnn_version':torch.backends.cudnn.version(),
                         'matmul_allow_tf32':torch.backends.cuda.matmul.allow_tf32,
                         'cudnn_allow_tf32':torch.backends.cudnn.allow_tf32}
    experiment_signature=signature(identity)
    output=args.resume or args.output or Path(__file__).parent/'runs'/datetime.now().strftime('livdet_pgd_parameter_search_%Y%m%d_%H%M%S_%f')
    if args.resume:
        old=json.loads((output/'config.json').read_text())
        if old['experiment_signature']!=experiment_signature:
            raise ValueError('Resume requires identical search, data, code, classifier and runtime')
    else:
        output.mkdir(parents=True,exist_ok=False)
        write_json(output/'config.json',{'experiment_signature':experiment_signature,**identity,
            'goal':'Compare classifier accuracies across PGD settings and archived attacks on identical samples',
            'sampling':'Entire fixed split in original order and class ratio' if args.sample_size==0 else 'Fixed first samples per class from seeded partition',
            'historical_reference_accuracy_percent':3.76,
            'evaluation_parameter_comparison':args.split=='evaluation',
            'sources':{'alpha_2_255_steps7_40_epsilon0_3':'PGD-Copy1.ipynb / PGD-CMFK.ipynb; exact archived generator unverified',
                       'epsilon0_03_8_255':'Hypotheses based on archived perturbation range and8bit storage',
                       'alpha0_01_random_logits':'Current implementation controls',
                       'tensor_gray':'Resize RGB then ToTensor then torchvision Grayscale, as in candidate notebooks'},
            'limitations':['No denoiser is loaded or used to rank configurations.',
                           'Matching accuracy alone does not identify the generator.',
                           'Random-start seed in historical generation is unverified.',
                           'Epsilon0.3 is a different-budget control and excluded from candidate ranking.',
                           'This is a finite source-grounded grid, not an exhaustive search of continuous hyperparameters.']})
        write_json(output/'manifest.json',selected)
        write_json(output/'input_hashes.json',pinned)
    livdet.RECOVERED_IMAGES.clear()
    livdet.RECOVERY_REPORT_PATH=output/'image_decode_recoveries.json'
    model=load_classifier(classifier_path,device)
    cache=[]
    for e in selected:
        canonical=livdet.read_tensor(e['path'])[None].to(device)
        rgb=livdet.read_tensor(e['path'],True)[None].to(device)
        tensor_gray=rgb_to_grayscale(rgb,num_output_channels=1)
        archived=livdet.read_tensor(e['attack_path'])[None].to(device)
        labels=torch.tensor([e['label']],device=device)
        with torch.no_grad():
            logits=model(archived)
            if not torch.isfinite(logits).all():
                raise ValueError('Nonfinite archived-input output')
            archived_pred=logits.argmax(1).item()
        # Keep the full-split image cache in CPU RAM, leaving GPU memory for PGD.
        cache.append((e,canonical.cpu(),tensor_gray.cpu(),archived.cpu(),labels.cpu(),archived_pred))
    baseline_correct=sum(pred==e['label'] for e,_,_,_,_,pred in cache)
    write_json(output/'archived_baseline.json',{'sample_count':len(cache),
        'classifier_accuracy_percent':100*baseline_correct/len(cache),
        'per_image':[{ 'id':e['id'],'label':e['label'],'prediction':pred} for e,_,_,_,_,pred in cache],
        'note':'Same-sample archived baseline is the primary comparison. Historical3.76% may use a different subset.',
        'historical_reference_accuracy_percent':3.76,
        'split':args.split,'full_split':args.sample_size==0})
    for number,case in enumerate(cases,1):
        folder=output/'cases'/case['case_id']
        if (folder/'completion.json').is_file():
            meta=json.loads((folder/'completion.json').read_text())
            if (meta['experiment_signature']!=experiment_signature or
                fingerprint(folder/'per_image.csv')!=meta['per_image_sha256'] or
                fingerprint(folder/'summary.json')!=meta['summary_sha256']):
                raise ValueError('Completed case changed')
            print('REUSE',number,'/',len(cases),case['case_id'],flush=True)
            continue
        folder.mkdir(parents=True,exist_ok=True)
        print('CASE',number,'/',len(cases),json.dumps(case),flush=True)
        records=[]
        started=time.monotonic()
        for image_number,(e,canonical_cpu,tensor_gray_cpu,archived_cpu,labels_cpu,archived_pred) in enumerate(cache,1):
            canonical=canonical_cpu.to(device)
            tensor_gray=tensor_gray_cpu.to(device)
            archived=archived_cpu.to(device)
            labels=labels_cpu.to(device)
            clean=canonical if case['preprocessing']=='pil-l' else tensor_gray
            start=start_for_id(clean,e['id'],args.seed,epsilon=case['epsilon'])
            generated=generate(model,clean,labels,case['alpha'],case['random_start'],
                case['loss']=='probability',case['selection']=='best',start,case['steps'],epsilon=case['epsilon'])
            if not torch.isfinite(generated).all() or (generated-clean).abs().max().item()>case['epsilon']+1e-6:
                raise ValueError('Invalid generated float image or budget violation')
            if generated.min()<0 or generated.max()>1:
                raise ValueError('Generated float image outside[0,1]')
            for encoding in identity['encodings']:
                value=encode(generated,encoding)
                with torch.no_grad():
                    logits=model(value)
                    if not torch.isfinite(logits).all():
                        raise ValueError('Nonfinite generated-input output')
                    predicted=logits.argmax(1).item()
                difference=(value.double()-archived.double())
                rmse=difference.square().mean().sqrt().item()
                l_inf=difference.abs().max().item()
                rounded_match=(value.mul(255).add(.5).clamp(0,255).floor()==
                               archived.mul(255).add(.5).clamp(0,255).floor()).float().mean().item()
                records.append({'id':e['id'],'label':e['label'],'encoding':encoding,
                    'prediction':predicted,'archived_prediction':archived_pred,
                    'classifier_correct':int(predicted==e['label']),
                    'prediction_agreement':int(predicted==archived_pred),
                    'rmse_to_archived':rmse,'linf_to_archived':l_inf,
                    'pixel_exact':int(torch.equal(value,archived)),
                    'rounded_pixel_match_fraction':rounded_match,
                    'clean_preprocessing_linf':(clean-canonical).abs().max().item(),
                    'generated_linf_from_preprocessed_clean':(generated-clean).abs().max().item(),
                    'encoded_linf_from_canonical_clean':(value-canonical).abs().max().item()})
            if image_number%100==0 or image_number==len(cache):
                print('PROGRESS',number,'/',len(cases),'images',image_number,'/',len(cache),flush=True)
        write_csv(folder/'per_image.csv',records,list(records[0]))
        rows=[]
        for encoding in identity['encodings']:
            group=[r for r in records if r['encoding']==encoding]
            n=len(group)
            rows.append({**case,'encoding':encoding,'split':args.split,'sample_count':n,'seed':args.seed,
                'classifier_accuracy_percent':100*sum(r['classifier_correct'] for r in group)/n,
                'archived_classifier_accuracy_percent':100*baseline_correct/n,
                'historical_reference_accuracy_percent':3.76,
                'delta_to_archived_pp':100*(sum(r['classifier_correct'] for r in group)-baseline_correct)/n,
                'delta_to_historical_3p76_pp':100*sum(r['classifier_correct'] for r in group)/n-3.76,
                'classifier_prediction_agreement_percent':100*sum(r['prediction_agreement'] for r in group)/n,
                'mean_rmse_to_archived':sum(r['rmse_to_archived'] for r in group)/n,
                'max_linf_to_archived':max(r['linf_to_archived'] for r in group),
                'pixel_exact_image_percent':100*sum(r['pixel_exact'] for r in group)/n,
                'mean_rounded_pixel_match_percent':100*sum(r['rounded_pixel_match_fraction'] for r in group)/n,
                'max_clean_preprocessing_linf':max(r['clean_preprocessing_linf'] for r in group),
                'max_encoded_attack_linf':max(r['encoded_linf_from_canonical_clean'] for r in group),
                'elapsed_seconds':time.monotonic()-started})
        write_json(folder/'summary.json',rows)
        write_json(folder/'completion.json',{'experiment_signature':experiment_signature,
            'per_image_sha256':fingerprint(folder/'per_image.csv'),'summary_sha256':fingerprint(folder/'summary.json')})
        aggregate(output,cases)
        print('CASE_COMPLETED',number,'/',len(cases),flush=True)
    aggregate(output,cases)
    write_json(output/'completion.json',{'case_count':len(cases),'experiment_signature':experiment_signature})
    print('RESULTS',output,flush=True)


if __name__=='__main__':
    main()
