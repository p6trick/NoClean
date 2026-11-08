"""Result I/O shared by the MNIST epsilon runner and offline plotting script."""
import csv
import json
from pathlib import Path

EPSILONS = [.05, .10, .15, .20, .30, .40]
SUPERVISIONS = ['clean', 'rectified']
LABELS = {'clean': 'Clean target', 'rectified': 'Rectified target (Ours)'}


def condition_dir(root, epsilon, supervision):
    key = format(float(epsilon), '.10g').replace('.', 'p')
    return Path(root) / 'conditions' / ('epsilon_' + key) / supervision


def write_json(path, value):
    path = Path(path)
    temporary = path.with_name(path.name + '.tmp')
    temporary.write_text(json.dumps(value, indent=2, ensure_ascii=False, allow_nan=False) + '\n')
    temporary.replace(path)


def write_csv(path, rows, fields):
    path = Path(path)
    temporary = path.with_name(path.name + '.tmp')
    with temporary.open('w', newline='') as stream:
        writer = csv.DictWriter(stream, fieldnames=fields, extrasaction='ignore')
        writer.writeheader()
        writer.writerows(rows)
    temporary.replace(path)


def read_conditions(root):
    root = Path(root)
    config = json.loads((root / 'config.json').read_text())
    rows, missing = [], []
    for epsilon in config['epsilons']:
        for supervision in SUPERVISIONS:
            path = condition_dir(root, epsilon, supervision) / 'metrics.json'
            if path.is_file():
                row = json.loads(path.read_text())
                if row['experiment_signature'] != config['experiment_signature']:
                    raise ValueError(f'Condition signature mismatch: {path}')
                if row['epsilon'] != epsilon or row['supervision'] != supervision:
                    raise ValueError(f'Condition identity mismatch: {path}')
                rows.append(row)
            else:
                missing.append((epsilon, supervision))
    return config, rows, missing


def write_aggregates(root):
    root = Path(root)
    config, rows, missing = read_conditions(root)
    epsilons = sorted(config['epsilons'])
    lookup = {(r['epsilon'], r['supervision']): r for r in rows}
    common = ['epsilon', 'supervision', 'sample_count', 'seed', 'alpha', 'steps',
              'random_start', 'restarts', 'loss_mode', 'candidate_selection', 'batch_size',
              'classifier_sha256', 'denoiser_sha256']
    write_csv(root / 'accuracy_long.csv', rows, common + [
        'accuracy_percent', 'correct_count', 'clean_defended_accuracy_percent',
        'clean_inclusive_robust_accuracy_percent', 'asr_clean_correct_percent',
        'asr_numerator', 'asr_denominator', 'all_sample_error_percent'])
    write_csv(root / 'perturbation_stats.csv', rows, common + [
        'avg_linf', 'max_linf', 'avg_l2', 'max_l2', 'denoised_clean_rmse_mean',
        'asr_clean_correct_percent', 'asr_numerator', 'asr_denominator',
        'input_min', 'input_max', 'constraint_violations',
        'gradient_calls', 'zero_gradient_samples', 'max_gradient_linf', 'nonzero_input_gradient_observed'])
    wide = []
    for supervision in SUPERVISIONS:
        row = {'Supervision': LABELS[supervision]}
        for epsilon in epsilons:
            metric = lookup.get((epsilon, supervision))
            row[f'{epsilon:.2f}'] = metric['accuracy_percent'] if metric else ''
        wide.append(row)
    write_csv(root / 'accuracy_by_epsilon.csv', wide, ['Supervision'] + [f'{e:.2f}' for e in epsilons])
    comparisons = []
    for supervision in SUPERVISIONS:
        metric = lookup.get((.4, supervision))
        expected = config['reference']['expected'][supervision]
        comparisons.append({
            'supervision': supervision, 'epsilon': .4,
            'sample_count': config['sample_count'],
            'scope': 'full evaluation' if config['sample_count'] == 5000 else 'partial check',
            'expected_accuracy_percent': expected['accuracy_percent'],
            'actual_accuracy_percent': metric['accuracy_percent'] if metric else '',
            'difference_pp': metric['reference_accuracy_difference_pp'] if metric else '',
            'prediction_mismatch_count': metric['reference_prediction_mismatch_count'] if metric else '',
            'clean_baseline_prediction_mismatches': metric['clean_baseline_prediction_mismatches'] if metric else '',
            'raw_clean_prediction_mismatches': metric['raw_clean_prediction_mismatches'] if metric else '',
            'exact_accuracy_match': metric['correct_count'] == expected['correct_count'] if metric else '',
            'status': ('PASS' if metric['correct_count'] == expected['correct_count'] else 'MISMATCH') if metric else 'PENDING',
        })
    write_csv(root / 'reference_reproduction.csv', comparisons, list(comparisons[0]))
    attack = config['attack']
    lines = ['# MNIST adaptive PGD epsilon sensitivity', '',
             f"완료 조건: {len(rows)}/{len(rows)+len(missing)}; 조건별 {config['sample_count']}장.", '',
             f"고정 설정: alpha={attack['alpha']:.15g}, steps={attack['steps']}, "
             f"random start={attack['random_start']}, seed={attack['seed']}, "
             f"batch={attack['batch_size']}, loss={attack['loss_mode']}, selection={attack['selection']}.", '',
             '## Adaptive PGD 정확도 (%)', '',
             '| Supervision | ' + ' | '.join(f'{e:.2f}' for e in epsilons) + ' |',
             '|---|' + '---:|' * len(epsilons)]
    for supervision in SUPERVISIONS:
        cells = [f"{lookup[e, supervision]['accuracy_percent']:.2f}" if (e, supervision) in lookup else '미완료' for e in epsilons]
        lines.append('| ' + LABELS[supervision] + ' | ' + ' | '.join(cells) + ' |')
    lines += ['', '## Epsilon 0.4 기준 재현', '',
              '| Supervision | 기준 (%) | 이번 (%) | 차이 (%p) | 상태 |',
              '|---|---:|---:|---:|---|']
    for row in comparisons:
        actual = f"{row['actual_accuracy_percent']:.2f}" if row['status'] != 'PENDING' else '미완료'
        delta = f"{row['difference_pp']:+.4f}" if row['status'] != 'PENDING' else ''
        lines.append(f"| {LABELS[row['supervision']]} | {row['expected_accuracy_percent']:.2f} | {actual} | {delta} | {row['status']} |")
    lines += ['', '## Gradient 진단', '',
              '| Epsilon | Supervision | Gradient calls | Zero-gradient samples | Nonzero observed |',
              '|---:|---|---:|---:|---|']
    for row in rows:
        lines.append(f"| {row['epsilon']:.2f} | {LABELS[row['supervision']]} | {row['gradient_calls']} | "
                     f"{row['zero_gradient_samples']} | {row['nonzero_input_gradient_observed']} |")
    lines += ['', '## 해석과 검증 범위', '',
              '- 주 지표는 기존 실행과 같은 마지막 PGD 후보의 전체 샘플 정확도다.',
              '- ASR은 각 방어 모델의 clean 입력 정답 샘플만 분모로 사용한다. 전체 오분류율과 구분한다.',
              '- clean 입력 오분류까지 실패로 포함한 정확도도 accuracy_long.csv에 별도로 기록한다.',
              '- random start가 꺼져 있으므로 기준값 차이를 random start로 설명하지 않는다.',
              '- 40×2/255≈0.313725이므로 random start가 없는 이 설정은 epsilon0.4 전체 반경을 탐색하지 못한다.',
              '- 마지막 반복·고정 step 수의 측정 곡선은 완전 최적화의 단조성을 보장하지 않는다. 곡선을 강제로 보정하지 않는다.',
              '- denoised RMSE는 실제 분류기 입력인 가중 회색조·clamp 후 출력과 clean 사이의 샘플별 RMSE 평균이다.',
              '- 이 결과는 과거 probability-CE 설정의 민감도 분석이며 더 강한 logits-CE 공격 결과를 대체하지 않는다.',
              '- 시각화에서 의미 훼손 여부는 정성적으로 판단하며 분류기 예측만으로 단정하지 않는다.', '',
              '환경·가중치·데이터 식별 정보: config.json, manifest.json, input_hashes.json.',
              'PNG와 시각화는 별도 분석 스크립트를 실행한 뒤 생성된다.', '']
    (root / 'summary.md').write_text('\n'.join(lines))
    return comparisons, missing
