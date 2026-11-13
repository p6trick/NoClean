# 고정 PGD 후보의 LivDet N2N 평가

전체 final-evaluation 1,250장 탐색에서 분류기 정확도 2.88%를 보인 후보를 고정해 N2N 방어 성능을 확인한다. 저장 공격 분류기 정확도는 3.76%였다. 이는 평가셋 탐색으로 찾은 후보이며 과거 생성 설정의 복원으로 단정하지 않는다.

후보: epsilon8/255, alpha2/255,40회, random start 없음, probability CE, RGB→tensor→torchvision grayscale, 마지막 반복, 입력 공격 이미지 truncate8. 원래 epsilon.03과 다르다. 양자화 후 canonical clean 대비 최대 변화는 탐색에서 약.03529였다. float 공격의 L∞ 제약과 최종 양자화 이미지의 변화량을 별도로 기록한다.

`evaluate_livdet_pgd_candidate.py`는 기존 `generate`와 `encode`를 재사용한다. 공격은 분류기에만 생성하며, 동일 이미지를 두 denoiser에 사용할 수 있다. 새 모델 학습이나 adaptive 공격은 수행하지 않는다. Denoiser 이후에는 기존 LivDet 평가의 RGB uint8→PIL grayscale 후처리를 유지한다.

## 서버 실행

기존 n2n_py311 가상환경, assets와 data를 사용한다. 새 패키지는 필요하지 않으며 requirements를 다시 설치할 필요가 없다. 이 서버는 드라이버에 맞춘 torch2.7.1+cu118을 사용한다.

```bash
cd /path/to/NoClean
git pull origin main
source n2n_py311/bin/activate
source .env

python reproduction/evaluate_livdet_pgd_candidate.py --check-only

nohup python -u reproduction/evaluate_livdet_pgd_candidate.py \
  --device cuda \
  --output ./reproduction/runs/livdet_pgd_candidate_rectified \
  > livdet_pgd_candidate.log 2>&1 &

tail -f livdet_pgd_candidate.log
```

기본값은 N2N_assets의 PGD rectified-target 모델만 평가한다. Clean-target 모델은 첫 번째 target-ablation에서 새로 학습한 별도 가중치이며 원래 assets 묶음에는 없다. 파일이 있다면 새 output 경로와 다음 옵션으로 두 모델을 함께 평가한다.

```bash
python -u reproduction/evaluate_livdet_pgd_candidate.py \
  --device cuda \
  --clean-checkpoint /path/to/target_ablation/pgd/seed_2026/clean/denoiser.pt \
  --output ./reproduction/runs/livdet_pgd_candidate_both
```

파일이 없으면 clean-target 행을 생성하지 않는다. Rectified-target 모델을 clean-target으로 대신 표시하지 않는다.

중단 후 같은 환경에서 재개:

```bash
python -u reproduction/evaluate_livdet_pgd_candidate.py \
  --device cuda \
  --resume ./reproduction/runs/livdet_pgd_candidate_rectified
```

완료된 샘플부터 이어서 처리한다. 두 모델 실행을 재개할 때는 같은 clean-checkpoint 옵션도 추가한다.

## 결과

- summary.csv: 분류기 단독과 사용한 N2N 각각의 clean / 저장 공격 / 후보 공격 정확도, 동일 샘플 저장 공격과 차이. count와 complete 확인.
- metrics.json: 전체 결과와 float/양자화 후 최대 perturbation. 분류기 후보 정확도가 탐색의 2.88%와 얼마나 다른지도 기록.
- config.json: 고정 공격 설정, 모델·코드·데이터 해시와 실행 환경. clean-target 포함 여부 명시.
- manifest.json, input_hashes.json, per_image.csv: 평가 샘플, 파일 식별과 개별 예측.
- completion.json: 전체 완료 기록.

summary.csv, metrics.json, config.json을 공유한다. 기존 저장 공격 방어 정확도 Ours89.76%, clean-target89.04%는 참고치이고, 같은 실행에서 재측정한 archived_attack_accuracy_percent가 직접 비교 기준이다. 학습·추론·테스트는 코드 작성 시 실행하지 않았다.


## 같은 조건의 adaptive BPDA-PGD

`--adaptive` 옵션으로 같은 non-adaptive 대조군을 다시 측정하고, 각 denoiser를 개별적으로 공격한다.

공격 경로는 float latent 입력→truncate8→N2N→기존 PNG 후처리→분류기다. Forward는 정확한 입력 내림 및 기존 PNG 처리를 유지한다. Backward는 입력 양자화를 identity BPDA로, 출력 PNG는 기존 `png_bpda`로 근사한다. `pgd_candidates`를 그대로 사용하며 probability CE, 마지막 반복,40회,alpha2/255,epsilon8/255,random start 없음,batch1을 유지한다. float 공격은 tensor-gray clean 중심에서 제한하고 양자화 후 canonical clean 대비 변화량도 별도로 기록한다.

완료된 non-adaptive 실행을 `--reference-run`으로 지정한다. 데이터·순서·labels·모델·설정 해시와 실행 환경이 같아야 하며 각 샘플의 non-adaptive 예측도 재측정해 일치 여부를 확인한다. 다르면 중단한다. 다른 서버에서 실행하려면 먼저 해당 서버에서 non-adaptive 기준 실행을 만들어야 한다. 기존 결과나 checkpoint를 덮어쓰지 않는다.

현재 PC에서 기존 non-adaptive 결과를 기준으로 실행하는 예:

```bash
cd /path/to/NoClean
conda activate n2n
source .env
export CLEAN_CKPT="$PWD/reproduction/runs/livdet_target_ablation_20261104_161606_395725/pgd/seed_2026/clean/denoiser.pt"

python reproduction/evaluate_livdet_pgd_candidate.py \
  --adaptive --clean-checkpoint "$CLEAN_CKPT" \
  --reference-run ./reproduction/runs/livdet_pgd_candidate_both_local \
  --check-only

nohup python -u reproduction/evaluate_livdet_pgd_candidate.py \
  --device cuda --adaptive --clean-checkpoint "$CLEAN_CKPT" \
  --reference-run ./reproduction/runs/livdet_pgd_candidate_both_local \
  --output ./reproduction/runs/livdet_pgd_candidate_adaptive_both_local \
  > livdet_pgd_candidate_adaptive_both_local.log 2>&1 &
```

중단 후 같은 옵션에서 `--output`을 다음으로 교체한다:

```bash
--resume ./reproduction/runs/livdet_pgd_candidate_adaptive_both_local
```

공유 파일: summary.csv,metrics.json,config.json. 표에는 `candidate_attack_accuracy_percent`와 `adaptive_pgd_percent`를 사용한다. `combined_accuracy_percent`는 clean·non-adaptive·adaptive를 모두 맞힌 비율로 별도 지표다. ASR은 양자화된 defended clean 입력을 맞힌 샘플을 분모로 하며 canonical PIL clean 정확도와 구분한다. metrics.json은 gradient 호출 수, zero-gradient 수, float/양자화 후 최대 perturbation을 기록한다. Probability CE와 BPDA의 공격 결과는 이 고정 설정에 대한 결과이며 최강 공격에 대한 robustness 보장은 아니다.

코드 수정 후 추론·테스트·설치는 직접 수행하지 않았다.
